"""Combined popcorn MPM simulation inside the full USD scene.

Loads the complete scene (popcorn machine, popcorn car, warehouse, lights)
and simulates popcorn particles inside the machine tray.  The scooper enters
through the machine's open side (low-X), sweeps through the popcorn, lifts
out, and dumps into the bucket.

Particle positions are derived from the PointInstancer in background.usda,
mapped into the machine box's world-space bounds.

Usage:
    uv run python simulate_popcorn.py
    uv run python simulate_popcorn.py --viewer rerun
    uv run python simulate_popcorn.py --record-mp4 ./popcorn.mp4
"""

import math
import time

import numpy as np
import warp as wp
from pxr import Usd, UsdGeom

import newton
import newton.examples
import newton.usd
from newton.solvers import SolverImplicitMPM
from video_recorder import VideoRecorder
from visualize_scene_usd import (
    SCENE_USD_PATH,
    load_scene,
)

_ASSET_ROOT = "/home/yvetted/project/genie_sim/source/geniesim/assets"

POPCORN_USD_PATH = f"{_ASSET_ROOT}/background/common/popcorn/benchmark_popcorn_001/Aligned.usd"
POPCORN_PRIM_PATH = "/World/body/visual"

SCOOP_USD_PATH = f"{_ASSET_ROOT}/background/common/benchmark_popcorn_scoop_001/Aligned.usd"

BUCKET_USD_PATH = f"{_ASSET_ROOT}/objects/benchmark/popcorn_bucket/benchmark_popcorn_bucket_003/Aligned.usda"

# ---------------------------------------------------------------------------
# Extract particle emit bounds from PointInstancer
# ---------------------------------------------------------------------------
def _get_popcorn_emit_bounds(
    scene_path: str,
    instancer_prim_path: str = "/World/background/popcornInstancer",
    machine_prim_path: str = "/World/background/benchmark_popcorn_machine_001",
) -> tuple[np.ndarray, np.ndarray]:
    """Derive particle emission box from the USD PointInstancer positions.

    The PointInstancer stores ~1200 popcorn positions in the scene's local
    frame (near the origin).  We compute an offset so that the bounding box
    of those positions is centred inside the machine's tray.

    Returns ``(emit_lo, emit_hi)`` in world-space metres.
    """
    stage = Usd.Stage.Open(scene_path)
    xc = UsdGeom.XformCache(Usd.TimeCode.Default())

    pi = UsdGeom.PointInstancer(stage.GetPrimAtPath(instancer_prim_path))
    positions = np.array(pi.GetPositionsAttr().Get(), dtype=np.float32)

    pi_xform = xc.GetLocalToWorldTransform(stage.GetPrimAtPath(instancer_prim_path))
    pi_trans = pi_xform.ExtractTranslation()
    positions += np.array([pi_trans[0], pi_trans[1], pi_trans[2]], dtype=np.float32)

    machine_xform = xc.GetLocalToWorldTransform(stage.GetPrimAtPath(machine_prim_path))
    machine_pos = machine_xform.ExtractTranslation()

    # Machine box interior (world-space) determined from collider_body_000/001/002:
    #   wall 000 (front):  Y ≈  0.0     X ∈ [0.78, 1.13]
    #   wall 001 (right):  X ≈  1.12    Y ∈ [-0.47, 0.0]
    #   wall 002 (back):   Y ≈ -0.475   X ∈ [0.78, 1.13]
    #   bottom (003):      Z ≈  0.99
    box_x = (0.79, 1.12)
    box_y = (-0.47, -0.005)
    box_z_bottom = 1.00
    box_center_x = (box_x[0] + box_x[1]) / 2.0
    box_center_y = (box_y[0] + box_y[1]) / 2.0

    pos_center = positions.mean(axis=0)
    pos_min = positions.min(axis=0)

    offset = np.array([
        box_center_x - pos_center[0],
        box_center_y - pos_center[1],
        box_z_bottom - pos_min[2],
    ], dtype=np.float32)

    world_positions = positions + offset
    emit_lo = world_positions.min(axis=0)
    emit_hi = world_positions.max(axis=0)

    print(f"PointInstancer: {len(positions)} positions")
    print(f"  local range : {positions.min(axis=0)} → {positions.max(axis=0)}")
    print(f"  offset      : {offset}")
    print(f"  emit_lo     : {emit_lo}")
    print(f"  emit_hi     : {emit_hi}")
    print(f"  machine pos : ({machine_pos[0]:.3f}, {machine_pos[1]:.3f}, {machine_pos[2]:.3f})")
    return emit_lo, emit_hi


# ---------------------------------------------------------------------------
# Scooper trajectory — enters machine box from the open left side (low X)
# ---------------------------------------------------------------------------
def _build_scoop_trajectory(
    bucket_pos: wp.vec3,
    time_scale: float = 1.0,
) -> list[tuple[float, wp.vec3, wp.quat]]:
    """Keyframed scooper trajectory entering the machine from the open side.

    Machine tray (world space):
        Three walls: front (Y≈0), right (X≈1.12), back (Y≈-0.475).
        Opening on the LEFT side at X ≈ 0.78.
        Bottom at Z ≈ 1.0, interior centre ≈ (0.95, -0.24, 1.2).

    The scooper's raw mesh has the bowl along +Y.  We rotate -90° around Z
    so the bowl faces +X — pointing *into* the box through the opening.

    Args:
        bucket_pos: Target bucket position in world space.
        time_scale: Multiplier applied to every keyframe timestamp. Values
            greater than 1 slow down the whole motion proportionally; values
            less than 1 speed it up. The shape of the motion is preserved.
    """
    # Bowl faces +X after this rotation (entering the box from the left).
    q_base = wp.quat_from_axis_angle(wp.vec3(0.0, 0.0, 1.0), math.radians(-90.0))

    q_idle = q_base
    q_tilt = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.radians(15.0)) * q_base
    q_lift = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.radians(-10.0)) * q_base
    q_dump = wp.quat_from_axis_angle(wp.vec3(0.0, 1.0, 0.0), math.radians(80.0)) * q_base

    # Machine box reference points
    opening_x = 0.78
    box_mid_y = -0.24
    box_bottom_z = 1.02
    scoop_z = box_bottom_z + 0.01

    # Positions along the trajectory
    outside = wp.vec3(
        opening_x - 0.35, box_mid_y, box_bottom_z + 0.20,
    )
    approach = wp.vec3(
        opening_x - 0.05, box_mid_y, scoop_z + 0.13,
    )
    inside_shallow = wp.vec3(
        opening_x + 0.10, box_mid_y, scoop_z + 0.06,
    )
    inside_deep = wp.vec3(0.98, box_mid_y, scoop_z + 0.03)
    lift_inside = wp.vec3(
        opening_x + 0.2, box_mid_y, box_bottom_z + 0.10,
    )
    lift_opening = wp.vec3(
        opening_x - 0.2, box_mid_y, box_bottom_z + 0.10,
    )

    # Bucket is right next to the machine opening — short descent
    bx = float(bucket_pos[0])
    by = float(bucket_pos[1])
    bz = float(bucket_pos[2])
    bowl_half_len = 0.1
    above_bucket = wp.vec3(
        bx - bowl_half_len, by, bz + 0.2,
    )

    keyframes: list[tuple[float, wp.vec3, wp.quat]] = [
        # settle — particles fall into the tray
        (0.0,  outside, q_idle),
        (2.0,  outside, q_idle),
        # approach — glide toward the opening, tilt down
        (3.5,  approach, q_tilt),
        # enter the box
        (5.0,  inside_shallow, q_tilt),
        # sweep deep into the pile
        (6.0,  inside_deep, q_tilt),
        # lift inside the box
        (8.0,  lift_inside, q_lift),
        (9.5,  lift_inside, q_lift),
        # pull back out through the opening
        (13.5, lift_opening, q_lift),
        (14.0, lift_opening, q_lift),
        # descend toward the bucket (right next to the opening)
        (17.0, above_bucket, q_lift),
        # dump popcorn into the bucket
        (17.5, above_bucket, q_dump),
        (19.0, above_bucket, q_dump),
        # level off
        (20.5, above_bucket, q_lift),
        (22.0, above_bucket, q_lift),
    ]
    return [(t * time_scale, p, q) for (t, p, q) in keyframes]


# ---------------------------------------------------------------------------
# Warp kernels
# ---------------------------------------------------------------------------
@wp.func
def _ensure_rotation(R: wp.mat33) -> wp.mat33:
    if wp.determinant(R) < 0.0:
        return wp.mat33(
            R[0, 0], R[0, 1], -R[0, 2],
            R[1, 0], R[1, 1], -R[1, 2],
            R[2, 0], R[2, 1], -R[2, 2],
        )
    return R


@wp.func
def _eval_keyframe(
    kf_times: wp.array[float],
    kf_pos: wp.array[wp.vec3],
    kf_rot: wp.array[wp.quat],
    kf_count: int,
    t: float,
) -> wp.transform:
    """Binary-search-style keyframe evaluation on the GPU."""
    last = kf_count - 1
    p = kf_pos[last]
    q = kf_rot[last]
    if t <= kf_times[0]:
        p = kf_pos[0]
        q = kf_rot[0]
    else:
        found = int(0)
        for i in range(1, kf_count):
            if found == 0 and t <= kf_times[i]:
                alpha = (t - kf_times[i - 1]) / (kf_times[i] - kf_times[i - 1])
                p = wp.lerp(kf_pos[i - 1], kf_pos[i], alpha)
                q = wp.quat_slerp(kf_rot[i - 1], kf_rot[i], alpha)
                found = 1
    return wp.transform(p, q)


@wp.kernel
def eval_trajectory_pair(
    kf_times: wp.array[float],
    kf_pos: wp.array[wp.vec3],
    kf_rot: wp.array[wp.quat],
    kf_count: int,
    t0: float,
    t1: float,
    out0: wp.array[wp.transform],
    out1: wp.array[wp.transform],
):
    """Evaluate the scooper trajectory at two time points on the GPU.

    Writes transforms for *t0* and *t1* directly into device-resident arrays.
    """
    out0[0] = _eval_keyframe(kf_times, kf_pos, kf_rot, kf_count, t0)
    out1[0] = _eval_keyframe(kf_times, kf_pos, kf_rot, kf_count, t1)


@wp.kernel
def build_instance_xforms(
    positions: wp.array(dtype=wp.vec3),
    def_grads: wp.array(dtype=wp.mat33),
    out_xforms: wp.array(dtype=wp.transform),
):
    tid = wp.tid()
    F = def_grads[tid]
    U = wp.mat33()
    sigma = wp.vec3()
    V = wp.mat33()
    wp.svd3(F, U, sigma, V)
    R = _ensure_rotation(U @ wp.transpose(V))
    q = wp.normalize(wp.quat_from_matrix(R))
    out_xforms[tid] = wp.transform(positions[tid], q)


@wp.kernel
def set_body_transform_lerp_dual(
    body_id: int,
    ratio_start: float,
    ratio_end: float,
    transform0: wp.array(dtype=wp.transform),
    transform1: wp.array(dtype=wp.transform),
    body_q_start: wp.array(dtype=wp.transform),
    body_q_end: wp.array(dtype=wp.transform),
):
    """Lerp/slerp between two keyframe transforms and write both substep
    endpoints directly into the two body_q buffers in a single launch.
    """
    tf0 = transform0[0]
    tf1 = transform1[0]
    pos0 = wp.transform_get_translation(tf0)
    pos1 = wp.transform_get_translation(tf1)
    rot0 = wp.transform_get_rotation(tf0)
    rot1 = wp.transform_get_rotation(tf1)
    body_q_start[body_id] = wp.transform(
        wp.lerp(pos0, pos1, ratio_start),
        wp.quat_slerp(rot0, rot1, ratio_start),
    )
    body_q_end[body_id] = wp.transform(
        wp.lerp(pos0, pos1, ratio_end),
        wp.quat_slerp(rot0, rot1, ratio_end),
    )


# ---------------------------------------------------------------------------
# Example
# ---------------------------------------------------------------------------
class Example:
    def __init__(self, viewer, options):
        self.fps = 60.0
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.sim_substeps = 2
        self.sim_dt = self.frame_dt / self.sim_substeps
        self.viewer = viewer
        self.bucket_pos = wp.vec3(0.6, 0.44, 1.2)
        self.bucket_xform = wp.transform(self.bucket_pos, wp.quat_identity())

        self._load_render_mesh(options)
        scoop_keyframes = self._build_model(options)
        self._create_solver(options.voxel_size)
        self._setup_viewer(options, scoop_keyframes)

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    def _load_render_mesh(self, options):
        """Load the popcorn USD mesh and prepare material arrays."""
        usd_stage = Usd.Stage.Open(POPCORN_USD_PATH)
        popcorn_mesh = newton.usd.get_mesh(
            usd_stage.GetPrimAtPath(POPCORN_PRIM_PATH),
            load_normals=True,
            load_uvs=True,
        )
        center = (popcorn_mesh.vertices.max(axis=0) + popcorn_mesh.vertices.min(axis=0)) * 0.5
        popcorn_mesh.vertices = popcorn_mesh.vertices - center
        popcorn_mesh.finalize()
        self.render_mesh = popcorn_mesh

        c = popcorn_mesh.color
        color_val = (
            wp.vec3(float(c[0]), float(c[1]), float(c[2]))
            if c is not None
            else wp.vec3(0.95, 0.85, 0.55)
        )
        r = popcorn_mesh.roughness if popcorn_mesh.roughness is not None else 0.4
        m = popcorn_mesh.metallic if popcorn_mesh.metallic is not None else 0.3
        tex = 1.0 if (popcorn_mesh.texture is not None and popcorn_mesh.uvs is not None) else 0.0
        self._popcorn_color = wp.array([color_val], dtype=wp.vec3)
        self._popcorn_material = wp.array([wp.vec4(r, m, 0.0, tex)], dtype=wp.vec4)

    def _build_model(
        self, options,
    ) -> list[tuple[float, wp.vec3, wp.quat]]:
        """Build the Newton model with particles, rigid bodies, and scene
        colliders.

        Returns the scooper keyframe trajectory for later GPU baking.
        """
        builder = newton.ModelBuilder()
        SolverImplicitMPM.register_custom_attributes(builder)

        voxel_size = options.voxel_size
        emit_lo, emit_hi = self._add_particles(builder, options)
        scoop_keyframes = self._add_scooper(builder)
        self._add_bucket(builder)
        self._add_scene_colliders(
            builder, options, emit_lo, emit_hi, scoop_keyframes, voxel_size,
        )

        self.model = builder.finalize()
        self.num_particles = self.model.particle_count
        print(
            f"Created {self.num_particles} popcorn particles, "
            f"{self.model.body_count} body(ies), "
            f"{self.model.shape_count} shape(s)"
        )

        all_idx = wp.array(
            np.arange(self.num_particles, dtype=np.int32),
            device=self.model.device,
        )
        self.model.mpm.young_modulus[all_idx].fill_(options.young_modulus)
        self.model.mpm.poisson_ratio[all_idx].fill_(0.3)
        self.model.mpm.friction[all_idx].fill_(0.4)
        self.model.mpm.yield_stress[all_idx].fill_(options.yield_stress)
        self.model.mpm.yield_pressure[all_idx].fill_(options.yield_pressure)

        return scoop_keyframes

    def _add_particles(
        self,
        builder: newton.ModelBuilder,
        options,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Emit popcorn particles into the machine tray.

        Returns ``(emit_lo, emit_hi)`` for collider AABB filtering.
        """
        emit_lo, emit_hi = _get_popcorn_emit_bounds(options.scene)
        if options.emit_lo is not None:
            emit_lo = np.array(options.emit_lo, dtype=np.float32)
        if options.emit_hi is not None:
            emit_hi = np.array(options.emit_hi, dtype=np.float32)

        emit_center = (emit_lo + emit_hi) * 0.5
        emit_lo = emit_center - (emit_center - emit_lo) * 0.7
        emit_hi = emit_center + (emit_hi - emit_center) * 0.7
        print(f"  shrunk emit_lo: {emit_lo}")
        print(f"  shrunk emit_hi: {emit_hi}")

        density = 20.0
        voxel_size = options.voxel_size
        particles_per_cell = options.particles_per_cell
        particle_res = np.array(
            np.ceil(particles_per_cell * (emit_hi - emit_lo) / voxel_size),
            dtype=int,
        )
        cell_size = (emit_hi - emit_lo) / particle_res
        cell_volume = float(np.prod(cell_size))
        radius = float(np.max(cell_size) * 0.5)
        mass = float(cell_volume * density)

        builder.add_particle_grid(
            pos=wp.vec3(emit_lo),
            rot=wp.quat_identity(),
            vel=wp.vec3(0.0),
            dim_x=particle_res[0] + 1,
            dim_y=particle_res[1] + 1,
            dim_z=particle_res[2] + 1,
            cell_x=cell_size[0],
            cell_y=cell_size[1],
            cell_z=cell_size[2],
            mass=mass,
            jitter=2.0 * radius,
            radius_mean=radius,
            flags=newton.ParticleFlags.ACTIVE,
        )
        total = int((particle_res[0] + 1) * (particle_res[1] + 1) * (particle_res[2] + 1))
        print(f"Particle grid: res={particle_res}, total={total}, cell_size={cell_size}")
        return emit_lo, emit_hi

    def _add_scooper(
        self, builder: newton.ModelBuilder,
    ) -> list[tuple[float, wp.vec3, wp.quat]]:
        """Add the kinematic scooper body.  Returns the keyframe trajectory."""
        scoop_keyframes = _build_scoop_trajectory(self.bucket_pos)
        scoop_start_xform = wp.transform(scoop_keyframes[0][1], scoop_keyframes[0][2])
        self._trajectory_end_time = scoop_keyframes[-1][0]

        stage = Usd.Stage.Open(SCOOP_USD_PATH)
        result = builder.add_usd(
            stage,
            floating=True,
            xform=scoop_start_xform,
            skip_mesh_approximation=True,
            hide_collision_shapes=True,
        )
        self.scoop_body_id = result["path_body_map"]["/World"]
        print(f"Scooper body id: {self.scoop_body_id}")
        return scoop_keyframes

    def _add_bucket(self, builder: newton.ModelBuilder):
        """Add the static bucket body."""
        stage = Usd.Stage.Open(BUCKET_USD_PATH)
        result = builder.add_usd(
            stage,
            floating=False,
            xform=self.bucket_xform,
            skip_mesh_approximation=True,
            hide_collision_shapes=True,
        )
        print(f"Bucket body id: {result['path_body_map']['/World/entity']}")

    def _add_scene_colliders(
        self,
        builder: newton.ModelBuilder,
        options,
        emit_lo: np.ndarray,
        emit_hi: np.ndarray,
        scoop_keyframes: list[tuple[float, wp.vec3, wp.quat]],
        voxel_size: float,
    ):
        """Load scene colliders and visual meshes, filtering far-field
        geometry with an AABB built from the emit bounds, scoop trajectory,
        and bucket position.
        """
        filter_pts = [
            np.asarray(emit_lo, dtype=np.float64),
            np.asarray(emit_hi, dtype=np.float64),
            np.array(self.bucket_pos, dtype=np.float64),
        ]
        for _t_kf, _pos, _q in scoop_keyframes:
            filter_pts.append(
                np.array([float(_pos[0]), float(_pos[1]), float(_pos[2])], dtype=np.float64)
            )
        filter_pts = np.stack(filter_pts, axis=0)
        margin = 0.35  # [m], covers scooper bowl + bucket mouth
        filter_lo = filter_pts.min(axis=0) - margin
        filter_hi = filter_pts.max(axis=0) + margin

        thin_mesh_thickness = voxel_size * 0.8
        cache_mode = options.mesh_cache_mode
        use_cache = cache_mode != "none"
        prefer_full = cache_mode == "full"
        print(f"Loading scene assets (cache mode: {cache_mode})...")
        scene_visuals, _ = load_scene(
            options.scene,
            builder=builder,
            collider_thickness=thin_mesh_thickness * 1.2,
            skip_visible_colliders=True,
            prefer_full_cache=prefer_full,
            use_cache=use_cache,
            collider_aabb_filter=(filter_lo, filter_hi),
        )

        visual_cfg = newton.ModelBuilder.ShapeConfig()
        visual_cfg.is_visible = True
        visual_cfg.has_shape_collision = False
        visual_cfg.has_particle_collision = False
        for mesh, xform in scene_visuals:
            mesh.finalize()
            builder.add_shape_mesh(body=-1, xform=xform, mesh=mesh, cfg=visual_cfg)

        ground_cfg = newton.ModelBuilder.ShapeConfig()
        ground_cfg.margin = 0.001
        builder.add_ground_plane(cfg=ground_cfg)

    def _create_solver(self, voxel_size: float):
        """Create the MPM solver and configure colliders."""
        thin_mesh_thickness = voxel_size * 0.8
        cfg = SolverImplicitMPM.Config(
            voxel_size=voxel_size,
            grid_type="fixed",
            critical_fraction=0.7,
            grid_padding=50,
            max_active_cell_count=1 << 11,
            max_iterations=20,
            solver="gauss-seidel",
            warmstart_mode="auto",
            collider_velocity_mode="finite_difference",
            sdf_sign_from_average_normal=False,
        )
        self.solver = SolverImplicitMPM(
            self.model, cfg, enable_timers=False, verbose=False,
        )

        self.solver.setup_collider(
            body_mass=wp.zeros_like(self.model.body_mass),
            collider_margins=[None, thin_mesh_thickness, thin_mesh_thickness],
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()
        self.scoop_xform_0 = wp.zeros(1, dtype=wp.transform, device=self.model.device)
        self.scoop_xform_1 = wp.zeros(1, dtype=wp.transform, device=self.model.device)

    def _setup_viewer(
        self,
        options,
        scoop_keyframes: list[tuple[float, wp.vec3, wp.quat]],
    ):
        """Configure viewer, bake keyframes to GPU, and set up recording."""
        self.instance_xforms = wp.zeros(
            self.num_particles, dtype=wp.transform, device=self.model.device,
        )

        dev = self.model.device
        self._kf_count = len(scoop_keyframes)
        self._kf_times = wp.array([t for t, _, _ in scoop_keyframes], dtype=float, device=dev)
        self._kf_pos = wp.array([p for _, p, _ in scoop_keyframes], dtype=wp.vec3, device=dev)
        self._kf_rot = wp.array([q for _, _, q in scoop_keyframes], dtype=wp.quat, device=dev)

        self.viewer.set_model(self.model)
        self.viewer.show_particles = options.show_particles
        # The built-in particle logger does two `.numpy()` readbacks every
        # frame to compact active particles — that forces a CPU-GPU sync and
        # hides pipelining gains. We already render popcorn ourselves via
        # `_render_popcorn()`, so stub it out to keep the step→render path
        # fully async.
        if not options.show_particles:
            self.viewer._log_particles = lambda _state: None

        self.viewer.set_camera(
            pos=wp.vec3(-0.5, 0.8, 1.8),
            pitch=-19.0,
            yaw=-36.0,
        )
        if hasattr(self.viewer, "renderer"):
            d = np.array((-1.0, 0.0, 0.3))
            self.viewer.renderer._sun_direction = d / np.linalg.norm(d)

        # FPS tracking
        self._step_count = 0
        self._fps_log_interval = 60
        self._frame_deltas: list[float] = []
        self._last_step_start_time: float | None = None

        self.recorder = VideoRecorder.from_options(
            self.viewer, int(self.fps), options,
        )
        self.capture()

    # ------------------------------------------------------------------
    def capture(self):
        self.graph = None
        self._capturing = True
        if wp.get_device().is_cuda and self.solver.grid_type == "fixed":
            if self.solver.max_active_cell_count < 0:
                print("CUDA Graph capture disabled (max_active_cell_count=-1 requires CPU readback)")
                self._capturing = False
                return
            if "jacobi" in self.solver.solver:
                print("CUDA Graph capture disabled (Jacobi solver requires dynamic allocation)")
                self._capturing = False
                return
            if self.solver.verbose:
                print("CUDA Graph capture disabled (verbose=True needs CPU readback each step)")
                self._capturing = False
                return
            if self.sim_substeps % 2 != 0:
                wp.utils.warn("Sim substeps must be even for graph capture of MPM step")
            else:
                with wp.ScopedCapture() as capture:
                    self.simulate()
                self.graph = capture.graph
        self._capturing = False

    def simulate(self):
        for i in range(self.sim_substeps):
            ratio_start = i / self.sim_substeps
            ratio_end = (i + 1) / self.sim_substeps

            wp.launch(
                set_body_transform_lerp_dual,
                dim=1,
                inputs=[
                    self.scoop_body_id,
                    ratio_start,
                    ratio_end,
                    self.scoop_xform_0,
                    self.scoop_xform_1,
                ],
                outputs=[self.state_0.body_q, self.state_1.body_q],
                device=self.model.device,
            )

            self.state_0.clear_forces()
            self.solver.step(self.state_0, self.state_1, None, None, self.sim_dt)
            self.solver.project_outside(self.state_1, self.state_1, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

        self.solver.update_particle_frames(self.state_0, self.state_0, self.frame_dt)

    def step(self):
        now = time.perf_counter()
        if self._last_step_start_time is not None:
            self._frame_deltas.append(now - self._last_step_start_time)
        self._last_step_start_time = now

        wp.launch(
            eval_trajectory_pair,
            dim=1,
            inputs=[
                self._kf_times, self._kf_pos, self._kf_rot,
                self._kf_count,
                self.sim_time, self.sim_time + self.frame_dt,
            ],
            outputs=[self.scoop_xform_0, self.scoop_xform_1],
            device=self.model.device,
        )

        if self.graph is not None:
            wp.capture_launch(self.graph)
        else:
            self.simulate()

        self._step_count += 1
        if self._step_count % self._fps_log_interval == 0 and self._frame_deltas:
            n = self._fps_log_interval
            recent = self._frame_deltas[-n:]
            avg_frame = sum(recent) / len(recent)
            fps = 1.0 / avg_frame if avg_frame > 0 else float("inf")

            total = sum(self._frame_deltas)
            overall_fps = len(self._frame_deltas) / total if total > 0 else 0.0

            print(
                f"[step {self._step_count:>5d}] "
                f"sim_t={self.sim_time:.2f}s | "
                f"frame={avg_frame * 1000:.1f}ms | "
                f"FPS: {fps:.1f} (avg {overall_fps:.1f})"
            )

        self.sim_time += self.frame_dt

        if self.recorder and self.sim_time >= self._trajectory_end_time:
            print(f"Trajectory finished at sim_time={self.sim_time:.2f}s, stopping recording.")
            self.viewer.close()

    def _render_popcorn(self):
        wp.launch(
            build_instance_xforms,
            dim=self.num_particles,
            inputs=[
                self.state_0.particle_q,
                self.state_0.mpm.particle_transform,
                self.instance_xforms,
            ],
            device=self.model.device,
        )
        self.viewer.log_shapes(
            "/popcorn",
            newton.GeoType.MESH,
            (1.0, 1.0, 1.0),
            self.instance_xforms,
            self._popcorn_color,
            self._popcorn_material,
            geo_src=self.render_mesh,
        )

    def render(self):
        self.viewer.begin_frame(self.sim_time)

        self.viewer.log_state(self.state_0)

        if not self.viewer.show_particles:
            self._render_popcorn()
        else:
            self.viewer.log_shapes(
                "/popcorn",
                newton.GeoType.MESH,
                (1.0, 1.0, 1.0),
                self.instance_xforms,
                self._popcorn_color,
                self._popcorn_material,
                geo_src=self.render_mesh,
            )

        self.viewer.end_frame()
        if self.recorder:
            self.recorder.capture_frame()

    def test_final(self):
        pass


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument("--voxel-size", "-dx", type=float, default=0.02)
    parser.add_argument("--particles-per-cell", "-ppc", type=float, default=1.0)
    parser.add_argument("--emit-lo", type=float, nargs=3, default=None,
                        help="Override lower bound of particle box [m]")
    parser.add_argument("--emit-hi", type=float, nargs=3, default=None,
                        help="Override upper bound of particle box [m]")
    parser.add_argument(
        "--scene", type=str, default=SCENE_USD_PATH,
        help="Path to the USD scene file",
    )
    parser.add_argument("--show-particles", action="store_true",
                        help="Render particles as point clouds via log_state")
    parser.add_argument(
        "--mesh-cache-mode",
        type=str,
        choices=["full", "decimated", "none"],
        default="decimated",
        help="Visual mesh loading strategy: 'full' = cached original meshes "
             "(with UVs/texture), 'decimated' = low-poly cached meshes "
             "(fast, no texture), 'none' = load directly from USD (slowest)",
    )
    parser.add_argument("--young-modulus", type=float, default=1.0e5,
                        help="Young's modulus for popcorn particles [Pa]")
    parser.add_argument("--yield-stress", type=float, default=100.0,
                        help="Yield stress for popcorn particles [Pa]")
    parser.add_argument("--yield-pressure", type=float, default=1.0e3,
                        help="Yield pressure for popcorn particles [Pa]")
    VideoRecorder.add_args(parser)

    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    try:
        newton.examples.run(example, args)
    finally:
        if example.recorder:
            example.recorder.finalize()
