"""Visualize a USD scene file as static background assets.

Loads all visual Mesh prims from a USD stage, applies their world transforms,
and renders them via Newton's viewer.  Designed to be composed with simulation
scripts (e.g. the popcorn MPM demo) by importing ``load_scene_meshes``.

Usage:
    uv run python visualize_scene_usd.py
    uv run python visualize_scene_usd.py --viewer rerun
    uv run python visualize_scene_usd.py --scene /path/to/other/scene.usda
"""

import hashlib
import tempfile
from pathlib import Path

import numpy as np
import warp as wp
from pxr import Gf, Usd, UsdGeom, UsdLux, UsdPhysics

import newton
import newton.examples
import newton.usd

_MESH_CACHE_DIR = Path(__file__).parent / ".mesh_cache"


SCENE_USD_PATH = (
    "/home/yvetted/project/genie_sim/source/geniesim/assets/background/popcorn/popcorn_1/background.usda"
)


def _mesh_cache_paths(scene_path: str, prim_path: str) -> tuple[Path, Path]:
    """Compute ``(full_file, decimated_file)`` cache paths for a prim.

    The hash is taken over ``"{scene_path}::{prim_path}"`` so entries are
    stable across runs but scene- and prim-specific.
    """
    key = f"{scene_path}::{prim_path}"
    h = hashlib.sha256(key.encode()).hexdigest()[:16]
    return _MESH_CACHE_DIR / f"{h}.full.npz", _MESH_CACHE_DIR / f"{h}.npz"


def _try_load_cached_mesh(
    scene_path: str, prim_path: str, *, prefer_full: bool = True
) -> newton.Mesh | None:
    """Try to load a cached mesh from the .mesh_cache directory.

    Two cache variants are supported:

    * **Full-fidelity** (``<hash>.full.npz``) — original vertices, indices,
      normals, UVs, and texture path.  Rendering quality is identical to
      loading from USD, but much faster.
    * **Decimated** (``<hash>.npz``) — reduced triangle count, no UVs.
      Fastest loading, suitable for background objects.

    Args:
        scene_path: USD stage file path (used for hash).
        prim_path: Prim path inside the stage (used for hash).
        prefer_full: When ``True`` (default), try the full-fidelity cache
            first, then fall back to decimated.  Set to ``False`` to always
            prefer the decimated version.
    """
    if not _MESH_CACHE_DIR.is_dir():
        return None
    full_file, dec_file = _mesh_cache_paths(scene_path, prim_path)
    candidates = [full_file, dec_file] if prefer_full else [dec_file, full_file]
    for cache_file in candidates:
        if cache_file.exists():
            return _load_mesh_from_npz(cache_file)
    return None


def _load_mesh_from_npz(cache_file: Path) -> newton.Mesh:
    """Reconstruct a :class:`newton.Mesh` from an ``.npz`` cache file."""
    data = np.load(cache_file, allow_pickle=True)
    vertices = data["vertices"]
    indices = data["indices"]

    color_arr = data["color"]
    color = tuple(float(c) for c in color_arr) if len(color_arr) > 0 else None
    roughness = float(data["roughness"])
    roughness = roughness if roughness >= 0 else None
    metallic = float(data["metallic"])
    metallic = metallic if metallic >= 0 else None

    normals = data["normals"] if "normals" in data else None
    uvs = data["uvs"] if "uvs" in data else None
    texture_str = str(data["texture"]) if "texture" in data else ""
    texture = texture_str if texture_str else None

    mesh = newton.Mesh(vertices, indices, compute_inertia=False)
    mesh.color = color
    mesh._roughness = roughness
    mesh._metallic = metallic
    if normals is not None:
        mesh._normals = normals
    if uvs is not None:
        mesh._uvs = uvs
    if texture is not None:
        mesh.texture = texture
    return mesh


def _save_mesh_to_npz(mesh: newton.Mesh, cache_file: Path) -> None:
    """Serialize a :class:`newton.Mesh` to an ``.npz`` cache file.

    Inverse of :func:`_load_mesh_from_npz`. Uses temp-file + atomic
    ``rename`` so a concurrent reader never observes a half-written
    file. ``color``/``roughness``/``metallic`` use the same sentinels
    as the loader (empty array / -1), so the format round-trips.

    Texture arrays (in-memory image data) are not written; only string
    texture paths are persisted, matching what the loader can consume.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    indices = np.asarray(mesh.indices, dtype=np.int32)

    color = mesh.color
    color_arr = (
        np.asarray(color, dtype=np.float32) if color is not None else np.zeros(0, dtype=np.float32)
    )
    roughness = np.float32(mesh._roughness if mesh._roughness is not None else -1.0)
    metallic = np.float32(mesh._metallic if mesh._metallic is not None else -1.0)

    kwargs: dict[str, np.ndarray] = {
        "vertices": vertices,
        "indices": indices,
        "color": color_arr,
        "roughness": roughness,
        "metallic": metallic,
    }
    if mesh._normals is not None:
        kwargs["normals"] = np.asarray(mesh._normals, dtype=np.float32)
    if mesh._uvs is not None:
        kwargs["uvs"] = np.asarray(mesh._uvs, dtype=np.float32)
    if isinstance(mesh.texture, str) and mesh.texture:
        kwargs["texture"] = np.asarray(mesh.texture)

    _MESH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Write via a file handle in the same directory so (a) np.savez does not
    # auto-append ``.npz`` to our temp name, and (b) ``Path.replace`` is a
    # same-filesystem atomic rename.
    with tempfile.NamedTemporaryFile(
        mode="wb",
        dir=_MESH_CACHE_DIR,
        prefix=cache_file.name + ".",
        suffix=".tmp",
        delete=False,
    ) as tmp_fh:
        np.savez(tmp_fh, **kwargs)
        tmp_path = Path(tmp_fh.name)
    try:
        tmp_path.replace(cache_file)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def _compute_world_aabb_for_prim(
    prim: Usd.Prim,
    mesh: newton.Mesh,
    xform_cache: UsdGeom.XformCache,
) -> tuple[np.ndarray, np.ndarray]:
    """World-space AABB of ``mesh`` under ``prim``'s full transform.

    Uses the complete 4x4 local-to-world matrix (preserves non-uniform
    scale authored in USD, which would be dropped by the quaternion-based
    ``wp.transform`` returned from :func:`newton.usd.get_transform`). The
    matrix is row-vector convention (``p_world = p_local @ M``) which
    matches USD's :class:`pxr.Gf.Matrix4d`.
    """
    world_mat = xform_cache.GetLocalToWorldTransform(prim)
    mat_np = np.asarray(world_mat, dtype=np.float64).reshape(4, 4)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    if verts.size == 0:
        # Fall back to the prim's origin if the mesh has no vertices.
        origin = mat_np[3, :3]
        return origin.copy(), origin.copy()
    verts_h = np.concatenate([verts, np.ones((len(verts), 1))], axis=1)
    world = verts_h @ mat_np  # (N, 4), row-vector convention
    w = world[:, 3:4]
    w = np.where(np.abs(w) < 1e-12, 1.0, w)
    world_xyz = world[:, :3] / w
    return world_xyz.min(axis=0), world_xyz.max(axis=0)


def _aabbs_overlap(
    alo: np.ndarray, ahi: np.ndarray, blo: np.ndarray, bhi: np.ndarray
) -> bool:
    """Return True iff two 3D axis-aligned boxes intersect (inclusive)."""
    return bool(np.all(alo <= bhi) and np.all(blo <= ahi))


def load_scene(
    scene_path: str,
    builder: newton.ModelBuilder | None = None,
    collider_thickness: float = 0.0,
    skip_visible_colliders: bool = False,
    prefer_full_cache: bool = True,
    use_cache: bool = True,
    write_cache: bool = True,
    collider_aabb_filter: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[list[tuple[newton.Mesh, wp.transform]], list[tuple[newton.Mesh, wp.transform]]]:
    """Load visual meshes and collision shapes from a USD scene in one pass.

    Traverses the stage once, classifying each ``Mesh`` prim as either a
    visual mesh or a collider shape (or both, though visible colliders can
    be skipped with *skip_visible_colliders*).

    Args:
        scene_path: File path to the USD stage.
        builder: If provided, collision shapes are registered as static
            shapes (body -1) on this builder.
        collider_thickness: Collision margin [m] for each collider shape.
        skip_visible_colliders: If ``True``, skip collision prims whose
            computed USD visibility is not ``invisible``.
        prefer_full_cache: If ``True`` (default), prefer full-fidelity
            cached meshes (with UVs/texture) over decimated ones.
        use_cache: If ``False``, skip cache lookup entirely and always
            load from USD.
        write_cache: If ``True`` (default), save freshly-loaded visual
            meshes to ``<hash>.full.npz`` so subsequent runs hit the
            disk cache. Has no effect on prims that already hit the
            cache, nor on collider meshes. Set to ``False`` (together
            with ``use_cache=False``) to leave the cache directory
            completely untouched.
        collider_aabb_filter: Optional ``(lo, hi)`` world-space AABB. If
            provided, collider prims whose transformed mesh AABB does not
            intersect this box are dropped (neither registered on the
            builder nor returned in the ``colliders`` list). Visual meshes
            are never filtered. Useful for excluding far-away colliders
            that cannot influence a particle simulation region, cutting
            MPM rasterization cost.

    Returns:
        ``(visual_meshes, colliders)`` — each a list of
        ``(mesh, world_transform)`` pairs.
    """
    stage = Usd.Stage.Open(scene_path)
    if stage is None:
        raise FileNotFoundError(f"Cannot open USD stage: {scene_path}")

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    visuals: list[tuple[newton.Mesh, wp.transform]] = []
    colliders: list[tuple[newton.Mesh, wp.transform]] = []

    collider_cfg = newton.ModelBuilder.ShapeConfig()
    collider_cfg.is_visible = False
    collider_cfg.has_particle_collision = True
    collider_cfg.has_shape_collision = True
    collider_cfg.margin = collider_thickness

    if collider_aabb_filter is not None:
        filter_lo = np.asarray(collider_aabb_filter[0], dtype=np.float64).reshape(3)
        filter_hi = np.asarray(collider_aabb_filter[1], dtype=np.float64).reshape(3)
        print(
            f"  [collider filter] keep only colliders intersecting "
            f"AABB lo={filter_lo.tolist()} hi={filter_hi.tolist()}"
        )
    else:
        filter_lo = filter_hi = None

    import time as _time

    n_collider_skipped = 0

    for prim in stage.Traverse():
        if prim.GetTypeName() != "Mesh":
            continue

        prim_path = str(prim.GetPath())
        imageable = UsdGeom.Imageable(prim)
        vis = imageable.ComputeVisibility(Usd.TimeCode.Default())

        xform = newton.usd.get_transform(
            prim, local=False, xform_cache=xform_cache
        )

        # --- Collider? ---
        is_collider = prim.HasAPI(UsdPhysics.CollisionAPI)
        if is_collider:
            enabled_attr = prim.GetAttribute("physics:collisionEnabled")
            if enabled_attr and enabled_attr.IsAuthored() and not enabled_attr.Get():
                is_collider = False

        if is_collider:
            if skip_visible_colliders and vis != UsdGeom.Tokens.invisible:
                pass  # fall through to visual mesh handling below
            else:
                _t = _time.perf_counter()
                mesh = newton.usd.get_mesh(prim)
                mesh.finalize()
                if filter_lo is not None:
                    wlo, whi = _compute_world_aabb_for_prim(prim, mesh, xform_cache)
                    if not _aabbs_overlap(wlo, whi, filter_lo, filter_hi):
                        n_collider_skipped += 1
                        print(
                            f"  collider {prim_path}: SKIPPED (outside filter AABB, "
                            f"lo={wlo.tolist()} hi={whi.tolist()})"
                        )
                        continue
                if builder is not None:
                    builder.add_shape_mesh(body=-1, xform=xform, mesh=mesh, cfg=collider_cfg)
                colliders.append((mesh, xform))
                print(
                    f"  collider {prim_path}: "
                    f"{len(mesh.vertices)} verts, {len(mesh.indices)} tris "
                    f"({_time.perf_counter() - _t:.3f}s)"
                )
                continue

        # --- Visual mesh ---
        if "collider" in prim_path.lower():
            continue
        if vis == UsdGeom.Tokens.invisible:
            continue

        _t = _time.perf_counter()
        cached = _try_load_cached_mesh(scene_path, prim_path, prefer_full=prefer_full_cache) if use_cache else None
        if cached is not None:
            mesh = cached
            tag = "cached"
        else:
            mesh = newton.usd.get_mesh(prim, load_normals=True, load_uvs=True)
            tag = "loaded"
            if write_cache:
                full_file, _ = _mesh_cache_paths(scene_path, prim_path)
                try:
                    _save_mesh_to_npz(mesh, full_file)
                    tag = "loaded+saved"
                except Exception as e:
                    print(f"    [warn] failed to write mesh cache for {prim_path}: {e}")
        visuals.append((mesh, xform))
        print(
            f"  visual ({tag}) {prim_path}: "
            f"{len(mesh.vertices)} verts, {len(mesh.indices)} tris "
            f"({_time.perf_counter() - _t:.3f}s)"
        )

    suffix = ""
    if collider_aabb_filter is not None:
        suffix = f" (filtered out {n_collider_skipped} far collider(s))"
    print(
        f"Loaded {len(visuals)} visual mesh(es), "
        f"{len(colliders)} collider(s) from {scene_path}{suffix}"
    )
    return visuals, colliders


def load_scene_meshes(
    scene_path: str,
) -> list[tuple[newton.Mesh, wp.transform]]:
    """Load all visual Mesh prims from a USD scene.

    Returns a list of ``(mesh, world_transform)`` pairs.  Only prims whose
    ``typeName`` is ``Mesh`` under the ``/World`` subtree are included;
    collider meshes (paths containing ``collider``) are skipped.
    """
    visuals, _ = load_scene(scene_path)
    return visuals


def load_scene_colliders(
    builder: newton.ModelBuilder,
    scene_path: str,
    thickness: float = 0.0,
    skip_visible: bool = False,
) -> list[tuple[newton.Mesh, wp.transform]]:
    """Load collision-enabled Mesh prims from a USD scene into *builder*.

    Traverses the stage looking for ``Mesh`` prims that carry the
    ``PhysicsCollisionAPI`` with ``collisionEnabled`` not explicitly ``False``.
    Each mesh is added as a static collision shape on the world body
    (body -1) so the MPM solver picks it up automatically.

    Args:
        builder: The :class:`~newton.ModelBuilder` to populate.
        scene_path: File path to the USD stage.
        thickness: Collision margin [m] written to each shape's
            :attr:`~newton.ModelBuilder.ShapeConfig.margin`.
        skip_visible: If ``True``, skip meshes whose computed USD visibility
            is not ``invisible``.  This filters out high-poly visual meshes
            that carry ``PhysicsCollisionAPI`` but are not intended as
            simulation colliders (the actual collision bodies are typically
            marked invisible in the USD stage).

    Returns:
        List of ``(mesh, world_transform)`` pairs for optional visualisation.
    """
    _, colliders = load_scene(
        scene_path,
        builder=builder,
        collider_thickness=thickness,
        skip_visible_colliders=skip_visible,
    )
    return colliders


def _color_temperature_to_rgb(kelvin: float) -> tuple[float, float, float]:
    """Convert colour temperature in Kelvin to linear RGB (Tanner Helland)."""
    temp = max(1000.0, min(kelvin, 40000.0)) / 100.0
    if temp <= 66.0:
        r = 1.0
        g = max(0.3900815787 * np.log(temp) - 0.6318414438, 0.0)
    else:
        r = max(1.2929362 * (temp - 60.0) ** -0.1332047592, 0.0)
        g = max(1.1298909 * (temp - 60.0) ** -0.0755148492, 0.0)
    if temp >= 66.0:
        b = 1.0
    elif temp <= 19.0:
        b = 0.0
    else:
        b = max(0.5432068 * np.log(temp - 10.0) - 1.1962541, 0.0)
    return (min(r, 1.0), min(g, 1.0), min(b, 1.0))


def load_scene_lights(
    scene_path: str,
) -> list[dict]:
    """Load SphereLight prims from a USD scene.

    Returns a list of dicts with keys ``position``, ``color``, ``intensity``,
    and ``radius``.
    """
    stage = Usd.Stage.Open(scene_path)
    if stage is None:
        return []

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    lights: list[dict] = []

    for prim in stage.Traverse():
        if not prim.IsA(UsdLux.SphereLight):
            continue

        imageable = UsdGeom.Imageable(prim)
        if imageable.ComputeVisibility(Usd.TimeCode.Default()) == UsdGeom.Tokens.invisible:
            continue

        light = UsdLux.SphereLight(prim)
        intensity = light.GetIntensityAttr().Get() or 10000.0
        radius = light.GetRadiusAttr().Get() or 0.5

        enable_temp = light.GetEnableColorTemperatureAttr().Get()
        if enable_temp:
            temp = light.GetColorTemperatureAttr().Get() or 6500.0
            color = _color_temperature_to_rgb(temp)
        else:
            c = light.GetColorAttr().Get() or Gf.Vec3f(1, 1, 1)
            color = (float(c[0]), float(c[1]), float(c[2]))

        world_xform = xform_cache.GetLocalToWorldTransform(prim)
        pos = world_xform.ExtractTranslation()

        lights.append({
            "position": (float(pos[0]), float(pos[1]), float(pos[2])),
            "color": color,
            "intensity": float(intensity),
            "radius": float(radius),
        })
        print(
            f"  light {prim.GetPath()}: pos={pos}, "
            f"intensity={intensity}, color={color}, radius={radius}"
        )

    print(f"Loaded {len(lights)} light(s) from {scene_path}")
    return lights


class Example:
    def __init__(self, viewer, args):
        self.fps = 60.0
        self.frame_dt = 1.0 / self.fps
        self.sim_time = 0.0
        self.viewer = viewer

        scene_path = args.scene
        self.scene_meshes = load_scene_meshes(scene_path)
        self.scene_lights = load_scene_lights(scene_path)

        self.mesh_render_data: list[
            tuple[newton.Mesh, wp.array, wp.array, wp.array]
        ] = []
        for mesh, xform in self.scene_meshes:
            mesh.finalize()
            xforms = wp.array([xform], dtype=wp.transform)

            c = mesh.color if mesh.color is not None else (0.75, 0.65, 0.50)
            color = wp.array(
                [wp.vec3(float(c[0]), float(c[1]), float(c[2]))], dtype=wp.vec3
            )
            r = mesh._roughness if mesh._roughness is not None else 0.6
            m = mesh._metallic if mesh._metallic is not None else 0.0
            tex = 1.0 if (mesh.texture is not None and mesh._uvs is not None) else 0.0
            material = wp.array(
                [wp.vec4(r, m, 0.0, tex)], dtype=wp.vec4
            )
            self.mesh_render_data.append((mesh, xforms, color, material))

        self.ground_color = wp.array(
            [wp.vec3(0.15, 0.15, 0.18)], dtype=wp.vec3
        )
        self.ground_mat = wp.array(
            [wp.vec4(0.5, 0.5, 1.0, 0.0)], dtype=wp.vec4
        )

        builder = newton.ModelBuilder()
        self.scene_colliders = load_scene_colliders(
            builder, scene_path, thickness=0.025
        )
        builder.add_ground_plane()
        self.model = builder.finalize()
        self.viewer.set_model(self.model)

        self.collider_render_data: list[
            tuple[newton.Mesh, wp.array, wp.array, wp.array]
        ] = []
        collider_color = wp.array(
            [wp.vec3(0.2, 0.8, 0.3)], dtype=wp.vec3
        )
        collider_mat = wp.array(
            [wp.vec4(0.8, 0.0, 0.0, 0.0)], dtype=wp.vec4
        )
        for mesh, xform in self.scene_colliders:
            xforms = wp.array([xform], dtype=wp.transform)
            self.collider_render_data.append(
                (mesh, xforms, collider_color, collider_mat)
            )

        self.viewer.set_camera(
            pos=wp.vec3(2.5, -1.5, 2.0),
            pitch=-15.0,
            yaw=-150.0,
        )

        self.light_render_data: list[tuple[wp.array, wp.array, wp.array]] = []
        for lt in self.scene_lights:
            p = lt["position"]
            c = lt["color"]
            xforms = wp.array(
                [wp.transform(p, wp.quat_identity())], dtype=wp.transform
            )
            color = wp.array(
                [wp.vec3(float(c[0]), float(c[1]), float(c[2]))],
                dtype=wp.vec3,
            )
            material = wp.array(
                [wp.vec4(0.0, 0.0, 0.0, 0.0)], dtype=wp.vec4
            )
            self.light_render_data.append((xforms, color, material))

    def step(self):
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)

        show_visual = getattr(self.viewer, "show_visual", True)
        show_collision = getattr(self.viewer, "show_collision", False)

        for i, (mesh, xforms, color, material) in enumerate(
            self.mesh_render_data
        ):
            self.viewer.log_shapes(
                f"/scene/mesh_{i}",
                newton.GeoType.MESH,
                (1.0, 1.0, 1.0),
                xforms,
                color,
                material,
                geo_src=mesh,
                hidden=not show_visual,
            )

        for i, (mesh, xforms, color, material) in enumerate(
            self.collider_render_data
        ):
            self.viewer.log_shapes(
                f"/scene/collider_{i}",
                newton.GeoType.MESH,
                (1.0, 1.0, 1.0),
                xforms,
                color,
                material,
                geo_src=mesh,
                hidden=not show_collision,
            )

        for i, (xforms, color, material) in enumerate(
            self.light_render_data
        ):
            r = self.scene_lights[i]["radius"]
            self.viewer.log_shapes(
                f"/scene/light_{i}",
                newton.GeoType.SPHERE,
                (r,),
                xforms,
                color,
                material,
                hidden=not show_visual,
            )

        self.viewer.log_shapes(
            "/ground",
            newton.GeoType.PLANE,
            (20.0, 20.0),
            wp.array([wp.transform_identity()], dtype=wp.transform),
            self.ground_color,
            self.ground_mat,
        )

        self.viewer.end_frame()

    def test_final(self):
        pass


if __name__ == "__main__":
    parser = newton.examples.create_parser()
    parser.add_argument(
        "--scene",
        type=str,
        default=SCENE_USD_PATH,
        help="Path to the USD scene file to visualize",
    )

    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    newton.examples.run(example, args)
