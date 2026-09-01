# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Simulate the full H2 humanoid with all six closed loops
# (VBD sparse articulation)
#
# `H2_loop.urdf` encodes the spanning tree plus paired `*_connect_lhs/rhs`
# marker links: the two attachment points of a *missing* rigid pushrod per
# loop. This example reconstructs each pushrod as a capsule body with a
# ball joint at each end, closing six loops:
#   - 2x ankle (A-crank drives the passive ankle pitch, rod ~371 mm)
#   - 2x knee (knee motor four-bar, rod ~172 mm)
#   - 2x waist (L/R torso constraint rods, ~65 mm, passive support; the
#     rockers act as crank arms when the waist is driven)
#
# The pelvis is fixed in the air; ankle A-cranks and knee motors are driven
# with sinusoids while knees/ankle pitches are passive linkage outputs. The
# waist rods are rigid and passive; waist motion comes either from driving
# the rockers as crank arms (--drive waist/all, --waist-antiphase for pitch)
# or from direct-driving waist_pitch (--waist-direct). Rod-end joint types
# are selectable for ankle/knee: --ankle-rod ss|su, --knee-rod ss|rr|su.
#
# Command:
#   uv run --extra examples python sim_h2_loop.py --viewer gl
#   uv run --extra examples python sim_h2_loop.py --viewer null --diagnose
###########################################################################

from __future__ import annotations

import math
import os
import xml.etree.ElementTree as ET

import numpy as np
import warp as wp

import newton
import newton.examples

DEFAULT_URDF = "/home/yvetted/Downloads/unitree_ros_h2/robots/h2_description/H2_loop.urdf"
DEFAULT_MJCF = os.path.join(os.path.dirname(os.path.abspath(__file__)), "h2_description", "H2loop_complete.xml")

_DRIVE_FREQUENCY = 0.25  # Hz

# Placeholder-mass links (markers, motor cranks) make the system singular.
_MIN_BODY_MASS = 0.02
_MIN_BODY_INERTIA = 2.0e-6

_ROD_COLOR = wp.vec3(0.80, 0.68, 0.28)
_ROD_RADIUS = 0.008
_ROD_LINEAR_DENSITY = 0.3  # kg/m for reconstructed pushrods

# (link_a, marker_a, link_b, marker_b, rod_label)
_LOOPS = (
    (
        "left_ankle_pitch_link",
        "left_ankle_connect_rhs",
        "left_ankle_A_link",
        "left_ankle_connect_lhs",
        "left_ankle_rod",
    ),
    (
        "right_ankle_pitch_link",
        "right_ankle_connect_lhs",
        "right_ankle_A_link",
        "right_ankle_connect_rhs",
        "right_ankle_rod",
    ),
    ("left_knee_link", "left_knee_connect_rhs", "left_knee_motor_link", "left_knee_connect_lhs", "left_knee_rod"),
    ("right_knee_link", "right_knee_connect_lhs", "right_knee_motor_link", "right_knee_connect_rhs", "right_knee_rod"),
    ("waist_yaw_link", "waist_connect_lhs", "torso_constraint_L_link", "torso_connect_lhs", "waist_rod_L"),
    ("waist_yaw_link", "waist_connect_rhs", "torso_constraint_R_link", "torso_connect_rhs", "waist_rod_R"),
)

# Same six loops, but anchored on the MJCF's rod sites instead of the URDF's
# `*_connect_*` marker bodies. The MJCF carries the sole capsules and the
# per-link collision whitelist, which the URDF does not.
# (link_a, site_a, link_b, site_b, rod_label)
_LOOPS_MJCF = (
    (
        "left_ankle_pitch_link",
        "left_ankle_output_rod_site",
        "left_ankle_A_link",
        "left_ankle_motor_rod_site",
        "left_ankle_rod",
    ),
    (
        "right_ankle_pitch_link",
        "right_ankle_output_rod_site",
        "right_ankle_A_link",
        "right_ankle_motor_rod_site",
        "right_ankle_rod",
    ),
    ("left_knee_link", "left_knee_output_rod_site", "left_knee_motor_link", "left_knee_motor_rod_site", "left_knee_rod"),
    (
        "right_knee_link",
        "right_knee_output_rod_site",
        "right_knee_motor_link",
        "right_knee_motor_rod_site",
        "right_knee_rod",
    ),
    ("waist_yaw_link", "left_waist_output_rod_site", "torso_constraint_L_link", "left_waist_motor_rod_site", "waist_rod_L"),
    (
        "waist_yaw_link",
        "right_waist_output_rod_site",
        "torso_constraint_R_link",
        "right_waist_motor_rod_site",
        "waist_rod_R",
    ),
)


# Linkage outputs: constrained by the reconstructed rods, not directly driven.
_PASSIVE_JOINTS = (
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "torso_constraint_L_joint",
    "torso_constraint_R_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)

# Sinusoidally driven cranks: name -> amplitude [rad] (knee motors get a
# raised-cosine flexion profile so they stay inside [-0.2, 2.9]).
_DRIVEN_JOINTS = {
    "left_ankle_A_joint": math.radians(12.0),
    "right_ankle_A_joint": math.radians(12.0),
    "left_knee_motor_joint": math.radians(25.0),
    "right_knee_motor_joint": math.radians(25.0),
}


_MJCF_ATTRIBUTES = ("armature", "damping", "frictionloss")


def _load_mjcf_joint_params(path: str) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Read per-joint physical parameters from a MuJoCo XML.

    Returns a name -> params map for named joints (armature [kg·m²],
    damping [N·m·s/rad], frictionloss [N·m], effort_limit [N·m] from
    ``actuatorfrcrange``), plus the defaults from ``<default><joint>``.
    """
    root = ET.parse(path).getroot()
    defaults = {}
    default_joint = root.find("./default/joint")
    if default_joint is not None:
        for attr in _MJCF_ATTRIBUTES:
            if attr in default_joint.attrib:
                defaults[attr] = float(default_joint.get(attr))
    per_joint = {}
    for joint in root.iter("joint"):
        name = joint.get("name")
        if name is None:
            continue
        params = {attr: float(joint.get(attr)) for attr in _MJCF_ATTRIBUTES if attr in joint.attrib}
        if "actuatorfrcrange" in joint.attrib:
            lo, hi = (float(v) for v in joint.get("actuatorfrcrange").split())
            params["effort_limit"] = max(abs(lo), abs(hi))
        if params:
            per_joint[name] = params
    return per_joint, defaults


# The MJCF names the ankle motor cranks differently from the URDF.
_MJCF_JOINT_ALIASES = {
    "left_ankle_A_joint": "left_ankle_motor_joint",
    "right_ankle_A_joint": "right_ankle_motor_joint",
}


def _apply_mjcf_joint_params(builder: newton.ModelBuilder, path: str) -> None:
    """Override the URDF joints' physical parameters with the MJCF-defined values.

    Notes on solver support (SolverVBD): armature needs
    ``rigid_joint_armature=True`` (revolute only); passive ``damping`` is
    applied through ``joint_target_kd`` since VBD reads kd as absolute damping
    and ignores target modes; ``joint_friction`` (frictionloss) and
    ``joint_effort_limit`` are stored on the model but not used by VBD.
    """
    per_joint, defaults = _load_mjcf_joint_params(path)
    matched = 0
    for j, label in enumerate(builder.joint_label):
        name = label.rsplit("/", 1)[-1]
        params = per_joint.get(_MJCF_JOINT_ALIASES.get(name, name))
        if params is None:
            if builder.joint_type[j] != int(newton.JointType.REVOLUTE):
                continue
            params = defaults
        else:
            matched += 1
            params = {**defaults, **params}
        qd_start = builder.joint_qd_start[j]
        lin, ang = builder.joint_dof_dim[j]
        for dof in range(qd_start, qd_start + lin + ang):
            if "armature" in params:
                builder.joint_armature[dof] = params["armature"]
            if "damping" in params:
                builder.joint_damping[dof] = params["damping"]
                # VBD applies joint_target_kd as absolute damping regardless of
                # target mode; _configure_drives later raises kd on driven/held
                # joints, so this survives only on passive linkage joints.
                builder.joint_target_kd[dof] = max(builder.joint_target_kd[dof], params["damping"])
            if "frictionloss" in params:
                builder.joint_friction[dof] = params["frictionloss"]
            if "effort_limit" in params:
                builder.joint_effort_limit[dof] = params["effort_limit"]
    print(f"Applied MJCF joint params: {matched}/{len(per_joint)} named joints matched, defaults={defaults}")


def _find(labels: list[str], suffix: str) -> int:
    for i, label in enumerate(labels):
        if label.rsplit("/", 1)[-1] == suffix:
            return i
    raise KeyError(f"Missing '{suffix}' in {[l.rsplit('/', 1)[-1] for l in labels]}")


def _site_world_pos(builder: newton.ModelBuilder, site_label: str) -> np.ndarray:
    """World position of an imported MJCF site.

    ``add_mjcf`` turns each ``<site>`` into a non-colliding shape on its parent
    body, so the anchor is the body pose composed with the shape's local xform.
    """
    for si, label in enumerate(builder.shape_label):
        if label.rsplit("/", 1)[-1] != site_label:
            continue
        body = builder.shape_body[si]
        x_b = wp.transform_identity() if body < 0 else wp.transform(*builder.body_q[body])
        return np.array(x_b * wp.transform(*builder.shape_transform[si]))[:3]
    raise KeyError(f"Missing site '{site_label}' among the imported shapes")


def _size_infinite_planes(builder: newton.ModelBuilder, extent: float) -> int:
    """Add a drawable, non-colliding companion quad for each infinite plane.

    MuJoCo encodes an unbounded plane as ``size="0 0 spacing"``; the importer
    carries the zeros into the shape scale, so the viewer renders a 0x0 quad
    and the floor is invisible. Resizing the collision plane itself is NOT an
    option: a finite plane collides as a bounded quad, and a full-body impact
    against it injects energy (>800 m/s runaway, robot tunnels through) where
    the infinite half-space settles at ~20 m/s. Verified on both the sparse
    articulation fork and the upstream vbd-sparse-direct branch; repro lives in
    the vbd-stable-eval branch (repro_h2_impact_blowup.py --finite-floor).

    Returns:
        The number of visual quads that were added.
    """
    added = 0
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.density = 0.0
    cfg.has_shape_collision = False
    cfg.has_particle_collision = False
    for si in range(len(builder.shape_type)):
        if int(builder.shape_type[si]) != int(newton.GeoType.PLANE):
            continue
        scale = builder.shape_scale[si]
        if scale[0] > 0.0 or scale[1] > 0.0:
            continue
        builder.add_shape_plane(
            width=extent,
            length=extent,
            body=builder.shape_body[si],
            xform=wp.transform(*builder.shape_transform[si]) if builder.shape_body[si] >= 0 else wp.transform(*builder.shape_transform[si]),
            cfg=cfg,
            label=f"{builder.shape_label[si].rsplit('/', 1)[-1]}_visual_quad",
        )
        added += 1
    return added


def _rest_pose_fk(builder: newton.ModelBuilder) -> None:
    """Propagate rest (q=0) poses through the tree joints.

    On this branch `add_urdf` leaves every `builder.body_q` at identity, so the
    imported rest state is inconsistent with the joint frames. URDF joints are
    emitted parents-first (DFS), so one forward pass assembles the mechanism.
    """
    for j in range(len(builder.joint_type)):
        parent = builder.joint_parent[j]
        child = builder.joint_child[j]
        x_p = wp.transform(*builder.joint_X_p[j])
        x_c = wp.transform(*builder.joint_X_c[j])
        t_parent = wp.transform_identity() if parent < 0 else wp.transform(*builder.body_q[parent])
        builder.body_q[child] = t_parent * x_p * wp.transform_inverse(x_c)


def _single_closed_loop_articulation(builder: newton.ModelBuilder, label: str) -> None:
    """Merge every joint (including the loop-closing balls) into one articulation.

    The VBD sparse articulation solver assembles the whole joint graph of an
    articulation as a block-sparse system, so loop-closing joints must live in
    the same articulation as the spanning tree.
    """
    joint_count = len(builder.joint_articulation)
    for joint_index in range(joint_count):
        builder.joint_articulation[joint_index] = 0
    builder.articulation_start = [0]
    builder.articulation_end = [joint_count]
    builder.articulation_label = [label]
    builder.articulation_world = [builder.current_world]


def _local_point(body_xform: wp.transform, world_p: np.ndarray) -> wp.vec3:
    inv = wp.transform_inverse(body_xform)
    return wp.transform_point(inv, wp.vec3(float(world_p[0]), float(world_p[1]), float(world_p[2])))


def _box_inertia(mass: float, hx: float, hy: float, hz: float) -> wp.mat33:
    x, y, z = 2.0 * hx, 2.0 * hy, 2.0 * hz
    return wp.mat33(
        (mass * (y * y + z * z) / 12.0, 0.0, 0.0),
        (0.0, mass * (x * x + z * z) / 12.0, 0.0),
        (0.0, 0.0, mass * (x * x + y * y) / 12.0),
    )


def _quat_between(a: np.ndarray, b: np.ndarray) -> wp.quat:
    a = a.astype(np.float64) / np.linalg.norm(a)
    b = b.astype(np.float64) / np.linalg.norm(b)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot < -0.999999:
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1.0e-8:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        axis /= np.linalg.norm(axis)
        return wp.quat_from_axis_angle(wp.vec3(*axis.astype(np.float32)), math.pi)
    cross = np.cross(a, b)
    scale = math.sqrt((1.0 + dot) * 2.0)
    inv = 1.0 / scale
    return wp.quat(float(cross[0] * inv), float(cross[1] * inv), float(cross[2] * inv), float(0.5 * scale))


def _add_rod_body(builder: newton.ModelBuilder, center: np.ndarray, rod_q: wp.quat, half_len: float, label: str) -> int:
    mass = max(_MIN_BODY_MASS, _ROD_LINEAR_DENSITY * 2.0 * half_len)
    rod = builder.add_link(
        xform=wp.transform(wp.vec3(*center.astype(np.float32)), rod_q),
        com=wp.vec3(0.0, 0.0, 0.0),
        inertia=_box_inertia(mass, _ROD_RADIUS, _ROD_RADIUS, half_len),
        mass=mass,
        label=label,
        lock_inertia=True,
    )
    cfg = newton.ModelBuilder.ShapeConfig()
    cfg.density = 0.0
    cfg.has_shape_collision = False
    cfg.has_particle_collision = False
    builder.add_shape_capsule(
        rod, radius=_ROD_RADIUS, half_height=half_len, cfg=cfg, color=_ROD_COLOR, label=f"{label}_bar"
    )
    return rod


def _end_frames(
    builder: newton.ModelBuilder, b_link: int, rod: int, p_anchor: np.ndarray, world_rot: wp.quat
) -> tuple[wp.transform, wp.transform]:
    """Parent/child joint frames at `p_anchor` whose world rotation is `world_rot`."""
    t_link = wp.transform(*builder.body_q[b_link])
    t_rod = wp.transform(*builder.body_q[rod])
    q_link = wp.transform_get_rotation(t_link)
    q_rod = wp.transform_get_rotation(t_rod)
    px = wp.transform(_local_point(t_link, p_anchor), wp.quat_inverse(q_link) * world_rot)
    cx = wp.transform(_local_point(t_rod, p_anchor), wp.quat_inverse(q_rod) * world_rot)
    return px, cx


def _add_pushrod(
    builder: newton.ModelBuilder,
    b_a: int,
    p_a: np.ndarray,
    b_b: int,
    p_b: np.ndarray,
    label: str,
    rod_type: str = "ss",
) -> str:
    """Reconstruct the missing rod between two anchor points; returns the b-end joint label.

    rod_type selects the end-joint implementation (see h2_closed_loops.md):
      ss  — ball-ball: pure distance constraint; leaves a free spin DOF about the rod axis.
      su  — ball + universal (D6 with 2 angular axes ⊥ rod): same net constraint, no spin DOF.
      rr  — revolute pins about the link-local Y axis: exact for planar sagittal loops
            (H2 knee four-bar); overconstrained for spatial loops — do not use on the ankle.
    """
    rod_vec = p_b - p_a
    rod_len = float(np.linalg.norm(rod_vec))
    rod_q = _quat_between(np.array([0.0, 0.0, 1.0], dtype=np.float32), rod_vec)

    rod = _add_rod_body(builder, 0.5 * (p_a + p_b), rod_q, 0.5 * rod_len, label)

    if rod_type == "ss":
        for b_link, p_anchor, end in ((b_a, p_a, "a"), (b_b, p_b, "b")):
            px, cx = _end_frames(builder, b_link, rod, p_anchor, wp.quat_identity())
            builder.add_joint_ball(
                parent=b_link, child=rod, parent_xform=px, child_xform=cx, label=f"{label}_ball_{end}"
            )
        return f"{label}_ball_b"

    if rod_type == "rr":
        # Pin both ends about the link-local Y axis (sagittal plane normal).
        for b_link, p_anchor, end in ((b_a, p_a, "a"), (b_b, p_b, "b")):
            q_link = wp.transform_get_rotation(wp.transform(*builder.body_q[b_link]))
            px, cx = _end_frames(builder, b_link, rod, p_anchor, q_link)
            builder.add_joint_revolute(
                parent=b_link,
                child=rod,
                parent_xform=px,
                child_xform=cx,
                axis=(0.0, 1.0, 0.0),
                label=f"{label}_pin_{end}",
            )
        return f"{label}_pin_b"

    if rod_type == "su":
        px, cx = _end_frames(builder, b_a, rod, p_a, wp.quat_identity())
        builder.add_joint_ball(parent=b_a, child=rod, parent_xform=px, child_xform=cx, label=f"{label}_ball_a")
        # Universal end: joint frame z along the rod; two angular DOFs ⊥ rod,
        # twist about the rod axis is locked.
        px, cx = _end_frames(builder, b_b, rod, p_b, rod_q)
        dof = newton.ModelBuilder.JointDofConfig
        builder.add_joint_d6(
            parent=b_b,
            child=rod,
            parent_xform=px,
            child_xform=cx,
            angular_axes=[dof(axis=(1.0, 0.0, 0.0)), dof(axis=(0.0, 1.0, 0.0))],
            label=f"{label}_uni_b",
        )
        return f"{label}_uni_b"

    raise ValueError(f"Unknown rod_type {rod_type!r}")


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = 0.002
        self.sim_substeps = max(1, round(self.frame_dt / self.sim_dt))
        self.sim_time = 0.0
        self.viewer = viewer

        drive_group = getattr(args, "drive", "all")
        knee_amp = math.radians(getattr(args, "knee_amplitude_deg", 25.0))
        self.driven_joints = {}
        if drive_group == "none":
            # Hold every motor at q=0 instead of leaving the cranks limp: the
            # drop test wants a stiff statue, not a marionette.
            for name in _DRIVEN_JOINTS:
                self.driven_joints[name] = 0.0
        elif drive_group != "waist":
            for name, amp in _DRIVEN_JOINTS.items():
                if drive_group == "knee" and "knee_motor" not in name:
                    continue
                if drive_group == "ankle" and "ankle_A" not in name:
                    continue
                self.driven_joints[name] = knee_amp if "knee_motor" in name else amp
        self.drive_waist = drive_group in ("all", "waist")
        if drive_group == "none":
            self.drive_waist = False
        self.waist_antiphase = getattr(args, "waist_antiphase", False)
        self.rocker_amplitude_deg = getattr(args, "rocker_amplitude_deg", 6.0)
        # Direct-drive the waist pitch joint (the URDF marks it effort=180,
        # i.e. an actuated serial interface); the rod linkage then complies.
        self.waist_direct = getattr(args, "waist_direct", False)
        if self.waist_direct:
            self.driven_joints["waist_pitch_joint"] = math.radians(10.0)

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=getattr(args, "gravity", -9.81))
        self.source = getattr(args, "source", "mjcf")
        # Free base + ground turns the fixed-base linkage rig into a drop test,
        # which is what you want when checking the foot colliders against the
        # floor; the default rig hangs the pelvis with nothing to stand on.
        self.floating = getattr(args, "floating", False)
        # The MJCF already places the pelvis at 1.03 m and carries its own
        # <geom name="floor"> at z=0, so it is imported unshifted; translating it
        # would lift that floor along with the robot. The URDF has no ground and
        # spawns at the origin, hence the 1.2 m default there.
        spawn_z = getattr(args, "spawn_height", None)
        if spawn_z is None:
            spawn_z = 0.0 if self.source == "mjcf" else 1.2
        # The broad phase pairs shapes per world; world -1 is the shared segment
        # and is only tested against regular worlds, never against itself. With
        # everything left at -1 (this builder's default) no candidate pairs
        # exist at all, so the drop test's robot and floor must live in a
        # regular world.
        if self.floating:
            builder.begin_world(label="h2")
        if self.source == "mjcf":
            # The MJCF is the collision authority: sole capsules plus a per-link
            # collision whitelist. Its <equality><tendon> loops are skipped —
            # the pushrods below reconstruct them as rigid bodies instead.
            collision_view = getattr(args, "collision_view", False)
            builder.add_mjcf(
                args.mjcf,
                xform=wp.transform(wp.vec3(0.0, 0.0, spawn_z), wp.quat_identity()),
                floating=self.floating,
                enable_self_collisions=False,
                parse_sites=True,
                skip_equality_constraints=True,
                parse_visuals=not collision_view,
                force_show_colliders=collision_view,
            )
        else:
            builder.add_urdf(
                args.urdf,
                xform=wp.transform(wp.vec3(0.0, 0.0, spawn_z), wp.quat_identity()),
                floating=self.floating,
                enable_self_collisions=False,
            )

        _rest_pose_fk(builder)
        self._bump_placeholder_masses(builder)

        # With a floating base the importer drops the asset's base pose: the free
        # joint's parent anchor stays at identity (the fixed-base path carries the
        # pose in X_p instead), so the rest FK assembles the robot around the
        # origin. Recover the pose from the MJCF's root <body pos> (plus the
        # spawn offset) and shift every rest transform before the pushrods are
        # anchored — the closed-loop graph forbids eval_fk, so states inherit
        # body_q as-is and everything must be consistent here.
        if self.floating:
            base_p = [0.0, 0.0, spawn_z]
            if self.source == "mjcf":
                root_body = ET.parse(args.mjcf).getroot().find("./worldbody/body")
                pos = [float(v) for v in root_body.get("pos", "0 0 0").split()]
                base_p = [pos[0], pos[1], pos[2] + spawn_z]
            base_tf = wp.transform(wp.vec3(*base_p), wp.quat_identity())
            for bi in range(len(builder.body_q)):
                builder.body_q[bi] = list(base_tf * wp.transform(*builder.body_q[bi]))
            for j in range(len(builder.joint_type)):
                if builder.joint_type[j] == newton.JointType.FREE:
                    child = builder.joint_child[j]
                    qs = builder.joint_q_start[j]
                    builder.joint_q[qs : qs + 7] = list(builder.body_q[child])

        # enable_self_collisions=False filters every shape pair inside the
        # import — and the MJCF's floor is part of the import, so the foot-floor
        # pairs land in the exclusion list too (18 robot shapes + 1 floor =
        # C(19,2) = 171 filtered pairs, i.e. all of them). Strip the pairs that
        # involve static geometry; self-collision stays off.
        if self.floating:
            static_shapes = {si for si, b in enumerate(builder.shape_body) if b < 0}
            before = len(builder.shape_collision_filter_pairs)
            builder.shape_collision_filter_pairs = type(builder.shape_collision_filter_pairs)(
                pair
                for pair in builder.shape_collision_filter_pairs
                if pair[0] not in static_shapes and pair[1] not in static_shapes
            )
            removed = before - len(builder.shape_collision_filter_pairs)
            if removed:
                print(f"[sim_h2_loop] unfiltered {removed} static-vs-robot collision pairs")

        extent = getattr(args, "ground_extent", 20.0)
        if _size_infinite_planes(builder, extent):
            print(f"[sim_h2_loop] added a {extent:g}x{extent:g} m visual quad over the MJCF floor (collision plane untouched)")
        elif getattr(args, "ground", False):
            builder.add_ground_plane()

        mjcf_path = getattr(args, "armature_mjcf", DEFAULT_MJCF)
        if mjcf_path:
            _apply_mjcf_joint_params(builder, mjcf_path)

        rod_types = {
            "ankle": getattr(args, "ankle_rod", "ss"),
            "knee": getattr(args, "knee_rod", "ss"),
            "waist": "ss",  # rigid passive rods; actuation goes through the rockers
        }
        body_labels = builder.body_label
        self.loop_ball_labels = []
        loops = _LOOPS_MJCF if self.source == "mjcf" else _LOOPS
        for link_a, marker_a, link_b, marker_b, rod_label in loops:
            group = "ankle" if "ankle" in rod_label else ("knee" if "knee" in rod_label else "waist")
            b_a = _find(body_labels, link_a)
            b_b = _find(body_labels, link_b)
            if self.source == "mjcf":
                p_a = _site_world_pos(builder, marker_a)
                p_b = _site_world_pos(builder, marker_b)
            else:
                p_a = np.array(builder.body_q[_find(body_labels, marker_a)])[:3]
                p_b = np.array(builder.body_q[_find(body_labels, marker_b)])[:3]
            end_label = _add_pushrod(builder, b_a, p_a, b_b, p_b, rod_label, rod_type=rod_types[group])
            self.loop_ball_labels.append(end_label)

        if self.floating:
            builder.end_world()

        _single_closed_loop_articulation(builder, "h2_loops")
        builder.color()

        # NB: do not call eval_fk on a closed-loop graph. Forward kinematics
        # assumes a spanning tree, but the rod bodies have two parents (the
        # loop closure). _rest_pose_fk already assembled the rest pose, so the
        # finalized model (and states derived from it) start consistent.
        self.model = builder.finalize(skip_validation_joints=True)
        # The linkage rig never needed contacts, but the drop test does:
        # without a pipeline the solver receives contacts=None and the
        # floor is decorative.
        self.collision_pipeline = None
        self.contacts = None
        if self.floating:
            # "explicit" broad phase (the default) only tests the pairs staged
            # in model.shape_contact_pairs, and this standalone build stages
            # none; nxn is cheap for a single robot plus a floor.
            self.model.request_contact_attributes("force")
            self.collision_pipeline = newton.CollisionPipeline(self.model, broad_phase="nxn")
            self.contacts = self.collision_pipeline.contacts()
            # Scratch buffer holding only the force-bearing subset for display:
            # the narrow phase reports every candidate within gap_sum (0.2 m by
            # default), so raw arrows also mark contacts carrying zero force.
            self._display_contacts = self.collision_pipeline.contacts()

        self._configure_drives()

        # Gains follow the branch's validated VBD sparse recipe from
        # reports/vbd_complex_linkages/bench_complex_linkages.py.
        solve_mode = getattr(args, "solve", "block_sparse_joints")
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=getattr(args, "iterations", 8),
            # A fallen torso mesh alone reaches ~200 plane contacts on one body;
            # the 64-entry default silently drops the excess.
            rigid_body_contact_buffer_size=getattr(args, "contact_buffer", 256),
            rigid_articulation_solve=solve_mode,
            rigid_joint_armature=solve_mode == "block_sparse_joints",
            rigid_articulation_level_parallel=getattr(args, "level_parallel", False),
            rigid_articulation_relaxation=0.65,
            rigid_articulation_diagonal_regularization=0.0,
            rigid_avbd_alpha=0.0,
            rigid_avbd_beta=0.0,
            rigid_joint_linear_ke=5.0e4,
            rigid_joint_angular_ke=5.0e4,
            rigid_joint_linear_kd=1.25e2,
            rigid_joint_angular_kd=1.25e2,
        )

        self.state_0 = self.model.state()
        self.state_1 = self.model.state()

        self.frame_index = 0
        self._pelvis_body = next(
            i for i, lb in enumerate(self.model.body_label) if lb.rsplit("/", 1)[-1] == "pelvis"
        )
        self.control = self.model.control()
        self.target_q = self.model.joint_target_q.numpy().copy()

        self.viewer.set_model(self.model)
        if hasattr(self.viewer, "set_camera"):
            self.viewer.set_camera(wp.vec3(2.2, -2.2, 1.4), -35.0, 25.0)

    @staticmethod
    def _bump_placeholder_masses(builder: newton.ModelBuilder) -> None:
        # Includes zero-mass sensor bodies (camera/IMU): the VBD solver treats
        # inv_mass == 0 as a static anchor, and their fixed joints would weld
        # the torso/head chain to the world, freezing the whole spine.
        for b in range(len(builder.body_mass)):
            if builder.body_mass[b] >= _MIN_BODY_MASS:
                continue
            builder.body_mass[b] = _MIN_BODY_MASS
            builder.body_inv_mass[b] = 1.0 / _MIN_BODY_MASS
            inertia = np.array(builder.body_inertia[b]).reshape(3, 3)
            inertia[np.diag_indices(3)] = np.maximum(np.diag(inertia), _MIN_BODY_INERTIA)
            builder.body_inertia[b] = wp.mat33(inertia)
            builder.body_inv_inertia[b] = wp.mat33(np.linalg.inv(inertia))

    def _configure_drives(self) -> None:
        joint_labels = [label.rsplit("/", 1)[-1] for label in self.model.joint_label]
        joint_type = self.model.joint_type.numpy()
        qd_start = self.model.joint_qd_start.numpy()

        mode = self.model.joint_target_mode.numpy()
        target_ke = self.model.joint_target_ke.numpy()
        target_kd = self.model.joint_target_kd.numpy()

        self.drive_dofs = {}
        for name, joint_index in ((n, joint_labels.index(n)) for n in joint_labels if n in self.driven_joints):
            dof = int(qd_start[joint_index])
            self.drive_dofs[name] = dof
            mode[dof] = int(newton.JointTargetMode.POSITION_VELOCITY)
            target_ke[dof] = 500.0
            target_kd[dof] = 25.0

        for joint_index, name in enumerate(joint_labels):
            if joint_type[joint_index] != int(newton.JointType.REVOLUTE) or name in self.drive_dofs:
                continue
            dof = int(qd_start[joint_index])
            if name in _PASSIVE_JOINTS and not (self.waist_direct and name == "waist_pitch_joint"):
                mode[dof] = int(newton.JointTargetMode.NONE)
            else:
                # Hold arms/head/hips/waist-yaw/ankle-roll at the rest pose.
                mode[dof] = int(newton.JointTargetMode.POSITION_VELOCITY)
                target_ke[dof] = 200.0
                target_kd[dof] = 20.0

        # Waist actuation: drive the rockers (torso_constraint_L/R) as motor
        # crank arms through the rigid rods (unless --waist-direct is used).
        if self.drive_waist and not self.waist_direct:
            rocker_amp = math.radians(self.rocker_amplitude_deg)
            for name in ("torso_constraint_L_joint", "torso_constraint_R_joint"):
                dof = int(qd_start[joint_labels.index(name)])
                self.drive_dofs[name] = dof
                self.driven_joints[name] = rocker_amp
                mode[dof] = int(newton.JointTargetMode.POSITION_VELOCITY)
                target_ke[dof] = 500.0
                target_kd[dof] = 25.0

        self.model.joint_target_mode.assign(mode)
        self.model.joint_target_ke.assign(target_ke)
        self.model.joint_target_kd.assign(target_kd)

    def simulate(self):
        for substep in range(self.sim_substeps):
            t = self.sim_time + substep * self.sim_dt
            phase = 2.0 * math.pi * _DRIVE_FREQUENCY * t
            for name, dof in self.drive_dofs.items():
                amp = self.driven_joints[name]
                if "knee_motor" in name:
                    # Raised cosine keeps knee flexion inside its [-0.2, 2.9] range.
                    self.target_q[dof] = amp * 0.5 * (1.0 - math.cos(phase))
                else:
                    # Ankle A-cranks or waist rocker cranks (anti-phase rockers -> pitch).
                    right_side = name.startswith("torso_constraint_R")
                    sign = -1.0 if self.waist_antiphase and right_side else 1.0
                    self.target_q[dof] = sign * amp * math.sin(phase)
            self.control.joint_target_q.assign(self.target_q)

            self.state_0.clear_forces()
            self.viewer.apply_forces(self.state_0)
            if self.collision_pipeline is not None:
                self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt
        if self.floating:
            # one line per rendered frame so viewer observations can be matched
            # to a frame number
            z = float(self.state_0.body_q.numpy()[self._pelvis_body][2])
            n = int(self.contacts.rigid_contact_count.numpy()[0]) if self.contacts is not None else 0
            loaded = getattr(self, "loaded_contact_count", -1)
            print(
                f"[frame {self.frame_index:5d}] t={self.sim_time:7.3f}s  pelvis_z={z:+.4f} m"
                f"  candidates={n}  loaded={loaded}"
            )
        self.frame_index += 1

    def _loaded_contacts(self):
        """Compact the contact buffer down to entries with nonzero force."""
        import numpy as np

        c = self.contacts
        self.solver.update_contacts(c, self.state_0)
        n = min(int(c.rigid_contact_count.numpy()[0]), c.rigid_contact_max)
        f = c.force.numpy()[:n]
        keep = np.flatnonzero(np.linalg.norm(f[:, :3], axis=1) > 1e-6)
        d = self._display_contacts
        d.rigid_contact_count.assign(np.array([len(keep)], dtype=np.int32))
        if len(keep):
            d.rigid_contact_shape0.numpy()[: len(keep)]  # ensure allocation
            d.rigid_contact_shape0.assign(
                np.resize(c.rigid_contact_shape0.numpy()[:n][keep], d.rigid_contact_max)
            )
            d.rigid_contact_shape1.assign(
                np.resize(c.rigid_contact_shape1.numpy()[:n][keep], d.rigid_contact_max)
            )
            d.rigid_contact_point0.assign(np.resize(c.rigid_contact_point0.numpy()[:n][keep], (d.rigid_contact_max, 3)))
            d.rigid_contact_offset0.assign(
                np.resize(c.rigid_contact_offset0.numpy()[:n][keep], (d.rigid_contact_max, 3))
            )
            d.rigid_contact_normal.assign(np.resize(c.rigid_contact_normal.numpy()[:n][keep], (d.rigid_contact_max, 3)))
        return d, len(keep)

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        if self.contacts is not None:
            display, self.loaded_contact_count = self._loaded_contacts()
            self.viewer.log_contacts(display, self.state_0)
        self.viewer.end_frame()

    def test_final(self):
        body_q = self.state_0.body_q.numpy()
        if not np.all(np.isfinite(body_q)):
            raise ValueError("Body poses became non-finite")
        gaps = self._loop_gaps()
        if max(gaps) > 2.0e-3:
            raise ValueError(f"Loop closure drifted: gaps {gaps} m")

    def _joint_angle_deg(self, joint_name: str) -> float:
        joint_labels = [label.rsplit("/", 1)[-1] for label in self.model.joint_label]
        j = joint_labels.index(joint_name)
        parent = int(self.model.joint_parent.numpy()[j])
        child = int(self.model.joint_child.numpy()[j])
        body_q = self.state_0.body_q.numpy()
        rest_q = self.model.body_q.numpy()

        def rel(idx: int, q_arr: np.ndarray) -> wp.quat:
            return wp.quat(*q_arr[idx, 3:7])

        q_parent = rel(parent, body_q) if parent >= 0 else wp.quat_identity()
        q_child = rel(child, body_q)
        r_parent = rel(parent, rest_q) if parent >= 0 else wp.quat_identity()
        r_child = rel(child, rest_q)
        now = wp.quat_inverse(q_parent) * q_child
        rest = wp.quat_inverse(r_parent) * r_child
        delta = wp.quat_inverse(rest) * now
        w = float(np.clip(abs(delta[3]), 0.0, 1.0))
        return math.degrees(2.0 * math.acos(w))

    def _loop_gaps(self) -> list[float]:
        joint_labels = [label.rsplit("/", 1)[-1] for label in self.model.joint_label]
        body_q = self.state_0.body_q.numpy()
        gaps = []
        for ball_label in self.loop_ball_labels:
            j = joint_labels.index(ball_label)
            parent = int(self.model.joint_parent.numpy()[j])
            child = int(self.model.joint_child.numpy()[j])
            xp = self.model.joint_X_p.numpy()[j]
            xc = self.model.joint_X_c.numpy()[j]
            wp0 = wp.transform_point(wp.transform(*body_q[parent]), wp.vec3(*xp[:3]))
            wc0 = wp.transform_point(wp.transform(*body_q[child]), wp.vec3(*xc[:3]))
            gaps.append(float(np.linalg.norm(np.array(wp0) - np.array(wc0))))
        return gaps

    def diagnose(self, frames: int) -> None:
        init_finite = bool(np.all(np.isfinite(self.state_0.body_q.numpy())))
        gaps = self._loop_gaps()
        print(f"initial body_q finite={init_finite}  initial max_gap_mm={max(gaps) * 1e3:.4f}")
        header = (
            f"{'frame':>5} {'knee_tgt':>9} {'l_knee':>7} {'r_knee':>7} {'l_pitch':>8} {'r_pitch':>8}"
            f" {'w_pitch':>8} {'max_gap_mm':>11} {'finite':>7}"
        )
        print(header)
        for frame in range(frames):
            self.step()
            knee_dof = self.drive_dofs.get("left_knee_motor_joint")
            knee_tgt = math.degrees(float(self.target_q[knee_dof])) if knee_dof is not None else 0.0
            l_knee = self._joint_angle_deg("left_knee_joint")
            r_knee = self._joint_angle_deg("right_knee_joint")
            l_pitch = self._joint_angle_deg("left_ankle_pitch_joint")
            r_pitch = self._joint_angle_deg("right_ankle_pitch_joint")
            w_pitch = self._joint_angle_deg("waist_pitch_joint")
            gaps = self._loop_gaps()
            finite = bool(np.all(np.isfinite(self.state_0.body_q.numpy())))
            print(
                f"{frame:>5} {knee_tgt:>9.2f} {l_knee:>7.2f} {r_knee:>7.2f} {l_pitch:>8.2f} {r_pitch:>8.2f}"
                f" {w_pitch:>8.2f} {max(gaps) * 1e3:>11.3f} {finite!s:>7}"
            )
            if not finite:
                break
        print("final per-loop gaps [mm]:", [round(g * 1e3, 3) for g in self._loop_gaps()])

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument(
            "--source",
            choices=["mjcf", "urdf"],
            default="mjcf",
            help="Asset to load. 'mjcf' brings the MJCF collision set (sole capsules + link whitelist).",
        )
        parser.add_argument("--mjcf", type=str, default=DEFAULT_MJCF, help="Path to H2loop_complete.xml.")
        parser.add_argument(
            "--floating",
            action="store_true",
            help="Give the pelvis a free joint instead of pinning it in the air.",
        )
        parser.add_argument(
            "--ground",
            action="store_true",
            help="Add a collidable ground plane at z=0 (only needed for --source urdf; the MJCF ships one).",
        )
        parser.add_argument(
            "--ground-extent", type=float, default=20.0, help="Drawn half-extent of the ground plane [m]."
        )
        parser.add_argument(
            "--collision-view",
            action="store_true",
            help="Import only the collision geometry (no visual meshes) and force it visible.",
        )
        parser.add_argument(
            "--contact-buffer",
            type=int,
            default=256,
            help="Per-body rigid contact list capacity (solver default is 64).",
        )
        parser.add_argument(
            "--spawn-height",
            type=float,
            default=None,
            help="Extra height applied to the imported asset [m]. Defaults to 0 for mjcf (it already "
            "positions the pelvis and its floor) and 1.2 for urdf.",
        )
        parser.add_argument("--urdf", type=str, default=DEFAULT_URDF, help="Path to H2_loop.urdf.")
        parser.add_argument(
            "--armature_mjcf",
            type=str,
            default=DEFAULT_MJCF,
            help="MuJoCo XML providing per-joint armature values; pass '' to disable.",
        )
        parser.add_argument("--diagnose", action="store_true", help="Print linkage/loop diagnostics and exit.")
        parser.add_argument("--solve", choices=["local", "block_sparse_joints"], default="block_sparse_joints")
        parser.add_argument("--gravity", type=float, default=-9.81)
        parser.add_argument("--iterations", type=int, default=8)
        parser.add_argument(
            "--level-parallel", action="store_true", help="Use the level-scheduled block-sparse solve kernel."
        )
        parser.add_argument(
            "--drive", choices=["all", "knee", "ankle", "waist", "none"], default="all", help="Which cranks to drive."
        )
        parser.add_argument("--knee-amplitude-deg", type=float, default=25.0)
        parser.add_argument("--ankle-rod", choices=["ss", "su"], default="ss", help="Ankle rod ends (spatial loop).")
        parser.add_argument("--knee-rod", choices=["ss", "rr", "su"], default="ss", help="Knee rod ends (planar loop).")
        parser.add_argument(
            "--waist-antiphase", action="store_true", help="Anti-phase rocker drive (pitch) instead of in-phase (roll)."
        )
        parser.add_argument(
            "--waist-direct", action="store_true", help="Direct-drive waist_pitch_joint; the rod linkage complies."
        )
        parser.add_argument(
            "--rocker-amplitude-deg", type=float, default=6.0, help="Waist rocker-crank drive amplitude."
        )
        parser.set_defaults(num_frames=240)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    if getattr(args, "diagnose", False):
        example.diagnose(args.num_frames)
    else:
        newton.examples.run(example, args)
