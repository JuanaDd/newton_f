# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Build the H2 MJCF tree with six massive ball-ended closure rods."""

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import warp as wp

import newton

DEFAULT_XML = Path(__file__).resolve().parents[1] / "h2_description/H2loop_complete.xml"
_MIN_BODY_MASS = 0.02
_MIN_BODY_INERTIA = 2e-6
_ROD_RADIUS = 0.008
_ROD_LINEAR_DENSITY = 0.3
_ROD_COLOR = wp.vec3(0.80, 0.68, 0.28)

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
    (
        "left_knee_link",
        "left_knee_output_rod_site",
        "left_knee_motor_link",
        "left_knee_motor_rod_site",
        "left_knee_rod",
    ),
    (
        "right_knee_link",
        "right_knee_output_rod_site",
        "right_knee_motor_link",
        "right_knee_motor_rod_site",
        "right_knee_rod",
    ),
    (
        "waist_yaw_link",
        "left_waist_output_rod_site",
        "torso_constraint_L_link",
        "left_waist_motor_rod_site",
        "waist_rod_L",
    ),
    (
        "waist_yaw_link",
        "right_waist_output_rod_site",
        "torso_constraint_R_link",
        "right_waist_motor_rod_site",
        "waist_rod_R",
    ),
)


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


def _rest_pose_fk(builder: newton.ModelBuilder) -> None:
    """Assemble the imported tree before adding its loop-closing joints."""
    for j in range(len(builder.joint_type)):
        parent = builder.joint_parent[j]
        child = builder.joint_child[j]
        x_p = wp.transform(*builder.joint_X_p[j])
        x_c = wp.transform(*builder.joint_X_c[j])
        t_parent = wp.transform_identity() if parent < 0 else wp.transform(*builder.body_q[parent])
        builder.body_q[child] = t_parent * x_p * wp.transform_inverse(x_c)


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


def build_builder(xml: str | Path = DEFAULT_XML) -> newton.ModelBuilder:
    """Import the collision geometry and reconstruct the six MJCF tendon loops."""
    builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
    builder.begin_world(label="h2")
    builder.add_mjcf(
        str(xml),
        floating=True,
        enable_self_collisions=False,
        parse_sites=True,
        skip_equality_constraints=True,
        parse_visuals=True,
    )
    _rest_pose_fk(builder)
    for b in range(len(builder.body_mass)):
        if builder.body_mass[b] < _MIN_BODY_MASS:
            builder.body_mass[b] = _MIN_BODY_MASS
            builder.body_inv_mass[b] = 1.0 / _MIN_BODY_MASS
            inertia = np.array(builder.body_inertia[b]).reshape(3, 3)
            inertia[np.diag_indices(3)] = np.maximum(np.diag(inertia), _MIN_BODY_INERTIA)
            builder.body_inertia[b] = wp.mat33(inertia)
            builder.body_inv_inertia[b] = wp.mat33(np.linalg.inv(inertia))
    root = ET.parse(xml).getroot()
    pos = np.fromstring(root.find("./worldbody/body").get("pos", "0 0 0"), sep=" ")
    base = wp.transform(wp.vec3(*pos), wp.quat_identity())
    for b in range(len(builder.body_q)):
        builder.body_q[b] = list(base * wp.transform(*builder.body_q[b]))
    defaults = root.find("./default/joint").attrib
    authored = {j.get("name"): j.attrib for j in root.iter("joint") if j.get("name")}
    for j, label in enumerate(builder.joint_label):
        if builder.joint_type[j] == newton.JointType.FREE:
            q = builder.joint_q_start[j]
            builder.joint_q[q : q + 7] = list(builder.body_q[builder.joint_child[j]])
        if builder.joint_type[j] != newton.JointType.REVOLUTE:
            continue
        params = {**defaults, **authored[label.rsplit("/", 1)[-1]]}
        dof = builder.joint_qd_start[j]
        builder.joint_armature[dof] = float(params["armature"])
        builder.joint_damping[dof] = float(params["damping"])
        builder.joint_target_kd[dof] = float(params["damping"])
        builder.joint_friction[dof] = float(params["frictionloss"])
    # The importer also filtered floor/robot pairs; retain only self filters.
    static = {i for i, body in enumerate(builder.shape_body) if body < 0}
    builder.shape_collision_filter_pairs = type(builder.shape_collision_filter_pairs)(
        pair for pair in builder.shape_collision_filter_pairs if not static.intersection(pair)
    )
    for link_a, site_a, link_b, site_b, label in _LOOPS_MJCF:
        a, b = _find(builder.body_label, link_a), _find(builder.body_label, link_b)
        pa, pb = _site_world_pos(builder, site_a), _site_world_pos(builder, site_b)
        direction = pb - pa
        quat = _quat_between(np.array([0.0, 0.0, 1.0]), direction)
        rod = _add_rod_body(builder, (pa + pb) * 0.5, quat, np.linalg.norm(direction) * 0.5, label)
        for body, point, end in ((a, pa, "a"), (b, pb, "b")):
            parent, child = _end_frames(builder, body, rod, point, wp.quat_identity())
            builder.add_joint_ball(
                parent=body, child=rod, parent_xform=parent, child_xform=child, label=f"{label}_ball_{end}"
            )
    count = len(builder.joint_type)
    builder.joint_articulation = [0] * count
    builder.articulation_start = [0]
    builder.articulation_end = [count]
    builder.articulation_label = ["h2_closed_loop"]
    builder.articulation_world = [builder.current_world]
    builder.end_world()
    builder.color()
    return builder
