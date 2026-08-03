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

import numpy as np
import warp as wp

import newton
import newton.examples

DEFAULT_URDF = "/home/yvetted/Downloads/unitree_ros_h2/robots/h2_description/H2_loop.urdf"

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


def _find(labels: list[str], suffix: str) -> int:
    for i, label in enumerate(labels):
        if label.rsplit("/", 1)[-1] == suffix:
            return i
    raise KeyError(f"Missing '{suffix}' in {[l.rsplit('/', 1)[-1] for l in labels]}")


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
        if drive_group != "waist":
            for name, amp in _DRIVEN_JOINTS.items():
                if drive_group == "knee" and "knee_motor" not in name:
                    continue
                if drive_group == "ankle" and "ankle_A" not in name:
                    continue
                self.driven_joints[name] = knee_amp if "knee_motor" in name else amp
        self.drive_waist = drive_group in ("all", "waist")
        self.waist_antiphase = getattr(args, "waist_antiphase", False)
        self.rocker_amplitude_deg = getattr(args, "rocker_amplitude_deg", 6.0)
        # Direct-drive the waist pitch joint (the URDF marks it effort=180,
        # i.e. an actuated serial interface); the rod linkage then complies.
        self.waist_direct = getattr(args, "waist_direct", False)
        if self.waist_direct:
            self.driven_joints["waist_pitch_joint"] = math.radians(10.0)

        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=getattr(args, "gravity", -9.81))
        builder.add_urdf(
            args.urdf,
            xform=wp.transform(wp.vec3(0.0, 0.0, 1.2), wp.quat_identity()),
            floating=False,
            enable_self_collisions=False,
        )

        _rest_pose_fk(builder)
        self._bump_placeholder_masses(builder)

        rod_types = {
            "ankle": getattr(args, "ankle_rod", "ss"),
            "knee": getattr(args, "knee_rod", "ss"),
            "waist": "ss",  # rigid passive rods; actuation goes through the rockers
        }
        body_labels = builder.body_label
        self.loop_ball_labels = []
        for link_a, marker_a, link_b, marker_b, rod_label in _LOOPS:
            group = "ankle" if "ankle" in rod_label else ("knee" if "knee" in rod_label else "waist")
            b_a = _find(body_labels, link_a)
            b_b = _find(body_labels, link_b)
            p_a = np.array(builder.body_q[_find(body_labels, marker_a)])[:3]
            p_b = np.array(builder.body_q[_find(body_labels, marker_b)])[:3]
            end_label = _add_pushrod(builder, b_a, p_a, b_b, p_b, rod_label, rod_type=rod_types[group])
            self.loop_ball_labels.append(end_label)

        _single_closed_loop_articulation(builder, "h2_loops")
        builder.color()

        # NB: do not call eval_fk on a closed-loop graph. Forward kinematics
        # assumes a spanning tree, but the rod bodies have two parents (the
        # loop closure). _rest_pose_fk already assembled the rest pose, so the
        # finalized model (and states derived from it) start consistent.
        self.model = builder.finalize(skip_validation_joints=True)

        self._configure_drives()

        # Gains follow the branch's validated VBD sparse recipe from
        # reports/vbd_complex_linkages/bench_complex_linkages.py.
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=getattr(args, "iterations", 8),
            rigid_articulation_solve=getattr(args, "solve", "block_sparse_joints"),
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
            self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0

    def step(self):
        self.simulate()
        self.sim_time += self.frame_dt

    def render(self):
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
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
        parser.add_argument("--urdf", type=str, default=DEFAULT_URDF, help="Path to H2_loop.urdf.")
        parser.add_argument("--diagnose", action="store_true", help="Print linkage/loop diagnostics and exit.")
        parser.add_argument("--solve", choices=["local", "block_sparse_joints"], default="block_sparse_joints")
        parser.add_argument("--gravity", type=float, default=-9.81)
        parser.add_argument("--iterations", type=int, default=8)
        parser.add_argument(
            "--drive", choices=["all", "knee", "ankle", "waist"], default="all", help="Which cranks to drive."
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
