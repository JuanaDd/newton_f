# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Simulate G1 left-ankle closed-loop linkage (VBD sparse articulation)
#
# The Unitree G1 ankle is a parallel mechanism: two motor cranks (A/B
# joints) drive the foot's passive pitch/roll joints through two pushrods.
# The URDF only encodes the spanning tree; the accompanying MJCF closes the
# two loops with `connect` equality constraints between the roll link and
# the rod tips. This example reproduces that closure with two ball joints
# (matching the branch's `left_ankle_loop_A/B_ball` benchmark joints).
#
# The URDF authors the rod joints as revolute, but physically (and in the
# vendor README) they are ball joints, so they are retyped to `ball` before
# import (this branch's URDF importer supports ball joints).
#
# Command:
#   uv run --extra examples python sim_g1_ankle.py --viewer gl
#   uv run --extra examples python sim_g1_ankle.py --viewer null --diagnose
###########################################################################

from __future__ import annotations

import math
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import warp as wp

import newton
import newton.examples

DEFAULT_URDF = "/home/yvetted/Downloads/loop_model/loop_model/ankle/g1_left_ankle_mujoco_loop_model/L_loop_ankle.urdf"

_DRIVE_AMPLITUDE = math.radians(15.0)
_DRIVE_FREQUENCY = 0.5  # Hz

# The URDF gives rods/cranks placeholder masses of 1e-6 kg, which makes the
# system numerically singular. Bump them to a plausible small-part mass.
_MIN_BODY_MASS = 0.02
_MIN_BODY_INERTIA = 2.0e-6

_BALL_ROD_JOINTS = ("left_ankle_A_rod_joint", "left_ankle_B_rod_joint")
_LOOPS = (
    ("left_ankle_constraint_A", "left_ankle_A_rod_link", "left_ankle_loop_A_ball"),
    ("left_ankle_constraint_B", "left_ankle_B_rod_link", "left_ankle_loop_B_ball"),
)


def _find(labels: list[str], suffix: str) -> int:
    for i, label in enumerate(labels):
        if label.rsplit("/", 1)[-1] == suffix:
            return i
    raise KeyError(f"Missing '{suffix}' in {[l.rsplit('/', 1)[-1] for l in labels]}")


def _retype_rod_joints_to_ball(urdf_path: str) -> str:
    """Rewrite the rod joints as ball joints and return a temp URDF path."""
    tree = ET.parse(urdf_path)
    for joint in tree.getroot().iter("joint"):
        if joint.get("name") in _BALL_ROD_JOINTS:
            joint.set("type", "ball")
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".urdf", prefix="g1_ankle_ball_", dir=str(Path(urdf_path).parent), delete=False
    )
    tree.write(tmp.name)
    return tmp.name


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


class Example:
    def __init__(self, viewer, args):
        self.fps = 60
        self.frame_dt = 1.0 / self.fps
        self.sim_dt = 0.002
        self.sim_substeps = max(1, round(self.frame_dt / self.sim_dt))
        self.sim_time = 0.0
        self.viewer = viewer
        self.drive_amplitude = math.radians(getattr(args, "drive_amplitude_deg", 15.0))

        retype = not getattr(args, "no_retype", False)
        ball_urdf = _retype_rod_joints_to_ball(args.urdf) if retype else args.urdf
        builder = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=getattr(args, "gravity", -9.81))
        try:
            builder.add_urdf(
                ball_urdf,
                xform=wp.transform(wp.vec3(0.0, 0.0, 1.0), wp.quat_identity()),
                floating=False,
                enable_self_collisions=False,
            )
        finally:
            if retype:  # only ever delete the temp copy, never the source URDF
                Path(ball_urdf).unlink(missing_ok=True)

        _rest_pose_fk(builder)
        self._bump_placeholder_masses(builder)

        body_labels = builder.body_label
        b_roll = _find(body_labels, "left_ankle_roll_link")
        loops = () if getattr(args, "no_loops", False) else _LOOPS
        for marker_name, rod_name, ball_label in loops:
            b_marker = _find(body_labels, marker_name)
            b_rod = _find(body_labels, rod_name)
            # The green/blue marker body sits exactly at the MJCF connect anchor.
            anchor = np.array(builder.body_q[b_marker])[:3]
            builder.add_joint_ball(
                parent=b_roll,
                child=b_rod,
                parent_xform=wp.transform(_local_point(builder.body_q[b_roll], anchor), wp.quat_identity()),
                child_xform=wp.transform(_local_point(builder.body_q[b_rod], anchor), wp.quat_identity()),
                label=ball_label,
            )

        _single_closed_loop_articulation(builder, "g1_ankle_loops")
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
            self.viewer.set_camera(wp.vec3(0.5, -0.7, 0.9), -15.0, 55.0)

    @staticmethod
    def _bump_placeholder_masses(builder: newton.ModelBuilder) -> None:
        for b in range(len(builder.body_mass)):
            if builder.body_mass[b] <= 0.0 or builder.body_mass[b] >= _MIN_BODY_MASS:
                continue
            builder.body_mass[b] = _MIN_BODY_MASS
            builder.body_inv_mass[b] = 1.0 / _MIN_BODY_MASS
            inertia = np.array(builder.body_inertia[b]).reshape(3, 3)
            inertia[np.diag_indices(3)] = np.maximum(np.diag(inertia), _MIN_BODY_INERTIA)
            builder.body_inertia[b] = wp.mat33(inertia)
            builder.body_inv_inertia[b] = wp.mat33(np.linalg.inv(inertia))

    def _configure_drives(self) -> None:
        joint_labels = [label.rsplit("/", 1)[-1] for label in self.model.joint_label]
        qd_start = self.model.joint_qd_start.numpy()

        lower = self.model.joint_limit_lower.numpy()
        upper = self.model.joint_limit_upper.numpy()
        mode = self.model.joint_target_mode.numpy()
        target_ke = self.model.joint_target_ke.numpy()
        target_kd = self.model.joint_target_kd.numpy()

        # Actuate the A/B cranks as position/velocity servos. The URDF locks
        # their range to [0, 0]; open it to the MJCF ctrlrange.
        self.drive_dofs = []
        for name, lo, hi in (
            ("left_ankle_A_joint", -0.8203, 1.4835),
            ("left_ankle_B_joint", -1.4835, 0.8203),
        ):
            dof = int(qd_start[joint_labels.index(name)])
            self.drive_dofs.append(dof)
            lower[dof], upper[dof] = lo, hi
            mode[dof] = int(newton.JointTargetMode.POSITION_VELOCITY)
            target_ke[dof] = 500.0
            target_kd[dof] = 25.0

        # Pitch/roll are the passive outputs of the parallel mechanism.
        for name in ("left_ankle_pitch_joint", "left_ankle_roll_joint"):
            dof = int(qd_start[joint_labels.index(name)])
            mode[dof] = int(newton.JointTargetMode.NONE)

        self.model.joint_limit_lower.assign(lower)
        self.model.joint_limit_upper.assign(upper)
        self.model.joint_target_mode.assign(mode)
        self.model.joint_target_ke.assign(target_ke)
        self.model.joint_target_kd.assign(target_kd)

    def simulate(self):
        for substep in range(self.sim_substeps):
            t = self.sim_time + substep * self.sim_dt
            # The B crank range is mirrored w.r.t. A (see the MJCF ctrlrange),
            # so B = -A sweeps ankle pitch while keeping roll near zero.
            drive = self.drive_amplitude * math.sin(2.0 * math.pi * _DRIVE_FREQUENCY * t)
            self.target_q[self.drive_dofs[0]] = drive
            self.target_q[self.drive_dofs[1]] = -drive
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
        if max(self._loop_gaps()) > 2.0e-3:
            raise ValueError(f"Loop closure drifted: gaps {self._loop_gaps()} m")

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
        for _, _, ball_label in _LOOPS:
            if ball_label not in joint_labels:
                gaps.append(0.0)
                continue
            j = joint_labels.index(ball_label)
            parent = int(self.model.joint_parent.numpy()[j])
            child = int(self.model.joint_child.numpy()[j])
            xp = self.model.joint_X_p.numpy()[j]
            xc = self.model.joint_X_c.numpy()[j]
            wp0 = wp.transform_point(wp.transform(*body_q[parent]), wp.vec3(*xp[:3]))
            wc0 = wp.transform_point(wp.transform(*body_q[child]), wp.vec3(*xc[:3]))
            gaps.append(float(np.linalg.norm(np.array(wp0) - np.array(wc0))))
        return gaps

    def check_rest(self) -> None:
        """Print per-joint rest-pose anchor error and per-body motion after one substep."""
        body_q0 = self.state_0.body_q.numpy().copy()
        jp_arr = self.model.joint_parent.numpy()
        jc_arr = self.model.joint_child.numpy()
        xp_arr = self.model.joint_X_p.numpy()
        xc_arr = self.model.joint_X_c.numpy()
        print("--- rest joint anchor gaps ---")
        for j, label in enumerate(self.model.joint_label):
            parent, child = int(jp_arr[j]), int(jc_arr[j])
            if child < 0:
                continue
            xp, xc = xp_arr[j], xc_arr[j]
            tp = wp.transform(*xp) if parent < 0 else wp.transform(*body_q0[parent]) * wp.transform(*xp)
            tc = wp.transform(*body_q0[child]) * wp.transform(*xc)
            lin = float(
                np.linalg.norm(np.array(wp.transform_get_translation(tp)) - np.array(wp.transform_get_translation(tc)))
            )
            qrel = wp.quat_inverse(wp.transform_get_rotation(tp)) * wp.transform_get_rotation(tc)
            ang = math.degrees(2.0 * math.acos(min(1.0, abs(float(qrel[3])))))
            print(f"  {label:<50} lin={lin * 1e3:9.4f} mm  ang={ang:8.3f} deg")
        self.state_0.clear_forces()
        self.solver.step(self.state_0, self.state_1, self.control, None, self.sim_dt)
        body_q1 = self.state_1.body_q.numpy()
        print("--- body displacement after one substep ---")
        for b, label in enumerate(self.model.body_label):
            dp = float(np.linalg.norm(body_q1[b, :3] - body_q0[b, :3]))
            qrel = wp.quat_inverse(wp.quat(*body_q0[b, 3:7])) * wp.quat(*body_q1[b, 3:7])
            da = math.degrees(2.0 * math.acos(min(1.0, abs(float(qrel[3])))))
            print(f"  {label:<50} dpos={dp * 1e3:9.4f} mm  drot={da:8.3f} deg")

    def diagnose(self, frames: int) -> None:
        init_finite = bool(np.all(np.isfinite(self.state_0.body_q.numpy())))
        gaps = self._loop_gaps()
        print(f"initial body_q finite={init_finite}  initial gaps_mm=({gaps[0] * 1e3:.4f}, {gaps[1] * 1e3:.4f})")
        print("body masses:", np.round(self.model.body_mass.numpy(), 4))
        print(
            f"{'frame':>5} {'drive_deg':>10} {'pitch_deg':>10} {'roll_deg':>9} {'gapA_mm':>9} {'gapB_mm':>9} {'finite':>7}"
        )
        for frame in range(frames):
            self.step()
            drive = math.degrees(float(self.target_q[self.drive_dofs[0]]))
            pitch = self._joint_angle_deg("left_ankle_pitch_joint")
            roll = self._joint_angle_deg("left_ankle_roll_joint")
            gaps = self._loop_gaps()
            finite = bool(np.all(np.isfinite(self.state_0.body_q.numpy())))
            print(
                f"{frame:>5} {drive:>10.2f} {pitch:>10.2f} {roll:>9.2f}"
                f" {gaps[0] * 1e3:>9.3f} {gaps[1] * 1e3:>9.3f} {finite!s:>7}"
            )
            if not finite:
                break

    @staticmethod
    def create_parser():
        parser = newton.examples.create_parser()
        parser.add_argument("--urdf", type=str, default=DEFAULT_URDF, help="Path to the G1 ankle URDF file.")
        parser.add_argument("--diagnose", action="store_true", help="Print drive/pitch/roll/loop diagnostics and exit.")
        parser.add_argument("--solve", choices=["local", "block_sparse_joints"], default="block_sparse_joints")
        parser.add_argument("--gravity", type=float, default=-9.81)
        parser.add_argument("--iterations", type=int, default=8)
        parser.add_argument("--drive-amplitude-deg", type=float, default=15.0)
        parser.add_argument("--no-retype", action="store_true", help="Keep rod joints revolute (debug).")
        parser.add_argument("--no-loops", action="store_true", help="Skip loop-closing ball joints (debug).")
        parser.add_argument(
            "--check-rest", action="store_true", help="Print rest-pose consistency diagnostics and exit."
        )
        parser.set_defaults(num_frames=240)
        return parser


if __name__ == "__main__":
    parser = Example.create_parser()
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)
    if getattr(args, "check_rest", False):
        example.check_rest()
    elif getattr(args, "diagnose", False):
        example.diagnose(args.num_frames)
    else:
        newton.examples.run(example, args)
