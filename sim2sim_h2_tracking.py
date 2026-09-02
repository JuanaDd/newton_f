# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Sim2sim: run the mjlab `Mjlab-Tracking-Flat-Unitree-H2` policy (trained on
# MuJoCo Warp) on the Newton VBD closed-loop H2 built by sim_h2_loop.py.
#
# The policy interface is reproduced from mjlab (tracking_env_cfg.py,
# mdp/observations.py, mdp/commands.py, h2_constants.py):
#   obs (194) = ref joint_pos(37) | ref joint_vel(37) | anchor_pos_b(3)
#             | anchor_ori_b(6, first two matrix columns, row-major)
#             | pelvis lin vel (pelvis frame, 3) | pelvis ang vel (3)
#             | joint_pos - default (37) | joint_vel (37) | last raw action (31)
#   target_q  = default_q + scale * action, PD kp/kd from armature formula,
#   50 Hz control, 200 Hz physics (decimation 4).
# The reference index in the observation is one frame ahead of the robot
# state (mjlab increments time_steps before computing the observation).
#
# Modes:
#   policy   - closed loop: reset to reference frame 0, roll the policy out
#              with mjlab's terminations; --mjlab-rollout compares the reset
#              observation / first action bit-for-bit and the tracking metrics
#              against a MuJoCo Warp rollout recorded by sim2sim_h2_dump_mjlab.py.
#   replay   - kinematic check: teleport along the reference and inspect the
#              policy's PD targets.
#   fk-check - verify the Newton model reproduces the npz body poses.
#
# Workflow (IsaacLab venv has torch + newton; mjlab dump runs under mjlab's uv):
#   cd ~/project/mjlab && uv run python ~/project/newton-vbd-sparse/sim2sim_h2_dump_mjlab.py \
#       ./model_51499.pt ./h2_360dankou.npz ~/project/newton-vbd-sparse/sim2sim_out/mjlab_rollout.npz
#   cd ~/project/newton-vbd-sparse && ~/project/IsaacLab/.venv/bin/python sim2sim_h2_tracking.py \
#       --mjlab-rollout sim2sim_out/mjlab_rollout.npz --out sim2sim_out/newton.npz
#   xvfb-run -a ~/project/IsaacLab/.venv/bin/python sim2sim_h2_tracking.py \
#       --no-terminations --viewer gl --video sim2sim_out/newton.mp4
#
# Findings (2026-09-02, model_51499.pt / h2_360dankou.npz): the VBD joint
# stiffness must be ~5e6 (IsaacLab H2 value) — at the example's 5e4 the knee
# pushrods stretch several mm under body weight and the robot collapses in
# 0.5 s. With 5e6 Newton tracks for 4.74 s vs 5.18 s in MuJoCo Warp, both
# failing on `ee_body_pos` at the 360-degree jump the policy has not mastered.
# Foot creep: VBD hard contacts use a per-step tangential penalty plus an AL
# dual; the dual only persists with contact matching + rigid_contact_history
# (default here). Without it the standing feet drift ~5 cm per 2 s vs ~2 cm
# in MuJoCo Warp; with it ~1.7 cm. friction_epsilon only affects soft contacts.
###########################################################################

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
import types

import numpy as np
import warp as wp

import newton

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sim_h2_loop as h2

# ---------------------------------------------------------------------------
# mjlab H2 policy interface constants
# ---------------------------------------------------------------------------

# 37 non-free joints in MuJoCo declaration order (== Newton MJCF DFS order).
MJ_JOINTS = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_roll_joint",
    "left_ankle_pitch_joint",
    "left_ankle_motor_joint",
    "left_knee_motor_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_roll_joint",
    "right_ankle_pitch_joint",
    "right_ankle_motor_joint",
    "right_knee_motor_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "head_pitch_joint",
    "head_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "torso_constraint_R_joint",
    "torso_constraint_L_joint",
)

# Linkage outputs closed by the pushrods; everything else is actuated (31).
PASSIVE_JOINTS = (
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
)
ACTUATED_JOINTS = tuple(j for j in MJ_JOINTS if j not in PASSIVE_JOINTS)
assert len(ACTUATED_JOINTS) == 31

# 38 bodies in MuJoCo order (== npz body axis == Newton MJCF DFS order).
MJ_BODIES = (
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "left_ankle_pitch_link",
    "left_ankle_A_link",
    "left_knee_motor_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "right_ankle_pitch_link",
    "right_ankle_A_link",
    "right_knee_motor_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "head_pitch_link",
    "head_yaw_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
    "torso_constraint_R_link",
    "torso_constraint_L_link",
)
ANCHOR_BODY = "torso_link"
IMU_BODY = "pelvis"
TRACKED_BODIES = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)

# h2_constants.py: kp = armature * (2*pi*10)^2, kd = 2 * 2.0 * armature * 2*pi*10,
# action scale = 0.25 * effort / kp.
_NATURAL_FREQ = 2.0 * math.pi * 10.0
_DAMPING_RATIO = 2.0
_ACTUATOR_GROUPS = (
    # (regex, armature, effort limit)
    (r".*_hip_(pitch|roll|yaw)_joint|.*_knee_motor_joint", 0.160478, 254.0),
    (r".*_ankle_(roll|motor)_joint", 0.0190185, 53.7),
    (r"waist_yaw_joint|.*_shoulder_.*_joint|.*_elbow_joint|torso_constraint_[LR]_joint", 0.0251019, 111.0),
    (r"head_.*_joint", 0.01, 50.0),
    (r".*_wrist_roll_joint", 0.01, 60.0),
    (r".*_wrist_(pitch|yaw)_joint", 0.01, 10.0),
)
DEFAULT_JOINT_POS = {
    "left_shoulder_pitch_joint": 0.297,
    "right_shoulder_pitch_joint": 0.297,
    "left_shoulder_roll_joint": 0.281,
    "right_shoulder_roll_joint": -0.281,
    "left_elbow_joint": 0.718,
    "right_elbow_joint": 0.718,
}

OBS_DIM = 2 * 37 + 3 + 6 + 3 + 3 + 37 + 37 + 31
ACT_DIM = 31
NORM_EPS = 1e-2  # rsl_rl EmpiricalNormalization


def actuator_params(name: str) -> tuple[float, float, float, float]:
    """Return (kp, kd, effort, action_scale) for an actuated joint."""
    for pattern, armature, effort in _ACTUATOR_GROUPS:
        if re.fullmatch(pattern, name):
            kp = armature * _NATURAL_FREQ**2
            kd = 2.0 * _DAMPING_RATIO * armature * _NATURAL_FREQ
            return kp, kd, effort, 0.25 * effort / kp
    raise KeyError(name)


# ---------------------------------------------------------------------------
# quaternion helpers (wxyz unless stated otherwise)
# ---------------------------------------------------------------------------


def q_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def q_inv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]]) / np.dot(q, q)


def q_rot(q, v):
    """Rotate vector v by quaternion q."""
    return q_to_mat(q) @ np.asarray(v, dtype=np.float64)


def q_to_mat(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def q_angle(q):
    """Rotation angle [rad] of a unit quaternion."""
    return 2.0 * math.acos(float(np.clip(abs(q[0]), 0.0, 1.0)))


def q_from_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    s = math.sin(0.5 * angle)
    return np.array([math.cos(0.5 * angle), axis[0] * s, axis[1] * s, axis[2] * s])


def q_between(a, b):
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    d = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if d < -0.999999:
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-8:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        return q_from_axis_angle(axis, math.pi)
    c = np.cross(a, b)
    s = math.sqrt((1.0 + d) * 2.0)
    return np.array([0.5 * s, c[0] / s, c[1] / s, c[2] / s])


def xyzw_to_wxyz(q):
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def wxyz_to_xyzw(q):
    return np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)


class Tf:
    """Minimal rigid transform (position + wxyz quaternion) for numpy FK."""

    __slots__ = ("p", "q")

    def __init__(self, p, q):
        self.p = np.asarray(p, dtype=np.float64)
        self.q = np.asarray(q, dtype=np.float64)

    @staticmethod
    def from_newton(t):
        t = np.asarray(t, dtype=np.float64)
        return Tf(t[:3], xyzw_to_wxyz(t[3:7]))

    def to_newton(self):
        return np.concatenate([self.p, wxyz_to_xyzw(self.q)])

    def __mul__(self, other):
        return Tf(self.p + q_rot(self.q, other.p), q_mul(self.q, other.q))

    def inv(self):
        qi = q_inv(self.q)
        return Tf(-q_rot(qi, self.p), qi)

    def point(self, v):
        return self.p + q_rot(self.q, v)


# ---------------------------------------------------------------------------
# policy
# ---------------------------------------------------------------------------


class Policy:
    def __init__(self, checkpoint: str, device: str):
        import torch

        self.torch = torch
        ck = torch.load(checkpoint, map_location=device, weights_only=False)
        sd = ck["actor_state_dict"]
        self.mean = sd["obs_normalizer._mean"].to(torch.float32)
        self.std = sd["obs_normalizer._std"].to(torch.float32)
        assert self.mean.shape[-1] == OBS_DIM, self.mean.shape
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(OBS_DIM, 512),
            torch.nn.ELU(),
            torch.nn.Linear(512, 256),
            torch.nn.ELU(),
            torch.nn.Linear(256, 128),
            torch.nn.ELU(),
            torch.nn.Linear(128, ACT_DIM),
        ).to(device)
        self.mlp.load_state_dict({k[len("mlp.") :]: v for k, v in sd.items() if k.startswith("mlp.")})
        self.mlp.eval()
        self.device = device
        self.iteration = ck.get("iter")

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        torch = self.torch
        with torch.no_grad():
            x = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
            x = (x - self.mean) / (self.std + NORM_EPS)
            return self.mlp(x)[0].cpu().numpy().astype(np.float64)


# ---------------------------------------------------------------------------
# reference motion
# ---------------------------------------------------------------------------


class Motion:
    def __init__(self, path: str):
        d = np.load(path)
        self.fps = float(np.asarray(d["fps"]).reshape(-1)[0])
        self.joint_pos = d["joint_pos"].astype(np.float64)  # [T, 37]
        self.joint_vel = d["joint_vel"].astype(np.float64)
        self.body_pos_w = d["body_pos_w"].astype(np.float64)  # [T, 38, 3]
        self.body_quat_w = d["body_quat_w"].astype(np.float64)  # wxyz
        self.body_lin_vel_w = d["body_lin_vel_w"].astype(np.float64)  # link-origin velocity
        self.body_ang_vel_w = d["body_ang_vel_w"].astype(np.float64)
        self.num_frames = self.joint_pos.shape[0]
        assert self.joint_pos.shape[1] == 37 and self.body_pos_w.shape[1] == 38, (
            self.joint_pos.shape,
            self.body_pos_w.shape,
        )


# ---------------------------------------------------------------------------
# Newton side
# ---------------------------------------------------------------------------


class NewtonH2:
    """Closed-loop H2 on SolverVBD with the mjlab PD interface."""

    def __init__(self, args, viewer):
        sim_args = types.SimpleNamespace(
            source="mjcf",
            mjcf=args.mjcf,
            armature_mjcf=args.mjcf,
            floating=True,
            drive="none",
            iterations=args.iterations,
            contact_buffer=args.contact_buffer,
            solve="block_sparse_joints",
            level_parallel=args.level_parallel,
            ankle_rod="ss",
            knee_rod="ss",
            gravity=-9.81,
        )
        self.ex = h2.Example(viewer, sim_args)
        self.model = self.ex.model
        if args.contact_ke is not None:
            ke = self.model.shape_material_ke.numpy()
            ke[:] = args.contact_ke
            self.model.shape_material_ke.assign(ke)
        if args.contact_history:
            # Persistent contacts let the augmented-Lagrangian friction dual
            # accumulate across steps; without it friction is a per-step
            # penalty and the feet creep under sustained tangential load.
            self.ex.collision_pipeline = newton.CollisionPipeline(
                self.model,
                broad_phase="nxn",
                contact_matching="sticky",
                contact_matching_pos_threshold=args.match_threshold,
            )
            self.ex.contacts = self.ex.collision_pipeline.contacts()
            self.ex._display_contacts = self.ex.collision_pipeline.contacts()
        # Rebuild the solver: the example's 5e4 joint stiffness lets the knee
        # pushrods stretch by >10 mm under body weight; IsaacLab's H2 tasks use 5e6.
        self.solver = newton.solvers.SolverVBD(
            self.model,
            iterations=args.iterations,
            friction_epsilon=args.friction_eps,
            rigid_contact_tangential_stiffness_scale=args.tangential_scale,
            rigid_body_contact_buffer_size=args.contact_buffer,
            rigid_contact_history=args.contact_history,
            rigid_articulation_solve="block_sparse_joints",
            rigid_joint_armature=True,
            rigid_articulation_level_parallel=args.level_parallel,
            rigid_articulation_relaxation=args.relaxation,
            rigid_articulation_diagonal_regularization=0.0,
            rigid_avbd_alpha=0.0,
            rigid_avbd_beta=0.0,
            rigid_joint_linear_ke=args.joint_ke,
            rigid_joint_angular_ke=args.joint_ke,
            rigid_joint_linear_kd=args.joint_kd,
            rigid_joint_angular_kd=args.joint_kd,
        )
        self.ex.solver = self.solver
        self.viewer = viewer
        self.sim_dt = args.physics_dt
        self.decimation = args.decimation
        self.control_dt = self.sim_dt * self.decimation
        self.effort_clip = args.effort_clip

        m = self.model
        self.joint_labels = [l.rsplit("/", 1)[-1] for l in m.joint_label]
        self.body_labels = [l.rsplit("/", 1)[-1] for l in m.body_label]
        self.joint_type = m.joint_type.numpy()
        self.joint_parent = m.joint_parent.numpy()
        self.joint_child = m.joint_child.numpy()
        self.joint_q_start = m.joint_q_start.numpy()
        self.joint_qd_start = m.joint_qd_start.numpy()
        self.joint_target_q_start = m.joint_target_q_start.numpy()
        self.joint_X_p = m.joint_X_p.numpy()
        self.joint_X_c = m.joint_X_c.numpy()
        self.joint_axis = m.joint_axis.numpy()
        self.body_com = m.body_com.numpy()
        self.rest_body_q = m.body_q.numpy().copy()

        self.j_index = {n: self.joint_labels.index(n) for n in MJ_JOINTS}
        self.b_index = {n: self.body_labels.index(n) for n in MJ_BODIES}
        self.free_joint = next(j for j, t in enumerate(self.joint_type) if t == int(newton.JointType.FREE))
        assert self.joint_child[self.free_joint] == self.b_index["pelvis"]

        # Coord / dof indices of the 37 mjlab joints and the 31 actuated ones.
        self.mj_coord = np.array([self.joint_q_start[self.j_index[n]] for n in MJ_JOINTS])
        self.mj_dof = np.array([self.joint_qd_start[self.j_index[n]] for n in MJ_JOINTS])
        self.act_joint_idx_in_37 = np.array([MJ_JOINTS.index(n) for n in ACTUATED_JOINTS])
        self.act_target_idx = np.array([self.joint_target_q_start[self.j_index[n]] for n in ACTUATED_JOINTS])
        self.act_coord = self.mj_coord[self.act_joint_idx_in_37]

        params = np.array([actuator_params(n) for n in ACTUATED_JOINTS])
        self.kp, self.kd, self.effort, self.action_scale = params.T
        self.default_q37 = np.array([DEFAULT_JOINT_POS.get(n, 0.0) for n in MJ_JOINTS])
        self.default_q31 = self.default_q37[self.act_joint_idx_in_37]

        self._configure_gains(args.extra_damping)
        self._configure_friction(args.foot_friction)
        self._show_collider_only_links()

        # Tree joints (parents first) and rod ball joints for the reset FK.
        self.tree_joints = [j for j in range(m.joint_count) if self.joint_type[j] != int(newton.JointType.BALL)]
        self.rods = self._collect_rods()

        self.state_0 = self.ex.state_0
        self.state_1 = self.ex.state_1
        self.control = self.ex.control
        self.contacts = self.ex.contacts
        self.collision_pipeline = self.ex.collision_pipeline
        self.target_q = m.joint_target_q.numpy().copy()
        self.sim_time = 0.0

        # Scratch arrays for eval_ik.
        self._ik_q = wp.zeros(m.joint_coord_count, dtype=float, device=m.device)
        self._ik_qd = wp.zeros(m.joint_dof_count, dtype=float, device=m.device)

    # -- setup ------------------------------------------------------------

    def _configure_gains(self, extra_damping: float) -> None:
        m = self.model
        mode = m.joint_target_mode.numpy()
        ke = m.joint_target_ke.numpy()
        kd = m.joint_target_kd.numpy()
        for j, name in enumerate(self.joint_labels):
            if self.joint_type[j] != int(newton.JointType.REVOLUTE):
                continue
            dof = int(self.joint_qd_start[j])
            if name in ACTUATED_JOINTS:
                i = ACTUATED_JOINTS.index(name)
                mode[dof] = int(newton.JointTargetMode.POSITION_VELOCITY)
                ke[dof] = self.kp[i]
                # kd already carries the MJCF passive damping (0.05); mjlab's
                # MuJoCo model applies the same passive damping on top of the
                # actuator kv, so add rather than replace.
                kd[dof] = kd[dof] + self.kd[i] + extra_damping
            else:
                mode[dof] = int(newton.JointTargetMode.NONE)
                ke[dof] = 0.0
        m.joint_target_mode.assign(mode)
        m.joint_target_ke.assign(ke)
        m.joint_target_kd.assign(kd)

    def _show_collider_only_links(self) -> None:
        """Make collision meshes visible on links that have no visual geom.

        H2loop_complete.xml gives torso/hip_yaw/shoulder_yaw/elbow/wrist links a
        single semi-transparent collision mesh and no visual; Newton imports
        colliders hidden, so those links vanish from renders.
        """
        m = self.model
        flags = m.shape_flags.numpy()
        body = m.shape_body.numpy()
        visible = int(newton.ShapeFlags.VISIBLE)
        has_visual = {int(b) for i, b in enumerate(body) if b >= 0 and flags[i] & visible}
        for i, label in enumerate(m.shape_label):
            if body[i] >= 0 and label.endswith("_collision") and int(body[i]) not in has_visual:
                flags[i] |= visible
        m.shape_flags.assign(flags)
        self.viewer.set_model(m)

    def _configure_friction(self, mu: float | None) -> None:
        """Match mjlab's foot-ground friction.

        mjlab gives the 8 sole capsules friction 0.6 with MuJoCo priority 1, so
        the pair uses 0.6 regardless of the floor. The Newton MJCF carries no
        friction attributes (everything imports at mu=1) and VBD combines pairs
        by the geometric mean, so set both the soles and the floor to `mu`.
        """
        if mu is None:
            return
        m = self.model
        shape_mu = m.shape_material_mu.numpy()
        labels = [l.rsplit("/", 1)[-1] for l in m.shape_label]
        hit = 0
        for i, name in enumerate(labels):
            if re.fullmatch(r"(left|right)_foot_(front|rear|inner|outer)_collision", name) or name == "floor":
                shape_mu[i] = mu
                hit += 1
        assert hit == 9, f"expected 8 sole capsules + floor, matched {hit}"
        m.shape_material_mu.assign(shape_mu)

    def _collect_rods(self):
        """Group the loop-closing ball joints by rod body: rod -> [(joint, link)]."""
        rods = {}
        for j in range(self.model.joint_count):
            if self.joint_type[j] != int(newton.JointType.BALL):
                continue
            rods.setdefault(int(self.joint_child[j]), []).append(j)
        for rod, joints in rods.items():
            assert len(joints) == 2, (self.body_labels[rod], joints)
        return rods

    # -- kinematics -------------------------------------------------------

    def fk(self, root_pos, root_quat_wxyz, q37, root_lin_vel_w=None, root_ang_vel_w=None, qd37=None):
        """Numpy FK on the spanning tree, then place the pushrods between their anchors.

        Returns (body_q [B,7] xyzw, body_qd [B,6] COM twist) in Newton layout.
        Velocities are propagated through the tree when qd37 is given.
        """
        nb = self.model.body_count
        X = [None] * nb
        V = [None] * nb  # (v_origin_world, w_world)
        q_by_joint = dict(zip(MJ_JOINTS, q37, strict=True))
        qd_by_joint = dict(zip(MJ_JOINTS, qd37 if qd37 is not None else np.zeros(37), strict=True))
        for j in self.tree_joints:
            parent, child = int(self.joint_parent[j]), int(self.joint_child[j])
            X_pj = Tf.from_newton(self.joint_X_p[j])
            X_cj = Tf.from_newton(self.joint_X_c[j])
            name = self.joint_labels[j]
            t = self.joint_type[j]
            if t == int(newton.JointType.FREE):
                X_j = Tf(root_pos, root_quat_wxyz)
                v_j_lin = np.zeros(3) if root_lin_vel_w is None else np.asarray(root_lin_vel_w)
                v_j_ang = np.zeros(3) if root_ang_vel_w is None else np.asarray(root_ang_vel_w)
            elif t == int(newton.JointType.REVOLUTE):
                axis = self.joint_axis[int(self.joint_qd_start[j])].astype(np.float64)
                X_j = Tf(np.zeros(3), q_from_axis_angle(axis, q_by_joint.get(name, 0.0)))
                v_j_lin = np.zeros(3)
                v_j_ang = axis * qd_by_joint.get(name, 0.0)
            elif t == int(newton.JointType.FIXED):
                X_j = Tf(np.zeros(3), np.array([1.0, 0, 0, 0]))
                v_j_lin = np.zeros(3)
                v_j_ang = np.zeros(3)
            else:
                raise NotImplementedError(f"joint type {t} ({name})")
            X_wp = Tf(np.zeros(3), np.array([1.0, 0, 0, 0])) if parent < 0 else X[parent]
            X_wpj = X_wp * X_pj
            X_wc = X_wpj * X_j * X_cj.inv()
            X[child] = X_wc
            if t == int(newton.JointType.FREE):
                # root velocity given at the pelvis origin, world frame
                V[child] = (v_j_lin, v_j_ang)
            else:
                v_p, w_p = (np.zeros(3), np.zeros(3)) if parent < 0 else V[parent]
                w_j = q_rot(X_wpj.q, v_j_ang)
                r = X_wc.p - X_wp.p if parent >= 0 else X_wc.p
                v_c = v_p + np.cross(w_p, r) + np.cross(w_j, X_wc.p - X_wpj.p)
                V[child] = (v_c, w_p + w_j)

        for rod, joints in self.rods.items():
            anchors_w, anchors_l, vel_w = [], [], []
            for j in joints:
                link = int(self.joint_parent[j])
                X_pj = Tf.from_newton(self.joint_X_p[j])
                X_cj = Tf.from_newton(self.joint_X_c[j])
                anchors_w.append(X[link].point(X_pj.p))
                anchors_l.append(X_cj.p)
                v_l, w_l = V[link]
                vel_w.append(v_l + np.cross(w_l, anchors_w[-1] - X[link].p))
            rest = Tf.from_newton(self.rest_body_q[rod])
            d_local = anchors_l[1] - anchors_l[0]
            d_rest_w = q_rot(rest.q, d_local)
            d_new_w = anchors_w[1] - anchors_w[0]
            q_rod = q_mul(q_between(d_rest_w, d_new_w), rest.q)
            p_rod = 0.5 * (anchors_w[0] + anchors_w[1]) - q_rot(q_rod, 0.5 * (anchors_l[0] + anchors_l[1]))
            X[rod] = Tf(p_rod, q_rod)
            L = np.linalg.norm(d_new_w)
            w_rod = np.cross(d_new_w, vel_w[1] - vel_w[0]) / (L * L) if L > 1e-9 else np.zeros(3)
            V[rod] = (0.5 * (vel_w[0] + vel_w[1]), w_rod)

        body_q = np.zeros((nb, 7))
        body_qd = np.zeros((nb, 6))
        for b in range(nb):
            if X[b] is None:
                body_q[b] = self.rest_body_q[b]
                continue
            body_q[b] = X[b].to_newton()
            v_o, w = V[b]
            com_w = q_rot(X[b].q, self.body_com[b].astype(np.float64))
            body_qd[b, :3] = v_o + np.cross(w, com_w)
            body_qd[b, 3:] = w
        return body_q, body_qd

    def set_state(self, body_q, body_qd):
        for s in (self.state_0, self.state_1):
            s.body_q.assign(body_q.astype(np.float32))
            s.body_qd.assign(body_qd.astype(np.float32))
        self.sync_joint_state(self.state_0)

    def sync_joint_state(self, state):
        newton.eval_ik(self.model, state, self._ik_q, self._ik_qd)
        state.joint_q.assign(self._ik_q)
        state.joint_qd.assign(self._ik_qd)

    def read_joints(self, state):
        """(q37, qd37) in mjlab order from the current body poses."""
        newton.eval_ik(self.model, state, self._ik_q, self._ik_qd)
        q = self._ik_q.numpy().astype(np.float64)
        qd = self._ik_qd.numpy().astype(np.float64)
        return q[self.mj_coord], qd[self.mj_dof]

    def body_pose(self, state_body_q, name):
        t = state_body_q[self.b_index[name]]
        return t[:3].astype(np.float64), xyzw_to_wxyz(t[3:7])

    def body_origin_velocity(self, state_body_q, state_body_qd, name):
        """(v_origin_world, w_world) of a body from Newton's COM twist."""
        b = self.b_index[name]
        _, q = self.body_pose(state_body_q, name)
        v_com = state_body_qd[b, :3].astype(np.float64)
        w = state_body_qd[b, 3:].astype(np.float64)
        com_w = q_rot(q, self.body_com[b].astype(np.float64))
        return v_com - np.cross(w, com_w), w

    def loop_gaps(self):
        self.ex.state_0 = self.state_0
        return self.ex._loop_gaps()

    # -- stepping -----------------------------------------------------------

    def apply_action(self, raw_action: np.ndarray) -> np.ndarray:
        """mjlab JointPositionAction: target = default + scale * a (no clipping)."""
        target = self.default_q31 + self.action_scale * raw_action
        self._pd_target = target
        return target

    def step_control(self) -> None:
        for _ in range(self.decimation):
            target = self._pd_target
            if self.effort_clip:
                # Emulate MuJoCo's actuator forcerange on the P term: |kp*(t-q)| <= effort.
                q = self.state_0.joint_q.numpy().astype(np.float64)[self.act_coord]
                lim = self.effort / self.kp
                target = q + np.clip(target - q, -lim, lim)
            self.target_q[self.act_target_idx] = target
            self.control.joint_target_q.assign(self.target_q.astype(np.float32))

            self.state_0.clear_forces()
            if self.collision_pipeline is not None:
                self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sync_joint_state(self.state_0)
            self.sim_time += self.sim_dt

    def render(self):
        self.ex.state_0 = self.state_0
        if hasattr(self.viewer, "set_camera"):
            # Follow the pelvis like mjlab's ASSET_BODY tracking camera.
            target = self.state_0.body_q.numpy()[self.b_index["pelvis"]][:3].astype(np.float64)
            target[2] = 0.9
            eye = target + np.array([1.6, -2.2, 0.55])
            front = target - eye
            front /= np.linalg.norm(front)
            self.viewer.set_camera(
                wp.vec3(*eye),
                float(np.degrees(np.arcsin(front[2]))),
                float(np.degrees(np.arctan2(front[1], front[0]))),
            )
        self.viewer.begin_frame(self.sim_time)
        self.viewer.log_state(self.state_0)
        self.viewer.end_frame()


# ---------------------------------------------------------------------------
# observation
# ---------------------------------------------------------------------------


def anchor_obs(robot_p, robot_q, ref_p, ref_q):
    """mjlab motion_anchor_pos_b / motion_anchor_ori_b."""
    q_inv = q_inv_unit(robot_q)
    pos_b = q_rot(q_inv, ref_p - robot_p)
    rel = q_mul(q_inv, ref_q)
    mat = q_to_mat(rel)
    ori_6 = mat[:, :2].reshape(-1)  # row-major over the (3,2) slice
    return pos_b, ori_6, rel


def q_inv_unit(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def build_obs(sim: NewtonH2, motion: Motion, t_ref: int, last_action: np.ndarray, state=None):
    state = state or sim.state_0
    body_q = state.body_q.numpy()
    body_qd = state.body_qd.numpy()
    q37, qd37 = sim.read_joints(state)

    robot_anchor_p, robot_anchor_q = sim.body_pose(body_q, ANCHOR_BODY)
    ref_anchor_p = motion.body_pos_w[t_ref, MJ_BODIES.index(ANCHOR_BODY)]
    ref_anchor_q = motion.body_quat_w[t_ref, MJ_BODIES.index(ANCHOR_BODY)]
    pos_b, ori_6, _ = anchor_obs(robot_anchor_p, robot_anchor_q, ref_anchor_p, ref_anchor_q)

    _, pelvis_q = sim.body_pose(body_q, IMU_BODY)
    v_w, w_w = sim.body_origin_velocity(body_q, body_qd, IMU_BODY)
    pelvis_q_inv = q_inv_unit(pelvis_q)
    lin_vel_b = q_rot(pelvis_q_inv, v_w)
    ang_vel_b = q_rot(pelvis_q_inv, w_w)

    obs = np.concatenate(
        [
            motion.joint_pos[t_ref],
            motion.joint_vel[t_ref],
            pos_b,
            ori_6,
            lin_vel_b,
            ang_vel_b,
            q37 - sim.default_q37,
            qd37,
            last_action,
        ]
    )
    assert obs.shape == (OBS_DIM,)
    return obs, {"q37": q37, "qd37": qd37, "body_q": body_q}


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


EE_BODIES = ("left_ankle_pitch_link", "right_ankle_pitch_link", "left_wrist_yaw_link", "right_wrist_yaw_link")
ACT_IDX_IN_37 = np.array([MJ_JOINTS.index(n) for n in ACTUATED_JOINTS])
METRIC_KEYS = (
    "anchor_pos_err",
    "anchor_pos_err_z",
    "anchor_ori_err_deg",
    "anchor_grav_z_diff",
    "ee_pos_err_z_max",
    "body_pos_err_global",
    "body_pos_err_rel",
    "joint_pos_rmse",
    "joint_pos_rmse_act",
    "pelvis_z",
)


def tracking_metrics(motion: Motion, t: int, body_pos, body_quat, q37):
    """Tracking errors of a robot state (38 body poses in MJ order, wxyz quats) vs reference frame t."""
    ia = MJ_BODIES.index(ANCHOR_BODY)
    robot_anchor_p, robot_anchor_q = body_pos[ia], body_quat[ia]
    ref_anchor_p, ref_anchor_q = motion.body_pos_w[t, ia], motion.body_quat_w[t, ia]
    _, _, rel = anchor_obs(robot_anchor_p, robot_anchor_q, ref_anchor_p, ref_anchor_q)

    R_r = q_to_mat(robot_anchor_q).T
    R_m = q_to_mat(ref_anchor_q).T
    glob, rel_err = [], []
    for name in TRACKED_BODIES:
        i = MJ_BODIES.index(name)
        rp, mp = body_pos[i], motion.body_pos_w[t, i]
        glob.append(np.linalg.norm(rp - mp))
        rel_err.append(np.linalg.norm(R_r @ (rp - robot_anchor_p) - R_m @ (mp - ref_anchor_p)))
    ee_z = max(abs(body_pos[MJ_BODIES.index(n), 2] - motion.body_pos_w[t, MJ_BODIES.index(n), 2]) for n in EE_BODIES)
    # mjlab bad_anchor_ori: projected gravity z in the anchor frame
    g = np.array([0.0, 0.0, -1.0])
    grav_diff = abs((R_r @ g)[2] - (R_m @ g)[2])
    jerr = q37 - motion.joint_pos[t]
    return {
        "anchor_pos_err": float(np.linalg.norm(robot_anchor_p - ref_anchor_p)),
        "anchor_pos_err_z": float(abs(robot_anchor_p[2] - ref_anchor_p[2])),
        "anchor_ori_err_deg": math.degrees(q_angle(rel)),
        "anchor_grav_z_diff": float(grav_diff),
        "ee_pos_err_z_max": float(ee_z),
        "body_pos_err_global": float(np.mean(glob)),
        "body_pos_err_rel": float(np.mean(rel_err)),
        "joint_pos_rmse": float(np.sqrt(np.mean(jerr**2))),
        "joint_pos_rmse_act": float(np.sqrt(np.mean(jerr[ACT_IDX_IN_37] ** 2))),
        "pelvis_z": float(body_pos[0, 2]),
    }


def mjlab_terminated(met: dict) -> str | None:
    """mjlab H2 tracking terminations (thresholds from tracking_env_cfg.py)."""
    if met["anchor_pos_err_z"] > 0.25:
        return "anchor_pos"
    if met["anchor_grav_z_diff"] > 0.8:
        return "anchor_ori"
    if met["ee_pos_err_z_max"] > 0.25:
        return "ee_body_pos"
    return None


def newton_body_arrays(sim: NewtonH2, body_q):
    idx = [sim.b_index[n] for n in MJ_BODIES]
    pos = body_q[idx, :3].astype(np.float64)
    quat = np.stack([xyzw_to_wxyz(body_q[i, 3:7]) for i in idx])
    return pos, quat


OBS_SLICES = {
    "ref_joint_pos": slice(0, 37),
    "ref_joint_vel": slice(37, 74),
    "anchor_pos_b": slice(74, 77),
    "anchor_ori_b": slice(77, 83),
    "base_lin_vel": slice(83, 86),
    "base_ang_vel": slice(86, 89),
    "joint_pos": slice(89, 126),
    "joint_vel": slice(126, 163),
    "last_action": slice(163, 194),
}


def compare_obs(obs_newton, obs_mjlab, label):
    print(f"[obs-check] {label}: per-term max |newton - mjlab|")
    for name, sl in OBS_SLICES.items():
        d = np.abs(obs_newton[sl] - obs_mjlab[sl])
        print(
            f"    {name:14s} max {d.max():.2e}  (mjlab range [{obs_mjlab[sl].min():+.3f}, {obs_mjlab[sl].max():+.3f}])"
        )


def summarize_rollout(rows, label):
    keys = [k for k in rows[0] if k in METRIC_KEYS]
    summary = {
        k: {"mean": float(np.mean([r[k] for r in rows])), "max": float(np.max([r[k] for r in rows]))} for k in keys
    }
    summary["steps"] = len(rows)
    print(f"\n[{label}] {len(rows)} control steps")
    for k in keys:
        print(f"    {k:22s} mean {summary[k]['mean']:9.4f}   max {summary[k]['max']:9.4f}")
    return summary


def mjlab_rollout_metrics(motion: Motion, path: str):
    """Metrics of a recorded mjlab rollout (dump_mjlab_rollout.py) with the same functions."""
    d = np.load(path)
    rows = []
    n_states = d["joint_pos"].shape[0]
    for i in range(1, n_states):
        met = tracking_metrics(
            motion,
            i,
            d["body_pos_w"][i].astype(np.float64),
            d["body_quat_w"][i].astype(np.float64),
            d["joint_pos"][i].astype(np.float64),
        )
        met["step"] = i - 1
        rows.append(met)
    return rows, d


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------


def reset_to_frame(sim: NewtonH2, motion: Motion, t: int) -> None:
    pel = MJ_BODIES.index("pelvis")
    body_q, body_qd = sim.fk(
        motion.body_pos_w[t, pel],
        motion.body_quat_w[t, pel],
        motion.joint_pos[t],
        root_lin_vel_w=motion.body_lin_vel_w[t, pel],
        root_ang_vel_w=motion.body_ang_vel_w[t, pel],
        qd37=motion.joint_vel[t],
    )
    sim.set_state(body_q, body_qd)


def check_fk_against_reference(sim: NewtonH2, motion: Motion, t: int) -> None:
    """Compare the Newton FK of reference joint angles with the npz body poses."""
    reset_to_frame(sim, motion, t)
    body_q = sim.state_0.body_q.numpy()
    errs = []
    for i, name in enumerate(MJ_BODIES):
        p, _ = sim.body_pose(body_q, name)
        errs.append((np.linalg.norm(p - motion.body_pos_w[t, i]), name))
    errs.sort(reverse=True)
    q37, qd37 = sim.read_joints(sim.state_0)
    print(
        f"[fk-check @frame {t}] max body pos err {errs[0][0] * 1e3:.2f} mm ({errs[0][1]}), "
        f"mean {np.mean([e for e, _ in errs]) * 1e3:.2f} mm; "
        f"eval_ik joint err max {np.max(np.abs(q37 - motion.joint_pos[t])):.2e} rad, "
        f"joint vel err max {np.max(np.abs(qd37 - motion.joint_vel[t])):.2e} rad/s; "
        f"loop gap {max(sim.loop_gaps()) * 1e3:.3f} mm"
    )
    for e, name in errs[:3]:
        print(f"    {name:32s} {e * 1e3:7.2f} mm")


def run_replay(sim: NewtonH2, motion: Motion, policy: Policy, args) -> dict:
    """Teleport along the reference and check the policy output kinematically."""
    last_action = np.zeros(ACT_DIM)
    tgt_err, ref_step = [], []
    for t in range(motion.num_frames - 1):
        reset_to_frame(sim, motion, t)
        obs, _ = build_obs(sim, motion, t + 1, last_action)
        a = policy(obs)
        target = sim.apply_action(a)
        ref_next = motion.joint_pos[t + 1, sim.act_joint_idx_in_37]
        tgt_err.append(target - ref_next)
        ref_step.append(motion.joint_pos[t + 1, sim.act_joint_idx_in_37] - motion.joint_pos[t, sim.act_joint_idx_in_37])
        last_action = a
        if args.render_every and t % args.render_every == 0:
            sim.render()
    tgt_err = np.array(tgt_err)
    ref_step = np.array(ref_step)
    per_joint = np.sqrt(np.mean(tgt_err**2, axis=0))
    print("[replay] PD target vs next reference joint pos (actuated joints):")
    print(
        f"    overall RMSE {np.sqrt(np.mean(tgt_err**2)):.4f} rad, "
        f"reference per-step motion RMS {np.sqrt(np.mean(ref_step**2)):.4f} rad"
    )
    order = np.argsort(-per_joint)
    for i in order[:8]:
        print(f"    {ACTUATED_JOINTS[i]:28s} rmse {per_joint[i]:.4f} rad  mean bias {tgt_err[:, i].mean():+.4f}")
    return {"target_rmse": float(np.sqrt(np.mean(tgt_err**2))), "per_joint_rmse": per_joint.tolist()}


def foot_drift(body_hist, motion: Motion, steps: int = 100) -> dict:
    """Path length of each foot in the ground plane minus the reference's, over the first `steps` steps."""
    n = min(steps, len(body_hist))
    out = {"steps": n}
    for key, name in (("left_cm", "left_ankle_pitch_link"), ("right_cm", "right_ankle_pitch_link")):
        i = MJ_BODIES.index(name)
        xy = np.array([b[0][i, :2] for b in body_hist[:n]])
        ref = motion.body_pos_w[1 : n + 1, i, :2]
        path = np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1))
        ref_path = np.sum(np.linalg.norm(np.diff(ref, axis=0), axis=1))
        out[key] = float((path - ref_path) * 100)
    return out


def run_policy(sim: NewtonH2, motion: Motion, policy: Policy, args) -> dict:
    mj = np.load(args.mjlab_rollout) if args.mjlab_rollout else None

    reset_to_frame(sim, motion, 0)
    last_action = np.zeros(ACT_DIM)
    rows = []
    frames = []
    actions = []
    body_hist = []
    t_start = time.time()
    fell_at, reason = None, None
    n_steps = motion.num_frames - 1 if args.max_steps is None else min(args.max_steps, motion.num_frames - 1)
    for k in range(n_steps):
        t_ref = k + 1  # mjlab: time_steps is incremented before the observation
        obs, _ = build_obs(sim, motion, t_ref, last_action)
        if not np.all(np.isfinite(obs)):
            print(f"[policy] non-finite observation at step {k}; aborting")
            fell_at, reason = k, "nan"
            break
        a = policy(obs)
        if mj is not None and k == 0:
            compare_obs(obs, mj["obs"][0], "reset (step 0)")
            print(f"[obs-check] step-0 action max |newton - mjlab| = {np.abs(a - mj['action'][0]).max():.2e}")
        sim.apply_action(a)
        sim.step_control()
        last_action = a
        actions.append(a)

        body_q = sim.state_0.body_q.numpy()
        q37, _ = sim.read_joints(sim.state_0)
        pos, quat = newton_body_arrays(sim, body_q)
        met = tracking_metrics(motion, t_ref, pos, quat, q37)
        met["step"] = k
        met["t"] = t_ref / motion.fps
        met["loop_gap_mm"] = float(max(sim.loop_gaps()) * 1e3)
        rows.append(met)
        body_hist.append((pos.copy(), quat.copy(), q37.copy()))
        if args.render_every and k % args.render_every == 0:
            sim.render()
            if args.video:
                frames.append(np.ascontiguousarray(sim.viewer.get_frame().numpy()[:, :, :3]))
        if k % args.log_every == 0:
            print(
                f"[step {k:4d} t={met['t']:5.2f}s] pelvis_z={met['pelvis_z']:.3f} "
                f"anchor_pos={met['anchor_pos_err'] * 100:5.1f}cm ori={met['anchor_ori_err_deg']:5.1f}deg "
                f"body_rel={met['body_pos_err_rel'] * 100:5.1f}cm joint_rmse={met['joint_pos_rmse']:.3f}rad "
                f"gap={met['loop_gap_mm']:.2f}mm |a|max={np.abs(a).max():.2f}"
            )
        if not np.isfinite(met["pelvis_z"]):
            print(f"[policy] simulation diverged at step {k}")
            fell_at, reason = k, "nan"
            break
        term = mjlab_terminated(met)
        if term and not args.no_terminations:
            fell_at, reason = k, term
            print(
                f"[policy] mjlab termination '{term}' at step {k} (t={met['t']:.2f}s): "
                f"anchor z err {met['anchor_pos_err_z']:.3f} m, grav diff {met['anchor_grav_z_diff']:.2f}, "
                f"ee z err {met['ee_pos_err_z_max']:.3f} m"
            )
            break
    wall = time.time() - t_start

    summary = summarize_rollout(
        rows, "newton" + (" (clip complete)" if fell_at is None else f" (terminated: {reason} @ step {fell_at})")
    )
    drift = foot_drift(body_hist, motion)
    print(
        f"    foot xy drift vs reference over first {drift['steps']} steps: L {drift['left_cm']:.2f} cm, R {drift['right_cm']:.2f} cm"
    )
    summary["foot_drift_cm"] = drift
    summary.update(
        {
            "steps_total": n_steps,
            "survived": fell_at is None,
            "fell_at_step": fell_at,
            "termination": reason,
            "wall_time_s": wall,
            "loop_gap_mm_max": float(np.max([r["loop_gap_mm"] for r in rows])),
        }
    )
    print(
        f"    loop gap max {summary['loop_gap_mm_max']:.3f} mm; wall time {wall:.1f}s "
        f"({len(rows) / max(wall, 1e-9):.1f} control steps/s)"
    )

    if mj is not None:
        mj_rows, _ = mjlab_rollout_metrics(motion, args.mjlab_rollout)
        mj_term = mjlab_terminated(mj_rows[-1]) if bool(mj["done"][-1]) else None
        mj_summary = summarize_rollout(
            mj_rows,
            "mjlab" + (f" (terminated: {mj_term} @ step {len(mj_rows) - 1})" if mj_term else " (clip complete)"),
        )
        n = min(len(rows), len(mj_rows))
        print(f"\n[compare] first {n} common steps, newton vs mjlab (mean):")
        for k in METRIC_KEYS:
            a_ = np.mean([r[k] for r in rows[:n]])
            b_ = np.mean([r[k] for r in mj_rows[:n]])
            print(f"    {k:22s} newton {a_:9.4f}   mjlab {b_:9.4f}")
        act_diff = np.abs(np.array(actions[:n]) - mj["action"][:n])
        print(f"    action |diff| mean over first 10 steps: {act_diff[:10].mean(axis=1).round(3).tolist()}")
        summary["mjlab"] = mj_summary

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        np.savez(
            args.out,
            actions=np.array(actions),
            body_pos_w=np.array([b[0] for b in body_hist]),
            body_quat_w=np.array([b[1] for b in body_hist]),
            joint_pos=np.array([b[2] for b in body_hist]),
            **{k: np.array([r[k] for r in rows]) for k in rows[0]},
        )
        with open(os.path.splitext(args.out)[0] + "_summary.json", "w") as f:
            json.dump(summary, f, indent=2)
        print(f"[policy] wrote {args.out}")
    if args.video and frames:
        import imageio  # noqa: PLC0415 - optional dependency

        with imageio.get_writer(
            args.video,
            fps=int(round(motion.fps / max(args.render_every, 1))),
            codec="libx264",
            quality=8,
            macro_block_size=1,
        ) as w:
            for fr in frames:
                w.append_data(fr)
        print(f"[policy] wrote {args.video} ({len(frames)} frames)")
    return summary


# ---------------------------------------------------------------------------


def make_viewer(args):
    if args.viewer == "null":
        return newton.viewer.ViewerNull(num_frames=10**9)
    if args.viewer == "gl":
        return newton.viewer.ViewerGL(width=args.width, height=args.height, headless=bool(args.video))
    raise ValueError(args.viewer)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=os.path.expanduser("~/project/mjlab/model_51499.pt"))
    p.add_argument("--motion", default=os.path.expanduser("~/project/mjlab/h2_360dankou.npz"))
    p.add_argument("--mjcf", default=h2.DEFAULT_MJCF)
    p.add_argument("--mode", choices=["policy", "replay", "fk-check"], default="policy")
    p.add_argument("--physics-dt", type=float, default=0.005, help="mjlab: 0.005 s")
    p.add_argument("--decimation", type=int, default=4, help="mjlab: 4 (50 Hz control)")
    p.add_argument("--iterations", type=int, default=8)
    p.add_argument("--contact-buffer", type=int, default=256)
    p.add_argument("--level-parallel", action="store_true")
    p.add_argument("--joint-ke", type=float, default=5.0e6, help="VBD rigid joint stiffness (IsaacLab H2: 5e6).")
    p.add_argument("--joint-kd", type=float, default=1.25e2, help="VBD rigid joint damping.")
    p.add_argument("--relaxation", type=float, default=0.65)
    p.add_argument(
        "--friction-eps",
        type=float,
        default=1e-2,
        help="VBD friction_epsilon [m/s]: slip-velocity scale of the regularized Coulomb friction (smaller = less creep).",
    )
    p.add_argument("--tangential-scale", type=float, default=1.0, help="VBD rigid_contact_tangential_stiffness_scale.")
    p.add_argument(
        "--contact-ke",
        type=float,
        default=None,
        help="Override shape_material_ke for all shapes [N/m] (import default 2.5e3).",
    )
    p.add_argument(
        "--no-contact-history",
        dest="contact_history",
        action="store_false",
        help="Disable sticky contact matching + rigid_contact_history (friction duals then reset every "
        "step and the feet creep ~2.5 cm/s while standing).",
    )
    p.add_argument(
        "--match-threshold", type=float, default=0.005, help="Sticky contact matching position threshold [m]."
    )
    p.add_argument(
        "--foot-friction",
        type=float,
        default=0.6,
        help="Foot-ground friction coefficient (mjlab: 0.6; pass a negative value to keep the import default 1.0).",
    )
    p.add_argument(
        "--no-effort-clip",
        dest="effort_clip",
        action="store_false",
        help="Disable the MuJoCo forcerange emulation on the PD P-term.",
    )
    p.add_argument("--extra-damping", type=float, default=0.0, help="Extra joint kd on actuated joints [N·m·s/rad].")
    p.add_argument(
        "--no-terminations", action="store_true", help="Run through the clip even after a mjlab-style termination."
    )
    p.add_argument(
        "--mjlab-rollout",
        type=str,
        default=None,
        help="npz from dump_mjlab_rollout.py; compares observations and metrics against mjlab.",
    )
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--log-every", type=int, default=25)
    p.add_argument("--viewer", choices=["null", "gl"], default="null")
    p.add_argument("--render-every", type=int, default=1, help="Render every N control steps (0 = never).")
    p.add_argument(
        "--video", type=str, default=None, help="Write an MP4 (requires --viewer gl; use xvfb-run headless)."
    )
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--out", type=str, default=None, help="Per-step metrics .npz path.")
    p.add_argument("--device", default="cuda:0" if wp.get_cuda_device_count() > 0 else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    if args.viewer == "null":
        args.render_every = 0
    if args.foot_friction is not None and args.foot_friction < 0:
        args.foot_friction = None
    wp.init()
    viewer = make_viewer(args)
    motion = Motion(args.motion)
    policy = Policy(args.checkpoint, args.device)
    sim = NewtonH2(args, viewer)
    if abs(sim.control_dt - 1.0 / motion.fps) > 1e-6:
        print(f"WARNING: control dt {sim.control_dt} != 1/fps {1.0 / motion.fps}")
    print(
        f"[sim2sim] policy iter {policy.iteration}, motion {motion.num_frames} frames @ {motion.fps:g} fps, "
        f"physics dt {sim.sim_dt} x{sim.decimation}, effort clip {'on' if sim.effort_clip else 'off'}"
    )
    if hasattr(viewer, "set_camera"):
        viewer.set_camera(wp.vec3(3.0, -3.0, 1.6), -20.0, 135.0)

    check_fk_against_reference(sim, motion, 0)
    if args.mode == "fk-check":
        for t in (motion.num_frames // 3, 2 * motion.num_frames // 3, motion.num_frames - 1):
            check_fk_against_reference(sim, motion, t)
        return
    if args.mode == "replay":
        run_replay(sim, motion, policy, args)
        return
    run_policy(sim, motion, policy, args)
    if args.viewer == "gl" and not args.video:
        # keep the window open on the final pose
        while viewer.is_running():
            sim.render()


if __name__ == "__main__":
    main()
