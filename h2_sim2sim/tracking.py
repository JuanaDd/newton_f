# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the mjlab H2 actor with 5 ms / 8 iteration Newton VBD physics."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path

import numpy as np
import warp as wp

import newton

from .model import DEFAULT_XML, build_builder

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


def make_solver(model: newton.Model):
    """Construct the fixed sim2sim solver configuration."""
    return newton.solvers.SolverVBD(
        model,
        iterations=8,
        rigid_body_contact_buffer_size=256,
        rigid_contact_history=True,
        rigid_articulation_solve="block_sparse_joints",
        rigid_joint_armature=True,
        rigid_articulation_relaxation=0.65,
        rigid_articulation_diagonal_regularization=0.0,
        rigid_avbd_alpha=0.0,
        rigid_avbd_beta=0.0,
        rigid_joint_linear_ke=5e6,
        rigid_joint_angular_ke=5e6,
        rigid_joint_linear_kd=125.0,
        rigid_joint_angular_kd=125.0,
    )


class NewtonH2:
    """Closed-loop H2 using the trained gains and uncapped implicit PD targets."""

    def __init__(self, xml=DEFAULT_XML, device="cuda:0"):
        with wp.ScopedDevice(device):
            self.model = build_builder(xml).finalize(skip_validation_joints=True)
        self.sim_dt, self.decimation, self.control_dt = 0.005, 4, 0.02

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
        self.act_dof = self.mj_dof[self.act_joint_idx_in_37]

        params = np.array([actuator_params(n) for n in ACTUATED_JOINTS])
        self.kp, self.kd, self.effort, self.action_scale = params.T
        self.default_q37 = np.array([DEFAULT_JOINT_POS.get(n, 0.0) for n in MJ_JOINTS])
        self.default_q31 = self.default_q37[self.act_joint_idx_in_37]

        self._configure_gains()
        self._release_position_limits()
        self._configure_friction(0.6)
        self.solver = make_solver(self.model)
        self.tree_joints = [j for j in range(m.joint_count) if self.joint_type[j] != int(newton.JointType.BALL)]
        self.rods = self._collect_rods()
        self.state_0, self.state_1 = m.state(), m.state()
        self.control = m.control()
        self.collision_pipeline = newton.CollisionPipeline(
            m,
            broad_phase="nxn",
            contact_matching="sticky",
            contact_matching_pos_threshold=0.005,
        )
        self.contacts = self.collision_pipeline.contacts()
        self.target_q = m.joint_target_q.numpy().copy()
        self.sim_time = 0.0
        self._ik_q = wp.zeros(m.joint_coord_count, dtype=float, device=m.device)
        self._ik_qd = wp.zeros(m.joint_dof_count, dtype=float, device=m.device)

    def _configure_gains(self) -> None:
        m = self.model
        mode = m.joint_target_mode.numpy()
        ke = m.joint_target_ke.numpy()
        kd = m.joint_target_kd.numpy()
        passive = m.joint_damping.numpy()
        for j, name in enumerate(self.joint_labels):
            if self.joint_type[j] != int(newton.JointType.REVOLUTE):
                continue
            dof = int(self.joint_qd_start[j])
            kd[dof] = passive[dof]
            if name in ACTUATED_JOINTS:
                i = ACTUATED_JOINTS.index(name)
                mode[dof] = int(newton.JointTargetMode.POSITION_VELOCITY)
                ke[dof] = self.kp[i]
                # Passive damping and actuator kd each enter exactly once.
                kd[dof] += self.kd[i]
            else:
                mode[dof] = int(newton.JointTargetMode.NONE)
                ke[dof] = 0.0
        m.joint_target_mode.assign(mode)
        m.joint_target_ke.assign(ke)
        m.joint_target_kd.assign(kd)

    def _release_position_limits(self) -> None:
        """Match the unlimited hinge coordinates used during H2 training."""
        for name, value in (
            ("joint_limit_lower", -np.inf),
            ("joint_limit_upper", np.inf),
            ("joint_limit_ke", 0.0),
            ("joint_limit_kd", 0.0),
        ):
            array = getattr(self.model, name)
            values = array.numpy()
            values[self.mj_dof] = value
            array.assign(values)

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

    def apply_action(self, raw_action: np.ndarray) -> np.ndarray:
        """mjlab JointPositionAction: target = default + scale * a (no clipping)."""
        target = self.default_q31 + self.action_scale * raw_action
        self._pd_target = target
        return target

    def step_control(self) -> None:
        for _ in range(self.decimation):
            # VBD does not enforce effort limits; keep the policy target unchanged.
            self.target_q[self.act_target_idx] = self._pd_target
            self.control.joint_target_q.assign(self.target_q.astype(np.float32))

            self.state_0.clear_forces()
            if self.collision_pipeline is not None:
                self.collision_pipeline.collide(self.state_0, self.contacts)
            self.solver.step(self.state_0, self.state_1, self.control, self.contacts, self.sim_dt)
            self.state_0, self.state_1 = self.state_1, self.state_0
            self.sync_joint_state(self.state_0)
            self.sim_time += self.sim_dt


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


EE_BODIES = ("left_ankle_pitch_link", "right_ankle_pitch_link", "left_wrist_yaw_link", "right_wrist_yaw_link")


def termination_reason(motion: Motion, t: int, positions, quaternions) -> str | None:
    """Check the original mjlab height and projected-gravity thresholds."""
    anchor = MJ_BODIES.index(ANCHOR_BODY)
    if abs(positions[anchor, 2] - motion.body_pos_w[t, anchor, 2]) > 0.25:
        return "anchor_pos"
    actual = q_to_mat(quaternions[anchor])
    reference = q_to_mat(motion.body_quat_w[t, anchor])
    if abs(actual[2, 2] - reference[2, 2]) > 0.8:
        return "anchor_ori"
    if any(
        abs(positions[MJ_BODIES.index(n), 2] - motion.body_pos_w[t, MJ_BODIES.index(n), 2]) > 0.25 for n in EE_BODIES
    ):
        return "ee_body_pos"
    return None


def newton_body_arrays(sim: NewtonH2, body_q):
    idx = [sim.b_index[n] for n in MJ_BODIES]
    pos = body_q[idx, :3].astype(np.float64)
    quat = np.stack([xyzw_to_wxyz(body_q[i, 3:7]) for i in idx])
    return pos, quat


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--xml", type=Path, default=DEFAULT_XML)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=476)
    parser.add_argument("--keep-running", action="store_true", help="Record beyond tracking thresholds without resets.")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    # The optimized sparse assembly can trigger an NVRTC precompiled-header crash.
    wp.config.use_precompiled_headers = False
    wp.init()
    motion = Motion(str(args.motion))
    if not np.isclose(motion.fps, 50):
        raise ValueError("This policy requires a 50 Hz reference")
    sim = NewtonH2(args.xml, args.device)
    policy = Policy(str(args.checkpoint), args.device)
    reset_to_frame(sim, motion, 0)
    last_action = np.zeros(ACT_DIM)
    records = {key: [] for key in ("t", "obs", "actions", "newton_body_q", "body_pos_w", "body_quat_w", "joint_pos")}
    first_failure = None
    for i in range(min(args.steps, motion.num_frames - 2)):
        obs, _ = build_obs(sim, motion, i + 1, last_action)
        if not np.isfinite(obs).all():
            raise FloatingPointError(f"Non-finite observation at control frame {i}")
        action = policy(obs)
        sim.apply_action(action)
        sim.step_control()
        body_q = sim.state_0.body_q.numpy()
        if not np.isfinite(body_q).all():
            raise FloatingPointError(f"Non-finite state at control frame {i}")
        positions, quaternions = newton_body_arrays(sim, body_q)
        joints, _ = sim.read_joints(sim.state_0)
        values = ((i + 1) * 0.02, obs, action, body_q, positions, quaternions, joints)
        for key, value in zip(records, values, strict=True):
            records[key].append(value)
        reason = termination_reason(motion, i + 1, positions, quaternions)
        if reason and first_failure is None:
            first_failure = {"time_s": (i + 1) * 0.02, "reason": reason}
        if i % 100 == 0:
            print(f"frame={i + 1} t={(i + 1) * 0.02:.2f} pelvis_z={positions[0, 2]:.3f}", flush=True)
        if reason and not args.keep_running:
            break
        last_action = action
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        **{key: np.asarray(value) for key, value in records.items()},
        newton_body_labels=np.array(sim.body_labels),
    )
    summary = {
        "steps": len(records["t"]),
        "first_threshold": first_failure,
        "auto_reset": False,
        "configuration": {"physics_dt": 0.005, "decimation": 4, "iterations": 8, "no_terminations": args.keep_running},
        "sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.checkpoint, args.motion, args.xml)
        },
    }
    args.output.with_name(args.output.stem + "_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
