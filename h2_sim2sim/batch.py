# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""GPU-batched version of the single-world H2 inference contract."""

import numpy as np
import warp as wp

import newton

from .model import build_builder
from .tracking import ANCHOR_BODY, MJ_BODIES, Motion, NewtonH2, Policy, make_solver, reset_to_frame


@wp.kernel
def observations(
    body_q: wp.array[wp.transform],
    body_qd: wp.array[wp.spatial_vector],
    com: wp.array[wp.vec3],
    q: wp.array[float],
    qd: wp.array[float],
    coords: wp.array[int],
    dofs: wp.array[int],
    defaults: wp.array[float],
    ref_q: wp.array2d[float],
    ref_qd: wp.array2d[float],
    ref_anchor: wp.array[wp.transform],
    frame: wp.array[int],
    previous: wp.array2d[float],
    bodies: int,
    coord_stride: int,
    dof_stride: int,
    anchor: int,
    pelvis: int,
    output: wp.array2d[float],
):
    world = wp.tid()
    t = frame[0]
    torso = body_q[world * bodies + anchor]
    reference = ref_anchor[t]
    torso_inv = wp.quat_inverse(wp.transform_get_rotation(torso))
    position = wp.quat_rotate(torso_inv, wp.transform_get_translation(reference) - wp.transform_get_translation(torso))
    orientation = wp.quat_to_matrix(torso_inv * wp.transform_get_rotation(reference))
    root = world * bodies + pelvis
    rotation = wp.transform_get_rotation(body_q[root])
    velocity = body_qd[root]
    angular = wp.vec3(velocity[3], velocity[4], velocity[5])
    linear = wp.vec3(velocity[0], velocity[1], velocity[2]) - wp.cross(angular, wp.quat_rotate(rotation, com[root]))
    lin_b = wp.quat_rotate(wp.quat_inverse(rotation), linear)
    ang_b = wp.quat_rotate(wp.quat_inverse(rotation), angular)
    for j in range(37):
        output[world, j] = ref_q[t, j]
        output[world, 37 + j] = ref_qd[t, j]
        output[world, 89 + j] = q[world * coord_stride + coords[j]] - defaults[j]
        output[world, 126 + j] = qd[world * dof_stride + dofs[j]]
    for j in range(3):
        output[world, 74 + j] = position[j]
        output[world, 77 + 2 * j] = orientation[j, 0]
        output[world, 78 + 2 * j] = orientation[j, 1]
        output[world, 83 + j] = lin_b[j]
        output[world, 86 + j] = ang_b[j]
    for j in range(31):
        output[world, 163 + j] = previous[world, j]


@wp.kernel
def targets_from_actions(
    actions: wp.array2d[float],
    indices: wp.array[int],
    defaults: wp.array[float],
    scales: wp.array[float],
    stride: int,
    targets: wp.array[float],
):
    world, j = wp.tid()
    targets[world * stride + indices[j]] = defaults[j] + scales[j] * actions[world, j]


@wp.kernel
def advance(frame: wp.array[int]):
    frame[0] += 1


class BatchH2:
    """Replicate the original model and keep the observation-policy loop on CUDA."""

    def __init__(self, checkpoint, motion, num_envs, level_parallel=False):
        self.motion = Motion(str(motion))
        self.single = NewtonH2(device="cuda:0")
        reset_to_frame(self.single, self.motion, 0)
        self.num_envs = num_envs
        builder = newton.ModelBuilder()
        builder.replicate(build_builder(), num_envs)
        self.model = builder.finalize(device="cuda:0", skip_validation_joints=True)
        m, s = self.model, self.single
        assert m.articulation_count == num_envs
        assert m.body_count == s.model.body_count * num_envs
        for name in (
            "joint_target_mode",
            "joint_target_ke",
            "joint_target_kd",
            "joint_limit_lower",
            "joint_limit_upper",
            "joint_limit_ke",
            "joint_limit_kd",
            "shape_material_mu",
        ):
            getattr(m, name).assign(np.tile(getattr(s.model, name).numpy(), num_envs))
        self.initial_q = wp.array(np.tile(s.state_0.body_q.numpy(), (num_envs, 1)), dtype=wp.transform)
        self.initial_qd = wp.array(np.tile(s.state_0.body_qd.numpy(), (num_envs, 1)), dtype=wp.spatial_vector)
        self.state_a, self.state_b, self.control = m.state(), m.state(), m.control()
        self.solver = make_solver(m)
        if level_parallel:
            if not hasattr(self.solver, "rigid_articulation_level_parallel"):
                raise ValueError("This solver revision has no level-scheduled solve")
            self.solver.rigid_articulation_level_parallel = True
        self.policy = Policy(str(checkpoint), "cuda:0")
        self.obs = wp.zeros((num_envs, 194), dtype=float)
        self.obs_torch = wp.to_torch(self.obs)
        self.actions = wp.zeros((num_envs, 31), dtype=float)
        self.frame = wp.array([1], dtype=int)
        self.coords = wp.array(s.mj_coord, dtype=int)
        self.dofs = wp.array(s.mj_dof, dtype=int)
        self.defaults37 = wp.array(s.default_q37, dtype=float)
        self.defaults31 = wp.array(s.default_q31, dtype=float)
        self.scales = wp.array(s.action_scale, dtype=float)
        self.target_indices = wp.array(s.act_target_idx, dtype=int)
        self.q = wp.zeros(m.joint_coord_count, dtype=float)
        self.qd = wp.zeros(m.joint_dof_count, dtype=float)
        self.ref_q = wp.array(self.motion.joint_pos, dtype=float)
        self.ref_qd = wp.array(self.motion.joint_vel, dtype=float)
        i = MJ_BODIES.index(ANCHOR_BODY)
        ref = np.concatenate((self.motion.body_pos_w[:, i], self.motion.body_quat_w[:, i, [1, 2, 3, 0]]), axis=1)
        self.ref_anchor = wp.array(ref, dtype=wp.transform)

    def compute_obs(self):
        s, m = self.single, self.model
        newton.eval_ik(m, self.state_a, self.q, self.qd)
        wp.launch(
            observations,
            self.num_envs,
            inputs=[
                self.state_a.body_q,
                self.state_a.body_qd,
                m.body_com,
                self.q,
                self.qd,
                self.coords,
                self.dofs,
                self.defaults37,
                self.ref_q,
                self.ref_qd,
                self.ref_anchor,
                self.frame,
                self.actions,
                s.model.body_count,
                s.model.joint_coord_count,
                s.model.joint_dof_count,
                s.b_index[ANCHOR_BODY],
                s.b_index["pelvis"],
                self.obs,
            ],
        )

    def physics(self):
        wp.launch(
            targets_from_actions,
            (self.num_envs, 31),
            inputs=[
                self.actions,
                self.target_indices,
                self.defaults31,
                self.scales,
                self.single.model.joint_target_q.shape[0],
                self.control.joint_target_q,
            ],
        )
        for _ in range(2):
            for source, destination in ((self.state_a, self.state_b), (self.state_b, self.state_a)):
                source.clear_forces()
                self.pipeline.collide(source, self.contacts)
                self.solver.step(source, destination, self.control, self.contacts, 0.005)
        wp.launch(advance, 1, inputs=[self.frame])

    def reset(self):
        """Start an independent clip with fresh solver and contact history."""
        self.obs_graph = self.physics_graph = None
        self.pipeline = newton.CollisionPipeline(
            self.model, broad_phase="nxn", contact_matching="sticky", contact_matching_pos_threshold=0.005
        )
        self.contacts = self.pipeline.contacts()
        self.solver.reset(self.state_a, flags=0)
        for state in (self.state_a, self.state_b):
            wp.copy(state.body_q, self.initial_q)
            wp.copy(state.body_qd, self.initial_qd)
        self.frame.assign(np.array([1], dtype=np.int32))
        self.actions.zero_()
        self.compute_obs()
        self.physics()
        self.solver.reset(self.state_a, flags=0)
        # Discard the compile-only step's contact matching history too.
        self.pipeline = newton.CollisionPipeline(
            self.model, broad_phase="nxn", contact_matching="sticky", contact_matching_pos_threshold=0.005
        )
        self.contacts = self.pipeline.contacts()
        for state in (self.state_a, self.state_b):
            wp.copy(state.body_q, self.initial_q)
            wp.copy(state.body_qd, self.initial_qd)
        self.frame.assign(np.array([1], dtype=np.int32))
        with wp.ScopedCapture() as capture:
            self.compute_obs()
        self.obs_graph = capture.graph
        with wp.ScopedCapture() as capture:
            self.physics()
        self.physics_graph = capture.graph

    def step(self, events=None):
        wp.capture_launch(self.obs_graph)
        policy = self.policy
        action = policy.mlp((self.obs_torch - policy.mean) / (policy.std + 0.01))
        wp.copy(self.actions, wp.from_torch(action))
        if events is not None:
            wp.record_event(events[0])
        wp.capture_launch(self.physics_graph)
        if events is not None:
            wp.record_event(events[1])

    def status(self):
        q = self.state_a.body_q.numpy().reshape(self.num_envs, self.single.model.body_count, 7)
        z = q[:, self.single.b_index["pelvis"], 2]
        return {
            "finite": bool(np.isfinite(q).all()),
            "pelvis_z_min_median_max": [float(z.min()), float(np.median(z)), float(z.max())],
            "pelvis_below_0_5m_count": int(np.count_nonzero(z < 0.5)),
        }
