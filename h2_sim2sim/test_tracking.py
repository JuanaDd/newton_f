# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the minimal H2 observation and actuator contract."""

import os
import unittest

import numpy as np
import warp as wp

from .tracking import ACT_DIM, ACTUATED_JOINTS, MJ_JOINTS, OBS_DIM, actuator_params, anchor_obs


class TestTracking(unittest.TestCase):
    @unittest.skipUnless(
        wp.is_cuda_available() and os.getenv("H2_CHECKPOINT") and os.getenv("H2_MOTION"),
        "CUDA and external H2_CHECKPOINT/H2_MOTION inputs required",
    )
    def test_batch_observations_match_single(self):
        """Match all 194 batched observations at standing and airborne poses."""
        from .batch import BatchH2  # noqa: PLC0415 - optional CUDA integration
        from .tracking import build_obs, reset_to_frame  # noqa: PLC0415

        wp.set_device("cuda:0")
        sim = BatchH2(os.environ["H2_CHECKPOINT"], os.environ["H2_MOTION"], 2)
        previous = np.arange(31, dtype=np.float32) / 10
        for frame in (0, 265, 288):
            reset_to_frame(sim.single, sim.motion, frame)
            sim.state_a.body_q.assign(np.tile(sim.single.state_0.body_q.numpy(), (2, 1)))
            sim.state_a.body_qd.assign(np.tile(sim.single.state_0.body_qd.numpy(), (2, 1)))
            sim.frame.assign(np.array([frame + 1], dtype=np.int32))
            sim.actions.assign(np.tile(previous, (2, 1)))
            sim.compute_obs()
            expected, _ = build_obs(sim.single, sim.motion, frame + 1, previous)
            np.testing.assert_allclose(sim.obs.numpy(), np.tile(expected, (2, 1)), rtol=2e-5, atol=2e-5)

    def test_policy_dimensions(self):
        """Preserve the checkpoint's joint ordering and tensor dimensions."""
        self.assertEqual(len(MJ_JOINTS), 37)
        self.assertEqual(len(ACTUATED_JOINTS), 31)
        self.assertEqual((OBS_DIM, ACT_DIM), (194, 31))

    def test_original_gains(self):
        """Keep the full PD damping and original action scaling."""
        kp, kd, effort, scale = actuator_params("left_hip_pitch_joint")
        self.assertAlmostEqual(kd, 40.332520469022626)
        self.assertAlmostEqual(kp * scale, 0.25 * effort)

    def test_anchor_orientation_order(self):
        """Flatten the first two rotation columns in row-major order."""
        pos, ori, _ = anchor_obs(
            np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]), np.array([1.0, 2.0, 3.0]), np.array([1.0, 0.0, 0.0, 0.0])
        )
        np.testing.assert_allclose(pos, [1.0, 2.0, 3.0])
        np.testing.assert_allclose(ori, [1.0, 0.0, 0.0, 1.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
