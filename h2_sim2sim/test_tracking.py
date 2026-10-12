# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the minimal H2 observation and actuator contract."""

import unittest

import numpy as np

from .tracking import ACT_DIM, ACTUATED_JOINTS, MJ_JOINTS, OBS_DIM, actuator_params, anchor_obs


class TestTracking(unittest.TestCase):
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
