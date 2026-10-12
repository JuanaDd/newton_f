# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check closure metrics independently of policy weights and GPU availability."""

import importlib.util
import unittest

import numpy as np

from .closure import LoopSpec, length_residual, residual_stats, summarize_record, transform_points
from .model import DEFAULT_XML


class TestClosure(unittest.TestCase):
    def test_signed_length_residual(self):
        """Distinguish extension and compression in meters, not squared distance."""
        anchors = np.array([[[0, 0, 0], [0.301, 0, 0]], [[0, 0, 0], [0, 0.298, 0]]])
        np.testing.assert_allclose(length_residual(anchors, np.array([0.3, 0.3])), [0.001, -0.002], atol=1e-15)

    def test_body_origin_rotation(self):
        """Transform an offset site with a rotated, translated body-origin pose."""
        pose = np.array([1, 2, 3, 0, 0, np.sqrt(0.5), np.sqrt(0.5)])
        np.testing.assert_allclose(transform_points(pose, [0.2, 0, 0]), [1, 2.2, 3], atol=1e-15)
        pose[3:] *= -1
        np.testing.assert_allclose(transform_points(pose, [0.2, 0, 0]), [1, 2.2, 3], atol=1e-15)

    def test_zero_quaternion_is_rejected(self):
        """Reject invalid quaternion data instead of silently drawing a finite curve."""
        with self.assertRaises(ValueError):
            transform_points(np.zeros(7), np.zeros(3))

    def test_rigid_motion_invariance(self):
        """Keep loop-length errors invariant under global rigid-body motion."""
        anchors = np.array([[[0, 0, 0], [0.25, 0.1, -0.1]], [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]]])
        pose = np.array([7, -3, 2, 0.5, 0.5, 0.5, 0.5])
        target = np.array([0.3, 0.4])
        np.testing.assert_allclose(
            length_residual(transform_points(pose, anchors), target), length_residual(anchors, target), atol=1e-14
        )

    def test_millimeter_statistics(self):
        """Aggregate absolute values without cancellation and convert meters to millimeters."""
        stats = residual_stats(np.array([-0.001, 0.001, 0.003]))
        self.assertAlmostEqual(stats["mean_abs_mm"], 5 / 3)
        self.assertAlmostEqual(stats["rms_mm"], np.sqrt(11 / 3))
        self.assertAlmostEqual(stats["max_abs_mm"], 3)

    def test_exclude_initial_sample(self):
        """Separate reset geometry from play statistics and retain peak sample time."""
        data = {
            "signed_error_m": np.array([[[1.0], [0.001], [-0.003]]]),
            "time_s": np.array([0, 0.02, 0.04]),
            "loop_names": np.array(["loop"]),
        }
        result = summarize_record(data)
        self.assertEqual(result["all_loops"]["max_abs_mm"], 3)
        self.assertEqual(result["initial_error_mm"], [[1000]])
        self.assertEqual(result["peak"]["time_s"], 0.04)
        self.assertEqual(result["peak"]["signed_mm"], -3)

    def test_body_names_not_indices(self):
        """Resolve prefixed body names independently of backend ordering."""
        spec = LoopSpec(
            np.array(["loop"]),
            np.array([["motor", "output"]]),
            np.zeros((1, 2, 3)),
            np.ones(1),
            np.zeros(1),
            np.zeros((1, 2)),
        )
        np.testing.assert_array_equal(spec.body_indices(["robot/output", "robot/pelvis", "robot/motor"]), [[2, 0]])

    def test_length_does_not_hide_ball_gap(self):
        """Demonstrate that transverse rod displacement can leave loop length exact."""
        anchors = np.array([[[0, 0, 0], [1, 0, 0]]])
        rod_ends = anchors + np.array([0, 0.01, 0])
        np.testing.assert_allclose(length_residual(anchors, np.array([1])), 0)
        np.testing.assert_allclose(np.linalg.norm(rod_ends - anchors, axis=-1), 0.01)

    @unittest.skipUnless(
        importlib.util.find_spec("mujoco") and DEFAULT_XML.exists(), "MuJoCo and external H2 XML required"
    )
    def test_compiled_h2_tendon_residual(self):
        """Match geometric length residuals to MuJoCo equality rows at a perturbed pose."""
        import mujoco

        model = mujoco.MjModel.from_xml_path(str(DEFAULT_XML))
        spec = LoopSpec.from_mujoco(model)
        data = mujoco.MjData(model)
        joint = model.joint("left_ankle_motor_joint")
        data.qpos[joint.qposadr[0]] += 0.1
        mujoco.mj_forward(model, data)
        pose = np.concatenate((data.xpos, data.xquat[:, [1, 2, 3, 0]]), axis=-1)
        indices = spec.body_indices([model.body(i).name for i in range(model.nbody)])
        geometric = length_residual(transform_points(pose[indices], spec.local_points), spec.target)
        np.testing.assert_allclose(geometric, data.ten_length[spec.tendon_ids] - spec.target, atol=1e-12)
        equality = data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY
        np.testing.assert_array_equal(data.efc_id[equality], np.arange(6))
        np.testing.assert_allclose(geometric, data.efc_pos[equality], atol=1e-12)
        self.assertGreater(np.max(np.abs(geometric)), 1e-5)


if __name__ == "__main__":
    unittest.main()
