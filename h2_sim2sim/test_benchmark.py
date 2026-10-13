# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Check the manager-free benchmark contract and optional CUDA equivalence."""

import importlib.util
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import warp as wp

from .benchmark import BatchMjlab, validate_window


class TestBenchmark(unittest.TestCase):
    def test_physics_includes_forward_and_frame_advance(self):
        """Include target processing, four substeps and final forward in physics."""
        calls = Mock()
        robot = SimpleNamespace(update=calls.update)
        sim = BatchMjlab.__new__(BatchMjlab)
        sim.actions = object()
        sim.robot = robot
        sim.motion = SimpleNamespace(time_steps=SimpleNamespace(add_=calls.advance))
        sim.env = SimpleNamespace(
            cfg=SimpleNamespace(decimation=4),
            physics_dt=0.005,
            action_manager=SimpleNamespace(process_action=calls.process, apply_action=calls.apply),
            scene=SimpleNamespace(write_data_to_sim=calls.write),
            sim=SimpleNamespace(step=calls.step, forward=calls.forward),
        )
        sim.physics()
        self.assertEqual(calls.process.call_count, 1)
        self.assertEqual(calls.step.call_count, 4)
        self.assertEqual(calls.forward.call_count, 1)
        self.assertEqual(
            [call[0] for call in calls.mock_calls],
            ["process"] + ["apply", "write", "step", "update"] * 4 + ["forward", "advance"],
        )

    def test_actor_observations_only(self):
        """Skip critic, rewards, termination, metrics and command resampling."""
        sim = BatchMjlab.__new__(BatchMjlab)
        compute = Mock(return_value="actor")
        sim.env = SimpleNamespace(observation_manager=SimpleNamespace(compute_group=compute))
        self.assertEqual(sim.compute_obs(), "actor")
        compute.assert_called_once_with("actor")

    def test_events_enclose_entire_physics_graph(self):
        """Place both CUDA events around the graph that includes final forward."""
        sim = BatchMjlab.__new__(BatchMjlab)
        calls = Mock()
        sim.obs_graph = SimpleNamespace(replay=calls.observations)
        sim.physics_graph = SimpleNamespace(replay=calls.physics)
        sim.obs = np.ones((2, 194))
        sim.actions = SimpleNamespace(copy_=calls.copy)
        sim.policy = SimpleNamespace(mean=0, std=1, mlp=calls.actor)
        with patch("h2_sim2sim.benchmark.wp.record_event", calls.event):
            sim.step(("start", "end"))
        self.assertEqual(
            [call[0] for call in calls.mock_calls], ["observations", "actor", "copy", "event", "physics", "event"]
        )
        self.assertEqual(
            calls.mock_calls[-3:],
            [unittest.mock.call.event("start"), unittest.mock.call.physics(), unittest.mock.call.event("end")],
        )

    def test_reject_reference_wrap_before_timing(self):
        """Reject windows requiring per-frame wrap detection or a reset."""
        validate_window(40, 400, 478)
        with self.assertRaises(ValueError):
            validate_window(40, 438, 478)
        for warmup, steps in ((-1, 1), (0, 0)):
            with self.assertRaises(ValueError):
                validate_window(warmup, steps, 478)


@unittest.skipUnless(
    os.environ.get("H2_CHECKPOINT") and os.environ.get("H2_MOTION") and importlib.util.find_spec("mjlab"),
    "H2_CHECKPOINT, H2_MOTION and CUDA mjlab required",
)
class TestMjlabCuda(unittest.TestCase):
    def test_actor_contract_and_rollout_repeatability(self):
        """Preserve actor inputs and bound drift by full play's repeatability."""
        import torch

        if not torch.cuda.is_available():
            self.skipTest("CUDA required")
        wp.init()
        wp.set_device("cuda:0")
        wp.config.use_precompiled_headers = False
        with torch.inference_mode(), torch.cuda.stream(wp.stream_to_torch("cuda:0")):
            sim = BatchMjlab(os.environ["H2_CHECKPOINT"], os.environ["H2_MOTION"], 8)
            try:
                steps = min(476, sim.motion.motion.time_step_total - 2)
                sim.env.cfg.auto_reset = True
                actor = sim.policy
                full_positions = []
                for _ in range(3):
                    obs, _ = sim.env.reset()
                    positions = []
                    for _ in range(steps):
                        action = actor.mlp((obs["actor"] - actor.mean) / (actor.std + 0.01))
                        obs, _, terminated, truncated, _ = sim.env.step(action)
                        self.assertFalse(bool((terminated | truncated).any()))
                        positions.append(sim.robot.data.body_link_pos_w.clone())
                    full_positions.append(torch.stack(positions).cpu().numpy())
                maxima = dict.fromkeys(("obs", "action"), 0.0)
                minimal_positions = []
                for _ in range(2):
                    sim.reset()
                    positions = []
                    with (
                        patch.object(sim.env, "step", side_effect=AssertionError("full env.step called")),
                        patch.object(sim.env.reward_manager, "compute", side_effect=AssertionError("reward called")),
                        patch.object(
                            sim.env.termination_manager, "compute", side_effect=AssertionError("termination called")
                        ),
                        patch.object(
                            sim.env.command_manager, "compute", side_effect=AssertionError("command manager called")
                        ),
                        patch.object(sim.env.metrics_manager, "compute", side_effect=AssertionError("metrics called")),
                        patch.object(torch.Tensor, "nonzero", side_effect=AssertionError("nonzero called")),
                    ):
                        for _ in range(steps):
                            expected_obs = sim.compute_obs()
                            expected_action = actor.mlp((expected_obs - actor.mean) / (actor.std + 0.01))
                            sim.step()
                            for key, actual, expected in (
                                ("obs", sim.obs, expected_obs),
                                ("action", sim.actions, expected_action),
                            ):
                                error = float((actual - expected).abs().max())
                                maxima[key] = max(maxima[key], error)
                                torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
                            self.assertTrue(bool(torch.isfinite(sim.env.sim.data.qpos).all()))
                            self.assertTrue(bool(torch.isfinite(sim.env.sim.data.qvel).all()))
                            positions.append(sim.robot.data.body_link_pos_w.clone())
                    self.assertTrue(bool((sim.motion.time_steps == steps + 1).all()))
                    self.assertEqual(sim.status()["pelvis_below_0_5m_count"], 0)
                    minimal_positions.append(torch.stack(positions).cpu().numpy())

                def distances(a, b):
                    distance = np.linalg.norm(a - b, axis=-1)
                    return float(np.sqrt(np.mean(distance**2))), float(distance.max())

                baseline = [distances(a, b) for i, a in enumerate(full_positions) for b in full_positions[:i]]
                comparison = [distances(a, b) for a in minimal_positions for b in full_positions]
                baseline_max = np.max(baseline, axis=0)
                comparison_max = np.max(comparison, axis=0)
                # Independent full-play runs are not bitwise repeatable. Do not
                # demand trajectory equality that the reference itself fails.
                self.assertTrue(np.all(comparison_max <= np.maximum(2 * baseline_max, [0.002, 0.005])))
                self.assertLess(comparison_max[1], 0.1)
                print(
                    f"MJWarp validation ({steps} frames, 8 envs): actor maxima={maxima}; "
                    f"full/full (RMS,max) m={baseline}; minimal/full (RMS,max) m={comparison}"
                )
            finally:
                sim.env.close()


if __name__ == "__main__":
    unittest.main()
