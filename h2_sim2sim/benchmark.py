# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Benchmark closed-loop H2 policy play and its physics-only CUDA time.

FPS counts environment control frames/s, each containing four 5 ms steps.
Run each backend separately with no concurrent GPU workloads.
"""

import argparse
import hashlib
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
import warp as wp


def summarize(seconds, num_envs, steps):
    fps = [num_envs * steps / value for value in seconds]
    return {
        "seconds": seconds,
        "env_fps": fps,
        "median_env_fps": float(np.median(fps)),
        "median_batch_fps": float(np.median(fps)) / num_envs,
        "median_physics_env_steps_per_s": 4 * float(np.median(fps)),
    }


def timed_steps(step, count):
    """Measure synchronized wall time including actor and environment."""
    wp.synchronize_device("cuda:0")
    start = time.perf_counter()
    for _ in range(count):
        step()
    wp.synchronize_device("cuda:0")
    return time.perf_counter() - start


def make_events(count):
    return [(wp.Event(enable_timing=True), wp.Event(enable_timing=True)) for _ in range(count)]


def event_seconds(events):
    return sum(wp.get_event_elapsed_time(a, b, synchronize=False) for a, b in events) / 1000


def validate_window(warmup, steps, total_frames):
    """Reject reference wraparound before entering a reset-free timed loop."""
    if warmup < 0 or steps < 1 or warmup + steps >= total_frames - 1:
        raise ValueError("Require nonnegative warmup, positive steps and no reference wrap")


class BatchMjlab:
    """Run the H2 mjlab actor contract without training-only task managers.

    Like BatchH2, capture observations and a four-substep physics update in
    separate CUDA graphs and run the same Torch actor outside both graphs.
    Call under inference_mode on the Warp stream. Only fixed, non-wrapping
    motion windows are supported; reset explicitly between trials.
    """

    def __init__(self, checkpoint, motion, num_envs):
        import torch

        from .record_mjlab import make_env  # noqa: PLC0415 - optional mjlab environment
        from .tracking import Policy  # noqa: PLC0415 - selected backend only

        self.env, _, _ = make_env(Path(checkpoint), Path(motion), num_envs)
        self.robot = self.env.scene["robot"]
        self.motion = self.env.command_manager.get_term("motion")
        self.policy = Policy(str(checkpoint), "cuda:0")
        self.actions = torch.zeros((num_envs, 31), device="cuda:0")
        if self.env.cfg.decimation != 4 or not np.isclose(self.env.physics_dt, 0.005):
            raise ValueError("Benchmark requires four 5 ms substeps")
        actor_cfg = self.env.cfg.observations["actor"]
        if actor_cfg.enable_corruption or actor_cfg.nan_policy != "disabled":
            raise ValueError("Benchmark requires deterministic, synchronization-free actor observations")
        if actor_cfg.history_length or any(t.history_length or t.delay_max_lag for t in actor_cfg.terms.values()):
            raise ValueError("Benchmark does not support observation history or delays")

    def compute_obs(self):
        """Compute only the 194 actor inputs, excluding critic and task metrics."""
        return self.env.observation_manager.compute_group("actor")

    def physics(self):
        """Apply controls, advance 20 ms and refresh derived state for observations."""
        env = self.env
        env.action_manager.process_action(self.actions)
        for _ in range(env.cfg.decimation):
            env.action_manager.apply_action()
            env.scene.write_data_to_sim()
            env.sim.step()
            self.robot.update(env.physics_dt)
        # mj_step leaves derived state one substep behind; retain the same
        # refresh as full play, and include its dynamics/constraints in physics.
        env.sim.forward()
        # Actor inputs need the frame index, not command metrics, body-pose
        # reward targets, resampling timers or their nonzero/where synchronizations.
        self.motion.time_steps.add_(1)

    def reset(self):
        """Warm kernels, capture fixed-shape work and restore the initial state."""
        import torch

        self.obs_graph = self.physics_graph = None
        self.env.reset()
        self.actions.zero_()
        # Capture the underlying launches, not mjlab's already-instantiated
        # step/forward graphs (nested graph launch is unsupported by this Warp).
        use_cuda_graph = self.env.sim.use_cuda_graph
        self.env.sim.use_cuda_graph = False
        try:
            self.compute_obs()
            self.physics()
            self.env.reset()
            # Torch must own capture so allocating observation/action kernels
            # retain graph-private storage instead of reusing freed temporaries.
            self.obs_graph = torch.cuda.CUDAGraph()
            self.physics_graph = torch.cuda.CUDAGraph()
            # mjlab TorchArray writes retain the stream used at construction.
            # Capture on that same stream so ctrl writes cannot escape capture.
            capture_stream = wp.stream_to_torch("cuda:0")
            with (
                wp.ScopedStream(wp.stream_from_torch(capture_stream)),
                torch.cuda.graph(self.obs_graph, stream=capture_stream),
            ):
                wp.capture_begin(force_module_load=False, external=True)
                try:
                    self.obs = self.compute_obs()
                finally:
                    wp.capture_end()
            with (
                wp.ScopedStream(wp.stream_from_torch(capture_stream)),
                torch.cuda.graph(self.physics_graph, stream=capture_stream),
            ):
                wp.capture_begin(force_module_load=False, external=True)
                try:
                    self.physics()
                finally:
                    wp.capture_end()
        finally:
            self.env.sim.use_cuda_graph = use_cuda_graph
        self.env.reset()
        self.actions.zero_()
        if not bool((self.motion.time_steps == 1).all()):
            raise ValueError("Expected deterministic motion initialization at frame 1")

    def step(self, events=None):
        self.obs_graph.replay()
        policy = self.policy
        action = policy.mlp((self.obs - policy.mean) / (policy.std + 0.01))
        self.actions.copy_(action)
        if events is not None:
            wp.record_event(events[0])
        self.physics_graph.replay()
        if events is not None:
            wp.record_event(events[1])

    def status(self):
        import torch

        z = self.robot.data.body_link_pos_w[:, 0, 2]
        return {
            "finite": bool(torch.isfinite(self.env.sim.data.qpos).all() & torch.isfinite(self.env.sim.data.qvel).all()),
            "termination_checks": False,
            "auto_reset": False,
            "pelvis_z_min_median_max": [float(z.min()), float(z.median()), float(z.max())],
            "pelvis_below_0_5m_count": int((z < 0.5).sum()),
        }


def benchmark_vbd(args):
    import torch

    from .batch import BatchH2  # noqa: PLC0415 - selected backend only

    sim = BatchH2(args.checkpoint, args.motion, args.num_envs, args.level_parallel)
    validate_window(args.warmup, args.steps, sim.motion.num_frames)
    wall, physics, statuses = [], [], []
    with torch.inference_mode():
        for repeat in range(args.repeats):
            sim.reset()
            timed_steps(sim.step, args.warmup)
            events = make_events(args.steps)
            event_iter = iter(events)
            wall.append(timed_steps(lambda event_iter=event_iter: sim.step(next(event_iter)), args.steps))
            physics.append(event_seconds(events))
            statuses.append(sim.status())
            if not statuses[-1]["finite"]:
                raise FloatingPointError("Non-finite VBD state")
            print(f"VBD repeat={repeat} wall={wall[-1]:.3f}s physics={physics[-1]:.3f}s {statuses[-1]}", flush=True)
    return {
        "policy_play": summarize(wall, args.num_envs, args.steps),
        "physics_cuda": summarize(physics, args.num_envs, args.steps),
        "status": statuses,
        "level_parallel": args.level_parallel,
        "physics_scope": "GPU target conversion + clear forces + collision + 4 VBD steps",
        "play_scope": "GPU IK/194 observations + actor + physics; no rewards/termination/reset managers or rendering",
    }


def benchmark_mjlab(args):
    import torch

    with torch.inference_mode():
        sim = BatchMjlab(args.checkpoint, args.motion, args.num_envs)
        try:
            validate_window(args.warmup, args.steps, sim.motion.motion.time_step_total)
            wall, physics, statuses = [], [], []
            for repeat in range(args.repeats):
                sim.reset()
                timed_steps(sim.step, args.warmup)
                events = make_events(args.steps)
                event_iter = iter(events)
                wall.append(timed_steps(lambda event_iter=event_iter: sim.step(next(event_iter)), args.steps))
                physics.append(event_seconds(events))
                statuses.append(sim.status())
                if not statuses[-1]["finite"]:
                    raise FloatingPointError("Non-finite mjlab state")
                print(
                    f"mjlab repeat={repeat} wall={wall[-1]:.3f}s physics={physics[-1]:.3f}s {statuses[-1]}", flush=True
                )
            model = sim.env.sim.mj_model
            return {
                "policy_play": summarize(wall, args.num_envs, args.steps),
                "physics_cuda": summarize(physics, args.num_envs, args.steps),
                "status": statuses,
                "physics_dt": float(model.opt.timestep),
                "iterations": int(model.opt.iterations),
                "ls_iterations": int(model.opt.ls_iterations),
                "physics_scope": "GPU target conversion + four MuJoCo Warp steps + final forward + frame advance",
                "play_scope": "194 actor observations + shared Torch actor + physics; no rewards/termination/reset managers or rendering",
                "cuda_graphs": "separate actor-observation and four-substep-plus-forward graphs; actor outside graphs",
            }
        finally:
            sim.env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backend", choices=("vbd", "mjlab"))
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--level-parallel", action="store_true")
    args = parser.parse_args()
    if args.num_envs < 1 or args.repeats < 1:
        parser.error("num-envs and repeats must be positive")
    if args.output.exists():
        raise FileExistsError(args.output)
    # Avoid an NVRTC PCH crash when compiling the large sparse assembly kernel.
    # This affects compilation only; use the same setting for both backends.
    wp.config.use_precompiled_headers = False
    wp.init()
    wp.set_device("cuda:0")
    import torch

    # Keep Torch operations, Warp launches and timing events on one stream,
    # including Torch operations captured inside the mjlab CUDA graphs.
    with torch.cuda.stream(wp.stream_to_torch("cuda:0")):
        result = (benchmark_vbd if args.backend == "vbd" else benchmark_mjlab)(args)
    result.update(
        {
            "backend": args.backend,
            "timing_protocol": "aligned_minimal_rollout_v2",
            "num_envs": args.num_envs,
            "warmup": args.warmup,
            "steps": args.steps,
            "repeats": args.repeats,
            "control_dt": 0.02,
            "use_precompiled_headers": wp.config.use_precompiled_headers,
            "gpu": wp.get_device().name,
            "hostname": platform.node(),
            "python": platform.python_version(),
            "versions": {name: importlib.metadata.version(name) for name in ("warp-lang", "numpy", "torch")},
            "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.checkpoint, args.motion)},
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
