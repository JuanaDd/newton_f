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


def benchmark_vbd(args):
    import torch

    from .batch import BatchH2  # noqa: PLC0415 - selected backend only

    sim = BatchH2(args.checkpoint, args.motion, args.num_envs, args.level_parallel)
    if args.steps + args.warmup >= sim.motion.num_frames - 1:
        raise ValueError("Measured clip would wrap the reference")
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

    from .record_mjlab import make_env  # noqa: PLC0415 - optional mjlab environment

    env, wrapped, policy = make_env(args.checkpoint, args.motion, args.num_envs)
    env.cfg.auto_reset = True
    total_frames = env.command_manager.get_term("motion").motion.time_step_total
    if args.steps + args.warmup >= total_frames - 1:
        raise ValueError("Measured clip would wrap the reference")
    wall, physics, statuses = [], [], []
    original_step = env.sim.step
    event_iter = None

    def timed_physics_step():
        if event_iter is None:
            original_step()
        else:
            a, b = next(event_iter)
            wp.record_event(a)
            original_step()
            wp.record_event(b)

    env.sim.step = timed_physics_step
    with torch.inference_mode():
        for repeat in range(args.repeats):
            obs, _ = wrapped.reset()
            done_count = torch.zeros((), dtype=torch.int64, device="cuda:0")

            def step(done_count=done_count):
                nonlocal obs
                obs, _, dones, _ = wrapped.step(policy(obs))
                done_count.add_(dones.sum())

            timed_steps(step, args.warmup)
            events = make_events(4 * args.steps)
            event_iter = iter(events)
            wall.append(timed_steps(step, args.steps))
            event_iter = None
            physics.append(event_seconds(events))
            z = env.scene["robot"].data.body_link_pos_w[:, 0, 2]
            statuses.append(
                {
                    "finite": bool(torch.isfinite(env.sim.data.qpos).all()),
                    "termination_events": int(done_count),
                    "pelvis_z_min_median_max": [float(z.min()), float(z.median()), float(z.max())],
                    "pelvis_below_0_5m_count": int((z < 0.5).sum()),
                }
            )
            if not statuses[-1]["finite"]:
                raise FloatingPointError("Non-finite mjlab state")
            print(f"mjlab repeat={repeat} wall={wall[-1]:.3f}s physics={physics[-1]:.3f}s {statuses[-1]}", flush=True)
    model = env.sim.mj_model
    result = {
        "policy_play": summarize(wall, args.num_envs, args.steps),
        "physics_cuda": summarize(physics, args.num_envs, args.steps),
        "status": statuses,
        "physics_dt": float(model.opt.timestep),
        "iterations": int(model.opt.iterations),
        "ls_iterations": int(model.opt.ls_iterations),
        "physics_scope": "sum of four MuJoCo Warp physics steps including collision",
        "play_scope": "actor + physics + observations/rewards/terminations/managers; no rendering",
    }
    env.close()
    return result


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
    if args.output.exists():
        raise FileExistsError(args.output)
    # Avoid an NVRTC PCH crash when compiling the large sparse assembly kernel.
    # This affects compilation only; use the same setting for both backends.
    wp.config.use_precompiled_headers = False
    wp.init()
    wp.set_device("cuda:0")
    result = (benchmark_vbd if args.backend == "vbd" else benchmark_mjlab)(args)
    result.update(
        {
            "backend": args.backend,
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
