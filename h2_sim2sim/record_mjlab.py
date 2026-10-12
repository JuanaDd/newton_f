# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Record the reference rollout using the H2-enabled mjlab environment."""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.tracking.config.h2.env_cfgs import unitree_h2_flat_tracking_env_cfg
from mjlab.tasks.tracking.config.h2.rl_cfg import unitree_h2_tracking_ppo_runner_cfg
from mjlab.tasks.tracking.config.h2.runner import H2TrackingOnPolicyRunner


def make_env(checkpoint: Path, motion: Path, num_envs: int = 1, keep_running: bool = False):
    """Create deterministic H2 play with the checkpoint's original actor."""
    cfg = unitree_h2_flat_tracking_env_cfg(play=True)
    cfg.scene.num_envs = num_envs
    cfg.auto_reset = False
    cfg.commands["motion"].motion_file = str(motion)
    if keep_running:
        cfg.terminations = {}
    env = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
    wrapped = RslRlVecEnvWrapper(env, clip_actions=None)
    agent_cfg = unitree_h2_tracking_ppo_runner_cfg()
    agent_cfg.actor_checkpoint = str(checkpoint)
    runner = H2TrackingOnPolicyRunner(wrapped, asdict(agent_cfg), device="cuda:0")
    return env, wrapped, runner.get_inference_policy(device="cuda:0")


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--motion", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", default=476, type=int)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    env, wrapped, policy = make_env(args.checkpoint, args.motion)
    robot = env.scene["robot"]
    motion = env.command_manager.get_term("motion")
    records = {k: [] for k in ("obs", "action", "done", "body_pos_w", "body_quat_w", "joint_pos")}

    def record_state():
        for key, value in (
            ("body_pos_w", robot.data.body_link_pos_w),
            ("body_quat_w", robot.data.body_link_quat_w),
            ("joint_pos", robot.data.joint_pos),
        ):
            records[key].append(value[0].cpu().numpy().copy())

    obs = wrapped.get_observations()
    assert int(motion.time_steps[0]) == 1
    with torch.inference_mode():
        for _ in range(min(args.steps, motion.motion.time_step_total - 2)):
            record_state()
            records["obs"].append(obs["actor"][0].cpu().numpy().copy())
            action = policy(obs)
            records["action"].append(action[0].cpu().numpy().copy())
            obs, _, dones, _ = wrapped.step(action)
            records["done"].append(bool(dones[0]))
            if records["done"][-1]:
                break
        record_state()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output,
        **{k: np.asarray(v) for k, v in records.items()},
        body_names=np.array(robot.body_names),
        joint_names=np.array(robot.joint_names),
    )
    metadata = {
        "step_dt": env.step_dt,
        "steps": len(records["action"]),
        "auto_reset": False,
        "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.checkpoint, args.motion)},
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    env.close()


if __name__ == "__main__":
    main()
