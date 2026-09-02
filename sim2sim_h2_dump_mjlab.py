"""Roll the H2 tracking policy out in mjlab (MuJoCo Warp) and dump the trajectory as sim2sim ground truth.

Usage (from ~/project/mjlab):
    uv run python <this file> <checkpoint.pt> <motion.npz> <out.npz> [--no-terminations] [--video out.mp4]
"""

import argparse
from dataclasses import asdict

import numpy as np
import torch  # noqa: TID253 - script runs under the mjlab environment
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.tracking.config.h2.env_cfgs import unitree_h2_flat_tracking_env_cfg
from mjlab.tasks.tracking.config.h2.rl_cfg import unitree_h2_tracking_ppo_runner_cfg
from mjlab.tasks.tracking.rl.runner import MotionTrackingOnPolicyRunner

ap = argparse.ArgumentParser()
ap.add_argument("checkpoint")
ap.add_argument("motion_file")
ap.add_argument("out_path")
ap.add_argument("--no-terminations", action="store_true", help="Run the whole clip even after a fall.")
ap.add_argument("--video", default=None, help="Write an MP4 of the rollout (offscreen render, 1280x720).")
cli = ap.parse_args()
checkpoint, motion_file, out_path = cli.checkpoint, cli.motion_file, cli.out_path
device = "cuda:0"

cfg = unitree_h2_flat_tracking_env_cfg(play=True)
cfg.scene.num_envs = 1
cfg.commands["motion"].motion_file = motion_file
cfg.commands["motion"].joint_position_range = (0.0, 0.0)
# Drop startup domain randomization (friction, COM, encoder bias) for a clean reference.
for key in [k for k, v in cfg.events.items() if v.mode == "startup"]:
    cfg.events.pop(key)
if cli.no_terminations:
    cfg.terminations = {}
if cli.video:
    cfg.viewer.width, cfg.viewer.height = 1280, 720
env = ManagerBasedRlEnv(cfg=cfg, device=device, render_mode="rgb_array" if cli.video else None)
wrapped = RslRlVecEnvWrapper(env, clip_actions=None)

agent_cfg = unitree_h2_tracking_ppo_runner_cfg()
runner = MotionTrackingOnPolicyRunner(wrapped, asdict(agent_cfg), device=device)
runner.load(checkpoint, load_cfg={"actor": True}, strict=True, map_location=device)
policy = runner.get_inference_policy(device=device)

robot = env.scene["robot"]
motion_term = env.command_manager.get_term("motion")
T = motion_term.motion.time_step_total
print("joint names:", robot.joint_names)
print("body names:", robot.body_names)
print("num frames:", T)

rec = {k: [] for k in ("obs", "action", "time_step", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "done")}
frames = []
obs_td = wrapped.get_observations()
with torch.no_grad():
    for k in range(T - 1):
        if cli.video:
            frames.append(env.render())
        rec["obs"].append(obs_td["actor"][0].cpu().numpy().copy())
        rec["time_step"].append(int(motion_term.time_steps[0].item()))
        rec["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
        rec["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
        rec["body_pos_w"].append(robot.data.body_link_pos_w[0].cpu().numpy().copy())
        rec["body_quat_w"].append(robot.data.body_link_quat_w[0].cpu().numpy().copy())
        action = policy(obs_td)
        rec["action"].append(action[0].cpu().numpy().copy())
        obs_td, _, dones, _ = wrapped.step(action)
        rec["done"].append(bool(dones[0].item()))
        if rec["done"][-1]:
            print(f"episode ended at step {k}")
            break
# final state after the last step
rec["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
rec["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
rec["body_pos_w"].append(robot.data.body_link_pos_w[0].cpu().numpy().copy())
rec["body_quat_w"].append(robot.data.body_link_quat_w[0].cpu().numpy().copy())
rec["time_step"].append(int(motion_term.time_steps[0].item()))

if cli.video:
    import imageio

    with imageio.get_writer(
        cli.video, fps=int(round(1.0 / env.step_dt)), codec="libx264", quality=8, macro_block_size=1
    ) as w:
        for fr in frames:
            w.append_data(fr)
    print("wrote", cli.video, len(frames), "frames")
np.savez(
    out_path,
    **{k: np.array(v) for k, v in rec.items()},
    joint_names=np.array(robot.joint_names),
    body_names=np.array(robot.body_names),
)
print("wrote", out_path, {k: np.array(v).shape for k, v in rec.items()})
env.close()
