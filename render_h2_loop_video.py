# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

###########################################################################
# Render an MP4 of the H2 six-loop example (sim_h2_loop.py) headlessly.
#
# Requires a display; on headless machines wrap with xvfb-run. imageio is
# injected ad hoc so it never becomes a project dependency:
#
#   xvfb-run -a uv run --extra examples --with imageio --with imageio-ffmpeg \
#       python render_h2_loop_video.py --output h2_loop_waist.mp4 -- \
#       --waist-rod ss --drive all --waist-direct
#
# Arguments after `--` are forwarded verbatim to sim_h2_loop.py.
###########################################################################

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import warp as wp


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("h2_loop_video.mp4"))
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("sim_args", nargs="*", help="Arguments forwarded to sim_h2_loop.py (after --).")
    args = parser.parse_args()

    spec = importlib.util.spec_from_file_location("sim_h2_loop", Path(__file__).with_name("sim_h2_loop.py"))
    sim = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim)
    sim_args = sim.Example.create_parser().parse_args(args.sim_args)

    import imageio

    import newton

    viewer = newton.viewer.ViewerGL(width=args.width, height=args.height, headless=True)
    example = sim.Example(viewer, sim_args)

    # Auto-frame the robot from its rest pose.
    body_pos = example.state_0.body_q.numpy()[:, :3]
    center = 0.5 * (body_pos.min(0) + body_pos.max(0))
    extent = max(float(np.max(body_pos.max(0) - body_pos.min(0))), 1.0)
    target = center + np.array([0.0, 0.0, 0.05])
    eye = target + np.array([1.3 * extent, -1.9 * extent, 0.45 * extent])
    front = target - eye
    front /= np.linalg.norm(front)
    viewer.set_camera(
        pos=wp.vec3(*eye),
        pitch=float(np.degrees(np.arcsin(front[2]))),
        yaw=float(np.degrees(np.arctan2(front[1], front[0]))),
    )

    frames = []
    for _ in range(args.frames):
        example.step()
        example.render()
        frames.append(np.ascontiguousarray(viewer.get_frame().numpy()[:, :, :3]))
    viewer.close()

    with imageio.get_writer(args.output, fps=example.fps, codec="libx264", quality=8, macro_block_size=1) as writer:
        for frame in frames:
            writer.append_data(frame)

    example.test_final()
    print(f"wrote {args.output} ({args.frames} frames @ {example.fps} fps); test_final passed")
    print("final loop gaps [mm]:", [round(g * 1e3, 3) for g in example._loop_gaps()])


if __name__ == "__main__":
    main()
