# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Render saved mjlab ghost and Newton solid world poses without dynamics.

Run with MUJOCO_GL=egl and uv run --no-sync --with pillow --with imageio-ffmpeg.
MuJoCo provides mesh rendering only; neither trajectory is spatially aligned.
"""

import argparse
import copy
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import imageio_ffmpeg
import mujoco  # noqa: TID253 - offline mesh rendering, not Newton simulation
import numpy as np
from PIL import Image, ImageDraw, ImageFont  # noqa: TID253 - offline video labels

GHOST = (0.08, 0.65, 1.0, 0.32)
SOLID = (1.0, 0.47, 0.12, 1.0)
WIDTH, HEIGHT, VIEW_HEIGHT = 1600, 900, 760
HEADER = 92


def rotation(quat: np.ndarray) -> np.ndarray:
    """Convert a WXYZ quaternion to a rotation matrix."""
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, np.asarray(quat, dtype=float))
    return matrix.reshape(3, 3)


def rendering_model(xml: Path) -> mujoco.MjModel:
    """Duplicate the original meshes with independent solid and ghost poses."""
    root = ET.parse(xml).getroot()
    root.find("compiler").set("meshdir", str(xml.resolve().parent / "meshes"))
    # These two copies are visual objects, not interacting simulated robots.
    for tag in ("actuator", "sensor", "keyframe", "equality", "tendon", "contact"):
        for element in root.findall(tag):
            root.remove(element)
    world = root.find("worldbody")
    robot = world.find("body")
    world.remove(robot)
    for prefix, color in (("mjlab", GHOST), ("newton", SOLID)):
        instance = copy.deepcopy(robot)
        for element in instance.iter():
            if "name" in element.attrib:
                element.set("name", f"{prefix}/{element.get('name')}")
            if element.tag == "geom":
                old = np.fromstring(element.get("rgba", "0.7 0.7 0.7 1"), sep=" ")
                shade = 0.65 + 0.35 * np.mean(old[:3])
                element.set("rgba", " ".join(map(str, [*(shade * np.array(color[:3])), color[3]])))
                element.set("contype", "0")
                element.set("conaffinity", "0")
        world.append(instance)
    for texture in root.findall("asset/texture"):
        if texture.get("type") == "skybox":
            texture.set("rgb1", "0.10 0.13 0.18")
            texture.set("rgb2", "0.10 0.13 0.18")
        else:
            texture.set("rgb1", "0.26 0.29 0.34")
            texture.set("rgb2", "0.22 0.25 0.30")
            texture.set("markrgb", "0.40 0.44 0.50")
    for material in root.findall("asset/material"):
        material.set("reflectance", "0")
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    model.vis.global_.offwidth = WIDTH
    model.vis.global_.offheight = VIEW_HEIGHT
    model.vis.global_.fovy = 35
    model.vis.headlight.ambient[:] = 0.35
    model.vis.headlight.diffuse[:] = 0.65
    model.stat.extent = 8
    return model


def geom_mapping(model: mujoco.MjModel, prefix: str, names: list[str]) -> list[tuple[int, int]]:
    """Require recorded poses for every robot geom, including non-tree motion."""
    mapping = []
    for gi, body in enumerate(model.geom_bodyid):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body)) or ""
        if name.startswith(prefix + "/"):
            mapping.append((gi, names.index(name.split("/", 1)[1])))
    if not mapping:
        raise ValueError(f"No robot geoms found for {prefix}")
    return mapping


def set_poses(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    mapping: list[tuple[int, int]],
    positions: np.ndarray,
    quaternions: np.ndarray,
) -> None:
    """Set every geom from its recorded body pose and compiled mesh offset."""
    for gi, body in mapping:
        body_rotation = rotation(quaternions[body])
        data.geom_xpos[gi] = positions[body] + body_rotation @ model.geom_pos[gi]
        data.geom_xmat[gi] = (body_rotation @ rotation(model.geom_quat[gi])).ravel()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mjlab", type=Path, required=True)
    parser.add_argument("--newton", type=Path, required=True)
    parser.add_argument("--xml", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preview", action="store_true", help="Render QA images only, without encoding video.")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mj, nx = np.load(args.mjlab), np.load(args.newton)
    mj_meta = json.loads(args.mjlab.with_suffix(".json").read_text())
    config = json.loads(args.newton.with_name(args.newton.stem + "_summary.json").read_text())["configuration"]
    if "pd_mode" in config:
        raise ValueError("Use a new rollout recorded after target projection was removed")
    times, poses = nx["t"], nx["newton_body_q"]
    count = len(poses)
    dt = config["physics_dt"] * config["decimation"]
    fps = 1 / dt
    if len(mj["body_pos_w"]) != count + 1 or not np.isclose(dt, mj_meta["step_dt"]):
        raise ValueError("Incompatible rollout lengths or control rates")
    np.testing.assert_allclose(times, np.arange(1, count + 1) * dt, atol=1e-5)
    if np.any(mj["done"]):
        raise ValueError("mjlab rollout contains termination/reset events")
    mj_pos, mj_quat = mj["body_pos_w"][1:], mj["body_quat_w"][1:]
    nx_pos, nx_quat = poses[:, :, :3], poses[:, :, [6, 3, 4, 5]]
    for values in (mj_pos, mj_quat, nx_pos, nx_quat):
        if not np.isfinite(values).all():
            raise ValueError("Non-finite recorded poses")
    mj_names, nx_names = list(mj["body_names"]), list(nx["newton_body_labels"])
    pelvis_delta = nx_pos[:, nx_names.index("pelvis")] - mj_pos[:, mj_names.index("pelvis")]
    model = rendering_model(args.xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    mj_map, nx_map = geom_mapping(model, "mjlab", mj_names), geom_mapping(model, "newton", nx_names)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.lookat[:] = [1.15, 0.0, 0.95]
    camera.distance, camera.azimuth, camera.elevation = 5.4, 120, -16
    option = mujoco.MjvOption()
    option.sitegroup[:] = 0
    option.flags[mujoco.mjtVisFlag.mjVIS_TENDON] = False
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font, small = ImageFont.truetype(font_path, 27), ImageFont.truetype(font_path, 22)
    qa_frames = {0, 199, 265, 288, 299, count - 1}
    writer = None
    if not args.preview:
        writer = imageio_ffmpeg.write_frames(
            str(args.output),
            (WIDTH, HEIGHT),
            fps=fps,
            codec="libx264",
            pix_fmt_out="yuv420p",
            macro_block_size=1,
            output_params=["-crf", "18", "-movflags", "+faststart"],
        )
        writer.send(None)
    try:
        with mujoco.Renderer(model, height=VIEW_HEIGHT, width=WIDTH) as renderer:
            for i in sorted(qa_frames) if args.preview else range(count):
                set_poses(model, data, mj_map, mj_pos[i], mj_quat[i])
                set_poses(model, data, nx_map, nx_pos[i], nx_quat[i])
                renderer.update_scene(data, camera=camera, scene_option=option)
                # Disable shadows: scene bodies stay at compile-time positions;
                # only the measured geom transforms define this visualization.
                renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
                frame = Image.new("RGB", (WIDTH, HEIGHT), (17, 23, 32))
                frame.paste(Image.fromarray(renderer.render().copy()), (0, HEADER))
                draw = ImageDraw.Draw(frame)
                draw.text((26, 12), "mjlab / MuJoCo Warp  |  ghost (32% opacity)", font=font, fill=(72, 191, 255))
                draw.text(
                    (830, 12),
                    f"Newton VBD  |  solid  |  {config['physics_dt'] * 1000:g} ms + {config['iterations']} iter",
                    font=font,
                    fill=(255, 166, 75),
                )
                draw.text(
                    (26, 53),
                    f"t = {times[i]:.2f} / {times[-1]:.2f} s   |   1x   |   policy {fps:g} Hz",
                    font=small,
                    fill="white",
                )
                draw.text(
                    (830, 53),
                    f"Pelvis gap: XY {100 * np.linalg.norm(pelvis_delta[i, :2]):.1f} cm   |   Z {100 * pelvis_delta[i, 2]:+.1f} cm",
                    font=small,
                    fill="white",
                )
                draw.text(
                    (26, 864),
                    "Same world coordinates + simulation time  |  No pose alignment  |  VBD: no target projection",
                    font=small,
                    fill=(202, 212, 225),
                )
                if i in qa_frames:
                    frame.save(args.output.with_name(f"{args.output.stem}_frame_{i:03d}.jpg"), quality=95)
                if writer is not None:
                    writer.send(np.asarray(frame))
                if i % 100 == 0:
                    print(f"Rendered {i + 1}/{count}", flush=True)
    finally:
        if writer is not None:
            writer.close()
    if args.preview:
        return
    metadata = {
        "frames": count,
        "fps": fps,
        "duration_s": count / fps,
        "resolution": [WIDTH, HEIGHT],
        "ghost_rgba": GHOST,
        "solid_rgba": SOLID,
        "frame_mapping": f"Newton row i overlays mjlab row i+1; both post-control states at (i+1)*{dt:g} s",
        "spatial_alignment": False,
        "physics_rerun": False,
        "renderer": f"MuJoCo {mujoco.__version__}",
        "newton_configuration": config,
        "camera_lookat": camera.lookat.tolist(),
        "camera_distance": camera.distance,
        "camera_azimuth": camera.azimuth,
        "camera_elevation": camera.elevation,
        "geoms_per_robot": {"mjlab": len(mj_map), "newton": len(nx_map)},
        "note": "Original XML robot geoms use measured world poses. Six added Newton constraint rods have no original XML mesh and are not drawn. No FK replaces recorded poses. No early termination/reset; this is a diagnostic rollout, not a claim of passing thresholds.",
        "sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.mjlab, args.newton, args.xml, args.output)
        },
    }
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
