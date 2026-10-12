# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Record and plot H2 loop-length residuals during closed-loop policy play."""

import argparse
import hashlib
import importlib.metadata
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import warp as wp

import newton

from .model import DEFAULT_XML


def transform_points(poses: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Transform local points using body-origin poses with xyzw quaternions."""
    poses = np.asarray(poses, dtype=np.float64)
    points = np.asarray(points, dtype=np.float64)
    q = poses[..., 3:7]
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norm < 1e-12):
        raise ValueError("Invalid zero quaternion")
    q = q / norm
    cross = 2 * np.cross(q[..., :3], points)
    return poses[..., :3] + points + q[..., 3:] * cross + np.cross(q[..., :3], cross)


def length_residual(anchors: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return signed endpoint-distance residuals [m], not squared distances."""
    return np.linalg.norm(anchors[..., 1, :] - anchors[..., 0, :], axis=-1) - target


@dataclass
class LoopSpec:
    """Describe constant-length, two-site tendon equalities."""

    names: np.ndarray
    bodies: np.ndarray
    local_points: np.ndarray
    target: np.ndarray
    tendon_ids: np.ndarray
    site_ids: np.ndarray

    @classmethod
    def from_mujoco(cls, model):
        """Read compiled tendon references rather than treating reset poses as rest."""
        import mujoco

        names, bodies, points, targets, tendons, sites = [], [], [], [], [], []
        for eq in range(model.neq):
            if model.eq_type[eq] != mujoco.mjtEq.mjEQ_TENDON:
                continue
            if model.eq_obj2id[eq] != -1 or not model.eq_active0[eq]:
                raise ValueError("Expected active constant-length tendon equalities")
            tendon = int(model.eq_obj1id[eq])
            start = model.tendon_adr[tendon]
            wraps = np.arange(start, start + model.tendon_num[tendon])
            if len(wraps) != 2 or np.any(model.wrap_type[wraps] != mujoco.mjtWrap.mjWRAP_SITE):
                raise ValueError("Expected a straight two-site tendon")
            pair = model.wrap_objid[wraps]
            names.append(model.tendon(tendon).name.rsplit("/", 1)[-1])
            bodies.append([model.body(int(model.site_bodyid[s])).name.rsplit("/", 1)[-1] for s in pair])
            points.append(model.site_pos[pair].copy())
            targets.append(float(model.tendon_length0[tendon] + model.eq_data[eq, 0]))
            tendons.append(tendon)
            sites.append(pair.copy())
        if len(names) != 6 or len(set(names)) != 6:
            raise ValueError("Expected six distinct H2 loop tendons")
        return cls(*(np.asarray(x) for x in (names, bodies, points, targets, tendons, sites)))

    def body_indices(self, labels: list[str]) -> np.ndarray:
        """Resolve body names without depending on backend body ordering."""
        indices = {label.rsplit("/", 1)[-1]: i for i, label in enumerate(labels)}
        return np.array([[indices[name] for name in pair] for pair in self.bodies])


def vbd_rod_geometry(sim, spec: LoopSpec):
    """Match both ball anchors and check VBD rod lengths against tendon targets."""
    single = sim.single
    parent_ids = spec.body_indices(single.body_labels)
    rods, local = [], []
    for pair, points in zip(parent_ids, spec.local_points, strict=True):
        rod_pair, point_pair = [], []
        for body, point in zip(pair, points, strict=True):
            matches = [
                j
                for j in range(single.model.joint_count)
                if single.joint_type[j] == newton.JointType.BALL
                and single.joint_parent[j] == body
                and np.allclose(single.joint_X_p[j, :3], point, atol=1e-6, rtol=0)
            ]
            if len(matches) != 1:
                raise ValueError(f"Cannot uniquely match ball anchor on {single.body_labels[body]}")
            j = matches[0]
            rod_pair.append(single.joint_child[j])
            point_pair.append(single.joint_X_c[j, :3])
        if rod_pair[0] != rod_pair[1]:
            raise ValueError("Loop ends do not share one VBD rod")
        rods.append(rod_pair)
        local.append(point_pair)
    local = np.asarray(local)
    rest = np.linalg.norm(local[:, 1] - local[:, 0], axis=-1)
    np.testing.assert_allclose(rest, spec.target, atol=1e-6, rtol=0)
    return parent_ids, np.asarray(rods), local, rest


def record_vbd(args):
    """Sample actual VBD body poses after each unchanged CUDA-graph play step."""
    import mujoco
    import torch

    from .batch import BatchH2  # noqa: PLC0415 - selected CUDA backend
    from .tracking import MJ_BODIES, termination_reason  # noqa: PLC0415 - selected backend

    spec = LoopSpec.from_mujoco(mujoco.MjModel.from_xml_path(str(DEFAULT_XML)))
    sim = BatchH2(args.checkpoint, args.motion, 1, args.level_parallel)
    if args.steps > sim.motion.num_frames - 2:
        raise ValueError("Requested recording would wrap the reference")
    parents, rods, local, rest = vbd_rod_geometry(sim, spec)
    tracked = [sim.single.b_index[name] for name in MJ_BODIES]
    anchors, gaps, statuses = [], [], []
    with torch.inference_mode():
        for repeat in range(args.repeats):
            torch.manual_seed(args.seed + repeat)
            sim.reset()
            run_anchors, run_gaps, first_threshold = [], [], None
            for frame in range(args.steps + 1):
                if frame:
                    sim.step()
                pose = sim.state_a.body_q.numpy()
                if not np.isfinite(pose).all():
                    raise FloatingPointError(f"Non-finite VBD pose at frame {frame}")
                points = transform_points(pose[parents], spec.local_points)
                rod_points = transform_points(pose[rods], local)
                run_anchors.append(points)
                run_gaps.append(np.linalg.norm(points - rod_points, axis=-1))
                selected = pose[tracked]
                reason = termination_reason(sim.motion, frame, selected[:, :3], selected[:, [6, 3, 4, 5]])
                if frame and reason and first_threshold is None:
                    first_threshold = {"time_s": frame * 0.02, "reason": reason}
            anchors.append(run_anchors)
            gaps.append(run_gaps)
            statuses.append({"first_threshold": first_threshold, **sim.status()})
            print(f"VBD repeat {repeat + 1}/{args.repeats}: {statuses[-1]}", flush=True)
    return (
        spec,
        np.asarray(anchors),
        {"ball_gap_m": np.asarray(gaps), "rod_length_m": rest},
        {
            "level_parallel": args.level_parallel,
            "physics_dt": 0.005,
            "decimation": 4,
            "iterations": 8,
            "cuda_graph": True,
            "statuses": statuses,
        },
    )


def record_mjwarp(args):
    """Sample refreshed mjlab kinematics and cross-check native tendon lengths."""
    import torch

    from .record_mjlab import make_env  # noqa: PLC0415 - optional H2 mjlab installation

    env, wrapped, policy = make_env(args.checkpoint, args.motion, keep_running=True)
    model = env.sim.mj_model
    spec = LoopSpec.from_mujoco(model)
    reference = env.command_manager.get_term("motion")
    if args.steps > reference.motion.time_step_total - 2:
        raise ValueError("Requested recording would wrap the reference")
    if not np.isclose(env.step_dt, 0.02) or not np.isclose(env.physics_dt, 0.005):
        raise ValueError("Expected 50 Hz policy and 5 ms physics in the H2 task")
    indices = spec.body_indices([model.body(i).name for i in range(model.nbody)])
    anchors, native, statuses = [], [], []
    site_difference = 0.0
    try:
        with torch.inference_mode():
            for repeat in range(args.repeats):
                torch.manual_seed(args.seed + repeat)
                obs, _ = wrapped.reset()
                if int(reference.time_steps[0]) != 1:
                    raise ValueError("Expected reference play to start at frame 1")
                run_anchors, run_native, done_count = [], [], 0
                for frame in range(args.steps + 1):
                    if frame:
                        obs, _, done, _ = wrapped.step(policy(obs))
                        done_count += int(done.sum())
                    # mjlab step()/reset() refreshes forward kinematics before returning.
                    # Do not add a forward solve or reconstruct VBD poses with FK.
                    pos = env.sim.data.xpos[0].cpu().numpy()
                    quat = env.sim.data.xquat[0].cpu().numpy()[:, [1, 2, 3, 0]]
                    pose = np.concatenate((pos, quat), axis=-1)
                    if not np.isfinite(pose).all():
                        raise FloatingPointError(f"Non-finite mjwarp pose at frame {frame}")
                    points = transform_points(pose[indices], spec.local_points)
                    sites = env.sim.data.site_xpos[0].cpu().numpy()[spec.site_ids]
                    site_difference = max(site_difference, float(np.max(np.abs(points - sites))))
                    length = env.sim.data.ten_length[0].cpu().numpy()[spec.tendon_ids]
                    np.testing.assert_allclose(
                        np.linalg.norm(points[:, 1] - points[:, 0], axis=-1), length, atol=2e-6, rtol=0
                    )
                    run_anchors.append(points)
                    run_native.append(length - spec.target)
                anchors.append(run_anchors)
                native.append(run_native)
                statuses.append({"termination_events": done_count, "finite": True})
                print(f"mjwarp repeat {repeat + 1}/{args.repeats}: {statuses[-1]}", flush=True)
        return (
            spec,
            np.asarray(anchors),
            {"native_error_m": np.asarray(native)},
            {
                "physics_dt": float(model.opt.timestep),
                "decimation": env.cfg.decimation,
                "iterations": int(model.opt.iterations),
                "ls_iterations": int(model.opt.ls_iterations),
                "integrator": int(model.opt.integrator),
                "solver": int(model.opt.solver),
                "eq_solref": model.eq_solref.tolist(),
                "eq_solimp": model.eq_solimp.tolist(),
                "site_reconstruction_max_m": site_difference,
                "cuda_graph": bool(env.sim.use_cuda_graph),
                "statuses": statuses,
            },
        )
    finally:
        env.close()


def residual_stats(error: np.ndarray) -> dict:
    """Summarize signed residuals in millimeters over all supplied samples."""
    mm = np.asarray(error) * 1000
    return {
        "mean_abs_mm": float(np.mean(np.abs(mm))),
        "rms_mm": float(np.sqrt(np.mean(mm**2))),
        "p95_abs_mm": float(np.percentile(np.abs(mm), 95)),
        "max_abs_mm": float(np.max(np.abs(mm))),
    }


def summarize_record(data) -> dict:
    """Exclude initialization from play statistics and retain per-loop and per-run results."""
    error = data["signed_error_m"][:, 1:]
    result = {
        "all_loops": residual_stats(error),
        "per_loop": {str(name): residual_stats(error[:, :, i]) for i, name in enumerate(data["loop_names"])},
        "per_repeat": [residual_stats(run) for run in error],
        "initial_error_mm": (data["signed_error_m"][:, 0] * 1000).tolist(),
    }
    maximum = np.unravel_index(np.abs(error).argmax(), error.shape)
    result["peak"] = {
        "repeat": int(maximum[0]),
        "time_s": float(data["time_s"][maximum[1] + 1]),
        "loop": str(data["loop_names"][maximum[2]]),
        "signed_mm": float(error[maximum] * 1000),
    }
    if "ball_gap_m" in data:
        result["ball_gap"] = residual_stats(data["ball_gap_m"][:, 1:])
    if "native_error_m" in data:
        result["native_agreement_max_mm"] = float(
            np.max(np.abs(data["native_error_m"] - data["signed_error_m"])) * 1000
        )
    return result


def record(args):
    """Save raw anchors, residuals and machine-independent experiment metadata."""
    if args.output.exists() or args.output.with_suffix(".json").exists():
        raise FileExistsError(args.output)
    if args.steps < 1 or args.repeats < 1:
        raise ValueError("Steps and repeats must be positive")
    if args.backend == "mjwarp" and args.level_parallel:
        raise ValueError("--level-parallel only applies to VBD")
    wp.config.use_precompiled_headers = False
    wp.init()
    wp.set_device("cuda:0")
    spec, anchors, extra, config = (record_vbd if args.backend == "vbd" else record_mjwarp)(args)
    data = {
        "time_s": np.arange(args.steps + 1) * 0.02,
        "loop_names": spec.names,
        "target_length_m": spec.target,
        "anchors_world_m": anchors,
        "signed_error_m": length_residual(anchors, spec.target),
        **extra,
    }
    metadata = {
        "backend": args.backend,
        "steps": args.steps,
        "repeats": args.repeats,
        "seed": args.seed,
        "sampling_dt": 0.02,
        "auto_reset": False,
        "keep_running": True,
        "gpu": wp.get_device().name,
        "versions": {name: importlib.metadata.version(name) for name in ("warp-lang", "numpy", "torch", "mujoco")},
        "input_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (args.checkpoint, args.motion, DEFAULT_XML)
        },
        "configuration": config,
        "statistics": summarize_record(data),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **data)
    args.output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata["statistics"], indent=2), flush=True)


def plot(args):
    """Plot repeat-mean absolute residuals with repeat-range bands, without smoothing."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sources = [("MuJoCo Warp", args.mjwarp, "#2378b5"), ("VBD level", args.vbd, "#d86a18")]
    if args.vbd_default:
        sources.append(("VBD default", args.vbd_default, "#299d78"))
    records = [(label, np.load(source), color) for label, source, color in sources]
    reference = records[0][1]
    summaries = {}
    for label, data, _ in records:
        np.testing.assert_array_equal(data["loop_names"], reference["loop_names"])
        np.testing.assert_allclose(data["target_length_m"], reference["target_length_m"], atol=1e-6, rtol=0)
        np.testing.assert_allclose(data["time_s"], reference["time_s"], atol=1e-8, rtol=0)
        summaries[label] = summarize_record(data)
    outputs = [args.output.with_suffix(suffix) for suffix in (".png", ".svg", ".json")]
    if any(p.exists() for p in outputs):
        raise FileExistsError("Choose a new plot output prefix")
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    display_names = [f"{side}_{joint}_rod_tendon" for joint in ("ankle", "knee", "waist") for side in ("left", "right")]
    for name, ax in zip(display_names, axes.flat, strict=True):
        loop = reference["loop_names"].tolist().index(name)
        for label, data, color in records:
            values = np.abs(data["signed_error_m"][:, :, loop]) * 1000
            time = data["time_s"]
            ax.plot(
                time,
                values.mean(axis=0),
                label=label,
                color=color,
                linewidth=1.25,
                linestyle="--" if label == "VBD default" else "-",
                zorder=3 if label == "VBD level" else 2,
            )
            ax.fill_between(time, values.min(axis=0), values.max(axis=0), color=color, alpha=0.13)
        ax.set_title(str(reference["loop_names"][loop]).removesuffix("_rod_tendon").replace("_", " ").title())
        ax.set_ylabel("Absolute length residual [mm]")
        ax.grid(alpha=0.25)
        ax.set_xlim(0, reference["time_s"][-1])
    for ax in axes[-1]:
        ax.set_xlabel("Policy-play time [s]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(records), bbox_to_anchor=(0.5, 0.945))
    fig.suptitle("H2 closed-loop constraint error | 5 ms physics, 50 Hz policy", fontsize=15, y=0.985)
    fig.text(
        0.5,
        0.018,
        "Lines: mean absolute error across repeats. Bands: min/max. Samples: every 20 ms; no resets or smoothing.",
        ha="center",
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.92))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outputs[0], dpi=150, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(outputs[1], bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    outputs[2].write_text(json.dumps(summaries, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    recorder = commands.add_parser("record")
    recorder.add_argument("backend", choices=("vbd", "mjwarp"))
    recorder.add_argument("--checkpoint", type=Path, required=True)
    recorder.add_argument("--motion", type=Path, required=True)
    recorder.add_argument("--output", type=Path, required=True)
    recorder.add_argument("--steps", type=int, default=476)
    recorder.add_argument("--repeats", type=int, default=3)
    recorder.add_argument("--seed", type=int, default=42)
    recorder.add_argument("--level-parallel", action="store_true")
    recorder.set_defaults(run=record)
    plotting = commands.add_parser("plot")
    plotting.add_argument("--mjwarp", type=Path, required=True)
    plotting.add_argument("--vbd", type=Path, required=True)
    plotting.add_argument("--vbd-default", type=Path)
    plotting.add_argument("--output", type=Path, required=True)
    plotting.set_defaults(run=plot)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
