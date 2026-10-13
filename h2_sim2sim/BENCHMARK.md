# H2 sim2sim throughput

Compare closed-loop H2 policy inference in mjlab / MuJoCo Warp with Newton VBD.
The current protocol aligns inference workloads; earlier full-play numbers and
VBD optimization measurements are retained separately below. See [README.md](README.md) for
the task, external inputs, policy contract and overlay-video commands.

## Configuration and timing

Recorded hardware: NVIDIA RTX PRO 5000 72GB Blackwell.
Recorded environment: Python 3.13.15, Torch 2.9.0+cu128, Warp
1.16.0.dev20260716, NumPy 2.5.1, MuJoCo / MuJoCo Warp 3.11.0, H2-enabled mjlab.

- Both backends: 5 ms physics steps, decimation 4, 50 Hz policy, original PD gains.
- VBD: 8 iterations, block-sparse articulated solve, relaxation 0.65,
  six closed-loop rods, coupled armature and sticky contact history.
- mjlab: `implicitfast`, Newton solver, maximum 100 solver iterations and
  50 line-search iterations. These settings come from the H2 mjlab task.
- Independent environments start at the same motion phase, without play randomization.
- Each trial resets, warms up 40 control frames and measures the next 400 frames
  (0.8–8.8 s, including jump and landing). Report medians of three trials.
- Run sequentially on an otherwise idle GPU. Exclude initialization, compilation,
  graph capture and rendering. This is policy play, not training throughput.

`environment FPS = environment count × measured control frames / elapsed seconds`.
One control frame includes four physics steps. Divide by the environment count
for whole-batch updates/s; multiply by four for physics-environment steps/s.

| Timing scope | Included work |
|---|---|
| VBD physics | One CUDA-event interval around target conversion, force clearing, collision, four VBD steps and reference-frame advance |
| mjlab physics | One CUDA-event interval around target conversion, four MuJoCo Warp steps, the final `forward()` and reference-frame advance |
| Both play / e2e | Synchronized wall time for 194 actor observations, the same normalized Torch actor, action transfer and the physics interval |

The JSON identifies this as `aligned_minimal_rollout_v2`. Both timed paths omit
rewards, termination checks, metrics, automatic resets, logging and rendering.
Only actor observations are computed; no critic observations or custom
reward-only contact sensing run. MuJoCo's native sensors remain part of physics
because the actor uses its IMU outputs. Actuator gains, physics settings and the
policy are unchanged. This is matched inference work, not a matched training loop
or a claim that the simulators execute identical algorithms.

The final mjlab `forward()` refreshes derived quantities after integration and
is retained, not removed. It runs dynamics/constraint calculations without
advancing time; it is not a fifth 5 ms integration step. See the
[MuJoCo Warp 3.11 implementation](https://github.com/google-deepmind/mujoco_warp/blob/v3.11.0/mujoco_warp/_src/forward.py#L1239).

The mjlab benchmark bypasses `env.step()` and command-manager `compute()`.
It increments the motion frame on the GPU directly, so neither reset detection
via `reset_buf.nonzero()` nor command timer/wrap checks enter the measured loop.
The requested window is validated before timing: no wraparound is allowed.
Initialization and explicit between-trial resets still use mjlab normally,
outside timing. Non-finite states are checked after timing on both backends.

Both backends use separate observation and four-substep physics CUDA graphs;
mjlab's physics graph also includes the final forward. Torch actor inference
runs outside the graphs, using the same `Policy` implementation. Torch, Warp
and timing events share one stream. mjlab's combined graphs are Torch-owned to
retain intermediate tensor storage and use the original simulation stream so
its cached-stream control writes remain inside capture. The protocol applies to
batch benchmarks, not the unmodified single-environment recorders or training.
The benchmark disables Warp precompiled headers to avoid an NVRTC compilation
failure encountered with this environment; it does not alter physics settings.

## Aligned inference results

New measurements use the protocol above for both optimized VBD (`--level-parallel`)
and mjlab, with 40 warmup frames, 400 measured frames and three repetitions.
The results below supersede the earlier 7.6x physics / 5.3x full-play comparison.

| Environments | VBD physics | MJWarp physics | Physics ratio | VBD e2e | MJWarp e2e | E2e ratio |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 66,568 | 214,466 | 3.22x | 65,930 | 206,512 | 3.13x |
| 4096 | 69,764 | 381,970 | 5.48x | 69,489 | 371,235 | 5.34x |
| 8192 | 68,866 | 429,063 | 6.23x | 68,644 | 418,395 | 6.10x |

Values are environment control-frame FPS; ratios are MJWarp / VBD. All 18
timing trials ended with finite states and no pelvis heights below 0.5 m.
This final-state check is not a per-frame tracking-success or training-stability
claim. [Recorded trial timings and metadata](results/aligned_benchmark.json)
retain each measurement without machine names or absolute input paths.

At 8192 environments, per-batch control-step timings are:

| Path | Physics | E2e | E2e minus physics | Difference / e2e |
|---|---:|---:|---:|---:|
| Earlier full mjlab play | 15.604 ms | 22.644 ms | 7.040 ms | 31.09% |
| Aligned minimal mjlab | 19.093 ms | 19.580 ms | 0.487 ms | 2.49% |
| Aligned minimal VBD | 118.955 ms | 119.340 ms | 0.385 ms | 0.32% |

MJWarp physics time increases because its scope now includes the final forward
and control preparation; this does not show a solver regression. E2e throughput
increases from 361,774 to 418,395 env-FPS (15.7%) after removing task-only work
and capturing the minimal loop. The reduced apparent overhead reflects both
reclassification and removed work, not a 7 ms solver speedup. Wall time minus
CUDA-event time is not a separately profiled CPU stage; it also includes actor,
observations, action transfer and scheduling effects. The comparison changes
several factors together and does not isolate the cost of any one manager.

## Earlier measurements: different play scopes

These are existing measurements from 2026-10-13, not new runs for this document.
The optimized row uses the final cleaned implementation with `--level-parallel`.
The unoptimized and mjlab rows retain their previously measured medians; all
use the same hardware and physical settings, but the legacy timing scopes below.

**Legacy scope:** mjlab physics counted only four `sim.step()` calls, excluding
the final `forward()`. mjlab play included reward/termination/metrics, command
managers and critic observations; VBD play did not. The nine legacy mjlab trials
had zero actual resets but retained reset checks. These numbers document the
earlier experiment, not an equal-work e2e comparison. Do not combine legacy
mjlab rows with the new timing protocol when computing a speed ratio.

| Environments | VBD before physics | VBD after physics | mjlab physics | VBD before play | VBD after play | mjlab play |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 35,459 | 66,578 | 262,296 | 35,276 | 65,939 | 127,448 |
| 4096 | 34,106 | 69,731 | 468,668 | 34,040 | 69,461 | 292,331 |
| 8192 | 32,729 | 68,698 | 524,984 | 32,680 | 68,480 | 361,774 |

All values are environment FPS. The optimization improves VBD throughput;
mjlab remains faster in both timing scopes. The nine optimized VBD trials
ended with finite states and no pelvis heights below 0.5 m. This final-state
check does not establish successful tracking at every earlier frame.

The solver optimizations include structured joint assembly, reduced launch and
scratch-buffer overhead, and a CUDA level schedule using four warps per
articulation with register-resident 6-by-6 solves. The level schedule is opt-in:
pass `--level-parallel` to the benchmark. Omitting it selects the default
schedule, which still contains shared optimizations and is not the unoptimized
baseline. Floating-point accumulation order can change.

The existing full-clip comparison found mean world-position error versus mjlab
of 21.748 cm before and 21.986 cm after optimization. Both first crossed the
`ee_body_pos` threshold at 5.84 s and continued without reset. The speedup is
not an accuracy fix or a claim of passing all tracking thresholds.

## Reproduce optimized VBD and mjlab

Prepare the external inputs and H2-enabled environment described in
[README.md](README.md). Run from this repository root. Use an isolated
benchmark environment, not an active training environment. The recorded Warp
version can be installed into that environment with:

```sh
H2_PYTHON=../mjlab/.venv/bin/python
uv pip install --python "$H2_PYTHON" --index https://pypi.nvidia.com/ \
  warp-lang==1.16.0.dev20260716
```

Keep MuJoCo and MuJoCo Warp at the versions above for the same measurement
configuration. Do not resynchronize another project's lockfile between runs.
Run all environment counts sequentially:

```sh
for h2_envs in 1024 4096 8192; do
  uv run --no-project --python "$H2_PYTHON" python -m h2_sim2sim.benchmark vbd \
    --num-envs "$h2_envs" --level-parallel \
    --checkpoint sim2sim_inputs/model_29999.pt \
    --motion sim2sim_inputs/h2_360dankou.npz \
    --warmup 40 --steps 400 --repeats 3 \
    --output "sim2sim_out/benchmark/vbd_after_${h2_envs}.json"

  uv run --no-project --python "$H2_PYTHON" python -m h2_sim2sim.benchmark mjlab \
    --num-envs "$h2_envs" \
    --checkpoint sim2sim_inputs/model_29999.pt \
    --motion sim2sim_inputs/h2_360dankou.npz \
    --warmup 40 --steps 400 --repeats 3 \
    --output "sim2sim_out/benchmark/mjlab_${h2_envs}.json"
done
```

The JSON includes the timing-protocol identifier, individual trial timings, medians, final-state checks,
software versions and input hashes. Output files must not already exist.
To time the optimized default schedule, omit `--level-parallel` and choose
different output names.

## Reproduce unoptimized VBD

Use the same benchmark harness with the unoptimized solver; dropping the level
flag alone is not sufficient. Prepare an isolated detached checkout of the
unoptimized implementation and copy only the current H2 harness and local
robot assets into it. Use the explicit revision below, not `HEAD^`, so later
analysis commits do not change the baseline. The solver files remain unmodified:

```sh
git worktree add --detach sim2sim_out/vbd_baseline 61f613dac293268e441246e551366ff01f761f35
git archive HEAD h2_sim2sim | tar -x -C sim2sim_out/vbd_baseline
cp -R h2_description sim2sim_out/vbd_baseline/h2_description

for h2_envs in 1024 4096 8192; do
  uv run --no-project --python "$H2_PYTHON" python -c \
    'import runpy, sys; sys.path.insert(0, "sim2sim_out/vbd_baseline"); runpy.run_module("h2_sim2sim.benchmark", run_name="__main__")' \
    vbd --num-envs "$h2_envs" \
    --checkpoint sim2sim_inputs/model_29999.pt \
    --motion sim2sim_inputs/h2_360dankou.npz \
    --warmup 40 --steps 400 --repeats 3 \
    --output "sim2sim_out/benchmark/vbd_before_${h2_envs}.json"
done
```

The Python search-path override selects both the harness and Newton from the
isolated checkout while input/output paths remain relative to the repository
root. This uses the same CUDA-graph timing path as the optimized measurements.

## Input identity and checks

The recorded measurements used these external inputs (SHA-256):

| Input | SHA-256 |
|---|---|
| `model_29999.pt` | `99534239b1b0eaeb8a99bd12db856aab831d236a89c61012724e6d2b76df18d2` |
| `h2_360dankou.npz` | `95370380434a609bb91e19617b6592051b3b8fea05ed947fb95446f6f6d2d52a` |
| `H2loop_complete.xml` | `5803ce64b4207867195128d5c94edb21c6265096c64475d7fc34a656e669e10e` |

Validate the aligned benchmark before comparing throughput:

```sh
uv run --no-project --python "$H2_PYTHON" python -m unittest h2_sim2sim.test_benchmark.TestBenchmark

H2_CHECKPOINT=sim2sim_inputs/model_29999.pt \
H2_MOTION=sim2sim_inputs/h2_360dankou.npz \
uv run --no-project --python "$H2_PYTHON" python -m unittest h2_sim2sim.test_benchmark
```

The four logic tests check the final forward and frame advance, actor-only
observations, event boundaries and rejection of reference wraparound. The CUDA
test runs 476 frames at eight environments: three full-play rollouts establish
repeatability, followed by two graph-captured minimal rollouts. It checks actor
observations/actions against eager evaluation of the same state, finite states,
final standing height, reference-frame advance and absence of task-manager calls.

Recorded actor-input and action differences were exactly zero. Independent
closed-loop trajectories are not bitwise repeatable: full/full body-position
3D RMS differences were 11.48–14.79 mm (maximum 51.75 mm); minimal/full differences
were 11.46–16.20 mm (maximum 51.51 mm). These are repeatability checks within
MJWarp, not MJWarp-to-VBD tracking errors or closed-loop constraint residuals.
The test bounds minimal/full drift by twice the observed full/full baseline
(with 2 mm RMS / 5 mm maximum floors) and an absolute 100 mm maximum guard;
it does not claim exact trajectory equivalence.
All 18 H2 tracking, closure and benchmark unittest cases also passed with the
external assets on CUDA. This is separate from the older 100-case solver suite.

Run solver and policy-interface checks with the external inputs enabled:

```sh
H2_CHECKPOINT=sim2sim_inputs/model_29999.pt \
H2_MOTION=sim2sim_inputs/h2_360dankou.npz \
uv run --no-project --python "$H2_PYTHON" python -c \
  'import warp as wp; wp.config.use_precompiled_headers = False; import unittest; unittest.main(module=None, argv=["unittest", "newton.tests.test_solver_vbd", "newton.tests.test_vbd_sparse_articulation", "h2_sim2sim.test_tracking"])'
```

The existing CUDA-enabled validation passed 100 unittest cases: 46 CUDA-related
cases and 54 CPU/logic cases. These include 256 random 6-by-6 matrix checks,
CPU/CUDA sparse-solve comparisons, and all 194 batched H2 observations against
the single-environment implementation. Counts depend on available devices;
these are not 100 full H2 rollouts or throughput trials.
