# H2 sim2sim throughput

Compare closed-loop H2 policy play in mjlab / MuJoCo Warp with Newton VBD,
before and after the VBD solver optimizations. See [README.md](README.md) for
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
| VBD physics | CUDA events around action-to-target conversion, force clearing, collision and four VBD steps |
| mjlab physics | CUDA events around four MuJoCo Warp steps, including collision and native integrator/sensors |
| VBD play | Synchronized wall time for GPU IK/observations, Torch actor and physics |
| mjlab play | Synchronized wall time for actor, physics, observations, rewards, termination/reset and other task managers |

Play timings have different task-manager overheads; neither includes rendering.
VBD continues without tracking-triggered resets. mjlab retains termination/reset
checks; the nine reported mjlab trials recorded zero termination/reset events.

VBD uses separate CUDA graphs for observations and the four-substep physics
update. Torch actor inference runs outside the graphs. This applies to both
benchmark schedules, not to the single-environment `tracking.py` recorder.
The benchmark disables Warp precompiled headers to avoid an NVRTC compilation
failure encountered with this environment; it does not alter physics settings.

## Results

These are existing measurements from 2026-10-13, not new runs for this document.
The optimized row uses the final cleaned implementation with `--level-parallel`.
The unoptimized and mjlab rows retain their previously measured medians; all
use the hardware and protocol above, but were measured in separate runs.

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

The JSON includes individual trial timings, medians, final-state checks,
software versions and input hashes. Output files must not already exist.
To time the optimized default schedule, omit `--level-parallel` and choose
different output names.

## Reproduce unoptimized VBD

Use the same benchmark harness with the unoptimized solver; dropping the level
flag alone is not sufficient. From the optimization commit, prepare an isolated
detached checkout of its parent and copy only the current H2 harness and local
robot assets into it. The solver files remain unmodified:

```sh
git worktree add --detach sim2sim_out/vbd_baseline HEAD^
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
