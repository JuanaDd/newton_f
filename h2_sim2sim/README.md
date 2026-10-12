# H2 MuJoCo Warp to Newton VBD

Run a trained Unitree H2 mimic policy in Newton VBD and compare it with the
same policy in mjlab using MuJoCo Warp. Each simulator computes observations
from its own state and closes the loop through the actor; this is not action
replay. The motion NPZ supplies the reference motion to track.

This package provides inference, trajectory recording and recorded-pose overlay
rendering. It does not train a policy or include robot assets or policy weights.

See [BENCHMARK.md](BENCHMARK.md) for mjlab and VBD throughput at 1024, 4096
and 8192 environments, optimization settings and benchmark reproduction commands.
The batch benchmark supports an opt-in CUDA level schedule; the single-environment
recorder below uses the default schedule and does not capture CUDA graphs.

## Configuration

| Setting | Value |
|---|---|
| Robot | `H2loop_complete.xml`, 37 joints, 31 actuated joints |
| Physics timestep | 5 ms |
| VBD iterations | 8 per physics step |
| Decimation / policy rate | 4 physics steps per action / 50 Hz |
| Closed loops | Six massive ball-ended rods reconstructed from XML tendon anchors |
| Articulated solver | Block-sparse joints, relaxation 0.65, coupled armature |
| Contact | Sticky matching and contact history |
| Joint constraint stiffness / damping | 5e6 / 125 |
| Actuation | Original PD gains and action scaling; passive damping 0.05 |
| Actor | 194 observations, ELU MLP 512/256/128, 31 actions |
| Observation normalization | Checkpoint statistics, epsilon 0.01 |

Hinge position limits are released by the runtime as required by the policy.
VBD does not enforce actuator effort limits; no torque projection or PD damping
scaling is applied. The XML is not modified on disk.

## External inputs and environment

Place the externally supplied files at these repository-relative locations:

```text
h2_description/
  H2loop_complete.xml
  meshes/                 # Mesh files referenced by the XML
sim2sim_inputs/
  model_29999.pt
  h2_360dankou.npz
```

These inputs and the generated `sim2sim_out/` directory are ignored by Git.
The motion must match the H2 joint/body ordering used in `tracking.py`.

Use a CUDA-capable Linux machine with an isolated environment containing
Newton's dependencies, CUDA-enabled Torch, NumPy, MuJoCo, Pillow and
imageio-ffmpeg. Recording the source simulation additionally requires an
H2-enabled mjlab installation with `mjlab.tasks.tracking.config.h2` (environment,
runner and RL configuration); a generic mjlab installation is not sufficient.

Run all commands from this repository root. Set `H2_PYTHON` to the environment's
Python executable; for an H2-enabled mjlab checkout beside this repository:

```sh
H2_PYTHON=../mjlab/.venv/bin/python
```

Use `uv run --no-project` to preserve that environment rather than resynchronizing
a different project's lockfile. Newton is imported from the current repository.

## Record and render

Record the mjlab / MuJoCo Warp reference and the Newton VBD rollout:

```sh
uv run --no-project --python "$H2_PYTHON" python -m h2_sim2sim.record_mjlab \
  --checkpoint sim2sim_inputs/model_29999.pt \
  --motion sim2sim_inputs/h2_360dankou.npz \
  --steps 476 --output sim2sim_out/mjlab.npz

uv run --no-project --python "$H2_PYTHON" python -m h2_sim2sim.tracking \
  --checkpoint sim2sim_inputs/model_29999.pt \
  --motion sim2sim_inputs/h2_360dankou.npz \
  --xml h2_description/H2loop_complete.xml \
  --steps 476 --output sim2sim_out/vbd.npz --keep-running

MUJOCO_GL=egl uv run --no-project --python "$H2_PYTHON" \
  python -m h2_sim2sim.render \
  --mjlab sim2sim_out/mjlab.npz --newton sim2sim_out/vbd.npz \
  --xml h2_description/H2loop_complete.xml \
  --output sim2sim_out/h2_sim2sim.mp4
```

The recorders refuse to overwrite existing output files; choose new names for
another run. Each recorder also writes a JSON summary beside its NPZ file.
Without `--keep-running`, VBD recording stops at the first tracking-threshold
crossing. With it, crossings are recorded but do not stop or reset the rollout.
The mjlab recorder stops on termination and does not automatically reset.

The video overlays solid orange Newton VBD poses and a blue, 32%-opacity
mjlab reference. MuJoCo is used only to render the recorded meshes, not to
simulate the orange robot. No time/position fitting or forward-kinematics
projection is applied. The six added constraint rods have no original XML
meshes and are not rendered.

For a complete 476-frame rollout, the video covers 9.52 s at 50 fps. A completed
video is not evidence that every tracking threshold passed; inspect the summaries.

## Checks

```sh
uv run --no-project --python "$H2_PYTHON" python -m unittest h2_sim2sim.test_tracking
```

The interface tests check policy dimensions, original PD parameters and anchor
orientation encoding. They do not measure full-trajectory tracking accuracy.
The batched-observation test additionally requires CUDA and the `H2_CHECKPOINT`
and `H2_MOTION` environment variables; see the benchmark checks for that command.
