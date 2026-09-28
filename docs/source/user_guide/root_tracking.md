# Privileged root-trajectory tracking

`sonic_v1_1_root_tracking` trains a small root-trajectory branch on top of the
frozen SONIC v1.1 actor. It targets offline G1 motion tracking where the current
robot root pose and the reference trajectory share a world coordinate system.
PPO acts in the 64-dimensional latent-residual space. The root branch, its
exploration standard deviation and the critic are trainable; the original
SONIC encoder, FSQ and decoders remain frozen.
`sonic_v1_1_3pt` uses the same residual design and adds the two wrist-link origins
to the root trajectory input and a fine wrist-position reward.

## Observation and model

The pelvis reference uses the same ten time indices as the original G1 motion
encoder: current time through 0.9 seconds ahead at 0.1-second spacing. Indices
past a motion's end repeat its final frame.

For each reference frame, `motion_root_trajectory_heading` contains:

1. `R_heading_actual.T @ (p_reference_world - p_actual_world)`, in metres.
2. The first two columns of `R_heading_actual.T @ R_reference_world`, flattened
   in row-major order: `[R00, R01, R10, R11, R20, R21]`.

Heading uses the same quaternion heading extraction as SONIC. Current measured
pelvis state supplies the origin and heading. World origins are included in the
reference positions, so global translation errors remain observable. The
reference trajectory stays fixed as the robot drifts.

The root observation shape is `[batch, 10, 9]`, flattened to 90 values. Kimodo's 50 Hz
MuJoCo `qpos` provides XYZ in columns `0:3` and a `wxyz` quaternion in `3:7`;
at 50 Hz these references use offsets `[0,5,10,...,45]`.

The original G1 encoder produces a pre-quantization latent of shape `[batch,2,32]`.
The branch concatenates its flattened 64 values, the 90 root values, and
93 current-state values from the original actor observations, giving 247 input
features. This latent comes from the original encoder before adding the branch
contribution or applying FSQ.

The current-state order is gravity direction (3), base angular velocity (3),
joint position (29), joint velocity (29), and previous body action (29). Each
value is taken from the newest frame of its own actor-observation history,
preserving the original coordinate conventions, noise and scaling. The 930-value
actor history is term-major in the environment's order: angular velocity, joint
position, joint velocity, previous action, gravity direction. Frames within each
term are oldest-to-newest. The current state is not its final 93 values.
No privileged linear velocity or
additional simulator state is used. No input normalization is added.

```text
concat(original_latent[64], root_observation[90], actor_current_state[93])
247 -> Linear(512) -> SiLU -> Linear(256) -> SiLU
   -> Linear(128) -> SiLU -> Linear(64)

z = original_g1_encoder(motion_observations)
mu_residual = root_branch(concat(flatten(z), flatten(root_observation), actor_current_state))
u ~ Normal(mu_residual, diag(sigma_residual**2))  # PPO action during training
token = FSQ(z + 0.1 * reshape(u, [2,32]))
action = original_decoder(token, proprioception)
```

The final Linear weight and bias start at zero, giving an initially zero
correction mean. The trainable latent exploration standard deviation starts at
0.05, so sampled training actions can still correct the latent initially.
Evaluation and ONNX export use the residual mean. PPO stores the sampled raw
64-dimensional residuals and computes log-probabilities in that space. The
frozen SONIC executes in the environment wrapper without gradients; the PPO
update does not differentiate through FSQ or the body decoder.

Root training defaults to `root_critic=xy_preview`, adding a 40-dimensional XY
preview to the original 1645 privileged observations (1685 total). Use
`root_critic=base` to select the original critic observations. This selection
retains the original hidden layers and scalar
value output. The ten frames use the actor's reference times and endpoint
padding. Each frame contains `[dx_ref, dy_ref, ex, ey]`, in meters:

- Reference displacement: future reference root minus the current reference root.
- Tracking error: future reference root minus the current measured pelvis.

Both vectors are rotated into the current measured heading frame. The first
frame's reference displacement is zero; its error is the current XY tracking
error. Frames are flattened in temporal order. The first-layer weights for
these 40 inputs start at zero, preserving the pretrained value prediction at
initialization. All critic weights remain trainable under the existing PPO
value loss. Original observation normalization statistics are preserved; the
preview has independent running statistics and sample count, synchronized
across training processes. Actor observations and deployment interfaces are
unchanged by this critic extension.

The auxiliary loss is `1.0 * mean(mu_residual**2)` on the unscaled residual
mean, averaged across samples and latent coordinates. Two reward terms smooth the
residual stream, `root_meta_action_rate` (`-0.1 * sum((u_t - u_previous)**2)` on
the sampled raw residual) and `root_full_token_rate`
(`-0.01 * sum((token_t - token_previous)**2)` on the actual post-FSQ token). Both
carry `weight: 0.0` in the training config, so the reward manager skips them and
the `Env/Episode_Reward/root_*_rate` entries stay at zero. Both exclude the first
step after an environment reset, and their state is cleared at reset. They use
the same reward-manager time scaling as other reward terms. No G1 reconstruction
or cross-modal auxiliary loss is used.

Original rewards are retained. The additional horizontal reward is
`0.5 * exp(-squared_xy_error / 0.1**2)`, comparing current measured and current
reference pelvis positions. Future poses provide preview information.

## v1_1_3pt wrist input and reward

The 3-point policy follows the pelvis and the origins of
`left_wrist_yaw_link` and `right_wrist_yaw_link`. For each of the same ten
reference times (0 through 0.9 seconds in 0.1-second increments), the wrist
observation contains 12 values in this order:

```text
[left_reference_xyz, right_reference_xyz, left_error_xyz, right_error_xyz]
```

Each reference wrist position is relative to that frame's reference pelvis,
rotated by the inverse of that pelvis's full orientation. The measured current
wrist positions are relative to the measured current pelvis, rotated by the
inverse of its full orientation. Each error is future reference local position
minus current measured local position. Positions are in metres. The ten frames
form a `[10,12]` observation (120 values); references beyond the end of a clip
repeat its last frame. The root branch therefore receives 64 original latent
values, 90 root values, 120 wrist values and 93 current-state values: 367 total.
The original G1 encoder and action decoder remain frozen.

The additional reward is the mean of the two per-wrist terms
`exp(-squared_local_position_error / 0.01**2)`, with weight 1.0. It evaluates
the current frame and uses wrist-link origins without offsets. The existing
5-point reward continues to use its configured offset points and 0.1-metre
scale. The optional XY-preview critic remains selected by default.

## Training

Use the [SONIC training environment](../getting_started/installation_training.md).
For an extracted Isaac Sim installation with a Conda environment, source the
installation's `setup_conda_env.sh` before launching Python so its bundled
PyTorch and simulator libraries are available.

Download the matching training checkpoint:

```bash
python download_from_hf.py --training --sonic-v1-1 --no-smpl
```

The experiment defaults to `sonic_v1_1/last.pt` and
`data/motion_lib_bones_seed/robot_filtered`. G1 is the only sampled modality.
The `dummy` SMPL setting supplies placeholder observations for the frozen SMPL
encoder; SMPL clips are not sampled or trained. The only auxiliary loss is the residual-mean L2 penalty.

```bash
./run_sonic_root_tracking.sh
```

For the 3-point configuration, use `./run_sonic_v1_1_3pt.sh` or launch
`gear_sonic/train_agent_trl.py` after sourcing the same Isaac Sim setup used
above, with `+exp=manager/universal_token/all_modes/sonic_v1_1_3pt`. It
initializes from `sonic_v1_1/last.pt` and defaults to
`outputs/sonic_v1_1_3pt`. Start new runs in a separate output directory.
Resume requires a 367-input 3-point checkpoint; root-only checkpoints have a
different first-layer input size.

For a 16-environment, 20-iteration validation run:

```bash
python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_v1_1_3pt \
  num_envs=16 headless=true use_wandb=false \
  experiment_dir=outputs/sonic_v1_1_3pt_smoke \
  ++algo.config.num_learning_iterations=20 \
  ++callbacks.model_save.save_frequency=10 \
  ++callbacks.model_save.save_last_frequency=10 \
  ++manager_env.commands.motion.motion_lib_cfg.max_unique_motions=16
```

The local launcher uses the project's Isaac Sim 5.1 setup, SONIC Conda Python,
the v1.1 checkpoint and the prepared `robot_filtered` motions. Its settings can be edited near the
top of the script or set through environment variables:

| Variable | Default | Meaning |
| --- | --- | --- |
| `SONIC_NUM_ENVS` | `4096` | Parallel simulation environments |
| `SONIC_CKPT_SAVE_EVERY` | `2000` | PPO iterations between numbered `model_step_*.pt` saves |
| `SONIC_MAX_CKPTS` | `5` | Number of newest numbered checkpoints to retain |
| `SONIC_LAST_SAVE_EVERY` | `50` | PPO iterations between updates to resumable `last.pt` |
| `SONIC_USE_WANDB` | `true` | Enable or disable W&B logging |
| `SONIC_WANDB_RUN_NAME` | `sonic_root_tracking_fused_orilatent` | W&B run name, independent of the save directory |
| `SONIC_OUTPUT_DIR` | `<repository>/outputs/sonic_root_tracking` | Run directory for checkpoints and logs |

The retention limit applies to numbered checkpoints; `last.pt` is maintained
separately. For example:

```bash
SONIC_NUM_ENVS=128 SONIC_CKPT_SAVE_EVERY=500 SONIC_MAX_CKPTS=3 \
  SONIC_LAST_SAVE_EVERY=25 SONIC_USE_WANDB=true \
  SONIC_OUTPUT_DIR=logs_rl/root_tracking_run_01 \
  ./run_sonic_root_tracking.sh
```

Additional Hydra overrides can be passed as script arguments.
`SONIC_ISAAC_SIM_ROOT`, `SONIC_PYTHON`, `SONIC_CHECKPOINT` and
`SONIC_MOTION_FILE` override the local installation paths.

Start an independent latent-action run in its own directory:

```bash
SONIC_OUTPUT_DIR="$PWD/outputs/root_latent_$(date +%Y%m%d_%H%M%S)" \
  SONIC_WANDB_RUN_NAME=sonic_root_latent_action ./run_sonic_root_tracking.sh
```

For a 16-environment, 20-iteration smoke run:

```bash
python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_v1_1_root_tracking \
  num_envs=16 headless=true use_wandb=false \
  experiment_dir=logs_rl/root_latent_action_smoke \
  ++algo.config.num_learning_iterations=20 \
  ++callbacks.model_save.save_frequency=10 \
  ++callbacks.model_save.save_last_frequency=10 \
  ++manager_env.commands.motion.motion_lib_cfg.max_unique_motions=16
```

Initial adaptation loads the original encoder, FSQ and decoder weights
strictly and, when `root_critic=xy_preview`, expands the pretrained critic's
first layer with 40 zero columns. It
initializes the root branch and a fresh 64-dimensional action
standard deviation, and creates a new optimizer. The original 29-dimensional
standard deviation is retained only as a diagnostic normalization buffer.
To resume, set `+resume=true` and `checkpoint=<latent-policy-checkpoint.pt>`;
the branch, exploration distribution, critic and optimizer are restored.
Set the same `root_critic` option used to train the checkpoint when resuming;
resume loads the critic and normalization states strictly, without automatic
architecture selection. To initialize the preview critic from an older
1645-input critic checkpoint, use `root_critic=xy_preview`, leave resume disabled,
and start a new run; the critic is expanded and optimizers are recreated.

The first branch weight has shape `[512,247]`, its output has 64 coordinates,
and its final layer is Linear. Resume checkpoints must match this architecture
and PPO action space. Begin a new run directory when starting from v1.1.

## Policy deviation logs

After each PPO iteration, the root-tracking experiment compares the updated
policy with the frozen v1.1 path on up to 256 evenly selected rollout states per
GPU. The baseline bypasses the root branch and shares the frozen original
encoder, FSQ and decoder. Both paths receive the same
observations. Diagnostics run in evaluation mode without gradients and add no
training loss. Set `algo.config.base_policy_diagnostic_samples=0` to disable them.

The regular training log, including W&B when enabled, records these metrics
under `base_policy/`, with `_mean` and `_p95` suffixes across sampled G1 states:

The training objectives are logged as `loss/aux_latent_l2_avg` (unweighted),
`Episode_Reward/root_meta_action_rate` and
`Episode_Reward/root_full_token_rate` (weighted episode reward statistics).

| Metric | Definition |
| --- | --- |
| `action_delta_rms` | RMS difference of deterministic decoded body actions (using the latent mean), before environment scaling/clipping. |
| `action_delta_standardized_rms` | Body-action difference normalized by the saved original body-policy standard deviation. This is not a distribution KL. |
| `residual_rms` | RMS pre-FSQ latent difference across all 64 coordinates. |
| `token_change_fraction` | Fraction of the 64 post-FSQ scalar values differing from the base, in `[0,1]`. |

`base_policy/sample_count` reports the combined diagnostic sample count across
GPUs. P95 is computed over the gathered sample values, not averaged across GPUs.
These are comparisons on the current rollout distribution after the PPO update;
they are distinct from PPO's previous-policy KL and from fixed-trajectory A/B
evaluation. Larger deviations can be useful corrections; assess them alongside
root tracking error and motion quality. The latent Gaussian passed through FSQ
and the decoder is not a Gaussian body-action distribution, so diagnostics do
not report an exact KL to the original body policy. Logging adds two decoder evaluations on
the sampled states per iteration.

## Paired evaluation

Use a fixed held-out motion-lib subset containing straight walking, turning and
stopping clips. Run both checkpoints with the same seed and environment count:

```bash
python gear_sonic/scripts/compare_root_tracking.py \
  --baseline sonic_v1_1/last.pt \
  --adapted logs_rl/root_latent_action_smoke/model_step_000020.pt \
  --motion-file /path/to/held_out_robot_motions \
  --output logs_rl/root_tracking_comparison \
  --num-envs 16 --seed 0 --max-motions 512
```

The script runs nominal and disturbed scenarios. The disturbed scenario adds
world-frame velocity `[0,0.3,0]` m/s at control step 50 (one second); use
`--push-step` to change its timing. Each paired run starts motions from their
first frame with the same evaluation reset procedure.

`comparison.json` summarizes frame-weighted XY, yaw and joint RMSE, completed
motion endpoint XY error, observed fall rate, and early termination rate.
It also reports left and right pelvis-local wrist-position RMSE, the fraction
of active frames within 1 cm for each wrist, and the fraction where both wrists
are within 1 cm. These wrist metrics use the link origins and full pelvis
orientation used by the 3-point reward.
Per-motion values and logs live in each scenario/controller directory. Failed
motions have a last-valid XY error and a null completed endpoint. Metrics sample
the measured state before each control advance, excluding states after reset.
An observed fall means pelvis height below 0.25 m above the environment origin
or pelvis tilt over 60 degrees at a sampled state. Early termination is reported
separately because tracking failures may terminate before a physical fall.

Twenty PPO iterations validate the training and export pipeline. Assess tracking
quality on held-out motions after a longer training run; frozen decoder weights
limit the branch's available corrective actions.

## ONNX export

```bash
python gear_sonic/eval_agent_trl.py \
  checkpoint=logs_rl/root_latent_action_smoke/model_step_000020.pt \
  +headless=true ++num_envs=1 +export_onnx_only=true
```

Each exported graph has a companion `.input.json` containing ordered input
fields, shapes and flattened offsets. Root-aware graphs also record the root
coordinate convention, time offsets and endpoint padding. The all-encoder
graph has 1,934 inputs and 64 outputs; the G1 encoder/decoder graph has 1,753
inputs and 29 outputs. The decoder-only graph retains its 994-to-29 interface.
The exported encoder uses the residual mean and includes the 0.1 scaling and
FSQ. Companion metadata records that execution convention.
The 64 latent conditioning values are computed inside the encoder graph; callers
supply the 90 root observation values and 93 current actor-state values together
with the original observations. The current-state values must match the newest
frames of the proprioception history supplied to the decoder. Each input manifest
records the exact actor-history indices used to extract them.
For `sonic_v1_1_3pt`, callers also supply the 120 wrist values in the order and
coordinate convention above; its `.input.json` records the full contract. Its
all-encoder input is 2,054 values, and its G1 encoder/decoder input is 1,873
values.

Deployment callers must supply the new root observation. Integrating this input
into humanoid-locomanip's native replay controller is a separate task.
