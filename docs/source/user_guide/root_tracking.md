# Privileged root-trajectory tracking

`sonic_v1_1_root_tracking` trains a small root-trajectory branch on top of the
frozen SONIC v1.1 actor. It targets offline G1 motion tracking where the current
robot root pose and the reference trajectory share a world coordinate system.
The training algorithm is SONIC PPO with G1 reconstruction loss. Only the root
branch and critic are trainable.

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
The branch concatenates its flattened 64 values followed by the 90 root values,
giving 154 input features. This latent comes from the original encoder before
adding the branch contribution or applying FSQ.

```text
concat(original_latent[64], root_observation[90])
154 -> Linear(512) -> SiLU -> Linear(256) -> SiLU
   -> Linear(128) -> SiLU -> Linear(64) -> LayerNorm(64)
   -> reshape [2,32]

z = original_g1_encoder(motion_observations)
token = FSQ(z + root_branch(concat(flatten(z), flatten(root_observation))))
action = original_decoder(token, proprioception)
```

Both LayerNorm affine parameters start at zero. The initial branch contribution
is exactly zero; the normalization parameters learn first, then gradients reach
the preceding linear layers. Original actor parameters, including exploration
standard deviation, remain fixed. Gradients through the frozen decoder and FSQ
are retained. Teleop and SMPL retain their original weights and behavior.

Original rewards are retained. The additional horizontal reward is
`0.5 * exp(-squared_xy_error / 0.1**2)`, comparing current measured and current
reference pelvis positions. Future poses provide preview information.

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
encoder; SMPL clips are not sampled or trained. Cross-modality auxiliary losses
are disabled, and G1 reconstruction retains coefficient `0.01`.

```bash
python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_v1_1_root_tracking \
  num_envs=4096 headless=true use_wandb=false
```

For a 16-environment, 20-iteration smoke run:

```bash
python gear_sonic/train_agent_trl.py \
  +exp=manager/universal_token/all_modes/sonic_v1_1_root_tracking \
  num_envs=16 headless=true use_wandb=false \
  experiment_dir=logs_rl/root_tracking_conditioned_smoke \
  ++algo.config.num_learning_iterations=20 \
  ++callbacks.model_save.save_frequency=10 \
  ++callbacks.model_save.save_last_frequency=10 \
  ++manager_env.commands.motion.motion_lib_cfg.max_unique_motions=16
```

Initial adaptation loads original actor and critic weights strictly and creates
a fresh optimizer. Only a wholly absent `actor_module.encoders.root` module may
be initialized locally. A partial branch, missing original parameter, or shape
mismatch fails loading. To resume adaptation, set `+resume=true` and
`checkpoint=<adapted-checkpoint.pt>`; the complete branch and optimizer state
are restored.

The branch checkpoint's first linear weight has shape `[512,154]`. Adaptation
starts from the original v1.1 checkpoint; resuming requires a checkpoint with
this matching branch architecture.

## Paired evaluation

Use a fixed held-out motion-lib subset containing straight walking, turning and
stopping clips. Run both checkpoints with the same seed and environment count:

```bash
python gear_sonic/scripts/compare_root_tracking.py \
  --baseline sonic_v1_1/last.pt \
  --adapted logs_rl/root_tracking_conditioned_smoke/model_step_000020.pt \
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
  checkpoint=logs_rl/root_tracking_conditioned_smoke/model_step_000020.pt \
  +headless=true ++num_envs=1 +export_onnx_only=true
```

Each exported graph has a companion `.input.json` containing ordered input
fields, shapes and flattened offsets. Root-aware graphs also record the root
coordinate convention, time offsets and endpoint padding. The all-encoder
graph has 1,841 inputs and 64 outputs; the G1 encoder/decoder graph has 1,660
inputs and 29 outputs. The decoder-only graph retains its 994-to-29 interface.
The 64 latent conditioning values are computed inside the encoder graph; callers
supply the 90 root observation values together with the original observations.

Deployment callers must supply the new root observation. Integrating this input
into humanoid-locomanip's native replay controller is a separate task.
