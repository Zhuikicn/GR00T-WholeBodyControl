"""PPO in root-correction latent space with a frozen SONIC action transform."""

import copy

import torch
from torch import nn

from gear_sonic.trl.modules.universal_token_modules import UniversalTokenModule


class RootLatentPolicy(UniversalTokenModule):
    def __init__(
        self, residual_scale=0.1, residual_l2_coef=1.0,
        actor_state_history_lengths=None, **kwargs
    ):
        if actor_state_history_lengths is not None:
            # Derived from actor_obs; it is not an additional environment sensor.
            kwargs["env_config"] = copy.deepcopy(kwargs["env_config"])
            kwargs["env_config"].obs.group_obs_dims.tokenizer["actor_current_state"] = [93]
        super().__init__(**kwargs)
        self.actor_state_history_lengths = actor_state_history_lengths
        if actor_state_history_lengths is not None:
            term_indices = {}
            offset = 0
            # PolicyCfg declaration order determines the flattened actor history.
            for name, dim in (
                ("base_ang_vel", 3), ("joint_pos", 29), ("joint_vel", 29),
                ("actions", 29), ("gravity_dir", 3),
            ):
                offset += dim * actor_state_history_lengths[name]
                term_indices[name] = list(range(offset - dim, offset))
            indices = [
                index
                for name in ("gravity_dir", "base_ang_vel", "joint_pos", "joint_vel", "actions")
                for index in term_indices[name]
            ]
            self.register_buffer("actor_state_indices", torch.tensor(indices), persistent=False)
        self.residual_scale = residual_scale
        self.residual_l2_coef = residual_l2_coef
        self.register_buffer("base_action_std", torch.ones(29))
        # Zero correction mean at initialization; latent exploration is separate.
        nn.init.zeros_(self.encoders["root"].module[-1].weight)
        nn.init.zeros_(self.encoders["root"].module[-1].bias)

    def parse_tokenizer_obs(self, input_data):
        obs = super().parse_tokenizer_obs(input_data)
        if self.actor_state_history_lengths is not None:
            # Each term's history is oldest-to-newest; reuse the exact noisy actor values.
            obs["actor_current_state"] = input_data["actor_obs"].index_select(
                -1, self.actor_state_indices
            )
        return obs

    def residual_mean(self, obs):
        with torch.no_grad():
            base = self._encode_single("g1", obs)
        return self._encode_single("root", obs, base_latent=base)

    def encode(self, encoder_name, tokenizer_obs, encoder_mask=None,
               no_quantization=False, frame_mask=None):
        # Deterministic execution/export uses the mean residual, scaled once.
        base = self._encode_single(encoder_name, tokenizer_obs, encoder_mask, frame_mask)
        latent = base
        if encoder_name == "g1":
            latent = base + self.residual_scale * self._encode_single(
                "root", tokenizer_obs, encoder_mask, frame_mask, base_latent=base
            )
        if no_quantization:
            return latent
        return self.quantizer(latent)[0], latent

    @torch.no_grad()
    def transform_actions(self, input_data, residual):
        """Convert sampled raw residuals to physical actions outside PPO's graph."""
        obs = self.parse_tokenizer_obs(input_data)
        base = self._encode_single("g1", obs)
        latent = base + self.residual_scale * residual.reshape_as(base)
        tokens = self.quantizer(latent)[0]
        batch, seq = input_data["actor_obs"].shape[:2]
        decoded = self.decode("g1_dyn", {
            **obs,
            "token": tokens.reshape(batch, seq, self.max_num_tokens, self.token_dim),
            "token_flattened": tokens.reshape(batch, seq, self.token_total_dim),
            "proprioception": torch.cat(
                [input_data[key] for key in self.proprioception_features], dim=-1
            ),
        })
        return decoded["action"], tokens.reshape(batch, seq, self.token_total_dim)

    @torch.no_grad()
    def base_policy_diagnostics(self, input_data, std):
        obs = self.parse_tokenizer_obs(input_data)
        residual = self.residual_mean(obs)
        action, tokens = self.transform_actions(input_data, residual)
        base_action, base_tokens = self.transform_actions(input_data, torch.zeros_like(residual))
        delta = (action.float() - base_action.float()).flatten(0, 1)
        return {
            "action_delta_rms": delta.square().mean(-1).sqrt(),
            "action_delta_standardized_rms": (delta / self.base_action_std).square().mean(-1).sqrt(),
            "residual_rms": (self.residual_scale * residual.float()).flatten(1).square().mean(-1).sqrt(),
            "token_change_fraction": (tokens != base_tokens).float().mean(-1).flatten(),
        }

    def forward(self, input_data, compute_aux_loss=False, return_dict=False,
                base_policy_diagnostics_std=None, **kwargs):
        if base_policy_diagnostics_std is not None:
            return self.base_policy_diagnostics(input_data, base_policy_diagnostics_std)
        obs = self.parse_tokenizer_obs(input_data)
        batch, seq = input_data["actor_obs"].shape[:2]
        mean = self.residual_mean(obs).reshape(batch, seq, self.token_total_dim)
        if compute_aux_loss or return_dict:
            return {
                "action_mean": mean,
                "aux_losses": {"latent_l2": mean.float().square().mean()},
                "aux_loss_coef": {"latent_l2": self.residual_l2_coef},
            }
        return mean
