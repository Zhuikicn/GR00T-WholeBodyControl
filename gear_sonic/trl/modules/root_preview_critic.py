"""Extend a pretrained SONIC critic with independently normalized root preview."""

import torch

from gear_sonic.trl.modules.actor_critic_modules import Critic
from gear_sonic.utils.running_mean_std import RunningMeanStd


class RootPreviewCritic(Critic):
    def __init__(self, env_config, algo_config, backbone, preview_dim=40, **kwargs):
        super().__init__(env_config, algo_config, backbone, **kwargs)
        self.preview_dim = preview_dim
        self.running_mean_std = RunningMeanStd(
            (env_config.robot.algo_obs_dim_dict.critic_obs - preview_dim,), per_channel=True
        )
        self.preview_running_mean_std = RunningMeanStd((preview_dim,), per_channel=True)
        with torch.no_grad():
            self.critic_module.module[0].weight[:, -preview_dim:].zero_()

    def load_pretrained_state_dict(self, state):
        """Expand only the original first layer; matching checkpoints load strictly."""
        state = dict(state)
        key = "critic_module.module.0.weight"
        target = self.critic_module.module[0].weight
        if state[key].shape[1] == target.shape[1] - self.preview_dim:
            state[key] = torch.cat(
                (state[key], state[key].new_zeros(target.shape[0], self.preview_dim)), dim=1
            )
            state.update({
                f"preview_running_mean_std.{key}": value
                for key, value in self.preview_running_mean_std.state_dict().items()
            })
        return self.load_state_dict(state, strict=True)

    def evaluate(self, obs_dict, **kwargs):
        obs_dict = obs_dict.copy()
        obs = obs_dict["critic_obs"]
        with torch.no_grad():
            obs_dict["critic_obs"] = torch.cat((
                self.running_mean_std(obs[..., :-self.preview_dim]),
                self.preview_running_mean_std(obs[..., -self.preview_dim:]),
            ), dim=-1)
        return self.critic(obs_dict, **kwargs)
