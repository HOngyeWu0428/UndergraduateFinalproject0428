from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from vrl_dissertation.agents.common.encoder import PixelEncoder
from vrl_dissertation.agents.sac.networks import Actor, Critic
from vrl_dissertation.augmentations.base import Augmentation, IdentityAugmentation


class SACAgent:
    def __init__(
        self,
        obs_shape: tuple[int, int, int],
        action_dim: int,
        feature_dim: int,
        hidden_dim: int,
        gamma: float,
        tau: float,
        lr: float,
        alpha_lr: float,
        init_temperature: float,
        log_std_min: float,
        log_std_max: float,
        device: torch.device,
        augmentation: Augmentation | None = None,
    ) -> None:
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.target_entropy = -float(action_dim)

        self.augmentation = augmentation if augmentation is not None else IdentityAugmentation()

        self.encoder = PixelEncoder(obs_shape, feature_dim).to(device)
        self.actor = Actor(feature_dim, action_dim, hidden_dim, log_std_min, log_std_max).to(device)
        self.critic = Critic(feature_dim, action_dim, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)

        self.log_alpha = torch.tensor(np.log(init_temperature), device=device, requires_grad=True)

        self.encoder_opt = optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=lr)
        self.alpha_opt = optim.Adam([self.log_alpha], lr=alpha_lr)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def act(self, obs: np.ndarray, sample: bool) -> np.ndarray:
        obs_t = torch.as_tensor(obs, device=self.device).float().unsqueeze(0) / 255.0
        z = self.encoder(obs_t)
        if sample:
            action, _ = self.actor.sample(z)
        else:
            mean, _ = self.actor(z)
            action = torch.tanh(mean)
        return action.squeeze(0).cpu().numpy()

    def update(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        obs = batch["obs"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_obs = batch["next_obs"]
        dones = batch["dones"]

        obs = self.augmentation(obs)
        next_obs = self.augmentation(next_obs)

        z = self.encoder(obs)
        with torch.no_grad():
            next_z = self.encoder(next_obs)
            next_action, next_log_prob = self.actor.sample(next_z)
            target_q1, target_q2 = self.critic_target(next_z, next_action)
            target_v = torch.min(target_q1, target_q2) - self.alpha.detach() * next_log_prob
            target_q = rewards + (1.0 - dones) * self.gamma * target_v

        q1, q2 = self.critic(z, actions)
        critic_loss = F.mse_loss(q1, target_q) + F.mse_loss(q2, target_q)

        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.encoder_opt.step()
        self.critic_opt.step()

        z_actor = self.encoder(obs).detach()
        new_action, log_prob = self.actor.sample(z_actor)
        q1_pi, q2_pi = self.critic(z_actor, new_action)
        q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha.detach() * log_prob - q_pi).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        with torch.no_grad():
            for p, p_t in zip(self.critic.parameters(), self.critic_target.parameters()):
                p_t.data.mul_(1 - self.tau)
                p_t.data.add_(self.tau * p.data)

        return {
            "critic_loss": float(critic_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
            "q": float(q_pi.mean().item()),
        }

    def state_dict(self) -> dict:
        return {
            "encoder": self.encoder.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "encoder_opt": self.encoder_opt.state_dict(),
            "actor_opt": self.actor_opt.state_dict(),
            "critic_opt": self.critic_opt.state_dict(),
            "alpha_opt": self.alpha_opt.state_dict(),
        }
