from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers import FrameStackObservation, GrayscaleObservation, ResizeObservation

from vrl_dissertation.agents.common.replay_buffer import ReplayBuffer
from vrl_dissertation.agents.sac.agent import SACAgent
from vrl_dissertation.augmentations.registry import build_augmentation
from vrl_dissertation.baseline_config import BaselineExperimentConfig
from vrl_dissertation.logging.checkpoint import save_checkpoint
from vrl_dissertation.logging.csv_logger import CSVLogger
from vrl_dissertation.utils.seed import set_seed


class SACTrainer:
    def __init__(self, cfg: BaselineExperimentConfig) -> None:
        self.cfg = cfg
        set_seed(cfg.train.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.run_dir = Path(cfg.output.run_dir) / cfg.experiment_name / f"seed_{cfg.train.seed}"
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.env = self._make_env(train=True)
        self.eval_env = self._make_env(train=False)

        obs, _ = self.env.reset(seed=cfg.train.seed)
        obs_shape = obs.shape
        action_dim = int(np.prod(self.env.action_space.shape))

        augmentation = build_augmentation(cfg.augmentation).to(self.device)

        self.agent = SACAgent(
            obs_shape=obs_shape,
            action_dim=action_dim,
            feature_dim=cfg.model.feature_dim,
            hidden_dim=cfg.model.hidden_dim,
            gamma=cfg.train.gamma,
            tau=cfg.train.tau,
            lr=cfg.train.lr,
            alpha_lr=cfg.train.alpha_lr,
            init_temperature=cfg.train.init_temperature,
            log_std_min=cfg.model.log_std_min,
            log_std_max=cfg.model.log_std_max,
            device=self.device,
            augmentation=augmentation,
        )

        self.replay = ReplayBuffer(obs_shape, action_dim, cfg.train.replay_capacity, self.device)
        self.csv = CSVLogger(self.run_dir / "train_metrics.csv")
        self.tb = None
        if cfg.output.use_tensorboard and importlib.util.find_spec("torch.utils.tensorboard") is not None:
            from torch.utils.tensorboard import SummaryWriter

            self.tb = SummaryWriter(log_dir=str(self.run_dir / "tb"))

    def _make_env(self, train: bool) -> gym.Env:
        env = gym.make(self.cfg.env.env_id)
        if train:
            env.reset(seed=self.cfg.train.seed)
        else:
            env.reset(seed=self.cfg.train.seed + 1000)
        env.action_space.seed(self.cfg.train.seed + (0 if train else 1000))

        obs_shape = env.observation_space.shape
        if len(obs_shape) == 3:
            if obs_shape[-1] == 3:
                env = GrayscaleObservation(env, keep_dim=True)
            env = ResizeObservation(env, (self.cfg.env.image_size, self.cfg.env.image_size))
            if self.cfg.env.frame_stack > 1:
                env = FrameStackObservation(env, stack_size=self.cfg.env.frame_stack)
        return env

    @staticmethod
    def _to_chw(obs: np.ndarray) -> np.ndarray:
        if obs.ndim == 4:  # (stack, H, W, C)
            obs = np.transpose(obs, (0, 3, 1, 2))
            obs = obs.reshape(-1, obs.shape[-2], obs.shape[-1])
        elif obs.ndim == 3 and obs.shape[-1] in (1, 3):
            obs = np.transpose(obs, (2, 0, 1))
        return obs.astype(np.uint8)

    def evaluate(self, step: int) -> dict[str, float]:
        returns, lengths = [], []
        for _ in range(self.cfg.train.eval_episodes):
            obs, _ = self.eval_env.reset()
            done = False
            truncated = False
            ep_ret = 0.0
            ep_len = 0
            while not (done or truncated):
                action = self.agent.act(self._to_chw(np.array(obs)), sample=False)
                obs, reward, done, truncated, _ = self.eval_env.step(action)
                ep_ret += reward
                ep_len += 1
            returns.append(ep_ret)
            lengths.append(ep_len)

        result = {
            "step": step,
            "eval_return": float(np.mean(returns)),
            "eval_len": float(np.mean(lengths)),
        }

        out_path = self.run_dir / "eval_metrics.csv"
        write_header = not out_path.exists()
        with out_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(result.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(result)
        return result

    def train(self) -> None:
        obs, _ = self.env.reset(seed=self.cfg.train.seed)
        obs = self._to_chw(np.array(obs))

        ep_return = 0.0
        ep_len = 0

        for step in range(1, self.cfg.train.total_steps + 1):
            if step <= self.cfg.train.start_steps:
                action = self.env.action_space.sample()
            else:
                action = self.agent.act(obs, sample=True)

            next_obs, reward, done, truncated, _ = self.env.step(action)
            next_obs = self._to_chw(np.array(next_obs))
            terminal = done or truncated

            self.replay.add(obs, action, reward, next_obs, terminal)
            obs = next_obs
            ep_return += reward
            ep_len += 1

            if terminal:
                self.csv.log({"step": step, "train_return": ep_return, "train_len": ep_len})
                obs, _ = self.env.reset()
                obs = self._to_chw(np.array(obs))
                ep_return = 0.0
                ep_len = 0

            if (
                step >= self.cfg.train.update_after
                and step % self.cfg.train.update_every == 0
                and len(self.replay) >= self.cfg.train.batch_size
            ):
                metrics = self.agent.update(self.replay.sample(self.cfg.train.batch_size))
                if step % self.cfg.output.log_interval == 0:
                    row = {"step": step, **metrics}
                    self.csv.log(row)
                    if self.tb is not None:
                        for k, v in metrics.items():
                            self.tb.add_scalar(f"train/{k}", v, global_step=step)

            if step % self.cfg.train.eval_interval == 0:
                eval_row = self.evaluate(step)
                if self.tb is not None:
                    self.tb.add_scalar("eval/return", eval_row["eval_return"], global_step=step)
                    self.tb.add_scalar("eval/len", eval_row["eval_len"], global_step=step)

            if step % self.cfg.output.checkpoint_interval == 0:
                save_checkpoint(
                    self.run_dir / "checkpoints" / f"step_{step}.pt",
                    {
                        "step": step,
                        "config": self.cfg,
                        "agent": self.agent.state_dict(),
                    },
                )

        save_checkpoint(
            self.run_dir / "checkpoints" / "final.pt",
            {
                "step": self.cfg.train.total_steps,
                "config": self.cfg,
                "agent": self.agent.state_dict(),
            },
        )
