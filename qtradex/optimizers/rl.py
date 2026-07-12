"""
Proximal Policy Optimization (PPO) optimizer for QTradeX.

Optional dependency: pip install stable-baselines3
Uses PPO to learn parameter mutation policies directly from trial-and-error.
"""

import math
import os
import time
import json
from copy import deepcopy
from random import random

import numpy as np

from qtradex.core import backtest
from qtradex.core.base_bot import Info
from qtradex.optimizers.utilities import bound_neurons

try:
    import gymnasium as gym
    from gymnasium import spaces
    import stable_baselines3 as sb3
    from stable_baselines3 import PPO
    _HAS_RL = True
except ImportError:
    _HAS_RL = False


class RLPPOoptions:
    """Configuration for RLPPO optimizer."""
    def __init__(self):
        self.total_timesteps = 50000     # total backtests across training
        self.learning_rate = 3e-4
        self.gamma = 0.99                # discount factor
        self.gae_lambda = 0.95
        self.clip_range = 0.2
        self.ent_coef = 0.01             # entropy bonus for exploration
        self.n_steps = 512               # steps per update
        self.batch_size = 64
        self.n_epochs = 10
        self.verbose = 1
        self.show_terminal = True
        self.print_tune = False
        self.epochs = math.inf
        self.improvements = math.inf
        self.select_data = False


class TradingEnv(gym.Env):
    """Gymnasium environment wrapping a QTradeX bot backtest.

    State: normalized tune params (concatenated into a flat array).
    Action: continuous deltas in [-1, 1] for each tunable param.
    Reward: sortino improvement over the best seen so far.
    """
    metadata = {"render_modes": []}

    def __init__(self, bot, data, wallet, **kwargs):
        super().__init__()
        self.bot = bot
        self.data = data
        self.wallet = wallet
        self.kwargs = kwargs

        # Extract tunable params and their clamps
        self.tunable_params = [p for p in bot.tune.keys() if bot.clamps[p][3]]
        self.clamp_info = {p: bot.clamps[p] for p in self.tunable_params}
        self.n_params = len(self.tunable_params)

        # Gym spaces
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_params,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_params + 2,), dtype=np.float32)

        self.best_sortino = -999.0
        self.episode_count = 0

    def _get_obs(self):
        """Normalize current tune params to [0, 1], append performance."""
        obs = []
        for p in self.tunable_params:
            lo, _, hi, _ = self.clamp_info[p]
            val = float(self.bot.tune[p])
            obs.append((val - lo) / (hi - lo))
        # Append rolling performance: last sortino, best sortino, progress
        obs.append(max(-1.0, min(1.0, self.best_sortino / 10.0)))
        obs.append(max(0.0, min(1.0, self.episode_count / 100)))
        return np.array(obs, dtype=np.float32)

    def _apply_action(self, action):
        """Scale action deltas by clamp ranges and apply to tune."""
        for i, p in enumerate(self.tunable_params):
            lo, _, hi, _ = self.clamp_info[p]
            span = hi - lo
            delta = float(action[i]) * span * 0.1  # 10% of range per step
            self.bot.tune[p] = float(self.bot.tune[p]) + delta
        bound_neurons(self.bot)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.bot.reset()
        self._apply_action(np.random.uniform(-0.5, 0.5, size=self.n_params))
        return self._get_obs(), {}

    def step(self, action):
        self._apply_action(action)
        self.episode_count += 1

        result = backtest(deepcopy(self.bot), self.data, deepcopy(self.wallet), plot=False, **self.kwargs)
        sortino = result.get("sortino_ratio", -999.0)
        roi = result.get("roi", 0)

        # Reward: sortino improvement over best
        reward = max(-1.0, min(1.0, (sortino - self.best_sortino) / 5.0))
        if sortino > self.best_sortino:
            self.best_sortino = sortino
            reward = 1.0  # big reward for new best

        # Penalize negative ROI
        if roi <= 0:
            reward -= 0.5

        done = self.episode_count >= 500  # reset after 500 episodes to avoid stale env
        return self._get_obs(), reward, done, False, {"sortino": sortino, "roi": roi}


class RLPPO:
    """PPO-based optimizer for QTradeX.

    Trains a policy to output parameter mutations that maximize sortino.
    Falls back to random search if stable-baselines3 is not installed.
    """
    def __init__(self, data, wallet=None, options=None):
        if wallet is None:
            raise ValueError("Wallet required")
        self.data = data
        self.wallet = wallet
        self.options = options or RLPPOoptions()

    def optimize(self, bot, **kwargs):
        bot.info = Info({"mode": "optimize"})
        bot.reset()

        if not _HAS_RL:
            print("stable-baselines3 not installed. Run: pip install stable-baselines3")
            return self._random_search(bot, **kwargs)

        env = TradingEnv(bot, self.data, self.wallet, **kwargs)

        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=self.options.learning_rate,
            n_steps=self.options.n_steps,
            batch_size=self.options.batch_size,
            n_epochs=self.options.n_epochs,
            gamma=self.options.gamma,
            gae_lambda=self.options.gae_lambda,
            clip_range=self.options.clip_range,
            ent_coef=self.options.ent_coef,
            verbose=self.options.verbose,
        )

        print("Training PPO...")
        model.learn(total_timesteps=self.options.total_timesteps)

        # Evaluate the trained policy
        print("Evaluating best policy...")
        obs, _ = env.reset()
        best_sortino = -999
        best_bot = deepcopy(bot)
        for _ in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            if info.get("sortino", -999) > best_sortino:
                best_sortino = info["sortino"]
                best_bot = deepcopy(env.bot)
            if done:
                break

        result = backtest(deepcopy(best_bot), self.data, deepcopy(self.wallet), plot=False, **kwargs)
        coords = [k for k in result.keys() if isinstance(result[k], (int, float))]
        best_bots = {c: [result.copy(), deepcopy(best_bot)] for c in coords}

        if self.options.print_tune:
            from qtradex.optimizers.utilities import print_tune
            print_tune(result, best_bot)

        return best_bots

    def _random_search(self, bot, **kwargs):
        """Fallback: random search with 1000 trials."""
        print("Running random search fallback...")
        coords = ["sortino_ratio", "roi"]
        best_bots = {c: [{"sortino_ratio": -999, "roi": 0}, deepcopy(bot)] for c in coords}

        for i in range(1000):
            trial = deepcopy(bot)
            for p in trial.tune.keys():
                if trial.clamps[p][3]:
                    lo, _, hi, _ = trial.clamps[p]
                    trial.tune[p] = lo + random() * (hi - lo)
            bound_neurons(trial)
            result = backtest(trial, self.data, deepcopy(self.wallet), plot=False, **kwargs)
            sortino = result.get("sortino_ratio", -999)
            if sortino > best_bots["sortino_ratio"][0].get("sortino_ratio", -999):
                best_bots["sortino_ratio"] = [result, deepcopy(trial)]

        return best_bots
