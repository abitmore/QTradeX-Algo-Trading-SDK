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


def _composite(result):
    """Composite score: sortino * (1 - max_dd) * trade_mult."""
    sortino = result.get("sortino_ratio", 0)
    dd = result.get("maximum_drawdown", 1)
    trades_raw = result.get("trades", [])
    n_trades = len(trades_raw) if isinstance(trades_raw, (list, tuple)) else result.get("trade_win_rate", 0) * 100
    trade_mult = min(1.0, n_trades / 30) if n_trades > 0 else 0
    return sortino * (1 - dd) * trade_mult


class RLPPOoptions:
    """Configuration for RLPPO optimizer."""
    def __init__(self):
        self.total_timesteps = 50000     # total backtests across training
        self.learning_rate = 3e-4
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_range = 0.2
        self.ent_coef = 0.01
        self.n_steps = 512
        self.batch_size = 64
        self.n_epochs = 10
        self.verbose = 1
        self.show_terminal = True
        self.print_tune = False
        self.timeout = 0
        self.epochs = math.inf
        self.improvements = math.inf
        self.select_data = False
        self.walk_forward = True   # split data, reward = min(train, val) composite


class TradingEnv(gym.Env if _HAS_RL else object):
    """Gymnasium environment wrapping a QTradeX bot backtest.

    State: normalized tune params.
    Action: continuous deltas in [-1, 1] for each tunable param.
    Reward: composite score (sortino * (1-DD) * trade_count).
    If walk_forward: data split 2/3 + 1/3, reward = min(train_c, val_c).
    """
    metadata = {"render_modes": []}

    def __init__(self, bot, data, wallet, options=None, **kwargs):
        super().__init__()
        self.bot = bot
        self.data = data
        self.wallet = wallet
        self.options = options or RLPPOoptions()
        self.kwargs = kwargs

        self.tunable_params = [p for p in bot.tune.keys() if bot.clamps[p][3]]
        self.clamp_info = {p: bot.clamps[p] for p in self.tunable_params}
        self.n_params = len(self.tunable_params)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.n_params,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(self.n_params + 2,), dtype=np.float32)

        # Walk-forward split
        if self.options.walk_forward:
            split = int(len(data) * 2 / 3)
            self._train_data = data[:split]
            self._val_data = data[split:]
        else:
            self._train_data = data
            self._val_data = None

        self.best_score = -999.0
        self.episode_count = 0

    def _get_obs(self):
        obs = []
        for p in self.tunable_params:
            lo, _, hi, _ = self.clamp_info[p]
            val = float(self.bot.tune[p])
            obs.append((val - lo) / (hi - lo))
        obs.append(max(-1.0, min(1.0, self.best_score / 10.0)))
        obs.append(max(0.0, min(1.0, self.episode_count / 100)))
        return np.array(obs, dtype=np.float32)

    def _apply_action(self, action):
        for i, p in enumerate(self.tunable_params):
            lo, _, hi, _ = self.clamp_info[p]
            span = hi - lo
            delta = float(action[i]) * span * 0.1
            self.bot.tune[p] = float(self.bot.tune[p]) + delta
        bound_neurons(self.bot)

    def _eval(self, data):
        """Backtest on data, return composite score."""
        r = backtest(deepcopy(self.bot), data, deepcopy(self.wallet), plot=False, **self.kwargs)
        return _composite(r)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.bot.reset()
        self._apply_action(np.random.uniform(-0.5, 0.5, size=self.n_params))
        return self._get_obs(), {}

    def step(self, action):
        self._apply_action(action)
        self.episode_count += 1

        train_c = self._eval(self._train_data)
        if self._val_data is not None:
            val_c = self._eval(self._val_data)
            composite = 0.7 * train_c + 0.3 * val_c
        else:
            composite = train_c

        reward = max(-1.0, min(1.0, (composite - self.best_score) / 2.0))
        if composite > self.best_score:
            self.best_score = composite
            reward = 1.0

        done = self.episode_count >= 500
        return self._get_obs(), reward, done, False, {"composite": composite}


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

        env = TradingEnv(bot, self.data, self.wallet, options=self.options, **kwargs)

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
        best_score = -999
        best_bot = deepcopy(bot)
        for _ in range(100):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            if info.get("composite", -999) > best_score:
                best_score = info["composite"]
                best_bot = deepcopy(env.bot)
            if done:
                break

        result = backtest(deepcopy(best_bot), self.data, deepcopy(self.wallet), plot=False, **kwargs)
        coords = [k for k in result.keys() if isinstance(result[k], (int, float))]
        best_bots = {c: [result.copy(), deepcopy(best_bot)] for c in coords}

        if self.options.print_tune:
            from qtradex.optimizers.utilities import print_tune
            print_tune(result, best_bot)

        from qtradex.optimizers.utilities import end_optimization
        end_optimization(best_bots, False)
        return best_bots

    def _random_search(self, bot, **kwargs):
        """Fallback: random search with 1000 trials."""
        print("Running random search fallback...")
        coords = ["sortino_ratio", "roi"]
        best_bots = {c: [{"sortino_ratio": -999, "roi": 0}, deepcopy(bot)] for c in coords}

        rs_start = time.time()
        for i in range(1000):
            if self.options.timeout and time.time() - rs_start > self.options.timeout:
                print(f"RL fallback timed out after {self.options.timeout}s")
                break
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

        from qtradex.optimizers.utilities import end_optimization
        end_optimization(best_bots, False)
        return best_bots
