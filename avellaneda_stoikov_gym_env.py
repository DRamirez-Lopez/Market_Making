"""
Avellaneda–Stoikov Market Making Gymnasium Environment (RL-ready)
-----------------------------------------------------------------
This module implements a self-contained reinforcement-learning environment for
market making inspired by Avellaneda & Stoikov (2008). It uses the Gymnasium API,
so it can plug it into SB3 / CleanRL / Ray RLlib.

Key features
- Mid-price follows arithmetic Brownian motion with volatility sigma.
- Order arrivals follow Poisson processes with exponential intensity vs. quote distance.
- Agent places ONE unit bid/ask per step by choosing (skew, half_spread).
- Reservation price from AS is used as a baseline (risk-aversion gamma).
- Reward = ΔPnL − inv_penalty * q^2 * dt (inventory risk shaping).

Usage
-----
$ pip install gymnasium numpy stable-baselines3

from avellaneda_stoikov_gym_env import MarketMakingEnv, make_env
env = make_env(seed=0)
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())

Training with SB3 (see __main__ block at bottom for a runnable example):
$ python avellaneda_stoikov_gym_env.py

Notes
-----
- Prices are in arbitrary units; you should calibrate sigma, A, k, tick_size to your asset.
- This is a pedagogical baseline. Realistic fills (queue position, partials),
  discrete ticks, and adverse selection models can be added later.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, Dict, Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces


@dataclass
class EnvConfig:
    T: float = 1.0                  # episode horizon in (trading) days
    dt: float = 1/390.0/10.0        # step ~ 1/10 of a minute in a 6.5h session
    sigma: float = 2.0              # mid-price vol (abs units per sqrt(day))
    s0: float = 100.0               # initial mid-price
    gamma: float = 0.1              # risk aversion (AS)
    A: float = 140.0                # base arrival intensity level (per day)
    k: float = 1.5                  # slope of intensity vs. distance
    tick_size: float = 0.01         # price tick
    max_inv: int = 50               # inventory hard cap (units)
    inv_penalty: float = 0.02       # quadratic inventory penalty coefficient
    max_half_spread: float = 1.5    # action bound (absolute dollars)
    max_skew: float = 1.5           # action bound (absolute dollars)
    fill_size: int = 1              # units per fill
    seed: int | None = None


class MarketMakingEnv(gym.Env):
    """
    Observation: (5,)
      [ mid_price_norm, inventory_norm, time_remaining, last_halfspread_norm, last_skew_norm ]
    Action: (2,) Box
      action[0] = skew adjustment (dollars) around reservation price (positive -> shift quotes downward)
      action[1] = half_spread (dollars, >= tick_size)
    Reward:
      ΔPnL − inv_penalty * q^2 * dt (shaping)
    Episode ends at t>=T or |q|>=max_inv (truncated if inventory cap hit)
    """
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self, cfg: EnvConfig = EnvConfig()):
        super().__init__()
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        # Spaces
        low = np.array([-cfg.max_skew, cfg.tick_size], dtype=np.float32)
        high = np.array([cfg.max_skew, cfg.max_half_spread], dtype=np.float32)
        self.action_space = spaces.Box(low=low, high=high, shape=(2,), dtype=np.float32)

        # Observations are roughly standardized
        self._price_scale = max(cfg.s0, 1.0)
        self._spread_scale = max(cfg.max_half_spread, cfg.tick_size)
        self._skew_scale = max(cfg.max_skew, cfg.tick_size)
        obs_low = np.array([0.0, -1.0, 0.0, 0.0, -1.0], dtype=np.float32)
        obs_high = np.array([np.inf, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # State vars
        self.reset(seed=cfg.seed)

    # ---- Core dynamics helpers -------------------------------------------------
    def _reservation_price(self, s: float, q: int, t: float) -> float:
        # r = s − q * gamma * sigma^2 * (T − t)  (Avellaneda–Stoikov heuristic)
        return s - q * self.cfg.gamma * (self.cfg.sigma ** 2) * (self.cfg.T - t)

    def _intensity(self, delta: float) -> float:
        # λ(δ) = A * exp(−k * δ); clamp delta>=0 for intensity definition
        return self.cfg.A * math.exp(-self.cfg.k * max(0.0, delta))

    def _step_price(self, s: float) -> float:
        # Arithmetic BM step: s_{t+dt} = s + sigma * sqrt(dt) * Z
        z = self.rng.standard_normal()
        return s + self.cfg.sigma * math.sqrt(self.cfg.dt) * z

    # ---- Gym API ---------------------------------------------------------------
    def _get_obs(self) -> np.ndarray:
        mid_price_norm = self.s / self._price_scale
        inventory_norm = np.clip(self.q / self.cfg.max_inv, -1.0, 1.0)
        time_remaining = max(self.cfg.T - self.t, 0.0) / self.cfg.T
        last_halfspread_norm = self.last_half_spread / self._spread_scale
        last_skew_norm = np.clip(self.last_skew / self._skew_scale, -1.0, 1.0)
        return np.array([
            mid_price_norm,
            inventory_norm,
            time_remaining,
            last_halfspread_norm,
            last_skew_norm,
        ], dtype=np.float32)

    def _get_info(self) -> Dict[str, Any]:
        return {
            "mid_price": self.s,
            "cash": self.cash,
            "inventory": int(self.q),
            "bid": self.bid,
            "ask": self.ask,
            "pnl": self.cash + self.q * self.s,
            "time": self.t,
        }

    def reset(self, *, seed: int | None = None, options: Dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.s = float(self.cfg.s0)
        self.q = 0
        self.cash = 0.0
        self.t = 0.0
        self.last_half_spread = self.cfg.tick_size
        self.last_skew = 0.0
        self.bid = self.s - self.last_half_spread
        self.ask = self.s + self.last_half_spread
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        skew, half_spread = float(action[0]), float(action[1])
        # Bound actions explicitly
        half_spread = float(np.clip(half_spread, self.cfg.tick_size, self.cfg.max_half_spread))
        skew = float(np.clip(skew, -self.cfg.max_skew, self.cfg.max_skew))

        # Baseline reservation price and quotes (ticks enforced)
        r = self._reservation_price(self.s, self.q, self.t)
        bid = r - half_spread - skew
        ask = r + half_spread - skew
        # Tick rounding
        bid = math.floor(bid / self.cfg.tick_size) * self.cfg.tick_size
        ask = math.ceil(ask / self.cfg.tick_size) * self.cfg.tick_size
        # Ensure bid < ask
        if ask - bid < 2 * self.cfg.tick_size:
            # widen minimally to respect ticks
            mid = 0.5 * (bid + ask)
            bid = mid - self.cfg.tick_size
            ask = mid + self.cfg.tick_size

        # Execution probabilities over dt (at most one fill each side per step)
        delta_b = max(self.s - bid, 0.0)
        delta_a = max(ask - self.s, 0.0)
        lam_b = self._intensity(delta_b)
        lam_a = self._intensity(delta_a)
        p_fill_b = 1.0 - math.exp(-lam_b * self.cfg.dt)
        p_fill_a = 1.0 - math.exp(-lam_a * self.cfg.dt)

        # Sample fills
        fill_b = self.rng.random() < p_fill_b  # we BUY at bid
        fill_a = self.rng.random() < p_fill_a  # we SELL at ask

        # Apply fills (one unit each side)
        cash_before = self.cash
        q_before = self.q
        if fill_b:
            self.cash -= bid * self.cfg.fill_size
            self.q += self.cfg.fill_size
        if fill_a:
            self.cash += ask * self.cfg.fill_size
            self.q -= self.cfg.fill_size

        # Price evolution after orders
        s_prev = self.s
        self.s = self._step_price(self.s)

        # Reward: mark-to-market PnL change minus inventory penalty
        pnl_prev = cash_before + q_before * s_prev
        pnl_now = self.cash + self.q * self.s
        pnl_delta = pnl_now - pnl_prev
        inv_pen = self.cfg.inv_penalty * (self.q ** 2) * self.cfg.dt
        reward = float(pnl_delta - inv_pen)

        # Advance time and write state
        self.t += self.cfg.dt
        self.last_half_spread = half_spread
        self.last_skew = skew
        self.bid, self.ask = bid, ask

        # Termination / Truncation
        terminated = self.t >= self.cfg.T
        truncated = abs(self.q) >= self.cfg.max_inv

        obs = self._get_obs()
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    # Optional
    def render(self):
        print(self._get_info())


# Convenience factory

def make_env(cfg: EnvConfig | None = None, seed: int | None = None) -> MarketMakingEnv:
    cfg = cfg or EnvConfig()
    if seed is not None:
        cfg.seed = seed
    return MarketMakingEnv(cfg)


# --------------------------- Training Example (SB3) ----------------------------
if __name__ == "__main__":
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv
        from stable_baselines3.common.monitor import Monitor
    except Exception as e:
        print("Stable-Baselines3 not available. Install with: pip install stable-baselines3")
        raise

    # Create and wrap env
    def _factory():
        env = make_env(seed=42)
        return Monitor(env)

    vec_env = DummyVecEnv([_factory])

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=1,
        n_steps=2048,
        batch_size=256,
        gae_lambda=0.95,
        gamma=0.999,          # long-ish horizon
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.01,
        tensorboard_log=None,
        device="auto",
    )

    timesteps = 200_000
    model.learn(total_timesteps=timesteps)

    # Quick evaluation rollout
    env = make_env(seed=123)
    obs, info = env.reset()
    total_reward = 0.0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if terminated or truncated:
            break
    final_pnl = info["pnl"]
    print(f"Episode finished. Total reward: {total_reward:.2f}  Final PnL: {final_pnl:.2f}")