import numpy as np
import gymnasium as gym
from gymnasium import spaces
from hedging.pricing import bs_call_price

class HedgingEnv(gym.Env):
    """Agent is SHORT one call option and hedges daily with shares. r = 0."""

    def __init__(self, n_steps=30, S0=100.0, K=100.0, sigma=0.2,
                 T=30/365, cost_rate=0.001, risk_aversion=1.0):
        super().__init__()
        self.n_steps, self.S0, self.K = n_steps, S0, K
        self.sigma, self.T = sigma, T
        self.cost_rate, self.lam = cost_rate, risk_aversion
        self.dt = T / n_steps
        self.observation_space = spaces.Box(
            low=np.array([-1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0], dtype=np.float32))
        # action in [-1, 1] is mapped to a hedge ratio in [0, 1]
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def _tau(self, t):               # time to maturity in years
        return self.T * (1 - t / self.n_steps)

    def _obs(self):
        m = np.clip(np.log(self.S / self.K), -1, 1)
        return np.array([m, 1 - self.t / self.n_steps, self.h], dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t, self.S, self.h = 0, self.S0, 0.0
        return self._obs(), {}

    def step(self, action):
        new_h = (float(np.clip(action[0], -1, 1)) + 1) / 2
        cost = self.cost_rate * abs(new_h - self.h) * self.S

        z = self.np_random.standard_normal()
        S_new = self.S * np.exp(-0.5 * self.sigma**2 * self.dt
                                + self.sigma * np.sqrt(self.dt) * z)

        c_old = bs_call_price(self.S, self.K, self._tau(self.t), 0.0, self.sigma)
        c_new = bs_call_price(S_new, self.K, self._tau(self.t + 1), 0.0, self.sigma)

        pnl = new_h * (S_new - self.S) - cost - (c_new - c_old)
        reward = pnl - self.lam * pnl**2

        self.t += 1
        self.S, self.h = S_new, new_h
        done = self.t >= self.n_steps
        return self._obs(), float(reward), done, False, {"pnl": pnl}