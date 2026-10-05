import numpy as np
from hedging.pricing import bs_call_delta

def run_policy(env, policy, n_episodes=2000, seed=123):
    totals = []
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done, total = False, 0.0
        while not done:
            obs, _, done, _, info = env.step(policy(obs, env))
            total += info["pnl"]
        totals.append(total)
    return np.array(totals)

def no_hedge(obs, env):
    return np.array([-1.0])                       # h = 0

def delta_hedge(obs, env):
    d = bs_call_delta(env.S, env.K, env._tau(env.t), 0.0, env.sigma)
    return np.array([2 * d - 1])                  # map [0,1] -> [-1,1]

def summarize(pnl):
    k = max(1, int(0.05 * len(pnl)))
    return {"mean": pnl.mean(), "std": pnl.std(),
            "CVaR95": np.sort(pnl)[:k].mean()}    # avg of worst 5% outcomes