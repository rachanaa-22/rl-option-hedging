import numpy as np, pandas as pd, matplotlib.pyplot as plt
from stable_baselines3 import PPO
from hedging.env import HedgingEnv
from hedging.evaluate import run_policy, no_hedge, delta_hedge, summarize

env = HedgingEnv(cost_rate=0.001)
model = PPO.load("models/ppo_hedger")
rl = lambda obs, e: model.predict(obs, deterministic=True)[0]

results = {"No hedge": run_policy(env, no_hedge),
           "Delta hedge": run_policy(env, delta_hedge),
           "RL (PPO)": run_policy(env, rl)}

table = pd.DataFrame({k: summarize(v) for k, v in results.items()}).T
print(table); table.to_csv("results/summary.csv")

for name, pnl in results.items():
    plt.hist(pnl, bins=60, alpha=0.5, label=name)
plt.xlabel("Total hedging P&L"); plt.legend()
plt.savefig("results/pnl_hist.png", dpi=150)