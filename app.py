import streamlit as st, numpy as np, matplotlib.pyplot as plt
from stable_baselines3 import PPO
from hedging.env import HedgingEnv
from hedging.evaluate import run_policy, no_hedge, delta_hedge, summarize

st.title("RL vs Delta Hedging")
cost = st.slider("Transaction cost rate", 0.0, 0.01, 0.001, 0.0005)
n = st.slider("Episodes", 200, 2000, 500, 100)

env = HedgingEnv(cost_rate=cost)
model = PPO.load("models/ppo_hedger")
rl = lambda obs, e: model.predict(obs, deterministic=True)[0]

res = {"No hedge": run_policy(env, no_hedge, n),
       "Delta": run_policy(env, delta_hedge, n),
       "RL": run_policy(env, rl, n)}
st.dataframe({k: summarize(v) for k, v in res.items()})
fig, ax = plt.subplots()
for k, v in res.items(): ax.hist(v, bins=50, alpha=0.5, label=k)
ax.legend(); st.pyplot(fig)