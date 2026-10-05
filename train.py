from stable_baselines3 import PPO
from hedging.env import HedgingEnv

env = HedgingEnv(cost_rate=0.001)
model = PPO("MlpPolicy", env, verbose=1, seed=0,
            learning_rate=3e-4, n_steps=2048, batch_size=64)
model.learn(total_timesteps=300_000)
model.save("models/ppo_hedger")