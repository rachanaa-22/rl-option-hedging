from stable_baselines3.common.env_checker import check_env
from hedging.env import HedgingEnv

env = HedgingEnv()
check_env(env)
print("check_env passed: the environment looks valid")