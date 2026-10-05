import numpy as np

def simulate_gbm(n_paths, n_steps, S0=100.0, mu=0.0, sigma=0.2, T=30/365, seed=0):
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    log_ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(log_ret, axis=1)], axis=1)
    return S0 * np.exp(log_paths)   # shape (n_paths, n_steps + 1)