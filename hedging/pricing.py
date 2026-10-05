import numpy as np
from scipy.stats import norm

def _d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

def bs_call_price(S, K, T, r, sigma):
    if T <= 1e-12:
        return max(S - K, 0.0)
    d1 = _d1(S, K, T, r, sigma)
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_call_delta(S, K, T, r, sigma):
    if T <= 1e-12:
        return 1.0 if S > K else 0.0
    return float(norm.cdf(_d1(S, K, T, r, sigma)))