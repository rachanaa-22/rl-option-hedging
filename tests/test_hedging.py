import numpy as np
from hedging.pricing import bs_call_price, bs_call_delta
from hedging.env import HedgingEnv
from hedging.evaluate import run_policy, no_hedge, delta_hedge


def test_price_is_intrinsic_at_expiry():
    assert bs_call_price(110, 100, 0, 0, 0.2) == 10


def test_delta_between_0_and_1():
    for S in [50, 100, 150]:
        assert 0 <= bs_call_delta(S, 100, 0.1, 0, 0.2) <= 1


def test_env_obs_shape_and_episode_length():
    env = HedgingEnv()
    obs, _ = env.reset(seed=1)
    assert obs.shape == (3,)
    steps, done = 0, False
    while not done:
        _, _, done, _, _ = env.step(env.action_space.sample())
        steps += 1
    assert steps == env.n_steps


def test_same_seed_same_result():
    a = run_policy(HedgingEnv(), delta_hedge, n_episodes=20)
    b = run_policy(HedgingEnv(), delta_hedge, n_episodes=20)
    assert np.allclose(a, b)


def test_delta_hedge_beats_no_hedge_without_costs():
    env = HedgingEnv(cost_rate=0.0)
    assert run_policy(env, delta_hedge, 500).std() < run_policy(env, no_hedge, 500).std()