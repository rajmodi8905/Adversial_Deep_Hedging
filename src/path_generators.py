"""
Stock Price Path Generators:
1. Geometric Brownian Motion (GBM)
2. Stress-Based Simulator (Merton Jump-Diffusion & Volatility Spikes)
3. Adversarial Market Scenario Generator
"""

import numpy as np


def generate_gbm_paths(S0: float, mu: float, sigma: float, T: float, num_steps: int, num_paths: int, seed: int = None):
    """
    Generates synthetic stock price paths using Geometric Brownian Motion (GBM).

    S_{t+dt} = S_t * exp((mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * Z)

    Returns:
        np.ndarray: Array of shape (num_paths, num_steps + 1) containing price paths.
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / num_steps
    # Standard normal increments
    Z = np.random.standard_normal(size=(num_paths, num_steps))

    # Log returns
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    log_returns = drift + diffusion

    # Cumulative log price paths
    log_paths = np.zeros((num_paths, num_steps + 1), dtype=np.float64)
    log_paths[:, 0] = np.log(S0)
    log_paths[:, 1:] = np.log(S0) + np.cumsum(log_returns, axis=1)

    return np.exp(log_paths)


def generate_stress_jump_paths(
    S0: float,
    mu: float,
    base_sigma: float,
    T: float,
    num_steps: int,
    num_paths: int,
    jump_intensity: float = 3.0,      # Average jumps per year
    jump_mean: float = -0.05,         # Average jump return (negative for flash crash stress)
    jump_std: float = 0.08,           # Jump amplitude dispersion
    vol_spike_prob: float = 0.15,     # Probability of intra-path volatility surge
    seed: int = None,
):
    """
    Generates stressed stock price paths with Merton Jump Diffusion and dynamic volatility bursts.
    Simulates flash crashes, heavy tails, and liquidity crises.

    Returns:
        np.ndarray: Array of shape (num_paths, num_steps + 1).
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / num_steps
    paths = np.zeros((num_paths, num_steps + 1), dtype=np.float64)
    paths[:, 0] = S0

    # Compensator for jump drift to keep asset price martingality under risk-neutral measure if desired
    k = np.exp(jump_mean + 0.5 * jump_std**2) - 1.0

    current_S = np.full(num_paths, S0, dtype=np.float64)
    current_sigma = np.full(num_paths, base_sigma, dtype=np.float64)

    for t in range(num_steps):
        # Volatility burst regime shift
        spike_mask = np.random.rand(num_paths) < (vol_spike_prob * dt * 252)
        current_sigma = np.where(spike_mask, base_sigma * np.random.uniform(1.8, 3.0, size=num_paths), base_sigma)

        # Brownian motion diffusion
        Z = np.random.standard_normal(num_paths)
        diff_drift = (mu - jump_intensity * k - 0.5 * current_sigma**2) * dt
        diff_term = current_sigma * np.sqrt(dt) * Z

        # Poisson jump arrivals
        num_jumps = np.random.poisson(jump_intensity * dt, size=num_paths)
        jump_sizes = np.zeros(num_paths)
        has_jumps = num_jumps > 0
        if np.any(has_jumps):
            # Sum of log-normal jumps
            for i in np.where(has_jumps)[0]:
                j_vals = np.random.normal(jump_mean, jump_std, size=num_jumps[i])
                jump_sizes[i] = np.sum(j_vals)

        log_ret = diff_drift + diff_term + jump_sizes
        current_S = current_S * np.exp(log_ret)
        paths[:, t + 1] = current_S

    return paths


def generate_adversarial_paths(
    S0: float,
    K: float,
    base_sigma: float,
    T: float,
    num_steps: int,
    num_paths: int,
    adversarial_mode: str = "mixed",
    seed: int = None,
):
    """
    Generates adversarial market scenarios specifically constructed to exploit delta-hedging weaknesses:
    1. 'pinning_whipsaw': Rapid oscillations across the strike K near expiry (maximum Gamma risk & transaction cost drag).
    2. 'trending_squeeze': Extreme directional momentum creating large gamma deficits.
    3. 'volatility_gap': Calm early period followed by an explosive end-of-horizon regime shift.
    4. 'mixed': A blend of challenging adversarial regimes.

    Returns:
        np.ndarray: Array of shape (num_paths, num_steps + 1).
    """
    if seed is not None:
        np.random.seed(seed)

    dt = T / num_steps
    paths = np.zeros((num_paths, num_steps + 1), dtype=np.float64)
    paths[:, 0] = S0

    for i in range(num_paths):
        mode = adversarial_mode
        if mode == "mixed":
            mode = np.random.choice(["pinning_whipsaw", "trending_squeeze", "volatility_gap"])

        path = np.zeros(num_steps + 1, dtype=np.float64)
        path[0] = S0

        if mode == "pinning_whipsaw":
            # Oscillate violently around the strike K
            for t in range(num_steps):
                time_frac = (t + 1) / num_steps
                # Mean-reverting force towards K with amplified noise
                pull = 15.0 * (K - path[t]) * dt
                noise = base_sigma * np.sqrt(dt) * path[t] * (1.5 + 2.0 * time_frac) * np.random.standard_normal()
                path[t + 1] = max(100.0, path[t] + pull + noise)

        elif mode == "trending_squeeze":
            # Strong drift with random direction to maximize delta mis-specification
            direction = np.random.choice([-1.0, 1.0])
            trend_mu = direction * 0.40  # 40% annualized directional drift
            for t in range(num_steps):
                ret = (trend_mu - 0.5 * base_sigma**2) * dt + base_sigma * np.sqrt(dt) * np.random.standard_normal()
                path[t + 1] = max(100.0, path[t] * np.exp(ret))

        elif mode == "volatility_gap":
            # Calm low-volatility first half, explosive volatility 2nd half
            for t in range(num_steps):
                cur_sigma = base_sigma * 0.5 if t < num_steps // 2 else base_sigma * 2.8
                ret = -0.5 * (cur_sigma**2) * dt + cur_sigma * np.sqrt(dt) * np.random.standard_normal()
                path[t + 1] = max(100.0, path[t] * np.exp(ret))

        paths[i] = path

    return paths
