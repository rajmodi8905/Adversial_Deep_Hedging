"""
Discrete-Time Delta Hedging Engine with Proportional Transaction Costs and P&L Accounting.
"""

import numpy as np
from .black_scholes import black_scholes_price, black_scholes_greeks


def compute_black_scholes_deltas(
    paths: np.ndarray,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call"
):
    """
    Computes theoretical Black-Scholes deltas along all paths at each discrete time step.

    Args:
        paths (np.ndarray): Shape (num_paths, num_steps + 1)
        K (float): Strike price
        T (float): Total maturity in years
        r (float): Risk-free rate
        sigma (float): Annualized volatility
        option_type (str): 'call' or 'put'

    Returns:
        np.ndarray: Deltas of shape (num_paths, num_steps + 1)
    """
    num_paths, total_steps = paths.shape
    num_steps = total_steps - 1
    dt = T / num_steps

    deltas = np.zeros_like(paths, dtype=np.float64)

    for step in range(total_steps):
        time_left = max(1e-7, T - step * dt)
        S_t = paths[:, step]
        greeks = black_scholes_greeks(S_t, K, time_left, r, sigma, option_type=option_type)
        deltas[:, step] = greeks["delta"]

    return deltas


def simulate_hedging_pnl(
    paths: np.ndarray,
    deltas: np.ndarray,
    K: float,
    T: float,
    r: float,
    cost_rate: float = 0.0,
    option_type: str = "call",
    include_initial_premium: bool = True,
    sigma_for_premium: float = 0.20
):
    """
    Simulates discrete-time delta hedging P&L across all paths with transaction costs.

    For an option issuer (seller of 1 contract):
    1. Initial cash received: C_0 (option premium)
    2. Trading P&L: sum_{k=0}^{N-1} delta_k * (S_{k+1} - S_k)
    3. Transaction costs: sum_{k=0}^N c * S_k * |delta_k - delta_{k-1}| with delta_{-1}=0, delta_N=0
    4. Payoff at expiry: -Z(S_T)

    Net P&L = C_0 - Z(S_T) + Trading_PnL - Total_Costs

    Returns:
        dict containing:
            'net_pnl': np.ndarray of shape (num_paths,)
            'trading_pnl': np.ndarray of shape (num_paths,)
            'total_costs': np.ndarray of shape (num_paths,)
            'option_payoffs': np.ndarray of shape (num_paths,)
            'initial_premium': float
    """
    num_paths, total_steps = paths.shape
    num_steps = total_steps - 1

    S0 = paths[:, 0]
    S_T = paths[:, -1]

    # Initial premium
    if include_initial_premium:
        init_premium = black_scholes_price(S0[0], K, T, r, sigma_for_premium, option_type=option_type)
    else:
        init_premium = 0.0

    # Option payoff at expiry
    if option_type.lower() == "call":
        payoff = np.maximum(0.0, S_T - K)
    else:
        payoff = np.maximum(0.0, K - S_T)

    # Trading P&L from underlying asset changes
    # delta held during interval [t_k, t_{k+1}] is delta[:, k]
    price_diffs = paths[:, 1:] - paths[:, :-1]
    trading_pnl_intervals = deltas[:, :-1] * price_diffs
    trading_pnl = np.sum(trading_pnl_intervals, axis=1)

    # Transaction costs across all rebalances
    # Start: rebalance from 0 to delta_0 at t=0
    # Steps: rebalance from delta_{k-1} to delta_k at t_k
    # End: liquidate delta_{N-1} to 0 at t=T
    costs = np.zeros(num_paths, dtype=np.float64)

    if cost_rate > 0.0:
        # Initial trade cost at t=0
        costs += cost_rate * paths[:, 0] * np.abs(deltas[:, 0])

        # Intermediate rebalancing costs
        rebalance_sizes = np.abs(deltas[:, 1:-1] - deltas[:, :-2])
        costs += np.sum(cost_rate * paths[:, 1:-1] * rebalance_sizes, axis=1)

        # Final liquidation trade cost at t=T
        costs += cost_rate * paths[:, -1] * np.abs(deltas[:, -2])

    net_pnl = init_premium - payoff + trading_pnl - costs

    return {
        "net_pnl": net_pnl,
        "trading_pnl": trading_pnl,
        "total_costs": costs,
        "option_payoffs": payoff,
        "initial_premium": init_premium,
    }
