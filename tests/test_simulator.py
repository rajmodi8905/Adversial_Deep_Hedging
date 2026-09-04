"""
Unit Tests for Delta Hedging & Stock Path Generation Simulator.
"""

import pytest
import numpy as np
from src.black_scholes import black_scholes_price, black_scholes_greeks, implied_volatility
from src.path_generators import generate_gbm_paths, generate_stress_jump_paths, generate_adversarial_paths
from src.hedging_engine import compute_black_scholes_deltas, simulate_hedging_pnl
from src.metrics import calculate_var, calculate_cvar, calculate_turnover


def test_black_scholes_put_call_parity():
    """Validates Put-Call Parity: C - P = S - K * exp(-r * T)."""
    S, K, T, r, sigma = 25000.0, 25000.0, 30.0 / 365.0, 0.07, 0.18
    call_price = black_scholes_price(S, K, T, r, sigma, option_type="call")
    put_price = black_scholes_price(S, K, T, r, sigma, option_type="put")

    lhs = call_price - put_price
    rhs = S - K * np.exp(-r * T)
    assert np.isclose(lhs, rhs, atol=1e-5), f"Put-Call parity violation: {lhs} != {rhs}"


def test_black_scholes_greeks_bounds():
    """Checks mathematical Greek boundaries."""
    S, K, T, r, sigma = 25000.0, 25000.0, 10.0 / 365.0, 0.07, 0.20
    greeks_call = black_scholes_greeks(S, K, T, r, sigma, option_type="call")
    greeks_put = black_scholes_greeks(S, K, T, r, sigma, option_type="put")

    # Delta bounds
    assert 0.0 < greeks_call["delta"] < 1.0
    assert -1.0 < greeks_put["delta"] < 0.0
    assert np.isclose(greeks_call["delta"] - greeks_put["delta"], 1.0, atol=1e-5)

    # Gamma and Vega positivity
    assert greeks_call["gamma"] > 0.0
    assert greeks_call["vega"] > 0.0


def test_implied_volatility_solver():
    """Validates that IV solver inverts the BS formula with high precision."""
    S, K, T, r, true_sigma = 25500.0, 25500.0, 7.0 / 365.0, 0.07, 0.22
    mkt_price = black_scholes_price(S, K, T, r, true_sigma, option_type="call")

    recovered_iv = implied_volatility(mkt_price, S, K, T, r, option_type="call")
    assert np.isclose(recovered_iv, true_sigma, atol=1e-4)


def test_path_generators():
    """Validates output shape, positivity, and non-degeneracy of all 3 path generators."""
    S0, K, T, num_steps, num_paths = 25000.0, 25000.0, 1.0 / 252.0, 50, 100

    # 1. GBM
    gbm = generate_gbm_paths(S0, mu=0.07, sigma=0.15, T=T, num_steps=num_steps, num_paths=num_paths, seed=42)
    assert gbm.shape == (num_paths, num_steps + 1)
    assert np.all(gbm > 0)
    assert np.allclose(gbm[:, 0], S0)

    # 2. Stress Jumps
    stress = generate_stress_jump_paths(S0, mu=0.07, base_sigma=0.15, T=T, num_steps=num_steps, num_paths=num_paths, seed=42)
    assert stress.shape == (num_paths, num_steps + 1)
    assert np.all(stress > 0)
    assert np.allclose(stress[:, 0], S0)

    # 3. Adversarial
    adv = generate_adversarial_paths(S0, K=K, base_sigma=0.15, T=T, num_steps=num_steps, num_paths=num_paths, seed=42)
    assert adv.shape == (num_paths, num_steps + 1)
    assert np.all(adv > 0)
    assert np.allclose(adv[:, 0], S0)


def test_hedging_pnl_conservation():
    """Validates cashflow conservation in the hedging simulation engine."""
    S0, K, T, num_steps, num_paths = 25000.0, 25000.0, 1.0 / 252.0, 20, 50
    paths = generate_gbm_paths(S0, mu=0.07, sigma=0.15, T=T, num_steps=num_steps, num_paths=num_paths, seed=123)
    deltas = compute_black_scholes_deltas(paths, K, T, r=0.07, sigma=0.15, option_type="call")

    sim_res = simulate_hedging_pnl(paths, deltas, K, T, r=0.07, cost_rate=0.001, sigma_for_premium=0.15)
    expected_net = sim_res["initial_premium"] - sim_res["option_payoffs"] + sim_res["trading_pnl"] - sim_res["total_costs"]

    assert np.allclose(sim_res["net_pnl"], expected_net, atol=1e-5)
    assert np.all(sim_res["total_costs"] >= 0.0)


def test_risk_metrics_cvar_le_var():
    """Validates that CVaR is always less than or equal to VaR for loss distributions."""
    pnl = np.random.normal(loc=0.0, scale=100.0, size=5000)
    var_5 = calculate_var(pnl, alpha=0.05)
    cvar_5 = calculate_cvar(pnl, alpha=0.05)

    assert cvar_5 <= var_5, f"CVaR ({cvar_5}) should be <= VaR ({var_5})"
