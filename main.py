"""
Main Pipeline Runner for Delta Hedging & Adversarial Stock Path Generation Simulator.

Executes:
1. NIFTY Real Market Calibration & Baseline Greeks Extraction
2. 3 Stock Path Generators: GBM, Stress/Jump Diffusion, and Adversarial Scenarios
3. Training & Evaluation of Deep Hedger (CVaR optimization under Transaction Costs)
4. Comprehensive Metric Comparison (Mean P&L, Hedging Error, VaR, CVaR, Turnover)
5. Generation of Publication-Quality Visualization Plots & Resume Metric Highlights
"""

import os
import numpy as np
import pandas as pd
from src.black_scholes import black_scholes_price, black_scholes_greeks, implied_volatility
from src.market_data import load_nifty_minute_data, get_options_snapshot
from src.path_generators import generate_gbm_paths, generate_stress_jump_paths, generate_adversarial_paths
from src.hedging_engine import compute_black_scholes_deltas, simulate_hedging_pnl
from src.deep_hedger import DeepHedger
from src.metrics import evaluate_hedging_performance, compare_strategies
from src.visualizer import plot_pnl_comparison, plot_sample_paths


def run_full_simulation():
    print("=" * 80)
    print("  DELTA HEDGING & ADVERSARIAL STOCK PATH GENERATION SIMULATOR")
    print("=" * 80)

    # -------------------------------------------------------------
    # 1. Market Data Calibration from NIFTY Option Minute Data
    # -------------------------------------------------------------
    print("\n[Step 1] Calibrating parameters on Real NSE NIFTY Option & Futures data...")
    data_dir = os.path.join(os.path.dirname(__file__), "drive-download-20260903T145837Z-1-001")
    csv_non_expiry = os.path.join(data_dir, "20260204_option_minute_prices_non_expiry.csv")
    csv_expiry = os.path.join(data_dir, "20260205_option_minute_prices_expiry.csv")

    df_non_exp = load_nifty_minute_data(csv_non_expiry)
    spot_price, options_snapshot = get_options_snapshot(df_non_exp, minute_str="110000", r=0.07, days_to_expiry=1.0)

    # Filter for ATM Call option
    calls = options_snapshot[options_snapshot["type"] == "CE"].copy()
    calls["moneyness_dist"] = np.abs(calls["strike"] - spot_price)
    atm_call = calls.sort_values("moneyness_dist").iloc[0]

    K = float(atm_call["strike"])
    S0 = float(spot_price)
    sigma_calibrated = float(atm_call["implied_vol"]) if not np.isnan(atm_call["implied_vol"]) else 0.15
    r = 0.07
    T = 1.0 / 252.0  # 1-day trading horizon (or 1/52 for weekly)
    cost_rate = 0.001  # 10 bps (0.10%) proportional transaction cost

    print(f"  Underlying Spot (NIFTY): {S0:.2f} INR")
    print(f"  Calibrated Strike (ATM Call): {K:.0f} INR")
    print(f"  Calibrated Implied Volatility: {sigma_calibrated:.2%}")
    print(f"  Horizon (T): {T:.4f} years (1 trading day)")
    print(f"  Transaction Cost Rate: {cost_rate:.2%}")

    # -------------------------------------------------------------
    # 2. Monte Carlo Stock Path Generation (10,000+ paths)
    # -------------------------------------------------------------
    num_paths = 10000
    num_steps = 100  # Rebalancing intervals (can be 50, 100, 250)
    print(f"\n[Step 2] Generating {num_paths:,} stock price paths across 3 techniques (Steps={num_steps})...")

    # Technique 1: Geometric Brownian Motion (GBM)
    gbm_train = generate_gbm_paths(S0, mu=r, sigma=sigma_calibrated, T=T, num_steps=num_steps, num_paths=num_paths, seed=42)
    gbm_test = generate_gbm_paths(S0, mu=r, sigma=sigma_calibrated, T=T, num_steps=num_steps, num_paths=num_paths, seed=1042)
    print("  ✓ Technique 1: Geometric Brownian Motion (GBM) generated.")

    # Technique 2: Stress-Based Simulator (Merton Jump Diffusion + Volatility Spikes)
    stress_paths = generate_stress_jump_paths(
        S0, mu=r, base_sigma=sigma_calibrated, T=T, num_steps=num_steps, num_paths=num_paths,
        jump_intensity=5.0, jump_mean=-0.03, jump_std=0.05, seed=999
    )
    print("  ✓ Technique 2: Stress-Based Jump Diffusion & Volatility Spikes generated.")

    # Technique 3: Adversarial Scenario Generator
    adv_paths = generate_adversarial_paths(
        S0, K=K, base_sigma=sigma_calibrated, T=T, num_steps=num_steps, num_paths=num_paths,
        adversarial_mode="mixed", seed=777
    )
    print("  ✓ Technique 3: Adversarial Market Scenarios generated.")

    # -------------------------------------------------------------
    # 3. Model Training: Neural Network Deep Hedger (PyTorch)
    # -------------------------------------------------------------
    print(f"\n[Step 3] Training Neural Deep Hedger with Differentiable CVaR (5%) Loss & Transaction Frictions...")
    hedger = DeepHedger(
        K=K, T=T, r=r, sigma=sigma_calibrated, cost_rate=cost_rate,
        hidden_dim=32, option_type="call", lr=0.005, alpha=0.05
    )
    loss_history = hedger.fit(gbm_train, epochs=40, batch_size=256, verbose=True)

    # -------------------------------------------------------------
    # 4. Strategy Benchmarking & Evaluation Across Test Environments
    # -------------------------------------------------------------
    print("\n[Step 4] Running Discrete Hedging Engine & Computing Risk Metrics...")

    results = []

    # Environment A: Normal GBM Test Set (No Frictions)
    bs_deltas_nofric = compute_black_scholes_deltas(gbm_test, K, T, r, sigma_calibrated, "call")
    pnl_bs_nofric = simulate_hedging_pnl(gbm_test, bs_deltas_nofric, K, T, r, cost_rate=0.0, sigma_for_premium=sigma_calibrated)
    results.append(evaluate_hedging_performance(pnl_bs_nofric, bs_deltas_nofric, "Black-Scholes (Zero Friction)"))

    # Environment B: Normal GBM Test Set (With 10 bps Friction)
    pnl_bs_fric = simulate_hedging_pnl(gbm_test, bs_deltas_nofric, K, T, r, cost_rate=cost_rate, sigma_for_premium=sigma_calibrated)
    results.append(evaluate_hedging_performance(pnl_bs_fric, bs_deltas_nofric, "Black-Scholes (With 10bps Cost)"))

    dh_deltas_gbm = hedger.predict_deltas(gbm_test)
    pnl_dh_fric = simulate_hedging_pnl(gbm_test, dh_deltas_gbm, K, T, r, cost_rate=cost_rate, sigma_for_premium=sigma_calibrated)
    results.append(evaluate_hedging_performance(pnl_dh_fric, dh_deltas_gbm, "Deep Hedger (With 10bps Cost)"))

    # Environment C: Stress-Based Jump Diffusion (Out-of-Distribution Test)
    bs_deltas_stress = compute_black_scholes_deltas(stress_paths, K, T, r, sigma_calibrated, "call")
    pnl_bs_stress = simulate_hedging_pnl(stress_paths, bs_deltas_stress, K, T, r, cost_rate=cost_rate, sigma_for_premium=sigma_calibrated)
    results.append(evaluate_hedging_performance(pnl_bs_stress, bs_deltas_stress, "Black-Scholes (Stress Jumps)"))

    dh_deltas_stress = hedger.predict_deltas(stress_paths)
    pnl_dh_stress = simulate_hedging_pnl(stress_paths, dh_deltas_stress, K, T, r, cost_rate=cost_rate, sigma_for_premium=sigma_calibrated)
    results.append(evaluate_hedging_performance(pnl_dh_stress, dh_deltas_stress, "Deep Hedger (Stress Jumps)"))

    # Environment D: Adversarial Scenario Test
    bs_deltas_adv = compute_black_scholes_deltas(adv_paths, K, T, r, sigma_calibrated, "call")
    pnl_bs_adv = simulate_hedging_pnl(adv_paths, bs_deltas_adv, K, T, r, cost_rate=cost_rate, sigma_for_premium=sigma_calibrated)
    results.append(evaluate_hedging_performance(pnl_bs_adv, bs_deltas_adv, "Black-Scholes (Adversarial)"))

    dh_deltas_adv = hedger.predict_deltas(adv_paths)
    pnl_dh_adv = simulate_hedging_pnl(adv_paths, dh_deltas_adv, K, T, r, cost_rate=cost_rate, sigma_for_premium=sigma_calibrated)
    results.append(evaluate_hedging_performance(pnl_dh_adv, dh_deltas_adv, "Deep Hedger (Adversarial)"))

    # Summary Table
    df_results = compare_strategies(results)
    print("\n" + "=" * 80)
    print("                     COMPREHENSIVE PERFORMANCE BENCHMARK")
    print("=" * 80)
    print(df_results.to_string(index=False))

    # -------------------------------------------------------------
    # 5. Visualizations
    # -------------------------------------------------------------
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    plot_pnl_comparison(
        pnl_bs_fric["net_pnl"],
        pnl_dh_fric["net_pnl"],
        alpha=0.05,
        title="P&L Distribution: Black-Scholes vs. Deep Hedger (GBM with Frictions)",
        save_path=os.path.join(results_dir, "pnl_distribution_comparison.png")
    )

    plot_sample_paths(
        gbm_test,
        bs_deltas_nofric,
        dh_deltas_gbm,
        K=K,
        num_sample_paths=3,
        save_path=os.path.join(results_dir, "sample_hedging_paths.png")
    )

    # -------------------------------------------------------------
    # 6. Quantitative Metrics for Resume
    # -------------------------------------------------------------
    bs_cvar = abs(results[1]["CVaR (5%)"])
    dh_cvar = abs(results[2]["CVaR (5%)"])
    cvar_reduction = ((bs_cvar - dh_cvar) / bs_cvar) * 100.0 if bs_cvar > 0 else 0.0

    bs_cost = results[1]["Mean Cost"]
    dh_cost = results[2]["Mean Cost"]
    cost_reduction = ((bs_cost - dh_cost) / bs_cost) * 100.0 if bs_cost > 0 else 0.0

    bs_stress_cvar = abs(results[3]["CVaR (5%)"])
    dh_stress_cvar = abs(results[4]["CVaR (5%)"])
    stress_cvar_red = ((bs_stress_cvar - dh_stress_cvar) / bs_stress_cvar) * 100.0 if bs_stress_cvar > 0 else 0.0

    print("\n" + "=" * 80)
    print("                 MEASURED QUANTITATIVE RESUME METRICS")
    print("=" * 80)
    print(f"• CVaR (95%) Reduction on Standard Market Paths: {cvar_reduction:.1f}%")
    print(f"• Transaction Cost Drag Reduction: {cost_reduction:.1f}%")
    print(f"• Downside CVaR (95%) Reduction under Stress/Jump Regimes: {stress_cvar_red:.1f}%")
    print("=" * 80)

    return df_results


if __name__ == "__main__":
    run_full_simulation()
