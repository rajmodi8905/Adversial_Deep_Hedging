"""
Quantitative Risk & Hedging Performance Metrics.
"""

import numpy as np
import pandas as pd


def calculate_var(pnl_array: np.ndarray, alpha: float = 0.05) -> float:
    """
    Calculates Value at Risk (VaR) at confidence level alpha.
    Convention: Return the quantile of P&L (e.g., negative number indicating loss).
    """
    return float(np.percentile(pnl_array, alpha * 100))


def calculate_cvar(pnl_array: np.ndarray, alpha: float = 0.05) -> float:
    """
    Calculates Conditional Value at Risk (CVaR / Expected Shortfall) at confidence level alpha.
    CVaR is the expectation of P&L conditional on P&L being <= VaR_alpha.
    """
    var_val = calculate_var(pnl_array, alpha)
    tail = pnl_array[pnl_array <= var_val]
    if len(tail) == 0:
        return var_val
    return float(np.mean(tail))


def calculate_turnover(deltas: np.ndarray) -> float:
    """
    Computes average portfolio turnover per path: sum_{k} |delta_{k} - delta_{k-1}|.
    """
    # Start from 0 to delta_0, then delta_k - delta_{k-1}, then delta_{N-1} to 0
    init_turn = np.abs(deltas[:, 0])
    inter_turn = np.sum(np.abs(deltas[:, 1:-1] - deltas[:, :-2]), axis=1)
    final_turn = np.abs(deltas[:, -2])
    total_turn = init_turn + inter_turn + final_turn
    return float(np.mean(total_turn))


def evaluate_hedging_performance(
    pnl_dict: dict,
    deltas: np.ndarray,
    strategy_name: str = "Strategy",
    alpha: float = 0.05
) -> dict:
    """
    Computes a comprehensive dictionary of risk and execution metrics.
    """
    net_pnl = pnl_dict["net_pnl"]
    costs = pnl_dict["total_costs"]

    mean_pnl = float(np.mean(net_pnl))
    std_pnl = float(np.std(net_pnl))  # Hedging error
    var_alpha = calculate_var(net_pnl, alpha=alpha)
    cvar_alpha = calculate_cvar(net_pnl, alpha=alpha)
    worst_pnl = float(np.min(net_pnl))
    mean_cost = float(np.mean(costs))
    turnover = calculate_turnover(deltas)

    return {
        "Strategy": strategy_name,
        "Mean P&L": mean_pnl,
        "Hedging Error (Std)": std_pnl,
        f"VaR ({int(alpha*100)}%)": var_alpha,
        f"CVaR ({int(alpha*100)}%)": cvar_alpha,
        "Mean Cost": mean_cost,
        "Avg Turnover": turnover,
        "Worst P&L": worst_pnl,
    }


def compare_strategies(metrics_list: list) -> pd.DataFrame:
    """
    Generates a structured comparison DataFrame from multiple strategy metrics.
    """
    df = pd.DataFrame(metrics_list)
    return df
