"""
Black-Scholes Analytical Pricing, Greeks Calculation, and Implied Volatility Solver.
"""

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq


def black_scholes_price(S, K, T, r, sigma, option_type="call"):
    """
    Calculates the analytical Black-Scholes price for European options.

    Args:
        S (float or np.ndarray): Current underlying spot/futures price.
        K (float or np.ndarray): Strike price.
        T (float or np.ndarray): Time to maturity in years (T > 0).
        r (float): Risk-free interest rate (annualized).
        sigma (float or np.ndarray): Volatility (annualized).
        option_type (str): 'call' or 'put'.

    Returns:
        float or np.ndarray: Theoretical option price.
    """
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)

    # Handle expired or zero-time options cleanly
    if np.any(T <= 1e-7):
        if option_type.lower() == "call":
            payoff = np.maximum(0.0, S - K)
        else:
            payoff = np.maximum(0.0, K - S)
        if np.isscalar(T) or T.ndim == 0:
            return float(payoff)
        return payoff

    sigma = np.maximum(sigma, 1e-6)
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if option_type.lower() == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type.lower() == "put":
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError(f"Invalid option_type: {option_type}. Must be 'call' or 'put'.")

    return price if isinstance(price, np.ndarray) and price.ndim > 0 else float(price)


def black_scholes_greeks(S, K, T, r, sigma, option_type="call"):
    """
    Calculates analytical Black-Scholes Greeks (Delta, Gamma, Vega, Theta).

    Returns:
        dict: {'delta': ..., 'gamma': ..., 'vega': ..., 'theta': ...}
    """
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    sigma = np.maximum(np.asarray(sigma, dtype=np.float64), 1e-6)

    is_expired = T <= 1e-7
    # For expired contracts:
    # Delta is step function at moneyness
    if np.all(is_expired):
        if option_type.lower() == "call":
            delta = np.where(S > K, 1.0, np.where(S == K, 0.5, 0.0))
        else:
            delta = np.where(S < K, -1.0, np.where(S == K, -0.5, 0.0))
        zeros = np.zeros_like(S)
        return {"delta": delta, "gamma": zeros, "vega": zeros, "theta": zeros}

    safe_T = np.where(is_expired, 1e-7, T)
    sqrt_T = np.sqrt(safe_T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * safe_T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    pdf_d1 = norm.pdf(d1)

    if option_type.lower() == "call":
        delta = norm.cdf(d1)
        theta = -(S * pdf_d1 * sigma) / (2 * sqrt_T) - r * K * np.exp(-r * safe_T) * norm.cdf(d2)
    elif option_type.lower() == "put":
        delta = norm.cdf(d1) - 1.0
        theta = -(S * pdf_d1 * sigma) / (2 * sqrt_T) + r * K * np.exp(-r * safe_T) * norm.cdf(-d2)
    else:
        raise ValueError(f"Invalid option_type: {option_type}")

    gamma = pdf_d1 / (S * sigma * sqrt_T)
    vega = S * pdf_d1 * sqrt_T  # 1% move: divide by 100 in practice if needed

    # Fix edge case for expired
    if np.any(is_expired):
        if option_type.lower() == "call":
            delta = np.where(is_expired, np.where(S > K, 1.0, 0.0), delta)
        else:
            delta = np.where(is_expired, np.where(S < K, -1.0, 0.0), delta)
        gamma = np.where(is_expired, 0.0, gamma)
        vega = np.where(is_expired, 0.0, vega)
        theta = np.where(is_expired, 0.0, theta)

    return {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}


def implied_volatility(market_price, S, K, T, r, option_type="call", low_vol=1e-4, high_vol=5.0):
    """
    Computes Implied Volatility (IV) using robust Brent's method with bisection fallback.

    Args:
        market_price (float): Observed option market price.
        S (float): Current underlying asset price.
        K (float): Strike price.
        T (float): Time to maturity in years.
        r (float): Risk-free interest rate.
        option_type (str): 'call' or 'put'.

    Returns:
        float: Solved implied volatility (or np.nan if no solution within bounds).
    """
    if T <= 1e-7 or market_price <= 0:
        return np.nan

    # Minimum intrinsic value check
    intrinsic = max(0.0, S - K) if option_type.lower() == "call" else max(0.0, K - S)
    if market_price < intrinsic:
        return np.nan

    def objective(sigma):
        return black_scholes_price(S, K, T, r, sigma, option_type) - market_price

    f_low = objective(low_vol)
    f_high = objective(high_vol)

    if f_low * f_high > 0:
        # Price is outside [low_vol, high_vol] range
        if f_low > 0:
            return low_vol
        return high_vol

    try:
        iv = brentq(objective, low_vol, high_vol, xtol=1e-6, maxiter=100)
        return float(iv)
    except Exception:
        # Fallback to simple bisection
        lo, hi = low_vol, high_vol
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            f_mid = objective(mid)
            if abs(f_mid) < 1e-5:
                return float(mid)
            if f_low * f_mid < 0:
                hi = mid
            else:
                lo = mid
                f_low = f_mid
        return float(0.5 * (lo + hi))
