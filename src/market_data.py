"""
Market Data Loader & Preprocessing for NSE NIFTY Options & Futures Minute Bars.
"""

import re
import numpy as np
import pandas as pd
from .black_scholes import implied_volatility, black_scholes_greeks


def parse_symbol(symbol: str):
    """
    Parses an NSE option/future symbol into structured components.
    Example:
        'NIFTY26FEBFUT' -> {'underlying': 'NIFTY', 'is_future': True, 'strike': None, 'type': 'FUT'}
        'NIFTY2621025550CE' -> {'underlying': 'NIFTY', 'is_future': False, 'strike': 25550.0, 'type': 'CE'}
        'NIFTY2621025550PE' -> {'underlying': 'NIFTY', 'is_future': False, 'strike': 25550.0, 'type': 'PE'}
    """
    symbol = str(symbol).strip()
    if symbol.endswith("FUT"):
        return {"underlying": "NIFTY", "is_future": True, "strike": None, "type": "FUT"}

    # Match strike and CE/PE
    m = re.search(r"(\d{5})(CE|PE)$", symbol)
    if m:
        strike = float(m.group(1))
        opt_type = m.group(2)
        return {"underlying": "NIFTY", "is_future": False, "strike": strike, "type": opt_type}

    return {"underlying": "NIFTY", "is_future": False, "strike": None, "type": "UNKNOWN"}


def load_nifty_minute_data(csv_path: str, scale_prices: bool = True):
    """
    Loads and cleans an NSE minute-bar CSV file.

    Prices in NSE raw feed are often quoted in paise (1/100 INR), e.g. 2575220 for 25752.20.
    If scale_prices=True and futures prices are > 100,000, scales by 1/100 to standard index points.

    Returns:
        pd.DataFrame: Cleaned dataframe with columns:
                      ['date', 'minute_end', 'symbol', 'price', 'is_future', 'strike', 'type']
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]

    price_col = "last_trade_price" if "last_trade_price" in df.columns else "price"

    # Scale from paise if needed
    sample_fut = df[df["symbol"].str.endswith("FUT")][price_col]
    if scale_prices and not sample_fut.empty and sample_fut.iloc[0] > 100000:
        scale_factor = 0.01
    else:
        scale_factor = 1.0

    df["price"] = df[price_col] * scale_factor

    # Parse symbols
    parsed = df["symbol"].apply(parse_symbol).apply(pd.Series)
    df = pd.concat([df, parsed], axis=1)

    # Convert minute_end to clean string / int
    df["minute_end"] = df["minute_end"].astype(str).str.zfill(6)
    return df


def get_options_snapshot(df: pd.DataFrame, minute_str: str = "110000", r: float = 0.07, days_to_expiry: float = 1.0):
    """
    Extracts an options chain snapshot at a specific minute (e.g. 11:00 AM),
    computes the underlying futures price, and solves for Implied Volatility & Greeks.

    Returns:
        spot_price (float), options_df (pd.DataFrame)
    """
    minute_df = df[df["minute_end"] == minute_str].copy()
    if minute_df.empty:
        # Pick the first available minute
        first_min = df["minute_end"].iloc[0]
        minute_df = df[df["minute_end"] == first_min].copy()

    fut_rows = minute_df[minute_df["is_future"] == True]
    if fut_rows.empty:
        # Fallback to ATM strike estimate
        spot_price = float(minute_df["strike"].dropna().median())
    else:
        spot_price = float(fut_rows["price"].iloc[0])

    options_df = minute_df[minute_df["is_future"] == False].copy()
    T = max(days_to_expiry / 365.0, 1e-4)

    # Compute IV and Greeks for each option in chain
    ivs = []
    deltas = []
    gammas = []
    thetas = []

    for _, row in options_df.iterrows():
        K = row["strike"]
        mkt_p = row["price"]
        otype = "call" if row["type"] == "CE" else "put"

        iv = implied_volatility(mkt_p, spot_price, K, T, r, option_type=otype)
        iv_clean = iv if not np.isnan(iv) else 0.15

        greeks = black_scholes_greeks(spot_price, K, T, r, iv_clean, option_type=otype)
        ivs.append(iv)
        deltas.append(greeks["delta"])
        gammas.append(greeks["gamma"])
        thetas.append(greeks["theta"])

    options_df["implied_vol"] = ivs
    options_df["bs_delta"] = deltas
    options_df["bs_gamma"] = gammas
    options_df["bs_theta"] = thetas
    options_df["underlying_price"] = spot_price
    options_df["T_years"] = T

    return spot_price, options_df
