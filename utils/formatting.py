import numpy as np
import pandas as pd


def is_valid(value):
    try:
        return (
            value is not None
            and not pd.isna(value)
            and np.isfinite(float(value))
        )
    except Exception:
        return False


def number(value, decimals=2):
    if not is_valid(value):
        return "N/A"

    return f"{float(value):,.{decimals}f}"


def percent(value, decimals=2):
    if not is_valid(value):
        return "N/A"

    return f"{float(value) * 100:.{decimals}f}%"


def money(value, symbol="$"):
    if not is_valid(value):
        return "N/A"

    value = float(value)

    if abs(value) >= 1e12:
        return f"{symbol}{value / 1e12:.2f}T"

    if abs(value) >= 1e9:
        return f"{symbol}{value / 1e9:.2f}B"

    if abs(value) >= 1e6:
        return f"{symbol}{value / 1e6:.2f}M"

    return f"{symbol}{value:,.2f}"


def multiple(value):
    if not is_valid(value):
        return "N/A"

    return f"{float(value):.2f}x"


def statement_table(df):
    if df is None or df.empty:
        return pd.DataFrame()

    result = df.copy()

    result.columns = [
        c.strftime("%Y-%m-%d")
        if hasattr(c, "strftime")
        else str(c)
        for c in result.columns
    ]

    return result / 1_000_000
