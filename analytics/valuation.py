import numpy as np
import pandas as pd


def info_value(info, key):
    value = info.get(key)

    if value is None:
        return np.nan

    try:
        return float(value)
    except Exception:
        return np.nan


def safe_divide(a, b):
    try:
        if pd.isna(a) or pd.isna(b) or b == 0:
            return np.nan

        return float(a) / float(b)

    except Exception:
        return np.nan


def calculate_valuation(
    info,
    fundamentals
):

    market_cap = info_value(
        info,
        "marketCap"
    )

    enterprise_value = info_value(
        info,
        "enterpriseValue"
    )

    trailing_pe = info_value(
        info,
        "trailingPE"
    )

    forward_pe = info_value(
        info,
        "forwardPE"
    )

    price_to_sales = info_value(
        info,
        "priceToSalesTrailing12Months"
    )

    price_to_book = info_value(
        info,
        "priceToBook"
    )

    peg = info_value(
        info,
        "trailingPegRatio"
    )

    ev_revenue = safe_divide(
        enterprise_value,
        fundamentals["revenue"]
    )

    ev_ebitda = safe_divide(
        enterprise_value,
        fundamentals["ebitda"]
    )

    ev_ebit = safe_divide(
        enterprise_value,
        fundamentals["ebit"]
    )

    price_fcf = safe_divide(
        market_cap,
        fundamentals["fcf"]
    )

    fcf_yield = safe_divide(
        fundamentals["fcf"],
        market_cap
    )

    earnings_yield = safe_divide(
        1,
        trailing_pe
    )

    dividend_yield = info_value(
        info,
        "dividendYield"
    )

    # yfinance versions/providers may expose this
    # field in different units. Sanity normalize.
    if (
        not pd.isna(dividend_yield)
        and dividend_yield > 0.20
    ):
        dividend_yield /= 100

    return {
        "market_cap": market_cap,
        "enterprise_value": enterprise_value,

        "trailing_pe": trailing_pe,
        "forward_pe": forward_pe,
        "peg": peg,

        "price_to_sales": price_to_sales,
        "price_to_book": price_to_book,

        "ev_revenue": ev_revenue,
        "ev_ebitda": ev_ebitda,
        "ev_ebit": ev_ebit,

        "price_fcf": price_fcf,

        "fcf_yield": fcf_yield,
        "earnings_yield": earnings_yield,
        "dividend_yield": dividend_yield,
    }
