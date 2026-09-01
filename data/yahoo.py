import time

import pandas as pd
import yfinance as yf


# ---------------------------------------------------------
# YFINANCE NETWORK CONFIGURATION
# ---------------------------------------------------------

# Retry transient Yahoo/network failures.
# Supported by recent versions of yfinance.
try:
    yf.config.network.retries = 3
except Exception:
    pass


# ---------------------------------------------------------
# HELPERS
# ---------------------------------------------------------

def safe_call(func, default=None, retries=2, delay=0.8):
    """
    Execute a yfinance request with a small retry mechanism.

    This is particularly useful on hosted environments where
    Yahoo endpoints can temporarily fail or rate-limit requests.
    """

    for attempt in range(retries + 1):
        try:
            result = func()

            if result is not None:
                return result

        except Exception:
            pass

        if attempt < retries:
            time.sleep(delay * (attempt + 1))

    return default


def safe_info(stock):
    """
    Retrieve Yahoo company metadata.

    get_info() is attempted first. The property .info is kept
    as a fallback for compatibility.
    """

    info = safe_call(
        stock.get_info,
        default=None,
        retries=2,
    )

    if isinstance(info, dict) and info:
        return info

    info = safe_call(
        lambda: stock.info,
        default={},
        retries=1,
    )

    if isinstance(info, dict):
        return info

    return {}


def safe_fast_info(stock):
    """
    Retrieve fast market information and convert it to a
    normal dictionary.
    """

    fast = safe_call(
        stock.get_fast_info,
        default=None,
        retries=2,
    )

    if fast is None:
        fast = safe_call(
            lambda: stock.fast_info,
            default=None,
            retries=1,
        )

    if fast is None:
        return {}

    try:
        return dict(fast)
    except Exception:
        return {}


def safe_dataframe(func):
    result = safe_call(
        func,
        default=pd.DataFrame(),
        retries=1,
    )

    if isinstance(result, pd.DataFrame):
        return result

    return pd.DataFrame()


def safe_dict(func):
    result = safe_call(
        func,
        default={},
        retries=1,
    )

    if isinstance(result, dict):
        return result

    return {}


def merge_fast_info_into_info(info, fast_info):
    """
    Fill selected market fields when quote-summary metadata
    is unavailable but Yahoo's faster price endpoint works.
    """

    info = dict(info or {})
    fast_info = fast_info or {}

    mapping = {
        "marketCap": "market_cap",
        "currentPrice": "last_price",
        "previousClose": "previous_close",
        "open": "open",
        "dayHigh": "day_high",
        "dayLow": "day_low",
        "fiftyTwoWeekHigh": "year_high",
        "fiftyTwoWeekLow": "year_low",
        "sharesOutstanding": "shares",
    }

    for info_key, fast_key in mapping.items():

        if info.get(info_key) is None:

            value = fast_info.get(fast_key)

            if value is not None:
                info[info_key] = value

    return info


# ---------------------------------------------------------
# MAIN DATA LOADER
# ---------------------------------------------------------

def get_stock_data(ticker, period="5y"):

    ticker = ticker.upper().strip()

    stock = yf.Ticker(ticker)

    # -----------------------------------------------------
    # PRICE HISTORY
    # -----------------------------------------------------

    history = safe_call(
        lambda: stock.history(
            period=period,
            auto_adjust=True,
            repair=True,
        ),
        default=pd.DataFrame(),
        retries=2,
    )

    if not isinstance(history, pd.DataFrame) or history.empty:
        raise ValueError(
            f"No price data found for {ticker}."
        )

    # -----------------------------------------------------
    # COMPANY / MARKET INFORMATION
    # -----------------------------------------------------

    fast_info = safe_fast_info(stock)
    info = safe_info(stock)

    # fast_info can still provide market cap / price data
    # when Yahoo's quote-summary endpoint is unavailable.
    info = merge_fast_info_into_info(
        info,
        fast_info,
    )

    # -----------------------------------------------------
    # FINANCIAL STATEMENTS
    # -----------------------------------------------------

    income = safe_dataframe(
        lambda: stock.get_income_stmt(
            freq="yearly"
        )
    )

    balance = safe_dataframe(
        lambda: stock.get_balance_sheet(
            freq="yearly"
        )
    )

    cashflow = safe_dataframe(
        lambda: stock.get_cash_flow(
            freq="yearly"
        )
    )

    quarterly_income = safe_dataframe(
        lambda: stock.get_income_stmt(
            freq="quarterly"
        )
    )

    # -----------------------------------------------------
    # WALL STREET / ANALYST DATA
    # -----------------------------------------------------

    recommendations = safe_dataframe(
        stock.get_recommendations
    )

    recommendations_summary = safe_dataframe(
        stock.get_recommendations_summary
    )

    analyst_price_targets = safe_dict(
        stock.get_analyst_price_targets
    )

    revenue_estimate = safe_dataframe(
        stock.get_revenue_estimate
    )

    earnings_estimate = safe_dataframe(
        stock.get_earnings_estimate
    )

    growth_estimates = safe_dataframe(
        stock.get_growth_estimates
    )

    # -----------------------------------------------------
    # RETURN NORMALIZED DATA
    # -----------------------------------------------------

    return {
        "ticker": ticker,
        "stock": stock,
        "history": history,
        "info": info,
        "fast_info": fast_info,
        "income": income,
        "balance": balance,
        "cashflow": cashflow,
        "quarterly_income": quarterly_income,
        "recommendations": recommendations,
        "recommendations_summary": recommendations_summary,
        "analyst_price_targets": analyst_price_targets,
        "revenue_estimate": revenue_estimate,
        "earnings_estimate": earnings_estimate,
        "growth_estimates": growth_estimates,
    }
