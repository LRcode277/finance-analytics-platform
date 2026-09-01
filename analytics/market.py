import numpy as np
import pandas as pd
import yfinance as yf

TRADING_DAYS = 252


def safe_divide(a, b):
    try:
        if pd.isna(a) or pd.isna(b) or float(b) == 0:
            return np.nan
        return float(a) / float(b)
    except Exception:
        return np.nan


def normalize_datetime_index(obj):
    obj = obj.copy()
    if not isinstance(obj.index, pd.DatetimeIndex):
        obj.index = pd.to_datetime(obj.index)
    if obj.index.tz is not None:
        obj.index = obj.index.tz_localize(None)
    return obj


def trailing_return(prices, trading_days):
    if len(prices) <= trading_days:
        return np.nan
    return prices.iloc[-1] / prices.iloc[-trading_days - 1] - 1


def calculate_market_metrics(history):
    history = normalize_datetime_index(history)
    prices = history["Close"].dropna()
    if len(prices) < 2:
        raise ValueError("Insufficient price history.")
    returns = prices.pct_change().dropna()
    years = (prices.index[-1] - prices.index[0]).days / 365.25
    total_return = prices.iloc[-1] / prices.iloc[0] - 1
    cagr = (prices.iloc[-1] / prices.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    volatility = returns.std(ddof=1) * np.sqrt(TRADING_DAYS)
    running_max = prices.cummax()
    drawdown = prices / running_max - 1
    return {
        "prices": prices, "returns": returns, "drawdown": drawdown,
        "current_price": prices.iloc[-1],
        "return_1m": trailing_return(prices, 21), "return_3m": trailing_return(prices, 63),
        "return_6m": trailing_return(prices, 126), "return_1y": trailing_return(prices, 252),
        "total_return": total_return, "cagr": cagr, "volatility": volatility,
        "max_drawdown": drawdown.min(),
    }


def _download_close(symbol, start=None, end=None, period=None):
    kwargs = {"auto_adjust": True, "progress": False}
    if period:
        kwargs["period"] = period
    else:
        kwargs.update({"start": start, "end": end})
    try:
        data = yf.download(symbol, **kwargs)
    except Exception:
        return pd.Series(dtype=float)
    if data is None or data.empty or "Close" not in data:
        return pd.Series(dtype=float)
    close = data["Close"]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:, 0]
    close = normalize_datetime_index(close.to_frame("value"))["value"]
    return close.dropna().astype(float)


def current_risk_free_rate(default=0.04):
    """Approximate annual risk-free rate using the latest 13-week T-bill (^IRX)."""
    irx = _download_close("^IRX", period="1mo")
    if irx.empty:
        return default
    rate = float(irx.iloc[-1]) / 100.0
    return rate if 0 <= rate <= 0.20 else default


def benchmark_metrics(history):
    history = normalize_datetime_index(history)
    prices = history["Close"].dropna().astype(float)
    start_date = prices.index[0].strftime("%Y-%m-%d")
    end_date = (prices.index[-1] + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    benchmark_close = _download_close("^GSPC", start=start_date, end=end_date)
    risk_free_rate = current_risk_free_rate()

    if benchmark_close.empty:
        return {"benchmark_prices": benchmark_close, "beta": np.nan, "sharpe": np.nan,
                "sortino": np.nan, "risk_free_rate": risk_free_rate}

    prices = normalize_datetime_index(prices.to_frame("stock_price"))["stock_price"]
    stock_returns = prices.pct_change().rename("stock")
    market_returns = benchmark_close.pct_change().rename("market")
    combined = pd.concat([stock_returns, market_returns], axis=1).dropna()

    beta = np.nan
    if len(combined) > 2:
        market_variance = combined["market"].var(ddof=1)
        if not pd.isna(market_variance) and market_variance > 0:
            beta = combined["stock"].cov(combined["market"]) / market_variance

    r = stock_returns.dropna()
    if len(r) < 2:
        return {"benchmark_prices": benchmark_close, "beta": beta, "sharpe": np.nan,
                "sortino": np.nan, "risk_free_rate": risk_free_rate}

    # Geometric annualized return is consistent with the CAGR displayed elsewhere.
    years = (prices.index[-1] - prices.index[0]).days / 365.25
    annual_return = (prices.iloc[-1] / prices.iloc[0]) ** (1 / years) - 1 if years > 0 else np.nan
    volatility = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    sharpe = safe_divide(annual_return - risk_free_rate, volatility)

    # Sortino: downside deviation relative to a daily minimum acceptable return (MAR).
    daily_mar = (1 + risk_free_rate) ** (1 / TRADING_DAYS) - 1
    shortfall = np.minimum(r - daily_mar, 0.0)
    downside_deviation = np.sqrt(np.mean(np.square(shortfall))) * np.sqrt(TRADING_DAYS)
    sortino = safe_divide(annual_return - risk_free_rate, downside_deviation)

    return {"benchmark_prices": benchmark_close, "beta": beta, "sharpe": sharpe,
            "sortino": sortino, "risk_free_rate": risk_free_rate}
