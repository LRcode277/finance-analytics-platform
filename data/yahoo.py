import pandas as pd
import yfinance as yf


def safe_info(stock):
    try:
        return stock.info or {}
    except Exception:
        return {}


def safe_fast_info(stock):
    try:
        return dict(stock.fast_info)
    except Exception:
        return {}


def safe_dataframe(func):
    try:
        df = func()
        if isinstance(df, pd.DataFrame):
            return df
    except Exception:
        pass

    return pd.DataFrame()


def get_stock_data(ticker, period="5y"):
    ticker = ticker.upper().strip()

    stock = yf.Ticker(ticker)

    history = stock.history(
        period=period,
        auto_adjust=True
    )

    if history.empty:
        raise ValueError(
            f"No price data found for {ticker}."
        )

    info = safe_info(stock)
    fast_info = safe_fast_info(stock)

    try:
        income = stock.financials
    except Exception:
        income = pd.DataFrame()

    try:
        balance = stock.balance_sheet
    except Exception:
        balance = pd.DataFrame()

    try:
        cashflow = stock.cashflow
    except Exception:
        cashflow = pd.DataFrame()

    try:
        quarterly_income = stock.quarterly_financials
    except Exception:
        quarterly_income = pd.DataFrame()

    try:
        recommendations = stock.recommendations
    except Exception:
        recommendations = pd.DataFrame()

    try:
        recommendations_summary = stock.recommendations_summary
    except Exception:
        recommendations_summary = pd.DataFrame()

    try:
        analyst_price_targets = stock.analyst_price_targets
        if analyst_price_targets is None:
            analyst_price_targets = {}
    except Exception:
        analyst_price_targets = {}

    try:
        revenue_estimate = stock.revenue_estimate
    except Exception:
        revenue_estimate = pd.DataFrame()

    try:
        earnings_estimate = stock.earnings_estimate
    except Exception:
        earnings_estimate = pd.DataFrame()

    try:
        growth_estimates = stock.growth_estimates
    except Exception:
        growth_estimates = pd.DataFrame()

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
