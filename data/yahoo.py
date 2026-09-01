import pandas as pd
import requests
import yfinance as yf


YAHOO_SEARCH_URL = "https://query1.finance.yahoo.com/v1/finance/search"
HEADERS = {"User-Agent": "Mozilla/5.0"}


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


def yahoo_search_fallback(ticker):
    """Use Yahoo's lightweight search endpoint to recover basic metadata.

    This endpoint is already used by the app's company search and often remains
    available when Yahoo's quote-summary endpoint (used by Ticker.info) is
    unavailable on hosted environments.
    """
    try:
        response = requests.get(
            YAHOO_SEARCH_URL,
            params={
                "q": ticker,
                "quotesCount": 10,
                "newsCount": 0,
                "enableFuzzyQuery": False,
            },
            headers=HEADERS,
            timeout=6,
        )
        response.raise_for_status()
        quotes = response.json().get("quotes", [])
    except Exception:
        return {}

    ticker_upper = ticker.upper()
    quote = next(
        (q for q in quotes if str(q.get("symbol", "")).upper() == ticker_upper),
        None,
    )
    if not quote:
        return {}

    # Translate search-response names to the keys expected by the rest of app.
    candidates = {
        "symbol": quote.get("symbol"),
        "longName": quote.get("longname") or quote.get("shortname"),
        "shortName": quote.get("shortname") or quote.get("longname"),
        "quoteType": quote.get("quoteType"),
        "sector": quote.get("sectorDisp") or quote.get("sector"),
        "industry": quote.get("industryDisp") or quote.get("industry"),
        "marketCap": quote.get("marketCap"),
        "currency": quote.get("currency"),
        "exchange": quote.get("exchange"),
        "fullExchangeName": quote.get("exchDisp"),
    }
    return {k: v for k, v in candidates.items() if v is not None}


def merge_missing(base, fallback):
    result = dict(base or {})
    for key, value in (fallback or {}).items():
        if result.get(key) in (None, "", "N/A"):
            result[key] = value
    return result


def merge_fast_info(info, fast_info):
    # fast_info keys vary slightly by yfinance version, so support both forms.
    aliases = {
        "marketCap": ("market_cap", "marketCap"),
        "currentPrice": ("last_price", "lastPrice"),
        "previousClose": ("previous_close", "previousClose"),
        "fiftyTwoWeekHigh": ("year_high", "yearHigh"),
        "fiftyTwoWeekLow": ("year_low", "yearLow"),
    }
    result = dict(info or {})
    for target, source_keys in aliases.items():
        if result.get(target) is not None:
            continue
        for source in source_keys:
            value = (fast_info or {}).get(source)
            if value is not None:
                result[target] = value
                break
    return result


def get_stock_data(ticker, period="5y"):
    ticker = ticker.upper().strip()
    stock = yf.Ticker(ticker)

    # IMPORTANT: keep the original, proven history path. Do not wrap this call
    # in a broad retry helper because API/version errors must not become a false
    # "No price data" result.
    history = stock.history(period=period, auto_adjust=True)

    if history.empty:
        raise ValueError(f"No price data found for {ticker}.")

    # Primary yfinance metadata + independent lightweight fallbacks.
    info = safe_info(stock)
    fast_info = safe_fast_info(stock)
    info = merge_fast_info(info, fast_info)
    info = merge_missing(info, yahoo_search_fallback(ticker))

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
