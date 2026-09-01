import numpy as np
import pandas as pd
import yfinance as yf

from data.alpha_vantage import get_overview_info, is_available as alpha_available
from data.finnhub import get_info as finnhub_info, get_peers as finnhub_peers, is_available as finnhub_available


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def safe_value(value):
    """Return NaN when Yahoo returns missing/invalid data."""
    if value is None:
        return np.nan

    try:
        if pd.isna(value):
            return np.nan
    except Exception:
        pass

    return value


def to_billions(value):
    """Convert raw dollar values to billions."""
    value = safe_value(value)

    if pd.isna(value):
        return np.nan

    return float(value) / 1_000_000_000


def to_percent(value):
    """Convert Yahoo decimal percentages to percentage points."""
    value = safe_value(value)

    if pd.isna(value):
        return np.nan

    return float(value) * 100


# ---------------------------------------------------------
# Peer universe
# ---------------------------------------------------------

SECTOR_PEERS = {
    "Technology": [
        "AAPL",
        "MSFT",
        "NVDA",
        "GOOGL",
        "META",
        "AVGO",
        "ORCL",
        "CRM",
        "ADBE",
    ],

    "Communication Services": [
        "GOOGL",
        "META",
        "NFLX",
        "DIS",
        "TMUS",
        "VZ",
        "T",
    ],

    "Consumer Cyclical": [
        "AMZN",
        "TSLA",
        "HD",
        "MCD",
        "NKE",
        "LOW",
        "SBUX",
    ],

    "Consumer Defensive": [
        "WMT",
        "COST",
        "PG",
        "KO",
        "PEP",
        "PM",
        "MO",
    ],

    "Financial Services": [
        "JPM",
        "BAC",
        "GS",
        "MS",
        "WFC",
        "C",
        "BLK",
    ],

    "Healthcare": [
        "LLY",
        "JNJ",
        "UNH",
        "ABBV",
        "MRK",
        "PFE",
        "TMO",
    ],

    "Industrials": [
        "GE",
        "CAT",
        "HON",
        "RTX",
        "UPS",
        "BA",
        "DE",
    ],

    "Energy": [
        "XOM",
        "CVX",
        "COP",
        "SLB",
        "EOG",
        "OXY",
    ],

    "Utilities": [
        "NEE",
        "SO",
        "DUK",
        "AEP",
        "SRE",
    ],

    "Real Estate": [
        "PLD",
        "AMT",
        "EQIX",
        "O",
        "SPG",
    ],
}


# ---------------------------------------------------------
# Determine peers
# ---------------------------------------------------------

def get_peer_tickers(ticker, info, max_peers=6):
    ticker = ticker.upper()

    sector = info.get("sector")

    # Prefer live industry peers from Finnhub on hosted deployments.
    peer_list = finnhub_peers(ticker, limit=max_peers + 3) if finnhub_available() else []
    if not peer_list:
        peer_list = SECTOR_PEERS.get(sector, []).copy()

    # Always include the company being analyzed
    if ticker in peer_list:
        peer_list.remove(ticker)

    peer_list.insert(0, ticker)

    return peer_list[:max_peers]


# ---------------------------------------------------------
# Download one company's metrics
# ---------------------------------------------------------

def get_company_metrics(ticker):
    ticker = ticker.upper()

    try:
        stock = yf.Ticker(ticker)
        try:
            info = stock.info or {}
        except Exception:
            info = {}

        # Finnhub basic financials/profile are cloud-friendly and fill Yahoo gaps.
        if finnhub_available():
            fallback = finnhub_info(ticker)
            for key, value in fallback.items():
                if info.get(key) in (None, "", "N/A"):
                    info[key] = value

        # Streamlit Cloud can receive empty Yahoo quote-summary responses.
        # Use one Alpha Vantage OVERVIEW call only when the core peer metrics
        # are absent. The provider adapter is cached for the process lifetime.
        core_keys = ("marketCap", "trailingPE", "totalRevenue")
        if alpha_available() and not any(info.get(k) is not None for k in core_keys):
            fallback = get_overview_info(ticker)
            for key, value in fallback.items():
                if info.get(key) in (None, "", "N/A"):
                    info[key] = value

        company_name = (
            info.get("shortName")
            or info.get("longName")
            or ticker
        )

        price = (
            info.get("currentPrice")
            or info.get("regularMarketPrice")
        )
        if price is None:
            try:
                fast = dict(stock.fast_info)
                price = fast.get("last_price") or fast.get("lastPrice")
                if info.get("marketCap") is None:
                    info["marketCap"] = fast.get("market_cap") or fast.get("marketCap")
            except Exception:
                pass

        gross_margin = info.get("grossMargins")
        if gross_margin is None:
            gp, rev = info.get("grossProfits"), info.get("totalRevenue")
            try:
                gross_margin = float(gp) / float(rev) if float(rev) else None
            except Exception:
                gross_margin = None

        return {
            "Company": company_name,
            "Ticker": ticker,

            "Price": safe_value(price),

            "Market Cap": to_billions(
                info.get("marketCap")
            ),

            "Revenue": to_billions(
                info.get("totalRevenue")
            ),

            "Revenue Growth": to_percent(
                info.get("revenueGrowth")
            ),

            "Earnings Growth": to_percent(
                info.get("earningsGrowth")
            ),

            "Gross Margin": to_percent(
                gross_margin
            ),

            "Operating Margin": to_percent(
                info.get("operatingMargins")
            ),

            "Net Margin": to_percent(
                info.get("profitMargins")
            ),

            "ROE": to_percent(
                info.get("returnOnEquity")
            ),

            "ROA": to_percent(
                info.get("returnOnAssets")
            ),

            "P/E": safe_value(
                info.get("trailingPE")
            ),

            "Forward P/E": safe_value(
                info.get("forwardPE")
            ),

            "Price / Sales": safe_value(
                info.get("priceToSalesTrailing12Months")
            ),

            "Price / Book": safe_value(
                info.get("priceToBook")
            ),

            "EV / Revenue": safe_value(
                info.get("enterpriseToRevenue")
            ),

            "EV / EBITDA": safe_value(
                info.get("enterpriseToEbitda")
            ),

            "PEG": safe_value(
                info.get("pegRatio")
            ),
        }

    except Exception:
        return {
            "Company": ticker,
            "Ticker": ticker,
        }


# ---------------------------------------------------------
# Build comparison table
# ---------------------------------------------------------

def build_peer_comparison(ticker, info, max_peers=6):
    peer_tickers = get_peer_tickers(
        ticker=ticker,
        info=info,
        max_peers=max_peers
    )

    rows = []

    for peer in peer_tickers:
        data = get_company_metrics(peer)
        rows.append(data)

    df = pd.DataFrame(rows)

    columns = [
        "Company",
        "Ticker",
        "Price",
        "Market Cap",
        "Revenue",
        "Revenue Growth",
        "Earnings Growth",
        "Gross Margin",
        "Operating Margin",
        "Net Margin",
        "ROE",
        "ROA",
        "P/E",
        "Forward P/E",
        "Price / Sales",
        "Price / Book",
        "EV / Revenue",
        "EV / EBITDA",
        "PEG",
    ]

    for column in columns:
        if column not in df.columns:
            df[column] = np.nan

    return df[columns]


# ---------------------------------------------------------
# Sector median
# ---------------------------------------------------------

def calculate_peer_medians(peer_df):
    numeric_columns = peer_df.select_dtypes(
        include=[np.number]
    ).columns

    medians = {}

    for column in numeric_columns:
        values = peer_df[column].dropna()

        if len(values) > 0:
            medians[column] = values.median()
        else:
            medians[column] = np.nan

    return medians


# ---------------------------------------------------------
# Relative valuation
# ---------------------------------------------------------

def relative_valuation_summary(peer_df, ticker):
    ticker = ticker.upper()

    company_rows = peer_df[
        peer_df["Ticker"] == ticker
    ]

    competitors = peer_df[
        peer_df["Ticker"] != ticker
    ]

    if company_rows.empty or competitors.empty:
        return {}

    company = company_rows.iloc[0]

    metrics = [
        "P/E",
        "Forward P/E",
        "Price / Sales",
        "EV / Revenue",
        "EV / EBITDA",
    ]

    results = {}

    for metric in metrics:
        company_value = company.get(metric)

        peer_values = (
            competitors[metric]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        if (
            pd.isna(company_value)
            or peer_values.empty
        ):
            results[metric] = {
                "company": np.nan,
                "peer_median": np.nan,
                "premium_discount": np.nan,
            }

            continue

        peer_median = peer_values.median()

        if peer_median == 0:
            premium_discount = np.nan
        else:
            premium_discount = (
                company_value / peer_median - 1
            ) * 100

        results[metric] = {
            "company": company_value,
            "peer_median": peer_median,
            "premium_discount": premium_discount,
        }

    return results
