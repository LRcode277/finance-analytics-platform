"""Alpha Vantage fallback adapter.

Keeps the rest of the application provider-agnostic by translating Alpha
Vantage responses into the Yahoo/yfinance-shaped dictionaries and DataFrames
already consumed by the analytics layer.
"""
import os
import time
from functools import lru_cache

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://www.alphavantage.co/query"
HEADERS = {"User-Agent": "FinanceAnalyticsPlatform/1.0"}


def get_api_key():
    """Read the key from Streamlit Secrets or an environment variable."""
    key = os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()
    if key:
        return key
    try:
        import streamlit as st
        return str(st.secrets.get("ALPHA_VANTAGE_API_KEY", "")).strip()
    except Exception:
        return ""


def is_available():
    return bool(get_api_key())


@lru_cache(maxsize=512)
def _request_cached(function, symbol, api_key):
    try:
        response = requests.get(
            BASE_URL,
            params={"function": function, "symbol": symbol, "apikey": api_key},
            headers=HEADERS,
            timeout=12,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            return {}
        # Alpha Vantage returns these keys when throttled or when an endpoint is
        # unavailable for the current plan. Treat them as missing fallback data.
        if any(k in data for k in ("Error Message", "Information", "Note")):
            return {}
        return data
    except Exception:
        return {}


def request(function, symbol):
    key = get_api_key()
    if not key:
        return {}
    return _request_cached(function, symbol.upper().strip(), key)


def _num(value):
    if value in (None, "", "None", "-", "N/A"):
        return np.nan
    try:
        return float(value)
    except Exception:
        return np.nan


def _ratio(value):
    value = _num(value)
    return value if not pd.isna(value) else np.nan


def overview_to_yahoo(overview):
    """Translate OVERVIEW fields to keys used by the existing app."""
    if not overview:
        return {}
    mapping = {
        "Symbol": "symbol", "Name": "longName", "AssetType": "quoteType",
        "Sector": "sector", "Industry": "industry", "Currency": "currency",
        "MarketCapitalization": "marketCap", "EnterpriseValue": "enterpriseValue",
        "PERatio": "trailingPE", "ForwardPE": "forwardPE", "PEGRatio": "trailingPegRatio",
        "PriceToSalesRatioTTM": "priceToSalesTrailing12Months", "PriceToBookRatio": "priceToBook",
        "EVToRevenue": "enterpriseToRevenue", "EVToEBITDA": "enterpriseToEbitda",
        "DividendYield": "dividendYield", "RevenueTTM": "totalRevenue",
        "RevenueGrowthTTM": "revenueGrowth", "QuarterlyRevenueGrowthYOY": "revenueGrowth",
        "QuarterlyEarningsGrowthYOY": "earningsGrowth", "GrossProfitTTM": "grossProfits",
        "ProfitMargin": "profitMargins", "OperatingMarginTTM": "operatingMargins",
        "ReturnOnEquityTTM": "returnOnEquity", "ReturnOnAssetsTTM": "returnOnAssets",
        "AnalystTargetPrice": "targetMeanPrice", "Beta": "beta",
        "EPS": "trailingEps", "DilutedEPSTTM": "trailingEps",
        "52WeekHigh": "fiftyTwoWeekHigh", "52WeekLow": "fiftyTwoWeekLow",
    }
    numeric_targets = {
        "marketCap", "enterpriseValue", "trailingPE", "forwardPE", "trailingPegRatio",
        "priceToSalesTrailing12Months", "priceToBook", "enterpriseToRevenue",
        "enterpriseToEbitda", "dividendYield", "totalRevenue", "revenueGrowth",
        "earningsGrowth", "grossProfits", "profitMargins", "operatingMargins",
        "returnOnEquity", "returnOnAssets", "targetMeanPrice", "beta", "trailingEps",
        "fiftyTwoWeekHigh", "fiftyTwoWeekLow",
    }
    out = {}
    for source, target in mapping.items():
        value = overview.get(source)
        if value in (None, "", "None", "-"):
            continue
        out[target] = _num(value) if target in numeric_targets else value
    if out.get("longName"):
        out["shortName"] = out["longName"]
    return out


def get_overview_info(symbol):
    return overview_to_yahoo(request("OVERVIEW", symbol))


# Alpha Vantage field -> yfinance statement row label
INCOME_FIELDS = {
    "totalRevenue": "Total Revenue", "grossProfit": "Gross Profit",
    "ebitda": "EBITDA", "ebit": "EBIT", "operatingIncome": "Operating Income",
    "netIncome": "Net Income", "incomeBeforeTax": "Pretax Income",
    "incomeTaxExpense": "Tax Provision", "interestExpense": "Interest Expense",
}
BALANCE_FIELDS = {
    "totalAssets": "Total Assets", "totalShareholderEquity": "Stockholders Equity",
    "shortLongTermDebtTotal": "Total Debt", "longTermDebt": "Long Term Debt",
    "cashAndShortTermInvestments": "Cash Cash Equivalents And Short Term Investments",
    "cashAndCashEquivalentsAtCarryingValue": "Cash And Cash Equivalents",
    "totalCurrentAssets": "Current Assets", "totalCurrentLiabilities": "Current Liabilities",
    "inventory": "Inventory",
}
CASH_FIELDS = {
    "operatingCashflow": "Operating Cash Flow", "capitalExpenditures": "Capital Expenditure",
}


def _reports_to_statement(payload, report_key, field_map):
    reports = payload.get(report_key, []) if isinstance(payload, dict) else []
    if not reports:
        return pd.DataFrame()
    columns = {}
    for report in reports:
        date = pd.to_datetime(report.get("fiscalDateEnding"), errors="coerce")
        if pd.isna(date):
            continue
        values = {}
        for source, label in field_map.items():
            value = _num(report.get(source))
            if not pd.isna(value):
                values[label] = value
        columns[date] = values
    if not columns:
        return pd.DataFrame()
    df = pd.DataFrame(columns).sort_index(axis=1, ascending=False)
    return df


def get_statements(symbol):
    income_raw = request("INCOME_STATEMENT", symbol)
    balance_raw = request("BALANCE_SHEET", symbol)
    cash_raw = request("CASH_FLOW", symbol)
    income = _reports_to_statement(income_raw, "annualReports", INCOME_FIELDS)
    quarterly_income = _reports_to_statement(income_raw, "quarterlyReports", INCOME_FIELDS)
    balance = _reports_to_statement(balance_raw, "annualReports", BALANCE_FIELDS)
    cashflow = _reports_to_statement(cash_raw, "annualReports", CASH_FIELDS)

    # Derive FCF in the same sign convention expected by fundamentals.py.
    if not cashflow.empty and "Operating Cash Flow" in cashflow.index and "Capital Expenditure" in cashflow.index:
        cfo = cashflow.loc["Operating Cash Flow"]
        capex = cashflow.loc["Capital Expenditure"].abs()
        cashflow.loc["Free Cash Flow"] = cfo - capex
    return income, balance, cashflow, quarterly_income


def get_estimate_tables(symbol):
    """Return display-friendly revenue, earnings and growth estimate tables."""
    payload = request("EARNINGS_ESTIMATES", symbol)
    rows = payload.get("estimates", []) if isinstance(payload, dict) else []
    if not rows:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}

    period_col = next((c for c in ["horizon", "fiscalDateEnding", "date"] if c in df.columns), None)
    if period_col is None:
        df["Period"] = range(1, len(df) + 1)
        period_col = "Period"

    def table(cols):
        available = [period_col] + [c for c in cols if c in df.columns]
        out = df[available].copy()
        for c in available[1:]:
            out[c] = pd.to_numeric(out[c], errors="coerce")
        return out.set_index(period_col)

    revenue = table(["revenueEstimateAverage", "revenueEstimateHigh", "revenueEstimateLow", "revenueEstimateAnalystCount"])
    earnings = table(["epsEstimateAverage", "epsEstimateHigh", "epsEstimateLow", "epsEstimateAnalystCount"])
    growth = table(["revenueEstimateGrowth", "epsEstimateGrowth"])

    analyst_count = np.nan
    for c in ["epsEstimateAnalystCount", "revenueEstimateAnalystCount"]:
        if c in df.columns:
            vals = pd.to_numeric(df[c], errors="coerce").dropna()
            if not vals.empty:
                analyst_count = float(vals.iloc[0])
                break
    meta = {"analyst_count": analyst_count}
    return revenue, earnings, growth, meta
