"""Finnhub fallback adapter for hosted deployments.

Uses endpoints available on the standard API where possible: company profile 2,
basic financials, peers and recommendation trends. Premium estimate/price-target
endpoints are attempted safely but an unavailable-plan response is treated as
missing data, so the app continues with Alpha Vantage/Yahoo fallbacks.
"""
import os
from functools import lru_cache

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://finnhub.io/api/v1"
HEADERS = {"User-Agent": "FinanceAnalyticsPlatform/1.0"}


def get_api_key():
    key = os.getenv("FINNHUB_API_KEY", "").strip()
    if key:
        return key
    try:
        import streamlit as st
        return str(st.secrets.get("FINNHUB_API_KEY", "")).strip()
    except Exception:
        return ""


def is_available():
    return bool(get_api_key())


@lru_cache(maxsize=1024)
def _get(path, symbol, extra_items, api_key):
    params = {"symbol": symbol, "token": api_key}
    params.update(dict(extra_items))
    try:
        r = requests.get(BASE_URL + path, params=params, headers=HEADERS, timeout=12)
        if r.status_code in (401, 403, 429):
            return {}
        r.raise_for_status()
        data = r.json()
        # Premium-plan errors commonly arrive as JSON objects with an error key.
        if isinstance(data, dict) and data.get("error"):
            return {}
        return data
    except Exception:
        return {}


def request(path, symbol, **params):
    key = get_api_key()
    if not key:
        return {}
    return _get(path, symbol.upper().strip(), tuple(sorted(params.items())), key)


def _num(v):
    try:
        if v in (None, "", "N/A", "-"):
            return np.nan
        return float(v)
    except Exception:
        return np.nan


def _first(mapping, *keys):
    for key in keys:
        v = _num((mapping or {}).get(key))
        if not pd.isna(v):
            return v
    return np.nan


def get_profile_info(symbol):
    p = request("/stock/profile2", symbol)
    if not isinstance(p, dict) or not p:
        return {}
    out = {
        "symbol": p.get("ticker"),
        "longName": p.get("name"),
        "shortName": p.get("name"),
        "industry": p.get("finnhubIndustry"),
        "currency": p.get("currency"),
        "exchange": p.get("exchange"),
    }
    # Finnhub profile2 market cap is reported in millions.
    mc = _num(p.get("marketCapitalization"))
    if not pd.isna(mc):
        out["marketCap"] = mc * 1_000_000
    return {k: v for k, v in out.items() if v not in (None, "")}


def get_metric_info(symbol):
    payload = request("/stock/metric", symbol, metric="all")
    m = payload.get("metric", {}) if isinstance(payload, dict) else {}
    if not m:
        return {}

    # Finnhub's metric names evolve; aliases make this adapter resilient.
    aliases = {
        "trailingPE": ("peTTM", "peBasicExclExtraTTM", "peNormalizedAnnual"),
        "forwardPE": ("forwardPE", "peExclExtraAnnual"),
        "priceToSalesTrailing12Months": ("psTTM", "psAnnual"),
        "priceToBook": ("pbAnnual", "pbQuarterly"),
        "enterpriseToEbitda": ("evEbitdaTTM", "evEbitdaAnnual"),
        "enterpriseToRevenue": ("evSalesTTM", "evSalesAnnual"),
        "trailingPegRatio": ("pegTTM", "pegAnnual"),
        "beta": ("beta",),
        "returnOnEquity": ("roeTTM", "roeRfy"),
        "returnOnAssets": ("roaTTM", "roaRfy"),
        "grossMargins": ("grossMarginTTM", "grossMarginAnnual"),
        "operatingMargins": ("operatingMarginTTM", "operatingMarginAnnual"),
        "profitMargins": ("netProfitMarginTTM", "netProfitMarginAnnual"),
        "revenueGrowth": ("revenueGrowthTTMYoy", "revenueGrowth3Y", "revenueGrowth5Y"),
        "earningsGrowth": ("epsGrowthTTMYoy", "epsGrowth3Y", "epsGrowth5Y"),
        "currentRatio": ("currentRatioAnnual", "currentRatioQuarterly"),
    }
    out = {}
    percent_targets = {
        "returnOnEquity", "returnOnAssets", "grossMargins", "operatingMargins",
        "profitMargins", "revenueGrowth", "earningsGrowth",
    }
    for target, keys in aliases.items():
        v = _first(m, *keys)
        if pd.isna(v):
            continue
        # Finnhub margin/growth/return metrics are percentage points; app expects decimals.
        out[target] = v / 100.0 if target in percent_targets else v

    dy = _first(m, "currentDividendYieldTTM", "dividendYieldIndicatedAnnual")
    if not pd.isna(dy):
        out["dividendYield"] = dy / 100.0
    return out


def get_info(symbol):
    out = get_profile_info(symbol)
    for k, v in get_metric_info(symbol).items():
        if out.get(k) in (None, "", "N/A"):
            out[k] = v
    return out


def get_peers(symbol, limit=10):
    data = request("/stock/peers", symbol, grouping="industry")
    if not isinstance(data, list):
        return []
    cleaned = []
    for item in data:
        t = str(item).upper().strip()
        if t and t not in cleaned:
            cleaned.append(t)
    return cleaned[:limit]


def get_recommendations(symbol):
    rows = request("/stock/recommendation", symbol)
    if not isinstance(rows, list) or not rows:
        return pd.DataFrame(), {}
    df = pd.DataFrame(rows)
    wanted = [c for c in ["period", "strongBuy", "buy", "hold", "sell", "strongSell"] if c in df.columns]
    display = df[wanted].copy() if wanted else df.copy()
    latest = rows[0]
    counts = {k: int(latest.get(k, 0) or 0) for k in ["strongBuy", "buy", "hold", "sell", "strongSell"]}
    total = sum(counts.values())
    # Map aggregate analyst votes to the same vocabulary yfinance uses.
    positive = counts["strongBuy"] + counts["buy"]
    negative = counts["sell"] + counts["strongSell"]
    if total == 0:
        key = "N/A"
    elif positive / total >= 0.65:
        key = "buy"
    elif negative / total >= 0.45:
        key = "sell"
    elif positive > negative:
        key = "hold"
    else:
        key = "underperform"
    return display, {"recommendationKey": key, "numberOfAnalystOpinions": total}


def get_price_targets(symbol):
    # Finnhub documents this endpoint as Premium. Safe attempt lets paid keys work
    # automatically; free keys simply fall through to Alpha Vantage mean target.
    p = request("/stock/price-target", symbol)
    if not isinstance(p, dict) or not p:
        return {}, {}
    targets = {
        "low": _num(p.get("targetLow")),
        "mean": _num(p.get("targetMean")),
        "median": _num(p.get("targetMedian")),
        "high": _num(p.get("targetHigh")),
    }
    targets = {k: v for k, v in targets.items() if not pd.isna(v)}
    meta = {"numberOfAnalystOpinions": _num(p.get("numberAnalysts"))}
    return targets, meta


def _estimate_table(payload, prefix):
    rows = payload.get("data", []) if isinstance(payload, dict) else []
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "period" in df.columns:
        df = df.set_index("period")
    return df


def get_estimate_tables(symbol):
    # These are Premium on Finnhub. They are useful automatically for a paid key,
    # while Alpha Vantage remains the fallback on free deployments.
    rev = request("/stock/revenue-estimate", symbol, freq="annual")
    eps = request("/stock/eps-estimate", symbol, freq="annual")
    return _estimate_table(rev, "revenue"), _estimate_table(eps, "eps")
