import numpy as np
import pandas as pd


def valid(x):
    try:
        return x is not None and not pd.isna(x) and np.isfinite(float(x))
    except Exception:
        return False


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(x)))


def linear(value, bad, good, higher_better=True):
    if not valid(value) or bad == good:
        return np.nan
    x = (float(value) - bad) / (good - bad)
    # bad maps to 0 and good maps to 1; this works whether good is
    # numerically above or below bad.
    return clamp(x)


def average_available(values):
    vals = [float(v) for v in values if valid(v)]
    return float(np.mean(vals)) if vals else np.nan


def weighted_component(items, max_points):
    vals = [v for v in items if valid(v)]
    if not vals:
        return np.nan
    return round(average_available(vals) * max_points, 1)


def _peer_relative_score(relative_summary, metric):
    item = (relative_summary or {}).get(metric, {})
    premium = item.get("premium_discount", np.nan)
    # 40%+ discount -> 1.0; equal to peers -> 0.60; 60%+ premium -> 0.0.
    if not valid(premium):
        return np.nan
    p = float(premium) / 100.0
    if p <= 0:
        return clamp(0.60 + (-p / 0.40) * 0.40)
    return clamp(0.60 - (p / 0.60) * 0.60)


def asset_scoring_status(info):
    quote_type = str(info.get("quoteType", "")).upper()
    sector = str(info.get("sector", ""))
    industry = str(info.get("industry", "")).lower()
    if quote_type in {"ETF", "MUTUALFUND", "INDEX"}:
        return False, "Fund/ETF scoring requires a portfolio-level methodology, so the corporate score is disabled."
    if sector == "Financial Services" or any(k in industry for k in ["bank", "insurance", "credit services"]):
        return False, "Financial institutions require sector-specific capital and valuation metrics; the generic corporate score is disabled."
    if sector == "Real Estate" or "reit" in industry:
        return False, "REITs should be assessed with FFO/AFFO and property-specific metrics; the generic corporate score is disabled."
    return True, ""


def calculate_platform_score(info, fundamentals, valuation, market, benchmark, estimates, relative_summary=None):
    enabled, reason = asset_scoring_status(info)
    if not enabled:
        return {"available": False, "reason": reason, "score": np.nan, "rating": "N/A",
                "components": {}, "strengths": [], "risks": []}

    f, v, m, b, e = fundamentals, valuation, market, benchmark, estimates

    valuation_score = weighted_component([
        _peer_relative_score(relative_summary, "P/E"),
        _peer_relative_score(relative_summary, "Forward P/E"),
        _peer_relative_score(relative_summary, "EV / EBITDA"),
        _peer_relative_score(relative_summary, "EV / Revenue"),
        linear(v.get("fcf_yield"), 0.00, 0.08, True),
    ], 25)

    growth_score = weighted_component([
        linear(f.get("revenue_cagr"), -0.02, 0.15, True),
        linear(f.get("eps_cagr"), -0.05, 0.20, True),
        linear(f.get("ebitda_cagr"), -0.03, 0.18, True),
        linear(f.get("fcf_cagr"), -0.05, 0.20, True),
    ], 20)

    quality_score = weighted_component([
        linear(f.get("roic"), 0.03, 0.25, True),
        linear(f.get("roe"), 0.05, 0.30, True),
        linear(f.get("operating_margin"), 0.03, 0.30, True),
        linear(f.get("fcf_margin"), 0.02, 0.25, True),
    ], 20)

    health_score = weighted_component([
        linear(f.get("net_debt_to_ebitda"), 4.0, 0.0, False) if valid(f.get("net_debt_to_ebitda")) else (1.0 if valid(f.get("net_debt")) and f.get("net_debt") < 0 else np.nan),
        linear(f.get("interest_coverage"), 1.5, 12.0, True),
        linear(f.get("current_ratio"), 0.8, 2.0, True),
    ], 15)

    risk_score = weighted_component([
        linear(m.get("volatility"), 0.55, 0.15, False),
        linear(abs(m.get("max_drawdown")) if valid(m.get("max_drawdown")) else np.nan, 0.60, 0.15, False),
        linear(b.get("beta"), 1.8, 0.7, False),
    ], 10)

    recommendation_map = {"strong_buy": 1.0, "buy": 0.85, "outperform": 0.80,
                          "hold": 0.55, "neutral": 0.50, "underperform": 0.25,
                          "sell": 0.15, "strong_sell": 0.0}
    rec = str(e.get("recommendation", "")).lower().replace(" ", "_")
    analyst_score = weighted_component([
        recommendation_map.get(rec, np.nan),
        linear(e.get("upside_mean"), -0.20, 0.30, True),
    ], 10)

    components = {"Valuation": valuation_score, "Growth": growth_score,
                  "Quality": quality_score, "Financial Health": health_score,
                  "Risk": risk_score, "Wall Street": analyst_score}
    available = {k: x for k, x in components.items() if valid(x)}
    maxes = {"Valuation": 25, "Growth": 20, "Quality": 20, "Financial Health": 15, "Risk": 10, "Wall Street": 10}
    if not available:
        return {"available": False, "reason": "Insufficient data to calculate a reliable score.",
                "score": np.nan, "rating": "N/A", "components": components, "strengths": [], "risks": []}

    earned = sum(available.values())
    possible = sum(maxes[k] for k in available)
    score = round(earned / possible * 100)
    rating = "ATTRACTIVE" if score >= 75 else "ABOVE AVERAGE" if score >= 65 else "NEUTRAL" if score >= 50 else "CAUTIOUS" if score >= 35 else "UNATTRACTIVE"

    strengths, risks = [], []
    if valid(f.get("roic")) and f["roic"] >= 0.15: strengths.append("Strong return on invested capital")
    if valid(f.get("fcf_margin")) and f["fcf_margin"] >= 0.12: strengths.append("Strong free-cash-flow generation")
    if valid(f.get("revenue_cagr")) and f["revenue_cagr"] >= 0.10: strengths.append("Strong historical revenue growth")
    if valid(f.get("net_debt")) and f["net_debt"] < 0: strengths.append("Net cash balance sheet")
    if valid(f.get("net_debt_to_ebitda")) and f["net_debt_to_ebitda"] > 3: risks.append("Elevated net debt relative to EBITDA")
    if valid(m.get("volatility")) and m["volatility"] > 0.40: risks.append("High historical share-price volatility")
    if valid(m.get("max_drawdown")) and m["max_drawdown"] < -0.45: risks.append("Large historical maximum drawdown")
    evrel = (relative_summary or {}).get("EV / EBITDA", {}).get("premium_discount", np.nan)
    if valid(evrel) and evrel > 30: risks.append("EV/EBITDA trades at a substantial premium to selected peers")
    if valid(e.get("upside_mean")) and e["upside_mean"] < -0.05: risks.append("Mean analyst target is below the current price")

    return {"available": True, "reason": "", "score": score, "rating": rating,
            "components": components, "component_max": maxes,
            "strengths": strengths[:4], "risks": risks[:4],
            "methodology": "Rules-based quantitative score; missing components are reweighted rather than treated as zero."}
