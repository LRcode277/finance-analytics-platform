import numpy as np
import pandas as pd


def safe_divide(a, b):
    try:
        if pd.isna(a) or pd.isna(b) or float(b) == 0:
            return np.nan
        return float(a) / float(b)
    except Exception:
        return np.nan


def value_from_statement(statement, names, column=0):
    if statement is None or statement.empty:
        return np.nan
    for name in names:
        if name in statement.index:
            try:
                row = statement.loc[name]
                value = row.iloc[column] if hasattr(row, "iloc") else row
                if not pd.isna(value):
                    return float(value)
            except Exception:
                pass
    return np.nan


def historical_row(statement, names):
    if statement is None or statement.empty:
        return pd.Series(dtype=float)
    for name in names:
        if name in statement.index:
            try:
                series = statement.loc[name].dropna()
                series.index = pd.to_datetime(series.index)
                return series.astype(float).sort_index()
            except Exception:
                pass
    return pd.Series(dtype=float)


def average_balance(statement, names):
    """Average of the two most recent balance-sheet observations."""
    series = historical_row(statement, names)
    if series.empty:
        return np.nan
    recent = series.sort_index(ascending=False).iloc[:2]
    return float(recent.mean()) if not recent.empty else np.nan


def cagr_from_series(series):
    series = series.dropna().sort_index()
    if len(series) < 2:
        return np.nan
    first, last = float(series.iloc[0]), float(series.iloc[-1])
    # CAGR is not economically meaningful when endpoints are <= 0.
    if first <= 0 or last <= 0:
        return np.nan
    try:
        years = (pd.Timestamp(series.index[-1]) - pd.Timestamp(series.index[0])).days / 365.25
    except Exception:
        years = len(series) - 1
    if years <= 0:
        return np.nan
    return (last / first) ** (1 / years) - 1


def _normalized_tax_rate(tax_provision, pretax_income):
    rate = safe_divide(tax_provision, pretax_income)
    # Keep a reported effective rate when plausible; otherwise use a clearly
    # documented normalized fallback rather than allowing extreme one-offs.
    if pd.isna(rate) or rate < 0 or rate > 0.50:
        return 0.21
    return rate


def calculate_fundamentals(income, balance, cashflow):
    revenue = value_from_statement(income, ["Total Revenue"])
    gross_profit = value_from_statement(income, ["Gross Profit"])
    ebitda = value_from_statement(income, ["EBITDA", "Normalized EBITDA"])
    ebit = value_from_statement(income, ["EBIT", "Operating Income"])
    operating_income = value_from_statement(income, ["Operating Income", "EBIT"])
    net_income = value_from_statement(income, ["Net Income", "Net Income Common Stockholders"])
    pretax_income = value_from_statement(income, ["Pretax Income"])
    tax_provision = value_from_statement(income, ["Tax Provision"])
    interest_raw = value_from_statement(income, ["Interest Expense", "Interest Expense Non Operating"])
    interest_expense = abs(interest_raw) if not pd.isna(interest_raw) else np.nan
    eps = value_from_statement(income, ["Diluted EPS", "Basic EPS"])

    total_assets = value_from_statement(balance, ["Total Assets"])
    equity = value_from_statement(balance, ["Stockholders Equity", "Total Stockholder Equity"])
    debt = value_from_statement(balance, ["Total Debt"])
    cash = value_from_statement(balance, ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents", "Cash"])
    current_assets = value_from_statement(balance, ["Current Assets", "Total Current Assets"])
    current_liabilities = value_from_statement(balance, ["Current Liabilities", "Total Current Liabilities"])
    inventory = value_from_statement(balance, ["Inventory"])

    operating_cash_flow = value_from_statement(cashflow, ["Operating Cash Flow", "Total Cash From Operating Activities"])
    capex_raw = value_from_statement(cashflow, ["Capital Expenditure", "Capital Expenditures"])
    # Yahoo commonly reports CapEx as a negative cash outflow. Store a positive
    # expenditure amount so FCF = CFO - CapEx is unambiguous.
    capex = abs(capex_raw) if not pd.isna(capex_raw) else np.nan
    provider_fcf = value_from_statement(cashflow, ["Free Cash Flow"])
    calculated_fcf = (
        operating_cash_flow - capex
        if not pd.isna(operating_cash_flow) and not pd.isna(capex)
        else np.nan
    )
    fcf = calculated_fcf if not pd.isna(calculated_fcf) else provider_fcf

    net_debt = debt - cash if not pd.isna(debt) and not pd.isna(cash) else np.nan
    effective_tax_rate = _normalized_tax_rate(tax_provision, pretax_income)
    nopat = ebit * (1 - effective_tax_rate) if not pd.isna(ebit) else np.nan

    avg_equity = average_balance(balance, ["Stockholders Equity", "Total Stockholder Equity"])
    avg_assets = average_balance(balance, ["Total Assets"])
    avg_debt = average_balance(balance, ["Total Debt"])
    avg_cash = average_balance(balance, ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents", "Cash"])

    invested_capital = (
        avg_debt + avg_equity - avg_cash
        if all(not pd.isna(x) for x in [avg_debt, avg_equity, avg_cash])
        else np.nan
    )

    # ROE uses net income; ROA below is an operating ROA, consistent with an
    # enterprise-level return measure: NOPAT / average total assets.
    roe = safe_divide(net_income, avg_equity) if not pd.isna(avg_equity) and avg_equity > 0 else np.nan
    roa = safe_divide(nopat, avg_assets) if not pd.isna(avg_assets) and avg_assets > 0 else np.nan
    roic = safe_divide(nopat, invested_capital) if not pd.isna(invested_capital) and invested_capital > 0 else np.nan

    revenue_history = historical_row(income, ["Total Revenue"])
    ebitda_history = historical_row(income, ["EBITDA", "Normalized EBITDA"])
    ebit_history = historical_row(income, ["EBIT", "Operating Income"])
    net_income_history = historical_row(income, ["Net Income", "Net Income Common Stockholders"])
    eps_history = historical_row(income, ["Diluted EPS", "Basic EPS"])
    fcf_history = historical_row(cashflow, ["Free Cash Flow"])
    cash_history = historical_row(balance, ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents", "Cash"])
    debt_history = historical_row(balance, ["Total Debt"])

    return {
        "revenue": revenue, "gross_profit": gross_profit, "ebitda": ebitda,
        "ebit": ebit, "operating_income": operating_income, "net_income": net_income,
        "eps": eps, "fcf": fcf, "provider_fcf": provider_fcf,
        "operating_cash_flow": operating_cash_flow, "capex": capex,
        "cash": cash, "debt": debt, "net_debt": net_debt, "equity": equity,
        "average_equity": avg_equity, "average_assets": avg_assets,
        "invested_capital": invested_capital, "nopat": nopat,
        "effective_tax_rate": effective_tax_rate,
        "gross_margin": safe_divide(gross_profit, revenue),
        "ebitda_margin": safe_divide(ebitda, revenue),
        "ebit_margin": safe_divide(ebit, revenue),
        "operating_margin": safe_divide(operating_income, revenue),
        "net_margin": safe_divide(net_income, revenue),
        "fcf_margin": safe_divide(fcf, revenue),
        "roe": roe, "roa": roa, "roic": roic,
        "debt_to_equity": safe_divide(debt, equity) if not pd.isna(equity) and equity > 0 else np.nan,
        "net_debt_to_ebitda": safe_divide(net_debt, ebitda) if not pd.isna(ebitda) and ebitda > 0 else np.nan,
        "current_ratio": safe_divide(current_assets, current_liabilities),
        "quick_ratio": safe_divide((current_assets - (0 if pd.isna(inventory) else inventory)) if not pd.isna(current_assets) else np.nan, current_liabilities),
        "interest_coverage": safe_divide(ebit, interest_expense),
        "revenue_cagr": cagr_from_series(revenue_history),
        "eps_cagr": cagr_from_series(eps_history),
        "ebitda_cagr": cagr_from_series(ebitda_history),
        "fcf_cagr": cagr_from_series(fcf_history),
        "revenue_history": revenue_history, "ebitda_history": ebitda_history,
        "ebit_history": ebit_history, "net_income_history": net_income_history,
        "eps_history": eps_history, "fcf_history": fcf_history,
        "cash_history": cash_history, "debt_history": debt_history,
    }
