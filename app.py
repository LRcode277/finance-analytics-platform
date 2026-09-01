import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from data.yahoo import get_stock_data

from analytics.search import search_companies

from analytics.market import (
    calculate_market_metrics,
    benchmark_metrics
)

from analytics.fundamentals import (
    calculate_fundamentals
)

from analytics.valuation import (
    calculate_valuation
)

from analytics.estimates import (
    calculate_estimates
)

from analytics.peers import (
    build_peer_comparison,
    calculate_peer_medians,
    relative_valuation_summary,
)

from analytics.scoring import (
    calculate_platform_score
)

from utils.formatting import (
    number,
    percent,
    money,
    multiple,
    statement_table,
    is_valid
)


# ======================================================
# CONFIGURATION
# ======================================================

st.set_page_config(
    page_title="Finance Analytics Platform",
    page_icon="📊",
    layout="wide"
)


# ======================================================
# CSS
# ======================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.18);
        padding: 14px;
        border-radius: 10px;
    }

    div[data-testid="stMetricLabel"] {
        font-size: 0.88rem;
    }

    .section-note {
        opacity: 0.65;
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ======================================================
# HELPERS
# ======================================================

def metric(label, value):
    st.metric(
        label,
        value
    )


def currency_symbol(currency):
    mapping = {
        "USD": "$",
        "EUR": "€",
        "GBP": "£",
        "CHF": "CHF "
    }

    return mapping.get(
        currency,
        f"{currency} "
    )


def financial_chart(
    title,
    series_dict,
    percentage=False
):

    fig = go.Figure()

    has_data = False

    for name, series in series_dict.items():

        if (
            series is None
            or len(series) == 0
        ):
            continue

        has_data = True

        values = series.values

        if percentage:
            values = values * 100

        fig.add_trace(
            go.Bar(
                x=series.index,
                y=values,
                name=name
            )
        )

    if not has_data:
        st.info(
            f"No historical data available for {title}."
        )
        return

    fig.update_layout(
        title=title,
        template="plotly_white",
        hovermode="x unified",
        height=420,
        barmode="group",
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


# ======================================================
# SIDEBAR
# ======================================================

with st.sidebar:

    st.title("Stock Research")

    # ---------------------------------
    # COMPANY / TICKER SEARCH
    # ---------------------------------

    search_query = st.text_input(
        "Search company or ticker",
        value="AAPL",
        placeholder="Apple, Microsoft, AAPL..."
    ).strip()

    selected_ticker = None

    if search_query:

        search_results = search_companies(
            search_query,
            max_results=8
        )

        if search_results:

            company_options = {
                result["label"]: result
                for result in search_results
            }

            selected_label = st.selectbox(
                "Company",
                options=list(company_options.keys())
            )

            selected_company = company_options[selected_label]

            selected_ticker = selected_company["ticker"]

            exchange = selected_company.get("exchange", "")

            if exchange:
                st.caption(
                    f"{selected_ticker} · {exchange}"
                )

        else:
            # Fallback if Yahoo search is unavailable
            selected_ticker = search_query.upper()

    # ---------------------------------
    # PRICE HISTORY
    # ---------------------------------

    period = st.selectbox(
        "Price history",
        [
            "1y",
            "3y",
            "5y",
            "10y"
        ],
        index=2
    )

    # ---------------------------------
    # ANALYZE
    # ---------------------------------

    analyze = st.button(
        "Analyze",
        type="primary",
        use_container_width=True
    )

    if analyze and selected_ticker:

        st.session_state["analysis_ticker"] = (
            selected_ticker
        )

    st.divider()

    st.caption(
        "Search by company name or ticker. "
        "Examples: Apple, Microsoft, NVDA, "
        "GOOGL, JPM, LLY"
    )


# ---------------------------------
# ACTIVE TICKER
# ---------------------------------

ticker = st.session_state.get(
    "analysis_ticker",
    selected_ticker
)

if not ticker:
    st.stop()


# ======================================================
# DATA ENGINE
# ======================================================

@st.cache_resource(
    ttl=900,
    show_spinner=False
)
def run_analysis(
    ticker_symbol,
    selected_period
):

    raw = get_stock_data(
        ticker_symbol,
        selected_period
    )

    market = calculate_market_metrics(
        raw["history"]
    )

    benchmark = benchmark_metrics(
        raw["history"]
    )

    fundamentals = calculate_fundamentals(
        raw["income"],
        raw["balance"],
        raw["cashflow"]
    )

    valuation = calculate_valuation(
        raw["info"],
        fundamentals
    )

    estimates = calculate_estimates(
        raw["info"],
        raw["analyst_price_targets"],
        market["current_price"]
    )

    return {
        "raw": raw,
        "market": market,
        "benchmark": benchmark,
        "fundamentals": fundamentals,
        "valuation": valuation,
        "estimates": estimates,
    }


try:

    with st.spinner(
        f"Analyzing {ticker}..."
    ):
        result = run_analysis(
            ticker,
            period
        )

except Exception as error:

    st.error(
        f"Could not analyze {ticker}."
    )

    st.code(str(error))

    st.stop()


raw = result["raw"]
market = result["market"]
benchmark = result["benchmark"]
fund = result["fundamentals"]
valuation = result["valuation"]
estimates = result["estimates"]

info = raw["info"]

company_name = info.get(
    "longName",
    info.get(
        "shortName",
        ticker
    )
)

sector = info.get(
    "sector",
    "N/A"
)

industry = info.get(
    "industry",
    "N/A"
)

quote_type = info.get(
    "quoteType",
    "N/A"
)

currency = info.get(
    "currency",
    "USD"
)

symbol = currency_symbol(
    currency
)


# ======================================================
# PEERS + PLATFORM SCORE
# ======================================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_peer_data(ticker_symbol, sector_name):
    return build_peer_comparison(
        ticker=ticker_symbol,
        info={"sector": sector_name},
        max_peers=6
    )

peer_df = load_peer_data(ticker, sector)
relative_valuation = relative_valuation_summary(
    peer_df, ticker
) if not peer_df.empty else {}

platform_score = calculate_platform_score(
    info=info,
    fundamentals=fund,
    valuation=valuation,
    market=market,
    benchmark=benchmark,
    estimates=estimates,
    relative_summary=relative_valuation
)


# ======================================================
# HEADER
# ======================================================

st.title(company_name)

header1, header2, header3, header4 = (
    st.columns(4)
)

header1.write(
    f"**Ticker:** {ticker}"
)

header2.write(
    f"**Sector:** {sector}"
)

header3.write(
    f"**Industry:** {industry}"
)

header4.write(
    f"**Asset:** {quote_type}"
)

st.caption(
    "Independent quantitative research dashboard. "
    "Platform ratings are rules-based research aids, not personalized investment recommendations."
)


# ======================================================
# HERO METRICS
# ======================================================

h1, h2, h3, h4, h5, h6 = (
    st.columns(6)
)

h1.metric(
    "Price",
    money(
        market["current_price"],
        symbol
    )
)

h2.metric(
    "Market Cap",
    money(
        valuation["market_cap"],
        symbol
    )
)

h3.metric(
    "5Y / Period CAGR",
    percent(
        market["cagr"]
    )
)

h4.metric(
    "1Y Return",
    percent(
        market["return_1y"]
    )
)

h5.metric(
    "Volatility",
    percent(
        market["volatility"]
    )
)

h6.metric(
    "Max Drawdown",
    percent(
        market["max_drawdown"]
    )
)


st.divider()


# ======================================================
# TABS
# ======================================================

(
    overview_tab,
    platform_tab,
    financials_tab,
    valuation_tab,
    risk_tab,
    analyst_tab,
    peers_tab,
    statements_tab
) = st.tabs(
    [
        "Overview",
        "Platform View",
        "Fundamentals",
        "Valuation",
        "Risk",
        "Wall Street",
        "Peers",
        "Statements"
    ]
)


# ======================================================
# OVERVIEW
# ======================================================

with overview_tab:

    st.subheader(
        "Price & Analyst Targets"
    )

    prices = market["prices"]

    last_date = prices.index[-1]
    current_price = prices.iloc[-1]

    target_date = (
        last_date
        + pd.DateOffset(years=1)
    )

    fig = go.Figure()

    # Historical price

    fig.add_trace(
        go.Scatter(
            x=prices.index,
            y=prices.values,
            mode="lines",
            name=ticker,
            line=dict(
                width=2.5
            ),
            hovertemplate=(
                "%{x|%d %b %Y}<br>"
                f"{symbol}"
                "%{y:.2f}"
                "<extra></extra>"
            )
        )
    )

    # Current price

    fig.add_trace(
        go.Scatter(
            x=[last_date],
            y=[current_price],
            mode="markers",
            name="Current",
            marker=dict(
                size=9
            )
        )
    )

    # ------------------------------------------
    # Analyst target projections
    # ------------------------------------------

    def target_line(
        price,
        label
    ):

        if not is_valid(price):
            return

        price = float(price)

        upside = (
            price / current_price - 1
        )

        fig.add_trace(
            go.Scatter(
                x=[
                    last_date,
                    target_date
                ],
                y=[
                    current_price,
                    price
                ],
                mode="lines",
                name=label,
                line=dict(
                    width=2,
                    dash="dash"
                ),
                hovertemplate=(
                    f"{label}<br>"
                    f"{symbol}{price:.2f}<br>"
                    f"{upside:+.1%}"
                    "<extra></extra>"
                )
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[target_date],
                y=[price],
                mode="markers+text",
                showlegend=False,
                marker=dict(
                    size=9
                ),
                text=[
                    f"{label}  "
                    f"{symbol}{price:.2f}"
                ],
                textposition="middle right",
                hovertemplate=(
                    f"{label}<br>"
                    f"{symbol}{price:.2f}<br>"
                    f"{upside:+.1%}"
                    "<extra></extra>"
                )
            )
        )

    target_line(
        estimates["high"],
        "High"
    )

    target_line(
        estimates["mean"],
        "Mean"
    )

    target_line(
        estimates["low"],
        "Low"
    )

    fig.add_vline(
        x=last_date.timestamp() * 1000,
        line_dash="dot",
        opacity=0.4
    )

    fig.update_layout(
        template="plotly_white",
        height=580,
        hovermode="x unified",
        xaxis_title=None,
        yaxis_title=f"Price ({currency})",
        margin=dict(
            l=20,
            r=150,
            t=20,
            b=20
        )
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.caption(
        "Dashed lines show available analyst "
        "price-target consensus extending from "
        "the latest market price. They are not "
        "forecasts generated by this platform."
    )

    # Performance

    st.subheader(
        "Performance"
    )

    p1, p2, p3, p4, p5 = (
        st.columns(5)
    )

    p1.metric(
        "1 Month",
        percent(
            market["return_1m"]
        )
    )

    p2.metric(
        "3 Months",
        percent(
            market["return_3m"]
        )
    )

    p3.metric(
        "6 Months",
        percent(
            market["return_6m"]
        )
    )

    p4.metric(
        "1 Year",
        percent(
            market["return_1y"]
        )
    )

    p5.metric(
        "Period CAGR",
        percent(
            market["cagr"]
        )
    )

    # Benchmark chart

    st.subheader(
        "Performance vs S&P 500"
    )

    benchmark_prices = (
        benchmark[
            "benchmark_prices"
        ]
    )

    compare_fig = go.Figure()

    stock_normalized = (
        prices / prices.iloc[0]
        * 100
    )

    compare_fig.add_trace(
        go.Scatter(
            x=stock_normalized.index,
            y=stock_normalized.values,
            name=ticker,
            mode="lines"
        )
    )

    if not benchmark_prices.empty:

        benchmark_normalized = (
            benchmark_prices
            / benchmark_prices.iloc[0]
            * 100
        )

        compare_fig.add_trace(
            go.Scatter(
                x=benchmark_normalized.index,
                y=benchmark_normalized.values,
                name="S&P 500",
                mode="lines"
            )
        )

    compare_fig.update_layout(
        template="plotly_white",
        height=450,
        yaxis_title="Growth of 100",
        hovermode="x unified"
    )

    st.plotly_chart(
        compare_fig,
        use_container_width=True
    )


# ======================================================
# PLATFORM VIEW
# ======================================================

with platform_tab:

    st.subheader("Platform View")

    if not platform_score["available"]:
        st.info(platform_score["reason"])
        st.caption(
            "The application intentionally avoids forcing the same scoring "
            "model onto asset classes or sectors where the underlying ratios "
            "are not economically comparable."
        )
    else:
        score_col, rating_col = st.columns([1, 2])
        score_col.metric("Platform Score", f'{platform_score["score"]} / 100')
        rating_col.metric("Quantitative Rating", platform_score["rating"])

        st.subheader("Score Breakdown")
        components = platform_score["components"]
        maxima = platform_score["component_max"]
        score_columns = st.columns(3)
        for idx, name in enumerate(["Valuation", "Growth", "Quality", "Financial Health", "Risk", "Wall Street"]):
            value = components.get(name, np.nan)
            display = f"{value:.1f} / {maxima[name]}" if is_valid(value) else "N/A"
            score_columns[idx % 3].metric(name, display)

        left, right = st.columns(2)
        with left:
            st.subheader("Key Strengths")
            if platform_score["strengths"]:
                for item in platform_score["strengths"]:
                    st.write(f"• {item}")
            else:
                st.caption("No major rule-based strength flag triggered.")

        with right:
            st.subheader("Key Risks")
            if platform_score["risks"]:
                for item in platform_score["risks"]:
                    st.write(f"• {item}")
            else:
                st.caption("No major rule-based risk flag triggered.")

        st.divider()
        st.caption(
            platform_score["methodology"] + " The score is a research aid, "
            "not a personalized investment recommendation. Wall Street data "
            "is third-party consensus data and is kept separate from the "
            "platform's calculated metrics."
        )


# ======================================================
# FUNDAMENTALS
# ======================================================

with financials_tab:

    st.subheader(
        "Growth"
    )

    g1, g2, g3, g4 = (
        st.columns(4)
    )

    g1.metric(
        "Revenue CAGR",
        percent(
            fund["revenue_cagr"]
        )
    )

    g2.metric(
        "EPS CAGR",
        percent(
            fund["eps_cagr"]
        )
    )

    g3.metric(
        "EBITDA CAGR",
        percent(
            fund["ebitda_cagr"]
        )
    )

    g4.metric(
        "FCF CAGR",
        percent(
            fund["fcf_cagr"]
        )
    )

    st.subheader(
        "Profitability"
    )

    m1, m2, m3, m4 = (
        st.columns(4)
    )

    m1.metric(
        "Gross Margin",
        percent(
            fund["gross_margin"]
        )
    )

    m2.metric(
        "EBITDA Margin",
        percent(
            fund["ebitda_margin"]
        )
    )

    m3.metric(
        "EBIT Margin",
        percent(
            fund["ebit_margin"]
        )
    )

    m4.metric(
        "Net Margin",
        percent(
            fund["net_margin"]
        )
    )

    m5, m6, m7, m8 = (
        st.columns(4)
    )

    m5.metric(
        "FCF Margin",
        percent(
            fund["fcf_margin"]
        )
    )

    m6.metric(
        "ROE",
        percent(
            fund["roe"]
        )
    )

    m7.metric(
        "ROA",
        percent(
            fund["roa"]
        )
    )

    m8.metric(
        "ROIC",
        percent(
            fund["roic"]
        )
    )

    st.subheader(
        "Financial Health"
    )

    b1, b2, b3, b4 = (
        st.columns(4)
    )

    b1.metric(
        "Cash",
        money(
            fund["cash"],
            symbol
        )
    )

    b2.metric(
        "Debt",
        money(
            fund["debt"],
            symbol
        )
    )

    b3.metric(
        "Net Debt",
        money(
            fund["net_debt"],
            symbol
        )
    )

    b4.metric(
        "Debt / Equity",
        multiple(
            fund["debt_to_equity"]
        )
    )

    b5, b6, b7, b8 = (
        st.columns(4)
    )

    b5.metric(
        "Net Debt / EBITDA",
        multiple(
            fund[
                "net_debt_to_ebitda"
            ]
        )
    )

    b6.metric(
        "Current Ratio",
        number(
            fund["current_ratio"]
        )
    )

    b7.metric(
        "Quick Ratio",
        number(
            fund["quick_ratio"]
        )
    )

    b8.metric(
        "Interest Coverage",
        multiple(
            fund[
                "interest_coverage"
            ]
        )
    )

    st.divider()

    financial_chart(
        "Revenue History",
        {
            "Revenue":
            fund["revenue_history"]
        }
    )

    financial_chart(
        "EBITDA & Net Income",
        {
            "EBITDA":
            fund["ebitda_history"],

            "Net Income":
            fund[
                "net_income_history"
            ]
        }
    )

    financial_chart(
        "Free Cash Flow",
        {
            "FCF":
            fund["fcf_history"]
        }
    )

    financial_chart(
        "EPS",
        {
            "EPS":
            fund["eps_history"]
        }
    )

    financial_chart(
        "Cash vs Debt",
        {
            "Cash":
            fund["cash_history"],

            "Debt":
            fund["debt_history"]
        }
    )


# ======================================================
# VALUATION
# ======================================================

with valuation_tab:

    st.subheader(
        "Trading Multiples"
    )

    v1, v2, v3, v4 = (
        st.columns(4)
    )

    v1.metric(
        "P/E",
        multiple(
            valuation["trailing_pe"]
        )
    )

    v2.metric(
        "Forward P/E",
        multiple(
            valuation["forward_pe"]
        )
    )

    v3.metric(
        "EV / EBITDA",
        multiple(
            valuation["ev_ebitda"]
        )
    )

    v4.metric(
        "EV / EBIT",
        multiple(
            valuation["ev_ebit"]
        )
    )

    v5, v6, v7, v8 = (
        st.columns(4)
    )

    v5.metric(
        "EV / Revenue",
        multiple(
            valuation["ev_revenue"]
        )
    )

    v6.metric(
        "Price / Sales",
        multiple(
            valuation[
                "price_to_sales"
            ]
        )
    )

    v7.metric(
        "Price / Book",
        multiple(
            valuation[
                "price_to_book"
            ]
        )
    )

    v8.metric(
        "PEG",
        number(
            valuation["peg"]
        )
    )

    st.subheader(
        "Cash Flow & Earnings"
    )

    y1, y2, y3, y4 = (
        st.columns(4)
    )

    y1.metric(
        "Price / FCF",
        multiple(
            valuation["price_fcf"]
        )
    )

    y2.metric(
        "FCF Yield",
        percent(
            valuation["fcf_yield"]
        )
    )

    y3.metric(
        "Earnings Yield",
        percent(
            valuation[
                "earnings_yield"
            ]
        )
    )

    y4.metric(
        "Dividend Yield",
        percent(
            valuation[
                "dividend_yield"
            ]
        )
    )


# ======================================================
# RISK
# ======================================================

with risk_tab:

    st.subheader(
        "Risk Analytics"
    )

    r1, r2, r3, r4 = (
        st.columns(4)
    )

    r1.metric(
        "Beta",
        number(
            benchmark["beta"]
        )
    )

    r2.metric(
        "Volatility",
        percent(
            market["volatility"]
        )
    )

    r3.metric(
        "Sharpe",
        number(
            benchmark["sharpe"]
        )
    )

    r4.metric(
        "Sortino",
        number(
            benchmark["sortino"]
        )
    )

    r5, r6 = st.columns(2)

    r5.metric(
        "Maximum Drawdown",
        percent(
            market["max_drawdown"]
        )
    )

    r6.metric(
        "Risk-Free Assumption",
        percent(
            benchmark[
                "risk_free_rate"
            ]
        )
    )

    st.caption(
        "Sharpe and Sortino use the latest available 13-week U.S. Treasury "
        "bill yield (^IRX) as a risk-free-rate proxy, with a 4% fallback if "
        "the market data is unavailable."
    )

    drawdown_fig = go.Figure()

    drawdown_fig.add_trace(
        go.Scatter(
            x=market[
                "drawdown"
            ].index,

            y=market[
                "drawdown"
            ].values * 100,

            fill="tozeroy",
            name="Drawdown"
        )
    )

    drawdown_fig.update_layout(
        title="Historical Drawdown",
        template="plotly_white",
        height=450,
        yaxis_title="Drawdown (%)"
    )

    st.plotly_chart(
        drawdown_fig,
        use_container_width=True
    )


# ======================================================
# WALL STREET
# ======================================================

with analyst_tab:

    st.subheader(
        "Analyst Consensus"
    )

    a1, a2, a3, a4 = (
        st.columns(4)
    )

    a1.metric(
        "Low Target",
        money(
            estimates["low"],
            symbol
        )
    )

    a2.metric(
        "Mean Target",
        money(
            estimates["mean"],
            symbol
        )
    )

    a3.metric(
        "Median Target",
        money(
            estimates["median"],
            symbol
        )
    )

    a4.metric(
        "High Target",
        money(
            estimates["high"],
            symbol
        )
    )

    a5, a6, a7 = (
        st.columns(3)
    )

    a5.metric(
        "Upside to Mean",
        percent(
            estimates[
                "upside_mean"
            ]
        )
    )

    recommendation = (
        estimates[
            "recommendation"
        ]
    )

    if isinstance(
        recommendation,
        str
    ):
        recommendation = (
            recommendation
            .replace("_", " ")
            .upper()
        )

    a6.metric(
        "Consensus",
        recommendation
    )

    analyst_count = (
        estimates["analysts"]
    )

    a7.metric(
        "Analysts",
        (
            str(int(analyst_count))
            if is_valid(
                analyst_count
            )
            else "N/A"
        )
    )

    st.caption(
        "Analyst targets and recommendations "
        "are third-party consensus data when "
        "available. They are not generated by "
        "this application."
    )

    # Revenue estimates

    st.subheader(
        "Revenue Estimates"
    )

    revenue_estimate = (
        raw["revenue_estimate"]
    )

    if (
        revenue_estimate is not None
        and not revenue_estimate.empty
    ):
        st.dataframe(
            revenue_estimate,
            use_container_width=True
        )
    else:
        st.info(
            "Revenue estimates unavailable."
        )

    # Earnings estimates

    st.subheader(
        "Earnings Estimates"
    )

    earnings_estimate = (
        raw["earnings_estimate"]
    )

    if (
        earnings_estimate is not None
        and not earnings_estimate.empty
    ):
        st.dataframe(
            earnings_estimate,
            use_container_width=True
        )
    else:
        st.info(
            "Earnings estimates unavailable."
        )

    # Growth estimates

    st.subheader(
        "Growth Estimates"
    )

    growth_estimates = (
        raw["growth_estimates"]
    )

    if (
        growth_estimates is not None
        and not growth_estimates.empty
    ):
        st.dataframe(
            growth_estimates,
            use_container_width=True
        )
    else:
        st.info(
            "Growth estimates unavailable."
        )

    st.subheader(
        "Recommendation Trends"
    )

    recommendation_summary = (
        raw[
            "recommendations_summary"
        ]
    )

    if (
        recommendation_summary
        is not None
        and not
        recommendation_summary.empty
    ):
        st.dataframe(
            recommendation_summary,
            use_container_width=True
        )
    else:
        st.info(
            "Recommendation history unavailable."
        )


# ======================================================
# PEERS
# ======================================================

with peers_tab:

    st.subheader(
        "Peer Comparison"
    )

    st.caption(
        "Peer sets are currently selected from "
        "a sector-based universe and are intended "
        "as a starting point for comparative analysis."
    )

    if peer_df.empty:

        st.info(
            "No peer comparison available "
            "for this sector."
        )

    else:

        formatted_peers = (
            peer_df.copy()
        )

        st.dataframe(
            formatted_peers,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Revenue Growth":
                    st.column_config.NumberColumn(
                        format="%.1f%%"
                    ),

                "Operating Margin":
                    st.column_config.NumberColumn(
                        format="%.1f%%"
                    ),

                "ROE":
                    st.column_config.NumberColumn(
                        format="%.1f%%"
                    ),
            }
        )

        # P/E peer chart

        chart_df = (
            peer_df[
                ["Ticker", "P/E"]
            ]
            .dropna()
        )

        if not chart_df.empty:

            peer_fig = go.Figure()

            peer_fig.add_trace(
                go.Bar(
                    x=chart_df[
                        "Ticker"
                    ],

                    y=chart_df[
                        "P/E"
                    ],

                    name="P/E"
                )
            )

            peer_fig.update_layout(
                title="Peer P/E Comparison",
                template="plotly_white",
                height=420,
                yaxis_title="P/E (x)"
            )

            st.plotly_chart(
                peer_fig,
                use_container_width=True
            )


# ======================================================
# STATEMENTS
# ======================================================

with statements_tab:

    st.caption(
        f"Financial statement values shown "
        f"in millions of {currency}."
    )

    (
        income_tab,
        balance_tab,
        cashflow_tab
    ) = st.tabs(
        [
            "Income Statement",
            "Balance Sheet",
            "Cash Flow"
        ]
    )

    with income_tab:

        table = statement_table(
            raw["income"]
        )

        if table.empty:
            st.info(
                "Income statement unavailable."
            )
        else:
            st.dataframe(
                table,
                use_container_width=True
            )

    with balance_tab:

        table = statement_table(
            raw["balance"]
        )

        if table.empty:
            st.info(
                "Balance sheet unavailable."
            )
        else:
            st.dataframe(
                table,
                use_container_width=True
            )

    with cashflow_tab:

        table = statement_table(
            raw["cashflow"]
        )

        if table.empty:
            st.info(
                "Cash-flow statement unavailable."
            )
        else:
            st.dataframe(
                table,
                use_container_width=True
            )


# ======================================================
# FOOTER
# ======================================================

st.divider()

st.caption(
    "Market and fundamental data: Yahoo Finance "
    "via yfinance. Financial metrics should be "
    "validated against company filings before being "
    "used in formal investment research."
)
