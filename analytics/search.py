import requests


YAHOO_SEARCH_URL = "https://query1.finance.yahoo.com/v1/finance/search"


def search_companies(query, max_results=8):
    """
    Search Yahoo Finance by company name or ticker.

    Returns:
        [
            {
                "ticker": "AAPL",
                "name": "Apple Inc.",
                "exchange": "NMS",
                "type": "EQUITY",
                "label": "Apple Inc. (AAPL)"
            }
        ]
    """

    query = query.strip()

    if len(query) < 1:
        return []

    params = {
        "q": query,
        "quotesCount": max_results,
        "newsCount": 0,
        "enableFuzzyQuery": True,
        "quotesQueryId": "tss_match_phrase_query",
    }

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:
        response = requests.get(
            YAHOO_SEARCH_URL,
            params=params,
            headers=headers,
            timeout=5,
        )

        response.raise_for_status()

        data = response.json()

    except Exception:
        return []

    results = []

    allowed_types = {
        "EQUITY",
        "ETF",
    }

    for quote in data.get("quotes", []):
        ticker = quote.get("symbol")
        name = (
            quote.get("longname")
            or quote.get("shortname")
            or ticker
        )

        quote_type = quote.get("quoteType")
        exchange = (
            quote.get("exchDisp")
            or quote.get("exchange")
            or ""
        )

        if not ticker:
            continue

        if quote_type not in allowed_types:
            continue

        results.append(
            {
                "ticker": ticker,
                "name": name,
                "exchange": exchange,
                "type": quote_type,
                "label": f"{name} ({ticker})",
            }
        )

    return results
