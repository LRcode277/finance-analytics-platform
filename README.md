
## Hosted data fallbacks
For hosted deployments, configure these environment variables/secrets (never commit keys):
- `ALPHA_VANTAGE_API_KEY`
- `FINNHUB_API_KEY`

Yahoo/yfinance remains the primary source. Finnhub fills company profile, basic financial metrics, industry peers and recommendation trends when Yahoo quote-summary endpoints are unavailable. Alpha Vantage fills company overview and financial statements. Some forward analyst datasets (notably full price-target ranges and forward estimates) are premium at Finnhub; the app degrades gracefully when the configured plan does not expose them.
