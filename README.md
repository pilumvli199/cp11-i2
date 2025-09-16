# Indian Market Analysis Bot (Angel One SmartAPI + OpenAI + Telegram)

This bot analyzes multi-timeframe charts (5m, 15m, 30m) for selected instruments (NIFTY, BANKNIFTY, RELIANCE),
detects signals (breakouts, candlestick patterns), integrates GPT analysis, and sends alerts with annotated charts to Telegram.

## Features
- Angel One SmartAPI login (supports **MPIN** and fallback Password+TOTP)
- Multi-timeframe candlestick analysis
- Candlestick & chart pattern detection
- GPT-4o-mini integration for advanced pattern insights
- Telegram alerts with annotated candlestick charts

## Setup
1. Clone repo / unzip project
2. Create `.env` file (copy from `.env.example` and fill in values)
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the bot locally:
   ```bash
   python main.py
   ```
5. Deploy to Railway.app / any cloud with Procfile

## Notes
- If Angel One enforces MPIN login, set `SMARTAPI_MPIN` in `.env` and leave password/TOTP blank.
- If MPIN not enforced, set password + TOTP secret.
- Avoid repeated login attempts (API has rate-limits).

## Author
Generated with ❤️ by GPT
