# APEX Trading Bot

Bybit USDT perpetual futures trading bot built in Python.

## Features
- Dynamic market scanning: top 50 USDT perps by 24H volume each cycle
- Layered filters: market regime (BTC EMA), funding rate, volume confirmation
- Signal confluence: EMA/VWAP/RSI/StochRSI/MACD (threshold: 4/6)
- Automatic breakeven SL management
- Dynamic reversal exit system (3/5 signals → market close)
- Discord alerts: entries, breakeven, exits, scan summaries, daily loss limit
- Max 3 concurrent positions, 2% risk per trade, up to 10× leverage

## Environment Variables
| Variable | Description |
|---|---|
| `BYBIT_API_KEY` | Bybit API key |
| `BYBIT_API_SECRET` | Bybit API secret |
| `DISCORD_WEBHOOK_URL` | Discord webhook for alerts |
| `PORT` | HTTP keep-alive port (default: 5000) |

## Deploy on Railway
1. Fork / import this repo into Railway
2. Set the environment variables above
3. Railway auto-detects the `Procfile` and runs `python3 main.py`

## Run locally
```bash
pip install -r requirements.txt
export BYBIT_API_KEY=...
export BYBIT_API_SECRET=...
export DISCORD_WEBHOOK_URL=...
python3 main.py
```
