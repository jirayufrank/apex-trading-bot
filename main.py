#!/usr/bin/env python3
"""
Bybit Futures Trading Bot
Scans a watchlist every hour, calculates technical indicators, and places
trades when confluence score >= 4/6 signals agree.
"""

import os
import json
import math
import time
import hmac
import hashlib
import logging
import datetime
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

import requests
import pandas as pd
import numpy as np
import ta

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY = os.environ.get("BYBIT_API_KEY", "")
API_SECRET = os.environ.get("BYBIT_API_SECRET", "")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")

BASE_URL = "https://api.bybit.com"

DYNAMIC_SCAN_TOP_N = 50          # Pull top N symbols by 24H volume
DYNAMIC_VOLUME_MIN_USD = 50_000_000  # Minimum 24H volume in USD
STABLECOIN_BASES = {"USDC", "BUSD", "TUSD", "USDP", "DAI", "FDUSD", "PYUSD"}

RISK_PER_TRADE = 0.02       # 2% of account equity per trade
MAX_LEVERAGE = 10
MAX_OPEN_POSITIONS = 3      # Hard cap on concurrent open positions
DAILY_LOSS_LIMIT = 0.05     # Stop trading if daily loss exceeds 5%
SCAN_INTERVAL_SECONDS = 3600  # 1 hour
CONFLUENCE_THRESHOLD = 4     # Minimum signals that must agree

TRADE_JOURNAL_FILE = "trade_journal.json"

# Populated at startup by load_instrument_cache()
# symbol -> {"min_qty": float, "qty_step": float}
_instrument_cache: dict = {}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot.log"),
    ],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Trade Journal
# ---------------------------------------------------------------------------


def get_dynamic_watchlist() -> list[str]:
    """
    Fetch top DYNAMIC_SCAN_TOP_N USDT perpetual futures from Bybit ranked by
    24H turnover, then filter out stablecoins and low-volume symbols.
    Falls back to an empty list on any error.
    """
    try:
        resp = requests.get(
            f"{BASE_URL}/v5/market/tickers",
            params={"category": "linear"},
            timeout=10,
        )
        data = resp.json()
        if data.get("retCode") != 0:
            log.warning("Dynamic watchlist fetch error: %s", data.get("retMsg"))
            return []

        tickers = data["result"]["list"]

        # Keep only USDT-margined perps (symbol ends with USDT)
        usdt_perps = [t for t in tickers if t["symbol"].endswith("USDT")]

        # Sort by 24H turnover descending, take top N
        def _turnover(t):
            try:
                return float(t.get("turnover24h", 0))
            except (ValueError, TypeError):
                return 0.0

        top = sorted(usdt_perps, key=_turnover, reverse=True)[:DYNAMIC_SCAN_TOP_N]

        qualified = []
        for t in top:
            symbol = t["symbol"]
            # Strip the trailing USDT to get the base asset
            base = symbol[:-4]
            if base in STABLECOIN_BASES:
                continue
            vol_usd = _turnover(t)
            if vol_usd < DYNAMIC_VOLUME_MIN_USD:
                continue
            qualified.append(symbol)

        total = len(top)
        passed = len(qualified)
        log.info("Dynamic scan: %d/%d symbols qualified (volume ≥ $%.0fM)",
                 passed, total, DYNAMIC_VOLUME_MIN_USD / 1_000_000)
        return qualified

    except Exception as e:
        log.warning("get_dynamic_watchlist failed: %s", e)
        return []


def discord_notify(message: str) -> None:
    """Send a message to the configured Discord webhook. Silent on failure."""
    if not DISCORD_WEBHOOK_URL:
        return
    try:
        requests.post(
            DISCORD_WEBHOOK_URL,
            json={"content": message},
            timeout=10,
        )
    except Exception as e:
        log.warning("Discord notify failed: %s", e)


def load_journal() -> list:
    if os.path.exists(TRADE_JOURNAL_FILE):
        with open(TRADE_JOURNAL_FILE, "r") as f:
            return json.load(f)
    return []


def save_journal(journal: list) -> None:
    with open(TRADE_JOURNAL_FILE, "w") as f:
        json.dump(journal, f, indent=2, default=str)


def log_trade(entry: dict) -> None:
    journal = load_journal()
    journal.append(entry)
    save_journal(journal)
    log.info("Trade logged: %s", entry)


# ---------------------------------------------------------------------------
# Bybit REST helpers
# ---------------------------------------------------------------------------


def _post_headers(body: str) -> dict:
    """Build signed headers for a Bybit V5 POST request.
    body must be the exact JSON string that will be sent as the request body.
    """
    timestamp = str(int(time.time() * 1000))
    recv_window = "5000"
    param_str = timestamp + API_KEY + recv_window + body
    signature = hmac.new(
        API_SECRET.encode(), param_str.encode(), hashlib.sha256
    ).hexdigest()
    return {
        "X-BAPI-API-KEY": API_KEY,
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-RECV-WINDOW": recv_window,
        "X-BAPI-SIGN": signature,
        "Content-Type": "application/json",
    }


def get_account_balance() -> Optional[float]:
    """Return USDT wallet balance."""
    url = f"{BASE_URL}/v5/account/wallet-balance"
    params = {"accountType": "UNIFIED", "coin": "USDT"}
    timestamp = str(int(time.time() * 1000))
    recv_window = "5000"
    param_str = timestamp + API_KEY + recv_window + "&".join(
        f"{k}={v}" for k, v in sorted(params.items())
    )
    signature = hmac.new(
        API_SECRET.encode(), param_str.encode(), hashlib.sha256
    ).hexdigest()
    headers = {
        "X-BAPI-API-KEY": API_KEY,
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-RECV-WINDOW": recv_window,
        "X-BAPI-SIGN": signature,
    }
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        data = resp.json()
        if data.get("retCode") == 0:
            for item in data["result"]["list"]:
                for coin in item.get("coin", []):
                    if coin["coin"] == "USDT":
                        return float(coin["walletBalance"])
    except Exception as e:
        log.error("Failed to fetch balance: %s", e)
    return None


def get_open_positions() -> dict:
    """
    Return a dict of currently open positions: {symbol: side}.
    Only includes positions where the absolute size is non-zero.
    """
    url = f"{BASE_URL}/v5/position/list"
    params = {"category": "linear", "settleCoin": "USDT"}
    timestamp = str(int(time.time() * 1000))
    recv_window = "5000"
    param_str = timestamp + API_KEY + recv_window + "&".join(
        f"{k}={v}" for k, v in sorted(params.items())
    )
    signature = hmac.new(
        API_SECRET.encode(), param_str.encode(), hashlib.sha256
    ).hexdigest()
    headers = {
        "X-BAPI-API-KEY": API_KEY,
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-RECV-WINDOW": recv_window,
        "X-BAPI-SIGN": signature,
    }
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        data = resp.json()
        if data.get("retCode") == 0:
            positions = {}
            for item in data["result"].get("list", []):
                if float(item.get("size", 0)) != 0:
                    positions[item["symbol"]] = item["side"]
            return positions
        log.warning("get_open_positions error: %s", data.get("retMsg"))
    except Exception as e:
        log.error("Failed to fetch open positions: %s", e)
    return {}


def get_position_details() -> list:
    """
    Return full details for all open positions.
    Each dict includes: symbol, side, size, avgPrice, stopLoss, markPrice.
    """
    url = f"{BASE_URL}/v5/position/list"
    params = {"category": "linear", "settleCoin": "USDT"}
    timestamp = str(int(time.time() * 1000))
    recv_window = "5000"
    param_str = timestamp + API_KEY + recv_window + "&".join(
        f"{k}={v}" for k, v in sorted(params.items())
    )
    signature = hmac.new(
        API_SECRET.encode(), param_str.encode(), hashlib.sha256
    ).hexdigest()
    headers = {
        "X-BAPI-API-KEY": API_KEY,
        "X-BAPI-TIMESTAMP": timestamp,
        "X-BAPI-RECV-WINDOW": recv_window,
        "X-BAPI-SIGN": signature,
    }
    try:
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        data = resp.json()
        if data.get("retCode") == 0:
            return [
                item for item in data["result"].get("list", [])
                if float(item.get("size", 0)) != 0
            ]
        log.warning("get_position_details error: %s", data.get("retMsg"))
    except Exception as e:
        log.error("Failed to fetch position details: %s", e)
    return []


def set_breakeven_sl(symbol: str, entry_price: float) -> bool:
    """Move the stop loss on an open position to entry price (breakeven)."""
    url = f"{BASE_URL}/v5/position/trading-stop"
    payload = {
        "category": "linear",
        "symbol": symbol,
        "stopLoss": str(round(entry_price, 4)),
        "slTriggerBy": "LastPrice",
        "positionIdx": 0,
    }
    try:
        body = json.dumps(payload)
        resp = requests.post(url, headers=_post_headers(body), data=body, timeout=10)
        data = resp.json()
        if data.get("retCode") == 0:
            return True
        log.error("set_breakeven_sl failed for %s: %s", symbol, data.get("retMsg"))
    except Exception as e:
        log.error("set_breakeven_sl exception for %s: %s", symbol, e)
    return False


def manage_breakeven(positions: list) -> None:
    """
    For each open position, check whether price has moved 1.5× the initial
    risk distance into profit (TP1). If so, move SL to entry (breakeven).

    Risk distance is inferred from the current SL: abs(entry - stopLoss).
    Already-breakeven positions (SL within 0.1% of entry) are skipped.
    """
    for pos in positions:
        symbol = pos.get("symbol", "")
        side = pos.get("side", "")
        try:
            entry = float(pos.get("avgPrice") or 0)
            sl = float(pos.get("stopLoss") or 0)
            mark = float(pos.get("markPrice") or 0)
        except (ValueError, TypeError):
            continue

        if entry == 0 or mark == 0:
            continue

        # No SL set — nothing to manage
        if sl == 0:
            log.info("%s — no stop loss set, skipping breakeven check.", symbol)
            continue

        # Already at breakeven (SL within 0.1% of entry)
        if abs(sl - entry) / entry < 0.001:
            log.info("%s — SL already at breakeven (SL=%.4f entry=%.4f).", symbol, sl, entry)
            continue

        risk_dist = abs(entry - sl)
        if side == "Buy":
            tp1 = entry + 1.5 * risk_dist
            tp1_reached = mark >= tp1
        else:
            tp1 = entry - 1.5 * risk_dist
            tp1_reached = mark <= tp1

        if tp1_reached:
            log.info(
                "%s — TP1 reached (mark=%.4f ≥ tp1=%.4f), moving SL to breakeven at %.4f",
                symbol, mark, tp1, entry,
            )
            if set_breakeven_sl(symbol, entry):
                log.info("%s — SL successfully moved to breakeven %.4f", symbol, entry)
                discord_notify(
                    f"🔒 **Breakeven** {symbol} | TP1 reached at `{mark:.4f}` "
                    f"| SL moved to entry `{entry:.4f}`"
                )
            else:
                log.warning("%s — failed to move SL to breakeven", symbol)
                discord_notify(f"⚠️ **{symbol}** — failed to move SL to breakeven")
        else:
            pct_to_tp1 = abs(tp1 - mark) / mark * 100
            log.info(
                "%s — TP1 not yet reached (mark=%.4f tp1=%.4f, %.2f%% away)",
                symbol, mark, tp1, pct_to_tp1,
            )


FUNDING_RATE_MAX = 0.001   # +0.1%
FUNDING_RATE_MIN = -0.001  # -0.1%


def get_funding_rate(symbol: str) -> Optional[float]:
    """Return the current funding rate for a linear futures symbol."""
    url = f"{BASE_URL}/v5/market/tickers"
    params = {"category": "linear", "symbol": symbol}
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data.get("retCode") == 0 and data["result"]["list"]:
            return float(data["result"]["list"][0]["fundingRate"])
        log.warning("get_funding_rate error for %s: %s", symbol, data.get("retMsg"))
    except Exception as e:
        log.error("Failed to fetch funding rate for %s: %s", symbol, e)
    return None


def get_klines(symbol: str, interval: str, limit: int = 300) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV klines from Bybit.
    interval: '240' = 4H, '60' = 1H
    """
    url = f"{BASE_URL}/v5/market/kline"
    params = {
        "category": "linear",
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
    }
    try:
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        if data.get("retCode") != 0:
            log.warning("Kline error for %s %s: %s", symbol, interval, data.get("retMsg"))
            return None
        raw = data["result"]["list"]
        df = pd.DataFrame(
            raw,
            columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"],
        )
        df = df.astype(
            {"open": float, "high": float, "low": float, "close": float, "volume": float}
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception as e:
        log.error("Kline fetch failed for %s %s: %s", symbol, interval, e)
        return None


def _classify_ema(close: "pd.Series") -> str:
    """Return BULL / BEAR / RANGE based on EMA 9/50/200 alignment."""
    ema9   = ta.trend.EMAIndicator(close, window=9).ema_indicator().iloc[-1]
    ema50  = ta.trend.EMAIndicator(close, window=50).ema_indicator().iloc[-1]
    ema200 = ta.trend.EMAIndicator(close, window=200).ema_indicator().iloc[-1]
    if ema9 > ema50 > ema200:
        return "BULL"
    if ema9 < ema50 < ema200:
        return "BEAR"
    return "RANGE"


def get_market_regime() -> dict:
    """
    Multi-timeframe regime using BTC 4H + 1H EMA 9/50/200.

    Returns a dict:
        regime_4h   : BULL | BEAR | RANGE
        regime_1h   : BULL | BEAR | RANGE
        allowed     : Buy | Sell | SKIP
        threshold   : minimum confluence score required (4 or 5)

    Alignment rules (no counter-trend trading):
        4H BULL + 1H BULL  → Buy,  threshold 4
        4H BULL + 1H RANGE → Buy,  threshold 5
        4H BEAR + 1H BEAR  → Sell, threshold 4
        4H BEAR + 1H RANGE → Sell, threshold 5
        anything else      → SKIP
    """
    df4h = get_klines("BTCUSDT", "240", limit=250)
    df1h = get_klines("BTCUSDT", "60",  limit=250)

    fallback = {"regime_4h": "RANGE", "regime_1h": "RANGE",
                "allowed": "SKIP", "threshold": 5}

    if df4h is None or len(df4h) < 200:
        log.warning("Regime check: insufficient 4H data — defaulting to SKIP")
        return fallback

    r4h = _classify_ema(df4h["close"])

    r1h = "RANGE"
    if df1h is not None and len(df1h) >= 50:
        r1h = _classify_ema(df1h["close"])

    # Determine allowed direction and score threshold
    if r4h == "BULL":
        if r1h == "BULL":
            allowed, threshold = "Buy", 4
        elif r1h == "RANGE":
            allowed, threshold = "Buy", 5
        else:                          # 1H BEAR conflicts with 4H BULL
            allowed, threshold = "SKIP", 5
    elif r4h == "BEAR":
        if r1h == "BEAR":
            allowed, threshold = "Sell", 4
        elif r1h == "RANGE":
            allowed, threshold = "Sell", 5
        else:                          # 1H BULL conflicts with 4H BEAR
            allowed, threshold = "SKIP", 5
    else:                              # 4H RANGE → no new entries
        allowed, threshold = "SKIP", 5

    log.info(
        "REGIME: 4H=%s 1H=%s → allowed=%s threshold=%d",
        r4h, r1h, allowed, threshold,
    )
    return {"regime_4h": r4h, "regime_1h": r1h,
            "allowed": allowed, "threshold": threshold}


def load_instrument_cache(symbols: list) -> None:
    """Fetch lot size constraints for each symbol and store in _instrument_cache."""
    url = f"{BASE_URL}/v5/market/instruments-info"
    for symbol in symbols:
        try:
            resp = requests.get(
                url,
                params={"category": "linear", "symbol": symbol},
                timeout=10,
            )
            data = resp.json()
            if data.get("retCode") == 0 and data["result"]["list"]:
                lot = data["result"]["list"][0]["lotSizeFilter"]
                _instrument_cache[symbol] = {
                    "min_qty": float(lot["minOrderQty"]),
                    "qty_step": float(lot["qtyStep"]),
                }
                log.info(
                    "Instrument %s — min_qty=%s  qty_step=%s",
                    symbol, lot["minOrderQty"], lot["qtyStep"],
                )
            else:
                log.warning(
                    "Could not fetch instrument info for %s: %s",
                    symbol, data.get("retMsg"),
                )
        except Exception as e:
            log.error("Instrument info error for %s: %s", symbol, e)


def snap_qty(symbol: str, qty: float) -> float:
    """
    Round qty DOWN to the symbol's qty_step and enforce min_qty.
    Returns 0.0 if the snapped qty falls below min_qty.
    """
    info = _instrument_cache.get(symbol)
    if not info:
        return round(qty, 3)  # no cache entry — use safe fallback

    step = info["qty_step"]
    min_qty = info["min_qty"]

    # Floor to nearest step to avoid 'qty too precise' rejections
    snapped = math.floor(qty / step) * step

    # Match decimal precision of the step value
    decimals = (
        len(str(step).rstrip("0").split(".")[-1]) if "." in str(step) else 0
    )
    snapped = round(snapped, decimals)

    if snapped < min_qty:
        return 0.0
    return snapped


def set_leverage(symbol: str, leverage: int) -> bool:
    url = f"{BASE_URL}/v5/position/set-leverage"
    payload = {
        "category": "linear",
        "symbol": symbol,
        "buyLeverage": str(leverage),
        "sellLeverage": str(leverage),
    }
    try:
        body = json.dumps(payload)
        resp = requests.post(url, headers=_post_headers(body), data=body, timeout=10)
        data = resp.json()
        if data.get("retCode") in (0, 110043):  # 110043 = leverage not modified
            return True
        log.warning("Set leverage failed for %s: %s", symbol, data.get("retMsg"))
    except Exception as e:
        log.error("Set leverage error: %s", e)
    return False


def place_order(
    symbol: str,
    side: str,
    qty: float,
    sl_price: float,
    tp_price: float,
) -> Optional[dict]:
    """Place a market order with SL and TP on Bybit linear futures."""
    url = f"{BASE_URL}/v5/order/create"
    payload = {
        "category": "linear",
        "symbol": symbol,
        "side": side,
        "orderType": "Market",
        "qty": str(qty),
        "stopLoss": str(round(sl_price, 4)),
        "takeProfit": str(round(tp_price, 4)),
        "timeInForce": "IOC",
        "slTriggerBy": "LastPrice",
        "tpTriggerBy": "LastPrice",
        "positionIdx": 0,
    }
    try:
        body = json.dumps(payload)
        resp = requests.post(url, headers=_post_headers(body), data=body, timeout=10)
        data = resp.json()
        if data.get("retCode") == 0:
            log.info("Order placed: %s %s %s qty=%s", side, symbol, data["result"]["orderId"], qty)
            return data["result"]
        log.error("Order failed for %s: %s", symbol, data.get("retMsg"))
    except Exception as e:
        log.error("Place order exception: %s", e)
    return None


def close_position(symbol: str, side: str, qty: float) -> Optional[dict]:
    """
    Close an open position with a reduce-only market order.
    `side` is the *position* side ("Buy" for long, "Sell" for short).
    The closing order uses the opposite side.
    """
    close_side = "Sell" if side == "Buy" else "Buy"
    url = f"{BASE_URL}/v5/order/create"
    payload = {
        "category": "linear",
        "symbol": symbol,
        "side": close_side,
        "orderType": "Market",
        "qty": str(qty),
        "timeInForce": "IOC",
        "reduceOnly": True,
        "positionIdx": 0,
    }
    try:
        body = json.dumps(payload)
        resp = requests.post(url, headers=_post_headers(body), data=body, timeout=10)
        data = resp.json()
        if data.get("retCode") == 0:
            log.info("Position closed: %s %s qty=%s orderId=%s",
                     symbol, close_side, qty, data["result"]["orderId"])
            return data["result"]
        log.error("close_position failed for %s: %s", symbol, data.get("retMsg"))
    except Exception as e:
        log.error("close_position exception for %s: %s", symbol, e)
    return None


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------


def compute_indicators(df: pd.DataFrame) -> dict:
    """
    Compute EMA 9/50/200, VWAP, RSI, Stochastic RSI, and MACD.
    Returns a dict with the latest values.
    """
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    ema9 = ta.trend.EMAIndicator(close, window=9).ema_indicator()
    ema50 = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    ema200 = ta.trend.EMAIndicator(close, window=200).ema_indicator()

    # VWAP (cumulative approximation over available data)
    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / volume.cumsum()

    rsi = ta.momentum.RSIIndicator(close, window=14).rsi()

    stoch_rsi = ta.momentum.StochRSIIndicator(close, window=14, smooth1=3, smooth2=3)
    stoch_k = stoch_rsi.stochrsi_k()
    stoch_d = stoch_rsi.stochrsi_d()

    macd_ind = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    macd_line = macd_ind.macd()
    macd_signal = macd_ind.macd_signal()

    return {
        "close": close.iloc[-1],
        "ema9": ema9.iloc[-1],
        "ema50": ema50.iloc[-1],
        "ema200": ema200.iloc[-1],
        "vwap": vwap.iloc[-1],
        "rsi": rsi.iloc[-1],
        "stoch_k": stoch_k.iloc[-1],
        "stoch_d": stoch_d.iloc[-1],
        "macd": macd_line.iloc[-1],
        "macd_signal": macd_signal.iloc[-1],
    }


def check_reversal_signals(symbol: str, position_side: str, df: pd.DataFrame) -> tuple[int, list[str]]:
    """
    Check 5 reversal signals on the 1H dataframe against an open position.

    position_side = "Buy"  → long position → look for bearish reversal
    position_side = "Sell" → short position → look for bullish reversal

    Returns (signal_count, signal_names).
    """
    if len(df) < 3:
        return 0, []

    close = df["close"]
    high  = df["high"]
    low   = df["low"]
    open_ = df["open"]

    # Indicator series
    ema9     = ta.trend.EMAIndicator(close, window=9).ema_indicator()
    rsi_ser  = ta.momentum.RSIIndicator(close, window=14).rsi()
    stoch    = ta.momentum.StochRSIIndicator(close, window=14, smooth1=3, smooth2=3)
    stoch_k  = stoch.stochrsi_k()
    stoch_d  = stoch.stochrsi_d()
    macd_ind = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    hist_ser = macd_ind.macd_diff()  # histogram = macd - signal

    # Current and previous bar values
    c_close = close.iloc[-1];    p_close = close.iloc[-2]
    c_open  = open_.iloc[-1];    p_open  = open_.iloc[-2]
    c_high  = high.iloc[-1];     p_high  = high.iloc[-2]   # noqa: F841
    c_low   = low.iloc[-1];      p_low   = low.iloc[-2]    # noqa: F841
    c_ema9  = ema9.iloc[-1];     p_ema9  = ema9.iloc[-2]
    c_rsi   = rsi_ser.iloc[-1];  p_rsi   = rsi_ser.iloc[-2]
    c_k     = stoch_k.iloc[-1];  p_k     = stoch_k.iloc[-2]
    c_d     = stoch_d.iloc[-1];  p_d     = stoch_d.iloc[-2]
    c_hist  = hist_ser.iloc[-1]; p_hist  = hist_ser.iloc[-2]

    signals: list[str] = []

    if position_side == "Sell":
        # --- Bullish reversal signals (exit short) ---

        # 1. RSI crosses above 50
        if p_rsi < 50 and c_rsi >= 50:
            signals.append("RSI>50")

        # 2. StochRSI K crosses above D from below 20
        if p_k < 20 and p_k <= p_d and c_k > c_d:
            signals.append("StochRSI_cross_up")

        # 3. MACD histogram flips negative → positive
        if p_hist < 0 and c_hist >= 0:
            signals.append("MACD_hist_flip_up")

        # 4. Price crosses back above EMA9
        if p_close < p_ema9 and c_close >= c_ema9:
            signals.append("price>EMA9")

        # 5. Bullish Engulfing or Hammer
        body    = abs(c_close - c_open)
        lo_wick = min(c_close, c_open) - c_low
        hi_wick = c_high - max(c_close, c_open)
        hammer = (body > 0 and lo_wick >= 2 * body and hi_wick <= body)
        bull_eng = (p_close < p_open and c_close > c_open
                    and c_open <= p_close and c_close >= p_open)
        if hammer or bull_eng:
            signals.append("BullishPattern")

    else:  # position_side == "Buy"
        # --- Bearish reversal signals (exit long) ---

        # 1. RSI drops below 50
        if p_rsi > 50 and c_rsi <= 50:
            signals.append("RSI<50")

        # 2. StochRSI K crosses below D from above 80
        if p_k > 80 and p_k >= p_d and c_k < c_d:
            signals.append("StochRSI_cross_down")

        # 3. MACD histogram flips positive → negative
        if p_hist > 0 and c_hist <= 0:
            signals.append("MACD_hist_flip_down")

        # 4. Price crosses below EMA9
        if p_close > p_ema9 and c_close <= c_ema9:
            signals.append("price<EMA9")

        # 5. Bearish Engulfing or Shooting Star
        body    = abs(c_close - c_open)
        lo_wick = min(c_close, c_open) - c_low
        hi_wick = c_high - max(c_close, c_open)
        shooting_star = (body > 0 and hi_wick >= 2 * body and lo_wick <= body)
        bear_eng = (p_close > p_open and c_close < c_open
                    and c_open >= p_close and c_close <= p_open)
        if shooting_star or bear_eng:
            signals.append("BearishPattern")

    log.info("%s reversal check (%s side): %d/5 — %s",
             symbol, position_side, len(signals), signals)
    return len(signals), signals


def manage_reversals(position_details: list) -> dict[str, bool]:
    """
    For every open position, run the reversal signal check on 1H data.
    Closes any position where ≥3 reversal signals fire.
    Returns a dict {symbol: True} for each position that was closed.
    """
    EXIT_THRESHOLD = 3
    closed: dict[str, bool] = {}

    for pos in position_details:
        symbol = pos["symbol"]
        side   = pos["side"]        # "Buy" or "Sell"
        qty    = float(pos["size"])
        entry  = float(pos.get("avgPrice", 0))
        mark   = float(pos.get("markPrice", 0))
        upnl   = float(pos.get("unrealisedPnl", 0))

        df1h = get_klines(symbol, "60", limit=100)
        if df1h is None or len(df1h) < 3:
            log.warning("%s — could not fetch 1H klines for reversal check", symbol)
            continue

        count, fired = check_reversal_signals(symbol, side, df1h)
        if count < EXIT_THRESHOLD:
            continue

        log.warning(
            "%s — EXIT triggered: %d/5 reversal signals (%s). Closing %s qty=%s entry=%.4f mark=%.4f uPnL=%.2f",
            symbol, count, fired, side, qty, entry, mark, upnl,
        )
        result = close_position(symbol, side, qty)
        if result:
            pnl_str = f"+${upnl:.2f}" if upnl >= 0 else f"-${abs(upnl):.2f}"
            direction = "LONG" if side == "Buy" else "SHORT"
            discord_notify(
                f"🔔 **EXIT {direction} {symbol}** | Reversal `{count}/5` signals "
                f"[{', '.join(fired)}] | PnL: `{pnl_str}`"
            )
            closed[symbol] = True
        else:
            discord_notify(f"⚠️ **{symbol}** — reversal exit order FAILED ({count}/5 signals)")

    return closed


# ---------------------------------------------------------------------------
# 15M momentum monitor + quick exits
# ---------------------------------------------------------------------------

QUICK_PROFIT_R       = 0.8    # Close if uPnL ≥ 0.8R within any cycle
TIME_EXIT_HOURS      = 6      # Close if held longer than this …
TIME_EXIT_MIN_R      = 0.3    # … and profit is below this R multiple
FUNDING_SPIKE_PCT    = 0.0005 # ±0.05% funding rate triggers exit while in profit


def check_15m_momentum(symbol: str, position_side: str) -> bool:
    """
    Return True if 15M momentum has reversed against the open position.
    Requires BOTH: EMA9 cross AND RSI crossing 50.
    """
    df = get_klines(symbol, "15", limit=60)
    if df is None or len(df) < 20:
        return False

    close = df["close"]
    ema9  = ta.trend.EMAIndicator(close, window=9).ema_indicator()
    rsi   = ta.momentum.RSIIndicator(close, window=14).rsi()

    c_close, p_close = close.iloc[-1], close.iloc[-2]
    c_ema9,  p_ema9  = ema9.iloc[-1],  ema9.iloc[-2]
    c_rsi,   p_rsi   = rsi.iloc[-1],   rsi.iloc[-2]

    if position_side == "Buy":
        # Bearish: price crosses below EMA9 AND RSI crosses below 50
        ema_cross = p_close >= p_ema9 and c_close < c_ema9
        rsi_flip  = p_rsi >= 50 and c_rsi < 50
    else:
        # Bullish: price crosses above EMA9 AND RSI crosses above 50
        ema_cross = p_close <= p_ema9 and c_close > c_ema9
        rsi_flip  = p_rsi <= 50 and c_rsi > 50

    return ema_cross and rsi_flip


def manage_quick_exits(position_details: list) -> dict[str, bool]:
    """
    Check every open position for quick-exit conditions each cycle:

    1. 0.8R profit reached → close immediately
    2. 15M EMA9 cross + RSI flip → momentum exit
    3. Funding rate spikes ≥ ±0.05% while position is in profit → close
    4. Held > 6 hours with < 0.3R profit → time exit

    Returns {symbol: True} for every position that was closed.
    """
    closed: dict[str, bool] = {}
    now_ms = int(time.time() * 1000)

    for pos in position_details:
        symbol = pos["symbol"]
        side   = pos["side"]
        qty    = float(pos.get("size", 0))
        entry  = float(pos.get("avgPrice", 0) or 0)
        sl     = float(pos.get("stopLoss", 0) or 0)
        upnl   = float(pos.get("unrealisedPnl", 0) or 0)
        created_ms = int(pos.get("createdTime", now_ms) or now_ms)

        if entry == 0 or qty == 0:
            continue

        risk_dist = abs(entry - sl) if sl != 0 else 0.0
        r_usdt    = risk_dist * qty          # 1R in USDT
        hours_held = (now_ms - created_ms) / 3_600_000

        exit_reason: Optional[str] = None

        # ── 1. Quick profit: ≥ 0.8R ────────────────────────────────────────
        if r_usdt > 0 and upnl >= QUICK_PROFIT_R * r_usdt:
            exit_reason = f"0.8R profit (`{upnl:.2f}` USDT ≥ `{QUICK_PROFIT_R * r_usdt:.2f}`)"

        # ── 2. 15M momentum shift ───────────────────────────────────────────
        if exit_reason is None:
            if check_15m_momentum(symbol, side):
                exit_reason = "15M momentum shift (EMA9 cross + RSI flip)"

        # ── 3. Funding rate spike while in profit ───────────────────────────
        if exit_reason is None and upnl > 0:
            funding = get_funding_rate(symbol)
            if funding is not None and abs(funding) >= FUNDING_SPIKE_PCT:
                exit_reason = f"funding spike `{funding * 100:.4f}%` while in profit"

        # ── 4. Time exit: > 6 h with < 0.3R profit ─────────────────────────
        if exit_reason is None:
            if hours_held > TIME_EXIT_HOURS and r_usdt > 0 and upnl < TIME_EXIT_MIN_R * r_usdt:
                exit_reason = (
                    f"time exit `{hours_held:.1f}h` held, "
                    f"profit `{upnl:.2f}` < `{TIME_EXIT_MIN_R * r_usdt:.2f}` (0.3R)"
                )

        if exit_reason is None:
            continue

        pnl_str = f"+${upnl:.2f}" if upnl >= 0 else f"-${abs(upnl):.2f}"
        direction_label = "LONG" if side == "Buy" else "SHORT"

        log.warning(
            "%s — QUICK EXIT [%s] | reason: %s | uPnL=%s | held=%.1fh",
            symbol, direction_label, exit_reason, pnl_str, hours_held,
        )

        result = close_position(symbol, side, qty)
        if result:
            # Choose right emoji per reason
            if "time exit" in exit_reason:
                icon = "⏰"
                label = "TIME EXIT"
            elif "momentum" in exit_reason:
                icon = "⚡"
                label = "QUICK EXIT"
            else:
                icon = "⚡"
                label = "QUICK EXIT"

            discord_notify(
                f"{icon} **{label} {direction_label} {symbol}** | "
                f"{exit_reason} | PnL: `{pnl_str}` | Held: `{hours_held:.1f}h`"
            )
            closed[symbol] = True
        else:
            discord_notify(
                f"⚠️ **{symbol}** — quick exit FAILED | reason: {exit_reason}"
            )

    return closed


# ---------------------------------------------------------------------------
# Signal scoring
# ---------------------------------------------------------------------------


def score_signals(ind4h: dict, ind1h: dict) -> tuple[int, str]:
    """
    Score 6 confluence signals. Returns (score, direction).
    direction is 'Buy' or 'Sell' based on the dominant side.
    """
    buy_signals = 0
    sell_signals = 0
    details = []

    # 1. EMA trend alignment (4H): price > EMA9 > EMA50 > EMA200 → bullish
    if ind4h["close"] > ind4h["ema9"] > ind4h["ema50"] > ind4h["ema200"]:
        buy_signals += 1
        details.append("EMA_bull")
    elif ind4h["close"] < ind4h["ema9"] < ind4h["ema50"] < ind4h["ema200"]:
        sell_signals += 1
        details.append("EMA_bear")

    # 2. VWAP (1H): price above VWAP → bullish
    if ind1h["close"] > ind1h["vwap"]:
        buy_signals += 1
        details.append("VWAP_bull")
    elif ind1h["close"] < ind1h["vwap"]:
        sell_signals += 1
        details.append("VWAP_bear")

    # 3. RSI (1H): oversold < 40 → bullish setup, overbought > 60 → bearish setup
    if ind1h["rsi"] < 40:
        buy_signals += 1
        details.append("RSI_bull")
    elif ind1h["rsi"] > 60:
        sell_signals += 1
        details.append("RSI_bear")

    # 4. Stochastic RSI (1H): K crosses above D and K < 80 → bullish
    if ind1h["stoch_k"] > ind1h["stoch_d"] and ind1h["stoch_k"] < 80:
        buy_signals += 1
        details.append("StochRSI_bull")
    elif ind1h["stoch_k"] < ind1h["stoch_d"] and ind1h["stoch_k"] > 20:
        sell_signals += 1
        details.append("StochRSI_bear")

    # 5. MACD (4H): MACD line above signal line → bullish
    if ind4h["macd"] > ind4h["macd_signal"]:
        buy_signals += 1
        details.append("MACD_bull")
    elif ind4h["macd"] < ind4h["macd_signal"]:
        sell_signals += 1
        details.append("MACD_bear")

    # 6. EMA crossover confirmation (1H): EMA9 > EMA50 → bullish momentum
    if ind1h["ema9"] > ind1h["ema50"]:
        buy_signals += 1
        details.append("EMA1H_bull")
    elif ind1h["ema9"] < ind1h["ema50"]:
        sell_signals += 1
        details.append("EMA1H_bear")

    if buy_signals >= sell_signals:
        return buy_signals, "Buy", details
    else:
        return sell_signals, "Sell", details


def calculate_position(
    balance: float,
    entry_price: float,
    sl_price: float,
    leverage: int,
) -> float:
    """
    Calculate position size based on 2% risk and max leverage.
    Returns quantity in base asset units.
    """
    risk_amount = balance * RISK_PER_TRADE
    sl_distance = abs(entry_price - sl_price)
    if sl_distance == 0:
        return 0.0
    qty_usdt = (risk_amount / sl_distance) * entry_price
    max_qty_usdt = balance * leverage
    qty_usdt = min(qty_usdt, max_qty_usdt)
    qty = qty_usdt / entry_price
    return round(qty, 3)


# ---------------------------------------------------------------------------
# Daily loss guard
# ---------------------------------------------------------------------------


def daily_loss_exceeded(starting_balance: float, current_balance: float) -> bool:
    if starting_balance <= 0:
        return False
    loss_pct = (starting_balance - current_balance) / starting_balance
    return loss_pct >= DAILY_LOSS_LIMIT


# ---------------------------------------------------------------------------
# Main scan loop
# ---------------------------------------------------------------------------


def scan_symbol(symbol: str, balance: float, regime: Optional[dict] = None) -> Optional[dict]:
    """Analyse one symbol and return a trade dict if a signal fires, else None."""
    log.info("Scanning %s …", symbol)

    df4h = get_klines(symbol, "240")
    df1h = get_klines(symbol, "60")

    if df4h is None or df1h is None or len(df4h) < 200 or len(df1h) < 50:
        log.warning("Insufficient data for %s", symbol)
        return None

    # --- Volume confirmation filter (4H) ---
    avg_vol_20 = df4h["volume"].iloc[-21:-1].mean()   # last 20 closed candles
    current_vol = df4h["volume"].iloc[-1]
    vol_ratio = current_vol / avg_vol_20 if avg_vol_20 > 0 else 0.0

    if vol_ratio < 1.2:
        log.info("%s — Volume: %.2fx avg — SKIP low volume confirmation", symbol, vol_ratio)
        return None
    log.info("%s — Volume: %.2fx avg — OK", symbol, vol_ratio)

    ind4h = compute_indicators(df4h)
    ind1h = compute_indicators(df1h)

    score, direction, details = score_signals(ind4h, ind1h)

    log.info(
        "%s | score=%d/6 dir=%s signals=%s",
        symbol, score, direction, details,
    )

    # --- Multi-timeframe regime filter ---
    if regime is None:
        regime = {"allowed": "SKIP", "threshold": 5, "regime_4h": "?", "regime_1h": "?"}

    allowed   = regime.get("allowed", "SKIP")
    threshold = regime.get("threshold", CONFLUENCE_THRESHOLD)

    if allowed == "SKIP":
        log.info(
            "%s SKIP — regime conflict/range (4H=%s 1H=%s), no new entries.",
            symbol, regime.get("regime_4h"), regime.get("regime_1h"),
        )
        return None
    if allowed != direction:
        log.info(
            "%s SKIP — regime allows %s only, signal direction is %s.",
            symbol, allowed, direction,
        )
        return None
    if score < threshold:
        log.info(
            "%s SKIP — score %d < required %d (4H=%s 1H=%s).",
            symbol, score, threshold, regime.get("regime_4h"), regime.get("regime_1h"),
        )
        return None

    entry = ind1h["close"]
    atr_approx = (df1h["high"].iloc[-14:] - df1h["low"].iloc[-14:]).mean()

    if direction == "Buy":
        sl = entry - 2 * atr_approx
        tp = entry + 4 * atr_approx
    else:
        sl = entry + 2 * atr_approx
        tp = entry - 4 * atr_approx

    leverage = min(MAX_LEVERAGE, 10)
    qty = calculate_position(balance, entry, sl, leverage)

    # RANGE regime: reduce position size by 50%
    if regime == "RANGE":
        qty *= 0.5
        log.info("%s RANGE regime — position size halved.", symbol)

    qty = snap_qty(symbol, qty)

    if qty <= 0:
        log.warning("Calculated qty is 0 for %s, skipping", symbol)
        return None

    set_leverage(symbol, leverage)
    result = place_order(symbol, direction, qty, sl, tp)

    trade = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "symbol": symbol,
        "direction": direction,
        "score": score,
        "signals": details,
        "entry": entry,
        "sl": round(sl, 4),
        "tp": round(tp, 4),
        "qty": qty,
        "leverage": leverage,
        "order_result": result,
        "ind_4h": {k: round(v, 6) if isinstance(v, float) else v for k, v in ind4h.items()},
        "ind_1h": {k: round(v, 6) if isinstance(v, float) else v for k, v in ind1h.items()},
    }

    log_trade(trade)

    # Discord alert on successful order
    if result:
        emoji = "🟢" if direction == "Buy" else "🔴"
        discord_notify(
            f"{emoji} **{direction} {symbol}** | Score {score}/6 | "
            f"Entry `{entry:.4f}` | SL `{round(sl, 4)}` | TP `{round(tp, 4)}` | "
            f"Qty `{qty}` × {leverage}x"
        )

    return trade


# ---------------------------------------------------------------------------
# UptimeRobot keep-alive HTTP server
# ---------------------------------------------------------------------------

HTTP_PORT = int(os.environ.get("PORT", 5000))


class _PingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = b"APEX Bot Running"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass  # silence default request logging


def _start_http_server():
    import socket
    for attempt in range(5):
        port = HTTP_PORT + attempt
        try:
            server = HTTPServer(("0.0.0.0", port), _PingHandler)
            server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            log.info("Keep-alive HTTP server listening on port %d", port)
            server.serve_forever()
            return
        except OSError as e:
            log.warning("Port %d in use (%s), trying %d …", port, e.strerror, port + 1)
            continue
    log.error("Could not bind HTTP server on ports %d–%d — keep-alive disabled",
              HTTP_PORT, HTTP_PORT + 4)


def main():
    log.info("=" * 60)
    log.info("Bybit Futures Trading Bot starting")
    log.info("Dynamic scan: top %d USDT perps, vol ≥ $%.0fM",
             DYNAMIC_SCAN_TOP_N, DYNAMIC_VOLUME_MIN_USD / 1_000_000)
    log.info("=" * 60)
    discord_notify(
        f"🚀 **APEX Bot started** | Dynamic scan: top {DYNAMIC_SCAN_TOP_N} USDT perps "
        f"with vol ≥ ${DYNAMIC_VOLUME_MIN_USD/1_000_000:.0f}M"
    )

    # Start keep-alive server in background thread for UptimeRobot
    t = threading.Thread(target=_start_http_server, daemon=True)
    t.start()

    if not API_KEY or not API_SECRET:
        log.error(
            "BYBIT_API_KEY and BYBIT_API_SECRET environment variables are not set. "
            "Please set them before running the bot."
        )
        return

    # Record the starting balance for the daily loss guard
    day_start = datetime.date.today()
    starting_balance = get_account_balance()
    if starting_balance is None:
        log.error("Could not fetch account balance. Check API credentials.")
        return

    log.info("Starting balance: %.2f USDT", starting_balance)

    while True:
        # Reset daily baseline at midnight
        today = datetime.date.today()
        if today != day_start:
            day_start = today
            starting_balance = get_account_balance() or starting_balance
            log.info("New day — reset daily baseline to %.2f USDT", starting_balance)

        current_balance = get_account_balance()
        if current_balance is None:
            log.warning("Could not fetch balance; retrying next cycle")
            time.sleep(SCAN_INTERVAL_SECONDS)
            continue

        if daily_loss_exceeded(starting_balance, current_balance):
            loss_pct = (starting_balance - current_balance) / starting_balance * 100
            log.warning(
                "Daily loss limit reached (%.2f → %.2f USDT). "
                "Pausing until tomorrow.",
                starting_balance, current_balance,
            )
            discord_notify(
                f"🛑 **Daily loss limit hit** | Start `{starting_balance:.2f}` → "
                f"Now `{current_balance:.2f}` USDT (`-{loss_pct:.1f}%`) | Pausing until tomorrow"
            )
            time.sleep(SCAN_INTERVAL_SECONDS)
            continue

        # --- Position check + breakeven management at start of every cycle ---
        open_positions = get_open_positions()
        pos_count = len(open_positions)
        log.info(
            "Open positions: %d/%d %s | Balance: %.2f USDT",
            pos_count, MAX_OPEN_POSITIONS, list(open_positions.keys()), current_balance,
        )

        # Run breakeven manager + all exit checks on all open positions
        if pos_count > 0:
            position_details = get_position_details()
            if position_details:
                manage_breakeven(position_details)
                closed_rev   = manage_reversals(position_details)
                closed_quick = manage_quick_exits(position_details)
                if closed_rev or closed_quick:
                    # Refresh position count after any forced exits
                    open_positions = get_open_positions()
                    pos_count = len(open_positions)

        if pos_count >= MAX_OPEN_POSITIONS:
            next_scan = datetime.datetime.utcnow() + datetime.timedelta(seconds=SCAN_INTERVAL_SECONDS)
            log.info(
                "Max positions reached (%d/%d). Skipping scan. Next at %s UTC",
                pos_count, MAX_OPEN_POSITIONS, next_scan.strftime("%H:%M:%S"),
            )
            time.sleep(SCAN_INTERVAL_SECONDS)
            continue

        # --- Build dynamic watchlist for this cycle ---
        watchlist = get_dynamic_watchlist()
        if not watchlist:
            log.warning("Dynamic watchlist returned 0 symbols — skipping scan cycle")
            time.sleep(SCAN_INTERVAL_SECONDS)
            continue

        # Refresh instrument lot-size cache for any new symbols
        load_instrument_cache(watchlist)

        # --- Market regime check (runs before every scan cycle) ---
        regime = get_market_regime()

        for symbol in watchlist:
            # Hard cap check before each symbol
            if len(open_positions) >= MAX_OPEN_POSITIONS:
                log.info(
                    "Max positions reached (%d/%d). Stopping scan early.",
                    len(open_positions), MAX_OPEN_POSITIONS,
                )
                break

            # Skip symbols already in an open position
            if symbol in open_positions:
                log.info("%s already has an open %s position — skipping.", symbol, open_positions[symbol])
                continue

            # Funding rate filter
            funding = get_funding_rate(symbol)
            if funding is None:
                log.warning("%s — could not fetch funding rate, skipping.", symbol)
                continue
            if funding > FUNDING_RATE_MAX:
                log.info("%s SKIP — funding rate too high (%.4f%%)", symbol, funding * 100)
                continue
            if funding < FUNDING_RATE_MIN:
                log.info("%s SKIP — funding rate too negative (%.4f%%)", symbol, funding * 100)
                continue
            log.info("%s funding rate OK: %.4f%%", symbol, funding * 100)

            try:
                trade = scan_symbol(symbol, current_balance, regime)
                # If an order was actually placed, refresh positions immediately
                if trade and trade.get("order_result"):
                    open_positions = get_open_positions()
                    log.info(
                        "Post-order positions: %d/%d %s",
                        len(open_positions), MAX_OPEN_POSITIONS, list(open_positions.keys()),
                    )
            except Exception as e:
                log.error("Unexpected error scanning %s: %s", symbol, e)

            time.sleep(2)  # brief pause between symbols to avoid rate limits

        next_scan = datetime.datetime.utcnow() + datetime.timedelta(seconds=SCAN_INTERVAL_SECONDS)
        log.info("Scan complete. Next scan at %s UTC", next_scan.strftime("%H:%M:%S"))
        regime_label = f"{regime.get('regime_4h','?')}/{regime.get('regime_1h','?')} → {regime.get('allowed','?')}"
        discord_notify(
            f"📊 **Scan complete** | Regime: `{regime_label}` | "
            f"Positions: `{len(open_positions)}/{MAX_OPEN_POSITIONS}` | "
            f"Balance: `{current_balance:.2f}` USDT | "
            f"Next scan: `{next_scan.strftime('%H:%M')} UTC`"
        )
        time.sleep(SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
