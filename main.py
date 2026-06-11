#!/usr/bin/env python3
"""
APEX Bybit Futures Trading Bot — Upgraded
Fixes applied:
  1. Symbol-specific regime (XAUT uses gold proxy, not BTC)
  2. TP snaps to nearest S/R pivot level
  3. Long entries disabled until track record exists (LONG_ENABLED = False)
  4. 15M Quick Exit requires 3-candle confirmation (45 min)
  5. Weekly circuit breaker (-10% → pause 48h)
  6. Correlation filter (max 2 BTC-correlated positions)
  7. Dynamic position sizing by confluence score
  8. Coin blacklist + spread check
  9. Daily performance summary at 00:00 UTC
 10. 6-hour health-check ping to Discord
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

API_KEY            = os.environ.get("BYBIT_API_KEY", "")
API_SECRET         = os.environ.get("BYBIT_API_SECRET", "")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")

BASE_URL = "https://api.bybit.com"

DYNAMIC_SCAN_TOP_N       = 50
DYNAMIC_VOLUME_MIN_USD   = 50_000_000
STABLECOIN_BASES         = {"USDC", "BUSD", "TUSD", "USDP", "DAI", "FDUSD", "PYUSD"}

# Risk & position limits
RISK_PER_TRADE      = 0.02    # base 2% — scaled by score below
MAX_LEVERAGE        = 10
MAX_OPEN_POSITIONS  = 3
DAILY_LOSS_LIMIT    = 0.05    # 5% daily  → pause until tomorrow
WEEKLY_LOSS_LIMIT   = 0.10    # 10% weekly → pause 48 h
SCAN_INTERVAL_SECONDS = 3600  # 1 hour
CONFLUENCE_THRESHOLD  = 4

# ── NEW: Long guard ─────────────────────────────────────────────────────────
# Set True only after Long logic has been validated with real trade data.
LONG_ENABLED = False

# ── NEW: Correlation filter ──────────────────────────────────────────────────
# Coins that move closely with BTC — limit concurrent exposure
MAX_CORRELATION_POSITIONS = 2
BTC_CORRELATED = {
    "ETHUSDT", "SOLUSDT", "BNBUSDT", "AVAXUSDT",
    "LINKUSDT", "DOTUSDT", "MATICUSDT", "LTCUSDT",
    "ADAUSDT", "ATOMUSDT", "NEARUSDT", "APTUSDT",
}

# ── NEW: Coin blacklist ──────────────────────────────────────────────────────
COIN_BLACKLIST = {
    "USDCUSDT", "TUSDUSDT", "BUSDUSDT", "USDTUSDT",
    "FDUSDUSDT", "LDOUSDT", "STETHUSDT", "WBTCUSDT",
    "SHIBUSDT", "PEPEUSDT", "FLOKIUSDT", "BTTUSDT",
}
MAX_SPREAD_PCT = 0.0015   # 0.15% max bid/ask spread

# ── NEW: Dynamic position sizing by confluence score ─────────────────────────
SCORE_SIZE_MAP = {4: 0.015, 5: 0.020, 6: 0.025}   # score → risk %

# Exit parameters
QUICK_PROFIT_R    = 0.8
TIME_EXIT_HOURS   = 6
TIME_EXIT_MIN_R   = 0.3
FUNDING_SPIKE_PCT = 0.0005
FUNDING_RATE_MAX  = 0.001
FUNDING_RATE_MIN  = -0.001

TRADE_JOURNAL_FILE = "trade_journal.json"

_instrument_cache: dict = {}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("bot.log")],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Trade Journal
# ---------------------------------------------------------------------------

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
# Discord
# ---------------------------------------------------------------------------

def discord_notify(message: str) -> None:
    if not DISCORD_WEBHOOK_URL:
        return
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": message}, timeout=10)
    except Exception as e:
        log.warning("Discord notify failed: %s", e)


# ---------------------------------------------------------------------------
# Dynamic watchlist
# ---------------------------------------------------------------------------

def get_dynamic_watchlist() -> list[str]:
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
        usdt_perps = [t for t in tickers if t["symbol"].endswith("USDT")]

        def _turnover(t):
            try:
                return float(t.get("turnover24h", 0))
            except (ValueError, TypeError):
                return 0.0

        top = sorted(usdt_perps, key=_turnover, reverse=True)[:DYNAMIC_SCAN_TOP_N]

        qualified = []
        for t in top:
            symbol = t["symbol"]
            # ── Blacklist filter ──
            if symbol in COIN_BLACKLIST:
                log.info("%s SKIP — blacklisted", symbol)
                continue
            base = symbol[:-4]
            if base in STABLECOIN_BASES:
                continue
            if _turnover(t) < DYNAMIC_VOLUME_MIN_USD:
                continue
            qualified.append(symbol)

        log.info("Dynamic scan: %d/%d symbols qualified", len(qualified), len(top))
        return qualified

    except Exception as e:
        log.warning("get_dynamic_watchlist failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# Bybit REST helpers
# ---------------------------------------------------------------------------

def _post_headers(body: str) -> dict:
    timestamp   = str(int(time.time() * 1000))
    recv_window = "5000"
    param_str   = timestamp + API_KEY + recv_window + body
    signature   = hmac.new(API_SECRET.encode(), param_str.encode(), hashlib.sha256).hexdigest()
    return {
        "X-BAPI-API-KEY":     API_KEY,
        "X-BAPI-TIMESTAMP":   timestamp,
        "X-BAPI-RECV-WINDOW": recv_window,
        "X-BAPI-SIGN":        signature,
        "Content-Type":       "application/json",
    }


def _signed_get(path: str, params: dict) -> Optional[dict]:
    timestamp   = str(int(time.time() * 1000))
    recv_window = "5000"
    param_str   = timestamp + API_KEY + recv_window + "&".join(
        f"{k}={v}" for k, v in sorted(params.items())
    )
    signature = hmac.new(API_SECRET.encode(), param_str.encode(), hashlib.sha256).hexdigest()
    headers = {
        "X-BAPI-API-KEY":     API_KEY,
        "X-BAPI-TIMESTAMP":   timestamp,
        "X-BAPI-RECV-WINDOW": recv_window,
        "X-BAPI-SIGN":        signature,
    }
    try:
        resp = requests.get(f"{BASE_URL}{path}", headers=headers, params=params, timeout=10)
        return resp.json()
    except Exception as e:
        log.error("GET %s failed: %s", path, e)
    return None


def get_account_balance() -> Optional[float]:
    data = _signed_get("/v5/account/wallet-balance", {"accountType": "UNIFIED", "coin": "USDT"})
    if data and data.get("retCode") == 0:
        for item in data["result"]["list"]:
            for coin in item.get("coin", []):
                if coin["coin"] == "USDT":
                    return float(coin["walletBalance"])
    return None


def get_open_positions() -> dict:
    data = _signed_get("/v5/position/list", {"category": "linear", "settleCoin": "USDT"})
    if data and data.get("retCode") == 0:
        return {
            item["symbol"]: item["side"]
            for item in data["result"].get("list", [])
            if float(item.get("size", 0)) != 0
        }
    return {}


def get_position_details() -> list:
    data = _signed_get("/v5/position/list", {"category": "linear", "settleCoin": "USDT"})
    if data and data.get("retCode") == 0:
        return [
            item for item in data["result"].get("list", [])
            if float(item.get("size", 0)) != 0
        ]
    return []


def set_breakeven_sl(symbol: str, entry_price: float) -> bool:
    payload = {
        "category":    "linear",
        "symbol":      symbol,
        "stopLoss":    str(round(entry_price, 4)),
        "slTriggerBy": "LastPrice",
        "positionIdx": 0,
    }
    body = json.dumps(payload)
    try:
        resp = requests.post(
            f"{BASE_URL}/v5/position/trading-stop",
            headers=_post_headers(body), data=body, timeout=10,
        )
        data = resp.json()
        return data.get("retCode") == 0
    except Exception as e:
        log.error("set_breakeven_sl exception for %s: %s", symbol, e)
    return False


def manage_breakeven(positions: list) -> None:
    for pos in positions:
        symbol = pos.get("symbol", "")
        side   = pos.get("side", "")
        try:
            entry = float(pos.get("avgPrice") or 0)
            sl    = float(pos.get("stopLoss") or 0)
            mark  = float(pos.get("markPrice") or 0)
        except (ValueError, TypeError):
            continue

        if entry == 0 or mark == 0 or sl == 0:
            continue
        if abs(sl - entry) / entry < 0.001:
            log.info("%s — SL already at breakeven (SL=%.4f entry=%.4f).", symbol, sl, entry)
            continue

        risk_dist = abs(entry - sl)
        tp1 = (entry + 1.5 * risk_dist) if side == "Buy" else (entry - 1.5 * risk_dist)
        tp1_reached = (mark >= tp1) if side == "Buy" else (mark <= tp1)

        if tp1_reached:
            if set_breakeven_sl(symbol, entry):
                log.info("%s — SL moved to breakeven %.4f", symbol, entry)
                discord_notify(
                    f"🔒 **Breakeven** {symbol} | TP1 `{mark:.4f}` | SL → entry `{entry:.4f}`"
                )
        else:
            pct = abs(tp1 - mark) / mark * 100
            log.info("%s — TP1 not yet reached (mark=%.4f tp1=%.4f, %.2f%% away)",
                     symbol, mark, tp1, pct)


def get_funding_rate(symbol: str) -> Optional[float]:
    try:
        resp = requests.get(
            f"{BASE_URL}/v5/market/tickers",
            params={"category": "linear", "symbol": symbol},
            timeout=10,
        )
        data = resp.json()
        if data.get("retCode") == 0 and data["result"]["list"]:
            return float(data["result"]["list"][0]["fundingRate"])
    except Exception as e:
        log.error("Funding rate error %s: %s", symbol, e)
    return None


def get_spread_pct(symbol: str) -> Optional[float]:
    """Return bid/ask spread as a fraction. None on error."""
    try:
        resp = requests.get(
            f"{BASE_URL}/v5/market/orderbook",
            params={"category": "linear", "symbol": symbol, "limit": 1},
            timeout=10,
        )
        data = resp.json()
        if data.get("retCode") == 0:
            bid = float(data["result"]["b"][0][0])
            ask = float(data["result"]["a"][0][0])
            mid = (bid + ask) / 2
            return (ask - bid) / mid if mid > 0 else None
    except Exception as e:
        log.warning("Spread check error %s: %s", symbol, e)
    return None


def get_klines(symbol: str, interval: str, limit: int = 300) -> Optional[pd.DataFrame]:
    try:
        resp = requests.get(
            f"{BASE_URL}/v5/market/kline",
            params={"category": "linear", "symbol": symbol,
                    "interval": interval, "limit": limit},
            timeout=10,
        )
        data = resp.json()
        if data.get("retCode") != 0:
            log.warning("Kline error for %s %s: %s", symbol, interval, data.get("retMsg"))
            return None
        df = pd.DataFrame(
            data["result"]["list"],
            columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"],
        )
        df = df.astype({"open": float, "high": float, "low": float,
                        "close": float, "volume": float})
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
        df.sort_values("timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception as e:
        log.error("Kline fetch failed for %s %s: %s", symbol, interval, e)
    return None


def load_instrument_cache(symbols: list) -> None:
    url = f"{BASE_URL}/v5/market/instruments-info"
    for symbol in symbols:
        try:
            resp = requests.get(url, params={"category": "linear", "symbol": symbol}, timeout=10)
            data = resp.json()
            if data.get("retCode") == 0 and data["result"]["list"]:
                lot = data["result"]["list"][0]["lotSizeFilter"]
                _instrument_cache[symbol] = {
                    "min_qty":  float(lot["minOrderQty"]),
                    "qty_step": float(lot["qtyStep"]),
                }
        except Exception as e:
            log.error("Instrument info error for %s: %s", symbol, e)


def snap_qty(symbol: str, qty: float) -> float:
    info = _instrument_cache.get(symbol)
    if not info:
        return round(qty, 3)
    step    = info["qty_step"]
    min_qty = info["min_qty"]
    snapped = math.floor(qty / step) * step
    decimals = len(str(step).rstrip("0").split(".")[-1]) if "." in str(step) else 0
    snapped  = round(snapped, decimals)
    return snapped if snapped >= min_qty else 0.0


def set_leverage(symbol: str, leverage: int) -> bool:
    payload = {
        "category":    "linear",
        "symbol":      symbol,
        "buyLeverage": str(leverage),
        "sellLeverage": str(leverage),
    }
    body = json.dumps(payload)
    try:
        resp = requests.post(
            f"{BASE_URL}/v5/position/set-leverage",
            headers=_post_headers(body), data=body, timeout=10,
        )
        data = resp.json()
        return data.get("retCode") in (0, 110043)
    except Exception as e:
        log.error("Set leverage error: %s", e)
    return False


def place_order(symbol: str, side: str, qty: float,
                sl_price: float, tp_price: float) -> Optional[dict]:
    payload = {
        "category":    "linear",
        "symbol":      symbol,
        "side":        side,
        "orderType":   "Market",
        "qty":         str(qty),
        "stopLoss":    str(round(sl_price, 4)),
        "takeProfit":  str(round(tp_price, 4)),
        "timeInForce": "IOC",
        "slTriggerBy": "LastPrice",
        "tpTriggerBy": "LastPrice",
        "positionIdx": 0,
    }
    body = json.dumps(payload)
    try:
        resp = requests.post(
            f"{BASE_URL}/v5/order/create",
            headers=_post_headers(body), data=body, timeout=10,
        )
        data = resp.json()
        if data.get("retCode") == 0:
            log.info("Order placed: %s %s qty=%s", side, symbol, qty)
            return data["result"]
        log.error("Order failed for %s: %s", symbol, data.get("retMsg"))
    except Exception as e:
        log.error("Place order exception: %s", e)
    return None


def close_position(symbol: str, side: str, qty: float) -> Optional[dict]:
    close_side = "Sell" if side == "Buy" else "Buy"
    payload = {
        "category":    "linear",
        "symbol":      symbol,
        "side":        close_side,
        "orderType":   "Market",
        "qty":         str(qty),
        "timeInForce": "IOC",
        "reduceOnly":  True,
        "positionIdx": 0,
    }
    body = json.dumps(payload)
    try:
        resp = requests.post(
            f"{BASE_URL}/v5/order/create",
            headers=_post_headers(body), data=body, timeout=10,
        )
        data = resp.json()
        if data.get("retCode") == 0:
            log.info("Position closed: %s %s qty=%s", symbol, close_side, qty)
            return data["result"]
        log.error("close_position failed for %s: %s", symbol, data.get("retMsg"))
    except Exception as e:
        log.error("close_position exception for %s: %s", symbol, e)
    return None


# ---------------------------------------------------------------------------
# Regime detection — symbol-specific
# ---------------------------------------------------------------------------

def _classify_ema(close: "pd.Series") -> str:
    ema9   = ta.trend.EMAIndicator(close, window=9).ema_indicator().iloc[-1]
    ema50  = ta.trend.EMAIndicator(close, window=50).ema_indicator().iloc[-1]
    ema200 = ta.trend.EMAIndicator(close, window=200).ema_indicator().iloc[-1]
    if ema9 > ema50 > ema200:
        return "BULL"
    if ema9 < ema50 < ema200:
        return "BEAR"
    return "RANGE"


def _compute_regime(proxy_symbol: str) -> dict:
    """Compute multi-TF regime for a given proxy symbol."""
    df4h = get_klines(proxy_symbol, "240", limit=250)
    df1h = get_klines(proxy_symbol, "60",  limit=250)

    fallback = {"regime_4h": "RANGE", "regime_1h": "RANGE",
                "allowed": "SKIP", "threshold": 5, "proxy": proxy_symbol}

    if df4h is None or len(df4h) < 200:
        log.warning("Regime: insufficient 4H data for %s — defaulting SKIP", proxy_symbol)
        return fallback

    r4h = _classify_ema(df4h["close"])
    r1h = "RANGE"
    if df1h is not None and len(df1h) >= 50:
        r1h = _classify_ema(df1h["close"])

    if r4h == "BULL":
        if r1h == "BULL":    allowed, threshold = "Buy",  4
        elif r1h == "RANGE": allowed, threshold = "Buy",  5
        else:                allowed, threshold = "SKIP", 5
    elif r4h == "BEAR":
        if r1h == "BEAR":    allowed, threshold = "Sell", 4
        elif r1h == "RANGE": allowed, threshold = "Sell", 5
        else:                allowed, threshold = "SKIP", 5
    else:
        allowed, threshold = "SKIP", 5

    log.info("REGIME [%s]: 4H=%s 1H=%s → allowed=%s threshold=%d",
             proxy_symbol, r4h, r1h, allowed, threshold)
    return {"regime_4h": r4h, "regime_1h": r1h,
            "allowed": allowed, "threshold": threshold, "proxy": proxy_symbol}


def get_market_regime() -> dict:
    """General market regime using BTC as proxy."""
    return _compute_regime("BTCUSDT")


def get_symbol_regime(symbol: str) -> dict:
    """
    Symbol-specific regime.
    Gold (XAUTUSDT) uses its own price action; all others use BTC.
    """
    proxy = "XAUTUSDT" if "XAU" in symbol else "BTCUSDT"
    return _compute_regime(proxy)


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def compute_indicators(df: pd.DataFrame) -> dict:
    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    volume = df["volume"]

    ema9   = ta.trend.EMAIndicator(close, window=9).ema_indicator()
    ema50  = ta.trend.EMAIndicator(close, window=50).ema_indicator()
    ema200 = ta.trend.EMAIndicator(close, window=200).ema_indicator()

    typical_price = (high + low + close) / 3
    vwap = (typical_price * volume).cumsum() / volume.cumsum()

    rsi = ta.momentum.RSIIndicator(close, window=14).rsi()

    stoch   = ta.momentum.StochRSIIndicator(close, window=14, smooth1=3, smooth2=3)
    stoch_k = stoch.stochrsi_k()
    stoch_d = stoch.stochrsi_d()

    macd_ind    = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    macd_line   = macd_ind.macd()
    macd_signal = macd_ind.macd_signal()

    return {
        "close": close.iloc[-1],
        "ema9":  ema9.iloc[-1],  "ema50":  ema50.iloc[-1],  "ema200": ema200.iloc[-1],
        "vwap":  vwap.iloc[-1],
        "rsi":   rsi.iloc[-1],
        "stoch_k": stoch_k.iloc[-1],  "stoch_d": stoch_d.iloc[-1],
        "macd":  macd_line.iloc[-1],  "macd_signal": macd_signal.iloc[-1],
    }


# ---------------------------------------------------------------------------
# Reversal signal checker
# ---------------------------------------------------------------------------

def check_reversal_signals(symbol: str, position_side: str,
                           df: pd.DataFrame) -> tuple[int, list[str]]:
    if len(df) < 3:
        return 0, []

    close  = df["close"]
    high   = df["high"]
    low    = df["low"]
    open_  = df["open"]

    ema9     = ta.trend.EMAIndicator(close, window=9).ema_indicator()
    rsi_ser  = ta.momentum.RSIIndicator(close, window=14).rsi()
    stoch    = ta.momentum.StochRSIIndicator(close, window=14, smooth1=3, smooth2=3)
    stoch_k  = stoch.stochrsi_k()
    stoch_d  = stoch.stochrsi_d()
    macd_ind = ta.trend.MACD(close, window_slow=26, window_fast=12, window_sign=9)
    hist_ser = macd_ind.macd_diff()

    c_close = close.iloc[-1];  p_close = close.iloc[-2]
    c_open  = open_.iloc[-1];  p_open  = open_.iloc[-2]
    c_high  = high.iloc[-1];   p_high  = high.iloc[-2]   # noqa
    c_low   = low.iloc[-1];    p_low   = low.iloc[-2]    # noqa
    c_ema9  = ema9.iloc[-1];   p_ema9  = ema9.iloc[-2]
    c_rsi   = rsi_ser.iloc[-1]; p_rsi  = rsi_ser.iloc[-2]
    c_k     = stoch_k.iloc[-1]; p_k    = stoch_k.iloc[-2]
    c_d     = stoch_d.iloc[-1]; p_d    = stoch_d.iloc[-2]
    c_hist  = hist_ser.iloc[-1]; p_hist = hist_ser.iloc[-2]

    signals: list[str] = []

    if position_side == "Sell":
        if p_rsi < 50 and c_rsi >= 50:
            signals.append("RSI>50")
        if p_k < 20 and p_k <= p_d and c_k > c_d:
            signals.append("StochRSI_cross_up")
        if p_hist < 0 and c_hist >= 0:
            signals.append("MACD_hist_flip_up")
        if p_close < p_ema9 and c_close >= c_ema9:
            signals.append("price>EMA9")
        body    = abs(c_close - c_open)
        lo_wick = min(c_close, c_open) - c_low
        hi_wick = c_high - max(c_close, c_open)
        hammer  = body > 0 and lo_wick >= 2 * body and hi_wick <= body
        bull_eng = (p_close < p_open and c_close > c_open
                    and c_open <= p_close and c_close >= p_open)
        if hammer or bull_eng:
            signals.append("BullishPattern")
    else:
        if p_rsi > 50 and c_rsi <= 50:
            signals.append("RSI<50")
        if p_k > 80 and p_k >= p_d and c_k < c_d:
            signals.append("StochRSI_cross_down")
        if p_hist > 0 and c_hist <= 0:
            signals.append("MACD_hist_flip_down")
        if p_close > p_ema9 and c_close <= c_ema9:
            signals.append("price<EMA9")
        body    = abs(c_close - c_open)
        lo_wick = min(c_close, c_open) - c_low
        hi_wick = c_high - max(c_close, c_open)
        shooting_star = body > 0 and hi_wick >= 2 * body and lo_wick <= body
        bear_eng = (p_close > p_open and c_close < c_open
                    and c_open >= p_close and c_close <= p_open)
        if shooting_star or bear_eng:
            signals.append("BearishPattern")

    log.info("%s reversal check (%s side): %d/5 — %s",
             symbol, position_side, len(signals), signals)
    return len(signals), signals


# ---------------------------------------------------------------------------
# 15M momentum — 3-candle confirmation (45 min)
# ---------------------------------------------------------------------------

def check_15m_momentum(symbol: str, position_side: str) -> bool:
    df = get_klines(symbol, "15", limit=60)
    if df is None or len(df) < 20:
        return False

    close = df["close"]
    ema9  = ta.trend.EMAIndicator(close, window=9).ema_indicator()
    rsi   = ta.momentum.RSIIndicator(close, window=14).rsi()

    if position_side == "Buy":
        c1 = close.iloc[-1] < ema9.iloc[-1] and rsi.iloc[-1] < 50
        c2 = close.iloc[-2] < ema9.iloc[-2] and rsi.iloc[-2] < 50
        c3 = close.iloc[-3] < ema9.iloc[-3] and rsi.iloc[-3] < 50
    else:
        c1 = close.iloc[-1] > ema9.iloc[-1] and rsi.iloc[-1] > 50
        c2 = close.iloc[-2] > ema9.iloc[-2] and rsi.iloc[-2] > 50
        c3 = close.iloc[-3] > ema9.iloc[-3] and rsi.iloc[-3] > 50

    confirmed = c1 and c2 and c3  # 3 consecutive candles = 45 min
    if confirmed:
        log.info("%s — 15M momentum confirmed (3/3 candles) for %s position",
                 symbol, position_side)
    return confirmed


# ---------------------------------------------------------------------------
# Unified exit manager
# ---------------------------------------------------------------------------

def manage_exits(position_details: list,
                 regime: Optional[dict] = None) -> dict[str, bool]:
    EXIT_THRESHOLD = 3
    closed: dict[str, bool] = {}
    now_ms = int(time.time() * 1000)

    for pos in position_details:
        symbol     = pos["symbol"]
        side       = pos["side"]
        qty        = float(pos.get("size", 0))
        entry      = float(pos.get("avgPrice", 0) or 0)
        sl         = float(pos.get("stopLoss", 0) or 0)
        upnl       = float(pos.get("unrealisedPnl", 0) or 0)
        created_ms = int(pos.get("createdTime", now_ms) or now_ms)

        if entry == 0 or qty == 0:
            continue

        risk_dist  = abs(entry - sl) if sl != 0 else 0.0
        r_usdt     = risk_dist * qty
        hours_held = (now_ms - created_ms) / 3_600_000
        pnl_str    = f"+${upnl:.2f}" if upnl >= 0 else f"-${abs(upnl):.2f}"
        dlabel     = "LONG" if side == "Buy" else "SHORT"

        exit_reason: Optional[str] = None
        icon  = "⚡"
        label = "QUICK EXIT"

        # ── Priority 1: 0.8R quick profit ──
        if r_usdt > 0 and upnl >= QUICK_PROFIT_R * r_usdt:
            exit_reason = f"0.8R profit (`{upnl:.2f}` ≥ `{QUICK_PROFIT_R * r_usdt:.2f}` USDT)"

        # ── Priority 2: reversal 3/5 + regime-conflict bonus ──
        if exit_reason is None:
            df1h = get_klines(symbol, "60", limit=100)
            if df1h is not None and len(df1h) >= 3:
                rev_count, rev_fired = check_reversal_signals(symbol, side, df1h)
                regime_bonus = 0
                if regime:
                    allowed = regime.get("allowed", "SKIP")
                    pos_dir = "Buy" if side == "Buy" else "Sell"
                    if allowed == "SKIP" or allowed != pos_dir:
                        regime_bonus = 1
                        log.info("%s — regime conflict bonus +1 (allowed=%s, pos=%s). "
                                 "Reversal: %d→%d/5 %s",
                                 symbol, allowed, pos_dir,
                                 rev_count, rev_count + regime_bonus, rev_fired)
                if rev_count + regime_bonus >= EXIT_THRESHOLD:
                    sig_list = rev_fired + (["RegimeConflict"] if regime_bonus else [])
                    exit_reason = (f"reversal `{rev_count + regime_bonus}/5` "
                                   f"[{', '.join(sig_list)}]")

        # ── Priority 3: 15M momentum (3-candle) ──
        if exit_reason is None and check_15m_momentum(symbol, side):
            exit_reason = "15M momentum (3-candle confirmed)"

        # ── Priority 4: funding spike while in profit ──
        if exit_reason is None and upnl > 0:
            funding = get_funding_rate(symbol)
            if funding is not None and abs(funding) >= FUNDING_SPIKE_PCT:
                exit_reason = f"funding spike `{funding * 100:.4f}%` while in profit"

        # ── Priority 5: time exit ──
        if exit_reason is None:
            if hours_held > TIME_EXIT_HOURS and r_usdt > 0 and upnl < TIME_EXIT_MIN_R * r_usdt:
                icon  = "⏰"
                label = "TIME EXIT"
                exit_reason = (f"held `{hours_held:.1f}h`, profit `{upnl:.2f}` "
                               f"< `{TIME_EXIT_MIN_R * r_usdt:.2f}` USDT (0.3R)")

        if exit_reason is None:
            continue

        log.warning("%s — %s [%s] | %s | uPnL=%s | held=%.1fh",
                    symbol, label, dlabel, exit_reason, pnl_str, hours_held)

        result = close_position(symbol, side, qty)
        if result:
            discord_notify(
                f"{icon} **{label} {dlabel} {symbol}** | "
                f"{exit_reason} | PnL: `{pnl_str}` | Held: `{hours_held:.1f}h`"
            )
            closed[symbol] = True
        else:
            discord_notify(f"⚠️ **{symbol}** — exit order FAILED | {exit_reason}")

    return closed


# ---------------------------------------------------------------------------
# Signal scoring
# ---------------------------------------------------------------------------

def score_signals(ind4h: dict, ind1h: dict) -> tuple[int, str, list]:
    buy_signals  = 0
    sell_signals = 0
    details: list[str] = []

    if ind4h["close"] > ind4h["ema9"] > ind4h["ema50"] > ind4h["ema200"]:
        buy_signals  += 1; details.append("EMA_bull")
    elif ind4h["close"] < ind4h["ema9"] < ind4h["ema50"] < ind4h["ema200"]:
        sell_signals += 1; details.append("EMA_bear")

    if ind1h["close"] > ind1h["vwap"]:
        buy_signals  += 1; details.append("VWAP_bull")
    elif ind1h["close"] < ind1h["vwap"]:
        sell_signals += 1; details.append("VWAP_bear")

    if ind1h["rsi"] < 40:
        buy_signals  += 1; details.append("RSI_bull")
    elif ind1h["rsi"] > 60:
        sell_signals += 1; details.append("RSI_bear")

    if ind1h["stoch_k"] > ind1h["stoch_d"] and ind1h["stoch_k"] < 80:
        buy_signals  += 1; details.append("StochRSI_bull")
    elif ind1h["stoch_k"] < ind1h["stoch_d"] and ind1h["stoch_k"] > 20:
        sell_signals += 1; details.append("StochRSI_bear")

    if ind4h["macd"] > ind4h["macd_signal"]:
        buy_signals  += 1; details.append("MACD_bull")
    elif ind4h["macd"] < ind4h["macd_signal"]:
        sell_signals += 1; details.append("MACD_bear")

    if ind1h["ema9"] > ind1h["ema50"]:
        buy_signals  += 1; details.append("EMA1H_bull")
    elif ind1h["ema9"] < ind1h["ema50"]:
        sell_signals += 1; details.append("EMA1H_bear")

    if buy_signals >= sell_signals:
        return buy_signals, "Buy", details
    return sell_signals, "Sell", details


def calculate_position(balance: float, entry_price: float,
                       sl_price: float, leverage: int,
                       score: int = 4) -> float:
    """Position size scaled by confluence score."""
    risk_pct    = SCORE_SIZE_MAP.get(score, RISK_PER_TRADE)
    risk_amount = balance * risk_pct
    sl_distance = abs(entry_price - sl_price)
    if sl_distance == 0:
        return 0.0
    qty_usdt = (risk_amount / sl_distance) * entry_price
    max_qty_usdt = balance * leverage
    qty_usdt = min(qty_usdt, max_qty_usdt)
    return round(qty_usdt / entry_price, 3)


# ---------------------------------------------------------------------------
# Loss guards
# ---------------------------------------------------------------------------

def daily_loss_exceeded(starting: float, current: float) -> bool:
    if starting <= 0:
        return False
    return (starting - current) / starting >= DAILY_LOSS_LIMIT


def weekly_loss_exceeded(starting: float, current: float) -> bool:
    if starting <= 0:
        return False
    return (starting - current) / starting >= WEEKLY_LOSS_LIMIT


# ---------------------------------------------------------------------------
# Daily performance summary
# ---------------------------------------------------------------------------

def send_daily_summary(balance: float) -> None:
    journal = load_journal()
    today   = datetime.date.today().isoformat()
    today_trades = [t for t in journal if str(t.get("timestamp", "")).startswith(today)]

    wins   = sum(1 for t in today_trades if t.get("order_result"))
    total  = len(today_trades)
    wr_pct = (wins / total * 100) if total > 0 else 0

    open_pos = get_open_positions()
    open_str = ", ".join(f"`{s}`" for s in open_pos) if open_pos else "None"

    discord_notify(
        f"📈 **DAILY SUMMARY** | {today}\n"
        f"• Trades taken: `{total}`\n"
        f"• Win rate: `{wr_pct:.0f}%` ({wins}/{total})\n"
        f"• Open positions: {open_str}\n"
        f"• Balance: `{balance:.2f}` USDT"
    )


# ---------------------------------------------------------------------------
# Main scan loop
# ---------------------------------------------------------------------------

def scan_symbol(symbol: str, balance: float,
                regime: Optional[dict] = None) -> Optional[dict]:
    log.info("Scanning %s …", symbol)

    # ── Spread check ──────────────────────────────────────────────────────
    spread = get_spread_pct(symbol)
    if spread is not None and spread > MAX_SPREAD_PCT:
        log.info("%s SKIP — spread too wide (%.4f%%)", symbol, spread * 100)
        return None

    df4h = get_klines(symbol, "240")
    df1h = get_klines(symbol, "60")

    if df4h is None or df1h is None or len(df4h) < 200 or len(df1h) < 50:
        log.warning("Insufficient data for %s", symbol)
        return None

    # ── Volume confirmation ────────────────────────────────────────────────
    avg_vol_20  = df4h["volume"].iloc[-21:-1].mean()
    current_vol = df4h["volume"].iloc[-1]
    vol_ratio   = current_vol / avg_vol_20 if avg_vol_20 > 0 else 0.0
    if vol_ratio < 1.2:
        log.info("%s — Volume %.2fx avg — SKIP", symbol, vol_ratio)
        return None

    ind4h = compute_indicators(df4h)
    ind1h = compute_indicators(df1h)
    score, direction, details = score_signals(ind4h, ind1h)

    log.info("%s | score=%d/6 dir=%s signals=%s", symbol, score, direction, details)

    # ── Long guard ────────────────────────────────────────────────────────
    if direction == "Buy" and not LONG_ENABLED:
        log.info("%s SKIP — Long entries disabled (LONG_ENABLED=False)", symbol)
        return None

    # ── Symbol-specific regime ────────────────────────────────────────────
    sym_regime = get_symbol_regime(symbol)
    allowed    = sym_regime.get("allowed", "SKIP")
    threshold  = sym_regime.get("threshold", CONFLUENCE_THRESHOLD)

    if allowed == "SKIP":
        log.info("%s SKIP — symbol regime conflict/range (proxy=%s)",
                 symbol, sym_regime.get("proxy"))
        return None
    if allowed != direction:
        log.info("%s SKIP — symbol regime allows %s only, signal=%s", symbol, allowed, direction)
        return None
    if score < threshold:
        log.info("%s SKIP — score %d < required %d", symbol, score, threshold)
        return None

    # ── Correlation filter ────────────────────────────────────────────────
    if symbol in BTC_CORRELATED:
        open_pos = get_open_positions()
        btc_corr_count = sum(1 for s in open_pos if s in BTC_CORRELATED)
        if btc_corr_count >= MAX_CORRELATION_POSITIONS:
            log.info("%s SKIP — BTC-correlated cap reached (%d/%d)",
                     symbol, btc_corr_count, MAX_CORRELATION_POSITIONS)
            return None

    entry      = ind1h["close"]
    atr_approx = (df1h["high"].iloc[-14:] - df1h["low"].iloc[-14:]).mean()

    # ── TP snapped to nearest S/R pivot ───────────────────────────────────
    recent_4h  = df4h.iloc[-20:]
    pivot_high = recent_4h["high"].max()
    pivot_low  = recent_4h["low"].min()
    pivot_mid  = (pivot_high + pivot_low) / 2

    if direction == "Buy":
        sl     = entry - 2 * atr_approx
        raw_tp = entry + 4 * atr_approx
        candidates = [p for p in [pivot_mid, pivot_high] if entry < p <= raw_tp * 1.02]
        tp = min(candidates) if candidates else raw_tp
    else:
        sl     = entry + 2 * atr_approx
        raw_tp = entry - 4 * atr_approx
        candidates = [p for p in [pivot_mid, pivot_low] if entry > p >= raw_tp * 0.98]
        tp = max(candidates) if candidates else raw_tp

    leverage = min(MAX_LEVERAGE, 10)
    qty      = calculate_position(balance, entry, sl, leverage, score)
    qty      = snap_qty(symbol, qty)

    if qty <= 0:
        log.warning("Calculated qty is 0 for %s, skipping", symbol)
        return None

    set_leverage(symbol, leverage)
    result = place_order(symbol, direction, qty, sl, tp)

    trade = {
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "symbol":    symbol,
        "direction": direction,
        "score":     score,
        "signals":   details,
        "entry":     entry,
        "sl":        round(sl, 4),
        "tp":        round(tp, 4),
        "qty":       qty,
        "leverage":  leverage,
        "order_result": result,
    }
    log_trade(trade)

    if result:
        emoji = "🟢" if direction == "Buy" else "🔴"
        discord_notify(
            f"{emoji} **{direction} {symbol}** | Score `{score}/6` | "
            f"Entry `{entry:.4f}` | SL `{round(sl, 4)}` | TP `{round(tp, 4)}` | "
            f"Qty `{qty}` × {leverage}x"
        )

    return trade


# ---------------------------------------------------------------------------
# Keep-alive HTTP server
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
        pass


def _start_http_server():
    import socket
    for attempt in range(5):
        port = HTTP_PORT + attempt
        try:
            server = HTTPServer(("0.0.0.0", port), _PingHandler)
            server.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            log.info("Keep-alive HTTP server on port %d", port)
            server.serve_forever()
            return
        except OSError as e:
            log.warning("Port %d in use (%s), trying %d …", port, e.strerror, port + 1)
    log.error("Could not bind HTTP server — keep-alive disabled")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    log.info("=" * 60)
    log.info("APEX Bybit Futures Trading Bot starting")
    log.info("Dynamic scan: top %d USDT perps, vol ≥ $%.0fM",
             DYNAMIC_SCAN_TOP_N, DYNAMIC_VOLUME_MIN_USD / 1_000_000)
    log.info("Long enabled: %s", LONG_ENABLED)
    log.info("=" * 60)

    discord_notify(
        f"🚀 **APEX Bot started** | Dynamic scan: top {DYNAMIC_SCAN_TOP_N} USDT perps "
        f"with vol ≥ ${DYNAMIC_VOLUME_MIN_USD/1_000_000:.0f}M | "
        f"Long: {'✅' if LONG_ENABLED else '❌ disabled'}"
    )

    threading.Thread(target=_start_http_server, daemon=True).start()

    if not API_KEY or not API_SECRET:
        log.error("API keys not set — exiting.")
        return

    starting_balance = get_account_balance()
    if starting_balance is None:
        log.error("Could not fetch balance. Check API credentials.")
        return

    log.info("Starting balance: %.2f USDT", starting_balance)

    day_start              = datetime.date.today()
    week_start             = day_start - datetime.timedelta(days=day_start.weekday())
    week_starting_balance  = starting_balance
    weekly_pause_until     = None
    last_health_check      = datetime.datetime.utcnow()
    daily_summary_sent     = False

    while True:
        now_utc = datetime.datetime.utcnow()
        today   = datetime.date.today()

        # ── Weekly pause check ──
        if weekly_pause_until and now_utc < weekly_pause_until:
            remaining = (weekly_pause_until - now_utc).total_seconds() / 3600
            log.warning("Weekly loss limit active — paused %.1fh remaining.", remaining)
            time.sleep(SCAN_INTERVAL_SECONDS)
            continue

        # ── Daily reset ──
        if today != day_start:
            day_start         = today
            starting_balance  = get_account_balance() or starting_balance
            daily_summary_sent = False
            log.info("New day — reset daily baseline to %.2f USDT", starting_balance)

        # ── Weekly reset ──
        current_week = today - datetime.timedelta(days=today.weekday())
        if current_week != week_start:
            week_start            = current_week
            week_starting_balance = get_account_balance() or starting_balance
            log.info("New week — reset weekly baseline to %.2f USDT", week_starting_balance)

        # ── Daily summary at 00:00 UTC ──
        if now_utc.hour == 0 and now_utc.minute < 10 and not daily_summary_sent:
            current_bal = get_account_balance()
            if current_bal:
                send_daily_summary(current_bal)
            daily_summary_sent = True

        # ── 6-hour health check ──
        if (now_utc - last_health_check).total_seconds() >= 21_600:
            current_bal = get_account_balance() or 0
            open_pos    = get_open_positions()
            next_scan   = now_utc + datetime.timedelta(seconds=SCAN_INTERVAL_SECONDS)
            discord_notify(
                f"💚 **APEX Bot alive** | "
                f"Positions: `{len(open_pos)}/{MAX_OPEN_POSITIONS}` | "
                f"Balance: `{current_bal:.2f}` USDT | "
                f"Next scan: `{next_scan.strftime('%H:%M')} UTC`"
            )
            last_health_check = now_utc

        current_balance = get_account_balance()
        if current_balance is None:
            log.warning("Could not fetch balance; retrying next cycle")
            time.sleep(SCAN_INTERVAL_SECONDS)
            continue

        # ── Daily loss guard ──
        if daily_loss_exceeded(starting_balance, current_balance):
            loss_pct = (starting_balance - current_balance) / starting_balance * 100
            log.warning("Daily loss limit — pausing until tomorrow.")
            discord_notify(
                f"🛑 **Daily loss limit hit** | "
                f"`{starting_balance:.2f}` → `{current_balance:.2f}` USDT "
                f"(`-{loss_pct:.1f}%`) | Pausing until tomorrow"
            )
            time.sleep(SCAN_INTERVAL_SECONDS)
            continue

        # ── Weekly loss guard ──
        if weekly_loss_exceeded(week_starting_balance, current_balance):
            weekly_pause_until = now_utc + datetime.timedelta(hours=48)
            loss_pct = (week_starting_balance - current_balance) / week_starting_balance * 100
            log.warning("Weekly loss limit hit — pausing 48 h.")
            discord_notify(
                f"🛑 **WEEKLY LOSS LIMIT HIT** | "
                f"`{week_starting_balance:.2f}` → `{current_balance:.2f}` USDT "
                f"(`-{loss_pct:.1f}%`) | Pausing **48 hours**"
            )
            time.sleep(SCAN_INTERVAL_SECONDS)
            continue

        # ── Regime (BTC proxy for general market) ──
        regime = get_market_regime()

        # ── Position management ──
        open_positions = get_open_positions()
        pos_count      = len(open_positions)
        log.info("Open positions: %d/%d %s | Balance: %.2f USDT",
                 pos_count, MAX_OPEN_POSITIONS,
                 list(open_positions.keys()), current_balance)

        if pos_count > 0:
            position_details = get_position_details()
            if position_details:
                manage_breakeven(position_details)
                closed = manage_exits(position_details, regime)
                if closed:
                    open_positions = get_open_positions()
                    pos_count      = len(open_positions)

        if pos_count >= MAX_OPEN_POSITIONS:
            next_scan = now_utc + datetime.timedelta(seconds=SCAN_INTERVAL_SECONDS)
            log.info("Max positions (%d/%d). Skipping scan. Next at %s UTC",
                     pos_count, MAX_OPEN_POSITIONS, next_scan.strftime("%H:%M:%S"))
            time.sleep(SCAN_INTERVAL_SECONDS)
            continue

        # ── Dynamic watchlist ──
        watchlist = get_dynamic_watchlist()
        if not watchlist:
            log.warning("Empty watchlist — skipping cycle")
            time.sleep(SCAN_INTERVAL_SECONDS)
            continue

        load_instrument_cache(watchlist)

        for symbol in watchlist:
            if len(open_positions) >= MAX_OPEN_POSITIONS:
                log.info("Max positions reached — stopping scan early.")
                break
            if symbol in open_positions:
                log.info("%s already open — skipping.", symbol)
                continue

            funding = get_funding_rate(symbol)
            if funding is None:
                continue
            if not (FUNDING_RATE_MIN <= funding <= FUNDING_RATE_MAX):
                log.info("%s SKIP — funding out of range (%.4f%%)", symbol, funding * 100)
                continue

            try:
                trade = scan_symbol(symbol, current_balance, regime)
                if trade and trade.get("order_result"):
                    open_positions = get_open_positions()
            except Exception as e:
                log.error("Unexpected error scanning %s: %s", symbol, e)

            time.sleep(2)

        next_scan    = now_utc + datetime.timedelta(seconds=SCAN_INTERVAL_SECONDS)
        regime_label = (f"{regime.get('regime_4h','?')}/{regime.get('regime_1h','?')}"
                        f" → {regime.get('allowed','?')}")
        log.info("Scan complete. Next scan at %s UTC", next_scan.strftime("%H:%M:%S"))
        discord_notify(
            f"📊 **Scan complete** | Regime: `{regime_label}` | "
            f"Positions: `{len(open_positions)}/{MAX_OPEN_POSITIONS}` | "
            f"Balance: `{current_balance:.2f}` USDT | "
            f"Next scan: `{next_scan.strftime('%H:%M')} UTC`"
        )
        time.sleep(SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
