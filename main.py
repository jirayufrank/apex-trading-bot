#!/usr/bin/env python3
"""
APEX Bybit Futures Trading Bot — v3 with Self-Learning System
New in v3:
  - Outcome tracking: records PnL, win/loss on every close
  - Signal weight learning: Long/Short weights adjusted separately
  - Coin behavior profiling: win rate, best hours, ATR per coin
  - Score bonus/penalty per coin based on historical win rate
  - Time-of-day position sizing (boost/reduce by hour)
  - Weekly learning report to Discord every Sunday 00:00 UTC
  - Safety limits: min 20 trades before adjusting, max ±50% weight change
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
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

import requests
import pandas as pd
import numpy as np
import ta

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_KEY             = os.environ.get("BYBIT_API_KEY", "")
API_SECRET          = os.environ.get("BYBIT_API_SECRET", "")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")

BASE_URL = "https://api.bybit.com"

DYNAMIC_SCAN_TOP_N     = 50
DYNAMIC_VOLUME_MIN_USD = 50_000_000
STABLECOIN_BASES       = {"USDC","BUSD","TUSD","USDP","DAI","FDUSD","PYUSD"}

RISK_PER_TRADE        = 0.02
MAX_LEVERAGE          = 10
MAX_OPEN_POSITIONS    = 3
DAILY_LOSS_LIMIT      = 0.05
WEEKLY_LOSS_LIMIT     = 0.10
SCAN_INTERVAL_SECONDS = 3600
CONFLUENCE_THRESHOLD  = 4

LONG_ENABLED = True

MAX_CORRELATION_POSITIONS = 2
BTC_CORRELATED = {
    "ETHUSDT","SOLUSDT","BNBUSDT","AVAXUSDT",
    "LINKUSDT","DOTUSDT","MATICUSDT","LTCUSDT",
    "ADAUSDT","ATOMUSDT","NEARUSDT","APTUSDT",
}

COIN_BLACKLIST = {
    "USDCUSDT","TUSDUSDT","BUSDUSDT","USDTUSDT",
    "FDUSDUSDT","LDOUSDT","STETHUSDT","WBTCUSDT",
    "SHIBUSDT","PEPEUSDT","FLOKIUSDT","BTTUSDT",
}
MAX_SPREAD_PCT = 0.0015
SCORE_SIZE_MAP = {4: 0.015, 5: 0.020, 6: 0.025}

QUICK_PROFIT_R    = 0.8
TIME_EXIT_HOURS   = 6
TIME_EXIT_MIN_R   = 0.3
FUNDING_SPIKE_PCT = 0.0005
FUNDING_RATE_MAX  = 0.001
FUNDING_RATE_MIN  = -0.001

TRADE_JOURNAL_FILE  = "trade_journal.json"
LEARNING_STATE_FILE = "learning_state.json"

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
# Learning state
# ---------------------------------------------------------------------------

DEFAULT_LEARNING_STATE = {
    "signal_weights": {
        "Buy":  {"EMA_bull":1.0,"VWAP_bull":1.0,"RSI_bull":1.0,"StochRSI_bull":1.0,"MACD_bull":1.0,"EMA1H_bull":1.0},
        "Sell": {"EMA_bear":1.0,"VWAP_bear":1.0,"RSI_bear":1.0,"StochRSI_bear":1.0,"MACD_bear":1.0,"EMA1H_bear":1.0},
    },
    "coin_profiles": {},
    "last_updated": None,
    "total_trades_analyzed": 0,
}

MIN_TRADES_TO_LEARN = 20
MAX_WEIGHT          = 1.5
MIN_WEIGHT          = 0.5


def load_learning_state() -> dict:
    if os.path.exists(LEARNING_STATE_FILE):
        try:
            with open(LEARNING_STATE_FILE) as f:
                state = json.load(f)
            for side in ("Buy","Sell"):
                if side not in state.get("signal_weights",{}):
                    state["signal_weights"][side] = DEFAULT_LEARNING_STATE["signal_weights"][side].copy()
            return state
        except Exception as e:
            log.warning("Could not load learning state: %s", e)
    return json.loads(json.dumps(DEFAULT_LEARNING_STATE))


def save_learning_state(state: dict) -> None:
    with open(LEARNING_STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Trade Journal
# ---------------------------------------------------------------------------

def load_journal() -> list:
    if os.path.exists(TRADE_JOURNAL_FILE):
        with open(TRADE_JOURNAL_FILE) as f:
            return json.load(f)
    return []


def save_journal(journal: list) -> None:
    with open(TRADE_JOURNAL_FILE, "w") as f:
        json.dump(journal, f, indent=2, default=str)


def log_trade(entry: dict) -> None:
    j = load_journal()
    j.append(entry)
    save_journal(j)


def update_trade_outcome(symbol: str, direction: str, pnl: float) -> None:
    """Find most recent open trade for symbol+direction and record outcome."""
    journal = load_journal()
    for trade in reversed(journal):
        if (trade.get("symbol") == symbol
                and trade.get("direction") == direction
                and trade.get("outcome") is None):
            trade["outcome"]  = "win" if pnl > 0 else "loss"
            trade["pnl"]      = round(pnl, 4)
            trade["closed_at"]= datetime.datetime.utcnow().isoformat()
            trade["hour_utc"] = datetime.datetime.utcnow().hour
            save_journal(journal)
            log.info("Outcome: %s %s pnl=%.4f → %s", symbol, direction, pnl, trade["outcome"])
            return


# ---------------------------------------------------------------------------
# Discord
# ---------------------------------------------------------------------------

def discord_notify(message: str) -> None:
    if not DISCORD_WEBHOOK_URL:
        return
    try:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": message}, timeout=10)
    except Exception as e:
        log.warning("Discord failed: %s", e)


# ---------------------------------------------------------------------------
# Weekly Self-Learning Engine
# ---------------------------------------------------------------------------

def run_weekly_learning() -> None:
    log.info("Running weekly self-learning …")
    journal   = load_journal()
    state     = load_learning_state()
    completed = [t for t in journal if t.get("outcome") in ("win","loss")]

    if len(completed) < MIN_TRADES_TO_LEARN:
        discord_notify(
            f"📚 **Weekly Learning** | Only `{len(completed)}/{MIN_TRADES_TO_LEARN}` "
            f"completed trades — skipping weight update."
        )
        return

    # 1. Signal win rates — Buy/Sell separately
    sig_stats: dict[str, dict[str, dict]] = {"Buy":{},"Sell":{}}
    for trade in completed:
        d = trade.get("direction","")
        if d not in ("Buy","Sell"): continue
        for sig in trade.get("signals",[]):
            b = sig_stats[d].setdefault(sig, {"wins":0,"total":0})
            b["total"] += 1
            if trade.get("outcome") == "win": b["wins"] += 1

    for direction in ("Buy","Sell"):
        for sig, stats in sig_stats[direction].items():
            if stats["total"] < 5: continue
            wr = stats["wins"] / stats["total"]
            cur = state["signal_weights"][direction].get(sig, 1.0)
            if   wr >= 0.70: new_w = min(cur * 1.15, MAX_WEIGHT)
            elif wr >= 0.55: new_w = cur
            else:            new_w = max(cur * 0.85, MIN_WEIGHT)
            state["signal_weights"][direction][sig] = round(new_w, 3)

    # 2. Coin profiling
    coin_data: dict[str,dict] = {}
    for trade in completed:
        sym = trade.get("symbol","")
        if not sym: continue
        p = coin_data.setdefault(sym, {"wl":0,"tl":0,"ws":0,"ts":0,"hrs":[],"pnls":[]})
        d = trade.get("direction","")
        o = trade.get("outcome")
        h = trade.get("hour_utc")
        if d == "Buy":  p["tl"]+=1; (p.__setitem__("wl", p["wl"]+1) if o=="win" else None)
        elif d=="Sell": p["ts"]+=1; (p.__setitem__("ws", p["ws"]+1) if o=="win" else None)
        if h is not None: p["hrs"].append(h)
        p["pnls"].append(trade.get("pnl",0))

    profiles = state.setdefault("coin_profiles",{})
    for sym, data in coin_data.items():
        total = data["tl"] + data["ts"]
        if total < 3: continue
        wr_l = data["wl"]/data["tl"] if data["tl"]>0 else None
        wr_s = data["ws"]/data["ts"] if data["ts"]>0 else None
        wr_a = (data["wl"]+data["ws"])/total
        score_bonus = -1 if wr_a>=0.70 else (0 if wr_a>=0.50 else (1 if wr_a>=0.35 else 2))
        winning_hrs = [t.get("hour_utc") for t in completed
                       if t.get("symbol")==sym and t.get("outcome")=="win" and t.get("hour_utc") is not None]
        best_hrs = [h for h,_ in Counter(winning_hrs).most_common(3)] if winning_hrs else []
        profiles[sym] = {
            "win_rate_long":  round(wr_l,3) if wr_l is not None else None,
            "win_rate_short": round(wr_s,3) if wr_s is not None else None,
            "win_rate_all":   round(wr_a,3),
            "score_bonus":    score_bonus,
            "best_hours_utc": best_hrs,
            "trades_count":   total,
        }

    state["last_updated"]         = datetime.datetime.utcnow().isoformat()
    state["total_trades_analyzed"]= len(completed)
    save_learning_state(state)

    # Report
    def _wr(sig, d):
        s = sig_stats[d].get(sig)
        return f"{s['wins']/s['total']*100:.0f}%" if s and s["total"]>=5 else "n/a"
    def _arr(w): return "↑" if w>1.05 else ("↓" if w<0.95 else "→")

    sell_lines = "\n".join(
        f"  `{s}`: {_wr(s,'Sell')} {_arr(w)} w={w}"
        for s,w in state["signal_weights"]["Sell"].items()
    )
    buy_lines = "\n".join(
        f"  `{s}`: {_wr(s,'Buy')} {_arr(w)} w={w}"
        for s,w in state["signal_weights"]["Buy"].items()
    )
    top_coins = sorted(profiles.items(), key=lambda x:x[1].get("win_rate_all",0), reverse=True)[:5]
    coin_lines = "\n".join(
        f"  `{sym}`: {p['win_rate_all']*100:.0f}% ({p['trades_count']} trades) "
        f"bonus={p['score_bonus']:+d} hrs={p['best_hours_utc']}"
        for sym,p in top_coins
    )
    overall = sum(1 for t in completed if t["outcome"]=="win")/len(completed)*100
    discord_notify(
        f"📚 **WEEKLY LEARNING REPORT** | {datetime.date.today()}\n"
        f"Analyzed: `{len(completed)}` trades | Win rate: `{overall:.0f}%`\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"**SHORT signals (↑=boost ↓=reduce):**\n{sell_lines}\n"
        f"**LONG signals:**\n{buy_lines}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"**Coin profiles (top 5):**\n{coin_lines}"
    )
    log.info("Weekly learning done. %d trades analyzed.", len(completed))


# ---------------------------------------------------------------------------
# Dynamic watchlist
# ---------------------------------------------------------------------------

def get_dynamic_watchlist() -> list[str]:
    try:
        resp = requests.get(f"{BASE_URL}/v5/market/tickers",
                            params={"category":"linear"}, timeout=10)
        data = resp.json()
        if data.get("retCode") != 0: return []
        def _tv(t):
            try: return float(t.get("turnover24h",0))
            except: return 0.0
        top = sorted([t for t in data["result"]["list"] if t["symbol"].endswith("USDT")],
                     key=_tv, reverse=True)[:DYNAMIC_SCAN_TOP_N]
        qualified = []
        for t in top:
            sym = t["symbol"]
            if sym in COIN_BLACKLIST: continue
            if sym[:-4] in STABLECOIN_BASES: continue
            if _tv(t) < DYNAMIC_VOLUME_MIN_USD: continue
            qualified.append(sym)
        log.info("Watchlist: %d/%d qualified", len(qualified), len(top))
        return qualified
    except Exception as e:
        log.warning("get_dynamic_watchlist: %s", e)
        return []


# ---------------------------------------------------------------------------
# Bybit REST helpers
# ---------------------------------------------------------------------------

def _post_headers(body: str) -> dict:
    ts  = str(int(time.time()*1000))
    rw  = "5000"
    sig = hmac.new(API_SECRET.encode(), (ts+API_KEY+rw+body).encode(), hashlib.sha256).hexdigest()
    return {"X-BAPI-API-KEY":API_KEY,"X-BAPI-TIMESTAMP":ts,
            "X-BAPI-RECV-WINDOW":rw,"X-BAPI-SIGN":sig,"Content-Type":"application/json"}

def _signed_get(path: str, params: dict) -> Optional[dict]:
    ts  = str(int(time.time()*1000))
    rw  = "5000"
    qs  = "&".join(f"{k}={v}" for k,v in sorted(params.items()))
    sig = hmac.new(API_SECRET.encode(), (ts+API_KEY+rw+qs).encode(), hashlib.sha256).hexdigest()
    hdrs= {"X-BAPI-API-KEY":API_KEY,"X-BAPI-TIMESTAMP":ts,"X-BAPI-RECV-WINDOW":rw,"X-BAPI-SIGN":sig}
    try:
        return requests.get(f"{BASE_URL}{path}", headers=hdrs, params=params, timeout=10).json()
    except Exception as e:
        log.error("GET %s: %s", path, e)
    return None

def get_account_balance() -> Optional[float]:
    data = _signed_get("/v5/account/wallet-balance", {"accountType":"UNIFIED","coin":"USDT"})
    if data and data.get("retCode")==0:
        for item in data["result"]["list"]:
            for coin in item.get("coin",[]):
                if coin["coin"]=="USDT": return float(coin["walletBalance"])
    return None

def get_open_positions() -> dict:
    data = _signed_get("/v5/position/list", {"category":"linear","settleCoin":"USDT"})
    if data and data.get("retCode")==0:
        return {item["symbol"]:item["side"] for item in data["result"].get("list",[])
                if float(item.get("size",0))!=0}
    return {}

def get_position_details() -> list:
    data = _signed_get("/v5/position/list", {"category":"linear","settleCoin":"USDT"})
    if data and data.get("retCode")==0:
        return [item for item in data["result"].get("list",[]) if float(item.get("size",0))!=0]
    return []

def set_breakeven_sl(symbol: str, entry_price: float) -> bool:
    payload = {"category":"linear","symbol":symbol,"stopLoss":str(round(entry_price,4)),
               "slTriggerBy":"LastPrice","positionIdx":0}
    body = json.dumps(payload)
    try:
        resp = requests.post(f"{BASE_URL}/v5/position/trading-stop",
                             headers=_post_headers(body), data=body, timeout=10)
        return resp.json().get("retCode")==0
    except Exception as e:
        log.error("set_breakeven_sl %s: %s", symbol, e)
    return False

def manage_breakeven(positions: list) -> None:
    for pos in positions:
        sym  = pos.get("symbol",""); side = pos.get("side","")
        try:
            entry = float(pos.get("avgPrice") or 0)
            sl    = float(pos.get("stopLoss") or 0)
            mark  = float(pos.get("markPrice") or 0)
        except: continue
        if entry==0 or mark==0 or sl==0: continue
        if abs(sl-entry)/entry < 0.001: log.info("%s SL at breakeven",sym); continue
        rd  = abs(entry-sl)
        tp1 = (entry+1.5*rd) if side=="Buy" else (entry-1.5*rd)
        if (mark>=tp1 if side=="Buy" else mark<=tp1):
            if set_breakeven_sl(sym, entry):
                discord_notify(f"🔒 **Breakeven** {sym} | TP1 `{mark:.4f}` | SL→`{entry:.4f}`")
        else:
            log.info("%s TP1 %.2f%% away", sym, abs(tp1-mark)/mark*100)

def get_funding_rate(symbol: str) -> Optional[float]:
    try:
        resp = requests.get(f"{BASE_URL}/v5/market/tickers",
                            params={"category":"linear","symbol":symbol}, timeout=10)
        data = resp.json()
        if data.get("retCode")==0 and data["result"]["list"]:
            return float(data["result"]["list"][0]["fundingRate"])
    except Exception as e:
        log.error("Funding %s: %s", symbol, e)
    return None

def get_spread_pct(symbol: str) -> Optional[float]:
    try:
        resp = requests.get(f"{BASE_URL}/v5/market/orderbook",
                            params={"category":"linear","symbol":symbol,"limit":1}, timeout=10)
        data = resp.json()
        if data.get("retCode")==0:
            bid = float(data["result"]["b"][0][0]); ask = float(data["result"]["a"][0][0])
            mid = (bid+ask)/2
            return (ask-bid)/mid if mid>0 else None
    except: pass
    return None

def get_klines(symbol: str, interval: str, limit: int=300) -> Optional[pd.DataFrame]:
    try:
        resp = requests.get(f"{BASE_URL}/v5/market/kline",
                            params={"category":"linear","symbol":symbol,"interval":interval,"limit":limit}, timeout=10)
        data = resp.json()
        if data.get("retCode")!=0: return None
        df = pd.DataFrame(data["result"]["list"],
                          columns=["timestamp","open","high","low","close","volume","turnover"])
        df = df.astype({"open":float,"high":float,"low":float,"close":float,"volume":float})
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms")
        df.sort_values("timestamp", inplace=True); df.reset_index(drop=True, inplace=True)
        return df
    except Exception as e:
        log.error("Klines %s %s: %s", symbol, interval, e)
    return None

def load_instrument_cache(symbols: list) -> None:
    url = f"{BASE_URL}/v5/market/instruments-info"
    for sym in symbols:
        try:
            resp = requests.get(url, params={"category":"linear","symbol":sym}, timeout=10)
            data = resp.json()
            if data.get("retCode")==0 and data["result"]["list"]:
                lot = data["result"]["list"][0]["lotSizeFilter"]
                _instrument_cache[sym] = {"min_qty":float(lot["minOrderQty"]),"qty_step":float(lot["qtyStep"])}
        except Exception as e:
            log.error("Instrument %s: %s", sym, e)

def snap_qty(symbol: str, qty: float) -> float:
    info = _instrument_cache.get(symbol)
    if not info: return round(qty, 3)
    step=info["qty_step"]; min_qty=info["min_qty"]
    snapped = math.floor(qty/step)*step
    dec = len(str(step).rstrip("0").split(".")[-1]) if "." in str(step) else 0
    snapped = round(snapped, dec)
    return snapped if snapped>=min_qty else 0.0

def set_leverage(symbol: str, leverage: int) -> bool:
    payload = {"category":"linear","symbol":symbol,"buyLeverage":str(leverage),"sellLeverage":str(leverage)}
    body = json.dumps(payload)
    try:
        resp = requests.post(f"{BASE_URL}/v5/position/set-leverage",
                             headers=_post_headers(body), data=body, timeout=10)
        return resp.json().get("retCode") in (0,110043)
    except Exception as e:
        log.error("set_leverage: %s", e)
    return False

def place_order(symbol:str, side:str, qty:float, sl_price:float, tp_price:float) -> Optional[dict]:
    payload = {"category":"linear","symbol":symbol,"side":side,"orderType":"Market","qty":str(qty),
               "stopLoss":str(round(sl_price,4)),"takeProfit":str(round(tp_price,4)),
               "timeInForce":"IOC","slTriggerBy":"LastPrice","tpTriggerBy":"LastPrice","positionIdx":0}
    body = json.dumps(payload)
    try:
        resp = requests.post(f"{BASE_URL}/v5/order/create", headers=_post_headers(body), data=body, timeout=10)
        data = resp.json()
        if data.get("retCode")==0: log.info("Order placed: %s %s qty=%s",side,symbol,qty); return data["result"]
        log.error("Order failed %s: %s", symbol, data.get("retMsg"))
    except Exception as e:
        log.error("place_order: %s", e)
    return None

def close_position(symbol:str, side:str, qty:float) -> Optional[dict]:
    cs = "Sell" if side=="Buy" else "Buy"
    payload = {"category":"linear","symbol":symbol,"side":cs,"orderType":"Market","qty":str(qty),
               "timeInForce":"IOC","reduceOnly":True,"positionIdx":0}
    body = json.dumps(payload)
    try:
        resp = requests.post(f"{BASE_URL}/v5/order/create", headers=_post_headers(body), data=body, timeout=10)
        data = resp.json()
        if data.get("retCode")==0: log.info("Closed: %s %s qty=%s",symbol,cs,qty); return data["result"]
        log.error("close_position %s: %s", symbol, data.get("retMsg"))
    except Exception as e:
        log.error("close_position: %s", e)
    return None


# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------

def _classify_ema(close: pd.Series) -> str:
    e9=ta.trend.EMAIndicator(close,window=9).ema_indicator().iloc[-1]
    e50=ta.trend.EMAIndicator(close,window=50).ema_indicator().iloc[-1]
    e200=ta.trend.EMAIndicator(close,window=200).ema_indicator().iloc[-1]
    if e9>e50>e200: return "BULL"
    if e9<e50<e200: return "BEAR"
    return "RANGE"

def _compute_regime(proxy: str) -> dict:
    df4h=get_klines(proxy,"240",limit=250); df1h=get_klines(proxy,"60",limit=250)
    fb={"regime_4h":"RANGE","regime_1h":"RANGE","allowed":"SKIP","threshold":5,"proxy":proxy}
    if df4h is None or len(df4h)<200: return fb
    r4h=_classify_ema(df4h["close"])
    r1h=_classify_ema(df1h["close"]) if df1h is not None and len(df1h)>=50 else "RANGE"
    if r4h=="BULL":
        if r1h=="BULL":    allowed,thr="Buy",4
        elif r1h=="RANGE": allowed,thr="Buy",5
        else:              allowed,thr="SKIP",5
    elif r4h=="BEAR":
        if r1h=="BEAR":    allowed,thr="Sell",4
        elif r1h=="RANGE": allowed,thr="Sell",5
        else:              allowed,thr="SKIP",5
    else: allowed,thr="SKIP",5
    log.info("REGIME [%s]: 4H=%s 1H=%s → %s thr=%d",proxy,r4h,r1h,allowed,thr)
    return {"regime_4h":r4h,"regime_1h":r1h,"allowed":allowed,"threshold":thr,"proxy":proxy}

def get_market_regime() -> dict: return _compute_regime("BTCUSDT")
def get_symbol_regime(symbol: str) -> dict:
    return _compute_regime("XAUTUSDT" if "XAU" in symbol else "BTCUSDT")


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------

def compute_indicators(df: pd.DataFrame) -> dict:
    c,h,l,v = df["close"],df["high"],df["low"],df["volume"]
    e9=ta.trend.EMAIndicator(c,window=9).ema_indicator()
    e50=ta.trend.EMAIndicator(c,window=50).ema_indicator()
    e200=ta.trend.EMAIndicator(c,window=200).ema_indicator()
    vwap=((h+l+c)/3*v).cumsum()/v.cumsum()
    rsi=ta.momentum.RSIIndicator(c,window=14).rsi()
    stoch=ta.momentum.StochRSIIndicator(c,window=14,smooth1=3,smooth2=3)
    macd_i=ta.trend.MACD(c,window_slow=26,window_fast=12,window_sign=9)
    return {"close":c.iloc[-1],"ema9":e9.iloc[-1],"ema50":e50.iloc[-1],"ema200":e200.iloc[-1],
            "vwap":vwap.iloc[-1],"rsi":rsi.iloc[-1],
            "stoch_k":stoch.stochrsi_k().iloc[-1],"stoch_d":stoch.stochrsi_d().iloc[-1],
            "macd":macd_i.macd().iloc[-1],"macd_signal":macd_i.macd_signal().iloc[-1]}


# ---------------------------------------------------------------------------
# Signal scoring — weighted by learning state, Long/Short separately
# ---------------------------------------------------------------------------

def score_signals(ind4h:dict, ind1h:dict,
                  learning_state:Optional[dict]=None) -> tuple[float,str,list]:
    wb = (learning_state or {}).get("signal_weights",{}).get("Buy",{})
    ws = (learning_state or {}).get("signal_weights",{}).get("Sell",{})
    bs=ss=0.0; details=[]

    if ind4h["close"]>ind4h["ema9"]>ind4h["ema50"]>ind4h["ema200"]:
        bs+=wb.get("EMA_bull",1.0);   details.append("EMA_bull")
    elif ind4h["close"]<ind4h["ema9"]<ind4h["ema50"]<ind4h["ema200"]:
        ss+=ws.get("EMA_bear",1.0);   details.append("EMA_bear")

    if ind1h["close"]>ind1h["vwap"]:
        bs+=wb.get("VWAP_bull",1.0);  details.append("VWAP_bull")
    elif ind1h["close"]<ind1h["vwap"]:
        ss+=ws.get("VWAP_bear",1.0);  details.append("VWAP_bear")

    if ind1h["rsi"]<40:
        bs+=wb.get("RSI_bull",1.0);   details.append("RSI_bull")
    elif ind1h["rsi"]>60:
        ss+=ws.get("RSI_bear",1.0);   details.append("RSI_bear")

    if ind1h["stoch_k"]>ind1h["stoch_d"] and ind1h["stoch_k"]<80:
        bs+=wb.get("StochRSI_bull",1.0); details.append("StochRSI_bull")
    elif ind1h["stoch_k"]<ind1h["stoch_d"] and ind1h["stoch_k"]>20:
        ss+=ws.get("StochRSI_bear",1.0); details.append("StochRSI_bear")

    if ind4h["macd"]>ind4h["macd_signal"]:
        bs+=wb.get("MACD_bull",1.0);  details.append("MACD_bull")
    elif ind4h["macd"]<ind4h["macd_signal"]:
        ss+=ws.get("MACD_bear",1.0);  details.append("MACD_bear")

    if ind1h["ema9"]>ind1h["ema50"]:
        bs+=wb.get("EMA1H_bull",1.0); details.append("EMA1H_bull")
    elif ind1h["ema9"]<ind1h["ema50"]:
        ss+=ws.get("EMA1H_bear",1.0); details.append("EMA1H_bear")

    if bs>=ss: return bs,"Buy",details
    return ss,"Sell",details


# ---------------------------------------------------------------------------
# Reversal signal checker
# ---------------------------------------------------------------------------

def check_reversal_signals(symbol:str, position_side:str,
                           df:pd.DataFrame) -> tuple[int,list[str]]:
    if len(df)<3: return 0,[]
    c,h,l,o=df["close"],df["high"],df["low"],df["open"]
    e9=ta.trend.EMAIndicator(c,window=9).ema_indicator()
    rs=ta.momentum.RSIIndicator(c,window=14).rsi()
    st=ta.momentum.StochRSIIndicator(c,window=14,smooth1=3,smooth2=3)
    sk,sd=st.stochrsi_k(),st.stochrsi_d()
    hist=ta.trend.MACD(c,window_slow=26,window_fast=12,window_sign=9).macd_diff()
    cc,pc=c.iloc[-1],c.iloc[-2]; co,po=o.iloc[-1],o.iloc[-2]
    ch,cl=h.iloc[-1],l.iloc[-1]
    ce,pe=e9.iloc[-1],e9.iloc[-2]
    cr,pr=rs.iloc[-1],rs.iloc[-2]
    ck,pk=sk.iloc[-1],sk.iloc[-2]; cd,pd2=sd.iloc[-1],sd.iloc[-2]
    ch2,ph2=hist.iloc[-1],hist.iloc[-2]
    sigs=[]
    if position_side=="Sell":
        if pr<50 and cr>=50: sigs.append("RSI>50")
        if pk<20 and pk<=pd2 and ck>cd: sigs.append("StochRSI_cross_up")
        if ph2<0 and ch2>=0: sigs.append("MACD_hist_flip_up")
        if pc<pe and cc>=ce: sigs.append("price>EMA9")
        body=abs(cc-co); lw=min(cc,co)-cl; hw=ch-max(cc,co)
        if (body>0 and lw>=2*body and hw<=body) or (pc<po and cc>co and co<=pc and cc>=po):
            sigs.append("BullishPattern")
    else:
        if pr>50 and cr<=50: sigs.append("RSI<50")
        if pk>80 and pk>=pd2 and ck<cd: sigs.append("StochRSI_cross_down")
        if ph2>0 and ch2<=0: sigs.append("MACD_hist_flip_down")
        if pc>pe and cc<=ce: sigs.append("price<EMA9")
        body=abs(cc-co); lw=min(cc,co)-cl; hw=ch-max(cc,co)
        if (body>0 and hw>=2*body and lw<=body) or (pc>po and cc<co and co>=pc and cc<=po):
            sigs.append("BearishPattern")
    log.info("%s reversal (%s): %d/5 %s",symbol,position_side,len(sigs),sigs)
    return len(sigs),sigs


# ---------------------------------------------------------------------------
# 15M momentum — 3-candle confirmation
# ---------------------------------------------------------------------------

def check_15m_momentum(symbol:str, position_side:str) -> bool:
    df=get_klines(symbol,"15",limit=60)
    if df is None or len(df)<20: return False
    c=df["close"]
    e9=ta.trend.EMAIndicator(c,window=9).ema_indicator()
    rsi=ta.momentum.RSIIndicator(c,window=14).rsi()
    if position_side=="Buy":
        c1=c.iloc[-1]<e9.iloc[-1] and rsi.iloc[-1]<50
        c2=c.iloc[-2]<e9.iloc[-2] and rsi.iloc[-2]<50
        c3=c.iloc[-3]<e9.iloc[-3] and rsi.iloc[-3]<50
    else:
        c1=c.iloc[-1]>e9.iloc[-1] and rsi.iloc[-1]>50
        c2=c.iloc[-2]>e9.iloc[-2] and rsi.iloc[-2]>50
        c3=c.iloc[-3]>e9.iloc[-3] and rsi.iloc[-3]>50
    if c1 and c2 and c3:
        log.info("%s 15M momentum confirmed 3/3 candles",symbol); return True
    return False


# ---------------------------------------------------------------------------
# Unified exit manager — records outcome
# ---------------------------------------------------------------------------

def manage_exits(position_details:list, regime:Optional[dict]=None) -> dict[str,bool]:
    EXIT_THRESHOLD=3; closed={}; now_ms=int(time.time()*1000)
    for pos in position_details:
        sym=pos["symbol"]; side=pos["side"]
        qty=float(pos.get("size",0)); entry=float(pos.get("avgPrice",0) or 0)
        sl=float(pos.get("stopLoss",0) or 0); upnl=float(pos.get("unrealisedPnl",0) or 0)
        created_ms=int(pos.get("createdTime",now_ms) or now_ms)
        if entry==0 or qty==0: continue
        rd=abs(entry-sl) if sl else 0.0; r_usdt=rd*qty
        hrs=(now_ms-created_ms)/3_600_000
        pnl_str=f"+${upnl:.2f}" if upnl>=0 else f"-${abs(upnl):.2f}"
        dlabel="LONG" if side=="Buy" else "SHORT"
        reason=None; icon="⚡"; label="QUICK EXIT"

        if r_usdt>0 and upnl>=QUICK_PROFIT_R*r_usdt:
            reason=f"0.8R profit (`{upnl:.2f}` ≥ `{QUICK_PROFIT_R*r_usdt:.2f}` USDT)"

        if reason is None:
            df1h=get_klines(sym,"60",limit=100)
            if df1h is not None and len(df1h)>=3:
                rc,rf=check_reversal_signals(sym,side,df1h)
                rb=0
                if regime:
                    al=regime.get("allowed","SKIP"); pd3="Buy" if side=="Buy" else "Sell"
                    if al=="SKIP" or al!=pd3: rb=1
                if rc+rb>=EXIT_THRESHOLD:
                    sl2=rf+(["RegimeConflict"] if rb else [])
                    reason=f"reversal `{rc+rb}/5` [{', '.join(sl2)}]"

        if reason is None and check_15m_momentum(sym,side):
            reason="15M momentum (3-candle)"

        if reason is None and upnl>0:
            fr=get_funding_rate(sym)
            if fr is not None and abs(fr)>=FUNDING_SPIKE_PCT:
                reason=f"funding spike `{fr*100:.4f}%`"

        if reason is None:
            if hrs>TIME_EXIT_HOURS and r_usdt>0 and upnl<TIME_EXIT_MIN_R*r_usdt:
                icon="⏰"; label="TIME EXIT"; reason=f"held `{hrs:.1f}h` profit<0.3R"

        if reason is None: continue
        log.warning("%s — %s [%s] | %s | uPnL=%s | held=%.1fh",sym,label,dlabel,reason,pnl_str,hrs)
        result=close_position(sym,side,qty)
        if result:
            update_trade_outcome(sym,side,upnl)   # ← record outcome
            discord_notify(f"{icon} **{label} {dlabel} {sym}** | {reason} | PnL:`{pnl_str}` | Held:`{hrs:.1f}h`")
            closed[sym]=True
        else:
            discord_notify(f"⚠️ **{sym}** — exit FAILED | {reason}")
    return closed


# ---------------------------------------------------------------------------
# Position sizing — score + time-of-day from coin profile
# ---------------------------------------------------------------------------

def calculate_position(balance:float, entry_price:float, sl_price:float,
                       leverage:int, score:float=4.0,
                       symbol:str="", learning_state:Optional[dict]=None) -> float:
    risk_pct=SCORE_SIZE_MAP.get(min(int(score),6), RISK_PER_TRADE)
    if learning_state and symbol:
        profile=learning_state.get("coin_profiles",{}).get(symbol,{})
        best_hrs=profile.get("best_hours_utc",[])
        now_h=datetime.datetime.utcnow().hour
        if best_hrs:
            if now_h in best_hrs:
                risk_pct*=1.10
            else:
                completed=[t for t in load_journal()
                           if t.get("symbol")==symbol and t.get("outcome") in ("win","loss")]
                bad_hrs=[t.get("hour_utc") for t in completed
                         if t.get("outcome")=="loss" and t.get("hour_utc") is not None]
                worst=[h for h,_ in Counter(bad_hrs).most_common(3)]
                if now_h in worst: risk_pct*=0.80
    sl_dist=abs(entry_price-sl_price)
    if sl_dist==0: return 0.0
    qty_usdt=min((balance*risk_pct/sl_dist)*entry_price, balance*leverage)
    return round(qty_usdt/entry_price, 3)


# ---------------------------------------------------------------------------
# Loss guards
# ---------------------------------------------------------------------------

def daily_loss_exceeded(s,c): return s>0 and (s-c)/s>=DAILY_LOSS_LIMIT
def weekly_loss_exceeded(s,c): return s>0 and (s-c)/s>=WEEKLY_LOSS_LIMIT


# ---------------------------------------------------------------------------
# Daily summary
# ---------------------------------------------------------------------------

def send_daily_summary(balance:float) -> None:
    journal=load_journal(); today=datetime.date.today().isoformat()
    today_t=[t for t in journal if str(t.get("timestamp","")).startswith(today)]
    comp=[t for t in today_t if t.get("outcome") in ("win","loss")]
    wins=sum(1 for t in comp if t["outcome"]=="win"); total=len(comp)
    wr=f"{wins/total*100:.0f}%" if total>0 else "n/a"
    pnl=sum(t.get("pnl",0) for t in comp)
    op=get_open_positions()
    discord_notify(
        f"📈 **DAILY SUMMARY** | {today}\n"
        f"• Trades: `{total}` | Win rate: `{wr}` ({wins}/{total})\n"
        f"• Realized PnL: `{'%+.2f'%pnl}` USDT\n"
        f"• Open: {', '.join(f'`{s}`' for s in op) if op else 'None'}\n"
        f"• Balance: `{balance:.2f}` USDT"
    )


# ---------------------------------------------------------------------------
# Main scan
# ---------------------------------------------------------------------------

def scan_symbol(symbol:str, balance:float, regime:Optional[dict]=None,
                learning_state:Optional[dict]=None) -> Optional[dict]:
    log.info("Scanning %s …",symbol)
    sp=get_spread_pct(symbol)
    if sp is not None and sp>MAX_SPREAD_PCT: log.info("%s SKIP spread",symbol); return None
    df4h=get_klines(symbol,"240"); df1h=get_klines(symbol,"60")
    if df4h is None or df1h is None or len(df4h)<200 or len(df1h)<50: return None
    avg_v=df4h["volume"].iloc[-21:-1].mean()
    if avg_v>0 and df4h["volume"].iloc[-1]/avg_v<1.2: log.info("%s SKIP low vol",symbol); return None
    ind4h=compute_indicators(df4h); ind1h=compute_indicators(df1h)
    score,direction,details=score_signals(ind4h,ind1h,learning_state)
    log.info("%s | score=%.2f dir=%s %s",symbol,score,direction,details)

    if direction=="Buy" and not LONG_ENABLED: return None

    sr=get_symbol_regime(symbol)
    allowed=sr.get("allowed","SKIP"); thr=float(sr.get("threshold",CONFLUENCE_THRESHOLD))
    if learning_state:
        prof=learning_state.get("coin_profiles",{}).get(symbol,{})
        sb=prof.get("score_bonus",0); thr+=sb
        if sb!=0: log.info("%s coin bonus %+d → thr=%.1f",symbol,sb,thr)

    if allowed=="SKIP": return None
    if allowed!=direction: return None
    if score<thr: log.info("%s SKIP score %.2f<%.1f",symbol,score,thr); return None

    if symbol in BTC_CORRELATED:
        op=get_open_positions()
        if sum(1 for s in op if s in BTC_CORRELATED)>=MAX_CORRELATION_POSITIONS: return None

    entry=ind1h["close"]; atr=(df1h["high"].iloc[-14:]-df1h["low"].iloc[-14:]).mean()
    r4h=df4h.iloc[-20:]; ph=r4h["high"].max(); pl=r4h["low"].min(); pm=(ph+pl)/2
    if direction=="Buy":
        sl=entry-2*atr; raw_tp=entry+4*atr
        cands=[p for p in [pm,ph] if entry<p<=raw_tp*1.02]
        tp=min(cands) if cands else raw_tp
    else:
        sl=entry+2*atr; raw_tp=entry-4*atr
        cands=[p for p in [pm,pl] if entry>p>=raw_tp*0.98]
        tp=max(cands) if cands else raw_tp

    leverage=min(MAX_LEVERAGE,10)
    qty=calculate_position(balance,entry,sl,leverage,score,symbol,learning_state)
    qty=snap_qty(symbol,qty)
    if qty<=0: return None

    set_leverage(symbol,leverage)
    result=place_order(symbol,direction,qty,sl,tp)
    trade={
        "timestamp":   datetime.datetime.utcnow().isoformat(),
        "symbol":      symbol, "direction": direction,
        "score":       round(score,3), "signals": details,
        "entry":       entry, "sl": round(sl,4), "tp": round(tp,4),
        "qty":         qty, "leverage": leverage,
        "hour_utc":    datetime.datetime.utcnow().hour,
        "outcome":     None, "pnl": None, "closed_at": None,
        "order_result":result,
    }
    log_trade(trade)
    if result:
        emoji="🟢" if direction=="Buy" else "🔴"
        discord_notify(f"{emoji} **{direction} {symbol}** | Score `{score:.1f}/6` | "
                       f"Entry `{entry:.4f}` | SL `{round(sl,4)}` | TP `{round(tp,4)}` | "
                       f"Qty `{qty}` × {leverage}x")
    return trade


# ---------------------------------------------------------------------------
# HTTP keep-alive
# ---------------------------------------------------------------------------

HTTP_PORT = int(os.environ.get("PORT", 5000))

class _PingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        body=b"APEX Bot Running v3"
        self.send_response(200); self.send_header("Content-Type","text/plain")
        self.send_header("Content-Length",str(len(body))); self.end_headers()
        self.wfile.write(body)
    def log_message(self,*a): pass

def _start_http_server():
    import socket
    for i in range(5):
        port=HTTP_PORT+i
        try:
            srv=HTTPServer(("0.0.0.0",port),_PingHandler)
            srv.socket.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            log.info("HTTP keep-alive on port %d",port); srv.serve_forever(); return
        except OSError as e:
            log.warning("Port %d in use (%s)",port,e.strerror)
    log.error("HTTP keep-alive disabled")


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    log.info("="*60)
    log.info("APEX Bot v3 — Self-Learning System ACTIVE")
    log.info("Long enabled: %s", LONG_ENABLED)
    log.info("="*60)
    discord_notify(f"🚀 **APEX Bot v3 started** | Self-Learning: ✅ | Long: {'✅' if LONG_ENABLED else '❌ disabled'}")
    threading.Thread(target=_start_http_server, daemon=True).start()
    if not API_KEY or not API_SECRET: log.error("API keys not set"); return
    starting_balance=get_account_balance()
    if starting_balance is None: log.error("Balance fetch failed"); return
    log.info("Starting balance: %.2f USDT", starting_balance)

    day_start=datetime.date.today()
    week_start=day_start-datetime.timedelta(days=day_start.weekday())
    week_bal=starting_balance; weekly_pause=None
    last_hc=datetime.datetime.utcnow(); daily_done=False; weekly_done=False

    while True:
        now=datetime.datetime.utcnow(); today=datetime.date.today()
        ls=load_learning_state()

        if weekly_pause and now<weekly_pause:
            log.warning("Weekly pause — %.1fh left",( weekly_pause-now).total_seconds()/3600)
            time.sleep(SCAN_INTERVAL_SECONDS); continue

        if today!=day_start:
            day_start=today; starting_balance=get_account_balance() or starting_balance
            daily_done=False; log.info("New day — %.2f USDT",starting_balance)

        cw=today-datetime.timedelta(days=today.weekday())
        if cw!=week_start:
            week_start=cw; week_bal=get_account_balance() or starting_balance; weekly_done=False
            log.info("New week — %.2f USDT",week_bal)

        if now.weekday()==6 and now.hour==0 and not weekly_done:
            run_weekly_learning(); weekly_done=True

        if now.hour==0 and now.minute<10 and not daily_done:
            b=get_account_balance()
            if b: send_daily_summary(b)
            daily_done=True

        if (now-last_hc).total_seconds()>=21_600:
            b=get_account_balance() or 0; op=get_open_positions()
            ns=now+datetime.timedelta(seconds=SCAN_INTERVAL_SECONDS)
            t_count=len([t for t in load_journal() if t.get("outcome") in ("win","loss")])
            discord_notify(f"💚 **APEX Bot alive** | Positions:`{len(op)}/{MAX_OPEN_POSITIONS}` | "
                           f"Balance:`{b:.2f}` USDT | Learned:`{t_count}` trades | Next:`{ns.strftime('%H:%M')} UTC`")
            last_hc=now

        bal=get_account_balance()
        if bal is None: time.sleep(SCAN_INTERVAL_SECONDS); continue

        if daily_loss_exceeded(starting_balance,bal):
            lp=(starting_balance-bal)/starting_balance*100
            discord_notify(f"🛑 **Daily loss limit** | `{starting_balance:.2f}`→`{bal:.2f}` (`-{lp:.1f}%`) | Pausing")
            time.sleep(SCAN_INTERVAL_SECONDS); continue

        if weekly_loss_exceeded(week_bal,bal):
            weekly_pause=now+datetime.timedelta(hours=48)
            lp=(week_bal-bal)/week_bal*100
            discord_notify(f"🛑 **Weekly loss limit** | `{week_bal:.2f}`→`{bal:.2f}` (`-{lp:.1f}%`) | Pausing 48h")
            time.sleep(SCAN_INTERVAL_SECONDS); continue

        regime=get_market_regime()
        op=get_open_positions(); pc=len(op)
        log.info("Positions:%d/%d %s | Balance:%.2f",pc,MAX_OPEN_POSITIONS,list(op.keys()),bal)

        if pc>0:
            pd4=get_position_details()
            if pd4:
                manage_breakeven(pd4)
                closed=manage_exits(pd4,regime)
                if closed: op=get_open_positions(); pc=len(op)

        if pc>=MAX_OPEN_POSITIONS:
            ns=now+datetime.timedelta(seconds=SCAN_INTERVAL_SECONDS)
            log.info("Max positions. Next %s UTC",ns.strftime("%H:%M:%S"))
            time.sleep(SCAN_INTERVAL_SECONDS); continue

        wl=get_dynamic_watchlist()
        if not wl: time.sleep(SCAN_INTERVAL_SECONDS); continue
        load_instrument_cache(wl)

        for sym in wl:
            if len(op)>=MAX_OPEN_POSITIONS: break
            if sym in op: continue
            fr=get_funding_rate(sym)
            if fr is None: continue
            if not (FUNDING_RATE_MIN<=fr<=FUNDING_RATE_MAX): continue
            try:
                trade=scan_symbol(sym,bal,regime,ls)
                if trade and trade.get("order_result"): op=get_open_positions()
            except Exception as e:
                log.error("Error scanning %s: %s",sym,e)
            time.sleep(2)

        ns=now+datetime.timedelta(seconds=SCAN_INTERVAL_SECONDS)
        rl=f"{regime.get('regime_4h','?')}/{regime.get('regime_1h','?')}→{regime.get('allowed','?')}"
        tc=len([t for t in load_journal() if t.get("outcome") in ("win","loss")])
        log.info("Scan complete. Next %s UTC",ns.strftime("%H:%M:%S"))
        discord_notify(f"📊 **Scan complete** | Regime:`{rl}` | Positions:`{len(op)}/{MAX_OPEN_POSITIONS}` | "
                       f"Balance:`{bal:.2f}` USDT | Learned:`{tc}` | Next:`{ns.strftime('%H:%M')} UTC`")
        time.sleep(SCAN_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()
