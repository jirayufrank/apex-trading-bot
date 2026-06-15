#!/usr/bin/env python3
"""
APEX Bybit Futures Trading Bot — v4.1 "Fast & Precise"
================================================================
New in v4.1:
  [S1] Split-speed loop — exits/breakeven/TP1 checked every 5 MINUTES,
       entry scans every 15 minutes aligned to candle close (xx:00/15/30/45).
       No more waiting up to an hour to react.
  [S2] Real ATR — true range (includes gaps vs previous close) via
       ta.volatility.AverageTrueRange. SLs are now correctly sized in
       volatile markets instead of being too tight.
  [S3] Maker entries — PostOnly limit at best bid/ask (maker fee 0.02%
       instead of taker 0.055% + slippage). Unfilled after retries → skip,
       never chase.
  [S4] Partial TP — close 50% at TP1 (1.5R), move SL to breakeven, let
       the runner ride to TP2. TP1 profit is tracked and added to the
       trade's final PnL.

Fixes from v3 (all 9 weaknesses):
  [C1] Closed-PnL reconciliation — trades closed by exchange SL/TP are now
       recorded with REAL realized PnL (incl. fees) via /v5/position/closed-pnl.
       Learning data is no longer survivorship-biased.
  [C2] Persistent storage — journal/learning files live in DATA_DIR
       (set env DATA_DIR=/data + attach a Railway Volume). Atomic writes
       prevent file corruption on crash/restart.
  [C3] Closed-candle indicators — all entry signals, volume filter, and
       regime detection use the LAST CLOSED candle (no repaint).
       Entry price still uses live last price.
  [C4] Signal attribution fix — journal stores only the signals of the
       chosen direction, so learning never mixes bull/bear signals.
  [M5] Sizing fix — risk tiers by score threshold (>=6: 2.5%, >=5: 2.0%,
       else 1.5%). A weak score can never get a bigger size than a strong one.
  [M6] Scheduling fix — daily summary fires on the FIRST scan of each new
       day (summarizing yesterday); weekly learning fires on the first scan
       of every Sunday. No more 10-minute windows that get skipped.
  [M7] Daily-anchored VWAP — resets every UTC day (real session VWAP),
       no longer depends on arbitrary lookback length.
  [M8] Initial-risk exits — R is computed from the ORIGINAL SL stored in
       the journal, so quick-profit / time exits keep working after
       breakeven moves the live SL to entry.
  [M9] Learning robustness — coin profiles need >=10 trades; time-of-day
       sizing needs >=15 trades and uses precomputed best/worst hours;
       signal weights mean-revert 10% toward 1.0 each week (anti-drift).
Extra stability:
  - HTTP retry session (3 retries, backoff) on all Bybit/Discord calls
  - Signed GET now sends the exact sorted query string it signed
    (fixes latent signature-mismatch bug)
  - Main loop wrapped in catch-all: one bad iteration can no longer
    crash the whole bot
  - Trades are journaled ONLY when the order actually fills
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
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
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

# [C2] Persistent data dir — on Railway: add a Volume mounted at /data
#      and set env var DATA_DIR=/data
DATA_DIR = os.environ.get("DATA_DIR", ".")
os.makedirs(DATA_DIR, exist_ok=True)

DYNAMIC_SCAN_TOP_N     = 50
DYNAMIC_VOLUME_MIN_USD = 50_000_000
STABLECOIN_BASES       = {"USDC","BUSD","TUSD","USDP","DAI","FDUSD","PYUSD"}

RISK_PER_TRADE        = 0.015          # fallback / lowest tier
MAX_LEVERAGE          = 10
MAX_OPEN_POSITIONS    = 3
DAILY_LOSS_LIMIT      = 0.05
WEEKLY_LOSS_LIMIT     = 0.10
CONFLUENCE_THRESHOLD  = 4

# [S1] Split-speed loop
FAST_INTERVAL_SECONDS = 300            # exits / TP1 / breakeven every 5 min
ENTRY_SCAN_MINUTES    = (0, 15, 30, 45)  # entry scans at candle closes
WATCHLIST_TTL_SECONDS = 3600           # refresh watchlist hourly

# [S3] Maker entry
ENTRY_FILL_TIMEOUT_S  = 90             # wait per attempt for PostOnly fill
ENTRY_MAX_ATTEMPTS    = 2              # re-quote once, then skip

# [S4] Partial take-profit
TP1_R                 = 1.5
TP1_CLOSE_FRACTION    = 0.5

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

QUICK_PROFIT_R    = 0.8
TIME_EXIT_HOURS   = 6
TIME_EXIT_MIN_R   = 0.3
FUNDING_SPIKE_PCT = 0.0005
FUNDING_RATE_MAX  = 0.001
FUNDING_RATE_MIN  = -0.001

TRADE_JOURNAL_FILE  = os.path.join(DATA_DIR, "trade_journal.json")
LEARNING_STATE_FILE = os.path.join(DATA_DIR, "learning_state.json")

# Reconciliation: how far back to look for closed PnL, and when to give up
RECONCILE_LOOKBACK_DAYS = 7
STALE_TRADE_DAYS        = 3   # open journal entry w/ no position & no match → unknown

_instrument_cache: dict = {}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(),
              logging.FileHandler(os.path.join(DATA_DIR, "bot.log"))],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HTTP session with retries  (stability)
# ---------------------------------------------------------------------------

SESSION = requests.Session()
_retry = Retry(total=3, backoff_factor=1.0,
               status_forcelist=[429, 500, 502, 503, 504],
               allowed_methods=frozenset(["GET", "POST"]))
SESSION.mount("https://", HTTPAdapter(max_retries=_retry))
SESSION.mount("http://", HTTPAdapter(max_retries=_retry))

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

MIN_TRADES_TO_LEARN     = 20
MIN_TRADES_PER_SIGNAL   = 8     # was 5 — fewer false adjustments
MIN_TRADES_PER_COIN     = 10    # [M9] was 3
MIN_TRADES_FOR_HOURS    = 15    # [M9] time-of-day sizing needs real sample
MAX_WEIGHT              = 1.5
MIN_WEIGHT              = 0.5
WEIGHT_MEAN_REVERSION   = 0.9   # [M9] pull 10% toward 1.0 weekly (anti-drift)


def _atomic_write_json(path: str, obj) -> None:
    """[C2] Write to temp file then atomically replace — no corrupt files."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    os.replace(tmp, path)


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
    _atomic_write_json(LEARNING_STATE_FILE, state)


# ---------------------------------------------------------------------------
# Trade Journal
# ---------------------------------------------------------------------------

def load_journal() -> list:
    if os.path.exists(TRADE_JOURNAL_FILE):
        try:
            with open(TRADE_JOURNAL_FILE) as f:
                return json.load(f)
        except Exception as e:
            log.warning("Journal load failed: %s", e)
    return []


def save_journal(journal: list) -> None:
    _atomic_write_json(TRADE_JOURNAL_FILE, journal)


def log_trade(entry: dict) -> None:
    j = load_journal()
    j.append(entry)
    save_journal(j)


def _ts_to_ms(iso_str: str) -> int:
    try:
        dt = datetime.datetime.fromisoformat(str(iso_str))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=datetime.timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        return 0


def _find_open_trade(symbol: str, direction: str) -> tuple:
    """Return (journal, index) of most recent open trade, index=-1 if none."""
    journal = load_journal()
    for i in range(len(journal)-1, -1, -1):
        t = journal[i]
        if (t.get("symbol")==symbol and t.get("direction")==direction
                and t.get("outcome") is None):
            return journal, i
    return journal, -1


def update_trade_outcome(symbol: str, direction: str, pnl: float,
                         exit_reason: str = "bot_exit",
                         exit_price: Optional[float] = None,
                         closed_ms: Optional[int] = None,
                         add_tp1: bool = True) -> None:
    """Find most recent open trade for symbol+direction and record outcome.
    add_tp1=True → adds stored TP1 partial profit to pnl (bot-side exits).
    add_tp1=False → pnl is already the full total (reconciliation path)."""
    journal, i = _find_open_trade(symbol, direction)
    if i < 0:
        log.warning("update_trade_outcome: no open journal entry for %s %s", symbol, direction)
        return
    trade = journal[i]
    total = pnl + (float(trade.get("tp1_realized") or 0) if add_tp1 else 0.0)
    closed_dt = (datetime.datetime.utcfromtimestamp(closed_ms/1000)
                 if closed_ms else datetime.datetime.utcnow())
    trade["outcome"]    = "win" if total > 0 else "loss"
    trade["pnl"]        = round(total, 4)
    trade["closed_at"]  = closed_dt.isoformat()
    trade["exit_reason"]= exit_reason
    if exit_price is not None:
        trade["exit_price"] = exit_price
    # hour_utc stays = ENTRY hour (set at entry) for time-of-day learning
    save_journal(journal)
    log.info("Outcome: %s %s pnl=%.4f → %s (%s)",
             symbol, direction, total, trade["outcome"], exit_reason)


# ---------------------------------------------------------------------------
# [C1] Closed-PnL reconciliation — the heart of accurate learning
# ---------------------------------------------------------------------------

def get_closed_pnl_records(start_ms: int) -> list:
    """Fetch closed PnL records from Bybit (realized PnL incl. fees)."""
    records, cursor = [], ""
    for _ in range(5):  # max 5 pages = 500 records
        params = {"category": "linear", "limit": "100", "startTime": str(start_ms)}
        if cursor:
            params["cursor"] = cursor
        data = _signed_get("/v5/position/closed-pnl", params)
        if not data or data.get("retCode") != 0:
            break
        result = data.get("result", {})
        records.extend(result.get("list", []))
        cursor = result.get("nextPageCursor", "")
        if not cursor:
            break
    return records


def reconcile_closed_trades() -> int:
    """
    Match journal entries with outcome=None against Bybit closed-PnL records.
    Catches trades closed by exchange-side SL/TP (or manual close) that the
    bot never saw. Returns number of trades reconciled.
    """
    journal = load_journal()
    pending = [t for t in journal if t.get("outcome") is None and t.get("order_result")]
    if not pending:
        return 0

    open_now = get_open_positions()
    now_ms = int(time.time() * 1000)
    earliest = min((_ts_to_ms(t.get("timestamp","")) for t in pending), default=now_ms)
    lookback_floor = now_ms - RECONCILE_LOOKBACK_DAYS * 86_400_000
    start_ms = max(min(earliest - 60_000, now_ms), lookback_floor)

    records = get_closed_pnl_records(start_ms)
    if not records:
        records = []
    # In closed-pnl, "side" is the CLOSING order side → position direction is the opposite
    def _rec_dir(r): return "Buy" if r.get("side") == "Sell" else "Sell"
    used_ids: set = set()
    fixed = 0

    for trade in pending:
        sym, d = trade.get("symbol",""), trade.get("direction","")
        # Still open on exchange? leave it
        if open_now.get(sym) == d:
            continue
        t_ms = _ts_to_ms(trade.get("timestamp",""))
        matches = sorted(
            [r for r in records
             if r.get("symbol")==sym and _rec_dir(r)==d
             and r.get("orderId") not in used_ids
             and int(r.get("updatedTime", 0)) >= t_ms - 60_000],
            key=lambda r: int(r.get("updatedTime", 0)),
        )
        if matches:
            # [S4] a trade can close in pieces (TP1 partial + runner) —
            # sum ALL matching close records for the true total PnL
            for r in matches:
                used_ids.add(r.get("orderId"))
            try:
                pnl = sum(float(r.get("closedPnl", 0)) for r in matches)
                exit_px = float(matches[-1].get("avgExitPrice", 0)) or None
            except Exception:
                pnl, exit_px = 0.0, None
            update_trade_outcome(
                sym, d, pnl,
                exit_reason="exchange_close",   # SL/TP hit or manual close
                exit_price=exit_px,
                closed_ms=int(matches[-1].get("updatedTime", now_ms)),
                add_tp1=False,                  # records already include TP1 piece
            )
            fixed += 1
            emoji = "✅" if pnl > 0 else "🛑"
            discord_notify(
                f"{emoji} **Reconciled {d} {sym}** | exchange close | "
                f"PnL `{pnl:+.2f}` USDT (incl. fees)"
            )
        else:
            # No position, no record, and it's old → mark unknown (excluded from learning)
            if t_ms and now_ms - t_ms > STALE_TRADE_DAYS * 86_400_000:
                j2 = load_journal()
                for t2 in reversed(j2):
                    if (t2.get("symbol")==sym and t2.get("direction")==d
                            and t2.get("outcome") is None
                            and t2.get("timestamp")==trade.get("timestamp")):
                        t2["outcome"] = "unknown"
                        t2["exit_reason"] = "stale_unmatched"
                        save_journal(j2)
                        log.warning("Marked stale trade unknown: %s %s", sym, d)
                        break
    if fixed:
        log.info("Reconciled %d exchange-closed trades", fixed)
    return fixed


# ---------------------------------------------------------------------------
# Discord
# ---------------------------------------------------------------------------

def discord_notify(message: str) -> None:
    if not DISCORD_WEBHOOK_URL:
        return
    try:
        SESSION.post(DISCORD_WEBHOOK_URL, json={"content": message}, timeout=10)
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

    # [M9] Mean-revert all weights 10% toward 1.0 before updating (anti-drift)
    for side in ("Buy","Sell"):
        for sig, w in state["signal_weights"][side].items():
            state["signal_weights"][side][sig] = round(1.0 + (w - 1.0) * WEIGHT_MEAN_REVERSION, 3)

    # 1. Signal win rates — Buy/Sell separately
    #    [C4] journal now stores only same-direction signals, so this is clean
    sig_stats: dict = {"Buy":{},"Sell":{}}
    for trade in completed:
        d = trade.get("direction","")
        if d not in ("Buy","Sell"): continue
        for sig in trade.get("signals",[]):
            if sig not in state["signal_weights"][d]:
                continue  # ignore foreign/legacy signal names
            b = sig_stats[d].setdefault(sig, {"wins":0,"total":0})
            b["total"] += 1
            if trade.get("outcome") == "win": b["wins"] += 1

    for direction in ("Buy","Sell"):
        for sig, stats in sig_stats[direction].items():
            if stats["total"] < MIN_TRADES_PER_SIGNAL: continue
            wr = stats["wins"] / stats["total"]
            cur = state["signal_weights"][direction].get(sig, 1.0)
            if   wr >= 0.70: new_w = min(cur * 1.15, MAX_WEIGHT)
            elif wr >= 0.55: new_w = cur
            else:            new_w = max(cur * 0.85, MIN_WEIGHT)
            state["signal_weights"][direction][sig] = round(new_w, 3)

    # 2. Coin profiling — [M9] min 10 trades; precompute best & worst hours
    coin_data: dict = {}
    for trade in completed:
        sym = trade.get("symbol","")
        if not sym: continue
        p = coin_data.setdefault(sym, {"wl":0,"tl":0,"ws":0,"ts":0,
                                       "win_hrs":[], "loss_hrs":[], "pnls":[]})
        d, o, h = trade.get("direction",""), trade.get("outcome"), trade.get("hour_utc")
        if d == "Buy":
            p["tl"] += 1
            if o == "win": p["wl"] += 1
        elif d == "Sell":
            p["ts"] += 1
            if o == "win": p["ws"] += 1
        if h is not None:
            (p["win_hrs"] if o=="win" else p["loss_hrs"]).append(h)
        p["pnls"].append(trade.get("pnl",0) or 0)

    profiles = state.setdefault("coin_profiles",{})
    for sym, data in coin_data.items():
        total = data["tl"] + data["ts"]
        if total < MIN_TRADES_PER_COIN: continue
        wr_l = data["wl"]/data["tl"] if data["tl"]>0 else None
        wr_s = data["ws"]/data["ts"] if data["ts"]>0 else None
        wr_a = (data["wl"]+data["ws"])/total
        score_bonus = -1 if wr_a>=0.70 else (0 if wr_a>=0.50 else (1 if wr_a>=0.35 else 2))
        best_hrs  = [h for h,_ in Counter(data["win_hrs"]).most_common(3)]  if data["win_hrs"]  else []
        worst_hrs = [h for h,_ in Counter(data["loss_hrs"]).most_common(3)] if data["loss_hrs"] else []
        profiles[sym] = {
            "win_rate_long":  round(wr_l,3) if wr_l is not None else None,
            "win_rate_short": round(wr_s,3) if wr_s is not None else None,
            "win_rate_all":   round(wr_a,3),
            "avg_pnl":        round(sum(data["pnls"])/total, 4),
            "score_bonus":    score_bonus,
            "best_hours_utc":  best_hrs,
            "worst_hours_utc": worst_hrs,
            "trades_count":   total,
        }

    state["last_updated"]          = datetime.datetime.utcnow().isoformat()
    state["total_trades_analyzed"] = len(completed)
    save_learning_state(state)

    # Report
    def _wr(sig, d):
        s = sig_stats[d].get(sig)
        return f"{s['wins']/s['total']*100:.0f}%" if s and s["total"]>=MIN_TRADES_PER_SIGNAL else "n/a"
    def _arr(w): return "↑" if w>1.05 else ("↓" if w<0.95 else "→")

    sell_lines = "\n".join(
        f"  `{s}`: {_wr(s,'Sell')} {_arr(w)} w={w}"
        for s,w in state["signal_weights"]["Sell"].items())
    buy_lines = "\n".join(
        f"  `{s}`: {_wr(s,'Buy')} {_arr(w)} w={w}"
        for s,w in state["signal_weights"]["Buy"].items())
    top_coins = sorted(profiles.items(), key=lambda x:x[1].get("win_rate_all",0), reverse=True)[:5]
    coin_lines = "\n".join(
        f"  `{sym}`: {p['win_rate_all']*100:.0f}% ({p['trades_count']}t) "
        f"bonus={p['score_bonus']:+d} avgPnL={p.get('avg_pnl',0):+.2f}"
        for sym,p in top_coins)
    overall = sum(1 for t in completed if t["outcome"]=="win")/len(completed)*100
    total_pnl = sum((t.get("pnl") or 0) for t in completed)
    discord_notify(
        f"📚 **WEEKLY LEARNING REPORT** | {datetime.date.today()}\n"
        f"Analyzed: `{len(completed)}` trades | Win rate: `{overall:.0f}%` | "
        f"Total PnL: `{total_pnl:+.2f}` USDT\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"**SHORT signals (↑=boost ↓=reduce):**\n{sell_lines}\n"
        f"**LONG signals:**\n{buy_lines}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"**Coin profiles (top 5):**\n{coin_lines if coin_lines else '  (need ≥10 trades/coin)'}"
    )
    log.info("Weekly learning done. %d trades analyzed.", len(completed))


# ---------------------------------------------------------------------------
# Dynamic watchlist
# ---------------------------------------------------------------------------

def get_dynamic_watchlist() -> list:
    try:
        resp = SESSION.get(f"{BASE_URL}/v5/market/tickers",
                           params={"category":"linear"}, timeout=10)
        data = resp.json()
        if data.get("retCode") != 0: return []
        def _tv(t):
            try: return float(t.get("turnover24h",0))
            except Exception: return 0.0
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
    ts  = str(int(time.time() * 1000))
    rw  = "5000"
    qs  = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
    sig = hmac.new(
        API_SECRET.encode(), (ts + API_KEY + rw + qs).encode(), hashlib.sha256
    ).hexdigest()
    headers = {
        "X-BAPI-API-KEY":     API_KEY,
        "X-BAPI-TIMESTAMP":   ts,
        "X-BAPI-RECV-WINDOW": rw,
        "X-BAPI-SIGN":        sig,
    }
    try:
        # Send the EXACT query string we signed (sorted) — fixes latent
        # signature-mismatch when param insertion order != sorted order
        resp = SESSION.get(f"{BASE_URL}{path}?{qs}", headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.json()
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
        resp = SESSION.post(f"{BASE_URL}/v5/position/trading-stop",
                            headers=_post_headers(body), data=body, timeout=10)
        return resp.json().get("retCode")==0
    except Exception as e:
        log.error("set_breakeven_sl %s: %s", symbol, e)
    return False

def manage_tp1(positions: list) -> None:
    """[S4] At TP1 (1.5R from ORIGINAL entry/SL): close 50%, move SL to
    breakeven, let the runner ride to TP2. Falls back to breakeven-only
    when the position is too small to split."""
    for pos in positions:
        sym  = pos.get("symbol",""); side = pos.get("side","")
        try:
            entry = float(pos.get("avgPrice") or 0)
            mark  = float(pos.get("markPrice") or 0)
            qty   = float(pos.get("size") or 0)
        except Exception:
            continue
        if entry==0 or mark==0 or qty==0: continue

        journal, i = _find_open_trade(sym, side)
        if i < 0: continue
        t = journal[i]
        if t.get("tp1_done"): continue
        try:
            e0 = float(t.get("entry") or entry)
            s0 = float(t.get("sl") or 0)
        except Exception:
            continue
        if s0 == 0: continue
        rd  = abs(e0 - s0)
        tp1 = (e0 + TP1_R*rd) if side=="Buy" else (e0 - TP1_R*rd)
        hit = (mark >= tp1) if side=="Buy" else (mark <= tp1)
        if not hit:
            log.info("%s TP1 %.2f%% away", sym, abs(tp1-mark)/mark*100)
            continue

        half = snap_qty(sym, qty*TP1_CLOSE_FRACTION)
        if half <= 0 or half >= qty:
            # too small to split → old behaviour: just protect with breakeven
            if set_breakeven_sl(sym, e0):
                journal[i]["tp1_done"] = True; save_journal(journal)
                discord_notify(f"🔒 **Breakeven** {sym} | TP1 `{mark:.4f}` | "
                               f"SL→`{e0:.4f}` (too small to split)")
            continue

        result = close_position(sym, side, half)
        if result:
            realized = (mark-e0)*half if side=="Buy" else (e0-mark)*half
            journal, i = _find_open_trade(sym, side)
            if i >= 0:
                journal[i]["tp1_done"]     = True
                journal[i]["tp1_realized"] = round(realized, 4)
                save_journal(journal)
            set_breakeven_sl(sym, e0)
            dlabel = "LONG" if side=="Buy" else "SHORT"
            discord_notify(f"🎯 **TP1 {dlabel} {sym}** | closed `{half}` @`{mark:.4f}` "
                           f"(`{realized:+.2f}` USDT est.) | SL→BE `{e0:.4f}` | "
                           f"runner `{round(qty-half,6)}`")

def get_funding_rate(symbol: str) -> Optional[float]:
    try:
        resp = SESSION.get(f"{BASE_URL}/v5/market/tickers",
                           params={"category":"linear","symbol":symbol}, timeout=10)
        data = resp.json()
        if data.get("retCode")==0 and data["result"]["list"]:
            return float(data["result"]["list"][0]["fundingRate"])
    except Exception as e:
        log.error("Funding %s: %s", symbol, e)
    return None

def get_spread_pct(symbol: str) -> Optional[float]:
    try:
        resp = SESSION.get(f"{BASE_URL}/v5/market/orderbook",
                           params={"category":"linear","symbol":symbol,"limit":1}, timeout=10)
        data = resp.json()
        if data.get("retCode")==0:
            bid = float(data["result"]["b"][0][0]); ask = float(data["result"]["a"][0][0])
            mid = (bid+ask)/2
            return (ask-bid)/mid if mid>0 else None
    except Exception:
        pass
    return None

def get_klines(symbol: str, interval: str, limit: int=300) -> Optional[pd.DataFrame]:
    try:
        resp = SESSION.get(f"{BASE_URL}/v5/market/kline",
                           params={"category":"linear","symbol":symbol,
                                   "interval":interval,"limit":limit}, timeout=10)
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
        if sym in _instrument_cache:   # cache hit — don't refetch every loop
            continue
        try:
            resp = SESSION.get(url, params={"category":"linear","symbol":sym}, timeout=10)
            data = resp.json()
            if data.get("retCode")==0 and data["result"]["list"]:
                inst = data["result"]["list"][0]
                lot  = inst["lotSizeFilter"]
                tick = float(inst.get("priceFilter",{}).get("tickSize", 0) or 0)
                _instrument_cache[sym] = {"min_qty":float(lot["minOrderQty"]),
                                          "qty_step":float(lot["qtyStep"]),
                                          "tick_size":tick}
        except Exception as e:
            log.error("Instrument %s: %s", sym, e)

def snap_price(symbol: str, price: float) -> float:
    """[S3] Snap price to the instrument's tick size (required for limit orders)."""
    info = _instrument_cache.get(symbol)
    tick = (info or {}).get("tick_size") or 0
    if tick <= 0: return round(price, 4)
    snapped = round(round(price/tick)*tick, 10)
    dec = len(str(tick).rstrip("0").split(".")[-1]) if "." in str(tick) else 0
    return round(snapped, dec)

def snap_qty(symbol: str, qty: float) -> float:
    info = _instrument_cache.get(symbol)
    if not info: return round(qty, 3)
    step=info["qty_step"]; min_qty=info["min_qty"]
    snapped = math.floor(qty/step)*step
    dec = len(str(step).rstrip("0").split(".")[-1]) if "." in str(step) else 0
    snapped = round(snapped, dec)
    return snapped if snapped>=min_qty else 0.0

def set_leverage(symbol: str, leverage: int) -> bool:
    payload = {"category":"linear","symbol":symbol,
               "buyLeverage":str(leverage),"sellLeverage":str(leverage)}
    body = json.dumps(payload)
    try:
        resp = SESSION.post(f"{BASE_URL}/v5/position/set-leverage",
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
        resp = SESSION.post(f"{BASE_URL}/v5/order/create",
                            headers=_post_headers(body), data=body, timeout=10)
        data = resp.json()
        if data.get("retCode")==0:
            log.info("Order placed: %s %s qty=%s",side,symbol,qty); return data["result"]
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
        resp = SESSION.post(f"{BASE_URL}/v5/order/create",
                            headers=_post_headers(body), data=body, timeout=10)
        data = resp.json()
        if data.get("retCode")==0:
            log.info("Closed: %s %s qty=%s",symbol,cs,qty); return data["result"]
        log.error("close_position %s: %s", symbol, data.get("retMsg"))
    except Exception as e:
        log.error("close_position: %s", e)
    return None


# ---------------------------------------------------------------------------
# [S3] Maker entry helpers — PostOnly limit at best bid/ask
# ---------------------------------------------------------------------------

def get_best_price(symbol: str, side: str) -> Optional[float]:
    """Maker price: best bid for Buy, best ask for Sell."""
    try:
        resp = SESSION.get(f"{BASE_URL}/v5/market/orderbook",
                           params={"category":"linear","symbol":symbol,"limit":1}, timeout=10)
        data = resp.json()
        if data.get("retCode")==0:
            if side=="Buy":
                return float(data["result"]["b"][0][0])
            return float(data["result"]["a"][0][0])
    except Exception as e:
        log.error("get_best_price %s: %s", symbol, e)
    return None

def place_limit_postonly(symbol:str, side:str, qty:float, price:float,
                         sl_price:float, tp_price:float) -> Optional[dict]:
    payload = {"category":"linear","symbol":symbol,"side":side,
               "orderType":"Limit","qty":str(qty),"price":str(price),
               "stopLoss":str(sl_price),"takeProfit":str(tp_price),
               "timeInForce":"PostOnly",
               "slTriggerBy":"LastPrice","tpTriggerBy":"LastPrice","positionIdx":0}
    body = json.dumps(payload)
    try:
        resp = SESSION.post(f"{BASE_URL}/v5/order/create",
                            headers=_post_headers(body), data=body, timeout=10)
        data = resp.json()
        if data.get("retCode")==0:
            log.info("PostOnly placed: %s %s qty=%s @%s",side,symbol,qty,price)
            return data["result"]
        log.error("PostOnly failed %s: %s", symbol, data.get("retMsg"))
    except Exception as e:
        log.error("place_limit_postonly: %s", e)
    return None

def get_order_status(symbol: str, order_id: str) -> Optional[str]:
    data = _signed_get("/v5/order/realtime",
                       {"category":"linear","orderId":order_id,"symbol":symbol})
    if data and data.get("retCode")==0:
        lst = data["result"].get("list",[])
        if lst: return lst[0].get("orderStatus")
    return None

def cancel_order(symbol: str, order_id: str) -> bool:
    payload = {"category":"linear","symbol":symbol,"orderId":order_id}
    body = json.dumps(payload)
    try:
        resp = SESSION.post(f"{BASE_URL}/v5/order/cancel",
                            headers=_post_headers(body), data=body, timeout=10)
        return resp.json().get("retCode")==0
    except Exception as e:
        log.error("cancel_order %s: %s", symbol, e)
    return False

def wait_for_fill(symbol: str, order_id: str, timeout_s: int) -> str:
    """Poll order status until Filled / terminal / timeout."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        st = get_order_status(symbol, order_id)
        if st == "Filled": return "Filled"
        if st in ("Cancelled","Rejected","Deactivated"): return st
        time.sleep(8)
    return "Timeout"


# ---------------------------------------------------------------------------
# Regime detection — [C3] uses CLOSED candles only
# ---------------------------------------------------------------------------

def _closed(df: pd.DataFrame) -> pd.DataFrame:
    """Drop the last (still-forming) candle. Bybit returns it as the newest row."""
    return df.iloc[:-1] if df is not None and len(df) > 1 else df

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
    if df4h is None or len(df4h)<201: return fb
    df4h, df1h = _closed(df4h), _closed(df1h) if df1h is not None else None
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
    elif r4h=="RANGE":
        if r1h=="BULL":    allowed,thr="Buy",5
        elif r1h=="BEAR":  allowed,thr="Sell",5
        else:              allowed,thr="SKIP",5
    log.info("REGIME [%s]: 4H=%s 1H=%s → %s thr=%d",proxy,r4h,r1h,allowed,thr)
    return {"regime_4h":r4h,"regime_1h":r1h,"allowed":allowed,"threshold":thr,"proxy":proxy}

def get_market_regime() -> dict: return _compute_regime("BTCUSDT")
def get_symbol_regime(symbol: str) -> dict:
    return _compute_regime("XAUTUSDT" if "XAU" in symbol else "BTCUSDT")


# ---------------------------------------------------------------------------
# Indicators — [C3] closed candles, [M7] daily-anchored VWAP
# ---------------------------------------------------------------------------

def _daily_anchored_vwap(df: pd.DataFrame) -> pd.Series:
    """Real session VWAP — resets every UTC day."""
    tp  = (df["high"] + df["low"] + df["close"]) / 3
    pv  = tp * df["volume"]
    day = df["timestamp"].dt.floor("D")
    cum_pv = pv.groupby(day).cumsum()
    cum_v  = df["volume"].groupby(day).cumsum().replace(0, np.nan)
    return cum_pv / cum_v

def compute_indicators(df: pd.DataFrame, use_closed: bool = True) -> dict:
    if use_closed:
        df = _closed(df)
    c,h,l,v = df["close"],df["high"],df["low"],df["volume"]
    e9=ta.trend.EMAIndicator(c,window=9).ema_indicator()
    e50=ta.trend.EMAIndicator(c,window=50).ema_indicator()
    e200=ta.trend.EMAIndicator(c,window=200).ema_indicator()
    vwap=_daily_anchored_vwap(df)
    rsi=ta.momentum.RSIIndicator(c,window=14).rsi()
    stoch=ta.momentum.StochRSIIndicator(c,window=14,smooth1=3,smooth2=3)
    macd_i=ta.trend.MACD(c,window_slow=26,window_fast=12,window_sign=9)
    return {"close":c.iloc[-1],"ema9":e9.iloc[-1],"ema50":e50.iloc[-1],"ema200":e200.iloc[-1],
            "vwap":vwap.iloc[-1],"rsi":rsi.iloc[-1],
            "stoch_k":stoch.stochrsi_k().iloc[-1],"stoch_d":stoch.stochrsi_d().iloc[-1],
            "macd":macd_i.macd().iloc[-1],"macd_signal":macd_i.macd_signal().iloc[-1]}


# ---------------------------------------------------------------------------
# Signal scoring — [C4] returns ONLY the chosen direction's signals
# ---------------------------------------------------------------------------

def score_signals(ind4h:dict, ind1h:dict,
                  learning_state:Optional[dict]=None) -> tuple:
    wb = (learning_state or {}).get("signal_weights",{}).get("Buy",{})
    ws = (learning_state or {}).get("signal_weights",{}).get("Sell",{})
    bs=ss=0.0
    buy_sigs=[]; sell_sigs=[]

    if ind4h["close"]>ind4h["ema9"]>ind4h["ema50"]>ind4h["ema200"]:
        bs+=wb.get("EMA_bull",1.0);   buy_sigs.append("EMA_bull")
    elif ind4h["close"]<ind4h["ema9"]<ind4h["ema50"]<ind4h["ema200"]:
        ss+=ws.get("EMA_bear",1.0);   sell_sigs.append("EMA_bear")

    if ind1h["close"]>ind1h["vwap"]:
        bs+=wb.get("VWAP_bull",1.0);  buy_sigs.append("VWAP_bull")
    elif ind1h["close"]<ind1h["vwap"]:
        ss+=ws.get("VWAP_bear",1.0);  sell_sigs.append("VWAP_bear")

    if ind1h["rsi"]<40:
        bs+=wb.get("RSI_bull",1.0);   buy_sigs.append("RSI_bull")
    elif ind1h["rsi"]>60:
        ss+=ws.get("RSI_bear",1.0);   sell_sigs.append("RSI_bear")

    if ind1h["stoch_k"]>ind1h["stoch_d"] and ind1h["stoch_k"]<80:
        bs+=wb.get("StochRSI_bull",1.0); buy_sigs.append("StochRSI_bull")
    elif ind1h["stoch_k"]<ind1h["stoch_d"] and ind1h["stoch_k"]>20:
        ss+=ws.get("StochRSI_bear",1.0); sell_sigs.append("StochRSI_bear")

    if ind4h["macd"]>ind4h["macd_signal"]:
        bs+=wb.get("MACD_bull",1.0);  buy_sigs.append("MACD_bull")
    elif ind4h["macd"]<ind4h["macd_signal"]:
        ss+=ws.get("MACD_bear",1.0);  sell_sigs.append("MACD_bear")

    if ind1h["ema9"]>ind1h["ema50"]:
        bs+=wb.get("EMA1H_bull",1.0); buy_sigs.append("EMA1H_bull")
    elif ind1h["ema9"]<ind1h["ema50"]:
        ss+=ws.get("EMA1H_bear",1.0); sell_sigs.append("EMA1H_bear")

    # [C4] return only the winning side's signals → clean attribution
    if bs>=ss: return bs,"Buy",buy_sigs
    return ss,"Sell",sell_sigs


# ---------------------------------------------------------------------------
# Reversal signal checker — [C3] closed candles
# ---------------------------------------------------------------------------

def check_reversal_signals(symbol:str, position_side:str,
                           df:pd.DataFrame) -> tuple:
    df = _closed(df)
    if df is None or len(df)<3: return 0,[]
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
# 15M momentum — 3-candle confirmation, [C3] closed candles
# ---------------------------------------------------------------------------

def check_15m_momentum(symbol:str, position_side:str) -> bool:
    df=get_klines(symbol,"15",limit=60)
    if df is None or len(df)<21: return False
    df=_closed(df)
    c=df["close"]
    e9=ta.trend.EMAIndicator(c,window=9).ema_indicator()
    rsi=ta.momentum.RSIIndicator(c,window=14).rsi()
    if position_side=="Buy":
        ok=all(c.iloc[-i]<e9.iloc[-i] and rsi.iloc[-i]<50 for i in (1,2,3))
    else:
        ok=all(c.iloc[-i]>e9.iloc[-i] and rsi.iloc[-i]>50 for i in (1,2,3))
    if ok: log.info("%s 15M momentum confirmed 3/3 closed candles",symbol)
    return ok


# ---------------------------------------------------------------------------
# [M8] Original risk lookup — exits keep working after breakeven
# ---------------------------------------------------------------------------

def get_initial_risk_usdt(symbol: str, direction: str, qty: float) -> float:
    """R in USDT based on ORIGINAL entry/SL stored at entry time."""
    journal = load_journal()
    for t in reversed(journal):
        if (t.get("symbol")==symbol and t.get("direction")==direction
                and t.get("outcome") is None):
            e0, s0 = t.get("entry"), t.get("sl")
            if e0 and s0:
                return abs(float(e0)-float(s0)) * qty
    return 0.0


# ---------------------------------------------------------------------------
# Unified exit manager — records outcome
# ---------------------------------------------------------------------------

def manage_exits(position_details:list, regime:Optional[dict]=None) -> dict:
    EXIT_THRESHOLD=3; closed={}; now_ms=int(time.time()*1000)
    for pos in position_details:
        sym=pos["symbol"]; side=pos["side"]
        qty=float(pos.get("size",0)); entry=float(pos.get("avgPrice",0) or 0)
        sl=float(pos.get("stopLoss",0) or 0); upnl=float(pos.get("unrealisedPnl",0) or 0)
        created_ms=int(pos.get("createdTime",now_ms) or now_ms)
        if entry==0 or qty==0: continue

        # [M8] R from original journal SL first; fall back to live SL
        r_usdt = get_initial_risk_usdt(sym, side, qty)
        if r_usdt <= 0:
            rd = abs(entry-sl) if sl else 0.0
            r_usdt = rd*qty

        hrs=(now_ms-created_ms)/3_600_000
        pnl_str=f"+${upnl:.2f}" if upnl>=0 else f"-${abs(upnl):.2f}"
        dlabel="LONG" if side=="Buy" else "SHORT"
        reason=None; icon="⚡"; label="QUICK EXIT"

        if r_usdt>0 and upnl>=QUICK_PROFIT_R*r_usdt:
            reason=f"0.8R profit (`{upnl:.2f}` ≥ `{QUICK_PROFIT_R*r_usdt:.2f}` USDT)"

        if reason is None:
            df1h=get_klines(sym,"60",limit=100)
            if df1h is not None and len(df1h)>=4:
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
            # uPnL is an estimate; reconciliation will overwrite with the real
            # closedPnl (incl. fees) on the next loop if a record is found.
            update_trade_outcome(sym,side,upnl,exit_reason=label.lower().replace(" ","_"))
            discord_notify(f"{icon} **{label} {dlabel} {sym}** | {reason} | "
                           f"PnL:`{pnl_str}` | Held:`{hrs:.1f}h`")
            closed[sym]=True
        else:
            discord_notify(f"⚠️ **{sym}** — exit FAILED | {reason}")
    return closed


# ---------------------------------------------------------------------------
# Position sizing — [M5] threshold tiers, [M9] hour adj needs ≥15 trades
# ---------------------------------------------------------------------------

def risk_pct_for_score(score: float) -> float:
    """Higher score → bigger size. Monotonic, no int() trap."""
    if score >= 6.0: return 0.025
    if score >= 5.0: return 0.020
    return 0.015

def calculate_position(balance:float, entry_price:float, sl_price:float,
                       leverage:int, score:float=4.0,
                       symbol:str="", learning_state:Optional[dict]=None) -> float:
    risk_pct = risk_pct_for_score(score)
    if learning_state and symbol:
        profile = learning_state.get("coin_profiles",{}).get(symbol,{})
        # [M9] only adjust by hour when sample is big enough; use precomputed hours
        if profile.get("trades_count",0) >= MIN_TRADES_FOR_HOURS:
            now_h = datetime.datetime.utcnow().hour
            if now_h in profile.get("best_hours_utc",[]):
                risk_pct *= 1.10
            elif now_h in profile.get("worst_hours_utc",[]):
                risk_pct *= 0.80
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
# Daily summary — [M6] now summarizes a specific date
# ---------------------------------------------------------------------------

def send_daily_summary(balance:float, for_date: Optional[datetime.date]=None) -> None:
    journal=load_journal()
    d=(for_date or datetime.date.today()).isoformat()
    day_t=[t for t in journal if str(t.get("timestamp","")).startswith(d)
           or str(t.get("closed_at","")).startswith(d)]
    comp=[t for t in day_t if t.get("outcome") in ("win","loss")]
    wins=sum(1 for t in comp if t["outcome"]=="win"); total=len(comp)
    wr=f"{wins/total*100:.0f}%" if total>0 else "n/a"
    pnl=sum((t.get("pnl") or 0) for t in comp)
    op=get_open_positions()
    discord_notify(
        f"📈 **DAILY SUMMARY** | {d}\n"
        f"• Trades closed: `{total}` | Win rate: `{wr}` ({wins}/{total})\n"
        f"• Realized PnL: `{'%+.2f'%pnl}` USDT (incl. fees where reconciled)\n"
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
    if df4h is None or df1h is None or len(df4h)<202 or len(df1h)<52: return None

    # [C3] volume filter on the LAST CLOSED 4H candle vs the 20 before it
    closed_vol = df4h["volume"].iloc[-2]
    avg_v      = df4h["volume"].iloc[-22:-2].mean()
    if avg_v>0 and closed_vol/avg_v<1.2:
        log.info("%s SKIP low vol (closed candle)",symbol); return None

    ind4h=compute_indicators(df4h); ind1h=compute_indicators(df1h)   # closed candles
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

    # [S2] Real ATR — true range includes gaps vs previous close
    df1h_c = _closed(df1h)
    atr = ta.volatility.AverageTrueRange(
        df1h_c["high"], df1h_c["low"], df1h_c["close"], window=14
    ).average_true_range().iloc[-1]
    if not atr or atr <= 0 or (isinstance(atr,float) and math.isnan(atr)): return None

    r4h=_closed(df4h).iloc[-20:]; ph=r4h["high"].max(); pl=r4h["low"].min(); pm=(ph+pl)/2
    leverage=min(MAX_LEVERAGE,10)
    set_leverage(symbol,leverage)

    # [S3] Maker entry: PostOnly limit at best bid/ask, re-quote once if unfilled
    result=None; entry=sl=tp=0.0; qty=0.0
    for attempt in range(ENTRY_MAX_ATTEMPTS):
        price=get_best_price(symbol,direction)
        if price is None: return None
        entry=snap_price(symbol,price)
        if direction=="Buy":
            sl=entry-2*atr; raw_tp=entry+4*atr
            cands=[p for p in [pm,ph] if entry<p<=raw_tp*1.02]
            tp=min(cands) if cands else raw_tp
        else:
            sl=entry+2*atr; raw_tp=entry-4*atr
            cands=[p for p in [pm,pl] if entry>p>=raw_tp*0.98]
            tp=max(cands) if cands else raw_tp
        sl=snap_price(symbol,sl); tp=snap_price(symbol,tp)
# [RR] Minimum R:R check
        risk   = abs(entry - sl)
        reward = abs(tp - entry)
        if risk <= 0 or reward / risk < 1.5:
            log.info("%s SKIP R:R too low (%.2f)", symbol, reward/risk if risk>0 else 0)
            return None

        # [RR] SL must not be closer than 2% to entry
        if abs(entry - sl) / entry < 0.02:
            log.info("%s SKIP SL too tight", symbol)
            return None
        qty=calculate_position(balance,entry,sl,leverage,score,symbol,learning_state)
        qty=snap_qty(symbol,qty)
        if qty<=0: return None

        res=place_limit_postonly(symbol,direction,qty,entry,sl,tp)
        if not res: return None
        oid=res.get("orderId","")
        status=wait_for_fill(symbol,oid,ENTRY_FILL_TIMEOUT_S)
        if status=="Filled":
            result=res; break
        if status=="Timeout":
            cancel_order(symbol,oid)
        log.info("%s entry attempt %d/%d not filled (%s) — re-quoting",
                 symbol,attempt+1,ENTRY_MAX_ATTEMPTS,status)

    if not result:
        log.info("%s SKIP — maker entry not filled, not chasing",symbol)
        return None

    trade={
        "timestamp":   datetime.datetime.utcnow().isoformat(),
        "symbol":      symbol, "direction": direction,
        "score":       round(score,3), "signals": details,   # [C4] same-side only
        "entry":       entry, "sl": sl, "tp": tp,
        "qty":         qty, "leverage": leverage,
        "hour_utc":    datetime.datetime.utcnow().hour,
        "outcome":     None, "pnl": None, "closed_at": None,
        "exit_reason": None,
        "tp1_done":    False, "tp1_realized": 0.0,           # [S4]
        "order_result":result,
    }
    log_trade(trade)
    emoji="🟢" if direction=="Buy" else "🔴"
    discord_notify(f"{emoji} **{direction} {symbol}** (maker) | Score `{score:.1f}/6` | "
                   f"Entry `{entry}` | SL `{sl}` | TP `{tp}` | "
                   f"Qty `{qty}` × {leverage}x")
    return trade


# ---------------------------------------------------------------------------
# HTTP keep-alive
# ---------------------------------------------------------------------------

HTTP_PORT = int(os.environ.get("PORT", 5000))

HTTP_PORT    = int(os.environ.get("PORT", 5000))
STATUS_TOKEN = os.environ.get("STATUS_TOKEN", "")
BOT_STARTED_AT = datetime.datetime.utcnow().isoformat()

class _PingHandler(BaseHTTPRequestHandler):
    def _send_json(self, obj, code=200):
        body = json.dumps(obj, indent=2, default=str).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _authorized(self, query: str) -> bool:
        if not STATUS_TOKEN:
            return True
        from urllib.parse import parse_qs
        return parse_qs(query).get("token", [""])[0] == STATUS_TOKEN

    def do_GET(self):
        from urllib.parse import urlparse
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/status":
                if not self._authorized(parsed.query):
                    return self._send_json({"error": "unauthorized"}, 401)
                journal   = load_journal()
                completed = [t for t in journal if t.get("outcome") in ("win","loss")]
                wins      = sum(1 for t in completed if t["outcome"]=="win")
                open_trades = [
                    {k: t.get(k) for k in ("symbol","direction","entry","sl","tp",
                                           "qty","score","signals","timestamp","tp1_done")}
                    for t in journal
                    if t.get("outcome") is None and t.get("order_result")
                ]
                ls = load_learning_state()
                self._send_json({
                    "bot": "APEX v4.1", "status": "running",
                    "started_at": BOT_STARTED_AT,
                    "time_utc": datetime.datetime.utcnow().isoformat(),
                    "balance_usdt": get_account_balance(),
                    "open_positions": get_open_positions(),
                    "open_trades_journal": open_trades,
                    "completed_trades": len(completed),
                    "wins": wins,
                    "win_rate": round(wins/len(completed),3) if completed else None,
                    "total_pnl_usdt": round(sum((t.get("pnl") or 0) for t in completed),4),
                    "last_5_closed": [
                        {k: t.get(k) for k in ("symbol","direction","outcome","pnl",
                                               "exit_reason","closed_at")}
                        for t in completed[-5:]
                    ],
                    "learning_last_updated": ls.get("last_updated"),
                    "signal_weights": ls.get("signal_weights"),
                })
            elif parsed.path == "/journal":
                if not self._authorized(parsed.query):
                    return self._send_json({"error": "unauthorized"}, 401)
                self._send_json(load_journal()[-20:])
            else:
                body = b"APEX Bot Running v4.1"
                self.send_response(200)
                self.send_header("Content-Type","text/plain")
                self.send_header("Content-Length",str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        except Exception as e:
            try: self._send_json({"error": str(e)}, 500)
            except Exception: pass

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
    log.info("APEX Bot v4.1 — Fast & Precise ACTIVE")
    log.info("Data dir: %s | Long enabled: %s", DATA_DIR, LONG_ENABLED)
    log.info("="*60)
    discord_notify(f"🚀 **APEX Bot v4.1 started** | Exits: every 5 min | "
                   f"Entries: every 15 min (candle close) | Maker entries: ✅ | "
                   f"Partial TP: ✅ | Data dir: `{DATA_DIR}` | "
                   f"Long: {'✅' if LONG_ENABLED else '❌'}")
    threading.Thread(target=_start_http_server, daemon=True).start()
    if not API_KEY or not API_SECRET: log.error("API keys not set"); return
    starting_balance=get_account_balance()
    if starting_balance is None:
        log.error("Balance fetch failed")
        discord_notify("💀 **APEX Bot CRASHED** | Balance fetch failed — API key issue!")
        return
    log.info("Starting balance: %.2f USDT", starting_balance)
    if DATA_DIR == ".":
        discord_notify("⚠️ **Warning**: DATA_DIR not set — journal/learning data "
                       "will be LOST on redeploy. Attach a Railway Volume and set "
                       "`DATA_DIR=/data`.")

    day_start=datetime.date.today()
    week_start=day_start-datetime.timedelta(days=day_start.weekday())
    week_bal=starting_balance; weekly_pause=None
    last_hc=datetime.datetime.utcnow()
    last_summary_date=datetime.date.today()   # [M6]
    weekly_done=False
    last_entry_slot=None                       # [S1] last 15-min slot scanned
    _wl_cache={"ts":0.0,"wl":[]}               # [S1] hourly watchlist cache

    def _sleep_to_next_tick():
        """Sleep to the next 5-min boundary +10s (so closed candles exist)."""
        s = FAST_INTERVAL_SECONDS - (time.time() % FAST_INTERVAL_SECONDS) + 10
        time.sleep(s)

    while True:
        try:
            now=datetime.datetime.utcnow(); today=datetime.date.today()
            ls=load_learning_state()

            # [C1] Reconcile exchange-closed trades EVERY tick, before anything else
            try:
                reconcile_closed_trades()
            except Exception as e:
                log.error("Reconcile failed: %s", e)

            if weekly_pause and now<weekly_pause:
                log.warning("Weekly pause — %.1fh left",(weekly_pause-now).total_seconds()/3600)
                _sleep_to_next_tick(); continue

            # [M6] New day → send summary for YESTERDAY on first tick of the day
            if today!=last_summary_date:
                b=get_account_balance()
                if b: send_daily_summary(b, for_date=last_summary_date)
                last_summary_date=today

            if today!=day_start:
                day_start=today; starting_balance=get_account_balance() or starting_balance
                log.info("New day — %.2f USDT",starting_balance)

            cw=today-datetime.timedelta(days=today.weekday())
            if cw!=week_start:
                week_start=cw; week_bal=get_account_balance() or starting_balance
                weekly_done=False
                log.info("New week — %.2f USDT",week_bal)

            # [M6] Weekly learning: first tick of every Sunday (any hour)
            if now.weekday()==6 and not weekly_done:
                run_weekly_learning(); weekly_done=True

            if (now-last_hc).total_seconds()>=21_600:
                b=get_account_balance() or 0; op=get_open_positions()
                t_count=len([t for t in load_journal() if t.get("outcome") in ("win","loss")])
                discord_notify(f"💚 **APEX Bot alive** | Positions:`{len(op)}/{MAX_OPEN_POSITIONS}` | "
                               f"Balance:`{b:.2f}` USDT | Learned:`{t_count}` trades")
                last_hc=now

            bal=get_account_balance()
            if bal is None: _sleep_to_next_tick(); continue

            if daily_loss_exceeded(starting_balance,bal):
                lp=(starting_balance-bal)/starting_balance*100
                log.warning("Daily loss limit hit (-%.1f%%) — paused",lp)
                _sleep_to_next_tick(); continue

            if weekly_loss_exceeded(week_bal,bal):
                weekly_pause=now+datetime.timedelta(hours=48)
                lp=(week_bal-bal)/week_bal*100
                discord_notify(f"🛑 **Weekly loss limit** | `{week_bal:.2f}`→`{bal:.2f}` "
                               f"(`-{lp:.1f}%`) | Pausing 48h")
                _sleep_to_next_tick(); continue

            # ---- FAST path (every 5 min): protect & manage open positions ----
            op=get_open_positions(); pc=len(op)
            if pc>0:
                pd4=get_position_details()
                if pd4:
                    regime_for_exit=get_market_regime()
                    manage_tp1(pd4)                       # [S4]
                    closed=manage_exits(pd4,regime_for_exit)
                    if closed: op=get_open_positions(); pc=len(op)
                log.info("[fast] Positions:%d/%d %s | Balance:%.2f",
                         pc,MAX_OPEN_POSITIONS,list(op.keys()),bal)

            # ---- ENTRY path (every 15 min, aligned to candle close) ----
            slot=(now.hour, (now.minute//15)*15)
            is_entry_tick = now.minute % 15 < 5 and slot != last_entry_slot
            if is_entry_tick and pc<MAX_OPEN_POSITIONS:
                last_entry_slot=slot
                regime=get_market_regime()

                # hourly watchlist cache
                if time.time()-_wl_cache["ts"]>WATCHLIST_TTL_SECONDS or not _wl_cache["wl"]:
                    wl=get_dynamic_watchlist()
                    if wl:
                        _wl_cache.update(ts=time.time(), wl=wl)
                        load_instrument_cache(wl)
                wl=_wl_cache["wl"]

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
                    time.sleep(1)

                rl=f"{regime.get('regime_4h','?')}/{regime.get('regime_1h','?')}→{regime.get('allowed','?')}"
                tc=len([t for t in load_journal() if t.get("outcome") in ("win","loss")])
                log.info("Entry scan complete [%02d:%02d] | Regime:%s",slot[0],slot[1],rl)
                # Discord summary only at the top of the hour — no 15-min spam
                if slot[1]==0:
                    discord_notify(f"📊 **Hourly scan** | Regime:`{rl}` | "
                                   f"Positions:`{len(op)}/{MAX_OPEN_POSITIONS}` | "
                                   f"Balance:`{bal:.2f}` USDT | Learned:`{tc}`")
            elif is_entry_tick:
                last_entry_slot=slot
                log.info("Entry slot %02d:%02d — max positions, skip scan",slot[0],slot[1])

            _sleep_to_next_tick()

        except Exception as e:
            # Catch-all: one bad iteration must never kill the bot
            log.exception("Main loop error: %s", e)
            discord_notify(f"⚠️ **Loop error (recovered)**: `{e}` — retrying in 5 min")
            time.sleep(300)


if __name__ == "__main__":
    main()
