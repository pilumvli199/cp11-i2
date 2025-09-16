#!/usr/bin/env python3
# main.py - Updated: MPIN-compatible SmartAPI login + safer backoff
# Keep .env updated with SMARTAPI_MPIN (preferred) OR SMARTAPI_PASSWORD + SMARTAPI_TOTP_SECRET

import os
import sys
import time
import json
import asyncio
import traceback
from datetime import datetime, timedelta
from tempfile import NamedTemporaryFile

import requests
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import date2num

from dotenv import load_dotenv
load_dotenv()

# optional libs with helpful messages
try:
    import pyotp
except Exception:
    print("Missing dependency 'pyotp'. Install: pip install pyotp")
    raise

try:
    import aiohttp
except Exception:
    print("Missing dependency 'aiohttp'. Install: pip install aiohttp")
    raise

# SmartAPI import (supports both package names)
try:
    try:
        from SmartApi import SmartConnect
    except Exception:
        from smartapi import SmartConnect
except Exception:
    print("Missing SmartAPI library. Install: pip install smartapi-python logzero websocket-client")
    raise

try:
    from openai import OpenAI
except Exception:
    print("Missing openai. Install: pip install openai")
    raise

# ---------------- CONFIG ----------------
SYMBOL_NAMES = ["NIFTY", "BANKNIFTY", "RELIANCE"]
TF_MAP = {"5m": "FIVE_MINUTE", "15m": "FIFTEEN_MINUTE", "30m": "THIRTY_MINUTE"}

POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", 120))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")

SMARTAPI_API_KEY = os.getenv("SMARTAPI_API_KEY")
SMARTAPI_API_SECRET = os.getenv("SMARTAPI_API_SECRET")
SMARTAPI_CLIENT_CODE = os.getenv("SMARTAPI_CLIENT_CODE")
SMARTAPI_PASSWORD = os.getenv("SMARTAPI_PASSWORD")
SMARTAPI_TOTP_SECRET = os.getenv("SMARTAPI_TOTP_SECRET")  # base32 secret
SMARTAPI_MPIN = os.getenv("SMARTAPI_MPIN")  # if Angel One enforces MPIN, set this

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_CLIENT = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

INSTR_MASTER_JSON_URL = "https://margincalculator.angelone.in/OpenAPI_File/files/OpenAPIScripMaster.json"
INSTR_CACHE_PATH = "/tmp/angel_instruments.json"
INSTR_CACHE_TTL = 24 * 3600

smartobj = None

# ---------------- Telegram helpers ----------------
async def send_text(session, text):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[Telegram] (skipped):", text)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        async with session.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}) as r:
            if r.status != 200:
                body = await r.text()
                print("Telegram send_text failed:", r.status, body[:300])
    except Exception as e:
        print("Telegram send_text exception:", e)

async def send_photo(session, caption, path):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        print("[Telegram] photo skipped:", caption)
        try: os.remove(path)
        except: pass
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
    try:
        with open(path, "rb") as f:
            data = aiohttp.FormData()
            data.add_field("chat_id", str(TELEGRAM_CHAT_ID))
            data.add_field("caption", caption)
            data.add_field("photo", f, filename=os.path.basename(path), content_type="image/png")
            async with aiohttp.ClientSession() as s:
                async with s.post(url, data=data, timeout=60) as r:
                    if r.status != 200:
                        body = await r.text()
                        print("Telegram send_photo failed:", r.status, body[:300])
    except Exception as e:
        print("Telegram send_photo exception:", e)
    finally:
        try: os.remove(path)
        except: pass

# ---------------- Instrument master ----------------
def download_instrument_master(force=False):
    try:
        if not force and os.path.exists(INSTR_CACHE_PATH):
            if time.time() - os.path.getmtime(INSTR_CACHE_PATH) < INSTR_CACHE_TTL:
                return json.load(open(INSTR_CACHE_PATH, "r"))
    except Exception:
        pass
    try:
        print("Downloading instrument master JSON...")
        r = requests.get(INSTR_MASTER_JSON_URL, timeout=30)
        r.raise_for_status()
        data = r.json()
        with open(INSTR_CACHE_PATH, "w") as f:
            json.dump(data, f)
        return data
    except Exception as e:
        print("Failed to download instrument master:", e)
        if os.path.exists(INSTR_CACHE_PATH):
            try:
                return json.load(open(INSTR_CACHE_PATH, "r"))
            except:
                pass
        return []

def build_symbol_token_map(instr_list):
    m = {}
    for item in instr_list:
        try:
            token = item.get("token") or item.get("symboltoken") or item.get("TOKEN")
            symbol = (item.get("symbol") or "").strip()
            name = (item.get("name") or "").strip()
            exch = item.get("exch_seg") or item.get("exchSeg") or ""
            entry = {"token": str(token), "symbol": symbol, "name": name, "exch": exch}
            if symbol:
                m.setdefault(symbol.upper(), []).append(entry)
            if name:
                m.setdefault(name.upper(), []).append(entry)
        except Exception:
            continue
    return m

def lookup_token(instr_map, friendly_name):
    if not instr_map:
        return None
    key = friendly_name.strip().upper()
    if key in instr_map:
        entries = instr_map[key]
        for e in entries:
            if e.get("exch") and "NSE" in e.get("exch"):
                return e
        return entries[0]
    for k, entries in instr_map.items():
        if key in k:
            return entries[0]
    for k, entries in instr_map.items():
        for e in entries:
            if key in (e.get("symbol","").upper()) or key in (e.get("name","").upper()):
                return e
    return None

# ---------------- SmartAPI login helpers ----------------
def generate_otps(secret):
    totp = pyotp.TOTP(secret)
    now = int(time.time())
    # prev, now, next
    return [totp.at(now - 30), totp.at(now), totp.at(now + 30)]

async def smartapi_login():
    """
    Attempt login with MPIN if provided, else password+TOTP fallback.
    Uses conservative retry/backoff to avoid rate-limits.
    """
    global smartobj
    if smartobj:
        return True

    # Basic env check
    if not SMARTAPI_API_KEY or not SMARTAPI_CLIENT_CODE:
        print("SmartAPI credentials incomplete (missing API_KEY or CLIENT_CODE).")
        return False

    loop = asyncio.get_running_loop()

    # If MPIN provided, prefer MPIN flow (one attempt)
    if SMARTAPI_MPIN and SMARTAPI_API_SECRET:
        print("[SmartAPI] Attempting MPIN login (using SMARTAPI_MPIN).")
        def _try_mpin():
            try:
                s = SmartConnect(api_key=SMARTAPI_API_KEY)
                resp = s.generateSession(SMARTAPI_CLIENT_CODE, SMARTAPI_MPIN, SMARTAPI_API_SECRET)
                # success detection differs across SDK versions
                if resp and (resp.get("status") is True or resp.get("data")):
                    try:
                        s.setfeedToken(resp['data'].get('feedToken'))
                    except Exception:
                        pass
                    return s, resp
                return None, resp
            except Exception as e:
                return None, {"status": False, "message": str(e)}

        smart, data = await loop.run_in_executor(None, _try_mpin)
        if smart:
            print("[SmartAPI] MPIN login success.")
            smartobj = smart
            return True
        else:
            print("[SmartAPI] MPIN login failed:", data)
            # if MPIN failed, do not aggressively fallback to password - return False
            return False

    # Otherwise try password+TOTP fallback with limited cycles
    if not (SMARTAPI_PASSWORD and SMARTAPI_TOTP_SECRET and SMARTAPI_API_SECRET):
        print("[SmartAPI] Password+TOTP credentials incomplete - set SMARTAPI_MPIN or set PASSWORD+TOTP+API_SECRET.")
        return False

    # Conservative login cycles to avoid rate-limit
    MAX_CYCLES = 3
    backoff = 30  # seconds, will exponential backoff
    for cycle in range(MAX_CYCLES):
        otps = generate_otps(SMARTAPI_TOTP_SECRET)
        print(f"[SmartAPI] Trying password+TOTP login (cycle {cycle+1}/{MAX_CYCLES}) with {len(otps)} candidate OTPs.")
        for otp in otps:
            print(f"[SmartAPI DEBUG] Trying OTP: {otp}")
            def _try_pwd(otp_local):
                try:
                    s = SmartConnect(api_key=SMARTAPI_API_KEY)
                    resp = s.generateSession(SMARTAPI_CLIENT_CODE, SMARTAPI_PASSWORD, otp_local)
                    if resp and (resp.get("status") is True or resp.get("data")):
                        try:
                            s.setfeedToken(resp['data'].get('feedToken'))
                        except Exception:
                            pass
                        return s, resp
                    return None, resp
                except Exception as e:
                    return None, {"status": False, "message": str(e)}
            smart, data = await loop.run_in_executor(None, lambda: _try_pwd(otp))
            # If server tells to use MPIN, abort and notify
            try:
                msg = (data.get("message") or "").upper() if isinstance(data, dict) else ""
            except Exception:
                msg = ""
            if msg and "MPIN" in msg:
                print("[SmartAPI] Server requires MPIN. Aborting password attempts. Please set SMARTAPI_MPIN in env.")
                async with aiohttp.ClientSession() as s:
                    await send_text(s, "⚠️ SmartAPI requires MPIN for login. Update `.env` with SMARTAPI_MPIN and redeploy.")
                return False

            # If success
            if smart:
                print("[SmartAPI] Password+TOTP login success.")
                smartobj = smart
                return True

            # If server rate-limited
            if isinstance(data, dict):
                dm = (data.get("message") or "").lower()
                if "access denied" in dm or "exceed" in dm or "rate" in dm:
                    print("[SmartAPI] Server rate-limited. Backing off.")
                    break  # break otps loop => will backoff below

            # small pause between OTP attempts to be polite
            await asyncio.sleep(0.5)

        # backoff before next cycle
        print(f"[SmartAPI] Login cycle {cycle+1} failed. Backing off {backoff}s before retrying.")
        await asyncio.sleep(backoff)
        backoff *= 2

    print("[SmartAPI] All login cycles failed. Check credentials (or add SMARTAPI_MPIN).")
    async with aiohttp.ClientSession() as s:
        await send_text(s, "❗ SmartAPI login failed after retries. Check credentials and whether MPIN is required.")
    return False

async def smartapi_get_candles(symboltoken, exchange, interval_str, from_dt, to_dt):
    if not await smartapi_login():
        return None
    params = {
        "exchange": exchange,
        "symboltoken": str(symboltoken),
        "interval": interval_str,
        "fromdate": from_dt.strftime("%Y-%m-%d %H:%M"),
        "todate": to_dt.strftime("%Y-%m-%d %H:%M")
    }
    def _call():
        try:
            return smartobj.getCandleData(params)
        except Exception as e:
            return {"status": False, "message": str(e)}
    loop = asyncio.get_running_loop()
    resp = await loop.run_in_executor(None, _call)
    if not resp or not resp.get("status"):
        print("[SmartAPI] getCandleData failed:", resp)
        return None
    rows = resp.get("data", [])
    candles = []; times = []
    for r in rows:
        try:
            t = int(r[0])//1000 if isinstance(r[0], (int,float)) else None
            o = float(r[1]); h = float(r[2]); l = float(r[3]); c = float(r[4]); v = float(r[5])
            candles.append([o,h,l,c,v]); times.append(t)
        except Exception:
            continue
    return {"candles": candles, "times": times}

# ---------------- (rest of bot: GPT analysis, detection, plotting) ----------------
async def analyze_patterns_with_gpt(symbol, candles, tf):
    if OPENAI_CLIENT is None or not candles:
        return None
    last = candles[-50:]
    data_str = "\n".join([f"{i}: O={o},H={h},L={l},C={c}" for i,(o,h,l,c,v) in enumerate(last)])
    prompt = (
        f"You are a professional technical analyst.\n"
        f"Analyze the following {tf} candles for {symbol}.\n"
        "Identify candlestick & chart patterns and give a short bias.\n"
        "Output concisely.\n\n"
        "Data:\n" + data_str
    )
    loop = asyncio.get_running_loop()
    def call_model():
        return OPENAI_CLIENT.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role":"user","content":prompt}],
            max_tokens=200,
            temperature=0.2
        )
    try:
        resp = await loop.run_in_executor(None, call_model)
        choice = resp.choices[0]
        if hasattr(choice, "message"):
            return choice.message.content.strip()
        else:
            return getattr(choice, "text", str(resp)).strip()
    except Exception as e:
        print("GPT error:", e)
        return None

def compute_levels(candles, lookback=50):
    if not candles:
        return (None,None,None)
    arr = candles[-lookback:] if len(candles)>=lookback else candles
    highs = sorted([c[1] for c in arr], reverse=True)
    lows = sorted([c[2] for c in arr])
    k = min(3, len(arr))
    res = sum(highs[:k])/k if highs else None
    sup = sum(lows[:k])/k if lows else None
    mid = (res+sup)/2 if (res is not None and sup is not None) else None
    return sup,res,mid

def detect_signal(sym, candles, tf):
    if not candles or len(candles)<5:
        return None
    last = candles[-1]; prev = candles[-2]
    sup,res,mid = compute_levels(candles)
    entry = last[3]; bias="NEUTRAL"; reason=[]; conf=50
    if res and entry>res:
        bias="BUY"; reason.append("Breakout"); conf+=15
    if sup and entry<sup:
        bias="SELL"; reason.append("Breakdown"); conf+=15
    if last[3]>last[0] and prev[3]<prev[0]:
        bias="BUY"; reason.append("Bullish Engulfing"); conf+=10
    if last[3]<last[0] and prev[3]>prev[0]:
        bias="SELL"; reason.append("Bearish Engulfing"); conf+=10
    sl=None; targets=[]
    if bias=="BUY":
        try: sl=min([c[2] for c in candles[-6:]])*0.997
        except: sl=entry*0.98
    if bias=="SELL":
        try: sl=max([c[1] for c in candles[-6:]])*1.003
        except: sl=entry*1.02
    if sl:
        risk=abs(entry-sl)
        if bias=="BUY":
            targets=[entry + risk*r for r in (1,2,3)]
        else:
            targets=[entry - risk*r for r in (1,2,3)]
    return {"tf":tf,"bias":bias,"entry":entry,"sl":sl,"targets":targets,"reason":"; ".join(reason),"conf":conf,"levels":{"sup":sup,"res":res,"mid":mid}}

def plot_multi_chart(tf_results, sym, trades):
    fig, axs = plt.subplots(3,1,figsize=(9,12),dpi=100,sharex=False)
    for ax, tf_key in zip(axs, TF_MAP.keys()):
        data = tf_results.get(tf_key, {})
        candles = data.get("candles"); times = data.get("times")
        trade = next((t for t in trades if t["tf"]==tf_key), None)
        if not candles:
            ax.set_title(f"{sym} {tf_key} - no data")
            continue
        dates = [datetime.utcfromtimestamp(t) for t in times]
        o=[c[0] for c in candles]; h=[c[1] for c in candles]; l=[c[2] for c in candles]; c_=[c[3] for c in candles]
        x = date2num(dates); width = 0.6*(x[1]-x[0]) if len(x)>1 else 0.4
        for xi,oi,hi,li,ci in zip(x,o,h,l,c_):
            col="white" if ci>=oi else "black"
            ax.vlines(xi,li,hi,color="black",linewidth=0.6)
            ax.add_patch(plt.Rectangle((xi-width/2,min(oi,ci)),width,max(0.0001,abs(ci-oi)),facecolor=col,edgecolor="black"))
        if trade:
            if trade.get("entry") is not None:
                ax.axhline(trade["entry"],color="blue",label=f"Entry {trade['entry']}")
            if trade.get("sl") is not None:
                ax.axhline(trade["sl"],color="red",linestyle="--",label=f"SL {trade['sl']}")
            for j,trg in enumerate(trade.get("targets",[])):
                ax.axhline(trg,color="green",linestyle=":",label=f"T{j+1} {trg}")
        levs = trade.get("levels") if trade else None
        if levs:
            if levs.get("res") is not None:
                ax.axhline(levs["res"],linestyle="--",color="orange",linewidth=0.6)
            if levs.get("sup") is not None:
                ax.axhline(levs["sup"],linestyle="--",color="purple",linewidth=0.6)
        if trade and trade.get("ai_patterns"):
            txt = trade["ai_patterns"][:300]
            ax.text(0.01,0.98,f"AI: {txt}",transform=ax.transAxes,fontsize=8,verticalalignment='top',bbox=dict(boxstyle="round,pad=0.3",facecolor="white",alpha=0.8))
        ax.set_title(f"{sym} {tf_key}")
        ax.legend(loc="upper left",fontsize="small")
    fig.autofmt_xdate()
    tmp = NamedTemporaryFile(delete=False,suffix=".png")
    fig.savefig(tmp.name,bbox_inches="tight")
    plt.close(fig)
    return tmp.name

# ---------------- MAIN LOOP ----------------
async def loop():
    instr_list = download_instrument_master(force=True)
    instr_map = build_symbol_token_map(instr_list)
    resolved = {}
    for n in SYMBOL_NAMES:
        e = lookup_token(instr_map, n)
        if e:
            resolved[n] = e
            print(f"[INSTR] Resolved {n} -> token={e.get('token')} exch={e.get('exch')}")
        else:
            print(f"[INSTR] Could not resolve {n}")

    async with aiohttp.ClientSession() as session:
        await send_text(session, f"Bot started (MPIN mode={'YES' if SMARTAPI_MPIN else 'NO'})")
        while True:
            try:
                for friendly, info in resolved.items():
                    token = info.get("token"); exch = info.get("exch") or "NSE"
                    if not token:
                        continue
                    tf_results = {}; trades=[]
                    for tf_key, tf_interval in TF_MAP.items():
                        minutes = 5 if tf_key=="5m" else (15 if tf_key=="15m" else 30)
                        total_minutes = minutes * 110
                        to_dt = datetime.now(); from_dt = to_dt - timedelta(minutes=total_minutes)
                        data = await smartapi_get_candles(token, exch, tf_interval, from_dt, to_dt)
                        if data and data.get("candles"):
                            tf_results[tf_key] = data
                            tr = detect_signal(friendly, data["candles"], tf_key)
                            if tr and tr.get("bias") in ("BUY","SELL") and tr.get("conf",0) >= 60:
                                gpt = await analyze_patterns_with_gpt(friendly, data["candles"], tf_key)
                                if gpt: tr["ai_patterns"]=gpt
                                trades.append(tr)
                    if trades:
                        lines=[f"🚨 *{friendly}* Signals (5m/15m/30m):"]
                        for t in trades:
                            lines.append(f"[{t['tf']}] {t['bias']} Entry=`{t['entry']}` SL=`{t['sl']}` Targets={t['targets']} | Conf={t['conf']}% | {t['reason']}")
                            if t.get("ai_patterns"):
                                lines.append(f"[{t['tf']}] AI: {t['ai_patterns']}")
                        text="\n".join(lines)
                        try:
                            chart = plot_multi_chart(tf_results, friendly, trades)
                        except Exception as e:
                            print("plot error:",e); chart=None
                        await send_text(session, text)
                        if chart:
                            await send_photo(session, text, chart)
                await asyncio.sleep(POLL_INTERVAL)
            except Exception as e:
                print("Main loop exception:", e)
                traceback.print_exc()
                await asyncio.sleep(30)

if __name__=="__main__":
    try:
        asyncio.run(loop())
    except KeyboardInterrupt:
        print("Stopped by user.")
