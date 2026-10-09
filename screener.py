import os
import json
import requests
import yfinance as yf
from google import genai
from google.genai import types

# -------------------------------------------------------------
# CONFIGURATION & PORTFOLIO DEFINITION
# -------------------------------------------------------------
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview")

# Replace or update these with your exact active positions
CURRENT_PORTFOLIO = [
    {"ticker": "IREN", "shares": 83, "avg_cost": 64.36, "currency": "USD"},
    {"ticker": "MSTR", "shares": 12, "avg_cost": 158.41, "currency": "USD"},
    {"ticker": "QQC.TO", "shares": 36, "avg_cost": 51.43, "currency": "CAD"},
]

WATCHLIST = [
    {"ticker": "NVDA", "benchmark": "SMH"},
    {"ticker": "AVGO", "benchmark": "SMH"},
    {"ticker": "AMD",  "benchmark": "SMH"},
    {"ticker": "MSFT", "benchmark": "XLK"},
    {"ticker": "AAPL", "benchmark": "XLK"},
    {"ticker": "AMZN", "benchmark": "XLY"},
    {"ticker": "META", "benchmark": "XLC"},
    {"ticker": "PLTR", "benchmark": "XLK"},
    {"ticker": "TSLA", "benchmark": "XLY"},
]

# -------------------------------------------------------------
# STEP A: MACRO REGIME CHECK (SPY, QQQ, SMH)
# -------------------------------------------------------------
def check_macro_regime():
    macro_tickers = ["SPY", "QQQ", "SMH"]
    data = yf.download(macro_tickers, period="2mo", interval="1d", progress=False)["Close"]
    status = {}
    favorable = True
    
    for ticker in macro_tickers:
        ema20 = data[ticker].ewm(span=20, adjust=False).mean().iloc[-1]
        price = data[ticker].iloc[-1]
        is_bullish = bool(price > ema20)
        if not is_bullish:
            favorable = False
        status[ticker] = {
            "price": round(float(price), 2),
            "ema20": round(float(ema20), 2),
            "above_20ema": is_bullish
        }
    return favorable, status

# -------------------------------------------------------------
# STEP B: AUDIT CURRENT HOLDINGS & PRE-SCREEN CANDIDATES
# -------------------------------------------------------------
def analyze_market_data():
    portfolio_metrics = []
    for item in CURRENT_PORTFOLIO:
        t = item["ticker"]
        hist = yf.download(t, period="3mo", interval="1d", progress=False)["Close"]
        if not hist.empty:
            current_px = float(hist.iloc[-1])
            sma50 = float(hist.rolling(50).mean().iloc[-1])
            pnl_pct = ((current_px - item["avg_cost"]) / item["avg_cost"]) * 100
            portfolio_metrics.append({
                "ticker": t,
                "current_price": round(current_px, 2),
                "avg_cost": item["avg_cost"],
                "pnl_pct": f"{pnl_pct:.2f}%",
                "below_50sma": current_px < sma50
            })

    # Pre-screen candidates for trend
    qualified_candidates = []
    for item in WATCHLIST:
        t = item["ticker"]
        bm = item["benchmark"]
        hist = yf.download([t, bm], period="3mo", interval="1d", progress=False)["Close"]
        if t in hist and bm in hist:
            t_px = float(hist[t].iloc[-1])
            t_ema20 = float(hist[t].ewm(span=20).mean().iloc[-1])
            t_sma50 = float(hist[t].rolling(50).mean().iloc[-1])
            
            # Check 3-month relative performance vs sector ETF
            t_perf = (t_px - float(hist[t].iloc[0])) / float(hist[t].iloc[0])
            bm_perf = (float(hist[bm].iloc[-1]) - float(hist[bm].iloc[0])) / float(hist[bm].iloc[0])
            
            if t_px > t_ema20 > t_sma50 and t_perf > bm_perf:
                qualified_candidates.append({
                    "ticker": t,
                    "price": round(t_px, 2),
                    "benchmark": bm,
                    "outperforming_benchmark": True
                })

    return portfolio_metrics, qualified_candidates

# -------------------------------------------------------------
# STEP C: GENERATE AI TRADE REPORT
# -------------------------------------------------------------
def generate_trade_report(macro_favorable, macro_data, portfolio_data, candidates):
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    prompt = f"""
    You are an expert quantitative swing trader. Evaluate this live data and provide recommendations following the user's strict swing criteria.

    INPUT DATA:
    - Macro Regime Favorable: {macro_favorable}
    - Macro Status: {json.dumps(macro_data)}
    - User Portfolio: {json.dumps(portfolio_data)}
    - Pre-Screened Candidates: {json.dumps(candidates)}

    CRITERIA TO ENFORCE:
    1. If macro is unfavorable, issue a capital preservation alert.
    2. Audit user holdings: Identify broken charts (trading below 50 SMA or heavy underperformance) and suggest reallocation.
    3. Pick the top 1-2 candidates meeting 1:2 to 1:3 Risk-to-Reward parameters with strict Entry, Stop Loss, and Targets.
    """

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                thinking_level="high"
            )
        )
    )
    return response.text

# -------------------------------------------------------------
# STEP D: SEND TO DISCORD
# -------------------------------------------------------------
def send_to_discord(text):
    # Discord limits messages to 2000 characters; split if necessary
    chunks = [text[i:i+1900] for i in range(0, len(text), 1900)]
    for chunk in chunks:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": chunk})

if __name__ == "__main__":
    is_macro_ok, macro_info = check_macro_regime()
    portfolio_info, candidates_info = analyze_market_data()
    report = generate_trade_report(is_macro_ok, macro_info, portfolio_info, candidates_info)
    send_to_discord(report)