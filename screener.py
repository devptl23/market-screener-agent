import os
import json
import requests
import yfinance as yf
from google import genai
from google.genai import errors
from google.genai import types

# -------------------------------------------------------------
# CONFIGURATION & PORTFOLIO DEFINITION
# -------------------------------------------------------------
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-pro-preview")
GEMINI_FALLBACK_MODEL = os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-3.8-flash")
MIN_AVG_DOLLAR_VOLUME = 20_000_000
MIN_SMALL_CAP_MARKET_CAP = 300_000_000
MAX_SMALL_CAP_MARKET_CAP = 5_000_000_000
MIN_TARGET_RETURN = 0.15
MAX_TARGET_RETURN = 0.20
MIN_REWARD_RISK = 3.0
MAX_REWARD_RISK = 5.0


def to_float(value):
    """Convert a scalar or one-column pandas object returned by yfinance to float."""
    while hasattr(value, "iloc"):
        value = value.iloc[-1]
    return float(value)

# Replace or update these with your exact active positions
CURRENT_PORTFOLIO = [
    {"ticker": "IREN", "shares": 83, "avg_cost": 64.36, "currency": "USD"},
    {"ticker": "MSTR", "shares": 12, "avg_cost": 158.41, "currency": "USD"},
    {"ticker": "QQC.TO", "shares": 36, "avg_cost": 51.43, "currency": "CAD"},
]

WATCHLIST = [
    {"ticker": "ACLS", "benchmark": "SMH"},
    {"ticker": "AMBA", "benchmark": "SMH"},
    {"ticker": "ARLO", "benchmark": "XLK"},
    {"ticker": "BL", "benchmark": "XLK"},
    {"ticker": "CELH", "benchmark": "XLP"},
    {"ticker": "FN", "benchmark": "SMH"},
    {"ticker": "PLAB", "benchmark": "SMH"},
    {"ticker": "PUBM", "benchmark": "XLC"},
    {"ticker": "RAMP", "benchmark": "XLK"},
    {"ticker": "TMDX", "benchmark": "XLV"},
    {"ticker": "VITL", "benchmark": "XLP"},
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
        ema20 = to_float(data[ticker].ewm(span=20, adjust=False).mean())
        price = to_float(data[ticker])
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
            current_px = to_float(hist)
            sma50 = to_float(hist.rolling(50).mean())
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
        try:
            market_cap = float(yf.Ticker(t).fast_info["market_cap"])
        except (KeyError, TypeError, ValueError):
            continue

        if not MIN_SMALL_CAP_MARKET_CAP <= market_cap <= MAX_SMALL_CAP_MARKET_CAP:
            continue

        raw_data = yf.download(
            [t, bm], period="3mo", interval="1d", progress=False
        )
        close = raw_data["Close"]
        volume = raw_data["Volume"]
        if t in close and bm in close:
            t_close = close[t].dropna()
            t_volume = volume[t].reindex(t_close.index).fillna(0)
            t_px = to_float(t_close)
            t_ema20 = to_float(t_close.ewm(span=20).mean())
            t_sma50 = to_float(t_close.rolling(50).mean())
            avg_dollar_volume = to_float(
                (t_close * t_volume).rolling(20).mean()
            )

            if avg_dollar_volume < MIN_AVG_DOLLAR_VOLUME:
                continue
            
            # Check 3-month relative performance vs sector ETF
            t_first_px = to_float(t_close.iloc[0])
            bm_first_px = to_float(close[bm].dropna().iloc[0])
            bm_last_px = to_float(close[bm])
            t_perf = (t_px - t_first_px) / t_first_px
            bm_perf = (bm_last_px - bm_first_px) / bm_first_px
            target_return = MIN_TARGET_RETURN
            target_price = t_px * (1 + target_return)
            atr14 = to_float(
                t_close.diff().abs().rolling(14).mean()
            )
            reward_risk = (target_price - t_px) / atr14 if atr14 else 0
            
            if (
                t_px > t_ema20 > t_sma50
                and t_perf > bm_perf
                and MIN_REWARD_RISK <= reward_risk <= MAX_REWARD_RISK
            ):
                qualified_candidates.append({
                    "ticker": t,
                    "price": round(t_px, 2),
                    "benchmark": bm,
                    "market_cap": round(market_cap),
                    "outperforming_benchmark": True,
                    "avg_dollar_volume": round(avg_dollar_volume),
                    "target_price": round(target_price, 2),
                    "target_return": f"{target_return:.0%}",
                    "atr14": round(atr14, 2),
                    "reward_to_risk": round(reward_risk, 2)
                })

    return portfolio_metrics, qualified_candidates

# -------------------------------------------------------------
# STEP C: GENERATE AI TRADE REPORT
# -------------------------------------------------------------
def generate_trade_report(macro_favorable, macro_data, portfolio_data, candidates):
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    prompt = f"""
    You are an expert quantitative swing trader producing a concise Discord report. Evaluate only the supplied live data. Do not invent prices, indicators, market caps, news, support levels, or institutional activity. Do not force a trade; return NO TRADE when the data does not support every requirement.

    INPUT DATA:
    - Macro Regime Favorable: {macro_favorable}
    - Macro Status: {json.dumps(macro_data)}
    - User Portfolio: {json.dumps(portfolio_data)}
    - Pre-Screened Candidates: {json.dumps(candidates)}

    CRITERIA TO ENFORCE:
    1. If macro is unfavorable, show a red capital-preservation alert and recommend NO TRADE unless a valid exception is supported by the data.
    2. Audit holdings using only the supplied fields. Never give an "immediate exit" order; label the setup as BROKEN, WATCH, or HOLD and state the objective reason.
    3. Only consider candidates with market cap between $300M and $5B and average dollar volume above $20M. Reject microcaps, thinly traded names, pump-and-dump setups, and extended or parabolic price action.
    4. Pick 0-2 candidates only when the setup supports a 15% to 20% target and a strict 1:3 to 1:5 reward-to-risk ratio. Use the supplied target and reward-to-risk values; do not replace them with 1:2 setups.
    5. A 15% to 20% return is a target, never a promise. State the invalidation condition and main risk for each trade.

    DISCORD FORMAT RULES:
    - Use Discord Markdown only. Do not use HTML, LaTeX, or wide ASCII tables.
    - Keep the report under 3,500 characters and make the first section scannable in under 10 seconds.
    - Use exactly these sections and numbering; do not repeat numbers:

    **MARKET SWEEP | [GREEN LIGHT / CAUTION / RED LIGHT]**
    `As of: [date/time if supplied]`
    `SPY [BULLISH/BEARISH] | QQQ [BULLISH/BEARISH] | SMH [BULLISH/BEARISH]`
    `Action: [LONGS ALLOWED / REDUCE RISK / NO NEW LONGS]`

    **QUICK TAKE**
    `Candidates: [count] | Valid setups: [count] | Target: 15-20% | R:R: 1:3-1:5`
    `Best idea: [ticker or NONE] | Risk: [LOW/MODERATE/HIGH]`

    **PORTFOLIO**
    - `TICKER` | P&L | Trend | **Action:** HOLD / WATCH / REDUCE

    **TRADE SETUPS**
    For each valid setup, use this compact card:
    **#1 TICKER | [LONG / NO TRADE]**
    `Entry: $X | Stop: $Y | Target: $Z`
    `Risk: X% | Expected return: Y% | R:R: 1:Z`
    `Why: one sentence. Invalidation: one sentence.`

    **RISK RULES**
    `Risk per trade: [state only if supplied] | Never risk more than planned | 15-20% is not guaranteed.`

    - If there are no valid setups, write **NO VALID SETUPS** and explain in one sentence.
    - Use consistent dollar formatting and short lines. Avoid long paragraphs, repeated disclaimers, and speculative claims.
    """

    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_level="high"
        )
    )

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config=config
        )
        return response.text
    except errors.ClientError as error:
        status_code = getattr(error, "code", getattr(error, "status_code", None))
        if status_code != 429 or GEMINI_MODEL == GEMINI_FALLBACK_MODEL:
            raise

        # Pro may have no free-tier quota; retry with the configured fallback.
        try:
            response = client.models.generate_content(
                model=GEMINI_FALLBACK_MODEL,
                contents=prompt,
                config=config
            )
            return response.text
        except errors.ClientError as fallback_error:
            fallback_status = getattr(
                fallback_error,
                "code",
                getattr(fallback_error, "status_code", None)
            )
            if fallback_status == 429:
                return (
                    "Gemini quota is exhausted for both "
                    f"{GEMINI_MODEL} and {GEMINI_FALLBACK_MODEL}. "
                    "Enable billing or wait for the quota to reset."
                )
            if fallback_status == 404:
                return (
                    f"Gemini fallback model {GEMINI_FALLBACK_MODEL} is unavailable. "
                    "Set GEMINI_FALLBACK_MODEL to an available model."
                )
            raise

# -------------------------------------------------------------
# STEP D: SEND TO DISCORD
# -------------------------------------------------------------
def send_to_discord(text):
    # Keep Discord messages below its limit without cutting a line in half.
    chunks = []
    current_chunk = ""
    for line in text.splitlines(keepends=True):
        if current_chunk and len(current_chunk) + len(line) > 1900:
            chunks.append(current_chunk.rstrip())
            current_chunk = ""
        current_chunk += line
    if current_chunk.strip():
        chunks.append(current_chunk.rstrip())

    for chunk in chunks:
        requests.post(DISCORD_WEBHOOK_URL, json={"content": chunk})

if __name__ == "__main__":
    is_macro_ok, macro_info = check_macro_regime()
    portfolio_info, candidates_info = analyze_market_data()
    report = generate_trade_report(is_macro_ok, macro_info, portfolio_info, candidates_info)
    send_to_discord(report)