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
TARGET_PROFIT_MIN = 200
TARGET_PROFIT_MAX = 500


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
    watchlist_candidates = []
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

            candidate = {
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
            }

            if t_px > t_ema20 > t_sma50 and t_perf > bm_perf:
                watchlist_candidates.append(candidate)
                if MIN_REWARD_RISK <= reward_risk <= MAX_REWARD_RISK:
                    qualified_candidates.append(candidate)

    watchlist_candidates.sort(
        key=lambda item: (item["reward_to_risk"], item["target_return"]),
        reverse=True
    )
    return portfolio_metrics, qualified_candidates, watchlist_candidates[:3]

# -------------------------------------------------------------
# STEP C: GENERATE AI TRADE REPORT
# -------------------------------------------------------------
def generate_trade_report(
    macro_favorable,
    macro_data,
    portfolio_data,
    candidates,
    watchlist_candidates
):
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    prompt = f"""
    You are an expert quantitative swing trader producing a detailed but well-structured Discord report. Evaluate only the supplied live data. Do not invent prices, indicators, market caps, news, support levels, or institutional activity. Do not force a trade; return NO TRADE when the data does not support every requirement.

    INPUT DATA:
    - Macro Regime Favorable: {macro_favorable}
    - Macro Status: {json.dumps(macro_data)}
    - User Portfolio: {json.dumps(portfolio_data)}
    - Pre-Screened Candidates: {json.dumps(candidates)}
    - Promising Watchlist Candidates (not necessarily entry-ready): {json.dumps(watchlist_candidates)}
    - Desired Gross Profit Range Per Trade: ${TARGET_PROFIT_MIN}-${TARGET_PROFIT_MAX}

    CRITERIA TO ENFORCE:
    1. If macro is unfavorable, show a red capital-preservation alert and recommend NO TRADE unless a valid exception is supported by the data.
    2. Audit every holding using only the supplied fields. For each holding, show ticker, current price, average cost, P&L, below-50-SMA status, classification (BROKEN/WATCH/HOLD), objective reasoning, and a practical action. Do not present a recommendation as guaranteed financial advice.
    3. Only consider candidates with market cap between $300M and $5B and average dollar volume above $20M. Reject microcaps, thinly traded names, pump-and-dump setups, and extended or parabolic price action.
    4. Pick 0-2 candidates only when the final target supports a 15% to 20% expected move AND the final reward-to-risk ratio is at least 1:3. Prefer 1:3 to 1:5. Never show a 1:2 or 1:2.8 trade as valid. If the supplied candidate has reward_to_risk below 3, reject it and say why.
    5. Use the supplied price, target_price, target_return, atr14, market_cap, avg_dollar_volume, and reward_to_risk. Use current price as the proposed entry unless the data clearly supports a different entry. For a provisional stop, use entry minus one ATR14 for a long setup and show the calculation. Do not invent chart support or resistance.
    6. A 15% to 20% return is a target, never a promise. State the invalidation condition, position-sizing caution, and main risk for each trade.
    7. Even when there are no valid trades, show up to three promising watchlist candidates from the supplied watchlist data. Label each WATCH ONLY and explain whether it failed the 1:3 R:R rule or another requirement. Do not present watchlist names as buy signals.
    8. The desired gross profit is $200-$500 per trade, but do not guess share count or account size. If account size and risk budget are not supplied, state that exact position sizing requires those inputs. You may show the formula: shares = desired dollar profit / (target price - entry price).

    DISCORD FORMAT RULES:
    - Use Discord Markdown only. Do not use HTML, LaTeX, or wide ASCII tables.
    - Keep the report detailed, but use short paragraphs, bullets, and separators so it remains easy to scan.
    - Put a quick dashboard first, followed by the complete detailed analysis.
    - Use exactly these sections and numbering; do not repeat numbers:

    **MARKET SWEEP | [GREEN LIGHT / CAUTION / RED LIGHT]**
    `As of: [date/time if supplied]`
    `SPY [BULLISH/BEARISH] | QQQ [BULLISH/BEARISH] | SMH [BULLISH/BEARISH]`
    `Action: [LONGS ALLOWED / REDUCE RISK / NO NEW LONGS]`

    **QUICK TAKE**
    `Candidates: [count] | Valid setups: [count] | Target: 15-20% | R:R: 1:3-1:5`
    `Best idea: [ticker or NONE] | Risk: [LOW/MODERATE/HIGH]`

    **PORTFOLIO AUDIT & CAPITAL REALLOCATION**
    | Ticker | Current | Avg Cost | P&L | Below 50 SMA | Action |
    Include one row for every holding, followed by a detailed diagnostic paragraph for every holding.

    **TOP QUANTITATIVE SWING TRADE SETUPS**
    Start with a compact side-by-side-style summary using separate blocks, then provide detailed analysis for every selected setup:
    **#1 TICKER | LONG**
    `Market cap: $X | Avg dollar volume: $X | Benchmark: TICKER`
    `Entry: $X | Stop: $Y | Target: $Z`
    `Risk: $X (X%) | Expected return: X% | Final R:R: 1:X`
    `Setup thesis:` two or three sentences based only on the supplied data.
    `Entry trigger:` explain what must happen before entry.
    `Invalidation:` explain the exact condition that cancels the trade.
    `Trade management:` explain partial profit-taking and stop management without guaranteeing a result.
    Include a clear **REJECTED / NO TRADE** subsection for candidates that fail the 1:3 minimum.

    **PROMISING WATCHLIST | NOT ENTRY SIGNALS**
    Show up to three names even when there are no valid trades:
    `TICKER | Why promising | Current filter status | What must improve before entry`
    Clearly label every one **WATCH ONLY**.

    **EXECUTION & RISK RULES**
    Include capital-reallocation considerations, stop-loss governance, position-sizing caution, and what to do if Target 1 is reached. State clearly that the 15-20% objective is not guaranteed.

    - If there are no valid setups, write **NO VALID SETUPS** and explain which filter failed.
    - Use consistent dollar formatting, readable Markdown tables, headings, and horizontal separators.
    - Preserve all supplied macro and portfolio information; do not omit details merely to shorten the message.
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
    except errors.APIError as error:
        status_code = getattr(error, "code", getattr(error, "status_code", None))
        if status_code not in (429, 503) or GEMINI_MODEL == GEMINI_FALLBACK_MODEL:
            raise

        # Pro may have no free-tier quota; retry with the configured fallback.
        try:
            response = client.models.generate_content(
                model=GEMINI_FALLBACK_MODEL,
                contents=prompt,
                config=config
            )
            return response.text
        except errors.APIError as fallback_error:
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
            if fallback_status == 503:
                return (
                    f"Gemini is temporarily unavailable for both {GEMINI_MODEL} "
                    f"and {GEMINI_FALLBACK_MODEL}. Try the next scheduled sweep."
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
    portfolio_info, candidates_info, watchlist_info = analyze_market_data()
    report = generate_trade_report(
        is_macro_ok,
        macro_info,
        portfolio_info,
        candidates_info,
        watchlist_info
    )
    send_to_discord(report)