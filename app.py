import io
import time
import requests
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =============================================================================
# STREAMLIT CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Nifty Total Market Research Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }

        .sub-title {
            color: #6b7280;
            font-size: 1rem;
            margin-bottom: 1.2rem;
        }

        .card {
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 14px;
            background: #ffffff;
            margin-bottom: 10px;
        }

        .success-card {
            border-left: 5px solid #16a34a;
        }

        .warning-card {
            border-left: 5px solid #f59e0b;
        }

        .danger-card {
            border-left: 5px solid #dc2626;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# CONSTANTS
# =============================================================================

NIFTY_TOTAL_MARKET_CSV = (
    "https://niftyindices.com/IndexConstituent/"
    "ind_niftytotalmarket_list.csv"
)

FALLBACK_SYMBOLS = [
    "RELIANCE",
    "TCS",
    "HDFCBANK",
    "ICICIBANK",
    "INFY",
    "SBIN",
    "LT",
    "ITC",
    "BHARTIARTL",
    "HINDUNILVR",
    "BAJFINANCE",
    "KOTAKBANK",
    "MARUTI",
    "SUNPHARMA",
    "TATAMOTORS",
    "TATASTEEL",
    "WIPRO",
    "HCLTECH",
    "ADANIENT",
    "POWERGRID",
]

PATTERN_OPTIONS = [
    "Any",
    "Double Top",
    "Double Bottom",
    "Head & Shoulders",
    "Inverse Head & Shoulders",
    "Rectangle",
    "Ascending Triangle",
    "Descending Triangle",
    "Symmetrical Triangle",
    "Reversal Bottom",
    "Reversal Top",
]

TIMEFRAMES = [
    "Daily",
    "Weekly",
    "Monthly",
]


# =============================================================================
# DATA FUNCTIONS
# =============================================================================

@st.cache_data(ttl=21600, show_spinner=False)
def get_nifty_total_market_members():
    """
    Downloads the official Nifty Total Market constituent list.

    If the official file is unavailable, blocked, slow, or malformed,
    the dashboard remains usable with a fallback stock list.
    """

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ),
        "Accept": "text/csv,application/csv,text/plain,*/*",
        "Referer": "https://www.niftyindices.com/",
    }

    try:
        response = requests.get(
            NIFTY_TOTAL_MARKET_CSV,
            headers=headers,
            timeout=15,
        )

        response.raise_for_status()

        data = pd.read_csv(io.BytesIO(response.content))
        data.columns = [str(column).strip() for column in data.columns]

        if "Symbol" not in data.columns:
            raise ValueError("Downloaded constituent file has no Symbol column.")

        if "Series" in data.columns:
            data = data[
                data["Series"]
                .astype(str)
                .str.upper()
                .eq("EQ")
            ].copy()

        data["Symbol"] = (
            data["Symbol"]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        data["Ticker"] = data["Symbol"] + ".NS"

        if "Company Name" not in data.columns:
            data["Company Name"] = data["Symbol"]

        if "Industry" not in data.columns:
            data["Industry"] = "Unknown"

        data = data[
            [
                "Company Name",
                "Industry",
                "Symbol",
                "Ticker",
            ]
        ]

        data = data.drop_duplicates("Symbol").reset_index(drop=True)

        if len(data) < 100:
            raise ValueError(
                f"Only {len(data)} stocks received; constituent data is incomplete."
            )

        return data, None

    except Exception as error:
        fallback = pd.DataFrame(
            {
                "Company Name": FALLBACK_SYMBOLS,
                "Industry": "Fallback universe",
                "Symbol": FALLBACK_SYMBOLS,
                "Ticker": [
                    f"{symbol}.NS"
                    for symbol in FALLBACK_SYMBOLS
                ],
            }
        )

        error_message = (
            f"Could not load the live Nifty Total Market constituent list. "
            f"Using a fallback list of {len(fallback)} large stocks. "
            f"Technical reason: {type(error).__name__}: {error}"
        )

        return fallback, error_message


@st.cache_data(ttl=900, show_spinner=False)
def fetch_price_data(ticker, period="5y"):
    """
    Fetches OHLCV data from Yahoo Finance.

    Cached for 15 minutes to reduce repeated requests.
    """

    try:
        stock = yf.Ticker(ticker)

        data = stock.history(
            period=period,
            auto_adjust=True,
        )

        if data is None or data.empty:
            return pd.DataFrame()

        data = data.rename(columns=str.lower)

        required_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        data = data[required_columns].dropna()

        data.index = pd.to_datetime(data.index)

        if getattr(data.index, "tz", None) is not None:
            data.index = data.index.tz_localize(None)

        return data

    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=43200, show_spinner=False)
def fetch_fundamentals(ticker):
    """
    Fetches available fundamental data from Yahoo Finance.

    Cached for 12 hours because fundamentals do not change intraday.
    """

    keys = [
        "marketCap",
        "sector",
        "industry",
        "trailingPE",
        "forwardPE",
        "priceToBook",
        "returnOnEquity",
        "returnOnAssets",
        "debtToEquity",
        "currentRatio",
        "profitMargins",
        "operatingMargins",
        "revenueGrowth",
        "earningsGrowth",
        "freeCashflow",
        "operatingCashflow",
        "dividendYield",
        "fiftyTwoWeekHigh",
        "fiftyTwoWeekLow",
    ]

    try:
        stock = yf.Ticker(ticker)
        stock_info = stock.get_info()

        return {
            key: stock_info.get(key)
            for key in keys
        }

    except Exception:
        return {}


# =============================================================================
# TECHNICAL ANALYSIS FUNCTIONS
# =============================================================================

def resample_prices(data, timeframe):
    """Converts daily OHLCV prices into weekly or monthly candles."""

    if data.empty:
        return data

    if timeframe == "Daily":
        return data.copy()

    if timeframe == "Weekly":
        rule = "W-FRI"
    else:
        rule = "ME"

    return (
        data.resample(rule)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .dropna()
    )


def get_pivots(values, pivot_type="high", order=3):
    """
    Finds simple local high or low pivot points.

    This is rule-based analysis, not machine-learning pattern recognition.
    """

    values = np.asarray(values, dtype=float)

    if len(values) < (2 * order) + 1:
        return []

    output = []

    for index in range(order, len(values) - order):
        window = values[
            index - order:index + order + 1
        ]

        if pivot_type == "high":
            if values[index] >= np.max(window):
                output.append(index)

        else:
            if values[index] <= np.min(window):
                output.append(index)

    return output


def make_signal(
    pattern,
    status,
    data,
    level,
    direction,
    notes,
):
    """Builds a standard pattern record."""

    current_price = float(data["close"].iloc[-1])

    average_volume = (
        data["volume"]
        .tail(21)
        .iloc[:-1]
        .mean()
    )

    latest_volume = float(data["volume"].iloc[-1])

    volume_change = (
        (latest_volume / max(average_volume, 1)) - 1
    ) * 100

    return {
        "Pattern": pattern,
        "Status": status,
        "Direction": direction,
        "Date": data.index[-1],
        "Level": float(level),
        "Current": current_price,
        "Return %": (
            (current_price / float(level)) - 1
        ) * 100,
        "Volume %": volume_change,
        "Notes": notes,
    }


def detect_patterns(data):
    """
    Detects simplified active price structures.

    Confirmed means price has closed through a defined breakout /
    breakdown level. In progress means the geometry is visible but
    the confirmation level has not been crossed.
    """

    if data is None or len(data) < 40:
        return []

    data = data.copy().dropna()

    if len(data) < 40:
        return []

    high = data["high"].to_numpy(dtype=float)
    low = data["low"].to_numpy(dtype=float)
    close = data["close"].to_numpy(dtype=float)
    open_price = data["open"].to_numpy(dtype=float)

    results = []

    pivot_highs = get_pivots(high, "high")
    pivot_lows = get_pivots(low, "low")

    # -------------------------------------------------------------------------
    # DOUBLE TOP
    # -------------------------------------------------------------------------

    if len(pivot_highs) >= 2:
        first_top, second_top = pivot_highs[-2:]

        top_similarity = abs(
            high[first_top] - high[second_top]
        ) / max(high[first_top], 1)

        if second_top - first_top >= 8 and top_similarity < 0.045:
            neckline = float(
                np.min(low[first_top:second_top + 1])
            )

            status = (
                "Confirmed"
                if close[-1] < neckline * 0.99
                else "In progress"
            )

            results.append(
                make_signal(
                    "Double Top",
                    status,
                    data,
                    neckline,
                    "Bearish",
                    "Two comparable highs. Confirmation requires a close below the neckline.",
                )
            )

    # -------------------------------------------------------------------------
    # DOUBLE BOTTOM
    # -------------------------------------------------------------------------

    if len(pivot_lows) >= 2:
        first_bottom, second_bottom = pivot_lows[-2:]

        bottom_similarity = abs(
            low[first_bottom] - low[second_bottom]
        ) / max(low[first_bottom], 1)

        if second_bottom - first_bottom >= 8 and bottom_similarity < 0.045:
            neckline = float(
                np.max(high[first_bottom:second_bottom + 1])
            )

            status = (
                "Confirmed"
                if close[-1] > neckline * 1.01
                else "In progress"
            )

            results.append(
                make_signal(
                    "Double Bottom",
                    status,
                    data,
                    neckline,
                    "Bullish",
                    "Two comparable lows. Confirmation requires a close above the neckline.",
                )
            )

    # -------------------------------------------------------------------------
    # HEAD AND SHOULDERS
    # -------------------------------------------------------------------------

    if len(pivot_highs) >= 3:
        left_shoulder, head, right_shoulder = pivot_highs[-3:]

        shoulder_average = (
            high[left_shoulder] +
            high[right_shoulder]
        ) / 2

        left_trough = np.min(
            low[left_shoulder:head + 1]
        )

        right_trough = np.min(
            low[head:right_shoulder + 1]
        )

        neckline = float(
            min(left_trough, right_trough)
        )

        shoulders_similar = abs(
            high[left_shoulder] -
            high[right_shoulder]
        ) / max(shoulder_average, 1)

        if (
            high[head] > shoulder_average * 1.04
            and shoulders_similar < 0.10
        ):
            status = (
                "Confirmed"
                if close[-1] < neckline * 0.99
                else "In progress"
            )

            results.append(
                make_signal(
                    "Head & Shoulders",
                    status,
                    data,
                    neckline,
                    "Bearish",
                    "Confirmation requires a close below neckline support.",
                )
            )

    # -------------------------------------------------------------------------
    # INVERSE HEAD AND SHOULDERS
    # -------------------------------------------------------------------------

    if len(pivot_lows) >= 3:
        left_shoulder, head, right_shoulder = pivot_lows[-3:]

        shoulder_average = (
            low[left_shoulder] +
            low[right_shoulder]
        ) / 2

        left_peak = np.max(
            high[left_shoulder:head + 1]
        )

        right_peak = np.max(
            high[head:right_shoulder + 1]
        )

        neckline = float(
            max(left_peak, right_peak)
        )

        shoulders_similar = abs(
            low[left_shoulder] -
            low[right_shoulder]
        ) / max(shoulder_average, 1)

        if (
            low[head] < shoulder_average * 0.96
            and shoulders_similar < 0.10
        ):
            status = (
                "Confirmed"
                if close[-1] > neckline * 1.01
                else "In progress"
            )

            results.append(
                make_signal(
                    "Inverse Head & Shoulders",
                    status,
                    data,
                    neckline,
                    "Bullish",
                    "Confirmation requires a close above neckline resistance.",
                )
            )

    # -------------------------------------------------------------------------
    # RECTANGLES AND TRIANGLES
    # -------------------------------------------------------------------------

    window_size = 30

    if len(data) >= window_size:
        recent_high = high[-window_size:]
        recent_low = low[-window_size:]

        x = np.arange(window_size)

        high_slope = (
            np.polyfit(x, recent_high, 1)[0]
            / max(np.mean(recent_high), 1)
        )

        low_slope = (
            np.polyfit(x, recent_low, 1)[0]
            / max(np.mean(recent_low), 1)
        )

        resistance = float(np.max(recent_high))
        support = float(np.min(recent_low))

        channel_width = (
            resistance - support
        ) / max(resistance, 1)

        if channel_width < 0.15:
            pattern_name = None

            if abs(high_slope) < 0.001 and abs(low_slope) < 0.001:
                pattern_name = "Rectangle"

            elif abs(high_slope) < 0.0007 and low_slope > 0.0007:
                pattern_name = "Ascending Triangle"

            elif high_slope < -0.0007 and abs(low_slope) < 0.0007:
                pattern_name = "Descending Triangle"

            elif high_slope < -0.0007 and low_slope > 0.0007:
                pattern_name = "Symmetrical Triangle"

            if pattern_name:
                if close[-1] > resistance * 1.01:
                    status = "Confirmed"
                    direction = "Bullish"
                    level = resistance

                elif close[-1] < support * 0.99:
                    status = "Confirmed"
                    direction = "Bearish"
                    level = support

                else:
                    status = "In progress"
                    direction = "Neutral"

                    if pattern_name == "Descending Triangle":
                        level = support
                    else:
                        level = resistance

                results.append(
                    make_signal(
                        pattern_name,
                        status,
                        data,
                        level,
                        direction,
                        "Breakout confirmation requires a close outside the pattern range.",
                    )
                )

    # -------------------------------------------------------------------------
    # REVERSAL CANDLE CANDIDATES
    # -------------------------------------------------------------------------

    latest_open = open_price[-1]
    latest_close = close[-1]
    latest_high = high[-1]
    latest_low = low[-1]

    body = abs(latest_close - latest_open)
    candle_range = max(latest_high - latest_low, 0.000001)

    lower_shadow = (
        min(latest_open, latest_close) -
        latest_low
    )

    upper_shadow = (
        latest_high -
        max(latest_open, latest_close)
    )

    prior_low = np.min(low[-11:-1])
    prior_high = np.max(high[-11:-1])

    if (
        lower_shadow / candle_range > 0.55
        and body / candle_range < 0.35
        and latest_close <= prior_low * 1.04
    ):
        results.append(
            make_signal(
                "Reversal Bottom",
                "Candidate",
                data,
                latest_low,
                "Bullish",
                "Hammer-like candle near local low. Await next-candle confirmation.",
            )
        )

    if (
        upper_shadow / candle_range > 0.55
        and body / candle_range < 0.35
        and latest_close >= prior_high * 0.96
    ):
        results.append(
            make_signal(
                "Reversal Top",
                "Candidate",
                data,
                latest_high,
                "Bearish",
                "Shooting-star-like candle near local high. Await next-candle confirmation.",
            )
        )

    return results


def calculate_support_resistance(data):
    """Returns nearest simple pivot-based support and resistance."""

    if len(data) < 30:
        return (
            float(data["low"].tail(10).min()),
            float(data["high"].tail(10).max()),
        )

    current_price = float(data["close"].iloc[-1])

    pivot_lows = get_pivots(
        data["low"].to_numpy(),
        "low",
    )

    pivot_highs = get_pivots(
        data["high"].to_numpy(),
        "high",
    )

    supports = [
        float(data["low"].iloc[index])
        for index in pivot_lows
        if data["low"].iloc[index] < current_price
    ]

    resistances = [
        float(data["high"].iloc[index])
        for index in pivot_highs
        if data["high"].iloc[index] > current_price
    ]

    support = (
        max(supports[-8:])
        if supports
        else float(data["low"].tail(20).min())
    )

    resistance = (
        min(resistances[-8:])
        if resistances
        else float(data["high"].tail(20).max())
    )

    return support, resistance


def calculate_trend(data):
    """Returns a simple moving-average trend classification."""

    if len(data) < 55:
        return "Insufficient data"

    close = float(data["close"].iloc[-1])
    ma20 = float(data["close"].rolling(20).mean().iloc[-1])
    ma50 = float(data["close"].rolling(50).mean().iloc[-1])

    if len(data) >= 200:
        ma200 = float(
            data["close"]
            .rolling(200)
            .mean()
            .iloc[-1]
        )

        if close > ma20 > ma50 > ma200:
            return "Strong bullish"

    if close > ma20 > ma50:
        return "Bullish"

    if close < ma20 < ma50:
        return "Bearish"

    return "Neutral / consolidating"


# =============================================================================
# FUNDAMENTAL FUNCTIONS
# =============================================================================

def calculate_fundamental_score(fundamentals):
    """
    Produces a simple, transparent 0-9 fundamental score.

    This score is not suitable for banks/NBFCs without sector-specific
    metrics such as NIM, CASA, GNPA, NNPA and capital adequacy.
    """

    score = 0
    strengths = []

    roe = fundamentals.get("returnOnEquity")
    debt_equity = fundamentals.get("debtToEquity")
    profit_margin = fundamentals.get("profitMargins")
    revenue_growth = fundamentals.get("revenueGrowth")
    free_cash_flow = fundamentals.get("freeCashflow")

    if isinstance(roe, (int, float)):
        if roe >= 0.20:
            score += 2
            strengths.append("ROE is at least 20%")

        elif roe >= 0.15:
            score += 1
            strengths.append("ROE is at least 15%")

    if isinstance(debt_equity, (int, float)):
        if debt_equity < 50:
            score += 2
            strengths.append("Low debt/equity")

        elif debt_equity < 100:
            score += 1
            strengths.append("Moderate debt/equity")

    if isinstance(profit_margin, (int, float)):
        if profit_margin >= 0.15:
            score += 2
            strengths.append("Profit margin is at least 15%")

        elif profit_margin >= 0.08:
            score += 1
            strengths.append("Profit margin is at least 8%")

    if isinstance(revenue_growth, (int, float)):
        if revenue_growth >= 0.12:
            score += 1
            strengths.append("Revenue growth is at least 12%")

    if isinstance(free_cash_flow, (int, float)):
        if free_cash_flow > 0:
            score += 2
            strengths.append("Positive free cash flow")

    if score >= 7:
        label = "Excellent"

    elif score >= 5:
        label = "Good"

    elif score >= 3:
        label = "Average"

    else:
        label = "Needs review"

    return score, label, strengths


def format_number(value, percentage=False):
    """Formats fundamental values safely."""

    if not isinstance(value, (int, float)):
        return "—"

    if pd.isna(value):
        return "—"

    if percentage:
        return f"{value * 100:.1f}%"

    return f"{value:.2f}"

# =============================================================================
# ADVANCED FUNDAMENTAL MOMENTUM ENGINE
# =============================================================================

def safe_number(value, default=None):
    """Safely convert a value to float."""
    try:
        if value is None:
            return default

        if pd.isna(value):
            return default

        return float(value)

    except Exception:
        return default


def percentage_change(current_value, previous_value):
    """
    Calculates percentage change safely.

    Example:
    current = 120
    previous = 100
    result = 20.0
    """

    current_value = safe_number(current_value)
    previous_value = safe_number(previous_value)

    if current_value is None or previous_value is None:
        return None

    if previous_value == 0:
        return None

    return (
        (current_value - previous_value)
        / abs(previous_value)
    ) * 100


def classify_growth(growth_rate):
    """Classifies a growth rate."""

    growth_rate = safe_number(growth_rate)

    if growth_rate is None:
        return "Unavailable"

    if growth_rate >= 20:
        return "Strong"

    if growth_rate >= 10:
        return "Healthy"

    if growth_rate >= 3:
        return "Moderate"

    if growth_rate >= -3:
        return "Flat"

    if growth_rate >= -10:
        return "Weak"

    return "Declining"


def classify_margin_change(change_in_bps):
    """
    Classifies a margin movement in basis points.

    100 basis points = 1 percentage point.
    """

    change_in_bps = safe_number(change_in_bps)

    if change_in_bps is None:
        return "Unavailable"

    if change_in_bps >= 150:
        return "Strong expansion"

    if change_in_bps >= 50:
        return "Expansion"

    if change_in_bps > -50:
        return "Stable"

    if change_in_bps > -150:
        return "Compression"

    return "Sharp compression"


def calculate_cagr(start_value, end_value, years):
    """Calculates CAGR safely."""

    start_value = safe_number(start_value)
    end_value = safe_number(end_value)
    years = safe_number(years)

    if (
        start_value is None
        or end_value is None
        or years is None
        or start_value <= 0
        or end_value <= 0
        or years <= 0
    ):
        return None

    return (
        (end_value / start_value) ** (1 / years) - 1
    ) * 100


def assess_annual_fundamentals(annual_data):
    """
    Scores annual fundamentals.

    Expected annual_data example:

    {
        "revenue_cagr_3y": 14.5,
        "profit_cagr_3y": 18.2,
        "eps_cagr_3y": 17.5,
        "roe": 21.0,
        "roce": 24.0,
        "operating_margin": 18.0,
        "margin_trend_bps": 120,
        "debt_to_equity": 0.25,
        "interest_coverage": 8.5,
        "free_cash_flow_positive": True,
        "cash_flow_conversion": 92.0
    }
    """

    score = 0
    positives = []
    concerns = []

    revenue_cagr = safe_number(
        annual_data.get("revenue_cagr_3y")
    )

    profit_cagr = safe_number(
        annual_data.get("profit_cagr_3y")
    )

    eps_cagr = safe_number(
        annual_data.get("eps_cagr_3y")
    )

    roe = safe_number(
        annual_data.get("roe")
    )

    roce = safe_number(
        annual_data.get("roce")
    )

    operating_margin = safe_number(
        annual_data.get("operating_margin")
    )

    margin_trend_bps = safe_number(
        annual_data.get("margin_trend_bps")
    )

    debt_to_equity = safe_number(
        annual_data.get("debt_to_equity")
    )

    interest_coverage = safe_number(
        annual_data.get("interest_coverage")
    )

    free_cash_flow_positive = annual_data.get(
        "free_cash_flow_positive"
    )

    cash_flow_conversion = safe_number(
        annual_data.get("cash_flow_conversion")
    )

    # Revenue CAGR score
    if revenue_cagr is not None:
        if revenue_cagr >= 15:
            score += 2
            positives.append(
                f"Strong 3-year revenue CAGR: {revenue_cagr:.1f}%"
            )

        elif revenue_cagr >= 8:
            score += 1
            positives.append(
                f"Healthy 3-year revenue CAGR: {revenue_cagr:.1f}%"
            )

        elif revenue_cagr < 0:
            concerns.append(
                f"Negative 3-year revenue CAGR: {revenue_cagr:.1f}%"
            )

    # Profit CAGR score
    if profit_cagr is not None:
        if profit_cagr >= 15:
            score += 2
            positives.append(
                f"Strong 3-year profit CAGR: {profit_cagr:.1f}%"
            )

        elif profit_cagr >= 8:
            score += 1
            positives.append(
                f"Healthy 3-year profit CAGR: {profit_cagr:.1f}%"
            )

        elif profit_cagr < 0:
            concerns.append(
                f"Negative 3-year profit CAGR: {profit_cagr:.1f}%"
            )

    # EPS CAGR score
    if eps_cagr is not None:
        if eps_cagr >= 15:
            score += 1
            positives.append(
                f"Strong 3-year EPS CAGR: {eps_cagr:.1f}%"
            )

        elif eps_cagr < 0:
            concerns.append(
                f"Negative 3-year EPS CAGR: {eps_cagr:.1f}%"
            )

    # ROE score
    if roe is not None:
        if roe >= 20:
            score += 2
            positives.append(
                f"Excellent ROE: {roe:.1f}%"
            )

        elif roe >= 15:
            score += 1
            positives.append(
                f"Good ROE: {roe:.1f}%"
            )

        elif roe < 10:
            concerns.append(
                f"Low ROE: {roe:.1f}%"
            )

    # ROCE score
    if roce is not None:
        if roce >= 20:
            score += 2
            positives.append(
                f"Excellent ROCE: {roce:.1f}%"
            )

        elif roce >= 15:
            score += 1
            positives.append(
                f"Good ROCE: {roce:.1f}%"
            )

        elif roce < 10:
            concerns.append(
                f"Low ROCE: {roce:.1f}%"
            )

    # Margin quality
    if operating_margin is not None:
        if operating_margin >= 15:
            score += 1
            positives.append(
                f"Healthy operating margin: {operating_margin:.1f}%"
            )

        elif operating_margin < 5:
            concerns.append(
                f"Low operating margin: {operating_margin:.1f}%"
            )

    # Margin trend
    if margin_trend_bps is not None:
        if margin_trend_bps >= 100:
            score += 1
            positives.append(
                f"Annual margin expansion: {margin_trend_bps:.0f} bps"
            )

        elif margin_trend_bps <= -100:
            concerns.append(
                f"Annual margin contraction: {margin_trend_bps:.0f} bps"
            )

    # Debt and interest coverage
    if debt_to_equity is not None:
        if debt_to_equity <= 0.5:
            score += 1
            positives.append(
                f"Low debt/equity: {debt_to_equity:.2f}"
            )

        elif debt_to_equity >= 2:
            concerns.append(
                f"High debt/equity: {debt_to_equity:.2f}"
            )

    if interest_coverage is not None:
        if interest_coverage >= 5:
            score += 1
            positives.append(
                f"Healthy interest coverage: {interest_coverage:.1f}x"
            )

        elif interest_coverage < 2:
            concerns.append(
                f"Weak interest coverage: {interest_coverage:.1f}x"
            )

    # Cash flow
    if free_cash_flow_positive is True:
        score += 1
        positives.append(
            "Positive free cash flow"
        )

    elif free_cash_flow_positive is False:
        concerns.append(
            "Negative free cash flow"
        )

    if cash_flow_conversion is not None:
        if cash_flow_conversion >= 80:
            score += 1
            positives.append(
                f"Good cash conversion: {cash_flow_conversion:.1f}%"
            )

        elif cash_flow_conversion < 50:
            concerns.append(
                f"Weak cash conversion: {cash_flow_conversion:.1f}%"
            )

    # Final annual classification
    if score >= 11:
        rating = "Strong"

    elif score >= 8:
        rating = "Strong / Improving"

    elif score >= 5:
        rating = "Stable"

    elif score >= 3:
        rating = "Mixed"

    else:
        rating = "Weak"

    return {
        "score": score,
        "rating": rating,
        "positives": positives,
        "concerns": concerns,
    }


def assess_quarterly_fundamentals(quarterly_data):
    """
    Scores quarterly growth and operating momentum.

    Expected quarterly_data example:

    {
        "revenue_yoy": 16.0,
        "profit_yoy": 22.0,
        "eps_yoy": 21.0,
        "ebitda_yoy": 18.0,
        "margin_change_bps_yoy": 120,
        "revenue_qoq": 4.0,
        "profit_qoq": 8.0,
        "operating_kpi_trend": "Improving",
        "management_guidance": "Positive"
    }
    """

    score = 0
    positives = []
    concerns = []

    revenue_yoy = safe_number(
        quarterly_data.get("revenue_yoy")
    )

    profit_yoy = safe_number(
        quarterly_data.get("profit_yoy")
    )

    eps_yoy = safe_number(
        quarterly_data.get("eps_yoy")
    )

    ebitda_yoy = safe_number(
        quarterly_data.get("ebitda_yoy")
    )

    margin_change_bps_yoy = safe_number(
        quarterly_data.get("margin_change_bps_yoy")
    )

    revenue_qoq = safe_number(
        quarterly_data.get("revenue_qoq")
    )

    profit_qoq = safe_number(
        quarterly_data.get("profit_qoq")
    )

    operating_kpi_trend = quarterly_data.get(
        "operating_kpi_trend"
    )

    management_guidance = quarterly_data.get(
        "management_guidance"
    )

    # Revenue momentum
    if revenue_yoy is not None:
        if revenue_yoy >= 15:
            score += 2
            positives.append(
                f"Strong revenue growth: {revenue_yoy:.1f}% YoY"
            )

        elif revenue_yoy >= 8:
            score += 1
            positives.append(
                f"Healthy revenue growth: {revenue_yoy:.1f}% YoY"
            )

        elif revenue_yoy < 0:
            concerns.append(
                f"Revenue decline: {revenue_yoy:.1f}% YoY"
            )

    # Profit momentum
    if profit_yoy is not None:
        if profit_yoy >= 20:
            score += 2
            positives.append(
                f"Strong profit growth: {profit_yoy:.1f}% YoY"
            )

        elif profit_yoy >= 10:
            score += 1
            positives.append(
                f"Healthy profit growth: {profit_yoy:.1f}% YoY"
            )

        elif profit_yoy < 0:
            concerns.append(
                f"Profit decline: {profit_yoy:.1f}% YoY"
            )

    # EPS momentum
    if eps_yoy is not None:
        if eps_yoy >= 15:
            score += 1
            positives.append(
                f"Strong EPS growth: {eps_yoy:.1f}% YoY"
            )

        elif eps_yoy < 0:
            concerns.append(
                f"EPS decline: {eps_yoy:.1f}% YoY"
            )

    # EBITDA momentum
    if ebitda_yoy is not None:
        if ebitda_yoy >= 15:
            score += 1
            positives.append(
                f"Strong EBITDA growth: {ebitda_yoy:.1f}% YoY"
            )

        elif ebitda_yoy < 0:
            concerns.append(
                f"EBITDA decline: {ebitda_yoy:.1f}% YoY"
            )

    # Margin movement
    if margin_change_bps_yoy is not None:
        if margin_change_bps_yoy >= 100:
            score += 2
            positives.append(
                f"Margin expanded by {margin_change_bps_yoy:.0f} bps YoY"
            )

        elif margin_change_bps_yoy >= 25:
            score += 1
            positives.append(
                f"Margin improved by {margin_change_bps_yoy:.0f} bps YoY"
            )

        elif margin_change_bps_yoy <= -100:
            concerns.append(
                f"Margin contracted by {margin_change_bps_yoy:.0f} bps YoY"
            )

    # QoQ direction
    if revenue_qoq is not None and profit_qoq is not None:
        if revenue_qoq > 0 and profit_qoq > 0:
            score += 1
            positives.append(
                "Sequential revenue and profit momentum is positive"
            )

        elif revenue_qoq < 0 and profit_qoq < 0:
            concerns.append(
                "Sequential revenue and profit momentum is negative"
            )

    # Sector KPI trend
    if operating_kpi_trend == "Improving":
        score += 2
        positives.append(
            "Sector-specific operating KPIs are improving"
        )

    elif operating_kpi_trend == "Stable":
        score += 1
        positives.append(
            "Sector-specific operating KPIs are stable"
        )

    elif operating_kpi_trend == "Deteriorating":
        concerns.append(
            "Sector-specific operating KPIs are deteriorating"
        )

    # Guidance
    if management_guidance == "Positive":
        score += 1
        positives.append(
            "Management guidance is positive"
        )

    elif management_guidance == "Negative":
        concerns.append(
            "Management guidance is negative"
        )

    # Final quarterly classification
    if score >= 9:
        rating = "Strong / Strengthening"

    elif score >= 6:
        rating = "Strong"

    elif score >= 4:
        rating = "Early Recovery"

    elif score >= 2:
        rating = "Mixed"

    else:
        rating = "Weak / Deteriorating"

    return {
        "score": score,
        "rating": rating,
        "positives": positives,
        "concerns": concerns,
    }


def calculate_combined_fundamental_status(
    annual_assessment,
    quarterly_assessment,
):
    """Combines long-term quality with recent financial momentum."""

    annual_rating = annual_assessment.get(
        "rating",
        "Weak",
    )

    quarterly_rating = quarterly_assessment.get(
        "rating",
        "Weak / Deteriorating",
    )

    annual_score = annual_assessment.get(
        "score",
        0,
    )

    quarterly_score = quarterly_assessment.get(
        "score",
        0,
    )

    combined_score = (
        annual_score * 0.60
        + quarterly_score * 0.40
    )

    if (
        annual_rating in [
            "Strong",
            "Strong / Improving",
        ]
        and quarterly_rating == "Strong / Strengthening"
    ):
        combined_rating = "Strong / Accelerating"

    elif (
        annual_rating in [
            "Strong",
            "Strong / Improving",
        ]
        and quarterly_rating == "Strong"
    ):
        combined_rating = "Strong"

    elif (
        annual_rating in [
            "Strong",
            "Strong / Improving",
        ]
        and quarterly_rating == "Early Recovery"
    ):
        combined_rating = "Strong / Recovering"

    elif (
        annual_rating == "Stable"
        and quarterly_rating == "Strong / Strengthening"
    ):
        combined_rating = "Improving / Accelerating"

    elif quarterly_rating == "Early Recovery":
        combined_rating = "Early Recovery"

    elif (
        annual_rating == "Weak"
        and quarterly_rating == "Weak / Deteriorating"
    ):
        combined_rating = "Weak / Deteriorating"

    elif combined_score >= 7:
        combined_rating = "Strong"

    elif combined_score >= 4:
        combined_rating = "Mixed"

    else:
        combined_rating = "Weak"

    return {
        "score": round(combined_score, 2),
        "rating": combined_rating,
    }


def determine_fundamental_transition(
    annual_rating,
    quarterly_rating,
):
    """Returns a readable fundamental transition label."""

    if (
        annual_rating == "Weak"
        and quarterly_rating == "Early Recovery"
    ):
        return "Weak → Early Recovery"

    if (
        annual_rating in [
            "Strong",
            "Strong / Improving",
        ]
        and quarterly_rating == "Early Recovery"
    ):
        return "Strong base → Early Recovery"

    if (
        annual_rating in [
            "Strong",
            "Strong / Improving",
        ]
        and quarterly_rating == "Strong / Strengthening"
    ):
        return "Early Recovery → Strong / Accelerating"

    if (
        annual_rating in [
            "Strong",
            "Strong / Improving",
        ]
        and quarterly_rating == "Strong"
    ):
        return "Strong → Strong / Stable"

    if quarterly_rating == "Weak / Deteriorating":
        return "Momentum deterioration"

    if quarterly_rating == "Mixed":
        return "Mixed / transition unclear"

    return "Stable / under observation"

# =============================================================================
# SECTOR-SPECIFIC FUNDAMENTAL KPI DEFINITIONS
# =============================================================================

SECTOR_KPI_LIBRARY = {
    "Banks / NBFCs": [
        "NIM",
        "GNPA",
        "NNPA",
        "Credit Cost",
        "CASA Ratio",
        "Loan Growth",
        "Deposit Growth",
        "Capital Adequacy Ratio",
        "ROA",
        "ROE",
        "Cost to Income Ratio",
        "Provision Coverage Ratio",
    ],

    "Insurance": [
        "VNB",
        "VNB Growth",
        "VNB Margin",
        "APE Growth",
        "Persistency Ratio",
        "Solvency Ratio",
        "Combined Ratio",
        "Embedded Value Growth",
    ],

    "EPC / Capital Goods / Defence": [
        "Order Book",
        "Order Inflow",
        "Order Book Growth",
        "Book to Bill Ratio",
        "Execution Rate",
        "Revenue Growth",
        "EBITDA Margin",
        "Working Capital Days",
    ],

    "IT Services": [
        "Constant Currency Growth",
        "Deal Wins",
        "TCV",
        "Attrition",
        "Utilization",
        "EBIT Margin",
        "Digital Revenue Mix",
        "Revenue per Employee",
    ],

    "Auto": [
        "Volume Growth",
        "Domestic Volume Growth",
        "Export Volume Growth",
        "Realization Growth",
        "EBITDA per Unit",
        "Market Share",
        "EV Mix",
        "Premium Segment Mix",
    ],

    "Cement": [
        "Volume Growth",
        "Realization per Tonne",
        "EBITDA per Tonne",
        "Capacity Utilization",
        "Capacity Addition",
        "Fuel Cost per Tonne",
    ],

    "Metals / Mining": [
        "Production Growth",
        "Sales Volume Growth",
        "Realization per Tonne",
        "Cost per Tonne",
        "EBITDA per Tonne",
        "Capacity Utilization",
        "Commodity Price Trend",
    ],

    "Telecom": [
        "ARPU",
        "Subscriber Growth",
        "Churn Rate",
        "Data Usage",
        "4G / 5G Subscriber Additions",
        "Capex",
        "EBITDA Margin",
        "Net Debt",
    ],

    "Real Estate": [
        "Pre-sales",
        "Booking Value",
        "Collections",
        "New Launches",
        "Net Debt",
        "Inventory",
        "Unsold Inventory",
        "Cash Flow from Operations",
    ],

    "Oil & Gas / Refining": [
        "GRM",
        "Throughput",
        "Crude Production",
        "Gas Production",
        "Realization",
        "Refining Margin",
        "Reserve Replacement Ratio",
        "Sales Volume Growth",
    ],

    "Retail / FMCG": [
        "Same Store Sales Growth",
        "Volume Growth",
        "Value Growth",
        "Gross Margin",
        "Store Additions",
        "Store Productivity",
        "Private Label Mix",
        "Distribution Expansion",
    ],

    "Pharma": [
        "US Sales Growth",
        "India Sales Growth",
        "ANDA Filings",
        "ANDA Approvals",
        "R&D as Percentage of Sales",
        "Product Concentration",
        "Complex Generics Mix",
        "API Revenue Growth",
    ],
}


# =============================================================================
# CHART FUNCTION
# =============================================================================

def create_chart(data, signals, title):
    """Creates a candlestick chart with moving averages and signal levels."""

    chart_data = data.tail(260).copy()

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.76, 0.24],
    )

    figure.add_trace(
        go.Candlestick(
            x=chart_data.index,
            open=chart_data["open"],
            high=chart_data["high"],
            low=chart_data["low"],
            close=chart_data["close"],
            name="Price",
        ),
        row=1,
        col=1,
    )

    figure.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data["close"].rolling(20).mean(),
            name="20 MA",
            line=dict(color="#2563eb", width=1.5),
        ),
        row=1,
        col=1,
    )

    figure.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data["close"].rolling(50).mean(),
            name="50 MA",
            line=dict(color="#f59e0b", width=1.5),
        ),
        row=1,
        col=1,
    )

    figure.add_trace(
        go.Bar(
            x=chart_data.index,
            y=chart_data["volume"],
            name="Volume",
            marker_color="#94a3b8",
        ),
        row=2,
        col=1,
    )

    for signal in signals:
        color = "#16a34a"

        if signal["Direction"] == "Bearish":
            color = "#dc2626"

        elif signal["Direction"] == "Neutral":
            color = "#f59e0b"

        figure.add_hline(
            y=signal["Level"],
            line_dash="dot",
            line_color=color,
            annotation_text=signal["Pattern"],
            row=1,
            col=1,
        )

    figure.update_layout(
        title=title,
        height=620,
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        legend_orientation="h",
        margin=dict(l=10, r=10, t=50, b=10),
    )

    return figure


# =============================================================================
# LOAD UNIVERSE ONCE
# =============================================================================

with st.spinner("Loading Nifty Total Market constituents..."):
    universe, universe_error = get_nifty_total_market_members()


# =============================================================================
# PAGE HEADER
# =============================================================================

st.markdown(
    '<div class="main-title">📊 Nifty Total Market (750) Research Dashboard</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-title">'
    'Daily, weekly and monthly technical research • '
    'fundamentals • support/resistance • pattern scanning'
    '</div>',
    unsafe_allow_html=True,
)

if universe_error:
    st.warning(universe_error)

if len(universe) < 100:
    st.error(
        "The live Nifty Total Market list is unavailable. "
        "The app is operating with the fallback stock universe."
    )


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.header("🔍 Research controls")

    mode = st.radio(
        "Mode",
        [
            "Stock research",
            "Pattern scanner",
        ],
    )

    minimum_market_cap = st.number_input(
        "Minimum market cap (₹ crore)",
        min_value=0,
        value=3000,
        step=1000,
    )

    st.caption(
        f"Loaded universe: {len(universe)} stocks"
    )

    if st.button("🔄 Refresh cached data"):
        st.cache_data.clear()
        st.rerun()

    st.divider()

    st.caption(
        "Price cache: 15 minutes\n\n"
        "Fundamental cache: 12 hours\n\n"
        "Universe cache: 6 hours"
    )


# =============================================================================
# STOCK RESEARCH MODE
# =============================================================================

if mode == "Stock research":
    symbols = sorted(universe["Symbol"].tolist())

    default_index = (
        symbols.index("RELIANCE")
        if "RELIANCE" in symbols
        else 0
    )

    selected_symbol = st.selectbox(
        "Search a Nifty Total Market stock",
        symbols,
        index=default_index,
    )

    selected_record = universe[
        universe["Symbol"] == selected_symbol
    ].iloc[0]

    ticker = selected_record["Ticker"]
    company_name = selected_record["Company Name"]
    industry = selected_record["Industry"]

    with st.spinner(f"Fetching latest data for {selected_symbol}..."):
        daily_data = fetch_price_data(ticker, "5y")
        fundamental_data = fetch_fundamentals(ticker)

    if daily_data.empty:
        st.error(
            "No price data was returned for this stock. "
            "Try again later or click Refresh cached data."
        )
        st.stop()

    market_cap_crore = (
        fundamental_data.get("marketCap") or 0
    ) / 10000000

    score, fundamental_label, strengths = (
        calculate_fundamental_score(fundamental_data)
    )

    current_price = float(daily_data["close"].iloc[-1])

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Last close",
        f"₹{current_price:,.2f}",
    )

    col2.metric(
        "Market cap",
        (
            f"₹{market_cap_crore:,.0f} Cr"
            if market_cap_crore > 0
            else "Unavailable"
        ),
    )

    col3.metric(
        "Daily trend",
        calculate_trend(daily_data),
    )

    col4.metric(
        "Fundamental score",
        f"{score}/9 · {fundamental_label}",
    )

    st.caption(
        f"{company_name} • {industry}"
    )

    if (
        market_cap_crore > 0
        and market_cap_crore < minimum_market_cap
    ):
        st.warning(
            f"{selected_symbol} is below your current minimum market-cap "
            f"filter of ₹{minimum_market_cap:,.0f} crore."
        )

    daily_tab, weekly_tab, monthly_tab, fundamental_tab = st.tabs(
        [
            "Daily",
            "Weekly",
            "Monthly",
            "Fundamentals",
        ]
    )

    for tab, timeframe in zip(
        [daily_tab, weekly_tab, monthly_tab],
        TIMEFRAMES,
    ):
        with tab:
            timeframe_data = resample_prices(
                daily_data,
                timeframe,
            )

            detected_signals = detect_patterns(
                timeframe_data
            )

            support, resistance = calculate_support_resistance(
                timeframe_data
            )

            metric1, metric2, metric3 = st.columns(3)

            metric1.metric(
                "Overall trend",
                calculate_trend(timeframe_data),
            )

            metric2.metric(
                "Nearest support",
                f"₹{support:,.2f}",
            )

            metric3.metric(
                "Nearest resistance",
                f"₹{resistance:,.2f}",
            )

            st.subheader(
                f"{timeframe} pattern status"
            )

            if detected_signals:
                signal_table = pd.DataFrame(
                    detected_signals
                )

                st.dataframe(
                    signal_table,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Date": st.column_config.DatetimeColumn(
                            "Date",
                            format="YYYY-MM-DD",
                        ),
                        "Level": st.column_config.NumberColumn(
                            "Breakout / Neckline",
                            format="₹%.2f",
                        ),
                        "Current": st.column_config.NumberColumn(
                            "Current price",
                            format="₹%.2f",
                        ),
                        "Return %": st.column_config.NumberColumn(
                            "Return since level",
                            format="%.2f%%",
                        ),
                        "Volume %": st.column_config.NumberColumn(
                            "Volume vs average",
                            format="%.1f%%",
                        ),
                    },
                )

            else:
                st.info(
                    "No supported active or confirmed pattern is currently "
                    "detected on this timeframe."
                )

            st.plotly_chart(
                create_chart(
                    timeframe_data,
                    detected_signals,
                    f"{selected_symbol} — {timeframe}",
                ),
                use_container_width=True,
            )

    with fundamental_tab:
        st.subheader(
            f"Fundamental health: {fundamental_label} ({score}/9)"
        )

        if strengths:
            st.success(
                "Positive signals: " +
                " • ".join(strengths)
            )

        else:
            st.info(
                "There is insufficient comparable Yahoo Finance data "
                "to identify strong fundamental signals."
            )

        free_cash_flow = fundamental_data.get(
            "freeCashflow"
        )

        if isinstance(free_cash_flow, (int, float)):
            free_cash_flow_value = (
                f"₹{free_cash_flow / 10000000:,.0f} Cr"
            )
        else:
            free_cash_flow_value = "—"

        fundamentals_table = pd.DataFrame(
            [
                ["Sector", fundamental_data.get("sector", "—")],
                ["Industry", fundamental_data.get("industry", "—")],
                ["Trailing P/E", format_number(fundamental_data.get("trailingPE"))],
                ["Forward P/E", format_number(fundamental_data.get("forwardPE"))],
                ["Price / Book", format_number(fundamental_data.get("priceToBook"))],
                ["ROE", format_number(fundamental_data.get("returnOnEquity"), True)],
                ["ROA", format_number(fundamental_data.get("returnOnAssets"), True)],
                ["Profit margin", format_number(fundamental_data.get("profitMargins"), True)],
                ["Operating margin", format_number(fundamental_data.get("operatingMargins"), True)],
                ["Revenue growth", format_number(fundamental_data.get("revenueGrowth"), True)],
                ["Earnings growth", format_number(fundamental_data.get("earningsGrowth"), True)],
                ["Debt / Equity", format_number(fundamental_data.get("debtToEquity"))],
                ["Current ratio", format_number(fundamental_data.get("currentRatio"))],
                ["Free cash flow", free_cash_flow_value],
                ["Dividend yield", format_number(fundamental_data.get("dividendYield"), True)],
            ],
            columns=[
                "Metric",
                "Value",
            ],
        )
st.divider()

st.subheader("📈 Annual and Quarterly Fundamental Momentum")

st.caption(
    "This section is designed to combine long-term business quality "
    "with recent quarterly momentum. Values not available from the "
    "current data source will display as unavailable until a dedicated "
    "NSE/XBRL or company-results ingestion pipeline is connected."
)

# -------------------------------------------------------------------------
# TEMPORARY DATA INPUT STRUCTURE
# -------------------------------------------------------------------------
#
# For now, these are placeholders that can be populated by:
#
# 1. NSE XBRL financial results
# 2. Company quarterly presentation extraction
# 3. Manual data entry
# 4. Supabase/PostgreSQL database
# 5. A licensed Indian market-data provider
#
# The scoring engine is ready now; the remaining task is data ingestion.
# -------------------------------------------------------------------------

annual_financial_data = {
    "revenue_cagr_3y": None,
    "profit_cagr_3y": None,
    "eps_cagr_3y": None,
    "roe": None,
    "roce": None,
    "operating_margin": None,
    "margin_trend_bps": None,
    "debt_to_equity": None,
    "interest_coverage": None,
    "free_cash_flow_positive": None,
    "cash_flow_conversion": None,
}

quarterly_financial_data = {
    "revenue_yoy": None,
    "profit_yoy": None,
    "eps_yoy": None,
    "ebitda_yoy": None,
    "margin_change_bps_yoy": None,
    "revenue_qoq": None,
    "profit_qoq": None,
    "operating_kpi_trend": None,
    "management_guidance": None,
}

annual_assessment = assess_annual_fundamentals(
    annual_financial_data
)

quarterly_assessment = assess_quarterly_fundamentals(
    quarterly_financial_data
)

combined_assessment = calculate_combined_fundamental_status(
    annual_assessment,
    quarterly_assessment,
)

transition = determine_fundamental_transition(
    annual_assessment["rating"],
    quarterly_assessment["rating"],
)

fundamental_col1, fundamental_col2 = st.columns(2)

with fundamental_col1:
    st.markdown(
        f"""
        <div class="card">
            <h4>Annual Fundamental Status</h4>
            <h2>{annual_assessment["rating"]}</h2>
            <p>Score: {annual_assessment["score"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with fundamental_col2:
    st.markdown(
        f"""
        <div class="card">
            <h4>Quarterly Fundamental Status</h4>
            <h2>{quarterly_assessment["rating"]}</h2>
            <p>Score: {quarterly_assessment["score"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

combined_col1, combined_col2 = st.columns(2)

with combined_col1:
    st.markdown(
        f"""
        <div class="card success-card">
            <h4>Combined Fundamental Status</h4>
            <h2>{combined_assessment["rating"]}</h2>
            <p>Combined score: {combined_assessment["score"]}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

with combined_col2:
    st.markdown(
        f"""
        <div class="card warning-card">
            <h4>Fundamental Transition</h4>
            <h2>{transition}</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("### Annual positives")

if annual_assessment["positives"]:
    for item in annual_assessment["positives"]:
        st.success(f"✅ {item}")
else:
    st.info(
        "Annual financial history has not yet been loaded "
        "from NSE/XBRL or another financial-results source."
    )

st.markdown("### Annual concerns")

if annual_assessment["concerns"]:
    for item in annual_assessment["concerns"]:
        st.warning(f"⚠️ {item}")
else:
    st.caption(
        "No annual risk flags are currently available."
    )

st.markdown("### Quarterly positives")

if quarterly_assessment["positives"]:
    for item in quarterly_assessment["positives"]:
        st.success(f"✅ {item}")
else:
    st.info(
        "Quarterly financial and operating KPI data "
        "has not yet been loaded."
    )

st.markdown("### Quarterly concerns")

if quarterly_assessment["concerns"]:
    for item in quarterly_assessment["concerns"]:
        st.warning(f"⚠️ {item}")
else:
    st.caption(
        "No quarterly risk flags are currently available."
    )
        st.dataframe(
            fundamentals_table,
            hide_index=True,
            use_container_width=True,
        )

        st.info(
            "For banks and NBFCs, Debt/Equity and free-cash-flow comparisons "
            "are less meaningful. Evaluate CASA, NIM, GNPA, NNPA, provision "
            "coverage, capital adequacy, loan growth and deposit growth."
        )


# =============================================================================
# PATTERN SCANNER MODE
# =============================================================================

else:
    st.subheader("Nifty Total Market pattern scanner")

    st.caption(
        "Scan stocks by technical pattern, timeframe, overall trend, "
        "pattern status, and minimum market capitalization."
    )

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        selected_pattern = st.selectbox(
            "Pattern",
            PATTERN_OPTIONS,
        )

        selected_timeframe = st.selectbox(
            "Timeframe",
            [
                "Any",
                "Daily",
                "Weekly",
                "Monthly",
            ],
        )

    with filter_col2:
        selected_overall_trend = st.selectbox(
            "Overall trend",
            [
                "Any",
                "Strong bullish",
                "Bullish",
                "Neutral / consolidating",
                "Bearish",
                "Insufficient data",
            ],
        )

        selected_status = st.selectbox(
            "Pattern status",
            [
                "Any",
                "Confirmed",
                "In progress",
                "Candidate",
            ],
        )

    st.info(
        "How the filters work: when Timeframe is set to Any, the scanner "
        "checks Daily, Weekly, and Monthly charts. A stock is returned for "
        "every timeframe where it matches your selected pattern, trend, "
        "status, and market-cap criteria."
    )

    st.warning(
        "A first full Nifty Total Market scan can take several minutes on "
        "free Streamlit hosting. Selecting Any timeframe requires three "
        "technical scans per stock: Daily, Weekly, and Monthly. "
        "Price data is cached for 15 minutes."
    )

    if st.button(
        "🔎 Scan current Nifty Total Market universe",
        type="primary",
    ):
        rows = []

        # "Any" means scan all available chart timeframes.
        if selected_timeframe == "Any":
            timeframes_to_scan = [
                "Daily",
                "Weekly",
                "Monthly",
            ]
        else:
            timeframes_to_scan = [
                selected_timeframe
            ]

        progress = st.progress(0)
        progress_status = st.empty()

        total_stocks = len(universe)

        for index, record in universe.iterrows():
            symbol = record["Symbol"]
            ticker = record["Ticker"]

            progress_status.caption(
                f"Scanning {index + 1:,} of {total_stocks:,}: "
                f"{symbol}"
            )

            try:
                # Download once per stock, then resample the same data into
                # Daily / Weekly / Monthly chart structures.
                stock_data = fetch_price_data(
                    ticker,
                    "5y",
                )

                if stock_data.empty:
                    continue

                stock_matches = []

                for timeframe in timeframes_to_scan:
                    timeframe_data = resample_prices(
                        stock_data,
                        timeframe,
                    )

                    if timeframe_data.empty:
                        continue

                    # Trend is calculated separately for each selected chart.
                    # Example: a stock can be Strong bullish on Daily but
                    # Neutral / consolidating on Monthly.
                    overall_trend = calculate_trend(
                        timeframe_data
                    )

                    # Apply the trend filter before processing pattern rows.
                    trend_matches = (
                        selected_overall_trend == "Any"
                        or overall_trend == selected_overall_trend
                    )

                    if not trend_matches:
                        continue

                    detected_signals = detect_patterns(
                        timeframe_data
                    )

                    for signal in detected_signals:
                        pattern_matches = (
                            selected_pattern == "Any"
                            or signal["Pattern"] == selected_pattern
                        )

                        status_matches = (
                            selected_status == "Any"
                            or signal["Status"] == selected_status
                        )

                        if pattern_matches and status_matches:
                            stock_matches.append(
                                {
                                    "Timeframe": timeframe,
                                    "Overall Trend": overall_trend,
                                    **signal,
                                }
                            )

                # Do not fetch fundamentals/market cap unless the stock has
                # already passed the technical-pattern and trend filters.
                if stock_matches:
                    stock_fundamentals = fetch_fundamentals(
                        ticker
                    )

                    market_cap_crore = (
                        stock_fundamentals.get("marketCap") or 0
                    ) / 10000000

                    # Strict market-cap rule:
                    # A missing market-cap value is treated as not eligible.
                    if market_cap_crore < minimum_market_cap:
                        continue

                    for signal in stock_matches:
                        rows.append(
                            {
                                "Stock": symbol,
                                "Company": record["Company Name"],
                                "Industry": record["Industry"],
                                "Market Cap (Cr)": round(
                                    market_cap_crore,
                                    0,
                                ),
                                **signal,
                            }
                        )

            except Exception:
                # A single unavailable Yahoo Finance symbol should not
                # terminate the complete Nifty Total Market scan.
                pass

            progress.progress(
                min(
                    (index + 1) / total_stocks,
                    1.0,
                )
            )

        progress.empty()
        progress_status.empty()

        st.subheader(
            f"Matching signals: {len(rows)}"
        )

        if rows:
            results = pd.DataFrame(rows)

            # Sort confirmed signals first, followed by strongest return.
            status_priority = {
                "Confirmed": 1,
                "In progress": 2,
                "Candidate": 3,
            }

            results["Status Priority"] = (
                results["Status"]
                .map(status_priority)
                .fillna(99)
            )

            results = results.sort_values(
                by=[
                    "Status Priority",
                    "Overall Trend",
                    "Return %",
                ],
                ascending=[
                    True,
                    True,
                    False,
                ],
            )

            results = results.drop(
                columns=["Status Priority"]
            )

            summary_col1, summary_col2, summary_col3, summary_col4 = (
                st.columns(4)
            )

            summary_col1.metric(
                "Total matches",
                len(results),
            )

            summary_col2.metric(
                "Confirmed",
                int(
                    (
                        results["Status"] == "Confirmed"
                    ).sum()
                ),
            )

            summary_col3.metric(
                "Strong bullish",
                int(
                    (
                        results["Overall Trend"]
                        == "Strong bullish"
                    ).sum()
                ),
            )

            summary_col4.metric(
                "Bullish direction",
                int(
                    (
                        results["Direction"]
                        == "Bullish"
                    ).sum()
                ),
            )

            st.dataframe(
                results,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Date": st.column_config.DatetimeColumn(
                        "Signal date",
                        format="YYYY-MM-DD",
                    ),
                    "Level": st.column_config.NumberColumn(
                        "Breakout / Neckline",
                        format="₹%.2f",
                    ),
                    "Current": st.column_config.NumberColumn(
                        "Current price",
                        format="₹%.2f",
                    ),
                    "Return %": st.column_config.NumberColumn(
                        "Return since level",
                        format="%.2f%%",
                    ),
                    "Volume %": st.column_config.NumberColumn(
                        "Volume vs average",
                        format="%.1f%%",
                    ),
                    "Market Cap (Cr)": st.column_config.NumberColumn(
                        "Market cap",
                        format="₹%d Cr",
                    ),
                },
            )

            csv_data = results.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Download scanner results as CSV",
                data=csv_data,
                file_name="nifty_total_market_pattern_scan.csv",
                mime="text/csv",
            )

        else:
            st.info(
                "No stocks matched your selected Pattern, Timeframe, "
                "Overall Trend, Pattern Status, and Market Cap filters. "
                "Try setting one or more filters to Any."
            )

# =============================================================================
# FOOTER
# =============================================================================

st.divider()

st.caption(
    "Data sources: Nifty Indices constituent CSV and Yahoo Finance. "
    "Pattern detection is rule-based and may generate false positives or "
    "miss valid patterns. Verify data, corporate actions, financial results, "
    "prices and investment suitability independently. "
    "This dashboard is for research and education only, not investment advice."
)
