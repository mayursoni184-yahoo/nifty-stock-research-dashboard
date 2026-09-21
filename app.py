import io
import warnings
from datetime import datetime, time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

from plotly.subplots import make_subplots


warnings.filterwarnings("ignore")


# =============================================================================
# STREAMLIT CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Nifty Total Market Research Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# STYLE
# =============================================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.35rem;
            font-weight: 800;
            color: #111827;
            margin-bottom: 0.12rem;
        }

        .sub-title {
            font-size: 1rem;
            color: #6b7280;
            margin-bottom: 1.15rem;
        }

        .research-card {
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 16px;
            background-color: #ffffff;
            margin-bottom: 12px;
        }

        .green-card {
            border-left: 6px solid #16a34a;
        }

        .blue-card {
            border-left: 6px solid #2563eb;
        }

        .orange-card {
            border-left: 6px solid #f59e0b;
        }

        .red-card {
            border-left: 6px solid #dc2626;
        }

        .gray-card {
            border-left: 6px solid #6b7280;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =============================================================================
# CONSTANTS
# =============================================================================

IST = ZoneInfo("Asia/Kolkata")

NSE_MARKET_OPEN = time(9, 15)
NSE_MARKET_CLOSE = time(15, 30)

NIFTY_TOTAL_MARKET_CSV = (
    "https://niftyindices.com/IndexConstituent/"
    "ind_niftytotalmarket_list.csv"
)

NIFTY_50_BENCHMARK = "^NSEI"

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

DASHBOARD_MODES = [
    "Daily Market Command Center",
    "Stock Research",
    "Winner Ranking Scanner",
    "Historical Filter Study",
     "Gold & Silver Decision Hub",
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

TREND_OPTIONS = [
    "Any",
    "Strong bullish",
    "Bullish",
    "Neutral / consolidating",
    "Bearish",
    "Insufficient data",
]

RS_STATUS_OPTIONS = [
    "Any",
    "Leader",
    "Strong",
    "Neutral",
    "Weak",
    "Insufficient data",
]

MTF_ALIGNMENT_OPTIONS = [
    "Any",
    "Strong multi-timeframe alignment",
    "Bullish multi-timeframe alignment",
    "Mixed timeframe alignment",
    "Bearish multi-timeframe alignment",
    "Insufficient data",
]

WINNER_CATEGORY_OPTIONS = [
    "Any",
    "Elite Candidate",
    "High-Conviction Watchlist",
    "Watchlist",
    "Neutral / Mixed",
    "Avoid / Weak",
]

RETURN_WINDOWS = {
    "5d": 5,
    "10d": 10,
    "20d": 20,
    "30d": 30,
    "60d": 60,
    "90d": 90,
    "6m": 126,
    "9m": 189,
    "12m": 252,
}

GOLD_ETF_TICKER = "GOLDBEES.NS"
SILVER_ETF_TICKER = "SILVERBEES.NS"

GOLD_FUTURES_TICKER = "GC=F"
SILVER_FUTURES_TICKER = "SI=F"

USD_INDEX_TICKER = "DX-Y.NYB"  # ICE U.S. Dollar Index

# =============================================================================
# SESSION STATE
# =============================================================================

if "selected_symbol" not in st.session_state:
    st.session_state["selected_symbol"] = "RELIANCE"

if "command_center_results" not in st.session_state:
    st.session_state["command_center_results"] = pd.DataFrame()

if "winner_scanner_results" not in st.session_state:
    st.session_state["winner_scanner_results"] = pd.DataFrame()

if "historical_filter_results" not in st.session_state:
    st.session_state["historical_filter_results"] = pd.DataFrame()

if "historical_filter_settings" not in st.session_state:
    st.session_state["historical_filter_settings"] = {}

if "live_filter_results" not in st.session_state:
    st.session_state["live_filter_results"] = pd.DataFrame()

if "live_filter_metadata" not in st.session_state:
    st.session_state["live_filter_metadata"] = {}


# =============================================================================
# GENERAL UTILITY FUNCTIONS
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


def format_price(value):
    """Format a value as Indian rupees."""

    value = safe_number(value)

    if value is None:
        return "Not available"

    return f"₹{value:,.2f}"


def format_percent(value):
    """Format signed percentage."""

    value = safe_number(value)

    if value is None:
        return "Not available"

    return f"{value:+.2f}%"


def format_market_cap(value):
    """Format market cap in crore rupees."""

    value = safe_number(value)

    if value is None or value <= 0:
        return "Not available"

    return f"₹{value / 10000000:,.0f} Cr"


def open_stock_research(symbol):
    """Save selected stock and request Stock Research navigation."""

    st.session_state["selected_symbol"] = symbol
    st.session_state["requested_mode"] = "Stock Research"


def get_ist_now():
    """Return current timestamp in Asia/Kolkata."""

    return datetime.now(IST)


def get_market_status():
    """
    Return a basic NSE market status.

    This checks weekday and regular session time only.
    It does not automatically verify special sessions or exchange holidays.
    """

    now_ist = get_ist_now()

    weekday = now_ist.weekday()
    current_time = now_ist.time()

    if weekday >= 5:
        return {
            "status": "CLOSED",
            "is_open": False,
            "now": now_ist,
        }

    if (
        NSE_MARKET_OPEN
        <= current_time
        <= NSE_MARKET_CLOSE
    ):
        return {
            "status": "OPEN",
            "is_open": True,
            "now": now_ist,
        }

    return {
        "status": "CLOSED",
        "is_open": False,
        "now": now_ist,
    }


def next_business_day_estimate(current_date):
    """
    Estimate next NSE trading day using pandas BusinessDay.

    This handles weekends but not NSE-specific holidays.
    A fully accurate version should use the official NSE holiday calendar.
    """

    return (
        pd.Timestamp(current_date)
        + pd.tseries.offsets.BusinessDay(1)
    ).date()


# =============================================================================
# NIFTY TOTAL MARKET UNIVERSE
# =============================================================================

@st.cache_data(ttl=21600, show_spinner=False)
def get_nifty_total_market_members():
    """
    Load current Nifty Total Market members.

    Use fallback symbols when official CSV cannot be reached.
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

        members = pd.read_csv(
            io.BytesIO(response.content)
        )

        members.columns = [
            str(column).strip()
            for column in members.columns
        ]

        if "Symbol" not in members.columns:
            raise ValueError(
                "Constituent file does not contain Symbol."
            )

        if "Series" in members.columns:
            members = members[
                members["Series"]
                .astype(str)
                .str.upper()
                .eq("EQ")
            ].copy()

        members["Symbol"] = (
            members["Symbol"]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        members["Ticker"] = (
            members["Symbol"] + ".NS"
        )

        if "Company Name" not in members.columns:
            members["Company Name"] = members["Symbol"]

        if "Industry" not in members.columns:
            members["Industry"] = "Unknown"

        members = members[
            [
                "Company Name",
                "Industry",
                "Symbol",
                "Ticker",
            ]
        ]

        members = (
            members
            .drop_duplicates("Symbol")
            .reset_index(drop=True)
        )

        if len(members) < 100:
            raise ValueError(
                "Too few valid Nifty Total Market members loaded."
            )

        return members, None

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

        warning = (
            "Could not load live Nifty Total Market members. "
            f"Using fallback stocks. Reason: {type(error).__name__}: {error}"
        )

        return fallback, warning


# =============================================================================
# PRICE AND FUNDAMENTAL DATA
# =============================================================================

@st.cache_data(ttl=900, show_spinner=False)
def fetch_price_data(ticker, period="5y"):
    """
    Fetch daily OHLCV data.

    This data is used for all historical studies and end-of-day
    technical calculations.
    """

    try:
        ticker_object = yf.Ticker(ticker)

        data = ticker_object.history(
            period=period,
            auto_adjust=True,
        )

        if data is None or data.empty:
            return pd.DataFrame()

        data = data.rename(
            columns=str.lower
        )

        data = data[
            [
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        ].dropna()

        data.index = pd.to_datetime(
            data.index
        )

        if getattr(data.index, "tz", None) is not None:
            data.index = data.index.tz_localize(None)

        return data

    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60, show_spinner=False)
def fetch_intraday_data(ticker):
    """
    Fetch current intraday data.

    5-minute interval is used rather than 1-minute to reduce rate-limit
    risk on free Yahoo Finance / Streamlit Cloud access.
    """

    try:
        ticker_object = yf.Ticker(ticker)

        intraday_data = ticker_object.history(
            period="1d",
            interval="5m",
            auto_adjust=False,
            prepost=False,
        )

        if intraday_data is None or intraday_data.empty:
            return pd.DataFrame()

        intraday_data = intraday_data.rename(
            columns=str.lower
        )

        intraday_data = intraday_data[
            [
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        ].dropna()

        intraday_data.index = pd.to_datetime(
            intraday_data.index
        )

        return intraday_data

    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=43200, show_spinner=False)
def fetch_fundamentals(ticker):
    """Fetch current summary fundamental fields."""

    fields = [
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
        "trailingEps",
        "fiftyTwoWeekHigh",
        "fiftyTwoWeekLow",
    ]

    try:
        ticker_object = yf.Ticker(ticker)
        info = ticker_object.get_info()

        return {
            field: info.get(field)
            for field in fields
        }

    except Exception:
        return {}


# =============================================================================
# TECHNICAL INDICATORS
# =============================================================================

def calculate_rsi(close_series, period=14):
    """Calculate rolling RSI."""

    delta = close_series.diff()

    gains = delta.clip(lower=0)

    losses = -delta.clip(upper=0)

    average_gain = gains.rolling(
        period
    ).mean()

    average_loss = losses.rolling(
        period
    ).mean()

    relative_strength = (
        average_gain
        / average_loss.replace(0, np.nan)
    )

    return 100 - (
        100 / (1 + relative_strength)
    )


def calculate_atr_series(data, period=14):
    """Calculate ATR series."""

    previous_close = data["close"].shift(1)

    true_range = pd.concat(
        [
            data["high"] - data["low"],
            (data["high"] - previous_close).abs(),
            (data["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    return true_range.rolling(period).mean()


def calculate_atr(data, period=14):
    """Return latest ATR."""

    if data is None or len(data) < period + 1:
        return None

    atr_value = calculate_atr_series(
        data,
        period,
    ).iloc[-1]

    if pd.isna(atr_value):
        return None

    return float(atr_value)


def resample_ohlcv(data, timeframe):
    """Resample Daily OHLCV data into Weekly or Monthly candles."""

    if data.empty:
        return data

    if timeframe == "Daily":
        return data.copy()

    if timeframe == "Weekly":
        frequency = "W-FRI"
    else:
        frequency = "ME"

    return (
        data
        .resample(frequency)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .dropna()
    )


def calculate_overall_trend(data):
    """Calculate moving-average trend."""

    if data is None or len(data) < 55:
        return "Insufficient data"

    close = float(
        data["close"].iloc[-1]
    )

    sma20 = float(
        data["close"]
        .rolling(20)
        .mean()
        .iloc[-1]
    )

    sma50 = float(
        data["close"]
        .rolling(50)
        .mean()
        .iloc[-1]
    )

    if len(data) >= 200:
        sma200 = float(
            data["close"]
            .rolling(200)
            .mean()
            .iloc[-1]
        )

        if close > sma20 > sma50 > sma200:
            return "Strong bullish"

    if close > sma20 > sma50:
        return "Bullish"

    if close < sma20 < sma50:
        return "Bearish"

    return "Neutral / consolidating"


def calculate_multitimeframe_alignment(
    daily_data,
    weekly_data,
    monthly_data,
):
    """Calculate alignment score across timeframes."""

    daily_trend = calculate_overall_trend(
        daily_data
    )

    weekly_trend = calculate_overall_trend(
        weekly_data
    )

    monthly_trend = calculate_overall_trend(
        monthly_data
    )

    daily_points = {
        "Strong bullish": 3,
        "Bullish": 2,
        "Neutral / consolidating": 1,
        "Bearish": 0,
        "Insufficient data": 0,
    }

    weekly_points = {
        "Strong bullish": 4,
        "Bullish": 3,
        "Neutral / consolidating": 1,
        "Bearish": 0,
        "Insufficient data": 0,
    }

    monthly_points = {
        "Strong bullish": 3,
        "Bullish": 2,
        "Neutral / consolidating": 1,
        "Bearish": 0,
        "Insufficient data": 0,
    }

    score = (
        daily_points.get(daily_trend, 0)
        + weekly_points.get(weekly_trend, 0)
        + monthly_points.get(monthly_trend, 0)
    )

    bullish_count = sum(
        trend in [
            "Strong bullish",
            "Bullish",
        ]
        for trend in [
            daily_trend,
            weekly_trend,
            monthly_trend,
        ]
    )

    bearish_count = sum(
        trend == "Bearish"
        for trend in [
            daily_trend,
            weekly_trend,
            monthly_trend,
        ]
    )

    if (
        daily_trend == "Strong bullish"
        and weekly_trend == "Strong bullish"
        and monthly_trend in [
            "Strong bullish",
            "Bullish",
        ]
    ):
        status = "Strong multi-timeframe alignment"

    elif bullish_count >= 2:
        status = "Bullish multi-timeframe alignment"

    elif bearish_count >= 2:
        status = "Bearish multi-timeframe alignment"

    else:
        status = "Mixed timeframe alignment"

    return {
        "daily": daily_trend,
        "weekly": weekly_trend,
        "monthly": monthly_trend,
        "score": score,
        "status": status,
    }


# =============================================================================
# RELATIVE STRENGTH
# =============================================================================

def align_stock_benchmark(stock_data, benchmark_data):
    """Align stock and benchmark close series."""

    if stock_data.empty or benchmark_data.empty:
        return pd.DataFrame()

    stock_close = stock_data[
        ["close"]
    ].rename(
        columns={
            "close": "stock_close",
        }
    )

    benchmark_close = benchmark_data[
        ["close"]
    ].rename(
        columns={
            "close": "benchmark_close",
        }
    )

    return stock_close.join(
        benchmark_close,
        how="inner",
    ).dropna()


def calculate_period_return(close_series, days):
    """Calculate return for selected trading days."""

    if len(close_series) <= days:
        return None

    current_close = float(
        close_series.iloc[-1]
    )

    old_close = float(
        close_series.iloc[-days - 1]
    )

    if old_close == 0:
        return None

    return (
        current_close / old_close - 1
    ) * 100


def calculate_relative_strength(
    stock_data,
    benchmark_data,
):
    """Calculate relative strength metrics versus Nifty 50."""

    aligned = align_stock_benchmark(
        stock_data,
        benchmark_data,
    )

    empty_result = {
        "status": "Insufficient data",
        "rs_trend": "Unavailable",
        "rs_line": pd.Series(dtype=float),
        "stock_1m": None,
        "stock_3m": None,
        "stock_6m": None,
        "stock_12m": None,
        "benchmark_1m": None,
        "benchmark_3m": None,
        "benchmark_6m": None,
        "benchmark_12m": None,
        "relative_1m": None,
        "relative_3m": None,
        "relative_6m": None,
        "relative_12m": None,
    }

    if len(aligned) < 30:
        return empty_result

    stock_close = aligned["stock_close"]
    benchmark_close = aligned["benchmark_close"]

    stock_1m = calculate_period_return(
        stock_close,
        21,
    )

    stock_3m = calculate_period_return(
        stock_close,
        63,
    )

    stock_6m = calculate_period_return(
        stock_close,
        126,
    )

    stock_12m = calculate_period_return(
        stock_close,
        252,
    )

    benchmark_1m = calculate_period_return(
        benchmark_close,
        21,
    )

    benchmark_3m = calculate_period_return(
        benchmark_close,
        63,
    )

    benchmark_6m = calculate_period_return(
        benchmark_close,
        126,
    )

    benchmark_12m = calculate_period_return(
        benchmark_close,
        252,
    )

    relative_1m = (
        stock_1m - benchmark_1m
        if stock_1m is not None
        and benchmark_1m is not None
        else None
    )

    relative_3m = (
        stock_3m - benchmark_3m
        if stock_3m is not None
        and benchmark_3m is not None
        else None
    )

    relative_6m = (
        stock_6m - benchmark_6m
        if stock_6m is not None
        and benchmark_6m is not None
        else None
    )

    relative_12m = (
        stock_12m - benchmark_12m
        if stock_12m is not None
        and benchmark_12m is not None
        else None
    )

    rs_line = stock_close / benchmark_close

    rs_trend = "Unavailable"

    if len(rs_line) >= 63:
        latest_rs = float(
            rs_line.iloc[-1]
        )

        old_rs = float(
            rs_line.iloc[-63]
        )

        if latest_rs > old_rs * 1.03:
            rs_trend = "Rising"

        elif latest_rs < old_rs * 0.97:
            rs_trend = "Falling"

        else:
            rs_trend = "Flat"

    available_returns = [
        value
        for value in [
            relative_1m,
            relative_3m,
            relative_6m,
            relative_12m,
        ]
        if value is not None
    ]

    if len(available_returns) < 2:
        status = "Insufficient data"

    else:
        positive_count = sum(
            value > 0
            for value in available_returns
        )

        strong_count = sum(
            value > 5
            for value in available_returns
        )

        negative_count = sum(
            value < 0
            for value in available_returns
        )

        if (
            positive_count >= 3
            and strong_count >= 2
            and rs_trend == "Rising"
        ):
            status = "Leader"

        elif (
            positive_count >= 2
            and rs_trend in [
                "Rising",
                "Flat",
            ]
        ):
            status = "Strong"

        elif negative_count >= 3:
            status = "Weak"

        else:
            status = "Neutral"

    return {
        "status": status,
        "rs_trend": rs_trend,
        "rs_line": rs_line,
        "stock_1m": stock_1m,
        "stock_3m": stock_3m,
        "stock_6m": stock_6m,
        "stock_12m": stock_12m,
        "benchmark_1m": benchmark_1m,
        "benchmark_3m": benchmark_3m,
        "benchmark_6m": benchmark_6m,
        "benchmark_12m": benchmark_12m,
        "relative_1m": relative_1m,
        "relative_3m": relative_3m,
        "relative_6m": relative_6m,
        "relative_12m": relative_12m,
    }


# =============================================================================
# SUPPORT / RESISTANCE
# =============================================================================

def find_pivots(values, pivot_type="high", order=3):
    """Find local pivot highs or lows."""

    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) < (order * 2) + 1:
        return []

    pivot_indexes = []

    for index in range(order, len(values) - order):
        window = values[
            index - order:index + order + 1
        ]

        if pivot_type == "high":
            if values[index] >= np.max(window):
                pivot_indexes.append(index)

        else:
            if values[index] <= np.min(window):
                pivot_indexes.append(index)

    return pivot_indexes


def calculate_support_resistance(data):
    """
    Calculate approximate recent support and resistance.

    This function is defined before any page rendering so Stock Research
    and Risk/Reward tabs do not produce NameError.
    """

    if data is None or data.empty:
        return None, None

    if len(data) < 30:
        return (
            float(data["low"].tail(10).min()),
            float(data["high"].tail(10).max()),
        )

    current_price = float(
        data["close"].iloc[-1]
    )

    low_values = data["low"].to_numpy(
        dtype=float
    )

    high_values = data["high"].to_numpy(
        dtype=float
    )

    low_pivots = find_pivots(
        low_values,
        pivot_type="low",
        order=3,
    )

    high_pivots = find_pivots(
        high_values,
        pivot_type="high",
        order=3,
    )

    support_candidates = [
        float(data["low"].iloc[index])
        for index in low_pivots
        if data["low"].iloc[index] < current_price
    ]

    resistance_candidates = [
        float(data["high"].iloc[index])
        for index in high_pivots
        if data["high"].iloc[index] > current_price
    ]

    support = (
        max(support_candidates[-8:])
        if support_candidates
        else float(data["low"].tail(20).min())
    )

    resistance = (
        min(resistance_candidates[-8:])
        if resistance_candidates
        else float(data["high"].tail(20).max())
    )

    return support, resistance


# =============================================================================
# PATTERN DETECTION
# =============================================================================

def build_pattern_signal(
    pattern,
    status,
    data,
    level,
    direction,
    notes,
    pattern_height=None,
):
    """Build a standard pattern-signal dictionary."""

    current_price = float(
        data["close"].iloc[-1]
    )

    average_volume = float(
        data["volume"]
        .tail(21)
        .iloc[:-1]
        .mean()
    )

    latest_volume = float(
        data["volume"].iloc[-1]
    )

    volume_change = (
        latest_volume / max(average_volume, 1) - 1
    ) * 100

    return {
        "Pattern": pattern,
        "Status": status,
        "Direction": direction,
        "Date": data.index[-1],
        "Level": float(level),
        "Current": current_price,
        "Return %": (
            current_price / float(level) - 1
        ) * 100,
        "Volume %": volume_change,
        "Pattern Height": pattern_height,
        "Notes": notes,
    }


def detect_patterns(data):
    """Detect current rule-based technical patterns."""

    if data is None or data.empty:
        return []

    data = data.dropna().copy()

    if len(data) < 40:
        return []

    high = data["high"].to_numpy(
        dtype=float
    )

    low = data["low"].to_numpy(
        dtype=float
    )

    close = data["close"].to_numpy(
        dtype=float
    )

    open_price = data["open"].to_numpy(
        dtype=float
    )

    patterns = []

    pivot_highs = find_pivots(
        high,
        "high",
        3,
    )

    pivot_lows = find_pivots(
        low,
        "low",
        3,
    )

    # DOUBLE TOP
    if len(pivot_highs) >= 2:
        first_top = pivot_highs[-2]
        second_top = pivot_highs[-1]

        top_difference = abs(
            high[first_top] - high[second_top]
        ) / max(high[first_top], 1)

        if (
            second_top - first_top >= 8
            and top_difference < 0.045
        ):
            neckline = float(
                np.min(
                    low[first_top:second_top + 1]
                )
            )

            peak = max(
                high[first_top],
                high[second_top],
            )

            status = (
                "Confirmed"
                if close[-1] < neckline * 0.99
                else "In progress"
            )

            patterns.append(
                build_pattern_signal(
                    "Double Top",
                    status,
                    data,
                    neckline,
                    "Bearish",
                    "Two comparable highs; confirmation is below neckline.",
                    peak - neckline,
                )
            )

    # DOUBLE BOTTOM
    if len(pivot_lows) >= 2:
        first_bottom = pivot_lows[-2]
        second_bottom = pivot_lows[-1]

        bottom_difference = abs(
            low[first_bottom] - low[second_bottom]
        ) / max(low[first_bottom], 1)

        if (
            second_bottom - first_bottom >= 8
            and bottom_difference < 0.045
        ):
            neckline = float(
                np.max(
                    high[first_bottom:second_bottom + 1]
                )
            )

            base = min(
                low[first_bottom],
                low[second_bottom],
            )

            status = (
                "Confirmed"
                if close[-1] > neckline * 1.01
                else "In progress"
            )

            patterns.append(
                build_pattern_signal(
                    "Double Bottom",
                    status,
                    data,
                    neckline,
                    "Bullish",
                    "Two comparable lows; confirmation is above neckline.",
                    neckline - base,
                )
            )

    # HEAD AND SHOULDERS
    if len(pivot_highs) >= 3:
        left_shoulder = pivot_highs[-3]
        head = pivot_highs[-2]
        right_shoulder = pivot_highs[-1]

        shoulder_average = (
            high[left_shoulder]
            + high[right_shoulder]
        ) / 2

        left_trough = float(
            np.min(
                low[left_shoulder:head + 1]
            )
        )

        right_trough = float(
            np.min(
                low[head:right_shoulder + 1]
            )
        )

        neckline = min(
            left_trough,
            right_trough,
        )

        shoulder_difference = abs(
            high[left_shoulder]
            - high[right_shoulder]
        ) / max(shoulder_average, 1)

        if (
            high[head] > shoulder_average * 1.04
            and shoulder_difference < 0.10
        ):
            status = (
                "Confirmed"
                if close[-1] < neckline * 0.99
                else "In progress"
            )

            patterns.append(
                build_pattern_signal(
                    "Head & Shoulders",
                    status,
                    data,
                    neckline,
                    "Bearish",
                    "Confirmation is below neckline support.",
                    high[head] - neckline,
                )
            )

    # INVERSE HEAD AND SHOULDERS
    if len(pivot_lows) >= 3:
        left_shoulder = pivot_lows[-3]
        head = pivot_lows[-2]
        right_shoulder = pivot_lows[-1]

        shoulder_average = (
            low[left_shoulder]
            + low[right_shoulder]
        ) / 2

        left_peak = float(
            np.max(
                high[left_shoulder:head + 1]
            )
        )

        right_peak = float(
            np.max(
                high[head:right_shoulder + 1]
            )
        )

        neckline = max(
            left_peak,
            right_peak,
        )

        shoulder_difference = abs(
            low[left_shoulder]
            - low[right_shoulder]
        ) / max(shoulder_average, 1)

        if (
            low[head] < shoulder_average * 0.96
            and shoulder_difference < 0.10
        ):
            status = (
                "Confirmed"
                if close[-1] > neckline * 1.01
                else "In progress"
            )

            patterns.append(
                build_pattern_signal(
                    "Inverse Head & Shoulders",
                    status,
                    data,
                    neckline,
                    "Bullish",
                    "Confirmation is above neckline resistance.",
                    neckline - low[head],
                )
            )

    # RECTANGLE / TRIANGLES
    window_size = 30

    if len(data) >= window_size:
        recent_highs = high[-window_size:]
        recent_lows = low[-window_size:]

        x_values = np.arange(
            window_size
        )

        high_slope = (
            np.polyfit(
                x_values,
                recent_highs,
                1,
            )[0]
            / max(np.mean(recent_highs), 1)
        )

        low_slope = (
            np.polyfit(
                x_values,
                recent_lows,
                1,
            )[0]
            / max(np.mean(recent_lows), 1)
        )

        resistance = float(
            np.max(recent_highs)
        )

        support = float(
            np.min(recent_lows)
        )

        height = resistance - support

        range_size = (
            height / max(resistance, 1)
        )

        if range_size < 0.15:
            pattern_name = None

            if (
                abs(high_slope) < 0.001
                and abs(low_slope) < 0.001
            ):
                pattern_name = "Rectangle"

            elif (
                abs(high_slope) < 0.0007
                and low_slope > 0.0007
            ):
                pattern_name = "Ascending Triangle"

            elif (
                high_slope < -0.0007
                and abs(low_slope) < 0.0007
            ):
                pattern_name = "Descending Triangle"

            elif (
                high_slope < -0.0007
                and low_slope > 0.0007
            ):
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
                    level = (
                        support
                        if pattern_name
                        == "Descending Triangle"
                        else resistance
                    )

                patterns.append(
                    build_pattern_signal(
                        pattern_name,
                        status,
                        data,
                        level,
                        direction,
                        "Confirmation requires close outside pattern range.",
                        height,
                    )
                )

    # REVERSAL BOTTOM / TOP
    latest_open = float(
        open_price[-1]
    )

    latest_close = float(
        close[-1]
    )

    latest_high = float(
        high[-1]
    )

    latest_low = float(
        low[-1]
    )

    body = abs(
        latest_close - latest_open
    )

    candle_range = max(
        latest_high - latest_low,
        0.000001,
    )

    lower_shadow = (
        min(latest_open, latest_close)
        - latest_low
    )

    upper_shadow = (
        latest_high
        - max(latest_open, latest_close)
    )

    previous_low = float(
        np.min(low[-11:-1])
    )

    previous_high = float(
        np.max(high[-11:-1])
    )

    if (
        lower_shadow / candle_range > 0.55
        and body / candle_range < 0.35
        and latest_close <= previous_low * 1.04
    ):
        patterns.append(
            build_pattern_signal(
                "Reversal Bottom",
                "Candidate",
                data,
                latest_low,
                "Bullish",
                "Hammer-like candle near local low.",
                None,
            )
        )

    if (
        upper_shadow / candle_range > 0.55
        and body / candle_range < 0.35
        and latest_close >= previous_high * 0.96
    ):
        patterns.append(
            build_pattern_signal(
                "Reversal Top",
                "Candidate",
                data,
                latest_high,
                "Bearish",
                "Shooting-star-like candle near local high.",
                None,
            )
        )

    return patterns


# =============================================================================
# RISK / REWARD FUNCTIONS
# =============================================================================

def calculate_pattern_target(pattern):
    """Calculate measured-move target."""

    level = safe_number(
        pattern.get("Level")
    )

    height = safe_number(
        pattern.get("Pattern Height")
    )

    direction = pattern.get(
        "Direction"
    )

    if level is None or height is None:
        return None

    if height <= 0:
        return None

    if direction == "Bullish":
        return level + height

    if direction == "Bearish":
        return level - height

    return None


def classify_entry_quality(
    current_price,
    breakout_level,
    direction,
):
    """Classify price extension from pattern level."""

    current_price = safe_number(
        current_price
    )

    breakout_level = safe_number(
        breakout_level
    )

    if current_price is None or breakout_level is None:
        return {
            "status": "Insufficient data",
            "distance_percent": None,
        }

    if breakout_level == 0:
        return {
            "status": "Insufficient data",
            "distance_percent": None,
        }

    if direction == "Bullish":
        distance = (
            (current_price - breakout_level)
            / breakout_level
        ) * 100

    elif direction == "Bearish":
        distance = (
            (breakout_level - current_price)
            / breakout_level
        ) * 100

    else:
        distance = (
            abs(current_price - breakout_level)
            / breakout_level
        ) * 100

    if distance <= 0:
        status = "Below / Near Breakout"
    elif distance <= 3:
        status = "Ideal Entry Zone"
    elif distance <= 7:
        status = "Acceptable Entry Zone"
    elif distance <= 12:
        status = "Extended"
    else:
        status = "Avoid Chasing"

    return {
        "status": status,
        "distance_percent": distance,
    }


def calculate_trade_plan(data, pattern):
    """Build a current rule-based trade plan."""

    if data is None or data.empty:
        return None

    current_price = float(
        data["close"].iloc[-1]
    )

    support, resistance = calculate_support_resistance(
        data
    )

    atr = calculate_atr(
        data,
        14,
    )

    direction = pattern.get(
        "Direction",
        "Neutral",
    )

    breakout_level = safe_number(
        pattern.get("Level")
    )

    if breakout_level is None:
        breakout_level = current_price

    target = calculate_pattern_target(
        pattern
    )

    entry_quality = classify_entry_quality(
        current_price,
        breakout_level,
        direction,
    )

    if direction == "Bullish":
        technical_stop = support

        atr_stop = (
            current_price - 2 * atr
            if atr is not None
            else None
        )

        stops = [
            value
            for value in [
                technical_stop,
                atr_stop,
            ]
            if value is not None
            and value < current_price
        ]

        selected_stop = (
            min(stops)
            if stops
            else None
        )

        if target is None:
            target = resistance

        if (
            target is None
            or target <= current_price
        ):
            if atr is not None:
                target = (
                    current_price
                    + 3 * atr
                )

        risk_per_share = (
            current_price - selected_stop
            if selected_stop is not None
            else None
        )

        reward_per_share = (
            target - current_price
            if target is not None
            else None
        )

    elif direction == "Bearish":
        technical_stop = resistance

        atr_stop = (
            current_price + 2 * atr
            if atr is not None
            else None
        )

        stops = [
            value
            for value in [
                technical_stop,
                atr_stop,
            ]
            if value is not None
            and value > current_price
        ]

        selected_stop = (
            max(stops)
            if stops
            else None
        )

        if target is None:
            target = support

        if (
            target is None
            or target >= current_price
        ):
            if atr is not None:
                target = (
                    current_price
                    - 3 * atr
                )

        risk_per_share = (
            selected_stop - current_price
            if selected_stop is not None
            else None
        )

        reward_per_share = (
            current_price - target
            if target is not None
            else None
        )

    else:
        return {
            "current_price": current_price,
            "direction": "Neutral",
            "breakout_level": breakout_level,
            "support": support,
            "resistance": resistance,
            "atr": atr,
            "entry_quality": entry_quality,
            "selected_stop": None,
            "target": None,
            "risk_per_share": None,
            "reward_per_share": None,
            "risk_reward": None,
        }

    risk_reward = None

    if (
        risk_per_share is not None
        and reward_per_share is not None
        and risk_per_share > 0
        and reward_per_share > 0
    ):
        risk_reward = (
            reward_per_share / risk_per_share
        )

    return {
        "current_price": current_price,
        "direction": direction,
        "breakout_level": breakout_level,
        "support": support,
        "resistance": resistance,
        "atr": atr,
        "entry_quality": entry_quality,
        "selected_stop": selected_stop,
        "target": target,
        "risk_per_share": risk_per_share,
        "reward_per_share": reward_per_share,
        "risk_reward": risk_reward,
    }


def calculate_position_size(
    portfolio_value,
    risk_percent,
    entry_price,
    stop_loss,
):
    """Calculate position size using defined portfolio risk."""

    portfolio_value = safe_number(
        portfolio_value
    )

    risk_percent = safe_number(
        risk_percent
    )

    entry_price = safe_number(
        entry_price
    )

    stop_loss = safe_number(
        stop_loss
    )

    if (
        portfolio_value is None
        or risk_percent is None
        or entry_price is None
        or stop_loss is None
    ):
        return None

    risk_per_share = abs(
        entry_price - stop_loss
    )

    if risk_per_share <= 0:
        return None

    maximum_loss = (
        portfolio_value * risk_percent / 100
    )

    quantity = int(
        maximum_loss / risk_per_share
    )

    position_value = (
        quantity * entry_price
    )

    allocation = (
        position_value / portfolio_value
    ) * 100

    return {
        "maximum_loss": maximum_loss,
        "risk_per_share": risk_per_share,
        "quantity": quantity,
        "position_value": position_value,
        "allocation": allocation,
    }


# =============================================================================
# WINNER SCORE ENGINE
# =============================================================================

def score_relative_strength(rs_data):
    """Score Relative Strength out of 20."""

    score = 0
    positives = []
    risks = []

    status = rs_data.get(
        "status",
        "Insufficient data",
    )

    rs_trend = rs_data.get(
        "rs_trend",
        "Unavailable",
    )

    relative_3m = safe_number(
        rs_data.get("relative_3m")
    )

    relative_6m = safe_number(
        rs_data.get("relative_6m")
    )

    if status == "Leader":
        score += 10
        positives.append(
            "Relative Strength status is Leader"
        )

    elif status == "Strong":
        score += 7
        positives.append(
            "Relative Strength status is Strong"
        )

    elif status == "Neutral":
        score += 3

    elif status == "Weak":
        risks.append(
            "Stock is underperforming Nifty 50"
        )

    if rs_trend == "Rising":
        score += 4
        positives.append(
            "Relative Strength line is rising"
        )

    elif rs_trend == "Falling":
        risks.append(
            "Relative Strength line is falling"
        )

    if relative_3m is not None:
        if relative_3m >= 15:
            score += 3
            positives.append(
                f"Strong 3M RS: {relative_3m:.1f}%"
            )

        elif relative_3m >= 5:
            score += 2

        elif relative_3m < 0:
            risks.append(
                f"Negative 3M RS: {relative_3m:.1f}%"
            )

    if relative_6m is not None:
        if relative_6m >= 20:
            score += 3
            positives.append(
                f"Strong 6M RS: {relative_6m:.1f}%"
            )

        elif relative_6m >= 8:
            score += 2

        elif relative_6m < 0:
            risks.append(
                f"Negative 6M RS: {relative_6m:.1f}%"
            )

    return {
        "score": min(score, 20),
        "positives": positives,
        "risks": risks,
    }


def score_alignment(alignment):
    """Score MTF alignment out of 20."""

    score = 0
    positives = []
    risks = []

    status = alignment.get(
        "status",
        "Insufficient data",
    )

    if status == "Strong multi-timeframe alignment":
        score += 20
        positives.append(
            "Strong multi-timeframe alignment"
        )

    elif status == "Bullish multi-timeframe alignment":
        score += 15
        positives.append(
            "Bullish multi-timeframe alignment"
        )

    elif status == "Mixed timeframe alignment":
        score += 6
        risks.append(
            "Timeframes are not fully aligned"
        )

    elif status == "Bearish multi-timeframe alignment":
        risks.append(
            "Most timeframes are bearish"
        )

    return {
        "score": min(score, 20),
        "positives": positives,
        "risks": risks,
    }


def score_fundamentals(fundamentals):
    """Score currently available basics out of 20."""

    score = 0
    positives = []
    risks = []

    roe = safe_number(
        fundamentals.get("returnOnEquity")
    )

    roa = safe_number(
        fundamentals.get("returnOnAssets")
    )

    profit_margin = safe_number(
        fundamentals.get("profitMargins")
    )

    operating_margin = safe_number(
        fundamentals.get("operatingMargins")
    )

    revenue_growth = safe_number(
        fundamentals.get("revenueGrowth")
    )

    earnings_growth = safe_number(
        fundamentals.get("earningsGrowth")
    )

    debt_equity = safe_number(
        fundamentals.get("debtToEquity")
    )

    free_cashflow = safe_number(
        fundamentals.get("freeCashflow")
    )

    if roe is not None:
        if roe >= 0.20:
            score += 4
            positives.append(
                f"Excellent ROE: {roe * 100:.1f}%"
            )

        elif roe >= 0.15:
            score += 3
            positives.append(
                f"Good ROE: {roe * 100:.1f}%"
            )

        elif roe < 0.08:
            risks.append(
                f"Low ROE: {roe * 100:.1f}%"
            )

    if roa is not None:
        if roa >= 0.10:
            score += 2
            positives.append(
                f"Strong ROA: {roa * 100:.1f}%"
            )

        elif roa >= 0.05:
            score += 1

    if profit_margin is not None:
        if profit_margin >= 0.15:
            score += 3
            positives.append(
                f"Strong profit margin: {profit_margin * 100:.1f}%"
            )

        elif profit_margin >= 0.08:
            score += 2

        elif profit_margin < 0.03:
            risks.append("Low profit margin")

    if operating_margin is not None:
        if operating_margin >= 0.15:
            score += 2
            positives.append(
                f"Healthy operating margin: {operating_margin * 100:.1f}%"
            )

        elif operating_margin >= 0.08:
            score += 1

    if revenue_growth is not None:
        if revenue_growth >= 0.15:
            score += 2
            positives.append(
                f"Strong revenue growth: {revenue_growth * 100:.1f}%"
            )

        elif revenue_growth >= 0.08:
            score += 1

        elif revenue_growth < 0:
            risks.append("Negative revenue growth")

    if earnings_growth is not None:
        if earnings_growth >= 0.20:
            score += 2
            positives.append(
                f"Strong earnings growth: {earnings_growth * 100:.1f}%"
            )

        elif earnings_growth >= 0.10:
            score += 1

        elif earnings_growth < 0:
            risks.append("Negative earnings growth")

    if debt_equity is not None:
        if debt_equity < 50:
            score += 2
            positives.append("Low debt/equity")

        elif debt_equity > 200:
            risks.append("High debt/equity")

    if free_cashflow is not None:
        if free_cashflow > 0:
            score += 2
            positives.append("Positive free cash flow")

        else:
            risks.append("Negative free cash flow")

    return {
        "score": min(score, 20),
        "positives": positives,
        "risks": risks,
    }


def score_technical_trends(
    daily_data,
    weekly_data,
    monthly_data,
):
    """Score trends out of 15."""

    score = 0
    positives = []
    risks = []

    daily = calculate_overall_trend(
        daily_data
    )

    weekly = calculate_overall_trend(
        weekly_data
    )

    monthly = calculate_overall_trend(
        monthly_data
    )

    if monthly == "Strong bullish":
        score += 6
        positives.append(
            "Strong Bullish Monthly trend"
        )

    elif monthly == "Bullish":
        score += 4
        positives.append(
            "Bullish Monthly trend"
        )

    elif monthly == "Bearish":
        risks.append("Bearish Monthly trend")

    if weekly == "Strong bullish":
        score += 5
        positives.append(
            "Strong Bullish Weekly trend"
        )

    elif weekly == "Bullish":
        score += 3
        positives.append(
            "Bullish Weekly trend"
        )

    elif weekly == "Bearish":
        risks.append("Bearish Weekly trend")

    if daily == "Strong bullish":
        score += 4
        positives.append(
            "Strong Bullish Daily trend"
        )

    elif daily == "Bullish":
        score += 2
        positives.append(
            "Bullish Daily trend"
        )

    elif daily == "Bearish":
        risks.append("Bearish Daily trend")

    return {
        "score": min(score, 15),
        "positives": positives,
        "risks": risks,
    }


def score_patterns(
    daily_patterns,
    weekly_patterns,
    monthly_patterns,
):
    """Score current technical patterns out of 15."""

    score = 0
    positives = []
    risks = []

    confirmed_bullish = False

    pattern_groups = [
        (
            "Daily",
            daily_patterns,
        ),
        (
            "Weekly",
            weekly_patterns,
        ),
        (
            "Monthly",
            monthly_patterns,
        ),
    ]

    for timeframe, patterns in pattern_groups:
        for pattern in patterns:
            status = pattern.get("Status")
            direction = pattern.get("Direction")
            name = pattern.get("Pattern")

            if (
                status == "Confirmed"
                and direction == "Bullish"
            ):
                confirmed_bullish = True

                if timeframe == "Monthly":
                    score += 8
                elif timeframe == "Weekly":
                    score += 6
                else:
                    score += 4

                positives.append(
                    f"Confirmed Bullish {name} on {timeframe}"
                )

            elif (
                status == "In progress"
                and direction == "Bullish"
            ):
                if timeframe == "Weekly":
                    score += 2
                elif timeframe == "Daily":
                    score += 1

                positives.append(
                    f"Bullish {name} forming on {timeframe}"
                )

            elif (
                status == "Confirmed"
                and direction == "Bearish"
            ):
                risks.append(
                    f"Confirmed Bearish {name} on {timeframe}"
                )

    if not confirmed_bullish:
        risks.append(
            "No confirmed bullish breakout detected"
        )

    return {
        "score": min(score, 15),
        "positives": positives,
        "risks": risks,
    }


def score_volume(
    daily_patterns,
    weekly_patterns,
):
    """Score breakout volume out of 5."""

    volumes = []

    for pattern in daily_patterns + weekly_patterns:
        if (
            pattern.get("Status") == "Confirmed"
            and pattern.get("Direction") == "Bullish"
        ):
            volume = safe_number(
                pattern.get("Volume %")
            )

            if volume is not None:
                volumes.append(volume)

    if not volumes:
        return {
            "score": 0,
            "positives": [],
            "risks": [],
        }

    highest_volume = max(volumes)

    if highest_volume >= 50:
        return {
            "score": 5,
            "positives": [
                f"Strong breakout volume: {highest_volume:.1f}% above average"
            ],
            "risks": [],
        }

    if highest_volume >= 25:
        return {
            "score": 4,
            "positives": [
                f"Good breakout volume: {highest_volume:.1f}% above average"
            ],
            "risks": [],
        }

    if highest_volume > 0:
        return {
            "score": 2,
            "positives": [
                f"Positive breakout volume: {highest_volume:.1f}% above average"
            ],
            "risks": [],
        }

    return {
        "score": 0,
        "positives": [],
        "risks": [
            "Breakout volume is below average"
        ],
    }


def score_valuation_risk(fundamentals):
    """Score valuation and leverage out of 5."""

    score = 0
    positives = []
    risks = []

    pe = safe_number(
        fundamentals.get("trailingPE")
    )

    pb = safe_number(
        fundamentals.get("priceToBook")
    )

    debt_equity = safe_number(
        fundamentals.get("debtToEquity")
    )

    if pe is not None:
        if 0 < pe <= 20:
            score += 2
            positives.append(
                f"Reasonable P/E: {pe:.1f}"
            )

        elif pe > 80:
            risks.append(
                f"Very high P/E: {pe:.1f}"
            )

    if pb is not None:
        if 0 < pb <= 3:
            score += 1
            positives.append(
                f"Reasonable Price/Book: {pb:.2f}"
            )

        elif pb > 12:
            risks.append(
                f"High Price/Book: {pb:.2f}"
            )

    if debt_equity is not None:
        if debt_equity < 50:
            score += 2
            positives.append(
                "Low financial leverage"
            )

        elif debt_equity > 250:
            risks.append(
                "High financial leverage"
            )

    return {
        "score": min(score, 5),
        "positives": positives,
        "risks": risks,
    }


def calculate_winner_score(
    fundamentals,
    rs_data,
    alignment,
    daily_data,
    weekly_data,
    monthly_data,
    daily_patterns,
    weekly_patterns,
    monthly_patterns,
):
    """Calculate Winner Score out of 100."""

    rs_component = score_relative_strength(
        rs_data
    )

    alignment_component = score_alignment(
        alignment
    )

    fundamental_component = score_fundamentals(
        fundamentals
    )

    technical_component = score_technical_trends(
        daily_data,
        weekly_data,
        monthly_data,
    )

    pattern_component = score_patterns(
        daily_patterns,
        weekly_patterns,
        monthly_patterns,
    )

    volume_component = score_volume(
        daily_patterns,
        weekly_patterns,
    )

    valuation_component = score_valuation_risk(
        fundamentals
    )

    score = (
        rs_component["score"]
        + alignment_component["score"]
        + fundamental_component["score"]
        + technical_component["score"]
        + pattern_component["score"]
        + volume_component["score"]
        + valuation_component["score"]
    )

    score = min(
        round(score, 1),
        100,
    )

    if score >= 80:
        category = "Elite Candidate"
    elif score >= 65:
        category = "High-Conviction Watchlist"
    elif score >= 50:
        category = "Watchlist"
    elif score >= 35:
        category = "Neutral / Mixed"
    else:
        category = "Avoid / Weak"

    positives = (
        rs_component["positives"]
        + alignment_component["positives"]
        + fundamental_component["positives"]
        + technical_component["positives"]
        + pattern_component["positives"]
        + volume_component["positives"]
        + valuation_component["positives"]
    )

    risks = (
        rs_component["risks"]
        + alignment_component["risks"]
        + fundamental_component["risks"]
        + technical_component["risks"]
        + pattern_component["risks"]
        + volume_component["risks"]
        + valuation_component["risks"]
    )

    return {
        "score": score,
        "category": category,
        "relative_strength_score": rs_component["score"],
        "alignment_score": alignment_component["score"],
        "fundamental_score": fundamental_component["score"],
        "technical_score": technical_component["score"],
        "pattern_score": pattern_component["score"],
        "volume_score": volume_component["score"],
        "valuation_score": valuation_component["score"],
        "positives": positives,
        "risks": risks,
    }


# =============================================================================
# HISTORICAL FILTER STUDY WITH PENDING-ENTRY SUPPORT
# =============================================================================

@st.cache_data(ttl=43200, show_spinner=False)
def prepare_historical_filter_indicators(
    ticker,
):
    """
    Prepare vectorized historical indicators.

    This logic is unchanged from the optimized historical study:
    all filters use completed historical daily close/volume values.
    """

    data = fetch_price_data(
        ticker,
        "5y",
    )

    if data.empty:
        return pd.DataFrame()

    data = data.copy()

    data["volume_sma_5"] = (
        data["volume"]
        .rolling(5)
        .mean()
    )

    data["sma_20"] = (
        data["close"]
        .rolling(20)
        .mean()
    )

    data["sma_200"] = (
        data["close"]
        .rolling(200)
        .mean()
    )

    data["atr_14"] = calculate_atr_series(
        data,
        14,
    )

    data["atr_pct"] = (
        data["atr_14"]
        / data["close"]
    ) * 100

    data["daily_rsi_14"] = calculate_rsi(
        data["close"],
        14,
    )

    data["close_1d_ago"] = data["close"].shift(1)

    data["close_2d_ago"] = data["close"].shift(2)

    data["daily_candle_return_pct"] = (
        (
            data["close"]
            / data["open"]
        ) - 1
    ) * 100

    data["volume_ratio_vs_5d_sma"] = (
        data["volume"]
        / data["volume_sma_5"]
    )

    weekly_data = (
        data[
            [
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        ]
        .resample("W-FRI")
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .dropna()
    )

    weekly_data["weekly_sma_20"] = (
        weekly_data["close"]
        .rolling(20)
        .mean()
    )

    weekly_data["weekly_rsi_14"] = calculate_rsi(
        weekly_data["close"],
        14,
    )

    weekly_features = weekly_data[
        [
            "close",
            "weekly_sma_20",
            "weekly_rsi_14",
        ]
    ].rename(
        columns={
            "close": "weekly_close",
        }
    )

    daily_dates = pd.DataFrame(
        {
            "date": data.index,
        }
    )

    weekly_dates = weekly_features.reset_index()

    weekly_dates = weekly_dates.rename(
        columns={
            weekly_dates.columns[0]: "date",
        }
    )

    merged = pd.merge_asof(
        daily_dates.sort_values("date"),
        weekly_dates.sort_values("date"),
        on="date",
        direction="backward",
    ).set_index("date")

    data["weekly_close"] = merged[
        "weekly_close"
    ]

    data["weekly_sma_20"] = merged[
        "weekly_sma_20"
    ]

    data["weekly_rsi_14"] = merged[
        "weekly_rsi_14"
    ]

    data["historical_filter_pass"] = (
        (data["volume"] > data["volume_sma_5"] * 1.5)
        & (data["daily_candle_return_pct"] > 2)
        & (data["close"] > data["sma_20"])
        & (data["close"] > data["sma_200"])
        & (data["weekly_close"] > data["weekly_sma_20"])
        & (data["atr_pct"] > 3)
        & (data["daily_rsi_14"] >= 55)
        & (data["daily_rsi_14"] <= 90)
        & (data["weekly_rsi_14"] >= 55)
        & (data["close"] > data["close_1d_ago"])
        & (data["close"] > data["close_2d_ago"])
    )

    return data


def calculate_window_metrics_vectorized(
    data,
    entry_indexes,
    window_days,
    include_minimum,
):
    """
    Calculate future maximum/minimum returns for qualifying signals.

    A result is NA if full future window has not elapsed.
    """

    max_returns = []
    min_returns = []
    hit_10pct = []

    high_values = data["high"].to_numpy(
        dtype=float
    )

    low_values = data["low"].to_numpy(
        dtype=float
    )

    close_values = data["close"].to_numpy(
        dtype=float
    )

    total_rows = len(data)

    for entry_index in entry_indexes:
        final_index = entry_index + window_days

        if final_index >= total_rows:
            max_returns.append(np.nan)
            min_returns.append(np.nan)
            hit_10pct.append(pd.NA)
            continue

        entry_price = close_values[entry_index]

        future_highs = high_values[
            entry_index + 1:final_index + 1
        ]

        max_return = (
            np.max(future_highs)
            / entry_price
            - 1
        ) * 100

        max_returns.append(max_return)
        hit_10pct.append(
            bool(max_return >= 10)
        )

        if include_minimum:
            future_lows = low_values[
                entry_index + 1:final_index + 1
            ]

            min_return = (
                np.min(future_lows)
                / entry_price
                - 1
            ) * 100

            min_returns.append(min_return)

        else:
            min_returns.append(np.nan)

    return (
        max_returns,
        min_returns,
        hit_10pct,
    )


def create_historical_filter_records(
    symbol,
    company_name,
    prepared_data,
):
    """
    Create one historical record per qualifying day.

    This version includes the final signal even if the next
    trading-day entry price is not yet available.
    """

    if prepared_data is None or prepared_data.empty:
        return pd.DataFrame()

    signal_data = prepared_data[
        prepared_data["historical_filter_pass"]
    ].copy()

    if signal_data.empty:
        return pd.DataFrame()

    all_data = prepared_data.copy()

    signal_indexes = all_data.index.get_indexer(
        signal_data.index
    )

    entry_indexes = signal_indexes + 1

    # Identify which signals have a valid next trading-day row.
    valid_entry_mask = entry_indexes < len(all_data)

    results = pd.DataFrame(
        {
            "stock_symbol": symbol,
            "stock_name": company_name,
            "signal_date": signal_data.index,
            "signal_day_price": signal_data[
                "close"
            ].to_numpy(),
            "entry_date": [
                all_data.index[idx]
                if valid
                else next_business_day_estimate(
                    signal_date
                )
                for idx, valid, signal_date in zip(
                    entry_indexes,
                    valid_entry_mask,
                    signal_data.index,
                )
            ],
            "entry_price": [
                all_data["close"].iloc[idx]
                if valid
                else np.nan
                for idx, valid in zip(
                    entry_indexes,
                    valid_entry_mask,
                )
            ],
            "entry_status": [
                "Completed Historical Entry"
                if valid
                else "Pending Next Trading Day"
                for valid in valid_entry_mask
            ],
            "entry_price_status": [
                "Available"
                if valid
                else "Not Available Yet"
                for valid in valid_entry_mask
            ],
            "daily_candle_return_pct": signal_data[
                "daily_candle_return_pct"
            ].to_numpy(),
            "volume_ratio_vs_5d_sma": signal_data[
                "volume_ratio_vs_5d_sma"
            ].to_numpy(),
            "daily_rsi_14": signal_data[
                "daily_rsi_14"
            ].to_numpy(),
            "weekly_rsi_14": signal_data[
                "weekly_rsi_14"
            ].to_numpy(),
            "atr_14_pct_of_close": signal_data[
                "atr_pct"
            ].to_numpy(),
            "historical_debt_equity": "Not evaluated",
            "historical_eps": "Not evaluated",
            "historical_roe": "Not evaluated",
        }
    )

    # For completed entries, calculate forward returns.
    completed_entry_indexes = entry_indexes[valid_entry_mask]

    for horizon_name, horizon_days in RETURN_WINDOWS.items():
        include_minimum = horizon_name in [
            "30d",
            "60d",
            "90d",
            "6m",
            "12m",
        ]

        max_returns, min_returns, hit_flags = (
            calculate_window_metrics_vectorized(
                all_data,
                completed_entry_indexes,
                horizon_days,
                include_minimum,
            )
        )

        # Create columns filled with NA.
        max_column = [np.nan] * len(results)
        min_column = [np.nan] * len(results)
        hit_column = [pd.NA] * len(results)

        # Overwrite completed entries only.
        completed_positions = np.where(valid_entry_mask)[0]

        for position, max_ret, min_ret, hit_flag in zip(
            completed_positions,
            max_returns,
            min_returns,
            hit_flags,
        ):
            max_column[position] = max_ret
            min_column[position] = min_ret
            hit_column[position] = hit_flag

        results[f"max_ret_{horizon_name}"] = max_column
        results[f"hit_10pct_{horizon_name}"] = pd.array(
            hit_column,
            dtype="boolean",
        )

        if include_minimum:
            results[f"min_ret_{horizon_name}"] = min_column

    return results


def run_optimized_historical_filter_study(
    universe,
    minimum_market_cap,
    scan_limit,
    progress_callback=None,
):
    """
    Run historical study across selected current market-cap eligible stocks.

    Current market-cap eligibility only.
    Historical D/E, EPS, ROE intentionally excluded.
    """

    output_frames = []

    selected_universe = universe.head(
        scan_limit
    ).copy()

    total_stocks = len(selected_universe)

    for position, (_, record) in enumerate(
        selected_universe.iterrows(),
        start=1,
    ):
        if progress_callback is not None:
            progress_callback(
                position,
                total_stocks,
                record["Symbol"],
            )

        try:
            fundamentals = fetch_fundamentals(
                record["Ticker"]
            )

            market_cap_crore = (
                safe_number(
                    fundamentals.get("marketCap"),
                    0,
                )
                / 10000000
            )

            if market_cap_crore < minimum_market_cap:
                continue

            prepared_data = (
                prepare_historical_filter_indicators(
                    record["Ticker"]
                )
            )

            if prepared_data.empty:
                continue

            stock_records = (
                create_historical_filter_records(
                    record["Symbol"],
                    record["Company Name"],
                    prepared_data,
                )
            )

            if stock_records.empty:
                continue

            stock_records[
                "current_market_cap_cr"
            ] = market_cap_crore

            output_frames.append(
                stock_records
            )

        except Exception:
            pass

    if not output_frames:
        return pd.DataFrame()

    return (
        pd.concat(
            output_frames,
            ignore_index=True,
        )
        .sort_values(
            by=[
                "signal_date",
                "entry_status",
            ],
            ascending=[
                False,
                True,
            ],
        )
        .reset_index(drop=True)
    )


def calculate_historical_study_summary(
    results,
):
    """Calculate historical study summary."""

    if results is None or results.empty:
        return {
            "total_signal_records": 0,
            "unique_stocks": 0,
            "unique_stock_months": 0,
            "unique_stock_quarters": 0,
            "consecutive_signal_records": 0,
            "pending_entries": 0,
            "latest_signal_date": None,
            "horizon_summary": pd.DataFrame(),
            "year_summary": pd.DataFrame(),
        }

    working = results.copy()

    working["signal_date"] = pd.to_datetime(
        working["signal_date"]
    )

    working["signal_year"] = (
        working["signal_date"].dt.year
    )

    working["signal_month"] = (
        working["signal_date"]
        .dt.to_period("M")
        .astype(str)
    )

    working["signal_quarter"] = (
        working["signal_date"]
        .dt.to_period("Q")
        .astype(str)
    )

    pending_entries = (
        working[
            working["entry_status"]
            == "Pending Next Trading Day"
        ]
        .shape[0]
    )

    latest_signal_date = (
        working["signal_date"].max()
        if not working.empty
        else None
    )

    unique_stock_months = (
        working[
            [
                "stock_symbol",
                "signal_month",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    unique_stock_quarters = (
        working[
            [
                "stock_symbol",
                "signal_quarter",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    consecutive_records = 0

    for _, group in working.groupby(
        "stock_symbol"
    ):
        dates = (
            group
            .sort_values("signal_date")[
                "signal_date"
            ]
            .tolist()
        )

        for index in range(1, len(dates)):
            if (
                dates[index] - dates[index - 1]
            ).days <= 4:
                consecutive_records += 1

    horizon_config = [
        (
            "5 Trading Days",
            "max_ret_5d",
            None,
            "hit_10pct_5d",
        ),
        (
            "10 Trading Days",
            "max_ret_10d",
            None,
            "hit_10pct_10d",
        ),
        (
            "20 Trading Days",
            "max_ret_20d",
            None,
            "hit_10pct_20d",
        ),
        (
            "30 Trading Days",
            "max_ret_30d",
            "min_ret_30d",
            "hit_10pct_30d",
        ),
        (
            "60 Trading Days",
            "max_ret_60d",
            "min_ret_60d",
            "hit_10pct_60d",
        ),
        (
            "90 Trading Days",
            "max_ret_90d",
            "min_ret_90d",
            "hit_10pct_90d",
        ),
        (
            "6 Months",
            "max_ret_6m",
            "min_ret_6m",
            "hit_10pct_6m",
        ),
        (
            "9 Months",
            "max_ret_9m",
            None,
            "hit_10pct_9m",
        ),
        (
            "12 Months",
            "max_ret_12m",
            "min_ret_12m",
            "hit_10pct_12m",
        ),
    ]

    horizon_rows = []

    for (
        label,
        max_column,
        min_column,
        hit_column,
    ) in horizon_config:
        max_values = working[
            max_column
        ].dropna()

        hit_values = working[
            hit_column
        ].dropna()

        if min_column is not None:
            min_values = working[
                min_column
            ].dropna()

            average_min = (
                min_values.mean()
                if not min_values.empty
                else None
            )

            median_min = (
                min_values.median()
                if not min_values.empty
                else None
            )
        else:
            average_min = None
            median_min = None

        horizon_rows.append(
            {
                "Horizon": label,
                "Completed Signals": len(
                    max_values
                ),
                "Average Max Return %": (
                    max_values.mean()
                    if not max_values.empty
                    else None
                ),
                "Median Max Return %": (
                    max_values.median()
                    if not max_values.empty
                    else None
                ),
                "Average Min Return %": average_min,
                "Median Min Return %": median_min,
                "10% Hit Rate %": (
                    hit_values.mean() * 100
                    if not hit_values.empty
                    else None
                ),
            }
        )

    horizon_summary = pd.DataFrame(
        horizon_rows
    )

    year_summary = (
        working
        .groupby("signal_year")
        .agg(
            total_signal_records=(
                "stock_symbol",
                "count",
            ),
            unique_stocks=(
                "stock_symbol",
                "nunique",
            ),
            average_max_ret_30d=(
                "max_ret_30d",
                "mean",
            ),
            median_max_ret_30d=(
                "max_ret_30d",
                "median",
            ),
            hit_10pct_30d_rate=(
                "hit_10pct_30d",
                "mean",
            ),
            average_max_ret_60d=(
                "max_ret_60d",
                "mean",
            ),
            median_max_ret_60d=(
                "max_ret_60d",
                "median",
            ),
            hit_10pct_60d_rate=(
                "hit_10pct_60d",
                "mean",
            ),
        )
        .reset_index()
    )

    if not year_summary.empty:
        year_summary[
            "hit_10pct_30d_rate"
        ] = (
            year_summary[
                "hit_10pct_30d_rate"
            ] * 100
        )

        year_summary[
            "hit_10pct_60d_rate"
        ] = (
            year_summary[
                "hit_10pct_60d_rate"
            ] * 100
        )

    return {
        "total_signal_records": len(working),
        "unique_stocks": working[
            "stock_symbol"
        ].nunique(),
        "unique_stock_months": unique_stock_months,
        "unique_stock_quarters": unique_stock_quarters,
        "consecutive_signal_records": consecutive_records,
        "pending_entries": pending_entries,
        "latest_signal_date": latest_signal_date,
        "horizon_summary": horizon_summary,
        "year_summary": year_summary,
    }


# =============================================================================
# LIVE / TODAY FILTER SCAN
# =============================================================================

def calculate_provisional_rsi(
    completed_close_series,
    current_live_price,
):
    """
    Calculate provisional daily RSI.

    Uses completed daily closes plus current live price as today's
    provisional close. This does not alter historical data.
    """

    provisional_close = pd.concat(
        [
            completed_close_series,
            pd.Series(
                [current_live_price],
                index=[
                    completed_close_series.index[-1]
                    + pd.Timedelta(days=1)
                ],
            ),
        ]
    )

    rsi_series = calculate_rsi(
        provisional_close,
        14,
    )

    rsi_value = rsi_series.iloc[-1]

    return (
        float(rsi_value)
        if not pd.isna(rsi_value)
        else None
    )


def calculate_live_weekly_values(
    completed_daily_data,
    current_live_price,
):
    """
    Calculate provisional current weekly close, weekly SMA20 and weekly RSI14.

    The current weekly candle uses latest live price as provisional close.
    Previous completed weeks remain unchanged.
    """

    weekly_completed = resample_ohlcv(
        completed_daily_data,
        "Weekly",
    )

    if weekly_completed.empty:
        return None, None, None

    current_timestamp = get_ist_now()

    current_week_end = (
        pd.Timestamp(current_timestamp.date())
        + pd.offsets.Week(
            weekday=4
        )
    ).normalize()

    provisional_weekly = weekly_completed.copy()

    if (
        len(provisional_weekly) > 0
        and provisional_weekly.index[-1]
        == current_week_end
    ):
        provisional_weekly.loc[
            provisional_weekly.index[-1],
            "close",
        ] = current_live_price
    else:
        new_week_row = pd.DataFrame(
            {
                "open": [current_live_price],
                "high": [current_live_price],
                "low": [current_live_price],
                "close": [current_live_price],
                "volume": [0],
            },
            index=[current_week_end],
        )

        provisional_weekly = pd.concat(
            [
                provisional_weekly,
                new_week_row,
            ]
        )

    weekly_close = float(
        provisional_weekly["close"].iloc[-1]
    )

    weekly_sma_20 = (
        float(
            provisional_weekly["close"]
            .rolling(20)
            .mean()
            .iloc[-1]
        )
        if len(provisional_weekly) >= 20
        else None
    )

    weekly_rsi_series = calculate_rsi(
        provisional_weekly["close"],
        14,
    )

    weekly_rsi_value = weekly_rsi_series.iloc[-1]

    weekly_rsi = (
        float(weekly_rsi_value)
        if not pd.isna(weekly_rsi_value)
        else None
    )

    return (
        weekly_close,
        weekly_sma_20,
        weekly_rsi,
    )


def calculate_live_filter_for_stock(
    record,
    minimum_market_cap,
    market_open,
):
    """
    Evaluate the current technical filter.

    If market is open:
    - completed historical daily values are retained
    - latest intraday price and cumulative volume are used for today's
      provisional close / volume

    If market is closed:
    - final/latest daily close and daily volume are used
    """

    ticker = record["Ticker"]

    daily_data = fetch_price_data(
        ticker,
        "5y",
    )

    if daily_data.empty or len(daily_data) < 220:
        return None

    fundamentals = fetch_fundamentals(
        ticker
    )

    market_cap_crore = (
        safe_number(
            fundamentals.get("marketCap"),
            0,
        )
        / 10000000
    )

    if market_cap_crore < minimum_market_cap:
        return None

    live_source = "Daily End-of-Day"
    live_timestamp = None
    signal_status = "Confirmed EOD"
    data_mode = "End-of-Day Confirmed"

    # -------------------------------------------------------------------------
    # MARKET OPEN: use intraday current price / volume.
    # -------------------------------------------------------------------------

    if market_open:
        intraday_data = fetch_intraday_data(
            ticker
        )

        if not intraday_data.empty:
            latest_intraday = intraday_data.iloc[-1]

            latest_price = float(
                latest_intraday["close"]
            )

            today_open = float(
                intraday_data["open"].iloc[0]
            )

            today_high = float(
                intraday_data["high"].max()
            )

            today_low = float(
                intraday_data["low"].min()
            )

            # Intraday bars commonly contain per-bar volume;
            # cumulative sum is used for current session volume.
            today_volume = float(
                intraday_data["volume"].sum()
            )

            timestamp = intraday_data.index[-1]

            if getattr(timestamp, "tzinfo", None) is not None:
                timestamp_ist = timestamp.tz_convert(
                    IST
                )
            else:
                timestamp_ist = timestamp.tz_localize(
                    IST
                )

            live_timestamp = timestamp_ist

            live_source = "Yahoo Finance 5-minute Intraday"

            signal_status = "Provisional Intraday"

            data_mode = "Provisional Intraday"

            # If the daily dataset already contains a row for today,
            # remove it to ensure all daily indicators use completed days.
            now_ist = get_ist_now()

            if (
                daily_data.index[-1].date()
                == now_ist.date()
            ):
                completed_data = daily_data.iloc[
                    :-1
                ].copy()
            else:
                completed_data = daily_data.copy()

        else:
            # Fallback to daily data if intraday unavailable.
            completed_data = daily_data.copy()

            latest_price = float(
                daily_data["close"].iloc[-1]
            )

            today_open = float(
                daily_data["open"].iloc[-1]
            )

            today_high = float(
                daily_data["high"].iloc[-1]
            )

            today_low = float(
                daily_data["low"].iloc[-1]
            )

            today_volume = float(
                daily_data["volume"].iloc[-1]
            )

            live_timestamp = daily_data.index[-1]

            live_source = (
                "Fallback: Latest Daily Data"
            )

            signal_status = "Fallback Daily Data"

            data_mode = (
                "Daily Fallback - Intraday Unavailable"
            )

    # -------------------------------------------------------------------------
    # MARKET CLOSED: use daily final/latest values.
    # -------------------------------------------------------------------------

    else:
        completed_data = daily_data.iloc[
            :-1
        ].copy()

        latest_price = float(
            daily_data["close"].iloc[-1]
        )

        today_open = float(
            daily_data["open"].iloc[-1]
        )

        today_high = float(
            daily_data["high"].iloc[-1]
        )

        today_low = float(
            daily_data["low"].iloc[-1]
        )

        today_volume = float(
            daily_data["volume"].iloc[-1]
        )

        live_timestamp = daily_data.index[-1]

    if len(completed_data) < 220:
        return None

    # -------------------------------------------------------------------------
    # All rolling indicators below use only previously completed daily data.
    # -------------------------------------------------------------------------

    prior_5d_average_volume = float(
        completed_data["volume"]
        .tail(5)
        .mean()
    )

    daily_sma_20 = float(
        completed_data["close"]
        .tail(20)
        .mean()
    )

    daily_sma_200 = float(
        completed_data["close"]
        .tail(200)
        .mean()
    )

    previous_day_close = float(
        completed_data["close"].iloc[-1]
    )

    two_day_ago_close = float(
        completed_data["close"].iloc[-2]
    )

    completed_atr = calculate_atr(
        completed_data,
        14,
    )

    atr_pct = (
        completed_atr / latest_price * 100
        if completed_atr is not None
        and latest_price != 0
        else None
    )

    provisional_daily_rsi = calculate_provisional_rsi(
        completed_data["close"],
        latest_price,
    )

    (
        provisional_weekly_close,
        provisional_weekly_sma_20,
        provisional_weekly_rsi,
    ) = calculate_live_weekly_values(
        completed_data,
        latest_price,
    )

    current_day_change_pct = (
        (latest_price / today_open - 1) * 100
        if today_open != 0
        else None
    )

    volume_ratio = (
        today_volume / prior_5d_average_volume
        if prior_5d_average_volume != 0
        else None
    )

    # Conditions matching your query.
    volume_condition = (
        volume_ratio is not None
        and volume_ratio > 1.5
    )

    candle_condition = (
        current_day_change_pct is not None
        and current_day_change_pct > 2
    )

    sma20_condition = (
        latest_price > daily_sma_20
    )

    sma200_condition = (
        latest_price > daily_sma_200
    )

    weekly_sma_condition = (
        provisional_weekly_close is not None
        and provisional_weekly_sma_20 is not None
        and provisional_weekly_close
        > provisional_weekly_sma_20
    )

    atr_condition = (
        atr_pct is not None
        and atr_pct > 3
    )

    daily_rsi_condition = (
        provisional_daily_rsi is not None
        and provisional_daily_rsi >= 55
        and provisional_daily_rsi <= 90
    )

    weekly_rsi_condition = (
        provisional_weekly_rsi is not None
        and provisional_weekly_rsi >= 55
    )

    close_1d_condition = (
        latest_price > previous_day_close
    )

    close_2d_condition = (
        latest_price > two_day_ago_close
    )

    all_filters_pass = (
        volume_condition
        and candle_condition
        and sma20_condition
        and sma200_condition
        and weekly_sma_condition
        and atr_condition
        and daily_rsi_condition
        and weekly_rsi_condition
        and close_1d_condition
        and close_2d_condition
    )

    return {
        "stock_symbol": record["Symbol"],
        "stock_name": record["Company Name"],
        "industry": record["Industry"],
        "market_cap_cr": market_cap_crore,
        "market_status": (
            "OPEN"
            if market_open
            else "CLOSED"
        ),
        "data_mode": data_mode,
        "signal_status": signal_status,
        "data_source": live_source,
        "signal_timestamp_ist": live_timestamp,
        "latest_price": latest_price,
        "today_open": today_open,
        "today_high": today_high,
        "today_low": today_low,
        "current_day_change_pct": current_day_change_pct,
        "current_volume": today_volume,
        "prior_5d_average_volume": prior_5d_average_volume,
        "volume_ratio_vs_prior_5d": volume_ratio,
        "daily_sma_20": daily_sma_20,
        "daily_sma_200": daily_sma_200,
        "weekly_close_or_live_price": provisional_weekly_close,
        "weekly_sma_20": provisional_weekly_sma_20,
        "atr_14_pct": atr_pct,
        "daily_rsi_14": provisional_daily_rsi,
        "weekly_rsi_14": provisional_weekly_rsi,
        "volume_condition": volume_condition,
        "candle_change_condition": candle_condition,
        "close_above_sma20": sma20_condition,
        "close_above_sma200": sma200_condition,
        "weekly_above_sma20": weekly_sma_condition,
        "atr_condition": atr_condition,
        "daily_rsi_condition": daily_rsi_condition,
        "weekly_rsi_condition": weekly_rsi_condition,
        "close_above_1d": close_1d_condition,
        "close_above_2d": close_2d_condition,
        "all_filters_pass": all_filters_pass,
        "current_debt_equity_reference": fundamentals.get(
            "debtToEquity"
        ),
        "current_eps_reference": fundamentals.get(
            "trailingEps"
        ),
        "current_roe_reference": fundamentals.get(
            "returnOnEquity"
        ),
    }


def run_live_filter_scan(
    universe,
    minimum_market_cap,
    scan_limit,
    market_open,
    progress_callback=None,
):
    """Run the new Live / Today technical filter scan."""

    rows = []

    selected_universe = universe.head(
        scan_limit
    ).copy()

    total = len(selected_universe)

    for position, (_, record) in enumerate(
        selected_universe.iterrows(),
        start=1,
    ):
        if progress_callback is not None:
            progress_callback(
                position,
                total,
                record["Symbol"],
            )

        try:
            result = calculate_live_filter_for_stock(
                record,
                minimum_market_cap,
                market_open,
            )

            if result is not None:
                rows.append(result)

        except Exception:
            pass

    if not rows:
        return pd.DataFrame()

    results = pd.DataFrame(rows)

    return results.sort_values(
        by=[
            "all_filters_pass",
            "current_day_change_pct",
            "volume_ratio_vs_prior_5d",
        ],
        ascending=[
            False,
            False,
            False,
        ],
    ).reset_index(drop=True)


# =============================================================================
# CURRENT SCANNER / COMMAND CENTER
# =============================================================================

def analyse_stock_for_scanner(
    record,
    benchmark_data,
    minimum_market_cap,
):
    """Build current stock record for Command Center and Ranking Scanner."""

    ticker = record["Ticker"]

    daily_data = fetch_price_data(
        ticker,
        "5y",
    )

    if daily_data.empty:
        return None

    fundamentals = fetch_fundamentals(
        ticker
    )

    market_cap_crore = (
        safe_number(
            fundamentals.get("marketCap"),
            0,
        )
        / 10000000
    )

    if market_cap_crore < minimum_market_cap:
        return None

    weekly_data = resample_ohlcv(
        daily_data,
        "Weekly",
    )

    monthly_data = resample_ohlcv(
        daily_data,
        "Monthly",
    )

    daily_patterns = detect_patterns(
        daily_data
    )

    weekly_patterns = detect_patterns(
        weekly_data
    )

    monthly_patterns = detect_patterns(
        monthly_data
    )

    rs_data = calculate_relative_strength(
        daily_data,
        benchmark_data,
    )

    alignment = calculate_multitimeframe_alignment(
        daily_data,
        weekly_data,
        monthly_data,
    )

    winner_score = calculate_winner_score(
        fundamentals,
        rs_data,
        alignment,
        daily_data,
        weekly_data,
        monthly_data,
        daily_patterns,
        weekly_patterns,
        monthly_patterns,
    )

    all_patterns = []

    for timeframe, patterns in [
        (
            "Daily",
            daily_patterns,
        ),
        (
            "Weekly",
            weekly_patterns,
        ),
        (
            "Monthly",
            monthly_patterns,
        ),
    ]:
        for pattern in patterns:
            copied_pattern = dict(pattern)
            copied_pattern["Timeframe"] = timeframe
            all_patterns.append(copied_pattern)

    bullish_confirmed = [
        pattern
        for pattern in all_patterns
        if (
            pattern["Direction"] == "Bullish"
            and pattern["Status"] == "Confirmed"
        )
    ]

    if bullish_confirmed:
        timeframe_priority = {
            "Monthly": 3,
            "Weekly": 2,
            "Daily": 1,
        }

        selected_pattern = sorted(
            bullish_confirmed,
            key=lambda item: timeframe_priority.get(
                item["Timeframe"],
                0,
            ),
            reverse=True,
        )[0]

    elif all_patterns:
        selected_pattern = all_patterns[0]

    else:
        selected_pattern = None

    if selected_pattern is not None:
        if selected_pattern["Timeframe"] == "Daily":
            plan_data = daily_data
        elif selected_pattern["Timeframe"] == "Weekly":
            plan_data = weekly_data
        else:
            plan_data = monthly_data

        trade_plan = calculate_trade_plan(
            plan_data,
            selected_pattern,
        )
    else:
        trade_plan = None

    support, resistance = calculate_support_resistance(
        daily_data
    )

    ma20 = (
        float(
            daily_data["close"]
            .rolling(20)
            .mean()
            .iloc[-1]
        )
        if len(daily_data) >= 20
        else None
    )

    ma50 = (
        float(
            daily_data["close"]
            .rolling(50)
            .mean()
            .iloc[-1]
        )
        if len(daily_data) >= 50
        else None
    )

    ma200 = (
        float(
            daily_data["close"]
            .rolling(200)
            .mean()
            .iloc[-1]
        )
        if len(daily_data) >= 200
        else None
    )

    return {
        "Stock": record["Symbol"],
        "Company": record["Company Name"],
        "Industry": record["Industry"],
        "Ticker": ticker,
        "Current Price": float(
            daily_data["close"].iloc[-1]
        ),
        "Market Cap (Cr)": market_cap_crore,
        "Winner Score": winner_score["score"],
        "Winner Category": winner_score["category"],
        "RS Status": rs_data["status"],
        "RS Trend": rs_data["rs_trend"],
        "RS 3M %": rs_data["relative_3m"],
        "RS 6M %": rs_data["relative_6m"],
        "Daily Trend": alignment["daily"],
        "Weekly Trend": alignment["weekly"],
        "Monthly Trend": alignment["monthly"],
        "MTF Score": alignment["score"],
        "MTF Alignment": alignment["status"],
        "Current Pattern": (
            selected_pattern["Pattern"]
            if selected_pattern is not None
            else "No active pattern"
        ),
        "Pattern Status": (
            selected_pattern["Status"]
            if selected_pattern is not None
            else "No Pattern"
        ),
        "Pattern Direction": (
            selected_pattern["Direction"]
            if selected_pattern is not None
            else "Neutral"
        ),
        "Pattern Timeframe": (
            selected_pattern["Timeframe"]
            if selected_pattern is not None
            else "Overall"
        ),
        "Breakout Level": (
            selected_pattern["Level"]
            if selected_pattern is not None
            else None
        ),
        "Pattern Volume %": (
            selected_pattern["Volume %"]
            if selected_pattern is not None
            else None
        ),
        "Support": support,
        "Resistance": resistance,
        "MA20": ma20,
        "MA50": ma50,
        "MA200": ma200,
        "Entry Quality": (
            trade_plan["entry_quality"]["status"]
            if trade_plan is not None
            else "No directional setup"
        ),
        "Distance From Breakout %": (
            trade_plan["entry_quality"]["distance_percent"]
            if trade_plan is not None
            else None
        ),
        "Stop Loss": (
            trade_plan["selected_stop"]
            if trade_plan is not None
            else None
        ),
        "Target": (
            trade_plan["target"]
            if trade_plan is not None
            else None
        ),
        "Risk Reward": (
            trade_plan["risk_reward"]
            if trade_plan is not None
            else None
        ),
        "Positive Evidence": " | ".join(
            winner_score["positives"][:5]
        ),
        "Risk Flags": " | ".join(
            winner_score["risks"][:5]
        ),
    }


def run_current_universe_scan(
    universe,
    benchmark_data,
    minimum_market_cap,
    scan_limit,
    progress_callback=None,
):
    """Run current command center / ranking scan."""

    rows = []

    selected_universe = universe.head(
        scan_limit
    ).copy()

    total = len(selected_universe)

    for position, (_, record) in enumerate(
        selected_universe.iterrows(),
        start=1,
    ):
        if progress_callback is not None:
            progress_callback(
                position,
                total,
                record["Symbol"],
            )

        try:
            result = analyse_stock_for_scanner(
                record,
                benchmark_data,
                minimum_market_cap,
            )

            if result is not None:
                rows.append(result)

        except Exception:
            pass

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows)


# =============================================================================
# MARKET REGIME
# =============================================================================

def calculate_market_breadth(scan_data):
    """Calculate breadth above Daily MA50 and MA200."""

    if scan_data is None or scan_data.empty:
        return {
            "above_50dma": None,
            "above_200dma": None,
            "count": 0,
        }

    valid_50 = scan_data.dropna(
        subset=[
            "Current Price",
            "MA50",
        ]
    )

    valid_200 = scan_data.dropna(
        subset=[
            "Current Price",
            "MA200",
        ]
    )

    above_50 = (
        (
            valid_50["Current Price"]
            > valid_50["MA50"]
        ).mean()
        * 100
        if not valid_50.empty
        else None
    )

    above_200 = (
        (
            valid_200["Current Price"]
            > valid_200["MA200"]
        ).mean()
        * 100
        if not valid_200.empty
        else None
    )

    return {
        "above_50dma": above_50,
        "above_200dma": above_200,
        "count": len(scan_data),
    }


def calculate_market_regime(
    nifty_data,
    breadth,
):
    """Classify broad market conditions."""

    nifty_trend = calculate_overall_trend(
        nifty_data
    )

    breadth_50 = safe_number(
        breadth.get("above_50dma")
    )

    if (
        nifty_trend == "Strong bullish"
        and breadth_50 is not None
        and breadth_50 >= 60
    ):
        regime = "Strong Bullish"

    elif (
        nifty_trend in [
            "Strong bullish",
            "Bullish",
        ]
        and breadth_50 is not None
        and breadth_50 >= 50
    ):
        regime = "Bullish"

    elif (
        nifty_trend == "Bearish"
        or (
            breadth_50 is not None
            and breadth_50 < 35
        )
    ):
        regime = "Defensive"

    else:
        regime = "Neutral"

    return {
        "regime": regime,
        "nifty_trend": nifty_trend,
        "breadth_50": breadth_50,
        "breadth_200": breadth.get(
            "above_200dma"
        ),
    }


# =============================================================================
# VISUALIZATIONS
# =============================================================================

def create_stock_chart(
    data,
    patterns,
    title,
):
    """Create candlestick chart with MA20/MA50, volume and pattern levels."""

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
            line=dict(
                color="#2563eb",
                width=1.5,
            ),
        ),
        row=1,
        col=1,
    )

    figure.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data["close"].rolling(50).mean(),
            name="50 MA",
            line=dict(
                color="#f59e0b",
                width=1.5,
            ),
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

    for pattern in patterns:
        color = "#16a34a"

        if pattern["Direction"] == "Bearish":
            color = "#dc2626"

        elif pattern["Direction"] == "Neutral":
            color = "#f59e0b"

        figure.add_hline(
            y=pattern["Level"],
            line_dash="dot",
            line_color=color,
            annotation_text=pattern["Pattern"],
            row=1,
            col=1,
        )

    figure.update_layout(
        title=title,
        height=620,
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        legend_orientation="h",
    )

    return figure


def create_relative_strength_chart(
    rs_line,
    symbol,
):
    """Create RS line chart."""

    figure = go.Figure()

    if rs_line is not None and not rs_line.empty:
        figure.add_trace(
            go.Scatter(
                x=rs_line.index,
                y=rs_line,
                mode="lines",
                name="RS Line",
                line=dict(
                    color="#2563eb",
                    width=2,
                ),
            )
        )

    figure.update_layout(
        title=f"{symbol} Relative Strength vs Nifty 50",
        height=360,
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Stock Close / Nifty 50 Close",
    )

    return figure

def calculate_position_outlook(
    daily_data,
    weekly_data,
    rs_data,
    daily_patterns,
    weekly_patterns,
    monthly_patterns,
    current_price,
    avg_price,
):
    """
    Calculate a simple 1–3 month technical outlook and suggested action.
    """

    daily_trend = calculate_overall_trend(daily_data)
    weekly_trend = calculate_overall_trend(weekly_data)

    rs_status = rs_data.get("status", "Insufficient data")
    rs_trend = rs_data.get("rs_trend", "Unavailable")

    # Aggregate patterns
    all_patterns = []
    for timeframe, patterns in [
        ("Daily", daily_patterns),
        ("Weekly", weekly_patterns),
        ("Monthly", monthly_patterns),
    ]:
        for pattern in patterns:
            copied = dict(pattern)
            copied["Timeframe"] = timeframe
            all_patterns.append(copied)

    bullish_confirmed = [
        p for p in all_patterns
        if p["Direction"] == "Bullish" and p["Status"] == "Confirmed"
    ]

    bearish_confirmed = [
        p for p in all_patterns
        if p["Direction"] == "Bearish" and p["Status"] == "Confirmed"
    ]

    # Outlook logic
    bullish_conditions = [
        daily_trend in ["Strong bullish", "Bullish"],
        weekly_trend in ["Strong bullish", "Bullish"],
        rs_status in ["Leader", "Strong"],
        rs_trend == "Rising",
        len(bearish_confirmed) == 0,
    ]

    bearish_conditions = [
        daily_trend == "Bearish",
        weekly_trend == "Bearish",
        rs_status == "Weak",
        rs_trend == "Falling",
        len(bearish_confirmed) > 0,
    ]

    bullish_score = sum(bullish_conditions)
    bearish_score = sum(bearish_conditions)

    if bullish_score >= 4:
        outlook = "Bullish"
        action = "Hold / Add"
    elif bearish_score >= 3:
        outlook = "Bearish"
        action = "Reduce / Exit"
    else:
        outlook = "Neutral"
        action = "Hold / Monitor"

    reasons = []

    if daily_trend in ["Strong bullish", "Bullish"]:
        reasons.append(f"Daily trend is {daily_trend}")
    elif daily_trend == "Bearish":
        reasons.append(f"Daily trend is {daily_trend}")

    if weekly_trend in ["Strong bullish", "Bullish"]:
        reasons.append(f"Weekly trend is {weekly_trend}")
    elif weekly_trend == "Bearish":
        reasons.append(f"Weekly trend is {weekly_trend}")

    if rs_status in ["Leader", "Strong"]:
        reasons.append(f"Relative Strength status is {rs_status}")
    elif rs_status == "Weak":
        reasons.append(f"Relative Strength status is {rs_status}")

    if rs_trend == "Rising":
        reasons.append("RS line is rising")
    elif rs_trend == "Falling":
        reasons.append("RS line is falling")

    if bullish_confirmed:
        reasons.append(
            f"Confirmed bullish pattern: {bullish_confirmed[0]['Pattern']} "
            f"on {bullish_confirmed[0]['Timeframe']}"
        )

    if bearish_confirmed:
        reasons.append(
            f"Confirmed bearish pattern: {bearish_confirmed[0]['Pattern']} "
            f"on {bearish_confirmed[0]['Timeframe']}"
        )

    # P&L-based nuance (optional, mild)
    if avg_price is not None and avg_price > 0:
        pnl_pct = (current_price / avg_price - 1) * 100
        if pnl_pct > 20 and outlook == "Bullish":
            reasons.append("Position is up >20% with bullish setup – consider partial profit on weakness")
        elif pnl_pct < -15 and outlook == "Bearish":
            reasons.append("Position is down >15% with bearish setup – consider reducing on rallies")

    return {
        "outlook": outlook,
        "action": action,
        "reasons": reasons,
    }


def fetch_metals_data():
    """
    Fetch price data for gold & silver ETFs and futures.
    Returns a dict with DataFrames and latest prices.
    """

    tickers = {
        "gold_etf": GOLD_ETF_TICKER,
        "silver_etf": SILVER_ETF_TICKER,
        "gold_futures": GOLD_FUTURES_TICKER,
        "silver_futures": SILVER_FUTURES_TICKER,
        "usd_index": USD_INDEX_TICKER,
    }

    data = {}

    for key, ticker in tickers.items():
        try:
            df = fetch_price_data(ticker, "5y")
            data[key] = df
        except Exception:
            data[key] = pd.DataFrame()

    result = {}

    for key, df in data.items():
        if df is not None and not df.empty:
            latest_price = float(df["close"].iloc[-1])
        else:
            latest_price = None

        result[key] = {
            "data": df,
            "latest_price": latest_price,
        }

    return result


def calculate_metals_outlook(
    etf_data,
    futures_data,
    usd_data,
    metal_type,
):
    """
    Calculate a simple outlook for gold or silver.
    metal_type: 'gold' or 'silver'
    """

    etf_df = etf_data.get("data", pd.DataFrame())
    futures_df = futures_data.get("data", pd.DataFrame())
    usd_df = usd_data.get("data", pd.DataFrame())

    if etf_df.empty:
        return {
            "outlook": "Insufficient data",
            "stance": "Hold / No fresh calls",
            "reasons": ["ETF price data unavailable."],
        }

    etf_trend = calculate_overall_trend(etf_df)
    futures_trend = calculate_overall_trend(futures_df) if not futures_df.empty else "Unavailable"

    etf_rsi = calculate_rsi(etf_df["close"], 14).iloc[-1] if not etf_df.empty else None

    # USD trend (simplified)
    if not usd_df.empty:
        usd_trend = calculate_overall_trend(usd_df)
    else:
        usd_trend = "Unavailable"

    reasons = []
    score = 0

    # Technical score
    if etf_trend in ["Strong bullish", "Bullish"]:
        score += 2
        reasons.append(f"ETF trend is {etf_trend}")
    elif etf_trend == "Bearish":
        score -= 2
        reasons.append(f"ETF trend is {etf_trend}")

    if futures_trend in ["Strong bullish", "Bullish"]:
        score += 2
        reasons.append(f"Futures trend is {futures_trend}")
    elif futures_trend == "Bearish":
        score -= 2
        reasons.append(f"Futures trend is {futures_trend}")

    if etf_rsi is not None:
        if etf_rsi < 40:
            score += 1
            reasons.append(f"ETF RSI is oversold ({etf_rsi:.1f})")
        elif etf_rsi > 65:
            score -= 1
            reasons.append(f"ETF RSI is overbought ({etf_rsi:.1f})")

    # Macro / fundamental context (static knowledge)
    if metal_type == "gold":
        reasons.append(
            "Central banks have been net buyers of gold, supporting long-term demand."
        )
        if usd_trend == "Bearish":
            score += 1
            reasons.append("USD trend is bearish, typically supportive for gold")
        elif usd_trend == "Strong bullish":
            score -= 1
            reasons.append("USD trend is strong, typically a headwind for gold")

        reasons.append(
            "Markets have been pricing in rate-cut expectations over the medium term, "
            "which can be supportive for gold if real yields fall."
        )

    elif metal_type == "silver":
        reasons.append(
            "Silver has been in a multi-year supply deficit, with 2026 deficit "
            "estimated around 46–67 million ounces."
        )
        reasons.append(
            "Solar demand for silver is falling in 2026 due to thrifting, but "
            "other industrial uses (EVs, AI, electronics) and investment demand "
            "remain supportive."
        )
        reasons.append(
            "Silver often moves with gold on monetary drivers but is more volatile "
            "and more sensitive to industrial demand."
        )

        if etf_trend in ["Strong bullish", "Bullish"]:
            score += 1
            reasons.append("Bullish ETF trend supports accumulation on dips")

    # Map score to stance
    if score >= 3:
        stance = "Accumulate on dips"
    elif score >= 1:
        stance = "Hold / Continue SIP"
    elif score <= -2:
        stance = "Trim on strength"
    else:
        stance = "Hold / No strong signal"

    if score >= 2:
        outlook = "Bullish"
    elif score <= -2:
        outlook = "Bearish"
    else:
        outlook = "Neutral"

    return {
        "outlook": outlook,
        "stance": stance,
        "reasons": reasons,
    }


# =============================================================================
# LOAD UNIVERSE AND BENCHMARK
# =============================================================================

with st.spinner(
    "Loading Nifty Total Market constituents..."
):
    stock_universe, universe_error = (
        get_nifty_total_market_members()
    )

with st.spinner(
    "Loading Nifty 50 benchmark..."
):
    nifty_50_data = fetch_price_data(
        NIFTY_50_BENCHMARK,
        "5y",
    )


# =============================================================================
# HEADER
# =============================================================================

st.markdown(
    """
    <div class="main-title">
        📊 Nifty Total Market Research Dashboard
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="sub-title">
        Daily Command Center • Stock Research • Winner Ranking •
        Historical Closing-Price Study • Live / Today Provisional Filter Scan
    </div>
    """,
    unsafe_allow_html=True,
)

if universe_error:
    st.warning(universe_error)

if stock_universe.empty:
    st.error(
        "No stock universe could be loaded."
    )
    st.stop()

if nifty_50_data.empty:
    st.warning(
        "Nifty 50 benchmark data is unavailable. "
        "Relative Strength results may be incomplete."
    )


# =============================================================================
# SIDEBAR
# =============================================================================

requested_mode = st.session_state.get(
    "requested_mode"
)

if requested_mode in DASHBOARD_MODES:
    default_mode = requested_mode
    del st.session_state["requested_mode"]
else:
    default_mode = "Daily Market Command Center"

with st.sidebar:
    st.header("🔍 Dashboard Controls")

    dashboard_mode = st.radio(
        "Dashboard Mode",
        DASHBOARD_MODES,
        index=DASHBOARD_MODES.index(
            default_mode
        ),
    )

    minimum_market_cap = st.number_input(
        "Minimum current market cap (₹ crore)",
        min_value=0,
        value=3000,
        step=1000,
    )

    st.caption(
        f"Loaded universe: {len(stock_universe)} stocks"
    )

    if st.button("🔄 Refresh cached data"):
        st.cache_data.clear()
        st.rerun()

    st.divider()

    st.caption(
        "Historical Study:\n\n"
        "• Closing-price logic only\n"
        "• Every qualifying daily signal retained\n"
        "• Next trading-day entry\n"
        "• Pending entries shown with NA entry price\n\n"
        "Live Scan:\n\n"
        "• Intraday Yahoo data during market hours\n"
        "• Provisional until market close\n"
        "• Daily-data fallback if intraday unavailable"
    )


# =============================================================================
# DAILY MARKET COMMAND CENTER
# =============================================================================

if dashboard_mode == "Daily Market Command Center":
    st.subheader(
        "Daily Market Command Center"
    )

    command_col1, command_col2 = st.columns(2)

    with command_col1:
        command_scan_limit = st.selectbox(
            "Stocks to scan",
            [
                50,
                100,
                250,
                500,
                750,
            ],
            index=1,
            key="command_scan_limit",
        )

    with command_col2:
        command_min_score = st.slider(
            "Minimum Winner Score",
            min_value=0,
            max_value=100,
            value=50,
            step=5,
            key="command_min_score",
        )

    if st.button(
        "▶ Run Daily Command Center Scan",
        type="primary",
    ):
        progress_bar = st.progress(0)
        progress_text = st.empty()

        def command_progress(
            position,
            total,
            symbol,
        ):
            progress_bar.progress(
                position / total
            )

            progress_text.caption(
                f"Scanning {position:,} of {total:,}: {symbol}"
            )

        command_results = run_current_universe_scan(
            stock_universe,
            nifty_50_data,
            minimum_market_cap,
            command_scan_limit,
            command_progress,
        )

        progress_bar.empty()
        progress_text.empty()

        st.session_state[
            "command_center_results"
        ] = command_results

    command_results = st.session_state[
        "command_center_results"
    ]

    if command_results is None or command_results.empty:
        st.info(
            "Click Run Daily Command Center Scan to start."
        )

    else:
        breadth = calculate_market_breadth(
            command_results
        )

        regime = calculate_market_regime(
            nifty_50_data,
            breadth,
        )

        if regime["regime"] in [
            "Strong Bullish",
            "Bullish",
        ]:
            regime_class = "green-card"
        elif regime["regime"] == "Defensive":
            regime_class = "red-card"
        else:
            regime_class = "orange-card"

        st.markdown(
            f"""
            <div class="research-card {regime_class}">
                <h3>Market Regime: {regime["regime"]}</h3>
                <p>
                    Nifty 50 Trend: {regime["nifty_trend"]} |
                    Breadth Above 50 DMA:
                    {
                        f"{regime["breadth_50"]:.1f}%"
                        if regime["breadth_50"] is not None
                        else "Not available"
                    }
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        r1, r2, r3, r4 = st.columns(4)

        r1.metric(
            "Nifty 50 Trend",
            regime["nifty_trend"],
        )

        r2.metric(
            "Above 50 DMA",
            (
                f"{regime['breadth_50']:.1f}%"
                if regime["breadth_50"] is not None
                else "Not available"
            ),
        )

        r3.metric(
            "Above 200 DMA",
            (
                f"{regime['breadth_200']:.1f}%"
                if regime["breadth_200"] is not None
                else "Not available"
            ),
        )

        r4.metric(
            "Eligible Stocks",
            breadth["count"],
        )

        st.divider()

        st.subheader(
            "🏆 Top Winner Candidates"
        )

        top_candidates = command_results[
            command_results["Winner Score"]
            >= command_min_score
        ].copy()

        top_candidates = top_candidates.sort_values(
            by=[
                "Winner Score",
                "MTF Score",
                "RS 6M %",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        ).head(15)

        if not top_candidates.empty:
            st.dataframe(
                top_candidates[
                    [
                        "Stock",
                        "Company",
                        "Winner Score",
                        "Winner Category",
                        "RS Status",
                        "MTF Score",
                        "MTF Alignment",
                        "Weekly Trend",
                        "Monthly Trend",
                        "Current Pattern",
                        "Pattern Status",
                        "Entry Quality",
                        "Risk Reward",
                    ]
                ],
                hide_index=True,
                use_container_width=True,
            )

            candidate_symbol = st.selectbox(
                "Open detailed Stock Research",
                top_candidates["Stock"].tolist(),
                key="command_open_stock",
            )

            if st.button(
                "Open Selected Stock Research",
                key="command_open_button",
            ):
                open_stock_research(
                    candidate_symbol
                )
                st.rerun()

        else:
            st.info(
                "No stocks meet selected Winner Score."
            )

        st.divider()

        st.subheader(
            "📈 Confirmed Bullish Breakouts"
        )

        confirmed = command_results[
            (
                command_results["Pattern Status"]
                == "Confirmed"
            )
            & (
                command_results["Pattern Direction"]
                == "Bullish"
            )
        ].copy()

        confirmed = confirmed.sort_values(
            by=[
                "Winner Score",
                "Pattern Volume %",
            ],
            ascending=[
                False,
                False,
            ],
        ).head(25)

        if not confirmed.empty:
            st.dataframe(
                confirmed[
                    [
                        "Stock",
                        "Company",
                        "Current Pattern",
                        "Pattern Timeframe",
                        "Pattern Volume %",
                        "Winner Score",
                        "RS Status",
                        "MTF Alignment",
                        "Entry Quality",
                        "Risk Reward",
                    ]
                ],
                hide_index=True,
                use_container_width=True,
            )

        else:
            st.info(
                "No confirmed bullish breakouts found."
            )

        st.divider()

        st.subheader(
            "🎯 Near Ideal Entry Zone"
        )

        ideal_entries = command_results[
            (
                command_results["Entry Quality"]
                .isin(
                    [
                        "Ideal Entry Zone",
                        "Acceptable Entry Zone",
                    ]
                )
            )
            & (
                command_results["Pattern Direction"]
                == "Bullish"
            )
            & (
                command_results["Winner Score"]
                >= 50
            )
        ].copy()

        ideal_entries = ideal_entries.sort_values(
            by=[
                "Winner Score",
                "Risk Reward",
            ],
            ascending=[
                False,
                False,
            ],
        ).head(25)

        if not ideal_entries.empty:
            st.dataframe(
                ideal_entries[
                    [
                        "Stock",
                        "Company",
                        "Current Pattern",
                        "Pattern Timeframe",
                        "Breakout Level",
                        "Current Price",
                        "Distance From Breakout %",
                        "Entry Quality",
                        "Stop Loss",
                        "Target",
                        "Risk Reward",
                        "Winner Score",
                    ]
                ],
                hide_index=True,
                use_container_width=True,
            )

        else:
            st.info(
                "No current bullish ideal-entry setups."
            )

        st.divider()

        st.subheader(
            "⚠️ Extended / Avoid Chasing"
        )

        extended = command_results[
            (
                command_results["Entry Quality"]
                .isin(
                    [
                        "Extended",
                        "Avoid Chasing",
                    ]
                )
            )
            & (
                command_results["Pattern Direction"]
                == "Bullish"
            )
        ].copy()

        extended = extended.sort_values(
            by=[
                "Distance From Breakout %",
                "Winner Score",
            ],
            ascending=[
                False,
                False,
            ],
        ).head(25)

        if not extended.empty:
            st.dataframe(
                extended[
                    [
                        "Stock",
                        "Company",
                        "Current Pattern",
                        "Pattern Timeframe",
                        "Breakout Level",
                        "Current Price",
                        "Distance From Breakout %",
                        "Entry Quality",
                        "Winner Score",
                        "RS Status",
                    ]
                ],
                hide_index=True,
                use_container_width=True,
            )

        else:
            st.info(
                "No extended bullish setups found."
            )


# =============================================================================
# STOCK RESEARCH
# =============================================================================

elif dashboard_mode == "Stock Research":
    st.subheader("Stock Research")

    symbols = sorted(
        stock_universe["Symbol"].tolist()
    )

    current_symbol = st.session_state.get(
        "selected_symbol",
        "RELIANCE",
    )

    if current_symbol not in symbols:
        current_symbol = symbols[0]

    selected_symbol = st.selectbox(
        "Search a Nifty Total Market stock",
        symbols,
        index=symbols.index(current_symbol),
        key="stock_research_symbol",
    )

    st.session_state["selected_symbol"] = (
        selected_symbol
    )

    selected_record = stock_universe[
        stock_universe["Symbol"]
        == selected_symbol
    ].iloc[0]

    with st.spinner(
        f"Loading research data for {selected_symbol}..."
    ):
        stock_data = fetch_price_data(
            selected_record["Ticker"],
            "5y",
        )

        fundamentals = fetch_fundamentals(
            selected_record["Ticker"]
        )

    if stock_data.empty:
        st.error(
            "Price data unavailable for this stock."
        )
        st.stop()

    daily_data = stock_data

    weekly_data = resample_ohlcv(
        stock_data,
        "Weekly",
    )

    monthly_data = resample_ohlcv(
        stock_data,
        "Monthly",
    )

    daily_patterns = detect_patterns(
        daily_data
    )

    weekly_patterns = detect_patterns(
        weekly_data
    )

    monthly_patterns = detect_patterns(
        monthly_data
    )

    rs_data = calculate_relative_strength(
        stock_data,
        nifty_50_data,
    )

    alignment = calculate_multitimeframe_alignment(
        daily_data,
        weekly_data,
        monthly_data,
    )

    winner_score = calculate_winner_score(
        fundamentals,
        rs_data,
        alignment,
        daily_data,
        weekly_data,
        monthly_data,
        daily_patterns,
        weekly_patterns,
        monthly_patterns,
    )

    current_price = float(
        stock_data["close"].iloc[-1]
    )

    h1, h2, h3, h4, h5 = st.columns(5)

    h1.metric(
        "Last Close",
        format_price(current_price),
    )

    h2.metric(
        "Market Cap",
        format_market_cap(
            fundamentals.get("marketCap")
        ),
    )

    h3.metric(
        "RS Status",
        rs_data["status"],
    )

    h4.metric(
        "MTF Alignment",
        f"{alignment['score']}/10",
    )

    h5.metric(
        "Winner Score",
        f"{winner_score['score']}/100",
        winner_score["category"],
    )

    st.caption(
        f"{selected_record['Company Name']} • "
        f"{selected_record['Industry']}"
    )

    (
        winner_tab,
        technical_tab,
        rs_tab,
        alignment_tab,
        risk_tab,
        fundamentals_tab,
        position_tab,
    ) = st.tabs(
        [
            "🏆 Winner Score",
            "Technical Research",
            "Relative Strength",
            "MTF Alignment",
            "Risk / Reward",
            "Fundamentals",
            "Position & Outlook",
        ]
    )

    with winner_tab:
        st.subheader("Winner Score Breakdown")

        breakdown = pd.DataFrame(
            [
                [
                    "Fundamental Quality",
                    winner_score[
                        "fundamental_score"
                    ],
                    20,
                ],
                [
                    "Relative Strength",
                    winner_score[
                        "relative_strength_score"
                    ],
                    20,
                ],
                [
                    "Multi-Timeframe Alignment",
                    winner_score[
                        "alignment_score"
                    ],
                    20,
                ],
                [
                    "Technical Trend",
                    winner_score[
                        "technical_score"
                    ],
                    15,
                ],
                [
                    "Pattern / Breakout Quality",
                    winner_score[
                        "pattern_score"
                    ],
                    15,
                ],
                [
                    "Volume Confirmation",
                    winner_score[
                        "volume_score"
                    ],
                    5,
                ],
                [
                    "Valuation / Risk",
                    winner_score[
                        "valuation_score"
                    ],
                    5,
                ],
            ],
            columns=[
                "Component",
                "Points",
                "Maximum",
            ],
        )

        breakdown["Strength %"] = (
            breakdown["Points"]
            / breakdown["Maximum"]
        ) * 100

        st.dataframe(
            breakdown,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Strength %": st.column_config.ProgressColumn(
                    "Strength",
                    min_value=0,
                    max_value=100,
                    format="%.0f%%",
                ),
            },
        )

        left_col, right_col = st.columns(2)

        with left_col:
            st.markdown("### Positive Evidence")

            if winner_score["positives"]:
                for item in winner_score["positives"][:18]:
                    st.success(f"✅ {item}")
            else:
                st.info(
                    "No positive rule-based evidence found."
                )

        with right_col:
            st.markdown("### Risk Flags")

            if winner_score["risks"]:
                for item in winner_score["risks"][:18]:
                    st.warning(f"⚠️ {item}")
            else:
                st.success(
                    "No major rule-based risks found."
                )

    with technical_tab:
        daily_tab, weekly_tab, monthly_tab = st.tabs(
            [
                "Daily",
                "Weekly",
                "Monthly",
            ]
        )

        for tab, name, data, patterns in [
            (
                daily_tab,
                "Daily",
                daily_data,
                daily_patterns,
            ),
            (
                weekly_tab,
                "Weekly",
                weekly_data,
                weekly_patterns,
            ),
            (
                monthly_tab,
                "Monthly",
                monthly_data,
                monthly_patterns,
            ),
        ]:
            with tab:
                support, resistance = calculate_support_resistance(
                    data
                )

                t1, t2, t3 = st.columns(3)

                t1.metric(
                    "Trend",
                    calculate_overall_trend(data),
                )

                t2.metric(
                    "Support",
                    format_price(support),
                )

                t3.metric(
                    "Resistance",
                    format_price(resistance),
                )

                if patterns:
                    st.dataframe(
                        pd.DataFrame(patterns),
                        hide_index=True,
                        use_container_width=True,
                    )
                else:
                    st.info(
                        "No supported active pattern detected."
                    )

                st.plotly_chart(
                    create_stock_chart(
                        data,
                        patterns,
                        f"{selected_symbol} — {name}",
                    ),
                    use_container_width=True,
                )

    with rs_tab:
        st.subheader(
            "Relative Strength versus Nifty 50"
        )

        rs1, rs2, rs3 = st.columns(3)

        rs1.metric(
            "RS Status",
            rs_data["status"],
        )

        rs2.metric(
            "RS Line Trend",
            rs_data["rs_trend"],
        )

        rs3.metric(
            "RS 6M",
            format_percent(
                rs_data["relative_6m"]
            ),
        )

        rs_table = pd.DataFrame(
            [
                [
                    "1 Month",
                    rs_data["stock_1m"],
                    rs_data["benchmark_1m"],
                    rs_data["relative_1m"],
                ],
                [
                    "3 Months",
                    rs_data["stock_3m"],
                    rs_data["benchmark_3m"],
                    rs_data["relative_3m"],
                ],
                [
                    "6 Months",
                    rs_data["stock_6m"],
                    rs_data["benchmark_6m"],
                    rs_data["relative_6m"],
                ],
                [
                    "12 Months",
                    rs_data["stock_12m"],
                    rs_data["benchmark_12m"],
                    rs_data["relative_12m"],
                ],
            ],
            columns=[
                "Period",
                "Stock Return %",
                "Nifty 50 Return %",
                "Relative Return %",
            ],
        )

        st.dataframe(
            rs_table,
            hide_index=True,
            use_container_width=True,
        )

        st.plotly_chart(
            create_relative_strength_chart(
                rs_data["rs_line"],
                selected_symbol,
            ),
            use_container_width=True,
        )

    with alignment_tab:
        st.subheader(
            "Multi-Timeframe Alignment"
        )

        a1, a2, a3 = st.columns(3)

        a1.metric(
            "Daily Trend",
            alignment["daily"],
        )

        a2.metric(
            "Weekly Trend",
            alignment["weekly"],
        )

        a3.metric(
            "Monthly Trend",
            alignment["monthly"],
        )

        a4, a5 = st.columns(2)

        a4.metric(
            "Alignment Score",
            f"{alignment['score']}/10",
        )

        a5.metric(
            "Alignment Status",
            alignment["status"],
        )

    with risk_tab:
        st.subheader(
            "Risk / Reward and Position Sizing"
        )

        selected_timeframe = st.selectbox(
            "Timeframe",
            [
                "Daily",
                "Weekly",
                "Monthly",
            ],
            key="risk_timeframe",
        )

        if selected_timeframe == "Daily":
            plan_data = daily_data
            plan_patterns = daily_patterns
        elif selected_timeframe == "Weekly":
            plan_data = weekly_data
            plan_patterns = weekly_patterns
        else:
            plan_data = monthly_data
            plan_patterns = monthly_patterns

        directional_patterns = [
            pattern
            for pattern in plan_patterns
            if pattern["Direction"]
            in [
                "Bullish",
                "Bearish",
            ]
        ]

        if directional_patterns:
            labels = [
                (
                    f"{index + 1}. "
                    f"{pattern['Pattern']} | "
                    f"{pattern['Status']} | "
                    f"{pattern['Direction']}"
                )
                for index, pattern in enumerate(
                    directional_patterns
                )
            ]

            chosen_label = st.selectbox(
                "Detected pattern",
                labels,
                key="risk_pattern",
            )

            chosen_index = labels.index(
                chosen_label
            )

            plan = calculate_trade_plan(
                plan_data,
                directional_patterns[chosen_index],
            )

            p1, p2, p3, p4 = st.columns(4)

            p1.metric(
                "Current Price",
                format_price(
                    plan["current_price"]
                ),
            )

            p2.metric(
                "Breakout / Neckline",
                format_price(
                    plan["breakout_level"]
                ),
            )

            p3.metric(
                "ATR",
                format_price(plan["atr"]),
            )

            p4.metric(
                "Entry Quality",
                plan["entry_quality"]["status"],
            )

            p5, p6, p7, p8 = st.columns(4)

            p5.metric(
                "Support",
                format_price(plan["support"]),
            )

            p6.metric(
                "Stop Loss",
                format_price(
                    plan["selected_stop"]
                ),
            )

            p7.metric(
                "Target",
                format_price(plan["target"]),
            )

            p8.metric(
                "Risk / Reward",
                (
                    f"1 : {plan['risk_reward']:.2f}"
                    if plan["risk_reward"] is not None
                    else "Not available"
                ),
            )

            st.divider()

            portfolio_value = st.number_input(
                "Portfolio value (₹)",
                min_value=10000.0,
                value=1000000.0,
                step=50000.0,
                key="portfolio_value",
            )

            risk_percent = st.slider(
                "Maximum risk per trade (%)",
                min_value=0.25,
                max_value=5.0,
                value=1.0,
                step=0.25,
                key="risk_percent",
            )

            position_size = calculate_position_size(
                portfolio_value,
                risk_percent,
                plan["current_price"],
                plan["selected_stop"],
            )

            if position_size is not None:
                q1, q2, q3, q4 = st.columns(4)

                q1.metric(
                    "Maximum Loss",
                    format_price(
                        position_size["maximum_loss"]
                    ),
                )

                q2.metric(
                    "Risk / Share",
                    format_price(
                        position_size["risk_per_share"]
                    ),
                )

                q3.metric(
                    "Maximum Quantity",
                    f"{position_size['quantity']:,}",
                )

                q4.metric(
                    "Position Value",
                    format_price(
                        position_size["position_value"]
                    ),
                )

                st.caption(
                    "Approximate portfolio allocation: "
                    f"{position_size['allocation']:.2f}%"
                )

        else:
            st.info(
                "No directional pattern available for selected timeframe."
            )

    with fundamentals_tab:
        st.subheader(
            "Current Fundamental Metrics"
        )

        roe = safe_number(
            fundamentals.get("returnOnEquity")
        )

        roa = safe_number(
            fundamentals.get("returnOnAssets")
        )

        profit_margin = safe_number(
            fundamentals.get("profitMargins")
        )

        operating_margin = safe_number(
            fundamentals.get("operatingMargins")
        )

        revenue_growth = safe_number(
            fundamentals.get("revenueGrowth")
        )

        earnings_growth = safe_number(
            fundamentals.get("earningsGrowth")
        )

        eps = safe_number(
            fundamentals.get("trailingEps")
        )

        free_cashflow = safe_number(
            fundamentals.get("freeCashflow")
        )

        fundamental_table = pd.DataFrame(
            [
                [
                    "Sector",
                    fundamentals.get(
                        "sector",
                        "Not available",
                    ),
                ],
                [
                    "Industry",
                    fundamentals.get(
                        "industry",
                        "Not available",
                    ),
                ],
                [
                    "Trailing P/E",
                    fundamentals.get(
                        "trailingPE",
                        "Not available",
                    ),
                ],
                [
                    "Forward P/E",
                    fundamentals.get(
                        "forwardPE",
                        "Not available",
                    ),
                ],
                [
                    "Price / Book",
                    fundamentals.get(
                        "priceToBook",
                        "Not available",
                    ),
                ],
                [
                    "ROE",
                    (
                        f"{roe * 100:.2f}%"
                        if roe is not None
                        else "Not available"
                    ),
                ],
                [
                    "ROA",
                    (
                        f"{roa * 100:.2f}%"
                        if roa is not None
                        else "Not available"
                    ),
                ],
                [
                    "Profit Margin",
                    (
                        f"{profit_margin * 100:.2f}%"
                        if profit_margin is not None
                        else "Not available"
                    ),
                ],
                [
                    "Operating Margin",
                    (
                        f"{operating_margin * 100:.2f}%"
                        if operating_margin is not None
                        else "Not available"
                    ),
                ],
                [
                    "Revenue Growth",
                    (
                        f"{revenue_growth * 100:.2f}%"
                        if revenue_growth is not None
                        else "Not available"
                    ),
                ],
                [
                    "Earnings Growth",
                    (
                        f"{earnings_growth * 100:.2f}%"
                        if earnings_growth is not None
                        else "Not available"
                    ),
                ],
                [
                    "Debt / Equity",
                    fundamentals.get(
                        "debtToEquity",
                        "Not available",
                    ),
                ],
                [
                    "EPS",
                    (
                        f"₹{eps:.2f}"
                        if eps is not None
                        else "Not available"
                    ),
                ],
                [
                    "Free Cash Flow",
                    (
                        f"₹{free_cashflow / 10000000:,.0f} Cr"
                        if free_cashflow is not None
                        else "Not available"
                    ),
                ],
            ],
            columns=[
                "Metric",
                "Value",
            ],
        )

        st.dataframe(
            fundamental_table,
            hide_index=True,
            use_container_width=True,
        )

    with position_tab:
        st.subheader("Position & Outlook (1–3 Months)")

        st.markdown(
            "Enter your average buy price and quantity to see a rule-based "
            "technical outlook and suggested action for the next 1–3 months."
        )

        avg_price = st.number_input(
            "Average buy price (₹)",
            min_value=0.0,
            value=0.0,
            step=0.01,
            key="avg_price_input",
        )

        qty = st.number_input(
            "Quantity",
            min_value=0,
            value=0,
            step=1,
            key="qty_input",
        )

        if avg_price <= 0 or qty <= 0:
            st.info("Enter average buy price and quantity to view outlook.")
        else:
            pnl = (current_price - avg_price) * qty
            pnl_pct = (current_price / avg_price - 1) * 100

            p1, p2, p3 = st.columns(3)

            p1.metric(
                "Current Price",
                format_price(current_price),
            )

            p2.metric(
                "Average Buy Price",
                format_price(avg_price),
            )

            p3.metric(
                "P&L %",
                f"{pnl_pct:.2f}%",
                delta=f"{pnl:,.0f}",
            )

            outlook_data = calculate_position_outlook(
                daily_data=daily_data,
                weekly_data=weekly_data,
                rs_data=rs_data,
                daily_patterns=daily_patterns,
                weekly_patterns=weekly_patterns,
                monthly_patterns=monthly_patterns,
                current_price=current_price,
                avg_price=avg_price,
            )

            st.divider()

            o1, o2 = st.columns(2)

            o1.metric(
                "Outlook (1–3 Months)",
                outlook_data["outlook"],
            )

            o2.metric(
                "Suggested Action",
                outlook_data["action"],
            )

            if outlook_data["reasons"]:
                st.markdown("### Key Reasons")
                for reason in outlook_data["reasons"]:
                    st.write(f"- {reason}")

            support, resistance = calculate_support_resistance(daily_data)
            atr = calculate_atr(daily_data, 14)

            st.divider()

            st.markdown("### Risk Levels")

            r1, r2, r3 = st.columns(3)

            r1.metric(
                "Support",
                format_price(support),
            )

            r2.metric(
                "Resistance",
                format_price(resistance),
            )

            if atr is not None:
                atr_stop_long = current_price - 2 * atr
                r3.metric(
                    "ATR-based Stop (2×ATR)",
                    format_price(atr_stop_long),
                )
            else:
                r3.metric(
                    "ATR-based Stop (2×ATR)",
                    "Not available",
                )

            st.caption(
                "This outlook is rule-based and technical only. It is not "
                "investment advice and does not guarantee future returns."
            )
# =============================================================================
# WINNER RANKING SCANNER
# =============================================================================

elif dashboard_mode == "Winner Ranking Scanner":
    st.subheader(
        "Winner Ranking Scanner"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        scanner_pattern = st.selectbox(
            "Pattern",
            PATTERN_OPTIONS,
            key="scanner_pattern",
        )

        scanner_timeframe = st.selectbox(
            "Timeframe",
            [
                "Any",
                "Daily",
                "Weekly",
                "Monthly",
            ],
            key="scanner_timeframe",
        )

        scanner_status = st.selectbox(
            "Pattern Status",
            [
                "Any",
                "Confirmed",
                "In progress",
                "Candidate",
            ],
            key="scanner_status",
        )

    with col2:
        scanner_trend = st.selectbox(
            "Overall Trend",
            TREND_OPTIONS,
            key="scanner_trend",
        )

        scanner_rs = st.selectbox(
            "Relative Strength",
            RS_STATUS_OPTIONS,
            key="scanner_rs",
        )

        scanner_alignment = st.selectbox(
            "MTF Alignment",
            MTF_ALIGNMENT_OPTIONS,
            key="scanner_alignment",
        )

    with col3:
        scanner_min_score = st.slider(
            "Minimum Winner Score",
            min_value=0,
            max_value=100,
            value=50,
            step=5,
            key="scanner_min_score",
        )

        scanner_category = st.selectbox(
            "Winner Category",
            WINNER_CATEGORY_OPTIONS,
            key="scanner_category",
        )

        scanner_limit = st.selectbox(
            "Stocks to Scan",
            [
                50,
                100,
                250,
                500,
                750,
            ],
            index=1,
            key="scanner_limit",
        )

    if st.button(
        "🏆 Run Winner Ranking Scan",
        type="primary",
    ):
        progress_bar = st.progress(0)
        progress_text = st.empty()

        def scanner_progress(
            position,
            total,
            symbol,
        ):
            progress_bar.progress(
                position / total
            )

            progress_text.caption(
                f"Scanning {position:,} of {total:,}: {symbol}"
            )

        scanner_results = run_current_universe_scan(
            stock_universe,
            nifty_50_data,
            minimum_market_cap,
            scanner_limit,
            scanner_progress,
        )

        progress_bar.empty()
        progress_text.empty()

        st.session_state[
            "winner_scanner_results"
        ] = scanner_results

    scanner_results = st.session_state[
        "winner_scanner_results"
    ]

    if scanner_results is None or scanner_results.empty:
        st.info(
            "Configure filters and run the Winner Ranking Scan."
        )

    else:
        filtered = scanner_results.copy()

        filtered = filtered[
            filtered["Winner Score"]
            >= scanner_min_score
        ]

        if scanner_category != "Any":
            filtered = filtered[
                filtered["Winner Category"]
                == scanner_category
            ]

        if scanner_rs != "Any":
            filtered = filtered[
                filtered["RS Status"]
                == scanner_rs
            ]

        if scanner_alignment != "Any":
            filtered = filtered[
                filtered["MTF Alignment"]
                == scanner_alignment
            ]

        if scanner_pattern != "Any":
            filtered = filtered[
                filtered["Current Pattern"]
                == scanner_pattern
            ]

        if scanner_status != "Any":
            filtered = filtered[
                filtered["Pattern Status"]
                == scanner_status
            ]

        if scanner_timeframe != "Any":
            filtered = filtered[
                filtered["Pattern Timeframe"]
                == scanner_timeframe
            ]

        if scanner_trend != "Any":
            filtered = filtered[
                (
                    filtered["Daily Trend"]
                    == scanner_trend
                )
                | (
                    filtered["Weekly Trend"]
                    == scanner_trend
                )
                | (
                    filtered["Monthly Trend"]
                    == scanner_trend
                )
            ]

        filtered = filtered.sort_values(
            by=[
                "Winner Score",
                "MTF Score",
                "RS 6M %",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )

        st.subheader(
            f"Ranked Results: {len(filtered)}"
        )

        if filtered.empty:
            st.info(
                "No stocks match all selected filters."
            )

        else:
            st.dataframe(
                filtered[
                    [
                        "Stock",
                        "Company",
                        "Industry",
                        "Market Cap (Cr)",
                        "Winner Score",
                        "Winner Category",
                        "RS Status",
                        "RS 3M %",
                        "RS 6M %",
                        "MTF Score",
                        "MTF Alignment",
                        "Daily Trend",
                        "Weekly Trend",
                        "Monthly Trend",
                        "Current Pattern",
                        "Pattern Status",
                        "Pattern Timeframe",
                        "Entry Quality",
                        "Risk Reward",
                    ]
                ],
                hide_index=True,
                use_container_width=True,
            )

            stock_to_open = st.selectbox(
                "Open detailed Stock Research",
                filtered["Stock"].tolist(),
                key="scanner_stock_to_open",
            )

            if st.button(
                "Open Selected Stock Research",
                key="scanner_open_stock",
            ):
                open_stock_research(
                    stock_to_open
                )
                st.rerun()

            ranking_csv = filtered.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Download Winner Ranking CSV",
                data=ranking_csv,
                file_name="nifty_winner_ranking.csv",
                mime="text/csv",
            )


# =============================================================================
# HISTORICAL FILTER STUDY WITH LIVE / TODAY TAB
# =============================================================================

elif dashboard_mode == "Historical Filter Study":
    st.subheader(
        "Historical Filter Study"
    )

    historical_tab, live_tab = st.tabs(
        [
            "Historical 5-Year Study",
            "Live / Today Filter Scan",
        ]
    )

elif dashboard_mode == "Gold & Silver Decision Hub":
    st.subheader("Gold & Silver Decision Hub")

    st.markdown(
        "This hub helps you decide whether it is a relatively good time to "
        "buy more, hold, or trim Gold and Silver ETFs, based on technicals "
        "and key macro/structural drivers."
    )

    with st.spinner("Loading metals data..."):
        metals = fetch_metals_data()

    gold_tab, silver_tab = st.tabs(["Gold", "Silver"])

    # -------------------------
    # GOLD TAB
    # -------------------------
    with gold_tab:
        st.markdown("### Gold (Nippon Gold BeES + Global Benchmarks)")

        gold_etf = metals.get("gold_etf", {})
        gold_futures = metals.get("gold_futures", {})
        usd_index = metals.get("usd_index", {})

        gold_etf_price = gold_etf.get("latest_price")
        gold_futures_price = gold_futures.get("latest_price")

        g1, g2 = st.columns(2)

        g1.metric(
            "Nippon Gold BeES (GOLDBEES.NS)",
            format_price(gold_etf_price) if gold_etf_price else "Not available",
        )

        g2.metric(
            "COMEX Gold Futures (GC=F)",
            f"${gold_futures_price:,.2f}" if gold_futures_price else "Not available",
        )

        gold_outlook = calculate_metals_outlook(
            etf_data=gold_etf,
            futures_data=gold_futures,
            usd_data=usd_index,
            metal_type="gold",
        )

        st.divider()

        o1, o2 = st.columns(2)

        o1.metric(
            "Outlook (1–3 Months)",
            gold_outlook["outlook"],
        )

        o2.metric(
            "Suggested Stance",
            gold_outlook["stance"],
        )

        if gold_outlook["reasons"]:
            st.markdown("### Key Drivers")
            for reason in gold_outlook["reasons"]:
                st.write(f"- {reason}")

        st.divider()

        st.markdown("### Buying Zones (Guidance Only)")

        if gold_etf.get("data") is not None and not gold_etf["data"].empty:
            support, resistance = calculate_support_resistance(gold_etf["data"])
            atr = calculate_atr(gold_etf["data"], 14)

            z1, z2, z3 = st.columns(3)

            z1.metric(
                "Support",
                format_price(support),
            )

            z2.metric(
                "Resistance",
                format_price(resistance),
            )

            if atr is not None and gold_etf_price is not None:
                weak_zone = gold_etf_price - 1.5 * atr
                strong_zone = gold_etf_price - 3 * atr

                z3.metric(
                    "Approx. Weak Buy Zone",
                    format_price(weak_zone),
                )

                st.caption(
                    f"Stronger buy zone around {format_price(strong_zone)} "
                    "(more volatile, use discretion)."
                )
        else:
            st.info("ETF data not available for zone calculation.")

        st.caption(
            "This is a rule-based, technical + macro summary. It is not "
            "investment advice and does not guarantee future returns."
        )

    # -------------------------
    # SILVER TAB
    # -------------------------
    with silver_tab:
        st.markdown("### Silver (Nippon Silver ETF + Global Benchmarks)")

        silver_etf = metals.get("silver_etf", {})
        silver_futures = metals.get("silver_futures", {})
        usd_index = metals.get("usd_index", {})

        silver_etf_price = silver_etf.get("latest_price")
        silver_futures_price = silver_futures.get("latest_price")

        s1, s2 = st.columns(2)

        s1.metric(
            "Nippon Silver ETF (SILVERBEES.NS)",
            format_price(silver_etf_price) if silver_etf_price else "Not available",
        )

        s2.metric(
            "COMEX Silver Futures (SI=F)",
            f"${silver_futures_price:,.2f}" if silver_futures_price else "Not available",
        )

        silver_outlook = calculate_metals_outlook(
            etf_data=silver_etf,
            futures_data=silver_futures,
            usd_data=usd_index,
            metal_type="silver",
        )

        st.divider()

        o1, o2 = st.columns(2)

        o1.metric(
            "Outlook (1–3 Months)",
            silver_outlook["outlook"],
        )

        o2.metric(
            "Suggested Stance",
            silver_outlook["stance"],
        )

        if silver_outlook["reasons"]:
            st.markdown("### Key Drivers")
            for reason in silver_outlook["reasons"]:
                st.write(f"- {reason}")

        st.divider()

        st.markdown("### Buying Zones (Guidance Only)")

        if silver_etf.get("data") is not None and not silver_etf["data"].empty:
            support, resistance = calculate_support_resistance(silver_etf["data"])
            atr = calculate_atr(silver_etf["data"], 14)

            z1, z2, z3 = st.columns(3)

            z1.metric(
                "Support",
                format_price(support),
            )

            z2.metric(
                "Resistance",
                format_price(resistance),
            )

            if atr is not None and silver_etf_price is not None:
                weak_zone = silver_etf_price - 1.5 * atr
                strong_zone = silver_etf_price - 3 * atr

                z3.metric(
                    "Approx. Weak Buy Zone",
                    format_price(weak_zone),
                )

                st.caption(
                    f"Stronger buy zone around {format_price(strong_zone)} "
                    "(more volatile, use discretion)."
                )
        else:
            st.info("ETF data not available for zone calculation.")

        st.caption(
            "This is a rule-based, technical + macro summary. It is not "
            "investment advice and does not guarantee future returns."
        )

    
    # =========================================================================
    # HISTORICAL TAB — PENDING-ENTRY SUPPORT
    # =========================================================================

    with historical_tab:
        st.markdown(
            """
            ### Historical 5-Year Technical Filter Study

            This is the existing closing-price study, now with pending-entry support.

            - Historical filters use completed daily candles only.
            - Signal is generated only after daily close.
            - Entry is on next available trading-day adjusted close.
            - Every repeated daily signal is included.
            - Incomplete future return windows display as `NA`.
            - The latest qualifying signal appears with pending entry status.
            """
        )

        st.markdown(
            """
            | Historical Condition | Status |
            |---|---|
            | Daily Volume > 1.5 × 5D Volume SMA | Evaluated |
            | Daily candle return > 2% | Evaluated |
            | Daily Close > 20 DMA | Evaluated |
            | Daily Close > 200 DMA | Evaluated |
            | Weekly Close > Weekly SMA20 | Evaluated |
            | ATR14 / Daily Close > 3% | Evaluated |
            | Daily RSI14 between 55 and 90 | Evaluated |
            | Weekly RSI14 >= 55 | Evaluated |
            | Daily Close > previous-day Close | Evaluated |
            | Daily Close > two-days-ago Close | Evaluated |
            | Historical Debt/Equity < 1 | Not evaluated in Option B |
            | Historical EPS > 0 | Not evaluated in Option B |
            | Historical ROE > 10% | Not evaluated in Option B |
            """
        )

        historical_scan_limit = st.selectbox(
            "Stocks to evaluate",
            [
                20,
                50,
                100,
                250,
                500,
                750,
            ],
            index=1,
            key="historical_scan_limit",
        )

        st.info(
            "Every qualifying day is retained. If a stock passes on "
            "three consecutive trading days, all three dates appear as "
            "separate historical records. The latest signal shows "
            "Pending Next Trading Day with NA entry price."
        )

        if st.button(
            "▶ Run Historical 5-Year Study",
            type="primary",
        ):
            progress_bar = st.progress(0)
            progress_text = st.empty()

            def historical_progress(
                position,
                total,
                symbol,
            ):
                progress_bar.progress(
                    position / total
                )

                progress_text.caption(
                    f"Evaluating {position:,} of {total:,}: {symbol}"
                )

            results = run_optimized_historical_filter_study(
                stock_universe,
                minimum_market_cap,
                historical_scan_limit,
                historical_progress,
            )

            progress_bar.empty()
            progress_text.empty()

            st.session_state[
                "historical_filter_results"
            ] = results

            st.session_state[
                "historical_filter_settings"
            ] = {
                "minimum_market_cap": minimum_market_cap,
                "scan_limit": historical_scan_limit,
            }

        historical_results = st.session_state[
            "historical_filter_results"
        ]

        if historical_results is None or historical_results.empty:
            st.info(
                "Click Run Historical 5-Year Study to generate results."
            )

        else:
            summary = calculate_historical_study_summary(
                historical_results
            )

            settings = st.session_state[
                "historical_filter_settings"
            ]

            st.markdown(
                f"""
                <div class="research-card blue-card">
                    <h3>Historical Study Configuration</h3>
                    <p>
                        Current Market Cap Eligibility:
                        Above ₹{settings.get("minimum_market_cap", 3000):,.0f} Cr |
                        Stocks Evaluated:
                        {settings.get("scan_limit", 50)} |
                        Signal Counting: Every qualifying day |
                        Signal Gap: 0 trading days
                    </p>
                    <p>
                        Entry: Next available trading-day adjusted close |
                        Historical Debt/Equity, EPS and ROE: Not evaluated
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            h1, h2, h3, h4, h5, h6 = st.columns(6)

            h1.metric(
                "Total Signal Records",
                summary["total_signal_records"],
            )

            h2.metric(
                "Unique Stocks",
                summary["unique_stocks"],
            )

            h3.metric(
                "Unique Stock-Months",
                summary["unique_stock_months"],
            )

            h4.metric(
                "Unique Stock-Quarters",
                summary["unique_stock_quarters"],
            )

            h5.metric(
                "Consecutive Signal Records",
                summary["consecutive_signal_records"],
            )

            h6.metric(
                "Pending Entries",
                summary["pending_entries"],
            )

            st.subheader(
                "Forward Outcome Summary"
            )

            st.dataframe(
                summary["horizon_summary"],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Average Max Return %": st.column_config.NumberColumn(
                        "Average Max Return",
                        format="%.2f%%",
                    ),
                    "Median Max Return %": st.column_config.NumberColumn(
                        "Median Max Return",
                        format="%.2f%%",
                    ),
                    "Average Min Return %": st.column_config.NumberColumn(
                        "Average Min Return",
                        format="%.2f%%",
                    ),
                    "Median Min Return %": st.column_config.NumberColumn(
                        "Median Min Return",
                        format="%.2f%%",
                    ),
                    "10% Hit Rate %": st.column_config.NumberColumn(
                        "10% Hit Rate",
                        format="%.2f%%",
                    ),
                },
            )

            st.subheader(
                "Historical Signals by Year"
            )

            if not summary["year_summary"].empty:
                st.dataframe(
                    summary["year_summary"],
                    hide_index=True,
                    use_container_width=True,
                )

            st.subheader(
                "Individual Historical Signal Records"
            )

            historical_display_columns = [
                "stock_symbol",
                "stock_name",
                "signal_date",
                "signal_day_price",
                "entry_date",
                "entry_price",
                "entry_status",
                "entry_price_status",
                "daily_candle_return_pct",
                "volume_ratio_vs_5d_sma",
                "daily_rsi_14",
                "weekly_rsi_14",
                "atr_14_pct_of_close",
                "max_ret_5d",
                "hit_10pct_5d",
                "max_ret_10d",
                "hit_10pct_10d",
                "max_ret_20d",
                "hit_10pct_20d",
                "max_ret_30d",
                "min_ret_30d",
                "hit_10pct_30d",
                "max_ret_60d",
                "min_ret_60d",
                "hit_10pct_60d",
                "max_ret_90d",
                "min_ret_90d",
                "hit_10pct_90d",
                "max_ret_6m",
                "min_ret_6m",
                "hit_10pct_6m",
                "max_ret_9m",
                "hit_10pct_9m",
                "max_ret_12m",
                "min_ret_12m",
                "hit_10pct_12m",
            ]

            st.dataframe(
                historical_results[
                    historical_display_columns
                ],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "signal_date": st.column_config.DateColumn(
                        "Signal Date"
                    ),
                    "entry_date": st.column_config.DateColumn(
                        "Entry Date"
                    ),
                    "signal_day_price": st.column_config.NumberColumn(
                        "Signal-Day Price",
                        format="₹%.2f",
                    ),
                    "entry_price": st.column_config.NumberColumn(
                        "Entry Price",
                        format="₹%.2f",
                    ),
                },
            )

            historical_csv = historical_results.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Download Historical Study CSV",
                data=historical_csv,
                file_name="nifty_historical_filter_study.csv",
                mime="text/csv",
            )

    # =========================================================================
    # LIVE / TODAY FILTER SCAN — UNCHANGED INTRADAY LOGIC
    # =========================================================================

    with live_tab:
        st.markdown(
            """
            ### Live / Today Filter Scan

            This is a separate current-day scan.

            - **Market Open:** latest intraday price and cumulative intraday
              volume are used. Results are provisional.
            - **Market Closed:** latest daily close and final daily volume
              are used. Results are end-of-day confirmed.
            - The Historical 5-Year Study above is not affected by this tab.
            """
        )

        market_status = get_market_status()

        if market_status["is_open"]:
            live_card_class = "orange-card"
            live_title = "⚠️ PROVISIONAL INTRADAY MODE"
            live_message = (
                "Current price and cumulative intraday volume are used. "
                "A stock can pass now and fail by market close."
            )
        else:
            live_card_class = "green-card"
            live_title = "✅ END-OF-DAY CONFIRMED MODE"
            live_message = (
                "Latest daily close and final daily volume are used."
            )

        st.markdown(
            f"""
            <div class="research-card {live_card_class}">
                <h3>{live_title}</h3>
                <p>
                    Market Status: {market_status["status"]} |
                    Current IST Time:
                    {market_status["now"].strftime("%Y-%m-%d %H:%M:%S")}
                </p>
                <p>{live_message}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        live_col1, live_col2 = st.columns(2)

        with live_col1:
            live_scan_limit = st.selectbox(
                "Stocks to evaluate for Live Scan",
                [
                    20,
                    50,
                    100,
                    250,
                    500,
                    750,
                ],
                index=1,
                key="live_scan_limit",
            )

        with live_col2:
            auto_info = (
                "Manual refresh is recommended. "
                "Yahoo Finance intraday data is cached for 60 seconds."
            )

            st.info(auto_info)

        st.warning(
            "Intraday Yahoo Finance data can be delayed, incomplete, "
            "or unavailable for some NSE stocks. If intraday data cannot "
            "be fetched, the scan falls back to the latest daily data and "
            "labels the result clearly."
        )

        if st.button(
            "⚡ Run / Refresh Live Filter Scan",
            type="primary",
        ):
            progress_bar = st.progress(0)
            progress_text = st.empty()

            def live_progress(
                position,
                total,
                symbol,
            ):
                progress_bar.progress(
                    position / total
                )

                progress_text.caption(
                    f"Checking {position:,} of {total:,}: {symbol}"
                )

            live_results = run_live_filter_scan(
                universe=stock_universe,
                minimum_market_cap=minimum_market_cap,
                scan_limit=live_scan_limit,
                market_open=market_status["is_open"],
                progress_callback=live_progress,
            )

            progress_bar.empty()
            progress_text.empty()

            st.session_state[
                "live_filter_results"
            ] = live_results

            st.session_state[
                "live_filter_metadata"
            ] = {
                "market_status": market_status["status"],
                "scan_timestamp": market_status["now"],
                "scan_limit": live_scan_limit,
                "market_open": market_status["is_open"],
            }

        live_results = st.session_state[
            "live_filter_results"
        ]

        live_metadata = st.session_state[
            "live_filter_metadata"
        ]

        if live_results is None or live_results.empty:
            st.info(
                "Click Run / Refresh Live Filter Scan to find "
                "current stocks passing the technical filter."
            )

        else:
            passed_results = live_results[
                live_results["all_filters_pass"]
            ].copy()

            total_checked = len(live_results)
            total_passed = len(passed_results)

            latest_scan_time = live_metadata.get(
                "scan_timestamp"
            )

            live_summary_col1, live_summary_col2, live_summary_col3, live_summary_col4 = (
                st.columns(4)
            )

            live_summary_col1.metric(
                "Stocks Checked",
                total_checked,
            )

            live_summary_col2.metric(
                "Stocks Passing All Filters",
                total_passed,
            )

            live_summary_col3.metric(
                "Market Status",
                live_metadata.get(
                    "market_status",
                    "Unknown",
                ),
            )

            live_summary_col4.metric(
                "Scan Time IST",
                (
                    latest_scan_time.strftime(
                        "%H:%M:%S"
                    )
                    if latest_scan_time is not None
                    else "Not available"
                ),
            )

            if market_status["is_open"]:
                st.warning(
                    "Passing stocks are provisional intraday signals. "
                    "Re-run or verify after market close before treating "
                    "them as completed daily signals."
                )
            else:
                st.success(
                    "The scan is using latest daily end-of-day values."
                )

            st.subheader(
                "Stocks Passing All Live / Today Filter Conditions"
            )

            if passed_results.empty:
                st.info(
                    "No stocks currently pass every required technical condition."
                )
            else:
                live_display_columns = [
                    "stock_symbol",
                    "stock_name",
                    "industry",
                    "market_status",
                    "data_mode",
                    "signal_status",
                    "data_source",
                    "signal_timestamp_ist",
                    "latest_price",
                    "today_open",
                    "today_high",
                    "today_low",
                    "current_day_change_pct",
                    "current_volume",
                    "prior_5d_average_volume",
                    "volume_ratio_vs_prior_5d",
                    "daily_sma_20",
                    "daily_sma_200",
                    "weekly_close_or_live_price",
                    "weekly_sma_20",
                    "atr_14_pct",
                    "daily_rsi_14",
                    "weekly_rsi_14",
                    "current_debt_equity_reference",
                    "current_eps_reference",
                    "current_roe_reference",
                ]

                st.dataframe(
                    passed_results[
                        live_display_columns
                    ],
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "signal_timestamp_ist": st.column_config.DatetimeColumn(
                            "Latest Data Timestamp",
                            format="YYYY-MM-DD HH:mm:ss",
                        ),
                        "latest_price": st.column_config.NumberColumn(
                            "Latest Price",
                            format="₹%.2f",
                        ),
                        "today_open": st.column_config.NumberColumn(
                            "Today Open",
                            format="₹%.2f",
                        ),
                        "today_high": st.column_config.NumberColumn(
                            "Today High",
                            format="₹%.2f",
                        ),
                        "today_low": st.column_config.NumberColumn(
                            "Today Low",
                            format="₹%.2f",
                        ),
                        "current_day_change_pct": st.column_config.NumberColumn(
                            "Day Change %",
                            format="%.2f%%",
                        ),
                        "current_volume": st.column_config.NumberColumn(
                            "Current Volume",
                            format="%.0f",
                        ),
                        "prior_5d_average_volume": st.column_config.NumberColumn(
                            "Prior 5D Avg Volume",
                            format="%.0f",
                        ),
                        "volume_ratio_vs_prior_5d": st.column_config.NumberColumn(
                            "Volume / Prior 5D",
                            format="%.2fx",
                        ),
                        "daily_sma_20": st.column_config.NumberColumn(
                            "Daily SMA20",
                            format="₹%.2f",
                        ),
                        "daily_sma_200": st.column_config.NumberColumn(
                            "Daily SMA200",
                            format="₹%.2f",
                        ),
                        "weekly_close_or_live_price": st.column_config.NumberColumn(
                            "Weekly Close / Live Price",
                            format="₹%.2f",
                        ),
                        "weekly_sma_20": st.column_config.NumberColumn(
                            "Weekly SMA20",
                            format="₹%.2f",
                        ),
                        "atr_14_pct": st.column_config.NumberColumn(
                            "ATR %",
                            format="%.2f%%",
                        ),
                        "daily_rsi_14": st.column_config.NumberColumn(
                            "Daily RSI14",
                            format="%.2f",
                        ),
                        "weekly_rsi_14": st.column_config.NumberColumn(
                            "Weekly RSI14",
                            format="%.2f",
                        ),
                        "current_roe_reference": st.column_config.NumberColumn(
                            "Current ROE",
                            format="%.2f",
                        ),
                    },
                )

                live_csv = passed_results.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "⬇️ Download Passing Live Filter Stocks CSV",
                    data=live_csv,
                    file_name="nifty_live_today_filter_scan.csv",
                    mime="text/csv",
                )

                passing_symbols = passed_results[
                    "stock_symbol"
                ].tolist()

                selected_live_symbol = st.selectbox(
                    "Open Stock Research",
                    passing_symbols,
                    key="live_open_stock_symbol",
                )

                if st.button(
                    "Open Selected Stock Research",
                    key="live_open_stock_button",
                ):
                    open_stock_research(
                        selected_live_symbol
                    )
                    st.rerun()

            st.divider()

            st.subheader(
                "Live Filter Audit — All Evaluated Stocks"
            )

            st.caption(
                "Use this table to understand why a stock did not pass."
            )

            audit_columns = [
                "stock_symbol",
                "stock_name",
                "data_mode",
                "latest_price",
                "current_day_change_pct",
                "volume_ratio_vs_prior_5d",
                "daily_rsi_14",
                "weekly_rsi_14",
                "atr_14_pct",
                "volume_condition",
                "candle_change_condition",
                "close_above_sma20",
                "close_above_sma200",
                "weekly_above_sma20",
                "atr_condition",
                "daily_rsi_condition",
                "weekly_rsi_condition",
                "close_above_1d",
                "close_above_2d",
                "all_filters_pass",
            ]

            st.dataframe(
                live_results[
                    audit_columns
                ],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "latest_price": st.column_config.NumberColumn(
                        "Latest Price",
                        format="₹%.2f",
                    ),
                    "current_day_change_pct": st.column_config.NumberColumn(
                        "Day Change %",
                        format="%.2f%%",
                    ),
                    "volume_ratio_vs_prior_5d": st.column_config.NumberColumn(
                        "Volume / Prior 5D",
                        format="%.2fx",
                    ),
                    "daily_rsi_14": st.column_config.NumberColumn(
                        "Daily RSI",
                        format="%.2f",
                    ),
                    "weekly_rsi_14": st.column_config.NumberColumn(
                        "Weekly RSI",
                        format="%.2f",
                    ),
                    "atr_14_pct": st.column_config.NumberColumn(
                        "ATR %",
                        format="%.2f%%",
                    ),
                },
            )


# =============================================================================
# FOOTER
# =============================================================================

st.divider()

st.caption(
    "Data sources: Nifty Indices constituent list and Yahoo Finance. "
    "Historical Filter Study uses completed daily closing-price data only "
    "and now includes pending entries for the latest signal. Live intraday results "
    "are provisional during market hours and may change before close. "
    "All indicators, patterns, rankings and filter results are for research "
    "only and are not investment advice."
)
