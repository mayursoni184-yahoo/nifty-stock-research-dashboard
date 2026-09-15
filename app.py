import io
import warnings

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import yfinance as yf

from plotly.subplots import make_subplots
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


warnings.filterwarnings("ignore")


# =============================================================================
# STREAMLIT CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Nifty Total Market Winner Dashboard",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# APPLICATION STYLE
# =============================================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.35rem;
            font-weight: 800;
            color: #111827;
            margin-bottom: 0.10rem;
        }

        .sub-title {
            font-size: 1rem;
            color: #6b7280;
            margin-bottom: 1.20rem;
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
    "Stock Probability Research",
    "Phase 3 Model Builder",
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

TARGET_OPTIONS = [5, 10, 15]

STOP_OPTIONS = [5, 8, 10, 12]

HORIZON_OPTIONS = [40, 60, 90]

MODEL_FEATURES = [
    "daily_trend_score",
    "weekly_trend_score",
    "rs_3m",
    "distance_52w_high",
    "volume_change",
    "atr_percent",
    "momentum_1m",
    "distance_ma50",
]


# =============================================================================
# SESSION STATE
# =============================================================================

if "selected_symbol" not in st.session_state:
    st.session_state["selected_symbol"] = "RELIANCE"

if "command_center_results" not in st.session_state:
    st.session_state["command_center_results"] = pd.DataFrame()

if "phase1_events" not in st.session_state:
    st.session_state["phase1_events"] = pd.DataFrame()

if "phase1_symbol" not in st.session_state:
    st.session_state["phase1_symbol"] = None

if "phase1_settings" not in st.session_state:
    st.session_state["phase1_settings"] = {}

if "phase2_events" not in st.session_state:
    st.session_state["phase2_events"] = pd.DataFrame()

if "phase2_symbol" not in st.session_state:
    st.session_state["phase2_symbol"] = None

if "phase3_dataset" not in st.session_state:
    st.session_state["phase3_dataset"] = pd.DataFrame()

if "phase3_model_package" not in st.session_state:
    st.session_state["phase3_model_package"] = None

if "phase3_settings" not in st.session_state:
    st.session_state["phase3_settings"] = {}


# =============================================================================
# GENERAL UTILITY FUNCTIONS
# =============================================================================

def safe_number(value, default=None):
    """Safely convert values to float."""

    try:
        if value is None:
            return default

        if pd.isna(value):
            return default

        return float(value)

    except Exception:
        return default


def format_price(value):
    """Format INR price."""

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
    """Save selected stock and set requested navigation destination."""

    st.session_state["selected_symbol"] = symbol
    st.session_state["requested_mode"] = "Stock Research"


def calculate_percent_change(current_value, previous_value):
    """Calculate percentage change safely."""

    current_value = safe_number(current_value)
    previous_value = safe_number(previous_value)

    if current_value is None:
        return None

    if previous_value is None:
        return None

    if previous_value == 0:
        return None

    return (
        (current_value - previous_value)
        / abs(previous_value)
    ) * 100


def probability_label(probability):
    """Convert model probability to a readable label."""

    probability = safe_number(probability)

    if probability is None:
        return "Insufficient data"

    if probability >= 0.70:
        return "High historical probability"

    if probability >= 0.60:
        return "Favourable historical probability"

    if probability >= 0.50:
        return "Slightly favourable"

    if probability >= 0.40:
        return "Neutral / uncertain"

    return "Low historical probability"


def sample_confidence(sample_size):
    """Classify model/backtest sample confidence."""

    if sample_size >= 400:
        return "Higher sample confidence"

    if sample_size >= 150:
        return "Medium"

    if sample_size >= 50:
        return "Low"

    return "Insufficient sample"


# =============================================================================
# NIFTY TOTAL MARKET UNIVERSE
# =============================================================================

@st.cache_data(ttl=21600, show_spinner=False)
def get_nifty_total_market_members():
    """
    Load current Nifty Total Market constituents.

    Uses a fallback stock universe if the official constituent CSV
    is unavailable from Streamlit Cloud.
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

        universe = pd.read_csv(
            io.BytesIO(response.content)
        )

        universe.columns = [
            str(column).strip()
            for column in universe.columns
        ]

        if "Symbol" not in universe.columns:
            raise ValueError(
                "Constituent data has no Symbol column."
            )

        if "Series" in universe.columns:
            universe = universe[
                universe["Series"]
                .astype(str)
                .str.upper()
                .eq("EQ")
            ].copy()

        universe["Symbol"] = (
            universe["Symbol"]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        universe["Ticker"] = (
            universe["Symbol"] + ".NS"
        )

        if "Company Name" not in universe.columns:
            universe["Company Name"] = universe["Symbol"]

        if "Industry" not in universe.columns:
            universe["Industry"] = "Unknown"

        universe = universe[
            [
                "Company Name",
                "Industry",
                "Symbol",
                "Ticker",
            ]
        ]

        universe = (
            universe
            .drop_duplicates("Symbol")
            .reset_index(drop=True)
        )

        if len(universe) < 100:
            raise ValueError(
                "Too few valid Nifty Total Market constituents received."
            )

        return universe, None

    except Exception as error:
        fallback_universe = pd.DataFrame(
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

        message = (
            "Live Nifty Total Market constituents could not be loaded. "
            f"Using fallback stocks. Reason: {type(error).__name__}: {error}"
        )

        return fallback_universe, message


# =============================================================================
# MARKET DATA FUNCTIONS
# =============================================================================

@st.cache_data(ttl=900, show_spinner=False)
def fetch_price_data(ticker, period="5y"):
    """Fetch OHLCV price data from Yahoo Finance."""

    try:
        ticker_object = yf.Ticker(ticker)

        price_data = ticker_object.history(
            period=period,
            auto_adjust=True,
        )

        if price_data is None or price_data.empty:
            return pd.DataFrame()

        price_data = price_data.rename(
            columns=str.lower
        )

        price_data = price_data[
            [
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        ].dropna()

        price_data.index = pd.to_datetime(
            price_data.index
        )

        if getattr(price_data.index, "tz", None) is not None:
            price_data.index = (
                price_data.index.tz_localize(None)
            )

        return price_data

    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=43200, show_spinner=False)
def fetch_fundamentals(ticker):
    """Fetch basic fundamental metrics from Yahoo Finance."""

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
        "fiftyTwoWeekHigh",
        "fiftyTwoWeekLow",
    ]

    try:
        ticker_object = yf.Ticker(ticker)
        stock_info = ticker_object.get_info()

        return {
            field: stock_info.get(field)
            for field in fields
        }

    except Exception:
        return {}


# =============================================================================
# TIMEFRAME AND TREND FUNCTIONS
# =============================================================================

def resample_ohlcv(data, timeframe):
    """Resample daily OHLCV into weekly or monthly candles."""

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
    """
    Determine price trend.

    Strong bullish:
    Close > MA20 > MA50 > MA200.

    Bullish:
    Close > MA20 > MA50.

    Bearish:
    Close < MA20 < MA50.
    """

    if data is None or len(data) < 55:
        return "Insufficient data"

    latest_close = float(
        data["close"].iloc[-1]
    )

    ma20 = float(
        data["close"]
        .rolling(20)
        .mean()
        .iloc[-1]
    )

    ma50 = float(
        data["close"]
        .rolling(50)
        .mean()
        .iloc[-1]
    )

    if len(data) >= 200:
        ma200 = float(
            data["close"]
            .rolling(200)
            .mean()
            .iloc[-1]
        )

        if latest_close > ma20 > ma50 > ma200:
            return "Strong bullish"

    if latest_close > ma20 > ma50:
        return "Bullish"

    if latest_close < ma20 < ma50:
        return "Bearish"

    return "Neutral / consolidating"


def trend_to_numeric(trend):
    """Convert trend label to model-friendly numeric score."""

    mapping = {
        "Strong bullish": 3,
        "Bullish": 2,
        "Neutral / consolidating": 1,
        "Bearish": 0,
        "Insufficient data": 0,
    }

    return mapping.get(trend, 0)


def calculate_multitimeframe_alignment(
    daily_data,
    weekly_data,
    monthly_data,
):
    """Calculate Daily/Weekly/Monthly alignment score out of 10."""

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

        structure = (
            "Daily, Weekly and Monthly trends align upward."
        )

    elif bullish_count >= 2:
        status = "Bullish multi-timeframe alignment"

        structure = (
            "Most chart timeframes are bullish."
        )

    elif bearish_count >= 2:
        status = "Bearish multi-timeframe alignment"

        structure = (
            "Most chart timeframes are bearish."
        )

    else:
        status = "Mixed timeframe alignment"

        structure = (
            "Daily, Weekly and Monthly trends are mixed."
        )

    return {
        "daily": daily_trend,
        "weekly": weekly_trend,
        "monthly": monthly_trend,
        "score": score,
        "status": status,
        "structure": structure,
    }


# =============================================================================
# RELATIVE STRENGTH ENGINE
# =============================================================================

def align_stock_benchmark(stock_data, benchmark_data):
    """Align stock and benchmark series to common dates."""

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

    aligned = stock_close.join(
        benchmark_close,
        how="inner",
    )

    return aligned.dropna()


def calculate_period_return(close_series, days):
    """Calculate return over an approximate trading-day period."""

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
    """
    Calculate return and relative strength versus Nifty 50.

    Relative Return = Stock Return - Nifty 50 Return.
    """

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
        latest_rs = float(rs_line.iloc[-1])
        old_rs = float(rs_line.iloc[-63])

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
        rs_status = "Insufficient data"

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
            rs_status = "Leader"

        elif (
            positive_count >= 2
            and rs_trend in [
                "Rising",
                "Flat",
            ]
        ):
            rs_status = "Strong"

        elif negative_count >= 3:
            rs_status = "Weak"

        else:
            rs_status = "Neutral"

    return {
        "status": rs_status,
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
# TECHNICAL FEATURES
# =============================================================================

def calculate_atr(data, period=14):
    """Calculate Average True Range."""

    if data is None or len(data) < period + 1:
        return None

    high = data["high"]
    low = data["low"]
    close = data["close"]

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = true_range.rolling(period).mean()

    atr_value = atr.iloc[-1]

    if pd.isna(atr_value):
        return None

    return float(atr_value)


def calculate_support_resistance(data):
    """Calculate recent support and resistance using local pivots."""

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
        "low",
        3,
    )

    high_pivots = find_pivots(
        high_values,
        "high",
        3,
    )

    supports = [
        float(data["low"].iloc[index])
        for index in low_pivots
        if data["low"].iloc[index] < current_price
    ]

    resistances = [
        float(data["high"].iloc[index])
        for index in high_pivots
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


def calculate_current_price_features(
    stock_data,
    benchmark_data,
):
    """
    Calculate current price-derived model features.

    These features contain only information observable on the current date.
    """

    if stock_data.empty:
        return {}

    daily_trend = calculate_overall_trend(
        stock_data
    )

    weekly_data = resample_ohlcv(
        stock_data,
        "Weekly",
    )

    weekly_trend = calculate_overall_trend(
        weekly_data
    )

    rs = calculate_relative_strength(
        stock_data,
        benchmark_data,
    )

    current_close = float(
        stock_data["close"].iloc[-1]
    )

    lookback = min(
        len(stock_data),
        252,
    )

    high_52w = float(
        stock_data["high"]
        .tail(lookback)
        .max()
    )

    distance_52w_high = (
        current_close / high_52w - 1
    ) * 100

    ma50 = (
        float(
            stock_data["close"]
            .rolling(50)
            .mean()
            .iloc[-1]
        )
        if len(stock_data) >= 50
        else None
    )

    distance_ma50 = (
        (
            current_close / ma50 - 1
        ) * 100
        if ma50 is not None
        and ma50 != 0
        else None
    )

    volume_change = None

    if len(stock_data) >= 21:
        average_volume = float(
            stock_data["volume"]
            .tail(21)
            .iloc[:-1]
            .mean()
        )

        latest_volume = float(
            stock_data["volume"].iloc[-1]
        )

        if average_volume != 0:
            volume_change = (
                latest_volume / average_volume - 1
            ) * 100

    atr = calculate_atr(
        stock_data,
        14,
    )

    atr_percent = (
        atr / current_close * 100
        if atr is not None
        and current_close != 0
        else None
    )

    momentum_1m = calculate_period_return(
        stock_data["close"],
        21,
    )

    return {
        "daily_trend_score": trend_to_numeric(
            daily_trend
        ),
        "weekly_trend_score": trend_to_numeric(
            weekly_trend
        ),
        "rs_3m": rs.get("relative_3m"),
        "distance_52w_high": distance_52w_high,
        "volume_change": volume_change,
        "atr_percent": atr_percent,
        "momentum_1m": momentum_1m,
        "distance_ma50": distance_ma50,
        "daily_trend_label": daily_trend,
        "weekly_trend_label": weekly_trend,
        "rs_status": rs.get("status"),
        "rs_trend": rs.get("rs_trend"),
    }


# =============================================================================
# PATTERN DETECTION
# =============================================================================

def find_pivots(values, pivot_type="high", order=3):
    """Find local high/low pivot indexes."""

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


def build_pattern_signal(
    pattern,
    status,
    data,
    level,
    direction,
    notes,
    pattern_height=None,
):
    """Build a normalized technical pattern row."""

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
    """Detect active technical patterns using rule-based conditions."""

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

            pattern_height = peak - neckline

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
                    "Two comparable peaks; confirmation is below neckline.",
                    pattern_height,
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

            pattern_height = neckline - base

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
                    pattern_height,
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
            pattern_height = (
                high[head] - neckline
            )

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
                    pattern_height,
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
            pattern_height = (
                neckline - low[head]
            )

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
                    pattern_height,
                )
            )

    # RECTANGLE / TRIANGLE
    pattern_window = 30

    if len(data) >= pattern_window:
        recent_highs = high[-pattern_window:]
        recent_lows = low[-pattern_window:]

        x_values = np.arange(
            pattern_window
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

        pattern_height = (
            resistance - support
        )

        range_size = (
            pattern_height
            / max(resistance, 1)
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

                    if pattern_name == "Descending Triangle":
                        level = support
                    else:
                        level = resistance

                patterns.append(
                    build_pattern_signal(
                        pattern_name,
                        status,
                        data,
                        level,
                        direction,
                        "Confirmation requires close outside pattern range.",
                        pattern_height,
                    )
                )

    # REVERSAL BOTTOM / TOP
    latest_open = float(open_price[-1])
    latest_close = float(close[-1])
    latest_high = float(high[-1])
    latest_low = float(low[-1])

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
# RISK/REWARD FUNCTIONS
# =============================================================================

def calculate_pattern_target(pattern):
    """Calculate measured-move pattern target."""

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
    """Classify whether price is extended from breakout."""

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
        distance_percent = (
            (current_price - breakout_level)
            / breakout_level
        ) * 100

    elif direction == "Bearish":
        distance_percent = (
            (breakout_level - current_price)
            / breakout_level
        ) * 100

    else:
        distance_percent = (
            abs(current_price - breakout_level)
            / breakout_level
        ) * 100

    if distance_percent <= 0:
        status = "Below / Near Breakout"

    elif distance_percent <= 3:
        status = "Ideal Entry Zone"

    elif distance_percent <= 7:
        status = "Acceptable Entry Zone"

    elif distance_percent <= 12:
        status = "Extended"

    else:
        status = "Avoid Chasing"

    return {
        "status": status,
        "distance_percent": distance_percent,
    }


def calculate_trade_plan(
    data,
    pattern,
):
    """Create a rule-based trade-plan research output."""

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

    entry_quality = classify_entry_quality(
        current_price,
        breakout_level,
        direction,
    )

    target = calculate_pattern_target(
        pattern
    )

    if direction == "Bullish":
        technical_stop = support

        atr_stop = (
            current_price - 2 * atr
            if atr is not None
            else None
        )

        valid_stops = [
            value
            for value in [
                technical_stop,
                atr_stop,
            ]
            if value is not None
            and value < current_price
        ]

        selected_stop = (
            min(valid_stops)
            if valid_stops
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

        valid_stops = [
            value
            for value in [
                technical_stop,
                atr_stop,
            ]
            if value is not None
            and value > current_price
        ]

        selected_stop = (
            max(valid_stops)
            if valid_stops
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
            reward_per_share
            / risk_per_share
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
    """Calculate quantity using maximum acceptable portfolio loss."""

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

    position_value = quantity * entry_price

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
    """Score relative strength out of 20."""

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
        positives.append("Relative Strength status is Leader")

    elif status == "Strong":
        score += 7
        positives.append("Relative Strength status is Strong")

    elif status == "Neutral":
        score += 3

    elif status == "Weak":
        risks.append("Stock is underperforming Nifty 50")

    if rs_trend == "Rising":
        score += 4
        positives.append("Relative Strength line is rising")

    elif rs_trend == "Falling":
        risks.append("Relative Strength line is falling")

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
    """Score multi-timeframe alignment out of 20."""

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
            "Strong Daily, Weekly and Monthly alignment"
        )

    elif status == "Bullish multi-timeframe alignment":
        score += 15
        positives.append(
            "Bullish multi-timeframe alignment"
        )

    elif status == "Mixed timeframe alignment":
        score += 6
        risks.append("Timeframes are not fully aligned")

    elif status == "Bearish multi-timeframe alignment":
        risks.append("Most timeframes are bearish")

    return {
        "score": min(score, 20),
        "positives": positives,
        "risks": risks,
    }


def score_fundamentals(fundamentals):
    """Score currently available basic fundamental quality out of 20."""

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
    """Score technical trend structure out of 15."""

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
        positives.append("Strong Bullish Monthly trend")

    elif monthly == "Bullish":
        score += 4
        positives.append("Bullish Monthly trend")

    elif monthly == "Bearish":
        risks.append("Bearish Monthly trend")

    if weekly == "Strong bullish":
        score += 5
        positives.append("Strong Bullish Weekly trend")

    elif weekly == "Bullish":
        score += 3
        positives.append("Bullish Weekly trend")

    elif weekly == "Bearish":
        risks.append("Bearish Weekly trend")

    if daily == "Strong bullish":
        score += 4
        positives.append("Strong Bullish Daily trend")

    elif daily == "Bullish":
        score += 2
        positives.append("Bullish Daily trend")

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
    """Score bullish pattern/breakout evidence out of 15."""

    score = 0
    positives = []
    risks = []

    groups = [
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

    confirmed_bullish = False

    for timeframe, patterns in groups:
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
    """Score volume confirmation out of 5."""

    volume_values = []

    for pattern in daily_patterns + weekly_patterns:
        if (
            pattern.get("Status") == "Confirmed"
            and pattern.get("Direction") == "Bullish"
        ):
            volume = safe_number(
                pattern.get("Volume %")
            )

            if volume is not None:
                volume_values.append(volume)

    if not volume_values:
        return {
            "score": 0,
            "positives": [],
            "risks": [],
        }

    best_volume = max(volume_values)

    if best_volume >= 50:
        return {
            "score": 5,
            "positives": [
                f"Strong breakout volume: {best_volume:.1f}% above average"
            ],
            "risks": [],
        }

    if best_volume >= 25:
        return {
            "score": 4,
            "positives": [
                f"Good breakout volume: {best_volume:.1f}% above average"
            ],
            "risks": [],
        }

    if best_volume > 0:
        return {
            "score": 2,
            "positives": [
                f"Positive breakout volume: {best_volume:.1f}% above average"
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
    """Score basic valuation/leverage data out of 5."""

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
            positives.append("Low financial leverage")

        elif debt_equity > 250:
            risks.append("High financial leverage")

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
    """Calculate transparent 100-point Winner Score."""

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
# PHASE 1 HISTORICAL SETUP STUDY
# =============================================================================

def historical_features(
    stock_data,
    benchmark_data,
    index,
):
    """
    Calculate features using only information available on a past date.

    This avoids direct future-price leakage during the historical setup study.
    """

    if index < 260:
        return None

    stock_slice = stock_data.iloc[
        :index + 1
    ].copy()

    event_date = stock_slice.index[-1]

    benchmark_slice = benchmark_data[
        benchmark_data.index <= event_date
    ].copy()

    if benchmark_slice.empty:
        return None

    return calculate_current_price_features(
        stock_slice,
        benchmark_slice,
    )


def historical_setup_qualifies(
    features,
    require_strong_daily,
    require_weekly_bullish,
    require_positive_rs,
    max_distance_from_52w,
    minimum_volume_change,
):
    """Check whether historical setup passes Phase 1 conditions."""

    if features is None:
        return False

    daily_trend = features.get(
        "daily_trend_label"
    )

    weekly_trend = features.get(
        "weekly_trend_label"
    )

    rs_3m = safe_number(
        features.get("rs_3m")
    )

    distance_52w = safe_number(
        features.get("distance_52w_high")
    )

    volume_change = safe_number(
        features.get("volume_change")
    )

    if require_strong_daily:
        daily_ok = (
            daily_trend == "Strong bullish"
        )
    else:
        daily_ok = daily_trend in [
            "Strong bullish",
            "Bullish",
        ]

    if require_weekly_bullish:
        weekly_ok = weekly_trend in [
            "Strong bullish",
            "Bullish",
        ]
    else:
        weekly_ok = True

    if require_positive_rs:
        rs_ok = (
            rs_3m is not None
            and rs_3m > 0
        )
    else:
        rs_ok = True

    high_ok = (
        distance_52w is not None
        and distance_52w >= max_distance_from_52w
    )

    if minimum_volume_change is None:
        volume_ok = True
    else:
        volume_ok = (
            volume_change is not None
            and volume_change >= minimum_volume_change
        )

    return (
        daily_ok
        and weekly_ok
        and rs_ok
        and high_ok
        and volume_ok
    )


def evaluate_triple_barrier(
    stock_data,
    entry_index,
    target_percent,
    stop_percent,
    horizon_days,
):
    """
    Evaluate which is reached first:
    target, stop, or time horizon.

    When target and stop are both touched within one daily candle,
    the event is marked Ambiguous Same Day because daily OHLC data
    cannot establish intraday sequence.
    """

    entry_price = float(
        stock_data["close"].iloc[entry_index]
    )

    entry_date = stock_data.index[
        entry_index
    ]

    target_price = entry_price * (
        1 + target_percent / 100
    )

    stop_price = entry_price * (
        1 - stop_percent / 100
    )

    final_index = min(
        entry_index + horizon_days,
        len(stock_data) - 1,
    )

    if final_index <= entry_index:
        return None

    future_data = stock_data.iloc[
        entry_index + 1:final_index + 1
    ].copy()

    maximum_high = float(
        future_data["high"].max()
    )

    minimum_low = float(
        future_data["low"].min()
    )

    outcome = "Time Expired"
    outcome_date = future_data.index[-1]
    days_to_outcome = len(future_data)

    for day_number, (date, row) in enumerate(
        future_data.iterrows(),
        start=1,
    ):
        target_touched = (
            float(row["high"]) >= target_price
        )

        stop_touched = (
            float(row["low"]) <= stop_price
        )

        if target_touched and stop_touched:
            outcome = "Ambiguous Same Day"
            outcome_date = date
            days_to_outcome = day_number
            break

        if target_touched:
            outcome = "Target Hit First"
            outcome_date = date
            days_to_outcome = day_number
            break

        if stop_touched:
            outcome = "Stop Hit First"
            outcome_date = date
            days_to_outcome = day_number
            break

    final_close = float(
        future_data["close"].iloc[-1]
    )

    return {
        "Entry Date": entry_date,
        "Entry Price": entry_price,
        "Target Price": target_price,
        "Stop Price": stop_price,
        "Outcome": outcome,
        "Outcome Date": outcome_date,
        "Days to Outcome": days_to_outcome,
        "Forward Return %": (
            final_close / entry_price - 1
        ) * 100,
        "Max Gain %": (
            maximum_high / entry_price - 1
        ) * 100,
        "Max Drawdown %": (
            minimum_low / entry_price - 1
        ) * 100,
        "Target Hit First": (
            1
            if outcome == "Target Hit First"
            else 0
        ),
    }


@st.cache_data(ttl=43200, show_spinner=False)
def run_phase1_historical_study(
    stock_data,
    benchmark_data,
    target_percent,
    stop_percent,
    horizon_days,
    setup_spacing,
    require_strong_daily,
    require_weekly_bullish,
    require_positive_rs,
    max_distance_from_52w,
    minimum_volume_change,
):
    """Run historical setup event study for one stock."""

    if stock_data.empty or benchmark_data.empty:
        return pd.DataFrame()

    if len(stock_data) < 400:
        return pd.DataFrame()

    events = []

    last_event_index = -setup_spacing

    final_index = (
        len(stock_data)
        - horizon_days
        - 1
    )

    for index in range(
        260,
        final_index,
    ):
        if (
            index - last_event_index
            < setup_spacing
        ):
            continue

        features = historical_features(
            stock_data,
            benchmark_data,
            index,
        )

        qualifies = historical_setup_qualifies(
            features,
            require_strong_daily,
            require_weekly_bullish,
            require_positive_rs,
            max_distance_from_52w,
            minimum_volume_change,
        )

        if not qualifies:
            continue

        result = evaluate_triple_barrier(
            stock_data,
            index,
            target_percent,
            stop_percent,
            horizon_days,
        )

        if result is None:
            continue

        result.update(
            {
                "Daily Trend": features.get(
                    "daily_trend_label"
                ),
                "Weekly Trend": features.get(
                    "weekly_trend_label"
                ),
                "RS 3M %": features.get(
                    "rs_3m"
                ),
                "Distance from 52W High %": features.get(
                    "distance_52w_high"
                ),
                "Volume vs 20D Avg %": features.get(
                    "volume_change"
                ),
                "ATR %": features.get(
                    "atr_percent"
                ),
                "Momentum 1M %": features.get(
                    "momentum_1m"
                ),
                "Distance from MA50 %": features.get(
                    "distance_ma50"
                ),
            }
        )

        events.append(result)

        last_event_index = index

    if not events:
        return pd.DataFrame()

    return (
        pd.DataFrame(events)
        .sort_values(
            by="Entry Date",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def summarise_phase1(events):
    """Summarise Phase 1 event-study results."""

    if events is None or events.empty:
        return {
            "count": 0,
            "target_rate": None,
            "stop_rate": None,
            "median_return": None,
            "average_return": None,
            "median_gain": None,
            "median_drawdown": None,
            "confidence": "Insufficient sample",
        }

    clean_events = events[
        events["Outcome"]
        != "Ambiguous Same Day"
    ].copy()

    if clean_events.empty:
        target_rate = None
        stop_rate = None
    else:
        target_rate = (
            (
                clean_events["Outcome"]
                == "Target Hit First"
            ).mean()
            * 100
        )

        stop_rate = (
            (
                clean_events["Outcome"]
                == "Stop Hit First"
            ).mean()
            * 100
        )

    return {
        "count": len(events),
        "target_rate": target_rate,
        "stop_rate": stop_rate,
        "median_return": float(
            events["Forward Return %"].median()
        ),
        "average_return": float(
            events["Forward Return %"].mean()
        ),
        "median_gain": float(
            events["Max Gain %"].median()
        ),
        "median_drawdown": float(
            events["Max Drawdown %"].median()
        ),
        "confidence": sample_confidence(
            len(events)
        ),
    }


# =============================================================================
# PHASE 2 SIMILARITY SCORING
# =============================================================================

def categorical_similarity(
    current_value,
    historical_value,
    points,
):
    """Score categorical similarity."""

    if current_value is None or historical_value is None:
        return points * 0.50

    if current_value == historical_value:
        return points

    bullish = [
        "Strong bullish",
        "Bullish",
    ]

    if (
        current_value in bullish
        and historical_value in bullish
    ):
        return points * 0.70

    return 0


def numeric_similarity(
    current_value,
    historical_value,
    tolerance,
    points,
):
    """Score numeric similarity with linear decay."""

    current_value = safe_number(
        current_value
    )

    historical_value = safe_number(
        historical_value
    )

    if current_value is None or historical_value is None:
        return points * 0.50

    difference = abs(
        current_value - historical_value
    )

    if difference <= tolerance:
        return points

    if difference >= tolerance * 2:
        return 0

    return points * (
        1
        - (
            (difference - tolerance)
            / tolerance
        )
    )


def calculate_similarity_score(
    current_features,
    historical_event,
):
    """Score historical event similarity to current setup out of 100."""

    scores = {
        "Daily Trend": categorical_similarity(
            current_features.get(
                "daily_trend_label"
            ),
            historical_event.get(
                "Daily Trend"
            ),
            15,
        ),
        "Weekly Trend": categorical_similarity(
            current_features.get(
                "weekly_trend_label"
            ),
            historical_event.get(
                "Weekly Trend"
            ),
            15,
        ),
        "RS 3M": numeric_similarity(
            current_features.get("rs_3m"),
            historical_event.get("RS 3M %"),
            8,
            15,
        ),
        "52W High Distance": numeric_similarity(
            current_features.get(
                "distance_52w_high"
            ),
            historical_event.get(
                "Distance from 52W High %"
            ),
            4,
            15,
        ),
        "Volume": numeric_similarity(
            current_features.get(
                "volume_change"
            ),
            historical_event.get(
                "Volume vs 20D Avg %"
            ),
            35,
            10,
        ),
        "ATR": numeric_similarity(
            current_features.get(
                "atr_percent"
            ),
            historical_event.get(
                "ATR %"
            ),
            1.5,
            10,
        ),
        "Momentum": numeric_similarity(
            current_features.get(
                "momentum_1m"
            ),
            historical_event.get(
                "Momentum 1M %"
            ),
            6,
            10,
        ),
        "MA50 Distance": numeric_similarity(
            current_features.get(
                "distance_ma50"
            ),
            historical_event.get(
                "Distance from MA50 %"
            ),
            5,
            10,
        ),
    }

    return {
        "score": round(
            sum(scores.values()),
            1,
        ),
        "components": scores,
    }


def run_phase2_similarity(
    events,
    current_features,
):
    """Add similarity scores to all Phase 1 events."""

    if events is None or events.empty:
        return pd.DataFrame()

    scored = events.copy()

    output = []

    for _, event in scored.iterrows():
        similarity = calculate_similarity_score(
            current_features,
            event,
        )

        row = event.to_dict()

        row["Similarity Score"] = similarity[
            "score"
        ]

        for name, value in similarity[
            "components"
        ].items():
            row[f"Similarity: {name}"] = value

        output.append(row)

    return (
        pd.DataFrame(output)
        .sort_values(
            by="Similarity Score",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def summarise_phase2(
    scored_events,
    minimum_similarity,
):
    """Summarise only historical events above selected similarity."""

    if scored_events is None or scored_events.empty:
        return {
            "count": 0,
            "target_rate": None,
            "stop_rate": None,
            "median_return": None,
            "median_gain": None,
            "median_drawdown": None,
            "confidence": "Insufficient sample",
        }

    selected = scored_events[
        scored_events["Similarity Score"]
        >= minimum_similarity
    ].copy()

    if selected.empty:
        return {
            "count": 0,
            "target_rate": None,
            "stop_rate": None,
            "median_return": None,
            "median_gain": None,
            "median_drawdown": None,
            "confidence": "Insufficient sample",
        }

    clean = selected[
        selected["Outcome"]
        != "Ambiguous Same Day"
    ].copy()

    target_rate = (
        (
            clean["Outcome"]
            == "Target Hit First"
        ).mean()
        * 100
        if not clean.empty
        else None
    )

    stop_rate = (
        (
            clean["Outcome"]
            == "Stop Hit First"
        ).mean()
        * 100
        if not clean.empty
        else None
    )

    return {
        "count": len(selected),
        "target_rate": target_rate,
        "stop_rate": stop_rate,
        "median_return": float(
            selected["Forward Return %"].median()
        ),
        "median_gain": float(
            selected["Max Gain %"].median()
        ),
        "median_drawdown": float(
            selected["Max Drawdown %"].median()
        ),
        "confidence": sample_confidence(
            len(selected)
        ),
    }


# =============================================================================
# PHASE 3 CROSS-SECTIONAL MODEL
# =============================================================================

@st.cache_data(ttl=43200, show_spinner=False)
def build_phase3_dataset(
    universe,
    benchmark_data,
    stock_limit,
    target_percent,
    stop_percent,
    horizon_days,
    setup_spacing,
    require_strong_daily,
    require_weekly_bullish,
    require_positive_rs,
    max_distance_from_52w,
    minimum_volume_change,
):
    """
    Build training dataset across multiple stocks using historic setups.

    The function only creates feature values using historical data
    visible at each setup date.
    """

    if benchmark_data.empty:
        return pd.DataFrame()

    rows = []

    selected_universe = universe.head(
        stock_limit
    ).copy()

    for _, record in selected_universe.iterrows():
        symbol = record["Symbol"]
        ticker = record["Ticker"]

        stock_data = fetch_price_data(
            ticker,
            "5y",
        )

        if stock_data.empty or len(stock_data) < 400:
            continue

        last_event_index = -setup_spacing

        final_index = (
            len(stock_data)
            - horizon_days
            - 1
        )

        for index in range(
            260,
            final_index,
        ):
            if (
                index - last_event_index
                < setup_spacing
            ):
                continue

            features = historical_features(
                stock_data,
                benchmark_data,
                index,
            )

            qualifies = historical_setup_qualifies(
                features,
                require_strong_daily,
                require_weekly_bullish,
                require_positive_rs,
                max_distance_from_52w,
                minimum_volume_change,
            )

            if not qualifies:
                continue

            result = evaluate_triple_barrier(
                stock_data,
                index,
                target_percent,
                stop_percent,
                horizon_days,
            )

            if result is None:
                continue

            if result["Outcome"] == "Ambiguous Same Day":
                continue

            row = {
                "Symbol": symbol,
                "Ticker": ticker,
                "Entry Date": result["Entry Date"],
                "Outcome": result["Outcome"],
                "Target Hit First": result[
                    "Target Hit First"
                ],
                "Forward Return %": result[
                    "Forward Return %"
                ],
            }

            for feature in MODEL_FEATURES:
                row[feature] = features.get(feature)

            rows.append(row)

            last_event_index = index

    if not rows:
        return pd.DataFrame()

    dataset = pd.DataFrame(rows)

    for feature in MODEL_FEATURES:
        dataset[feature] = pd.to_numeric(
            dataset[feature],
            errors="coerce",
        )

    dataset = dataset.dropna(
        subset=MODEL_FEATURES
    )

    return dataset.reset_index(drop=True)


def train_phase3_model(dataset):
    """
    Train regularized Logistic Regression.

    Target:
    1 = Target Hit First
    0 = Stop Hit First or Time Expired.
    """

    if dataset is None or dataset.empty:
        return None

    if len(dataset) < 50:
        return None

    features = dataset[
        MODEL_FEATURES
    ].copy()

    target = dataset[
        "Target Hit First"
    ].astype(int)

    if target.nunique() < 2:
        return None

    pipeline = Pipeline(
        steps=[
            (
                "scale",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    C=0.5,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    pipeline.fit(
        features,
        target,
    )

    probabilities = pipeline.predict_proba(
        features
    )[:, 1]

    predicted = (
        probabilities >= 0.50
    ).astype(int)

    accuracy = accuracy_score(
        target,
        predicted,
    )

    try:
        auc = roc_auc_score(
            target,
            probabilities,
        )
    except Exception:
        auc = None

    logistic_model = pipeline.named_steps[
        "model"
    ]

    coefficients = pd.DataFrame(
        {
            "Feature": MODEL_FEATURES,
            "Coefficient": logistic_model.coef_[0],
            "Absolute Impact": np.abs(
                logistic_model.coef_[0]
            ),
        }
    ).sort_values(
        by="Absolute Impact",
        ascending=False,
    )

    return {
        "model": pipeline,
        "rows": len(dataset),
        "positive_rows": int(target.sum()),
        "accuracy": accuracy,
        "auc": auc,
        "coefficients": coefficients,
    }


def predict_phase3_probability(
    model_package,
    current_features,
):
    """Predict target-before-stop probability for present stock setup."""

    if model_package is None:
        return None

    model_input = {}

    for feature in MODEL_FEATURES:
        value = safe_number(
            current_features.get(feature)
        )

        if value is None:
            return None

        model_input[feature] = value

    input_dataframe = pd.DataFrame(
        [model_input]
    )

    probability = model_package[
        "model"
    ].predict_proba(
        input_dataframe
    )[0, 1]

    return {
        "probability": float(probability),
        "label": probability_label(
            probability
        ),
    }


# =============================================================================
# COMMAND CENTER ANALYSIS
# =============================================================================

def analyse_stock_for_scanner(
    record,
    benchmark_data,
    minimum_market_cap,
):
    """
    Analyse one stock for Command Center and Winner Ranking Scanner.
    """

    symbol = record["Symbol"]
    ticker = record["Ticker"]

    stock_data = fetch_price_data(
        ticker,
        "5y",
    )

    if stock_data.empty:
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
            copied = dict(pattern)
            copied["Timeframe"] = timeframe
            all_patterns.append(copied)

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
        timeframe = selected_pattern["Timeframe"]

        if timeframe == "Weekly":
            plan_data = weekly_data
        elif timeframe == "Monthly":
            plan_data = monthly_data
        else:
            plan_data = daily_data

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
        "Stock": symbol,
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
        "Risk Flags": " | ".join(
            winner_score["risks"][:5]
        ),
        "Positive Evidence": " | ".join(
            winner_score["positives"][:5]
        ),
    }


def run_universe_scan(
    universe,
    benchmark_data,
    minimum_market_cap,
    scan_limit,
    progress_callback=None,
):
    """Run current technical/Winner Score analysis on selected universe size."""

    scan_rows = []

    scan_universe = universe.head(
        scan_limit
    ).copy()

    total = len(scan_universe)

    for index, (_, record) in enumerate(
        scan_universe.iterrows(),
        start=1,
    ):
        if progress_callback is not None:
            progress_callback(
                index,
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
                scan_rows.append(result)

        except Exception:
            pass

    if not scan_rows:
        return pd.DataFrame()

    return pd.DataFrame(scan_rows)


# =============================================================================
# MARKET REGIME AND BREADTH
# =============================================================================

def calculate_market_breadth(scan_dataframe):
    """Calculate percentage of scanned stocks above Daily MA50 and MA200."""

    if scan_dataframe is None or scan_dataframe.empty:
        return {
            "above_50dma": None,
            "above_200dma": None,
            "count": 0,
        }

    valid_50 = scan_dataframe.dropna(
        subset=[
            "Current Price",
            "MA50",
        ]
    )

    valid_200 = scan_dataframe.dropna(
        subset=[
            "Current Price",
            "MA200",
        ]
    )

    above_50dma = (
        (
            valid_50["Current Price"]
            > valid_50["MA50"]
        ).mean()
        * 100
        if not valid_50.empty
        else None
    )

    above_200dma = (
        (
            valid_200["Current Price"]
            > valid_200["MA200"]
        ).mean()
        * 100
        if not valid_200.empty
        else None
    )

    return {
        "above_50dma": above_50dma,
        "above_200dma": above_200dma,
        "count": len(scan_dataframe),
    }


def calculate_market_regime(
    nifty_data,
    breadth,
):
    """Classify broad market regime."""

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
# PHASE 3 MODEL FUNCTIONS
# =============================================================================

@st.cache_data(ttl=43200, show_spinner=False)
def build_phase3_dataset(
    universe,
    benchmark_data,
    stock_limit,
    target_percent,
    stop_percent,
    horizon_days,
    setup_spacing,
    require_strong_daily,
    require_weekly_bullish,
    require_positive_rs,
    max_distance_from_52w,
):
    """Build historical cross-sectional training dataset."""

    records = []

    selected_universe = universe.head(
        stock_limit
    ).copy()

    for _, record in selected_universe.iterrows():
        stock_data = fetch_price_data(
            record["Ticker"],
            "5y",
        )

        if stock_data.empty or len(stock_data) < 400:
            continue

        last_event_index = -setup_spacing

        final_index = (
            len(stock_data)
            - horizon_days
            - 1
        )

        for index in range(
            260,
            final_index,
        ):
            if (
                index - last_event_index
                < setup_spacing
            ):
                continue

            features = historical_features(
                stock_data,
                benchmark_data,
                index,
            )

            qualifies = historical_setup_qualifies(
                features,
                require_strong_daily,
                require_weekly_bullish,
                require_positive_rs,
                max_distance_from_52w,
                None,
            )

            if not qualifies:
                continue

            outcome = evaluate_triple_barrier(
                stock_data,
                index,
                target_percent,
                stop_percent,
                horizon_days,
            )

            if outcome is None:
                continue

            if outcome["Outcome"] == "Ambiguous Same Day":
                continue

            row = {
                "Symbol": record["Symbol"],
                "Entry Date": outcome["Entry Date"],
                "Outcome": outcome["Outcome"],
                "Target Hit First": outcome[
                    "Target Hit First"
                ],
                "Forward Return %": outcome[
                    "Forward Return %"
                ],
            }

            for feature in MODEL_FEATURES:
                row[feature] = features.get(
                    feature
                )

            records.append(row)

            last_event_index = index

    if not records:
        return pd.DataFrame()

    dataset = pd.DataFrame(records)

    for feature in MODEL_FEATURES:
        dataset[feature] = pd.to_numeric(
            dataset[feature],
            errors="coerce",
        )

    dataset = dataset.dropna(
        subset=MODEL_FEATURES
    )

    return dataset.reset_index(drop=True)


def train_phase3_model(dataset):
    """Train logistic regression Phase 3 probability model."""

    if dataset is None or dataset.empty:
        return None

    if len(dataset) < 50:
        return None

    features = dataset[
        MODEL_FEATURES
    ].copy()

    target = dataset[
        "Target Hit First"
    ].astype(int)

    if target.nunique() < 2:
        return None

    model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "logistic",
                LogisticRegression(
                    max_iter=2000,
                    C=0.5,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(
        features,
        target,
    )

    probabilities = model.predict_proba(
        features
    )[:, 1]

    predictions = (
        probabilities >= 0.50
    ).astype(int)

    accuracy = accuracy_score(
        target,
        predictions,
    )

    try:
        auc = roc_auc_score(
            target,
            probabilities,
        )
    except Exception:
        auc = None

    logistic = model.named_steps[
        "logistic"
    ]

    coefficients = pd.DataFrame(
        {
            "Feature": MODEL_FEATURES,
            "Coefficient": logistic.coef_[0],
            "Absolute Impact": np.abs(
                logistic.coef_[0]
            ),
        }
    ).sort_values(
        by="Absolute Impact",
        ascending=False,
    )

    return {
        "model": model,
        "rows": len(dataset),
        "positive_rows": int(
            target.sum()
        ),
        "accuracy": accuracy,
        "auc": auc,
        "coefficients": coefficients,
    }


def predict_phase3_probability(
    model_package,
    current_features,
):
    """Predict target-before-stop model probability."""

    if model_package is None:
        return None

    model_input = {}

    for feature in MODEL_FEATURES:
        value = safe_number(
            current_features.get(feature)
        )

        if value is None:
            return None

        model_input[feature] = value

    input_dataframe = pd.DataFrame(
        [model_input]
    )

    probability = model_package[
        "model"
    ].predict_proba(
        input_dataframe
    )[0, 1]

    return {
        "probability": float(probability),
        "label": probability_label(
            probability
        ),
    }


# =============================================================================
# VISUALISATION FUNCTIONS
# =============================================================================

def create_stock_chart(
    data,
    patterns,
    title,
):
    """Create candlestick, moving averages, volume and pattern-level chart."""

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


def create_rs_chart(rs_line, symbol):
    """Create relative strength line chart."""

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
        height=350,
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Stock Close / Nifty 50 Close",
    )

    return figure


def create_phase2_chart(
    events,
    target_percent,
    stop_percent,
):
    """Create similarity score versus historical forward outcome chart."""

    figure = go.Figure()

    if events is None or events.empty:
        figure.update_layout(
            title="No similarity events available",
            template="plotly_white",
        )

        return figure

    colors = {
        "Target Hit First": "#16a34a",
        "Stop Hit First": "#dc2626",
        "Time Expired": "#f59e0b",
        "Ambiguous Same Day": "#6b7280",
    }

    for outcome in events["Outcome"].unique():
        subset = events[
            events["Outcome"] == outcome
        ]

        figure.add_trace(
            go.Scatter(
                x=subset["Similarity Score"],
                y=subset["Forward Return %"],
                mode="markers",
                name=outcome,
                marker=dict(
                    size=10,
                    color=colors.get(
                        outcome,
                        "#2563eb",
                    ),
                ),
            )
        )

    figure.add_hline(
        y=target_percent,
        line_dash="dash",
        line_color="#16a34a",
        annotation_text=f"Target +{target_percent}%",
    )

    figure.add_hline(
        y=-stop_percent,
        line_dash="dash",
        line_color="#dc2626",
        annotation_text=f"Stop -{stop_percent}%",
    )

    figure.update_layout(
        title="Historical Similarity vs Forward Outcome",
        height=420,
        template="plotly_white",
        xaxis_title="Similarity Score",
        yaxis_title="Forward Return %",
    )

    return figure


# =============================================================================
# DATA LOADING
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
        🏆 Nifty Total Market Winner Research Dashboard
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="sub-title">
        Daily Command Center • Stock Research • Winner Ranking •
        Relative Strength • Pattern Analysis • Phase 1–3 Probability Research
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
        "Relative Strength and Phase 1–3 modules may be incomplete."
    )


# =============================================================================
# SIDEBAR NAVIGATION
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
        "Minimum market cap (₹ crore)",
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
        "Prices cache: 15 minutes\n\n"
        "Fundamentals cache: 12 hours\n\n"
        "Historical studies cache: 12 hours\n\n"
        "Constituent list cache: 6 hours"
    )


# =============================================================================
# DAILY MARKET COMMAND CENTER
# =============================================================================

if dashboard_mode == "Daily Market Command Center":
    st.subheader(
        "Daily Market Command Center"
    )

    st.caption(
        "Run a daily scan to identify market regime, leading stocks, "
        "confirmed bullish breakouts, ideal entries, extended setups and risk alerts."
    )

    scan_column_1, scan_column_2 = st.columns(2)

    with scan_column_1:
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

    with scan_column_2:
        minimum_score_command = st.slider(
            "Minimum Winner Score for top candidates",
            min_value=0,
            max_value=100,
            value=50,
            step=5,
            key="command_min_score",
        )

    st.warning(
        "Start with 50 or 100 stocks on free Streamlit Cloud. "
        "A complete Nifty 750 scan may take several minutes."
    )

    if st.button(
        "▶ Run Daily Command Center Scan",
        type="primary",
    ):
        progress_bar = st.progress(0)
        progress_text = st.empty()

        def progress_callback(position, total, symbol):
            progress_bar.progress(
                position / total
            )

            progress_text.caption(
                f"Scanning {position:,} of {total:,}: {symbol}"
            )

        with st.spinner(
            "Running current market analysis..."
        ):
            scan_results = run_universe_scan(
                stock_universe,
                nifty_50_data,
                minimum_market_cap,
                command_scan_limit,
                progress_callback,
            )

        progress_bar.empty()
        progress_text.empty()

        st.session_state[
            "command_center_results"
        ] = scan_results

    command_results = st.session_state[
        "command_center_results"
    ]

    if command_results is None or command_results.empty:
        st.info(
            "Click Run Daily Command Center Scan to create today's dashboard."
        )

    else:
        breadth = calculate_market_breadth(
            command_results
        )

        market_regime = calculate_market_regime(
            nifty_50_data,
            breadth,
        )

        if market_regime["regime"] in [
            "Strong Bullish",
            "Bullish",
        ]:
            regime_class = "green-card"

        elif market_regime["regime"] == "Defensive":
            regime_class = "red-card"

        else:
            regime_class = "orange-card"

        st.markdown(
            f"""
            <div class="research-card {regime_class}">
                <h3>Market Regime: {market_regime["regime"]}</h3>
                <p>
                    Nifty 50 Trend: {market_regime["nifty_trend"]} |
                    Breadth Above 50 DMA: {
                        f"{market_regime['breadth_50']:.1f}%"
                        if market_regime["breadth_50"] is not None
                        else "Not available"
                    }
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        regime_col1, regime_col2, regime_col3, regime_col4 = (
            st.columns(4)
        )

        regime_col1.metric(
            "Nifty 50 Trend",
            market_regime["nifty_trend"],
        )

        regime_col2.metric(
            "Breadth Above 50 DMA",
            (
                f"{market_regime['breadth_50']:.1f}%"
                if market_regime["breadth_50"] is not None
                else "Not available"
            ),
        )

        regime_col3.metric(
            "Breadth Above 200 DMA",
            (
                f"{market_regime['breadth_200']:.1f}%"
                if market_regime["breadth_200"] is not None
                else "Not available"
            ),
        )

        regime_col4.metric(
            "Eligible Stocks Scanned",
            breadth["count"],
        )

        st.divider()

        # ---------------------------------------------------------------------
        # TOP WINNER CANDIDATES
        # ---------------------------------------------------------------------

        st.subheader(
            "🏆 Top Winner Candidates"
        )

        top_candidates = command_results[
            command_results["Winner Score"]
            >= minimum_score_command
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
                column_config={
                    "Winner Score": st.column_config.ProgressColumn(
                        "Winner Score",
                        min_value=0,
                        max_value=100,
                        format="%.1f",
                    ),
                    "MTF Score": st.column_config.NumberColumn(
                        "MTF Score",
                        format="%.0f/10",
                    ),
                    "Risk Reward": st.column_config.NumberColumn(
                        "Risk / Reward",
                        format="1 : %.2f",
                    ),
                },
            )

            research_symbol = st.selectbox(
                "Open detailed Stock Research",
                top_candidates["Stock"].tolist(),
                key="top_candidates_research_symbol",
            )

            if st.button(
                "Open Selected Stock Research",
                key="open_top_candidate",
            ):
                open_stock_research(
                    research_symbol
                )
                st.rerun()

        else:
            st.info(
                "No stocks met the selected minimum Winner Score."
            )

        st.divider()

        # ---------------------------------------------------------------------
        # CONFIRMED BREAKOUTS
        # ---------------------------------------------------------------------

        st.subheader(
            "📈 New Confirmed Bullish Breakouts"
        )

        confirmed_breakouts = command_results[
            (
                command_results["Pattern Status"]
                == "Confirmed"
            )
            & (
                command_results["Pattern Direction"]
                == "Bullish"
            )
        ].copy()

        confirmed_breakouts = confirmed_breakouts.sort_values(
            by=[
                "Winner Score",
                "Pattern Volume %",
            ],
            ascending=[
                False,
                False,
            ],
        ).head(25)

        if not confirmed_breakouts.empty:
            st.dataframe(
                confirmed_breakouts[
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
                column_config={
                    "Pattern Volume %": st.column_config.NumberColumn(
                        "Volume vs Average",
                        format="%.1f%%",
                    ),
                    "Winner Score": st.column_config.ProgressColumn(
                        "Winner Score",
                        min_value=0,
                        max_value=100,
                        format="%.1f",
                    ),
                    "Risk Reward": st.column_config.NumberColumn(
                        "Risk / Reward",
                        format="1 : %.2f",
                    ),
                },
            )

        else:
            st.info(
                "No confirmed bullish breakouts were found."
            )

        st.divider()

        # ---------------------------------------------------------------------
        # IDEAL ENTRY ZONE
        # ---------------------------------------------------------------------

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
                column_config={
                    "Breakout Level": st.column_config.NumberColumn(
                        "Breakout Level",
                        format="₹%.2f",
                    ),
                    "Current Price": st.column_config.NumberColumn(
                        "Current Price",
                        format="₹%.2f",
                    ),
                    "Distance From Breakout %": st.column_config.NumberColumn(
                        "Distance",
                        format="%.2f%%",
                    ),
                    "Stop Loss": st.column_config.NumberColumn(
                        "Stop Loss",
                        format="₹%.2f",
                    ),
                    "Target": st.column_config.NumberColumn(
                        "Target",
                        format="₹%.2f",
                    ),
                    "Risk Reward": st.column_config.NumberColumn(
                        "Risk / Reward",
                        format="1 : %.2f",
                    ),
                    "Winner Score": st.column_config.ProgressColumn(
                        "Winner Score",
                        min_value=0,
                        max_value=100,
                        format="%.1f",
                    ),
                },
            )

        else:
            st.info(
                "No current bullish setups are near ideal entry conditions."
            )

        st.divider()

        # ---------------------------------------------------------------------
        # EXTENDED SETUPS
        # ---------------------------------------------------------------------

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
                column_config={
                    "Breakout Level": st.column_config.NumberColumn(
                        "Breakout Level",
                        format="₹%.2f",
                    ),
                    "Current Price": st.column_config.NumberColumn(
                        "Current Price",
                        format="₹%.2f",
                    ),
                    "Distance From Breakout %": st.column_config.NumberColumn(
                        "Distance",
                        format="%.2f%%",
                    ),
                    "Winner Score": st.column_config.ProgressColumn(
                        "Winner Score",
                        min_value=0,
                        max_value=100,
                        format="%.1f",
                    ),
                },
            )

        else:
            st.info(
                "No materially extended bullish setups were found."
            )

        st.divider()

        # ---------------------------------------------------------------------
        # RISK ALERTS
        # ---------------------------------------------------------------------

        st.subheader(
            "🚨 Risk Alerts"
        )

        risk_rows = []

        for _, row in command_results.iterrows():
            current_price = safe_number(
                row["Current Price"]
            )

            ma20 = safe_number(
                row["MA20"]
            )

            if (
                current_price is not None
                and ma20 is not None
                and current_price < ma20
            ):
                risk_rows.append(
                    {
                        "Stock": row["Stock"],
                        "Risk Alert": "Price below Daily 20 DMA",
                        "Details": (
                            f"Current price {format_price(current_price)} "
                            f"is below MA20 {format_price(ma20)}."
                        ),
                        "Winner Score": row[
                            "Winner Score"
                        ],
                    }
                )

            if row["Weekly Trend"] == "Bearish":
                risk_rows.append(
                    {
                        "Stock": row["Stock"],
                        "Risk Alert": "Weekly Trend Bearish",
                        "Details": "Weekly trend is Bearish.",
                        "Winner Score": row[
                            "Winner Score"
                        ],
                    }
                )

            if row["Monthly Trend"] == "Bearish":
                risk_rows.append(
                    {
                        "Stock": row["Stock"],
                        "Risk Alert": "Monthly Trend Bearish",
                        "Details": "Monthly trend is Bearish.",
                        "Winner Score": row[
                            "Winner Score"
                        ],
                    }
                )

            if (
                row["Pattern Status"] == "Confirmed"
                and row["Pattern Direction"] == "Bearish"
            ):
                risk_rows.append(
                    {
                        "Stock": row["Stock"],
                        "Risk Alert": "Confirmed Bearish Pattern",
                        "Details": (
                            f"{row['Current Pattern']} confirmed "
                            f"on {row['Pattern Timeframe']}."
                        ),
                        "Winner Score": row[
                            "Winner Score"
                        ],
                    }
                )

            if row["RS Status"] == "Weak":
                risk_rows.append(
                    {
                        "Stock": row["Stock"],
                        "Risk Alert": "Weak Relative Strength",
                        "Details": (
                            "Underperforming Nifty 50 over multiple measured periods."
                        ),
                        "Winner Score": row[
                            "Winner Score"
                        ],
                    }
                )

            risk_reward = safe_number(
                row["Risk Reward"]
            )

            if (
                risk_reward is not None
                and risk_reward < 1.5
                and row["Pattern Direction"] == "Bullish"
            ):
                risk_rows.append(
                    {
                        "Stock": row["Stock"],
                        "Risk Alert": "Low Risk / Reward",
                        "Details": (
                            f"Rule-based risk/reward is only 1 : {risk_reward:.2f}."
                        ),
                        "Winner Score": row[
                            "Winner Score"
                        ],
                    }
                )

        if risk_rows:
            risk_dataframe = pd.DataFrame(
                risk_rows
            ).sort_values(
                by="Winner Score",
                ascending=False,
            )

            st.dataframe(
                risk_dataframe,
                hide_index=True,
                use_container_width=True,
            )

        else:
            st.success(
                "No major rule-based risk alerts found in scanned results."
            )

        st.divider()

        command_csv = command_results.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "⬇️ Download Command Center CSV",
            data=command_csv,
            file_name="nifty_daily_command_center.csv",
            mime="text/csv",
        )


# =============================================================================
# STOCK RESEARCH
# =============================================================================

elif dashboard_mode == "Stock Research":
    st.subheader("Stock Research")

    available_symbols = sorted(
        stock_universe["Symbol"].tolist()
    )

    session_symbol = st.session_state.get(
        "selected_symbol",
        "RELIANCE",
    )

    if session_symbol not in available_symbols:
        session_symbol = available_symbols[0]

    selected_symbol = st.selectbox(
        "Search a Nifty Total Market stock",
        available_symbols,
        index=available_symbols.index(
            session_symbol
        ),
        key="stock_research_symbol",
    )

    st.session_state["selected_symbol"] = (
        selected_symbol
    )

    selected_record = stock_universe[
        stock_universe["Symbol"] == selected_symbol
    ].iloc[0]

    selected_ticker = selected_record["Ticker"]

    with st.spinner(
        f"Loading detailed research for {selected_symbol}..."
    ):
        stock_data = fetch_price_data(
            selected_ticker,
            "5y",
        )

        fundamentals = fetch_fundamentals(
            selected_ticker
        )

    if stock_data.empty:
        st.error(
            "No price data is currently available for this stock."
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

    current_features = calculate_current_price_features(
        stock_data,
        nifty_50_data,
    )

    phase3_prediction = predict_phase3_probability(
        st.session_state[
            "phase3_model_package"
        ],
        current_features,
    )

    current_price = float(
        stock_data["close"].iloc[-1]
    )

    header_1, header_2, header_3, header_4, header_5 = (
        st.columns(5)
    )

    header_1.metric(
        "Last Close",
        format_price(current_price),
    )

    header_2.metric(
        "Market Cap",
        format_market_cap(
            fundamentals.get("marketCap")
        ),
    )

    header_3.metric(
        "RS Status",
        rs_data["status"],
    )

    header_4.metric(
        "MTF Alignment",
        (
            f"{alignment['score']}/10"
            if alignment["score"] is not None
            else "Not available"
        ),
    )

    header_5.metric(
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
        phase1_tab,
        phase2_tab,
        phase3_tab,
        fundamentals_tab,
    ) = st.tabs(
        [
            "🏆 Winner Score",
            "Technical Research",
            "Relative Strength",
            "MTF Alignment",
            "Risk / Reward",
            "🎯 Phase 1 Historical",
            "🧩 Phase 2 Similarity",
            "🤖 Phase 3 Probability",
            "Fundamentals",
        ]
    )

    # =========================================================================
    # WINNER SCORE TAB
    # =========================================================================

    with winner_tab:
        st.subheader("Winner Score Breakdown")

        score_table = pd.DataFrame(
            [
                [
                    "Fundamental Quality",
                    winner_score["fundamental_score"],
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

        score_table["Strength %"] = (
            score_table["Points"]
            / score_table["Maximum"]
        ) * 100

        st.dataframe(
            score_table,
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

        positive_col, risk_col = st.columns(2)

        with positive_col:
            st.markdown("### Positive Evidence")

            if winner_score["positives"]:
                for item in winner_score["positives"][:18]:
                    st.success(f"✅ {item}")
            else:
                st.info(
                    "No significant positive rule-based evidence found."
                )

        with risk_col:
            st.markdown("### Risk Flags")

            if winner_score["risks"]:
                for item in winner_score["risks"][:18]:
                    st.warning(f"⚠️ {item}")
            else:
                st.success(
                    "No major rule-based risk flags found."
                )

    # =========================================================================
    # TECHNICAL RESEARCH TAB
    # =========================================================================

    with technical_tab:
        daily_chart_tab, weekly_chart_tab, monthly_chart_tab = st.tabs(
            [
                "Daily",
                "Weekly",
                "Monthly",
            ]
        )

        chart_views = [
            (
                daily_chart_tab,
                "Daily",
                daily_data,
                daily_patterns,
            ),
            (
                weekly_chart_tab,
                "Weekly",
                weekly_data,
                weekly_patterns,
            ),
            (
                monthly_chart_tab,
                "Monthly",
                monthly_data,
                monthly_patterns,
            ),
        ]

        for tab, name, data, patterns in chart_views:
            with tab:
                support, resistance = calculate_support_resistance(
                    data
                )

                col_1, col_2, col_3 = st.columns(3)

                col_1.metric(
                    "Trend",
                    calculate_overall_trend(data),
                )

                col_2.metric(
                    "Support",
                    format_price(support),
                )

                col_3.metric(
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
                        "No supported active technical pattern detected."
                    )

                st.plotly_chart(
                    create_stock_chart(
                        data,
                        patterns,
                        f"{selected_symbol} — {name}",
                    ),
                    use_container_width=True,
                )

    # =========================================================================
    # RELATIVE STRENGTH TAB
    # =========================================================================

    with rs_tab:
        st.subheader(
            "Relative Strength versus Nifty 50"
        )

        rs_1, rs_2, rs_3 = st.columns(3)

        rs_1.metric(
            "RS Status",
            rs_data["status"],
        )

        rs_2.metric(
            "RS Line Trend",
            rs_data["rs_trend"],
        )

        rs_3.metric(
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
            create_rs_chart(
                rs_data["rs_line"],
                selected_symbol,
            ),
            use_container_width=True,
        )

    # =========================================================================
    # MULTI-TIMEFRAME ALIGNMENT TAB
    # =========================================================================

    with alignment_tab:
        st.subheader(
            "Daily, Weekly and Monthly Alignment"
        )

        align_1, align_2, align_3 = st.columns(3)

        align_1.metric(
            "Daily Trend",
            alignment["daily"],
        )

        align_2.metric(
            "Weekly Trend",
            alignment["weekly"],
        )

        align_3.metric(
            "Monthly Trend",
            alignment["monthly"],
        )

        align_score_col, align_status_col = st.columns(2)

        align_score_col.metric(
            "Alignment Score",
            (
                f"{alignment['score']}/10"
                if alignment["score"] is not None
                else "Not available"
            ),
        )

        align_status_col.metric(
            "Alignment Status",
            alignment["status"],
        )

        st.info(
            alignment["structure"]
        )

    # =========================================================================
    # RISK / REWARD TAB
    # =========================================================================

    with risk_tab:
        st.subheader(
            "Risk / Reward and Position Sizing"
        )

        st.caption(
            "Research levels only. This does not provide a recommendation "
            "to buy, sell, enter, exit, or allocate capital."
        )

        plan_timeframe = st.selectbox(
            "Timeframe",
            [
                "Daily",
                "Weekly",
                "Monthly",
            ],
            key="risk_timeframe",
        )

        if plan_timeframe == "Daily":
            plan_data = daily_data
            plan_patterns = daily_patterns
        elif plan_timeframe == "Weekly":
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

            selected_label = st.selectbox(
                "Detected pattern",
                labels,
                key="risk_pattern",
            )

            selected_index = labels.index(
                selected_label
            )

            chosen_pattern = directional_patterns[
                selected_index
            ]

            plan = calculate_trade_plan(
                plan_data,
                chosen_pattern,
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

            q1, q2, q3, q4 = st.columns(4)

            q1.metric(
                "Support",
                format_price(plan["support"]),
            )

            q2.metric(
                "Stop",
                format_price(
                    plan["selected_stop"]
                ),
            )

            q3.metric(
                "Target",
                format_price(plan["target"]),
            )

            q4.metric(
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
                s1, s2, s3, s4 = st.columns(4)

                s1.metric(
                    "Maximum Loss",
                    format_price(
                        position_size["maximum_loss"]
                    ),
                )

                s2.metric(
                    "Risk / Share",
                    format_price(
                        position_size["risk_per_share"]
                    ),
                )

                s3.metric(
                    "Maximum Quantity",
                    f"{position_size['quantity']:,}",
                )

                s4.metric(
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
                "No directional technical pattern is currently available "
                "for the selected timeframe."
            )

    # =========================================================================
    # PHASE 1 TAB
    # =========================================================================

    with phase1_tab:
        st.subheader(
            "Phase 1: Historical Setup Event Study"
        )

        st.caption(
            "Tests what happened after prior bullish price setups in this stock. "
            "The study is historical research, not a future prediction."
        )

        p1_col1, p1_col2, p1_col3 = st.columns(3)

        with p1_col1:
            phase1_target = st.selectbox(
                "Target",
                TARGET_OPTIONS,
                index=1,
                format_func=lambda value: f"+{value}%",
                key="phase1_target",
            )

            phase1_stop = st.selectbox(
                "Stop",
                STOP_OPTIONS,
                index=1,
                format_func=lambda value: f"-{value}%",
                key="phase1_stop",
            )

        with p1_col2:
            phase1_horizon = st.selectbox(
                "Horizon",
                HORIZON_OPTIONS,
                index=1,
                format_func=lambda value: f"{value} Trading Days",
                key="phase1_horizon",
            )

            phase1_spacing = st.selectbox(
                "Minimum Setup Gap",
                [
                    10,
                    15,
                    20,
                    30,
                    40,
                ],
                index=2,
                key="phase1_spacing",
            )

        with p1_col3:
            phase1_strong_daily = st.checkbox(
                "Require Strong Bullish Daily",
                value=False,
                key="phase1_daily",
            )

            phase1_weekly_bullish = st.checkbox(
                "Require Bullish Weekly",
                value=True,
                key="phase1_weekly",
            )

            phase1_positive_rs = st.checkbox(
                "Require Positive 3M RS",
                value=True,
                key="phase1_rs",
            )

        phase1_high_distance = st.selectbox(
            "Maximum Distance Below 52W High",
            [
                -3.0,
                -5.0,
                -7.0,
                -10.0,
                -15.0,
            ],
            index=2,
            format_func=lambda value: (
                f"Within {abs(value):.0f}% of 52W High"
            ),
            key="phase1_52w",
        )

        if st.button(
            "Run Phase 1 Historical Study",
            type="primary",
        ):
            with st.spinner(
                "Evaluating historical setups..."
            ):
                events = run_phase1_historical_study(
                    stock_data,
                    nifty_50_data,
                    phase1_target,
                    phase1_stop,
                    phase1_horizon,
                    phase1_spacing,
                    phase1_strong_daily,
                    phase1_weekly_bullish,
                    phase1_positive_rs,
                    phase1_high_distance,
                    None,
                )

            st.session_state["phase1_events"] = events
            st.session_state["phase1_symbol"] = selected_symbol
            st.session_state["phase1_settings"] = {
                "target": phase1_target,
                "stop": phase1_stop,
                "horizon": phase1_horizon,
            }

        phase1_events = st.session_state[
            "phase1_events"
        ]

        if (
            phase1_events is not None
            and not phase1_events.empty
            and st.session_state["phase1_symbol"]
            == selected_symbol
        ):
            phase1_summary = summarise_phase1(
                phase1_events
            )

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Historical Setups",
                phase1_summary["count"],
            )

            c2.metric(
                "Target Hit First",
                (
                    f"{phase1_summary['target_rate']:.1f}%"
                    if phase1_summary["target_rate"] is not None
                    else "Not available"
                ),
            )

            c3.metric(
                "Stop Hit First",
                (
                    f"{phase1_summary['stop_rate']:.1f}%"
                    if phase1_summary["stop_rate"] is not None
                    else "Not available"
                ),
            )

            c4.metric(
                "Confidence",
                phase1_summary["confidence"],
            )

            st.dataframe(
                phase1_events,
                hide_index=True,
                use_container_width=True,
            )

        else:
            st.info(
                "Configure the study and click Run Phase 1 Historical Study."
            )

    # =========================================================================
    # PHASE 2 TAB
    # =========================================================================

    with phase2_tab:
        st.subheader(
            "Phase 2: Similar Historical Setup Analysis"
        )

        st.caption(
            "Ranks Phase 1 historical setups by resemblance to the present "
            "stock setup using trend, relative strength, volatility, volume, "
            "momentum and distance-from-high features."
        )

        phase1_events = st.session_state[
            "phase1_events"
        ]

        if (
            phase1_events is None
            or phase1_events.empty
            or st.session_state["phase1_symbol"]
            != selected_symbol
        ):
            st.info(
                "Run Phase 1 Historical Study for this stock first."
            )

        else:
            similarity_threshold = st.slider(
                "Minimum Similarity Score",
                min_value=40,
                max_value=95,
                value=70,
                step=5,
                key="phase2_similarity",
            )

            top_case_count = st.selectbox(
                "Top Similar Cases to Display",
                [
                    5,
                    10,
                    15,
                    20,
                ],
                index=1,
                key="phase2_cases",
            )

            if st.button(
                "Calculate Similar Historical Cases",
                type="primary",
            ):
                scored_events = run_phase2_similarity(
                    phase1_events,
                    current_features,
                )

                st.session_state[
                    "phase2_events"
                ] = scored_events

                st.session_state[
                    "phase2_symbol"
                ] = selected_symbol

            phase2_events = st.session_state[
                "phase2_events"
            ]

            if (
                phase2_events is not None
                and not phase2_events.empty
                and st.session_state["phase2_symbol"]
                == selected_symbol
            ):
                phase2_summary = summarise_phase2(
                    phase2_events,
                    similarity_threshold,
                )

                similar_events = phase2_events[
                    phase2_events[
                        "Similarity Score"
                    ]
                    >= similarity_threshold
                ].copy()

                s1, s2, s3, s4 = st.columns(4)

                s1.metric(
                    "Similar Cases",
                    phase2_summary["count"],
                )

                s2.metric(
                    "Target Hit First",
                    (
                        f"{phase2_summary['target_rate']:.1f}%"
                        if phase2_summary["target_rate"] is not None
                        else "Not available"
                    ),
                )

                s3.metric(
                    "Stop Hit First",
                    (
                        f"{phase2_summary['stop_rate']:.1f}%"
                        if phase2_summary["stop_rate"] is not None
                        else "Not available"
                    ),
                )

                s4.metric(
                    "Confidence",
                    phase2_summary["confidence"],
                )

                settings = st.session_state[
                    "phase1_settings"
                ]

                st.plotly_chart(
                    create_phase2_chart(
                        similar_events,
                        settings.get("target", 10),
                        settings.get("stop", 8),
                    ),
                    use_container_width=True,
                )

                display_columns = [
                    "Similarity Score",
                    "Entry Date",
                    "Entry Price",
                    "Daily Trend",
                    "Weekly Trend",
                    "RS 3M %",
                    "Distance from 52W High %",
                    "Volume vs 20D Avg %",
                    "ATR %",
                    "Momentum 1M %",
                    "Distance from MA50 %",
                    "Outcome",
                    "Forward Return %",
                    "Max Gain %",
                    "Max Drawdown %",
                ]

                st.dataframe(
                    phase2_events[
                        display_columns
                    ].head(top_case_count),
                    hide_index=True,
                    use_container_width=True,
                )

            else:
                st.info(
                    "Set the similarity threshold and click "
                    "Calculate Similar Historical Cases."
                )

    # =========================================================================
    # PHASE 3 PROBABILITY TAB
    # =========================================================================

    with phase3_tab:
        st.subheader(
            "Phase 3: Cross-Sectional Probability Estimate"
        )

        model_package = st.session_state[
            "phase3_model_package"
        ]

        model_settings = st.session_state[
            "phase3_settings"
        ]

        if model_package is None:
            st.warning(
                "Phase 3 model has not been trained yet. "
                "Open Phase 3 Model Builder from the sidebar."
            )

        else:
            prediction = predict_phase3_probability(
                model_package,
                current_features,
            )

            if prediction is None:
                st.warning(
                    "Current data does not contain all required features."
                )

            else:
                probability = (
                    prediction["probability"] * 100
                )

                if probability >= 70:
                    probability_class = "green-card"
                elif probability >= 60:
                    probability_class = "blue-card"
                elif probability >= 50:
                    probability_class = "orange-card"
                else:
                    probability_class = "red-card"

                st.markdown(
                    f"""
                    <div class="research-card {probability_class}">
                        <h2>{probability:.1f}%</h2>
                        <h3>{prediction["label"]}</h3>
                        <p>
                            Historical model estimate of reaching
                            +{model_settings.get("target", 10)}%
                            before -{model_settings.get("stop", 8)}%
                            within {model_settings.get("horizon", 60)}
                            trading days.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                model_col1, model_col2, model_col3 = st.columns(3)

                model_col1.metric(
                    "Training Events",
                    model_package["rows"],
                )

                model_col2.metric(
                    "Positive Events",
                    model_package["positive_rows"],
                )

                model_col3.metric(
                    "Sample Confidence",
                    sample_confidence(
                        model_package["rows"]
                    ),
                )

                st.warning(
                    "This is a model estimate, not a forecast or guarantee. "
                    "It has not yet completed Phase 4 walk-forward validation."
                )

    # =========================================================================
    # FUNDAMENTALS TAB
    # =========================================================================

    with fundamentals_tab:
        st.subheader(
            "Available Fundamental Metrics"
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

        dividend_yield = safe_number(
            fundamentals.get("dividendYield")
        )

        free_cashflow = safe_number(
            fundamentals.get("freeCashflow")
        )

        metrics_table = pd.DataFrame(
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
                    "Current Ratio",
                    fundamentals.get(
                        "currentRatio",
                        "Not available",
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
                [
                    "Dividend Yield",
                    (
                        f"{dividend_yield * 100:.2f}%"
                        if dividend_yield is not None
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
            metrics_table,
            hide_index=True,
            use_container_width=True,
        )

        st.info(
            "Annual and quarterly point-in-time fundamentals and sector-specific "
            "KPIs should be added through NSE filings or a dedicated data source "
            "before they are used in historical probability-model training."
        )


# =============================================================================
# WINNER RANKING SCANNER
# =============================================================================

elif dashboard_mode == "Winner Ranking Scanner":
    st.subheader(
        "Winner Ranking Scanner"
    )

    st.caption(
        "Rank stocks using Winner Score, Relative Strength, "
        "multi-timeframe alignment, patterns and trend filters."
    )

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
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

    with filter_col2:
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

    with filter_col3:
        minimum_winner_score = st.slider(
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
            "Maximum Stocks to Scan",
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

        def scanner_progress(position, total, symbol):
            progress_bar.progress(
                position / total
            )

            progress_text.caption(
                f"Scanning {position:,} of {total:,}: {symbol}"
            )

        with st.spinner(
            "Calculating Winner Scores..."
        ):
            scan_results = run_universe_scan(
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
        ] = scan_results

    scanner_results = st.session_state.get(
        "winner_scanner_results",
        pd.DataFrame(),
    )

    if scanner_results is None or scanner_results.empty:
        st.info(
            "Configure filters and click Run Winner Ranking Scan."
        )

    else:
        filtered = scanner_results.copy()

        filtered = filtered[
            filtered["Winner Score"]
            >= minimum_winner_score
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
                column_config={
                    "Winner Score": st.column_config.ProgressColumn(
                        "Winner Score",
                        min_value=0,
                        max_value=100,
                        format="%.1f",
                    ),
                    "MTF Score": st.column_config.NumberColumn(
                        "MTF Score",
                        format="%.0f/10",
                    ),
                    "RS 3M %": st.column_config.NumberColumn(
                        "RS 3M",
                        format="%.2f%%",
                    ),
                    "RS 6M %": st.column_config.NumberColumn(
                        "RS 6M",
                        format="%.2f%%",
                    ),
                    "Risk Reward": st.column_config.NumberColumn(
                        "Risk / Reward",
                        format="1 : %.2f",
                    ),
                    "Market Cap (Cr)": st.column_config.NumberColumn(
                        "Market Cap",
                        format="₹%d Cr",
                    ),
                },
            )

            selected_result = st.selectbox(
                "Open detailed Stock Research",
                filtered["Stock"].tolist(),
                key="winner_scanner_research_symbol",
            )

            if st.button(
                "Open Selected Stock Research",
                key="open_winner_scanner_stock",
            ):
                open_stock_research(
                    selected_result
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
# STOCK PROBABILITY RESEARCH
# =============================================================================

elif dashboard_mode == "Stock Probability Research":
    st.subheader(
        "Stock Probability Research"
    )

    model_package = st.session_state[
        "phase3_model_package"
    ]

    model_settings = st.session_state[
        "phase3_settings"
    ]

    if model_package is None:
        st.warning(
            "No Phase 3 probability model has been trained yet. "
            "Go to Phase 3 Model Builder first."
        )

    else:
        symbols = sorted(
            stock_universe["Symbol"].tolist()
        )

        session_symbol = st.session_state.get(
            "selected_symbol",
            "RELIANCE",
        )

        if session_symbol not in symbols:
            session_symbol = symbols[0]

        probability_symbol = st.selectbox(
            "Select Stock",
            symbols,
            index=symbols.index(session_symbol),
            key="probability_research_symbol",
        )

        selected_record = stock_universe[
            stock_universe["Symbol"]
            == probability_symbol
        ].iloc[0]

        with st.spinner(
            f"Loading probability features for {probability_symbol}..."
        ):
            stock_data = fetch_price_data(
                selected_record["Ticker"],
                "5y",
            )

        if stock_data.empty:
            st.error(
                "No price data is available for this stock."
            )

        else:
            current_features = calculate_current_price_features(
                stock_data,
                nifty_50_data,
            )

            prediction = predict_phase3_probability(
                model_package,
                current_features,
            )

            if prediction is None:
                st.warning(
                    "This stock does not have enough usable current model features."
                )

            else:
                probability = (
                    prediction["probability"] * 100
                )

                if probability >= 70:
                    card_class = "green-card"
                elif probability >= 60:
                    card_class = "blue-card"
                elif probability >= 50:
                    card_class = "orange-card"
                else:
                    card_class = "red-card"

                st.markdown(
                    f"""
                    <div class="research-card {card_class}">
                        <h2>{probability:.1f}%</h2>
                        <h3>{prediction["label"]}</h3>
                        <p>
                            Estimated probability of reaching
                            +{model_settings.get("target", 10)}%
                            before -{model_settings.get("stop", 8)}%
                            inside {model_settings.get("horizon", 60)}
                            trading days, based on the current price-derived setup.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                p1, p2, p3 = st.columns(3)

                p1.metric(
                    "Training Events",
                    model_package["rows"],
                )

                p2.metric(
                    "Positive Events",
                    model_package["positive_rows"],
                )

                p3.metric(
                    "Training AUC",
                    (
                        f"{model_package['auc']:.3f}"
                        if model_package["auc"] is not None
                        else "Not available"
                    ),
                )

                current_feature_table = pd.DataFrame(
                    [
                        [
                            "Daily Trend",
                            current_features.get(
                                "daily_trend_label"
                            ),
                        ],
                        [
                            "Weekly Trend",
                            current_features.get(
                                "weekly_trend_label"
                            ),
                        ],
                        [
                            "RS 3M %",
                            current_features.get(
                                "rs_3m"
                            ),
                        ],
                        [
                            "Distance from 52W High %",
                            current_features.get(
                                "distance_52w_high"
                            ),
                        ],
                        [
                            "Volume Change %",
                            current_features.get(
                                "volume_change"
                            ),
                        ],
                        [
                            "ATR %",
                            current_features.get(
                                "atr_percent"
                            ),
                        ],
                        [
                            "Momentum 1M %",
                            current_features.get(
                                "momentum_1m"
                            ),
                        ],
                        [
                            "Distance from MA50 %",
                            current_features.get(
                                "distance_ma50"
                            ),
                        ],
                    ],
                    columns=[
                        "Current Model Feature",
                        "Value",
                    ],
                )

                st.dataframe(
                    current_feature_table,
                    hide_index=True,
                    use_container_width=True,
                )

                if st.button(
                    "Open This Stock in Stock Research",
                    key="open_probability_stock",
                ):
                    open_stock_research(
                        probability_symbol
                    )
                    st.rerun()

                st.warning(
                    "The Phase 3 probability is an in-sample model output. "
                    "It is not a forecast or guarantee. Phase 4 walk-forward "
                    "validation is required before treating model probabilities "
                    "as reliable decision support."
                )


# =============================================================================
# PHASE 3 MODEL BUILDER
# =============================================================================

elif dashboard_mode == "Phase 3 Model Builder":
    st.subheader(
        "Phase 3: Cross-Sectional Probability Model Builder"
    )

    st.caption(
        "Build a Logistic Regression model using historical price-derived "
        "setups across multiple stocks. The model estimates the probability "
        "of hitting selected upside target before selected downside stop."
    )

    st.warning(
        "Start with 10–20 stocks for a test. A large historical cross-sectional "
        "dataset takes time to build on free Streamlit Cloud."
    )

    build_col1, build_col2, build_col3 = st.columns(3)

    with build_col1:
        model_target = st.selectbox(
            "Target",
            TARGET_OPTIONS,
            index=1,
            format_func=lambda value: f"+{value}%",
            key="model_target",
        )

        model_stop = st.selectbox(
            "Stop",
            STOP_OPTIONS,
            index=1,
            format_func=lambda value: f"-{value}%",
            key="model_stop",
        )

    with build_col2:
        model_horizon = st.selectbox(
            "Horizon",
            HORIZON_OPTIONS,
            index=1,
            format_func=lambda value: f"{value} Trading Days",
            key="model_horizon",
        )

        model_spacing = st.selectbox(
            "Minimum Setup Spacing",
            [
                10,
                15,
                20,
                30,
                40,
            ],
            index=2,
            key="model_spacing",
        )

    with build_col3:
        model_stock_limit = st.selectbox(
            "Stocks in Training Dataset",
            [
                10,
                20,
                30,
                50,
                75,
                100,
            ],
            index=1,
            key="model_stock_limit",
        )

        model_max_distance = st.selectbox(
            "Maximum Distance Below 52W High",
            [
                -3.0,
                -5.0,
                -7.0,
                -10.0,
                -15.0,
            ],
            index=2,
            format_func=lambda value: (
                f"Within {abs(value):.0f}%"
            ),
            key="model_52w",
        )

    rule_col1, rule_col2, rule_col3 = st.columns(3)

    with rule_col1:
        model_strong_daily = st.checkbox(
            "Require Strong Bullish Daily",
            value=False,
            key="model_daily",
        )

    with rule_col2:
        model_weekly_bullish = st.checkbox(
            "Require Bullish Weekly",
            value=True,
            key="model_weekly",
        )

    with rule_col3:
        model_positive_rs = st.checkbox(
            "Require Positive 3M RS",
            value=True,
            key="model_rs",
        )

    if st.button(
        "Build Phase 3 Probability Model",
        type="primary",
    ):
        progress_bar = st.progress(0)
        status_text = st.empty()

        status_text.caption(
            "Building historical cross-sectional dataset. "
            "This can take time on free hosting."
        )

        with st.spinner(
            "Creating historical setup events..."
        ):
            dataset = build_phase3_dataset(
                stock_universe,
                nifty_50_data,
                model_stock_limit,
                model_target,
                model_stop,
                model_horizon,
                model_spacing,
                model_strong_daily,
                model_weekly_bullish,
                model_positive_rs,
                model_max_distance,
            )

        progress_bar.empty()
        status_text.empty()

        with st.spinner(
            "Training regularized Logistic Regression..."
        ):
            model_package = train_phase3_model(
                dataset
            )

        st.session_state[
            "phase3_dataset"
        ] = dataset

        st.session_state[
            "phase3_model_package"
        ] = model_package

        st.session_state[
            "phase3_settings"
        ] = {
            "target": model_target,
            "stop": model_stop,
            "horizon": model_horizon,
            "spacing": model_spacing,
            "stock_limit": model_stock_limit,
        }

    dataset = st.session_state[
        "phase3_dataset"
    ]

    model_package = st.session_state[
        "phase3_model_package"
    ]

    if dataset is None or dataset.empty:
        st.info(
            "Configure parameters and click Build Phase 3 Probability Model."
        )

    else:
        st.subheader(
            "Cross-Sectional Dataset Summary"
        )

        d1, d2, d3, d4 = st.columns(4)

        d1.metric(
            "Historical Events",
            len(dataset),
        )

        d2.metric(
            "Stocks Included",
            dataset["Symbol"].nunique(),
        )

        d3.metric(
            "Target Hit First",
            int(
                dataset["Target Hit First"].sum()
            ),
        )

        d4.metric(
            "Target Hit Rate",
            f"{dataset['Target Hit First'].mean() * 100:.1f}%",
        )

        if model_package is None:
            st.error(
                "Model could not be trained. You need at least 50 usable events "
                "and both positive and non-positive historical outcomes."
            )

        else:
            st.subheader(
                "Phase 3 Training Diagnostics"
            )

            m1, m2, m3, m4 = st.columns(4)

            m1.metric(
                "Training Rows",
                model_package["rows"],
            )

            m2.metric(
                "Positive Outcomes",
                model_package["positive_rows"],
            )

            m3.metric(
                "Training Accuracy",
                f"{model_package['accuracy'] * 100:.1f}%",
            )

            m4.metric(
                "Training ROC AUC",
                (
                    f"{model_package['auc']:.3f}"
                    if model_package["auc"] is not None
                    else "Not available"
                ),
            )

            st.caption(
                "These are in-sample training diagnostics only. "
                "Phase 4 will add time-based walk-forward validation."
            )

            st.subheader(
                "Feature Influence"
            )

            st.dataframe(
                model_package["coefficients"],
                hide_index=True,
                use_container_width=True,
            )

            st.subheader(
                "Training Dataset Preview"
            )

            st.dataframe(
                dataset
                .sort_values(
                    by="Entry Date",
                    ascending=False,
                )
                .head(100),
                hide_index=True,
                use_container_width=True,
            )

            phase3_csv = dataset.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Download Phase 3 Dataset CSV",
                data=phase3_csv,
                file_name="phase3_cross_sectional_dataset.csv",
                mime="text/csv",
            )


# =============================================================================
# FOOTER
# =============================================================================

st.divider()

st.caption(
    "Data sources: Nifty Indices constituent list and Yahoo Finance. "
    "All patterns, Winner Scores, risk/reward levels, historical studies, "
    "similarity scores, and Phase 3 probabilities are rule-based or model-based "
    "research tools. They can be incomplete, delayed, or incorrect and do not "
    "constitute investment advice, recommendations, or guaranteed outcomes."
)
