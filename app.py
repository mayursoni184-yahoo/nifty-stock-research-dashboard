import io
import warnings
import requests
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    log_loss,
    brier_score_loss,
)

warnings.filterwarnings("ignore")


# =============================================================================
# STREAMLIT CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Nifty Total Market Probability Dashboard",
    page_icon="🎯",
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
            font-size: 2.3rem;
            font-weight: 800;
            color: #111827;
            margin-bottom: 0.15rem;
        }

        .sub-title {
            color: #6b7280;
            font-size: 1rem;
            margin-bottom: 1.2rem;
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

TARGET_OPTIONS = [5, 10, 15]
STOP_OPTIONS = [5, 8, 10, 12]
HORIZON_OPTIONS = [40, 60, 90]

FEATURE_COLUMNS = [
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
    """Format a price as Indian rupees."""

    value = safe_number(value)

    if value is None:
        return "Not available"

    return f"₹{value:,.2f}"


def format_percent(value):
    """Format a signed percentage."""

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


def classify_probability(probability):
    """Classify a probability estimate into a readable label."""

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


def model_confidence_label(training_rows, positive_rows):
    """Return simple confidence label based on sample size."""

    if training_rows < 50:
        return "Insufficient sample"

    if positive_rows < 10:
        return "Insufficient positive outcomes"

    if training_rows < 150:
        return "Low"

    if training_rows < 400:
        return "Medium"

    return "Higher sample confidence"


# =============================================================================
# NIFTY TOTAL MARKET UNIVERSE
# =============================================================================

@st.cache_data(ttl=21600, show_spinner=False)
def get_nifty_total_market_members():
    """
    Load Nifty Total Market constituents.

    Uses a fallback list if the official file is not accessible
    from Streamlit Cloud.
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
                "Too few valid index constituents were loaded."
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
            "Could not load live Nifty Total Market constituents. "
            f"Using fallback universe. Reason: {type(error).__name__}: {error}"
        )

        return fallback, warning


# =============================================================================
# DATA DOWNLOAD FUNCTIONS
# =============================================================================

@st.cache_data(ttl=900, show_spinner=False)
def fetch_price_data(ticker, period="5y"):
    """Fetch OHLCV data from Yahoo Finance."""

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


@st.cache_data(ttl=43200, show_spinner=False)
def fetch_fundamentals(ticker):
    """Fetch available summary fundamental data from Yahoo Finance."""

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
        info = ticker_object.get_info()

        return {
            field: info.get(field)
            for field in fields
        }

    except Exception:
        return {}


# =============================================================================
# TIMEFRAME AND TREND FUNCTIONS
# =============================================================================

def resample_ohlcv(data, timeframe):
    """Resample daily OHLCV data into weekly or monthly candles."""

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
    Calculate moving-average trend.

    Strong bullish:
    Close > MA20 > MA50 > MA200.

    Bullish:
    Close > MA20 > MA50.

    Bearish:
    Close < MA20 < MA50.
    """

    if data is None or len(data) < 55:
        return "Insufficient data"

    close = float(data["close"].iloc[-1])

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

        if close > ma20 > ma50 > ma200:
            return "Strong bullish"

    if close > ma20 > ma50:
        return "Bullish"

    if close < ma20 < ma50:
        return "Bearish"

    return "Neutral / consolidating"


def trend_to_score(trend):
    """Convert a trend label into numerical model feature."""

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
    """Calculate Daily, Weekly and Monthly trend alignment."""

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
# RELATIVE STRENGTH FUNCTIONS
# =============================================================================

def align_stock_benchmark(
    stock_data,
    benchmark_data,
):
    """Align stock and benchmark data on common dates."""

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

    joined = stock_close.join(
        benchmark_close,
        how="inner",
    )

    return joined.dropna()


def calculate_return_from_close(
    close_series,
    days,
):
    """Calculate return from a close-price series."""

    if len(close_series) <= days:
        return None

    current = float(close_series.iloc[-1])
    old = float(close_series.iloc[-days - 1])

    if old == 0:
        return None

    return (
        current / old - 1
    ) * 100


def calculate_relative_strength(
    stock_data,
    benchmark_data,
):
    """Calculate RS status and returns versus Nifty 50."""

    aligned = align_stock_benchmark(
        stock_data,
        benchmark_data,
    )

    empty_result = {
        "status": "Insufficient data",
        "rs_trend": "Unavailable",
        "rs_line": pd.Series(dtype=float),
        "relative_1m": None,
        "relative_3m": None,
        "relative_6m": None,
        "relative_12m": None,
    }

    if len(aligned) < 30:
        return empty_result

    stock_close = aligned["stock_close"]
    benchmark_close = aligned["benchmark_close"]

    stock_1m = calculate_return_from_close(
        stock_close,
        21,
    )

    stock_3m = calculate_return_from_close(
        stock_close,
        63,
    )

    stock_6m = calculate_return_from_close(
        stock_close,
        126,
    )

    stock_12m = calculate_return_from_close(
        stock_close,
        252,
    )

    benchmark_1m = calculate_return_from_close(
        benchmark_close,
        21,
    )

    benchmark_3m = calculate_return_from_close(
        benchmark_close,
        63,
    )

    benchmark_6m = calculate_return_from_close(
        benchmark_close,
        126,
    )

    benchmark_12m = calculate_return_from_close(
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

    if len(rs_line) >= 63:
        current_rs = float(rs_line.iloc[-1])
        old_rs = float(rs_line.iloc[-63])

        if current_rs > old_rs * 1.03:
            rs_trend = "Rising"

        elif current_rs < old_rs * 0.97:
            rs_trend = "Falling"

        else:
            rs_trend = "Flat"

    else:
        rs_trend = "Unavailable"

    relative_returns = [
        value
        for value in [
            relative_1m,
            relative_3m,
            relative_6m,
            relative_12m,
        ]
        if value is not None
    ]

    if len(relative_returns) < 2:
        status = "Insufficient data"

    else:
        positive_count = sum(
            value > 0
            for value in relative_returns
        )

        strong_count = sum(
            value > 5
            for value in relative_returns
        )

        negative_count = sum(
            value < 0
            for value in relative_returns
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
        "relative_1m": relative_1m,
        "relative_3m": relative_3m,
        "relative_6m": relative_6m,
        "relative_12m": relative_12m,
    }


# =============================================================================
# TECHNICAL FEATURE FUNCTIONS
# =============================================================================

def calculate_atr_percent(
    data,
    period=14,
):
    """Calculate ATR as percentage of close price."""

    if data is None or len(data) < period + 1:
        return None

    high = data["high"]
    low = data["low"]
    close = data["close"]

    prior_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - prior_close).abs(),
            (low - prior_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = true_range.rolling(period).mean().iloc[-1]

    if pd.isna(atr):
        return None

    current_close = float(close.iloc[-1])

    if current_close == 0:
        return None

    return (
        float(atr) / current_close
    ) * 100


def calculate_current_features(
    stock_data,
    benchmark_data,
):
    """
    Extract current price-derived model features.

    No fundamentals are included in model features because this
    Phase 3 model remains price-data based.
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

    relative_strength = calculate_relative_strength(
        stock_data,
        benchmark_data,
    )

    current_close = float(
        stock_data["close"].iloc[-1]
    )

    high_52w = float(
        stock_data["high"]
        .tail(252)
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
        (current_close / ma50 - 1) * 100
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

        current_volume = float(
            stock_data["volume"].iloc[-1]
        )

        if average_volume != 0:
            volume_change = (
                current_volume / average_volume - 1
            ) * 100

    momentum_1m = calculate_return_from_close(
        stock_data["close"],
        21,
    )

    atr_percent = calculate_atr_percent(
        stock_data,
        14,
    )

    return {
        "daily_trend_score": trend_to_score(
            daily_trend
        ),
        "weekly_trend_score": trend_to_score(
            weekly_trend
        ),
        "rs_3m": relative_strength.get(
            "relative_3m"
        ),
        "distance_52w_high": distance_52w_high,
        "volume_change": volume_change,
        "atr_percent": atr_percent,
        "momentum_1m": momentum_1m,
        "distance_ma50": distance_ma50,
        "daily_trend_label": daily_trend,
        "weekly_trend_label": weekly_trend,
        "rs_status": relative_strength.get(
            "status"
        ),
        "rs_trend": relative_strength.get(
            "rs_trend"
        ),
    }


def calculate_historical_features(
    stock_data,
    benchmark_data,
    index,
):
    """
    Extract historical model features using only data available
    on or before index. This prevents future leakage.
    """

    if index < 260:
        return None

    historical_stock = stock_data.iloc[
        :index + 1
    ].copy()

    event_date = historical_stock.index[-1]

    historical_benchmark = benchmark_data[
        benchmark_data.index <= event_date
    ].copy()

    if historical_benchmark.empty:
        return None

    return calculate_current_features(
        historical_stock,
        historical_benchmark,
    )


def qualifies_historical_setup(
    features,
    require_strong_daily,
    require_weekly_bullish,
    require_positive_rs,
    max_distance_52w,
    minimum_volume_change,
):
    """
    Determine whether a historical feature snapshot qualifies
    as Phase 1/Phase 3 bullish event.
    """

    if features is None:
        return False

    daily_label = features.get(
        "daily_trend_label"
    )

    weekly_label = features.get(
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
            daily_label == "Strong bullish"
        )
    else:
        daily_ok = daily_label in [
            "Strong bullish",
            "Bullish",
        ]

    if require_weekly_bullish:
        weekly_ok = weekly_label in [
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
        and distance_52w >= max_distance_52w
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


# =============================================================================
# TRIPLE-BARRIER OUTCOME FUNCTIONS
# =============================================================================

def evaluate_triple_barrier(
    stock_data,
    entry_index,
    target_percent,
    stop_percent,
    horizon_days,
):
    """
    Evaluate target, stop and time barrier from a historical entry date.

    Target Hit First: price high touches target before low touches stop.
    Stop Hit First: price low touches stop before high touches target.
    Time Expired: neither occurs inside horizon.
    Ambiguous Same Day: both are touched within one daily candle.
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

    last_index = min(
        entry_index + horizon_days,
        len(stock_data) - 1,
    )

    if last_index <= entry_index:
        return None

    future = stock_data.iloc[
        entry_index + 1:last_index + 1
    ].copy()

    maximum_high = float(
        future["high"].max()
    )

    minimum_low = float(
        future["low"].min()
    )

    mfe_percent = (
        maximum_high / entry_price - 1
    ) * 100

    mae_percent = (
        minimum_low / entry_price - 1
    ) * 100

    outcome = "Time Expired"
    outcome_date = future.index[-1]
    days_to_outcome = len(future)

    for day_number, (date, row) in enumerate(
        future.iterrows(),
        start=1,
    ):
        target_hit = float(row["high"]) >= target_price
        stop_hit = float(row["low"]) <= stop_price

        if target_hit and stop_hit:
            outcome = "Ambiguous Same Day"
            outcome_date = date
            days_to_outcome = day_number
            break

        if target_hit:
            outcome = "Target Hit First"
            outcome_date = date
            days_to_outcome = day_number
            break

        if stop_hit:
            outcome = "Stop Hit First"
            outcome_date = date
            days_to_outcome = day_number
            break

    final_close = float(
        future["close"].iloc[-1]
    )

    forward_return = (
        final_close / entry_price - 1
    ) * 100

    return {
        "entry_date": entry_date,
        "entry_price": entry_price,
        "target_price": target_price,
        "stop_price": stop_price,
        "outcome": outcome,
        "outcome_date": outcome_date,
        "days_to_outcome": days_to_outcome,
        "final_close": final_close,
        "forward_return": forward_return,
        "mfe_percent": mfe_percent,
        "mae_percent": mae_percent,
        "target_hit_first": (
            1
            if outcome == "Target Hit First"
            else 0
        ),
    }


# =============================================================================
# PHASE 3 CROSS-SECTIONAL DATASET CREATION
# =============================================================================

@st.cache_data(ttl=43200, show_spinner=False)
def build_cross_sectional_dataset(
    universe_records,
    benchmark_data,
    stock_limit,
    target_percent,
    stop_percent,
    horizon_days,
    setup_spacing_days,
    require_strong_daily,
    require_weekly_bullish,
    require_positive_rs,
    max_distance_52w,
    minimum_volume_change,
):
    """
    Create a cross-sectional historical setup dataset.

    For each selected stock:
    - retrieve 5-year OHLCV history
    - evaluate historical setup conditions
    - record current-at-that-time feature values
    - label outcome through triple-barrier evaluation

    This is intentionally limited by stock_limit for free hosting.
    """

    if benchmark_data is None or benchmark_data.empty:
        return pd.DataFrame()

    rows = []

    selected_universe = universe_records.head(
        stock_limit
    ).copy()

    for _, record in selected_universe.iterrows():
        symbol = record["Symbol"]
        ticker = record["Ticker"]

        stock_data = fetch_price_data(
            ticker,
            "5y",
        )

        if stock_data.empty:
            continue

        if len(stock_data) < 400:
            continue

        final_index = (
            len(stock_data)
            - horizon_days
            - 1
        )

        last_event_index = -setup_spacing_days

        for index in range(
            260,
            final_index,
        ):
            if (
                index - last_event_index
                < setup_spacing_days
            ):
                continue

            features = calculate_historical_features(
                stock_data,
                benchmark_data,
                index,
            )

            if not qualifies_historical_setup(
                features=features,
                require_strong_daily=require_strong_daily,
                require_weekly_bullish=require_weekly_bullish,
                require_positive_rs=require_positive_rs,
                max_distance_52w=max_distance_52w,
                minimum_volume_change=minimum_volume_change,
            ):
                continue

            outcome = evaluate_triple_barrier(
                stock_data=stock_data,
                entry_index=index,
                target_percent=target_percent,
                stop_percent=stop_percent,
                horizon_days=horizon_days,
            )

            if outcome is None:
                continue

            if outcome["outcome"] == "Ambiguous Same Day":
                continue

            row = {
                "symbol": symbol,
                "ticker": ticker,
                "entry_date": outcome["entry_date"],
                "entry_price": outcome["entry_price"],
                "outcome": outcome["outcome"],
                "target_hit_first": outcome[
                    "target_hit_first"
                ],
                "forward_return": outcome[
                    "forward_return"
                ],
                "mfe_percent": outcome[
                    "mfe_percent"
                ],
                "mae_percent": outcome[
                    "mae_percent"
                ],
                "days_to_outcome": outcome[
                    "days_to_outcome"
                ],
            }

            for feature in FEATURE_COLUMNS:
                row[feature] = features.get(
                    feature
                )

            rows.append(row)

            last_event_index = index

    if not rows:
        return pd.DataFrame()

    dataset = pd.DataFrame(rows)

    for feature in FEATURE_COLUMNS:
        dataset[feature] = pd.to_numeric(
            dataset[feature],
            errors="coerce",
        )

    dataset = dataset.dropna(
        subset=FEATURE_COLUMNS
    )

    return dataset.reset_index(drop=True)


def train_probability_model(
    dataset,
):
    """
    Train a regularized logistic regression model.

    Target variable:
    1 = Target Hit First
    0 = Stop Hit First or Time Expired

    This Phase 3 version trains on all available cross-sectional
    historical events. Phase 4 will add time-based walk-forward
    validation and probability calibration.
    """

    if dataset is None or dataset.empty:
        return None

    if len(dataset) < 50:
        return None

    target = dataset[
        "target_hit_first"
    ].astype(int)

    if target.nunique() < 2:
        return None

    features = dataset[
        FEATURE_COLUMNS
    ].copy()

    model = Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "logistic_regression",
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

    predicted_probability = model.predict_proba(
        features
    )[:, 1]

    predicted_class = (
        predicted_probability >= 0.50
    ).astype(int)

    accuracy = accuracy_score(
        target,
        predicted_class,
    )

    try:
        auc = roc_auc_score(
            target,
            predicted_probability,
        )
    except Exception:
        auc = None

    try:
        brier = brier_score_loss(
            target,
            predicted_probability,
        )
    except Exception:
        brier = None

    try:
        loss = log_loss(
            target,
            predicted_probability,
        )
    except Exception:
        loss = None

    scaler = model.named_steps["scaler"]
    logistic = model.named_steps[
        "logistic_regression"
    ]

    coefficients = pd.DataFrame(
        {
            "Feature": FEATURE_COLUMNS,
            "Coefficient": logistic.coef_[0],
            "Absolute Impact": np.abs(
                logistic.coef_[0]
            ),
            "Feature Mean": scaler.mean_,
        }
    ).sort_values(
        by="Absolute Impact",
        ascending=False,
    )

    return {
        "model": model,
        "features": features,
        "target": target,
        "predicted_probability": predicted_probability,
        "accuracy": accuracy,
        "auc": auc,
        "brier": brier,
        "log_loss": loss,
        "coefficients": coefficients,
        "training_rows": len(dataset),
        "positive_rows": int(target.sum()),
        "negative_rows": int(
            len(target) - target.sum()
        ),
    }


def predict_current_probability(
    model_package,
    current_features,
):
    """
    Predict target-before-stop probability for current stock state.
    """

    if model_package is None:
        return None

    input_row = {}

    for feature in FEATURE_COLUMNS:
        value = safe_number(
            current_features.get(feature)
        )

        if value is None:
            return None

        input_row[feature] = value

    input_dataframe = pd.DataFrame(
        [input_row]
    )

    probability = model_package[
        "model"
    ].predict_proba(
        input_dataframe
    )[0, 1]

    return {
        "probability": float(probability),
        "classification": classify_probability(
            probability
        ),
        "input_features": input_dataframe,
    }


# =============================================================================
# CHART FUNCTIONS
# =============================================================================

def create_candlestick_chart(
    data,
    title,
):
    """Create a candlestick chart with MA20, MA50 and volume."""

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

    figure.update_layout(
        title=title,
        height=620,
        template="plotly_white",
        xaxis_rangeslider_visible=False,
        legend_orientation="h",
    )

    return figure


def create_rs_chart(
    rs_line,
    symbol,
):
    """Create Relative Strength line chart."""

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


def create_probability_distribution_chart(
    dataset,
    predicted_probability,
):
    """Create target outcome distribution chart."""

    figure = go.Figure()

    if dataset is not None and not dataset.empty:
        outcome_counts = (
            dataset["outcome"]
            .value_counts()
            .reset_index()
        )

        outcome_counts.columns = [
            "Outcome",
            "Count",
        ]

        colors = {
            "Target Hit First": "#16a34a",
            "Stop Hit First": "#dc2626",
            "Time Expired": "#f59e0b",
        }

        figure.add_trace(
            go.Bar(
                x=outcome_counts["Outcome"],
                y=outcome_counts["Count"],
                marker_color=[
                    colors.get(
                        value,
                        "#6b7280",
                    )
                    for value in outcome_counts[
                        "Outcome"
                    ]
                ],
                name="Historical Outcomes",
            )
        )

    if predicted_probability is not None:
        figure.add_annotation(
            text=(
                "Current Model Estimate: "
                f"{predicted_probability * 100:.1f}%"
            ),
            xref="paper",
            yref="paper",
            x=0.5,
            y=1.14,
            showarrow=False,
            font=dict(
                size=16,
                color="#2563eb",
            ),
        )

    figure.update_layout(
        title="Cross-Sectional Historical Outcome Distribution",
        height=380,
        template="plotly_white",
        yaxis_title="Historical Setup Count",
    )

    return figure


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
        🎯 Nifty Total Market Probability Research Dashboard
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="sub-title">
        Phase 3 Cross-Sectional Probability Model •
        Historical Setup Research • Relative Strength •
        Trend Alignment • Rule-Based Market Analysis
    </div>
    """,
    unsafe_allow_html=True,
)

if universe_error:
    st.warning(universe_error)

if stock_universe.empty:
    st.error(
        "No stock universe is available."
    )
    st.stop()

if nifty_50_data.empty:
    st.warning(
        "Nifty 50 benchmark data is unavailable. "
        "Probability-model features requiring Relative Strength may not work."
    )


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.header("🔍 Dashboard Controls")

    mode = st.radio(
        "Mode",
        [
            "Stock Probability Research",
            "Phase 3 Model Builder",
        ],
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
        "Price cache: 15 minutes\n\n"
        "Fundamental cache: 12 hours\n\n"
        "Cross-sectional dataset cache: 12 hours\n\n"
        "Constituent data cache: 6 hours"
    )


# =============================================================================
# PHASE 3 MODEL BUILDER
# =============================================================================

if mode == "Phase 3 Model Builder":
    st.subheader(
        "Phase 3: Cross-Sectional Probability Model Builder"
    )

    st.caption(
        "This creates a historical dataset across multiple stocks and "
        "trains a Logistic Regression model to estimate the probability "
        "of reaching a target before a stop within a specified horizon."
    )

    st.warning(
        "This is a research prototype. It uses historical OHLCV-derived "
        "features only. It does not include point-in-time fundamentals, "
        "transaction costs, slippage, survivorship-bias adjustments, "
        "or out-of-sample validation. Phase 4 will address validation."
    )

    setting_col1, setting_col2, setting_col3 = st.columns(3)

    with setting_col1:
        model_target = st.selectbox(
            "Target",
            TARGET_OPTIONS,
            index=1,
            format_func=lambda value: f"+{value}%",
        )

        model_stop = st.selectbox(
            "Stop",
            STOP_OPTIONS,
            index=1,
            format_func=lambda value: f"-{value}%",
        )

    with setting_col2:
        model_horizon = st.selectbox(
            "Horizon",
            HORIZON_OPTIONS,
            index=1,
            format_func=lambda value: (
                f"{value} Trading Days"
            ),
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
            format_func=lambda value: (
                f"{value} Trading Days"
            ),
        )

    with setting_col3:
        model_stock_limit = st.selectbox(
            "Stocks for Cross-Sectional Dataset",
            [
                10,
                20,
                30,
                50,
                75,
                100,
            ],
            index=1,
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
        )

    setup_col1, setup_col2, setup_col3 = st.columns(3)

    with setup_col1:
        model_strong_daily = st.checkbox(
            "Require Strong Bullish Daily",
            value=False,
        )

    with setup_col2:
        model_weekly_bullish = st.checkbox(
            "Require Bullish Weekly",
            value=True,
        )

    with setup_col3:
        model_positive_rs = st.checkbox(
            "Require Positive 3M Relative Strength",
            value=True,
        )

    require_volume = st.checkbox(
        "Require Volume Confirmation",
        value=False,
    )

    if require_volume:
        model_minimum_volume = st.selectbox(
            "Minimum Volume Above 20D Average",
            [
                0,
                10,
                25,
                50,
            ],
            index=1,
            format_func=lambda value: (
                f"+{value}%"
            ),
        )
    else:
        model_minimum_volume = None

    st.info(
        "Start with 10 or 20 stocks on free Streamlit Cloud. "
        "The model-building process downloads historical price data and "
        "may take several minutes. After the first run, results are cached."
    )

    if st.button(
        "Build Phase 3 Cross-Sectional Model",
        type="primary",
    ):
        with st.spinner(
            "Building cross-sectional historical setup dataset..."
        ):
            dataset = build_cross_sectional_dataset(
                universe_records=stock_universe,
                benchmark_data=nifty_50_data,
                stock_limit=model_stock_limit,
                target_percent=model_target,
                stop_percent=model_stop,
                horizon_days=model_horizon,
                setup_spacing_days=model_spacing,
                require_strong_daily=model_strong_daily,
                require_weekly_bullish=model_weekly_bullish,
                require_positive_rs=model_positive_rs,
                max_distance_52w=model_max_distance,
                minimum_volume_change=model_minimum_volume,
            )

        with st.spinner(
            "Training Logistic Regression probability model..."
        ):
            model_package = train_probability_model(
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
            "strong_daily": model_strong_daily,
            "weekly_bullish": model_weekly_bullish,
            "positive_rs": model_positive_rs,
            "max_distance": model_max_distance,
            "minimum_volume": model_minimum_volume,
        }

    dataset = st.session_state.get(
        "phase3_dataset"
    )

    model_package = st.session_state.get(
        "phase3_model_package"
    )

    settings = st.session_state.get(
        "phase3_settings"
    )

    if dataset is not None and not dataset.empty:
        st.divider()

        st.subheader(
            "Cross-Sectional Dataset Summary"
        )

        dataset_col1, dataset_col2, dataset_col3, dataset_col4 = (
            st.columns(4)
        )

        dataset_col1.metric(
            "Historical Setup Rows",
            len(dataset),
        )

        dataset_col2.metric(
            "Stocks Represented",
            dataset["symbol"].nunique(),
        )

        dataset_col3.metric(
            "Target Hit First",
            int(
                dataset["target_hit_first"].sum()
            ),
        )

        dataset_col4.metric(
            "Target Hit Rate",
            (
                f"{dataset['target_hit_first'].mean() * 100:.1f}%"
            ),
        )

        st.caption(
            "Model dataset configuration: "
            f"Target +{settings['target']}%, "
            f"Stop -{settings['stop']}%, "
            f"Horizon {settings['horizon']} trading days, "
            f"Setup spacing {settings['spacing']} trading days."
        )

        if model_package is None:
            st.error(
                "The dataset was built, but the probability model could "
                "not be trained. This generally happens when there are "
                "fewer than 50 valid rows or only one outcome class."
            )

        else:
            st.subheader(
                "Model Training Diagnostics"
            )

            diagnostic_col1, diagnostic_col2, diagnostic_col3, diagnostic_col4 = (
                st.columns(4)
            )

            diagnostic_col1.metric(
                "Training Rows",
                model_package["training_rows"],
            )

            diagnostic_col2.metric(
                "Positive Outcomes",
                model_package["positive_rows"],
            )

            diagnostic_col3.metric(
                "Training Accuracy",
                f"{model_package['accuracy'] * 100:.1f}%",
            )

            diagnostic_col4.metric(
                "Confidence",
                model_confidence_label(
                    model_package["training_rows"],
                    model_package["positive_rows"],
                ),
            )

            quality_col1, quality_col2, quality_col3 = (
                st.columns(3)
            )

            quality_col1.metric(
                "ROC AUC",
                (
                    f"{model_package['auc']:.3f}"
                    if model_package["auc"] is not None
                    else "Not available"
                ),
            )

            quality_col2.metric(
                "Brier Score",
                (
                    f"{model_package['brier']:.3f}"
                    if model_package["brier"] is not None
                    else "Not available"
                ),
            )

            quality_col3.metric(
                "Log Loss",
                (
                    f"{model_package['log_loss']:.3f}"
                    if model_package["log_loss"] is not None
                    else "Not available"
                ),
            )

            st.caption(
                "These diagnostics are in-sample training diagnostics only. "
                "They should not be treated as evidence of future performance. "
                "Phase 4 will provide time-based out-of-sample validation."
            )

            st.subheader(
                "Model Feature Influence"
            )

            st.dataframe(
                model_package["coefficients"],
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Coefficient": st.column_config.NumberColumn(
                        "Coefficient",
                        format="%.4f",
                    ),
                    "Absolute Impact": st.column_config.ProgressColumn(
                        "Relative Impact",
                        min_value=0,
                        max_value=max(
                            1,
                            float(
                                model_package[
                                    "coefficients"
                                ][
                                    "Absolute Impact"
                                ].max()
                            ),
                        ),
                        format="%.4f",
                    ),
                    "Feature Mean": st.column_config.NumberColumn(
                        "Training Mean",
                        format="%.3f",
                    ),
                },
            )

            dataset_preview = dataset.copy()

            dataset_preview["entry_date"] = pd.to_datetime(
                dataset_preview["entry_date"]
            )

            dataset_preview = dataset_preview.sort_values(
                by="entry_date",
                ascending=False,
            )

            st.subheader(
                "Dataset Preview"
            )

            st.dataframe(
                dataset_preview.head(100),
                hide_index=True,
                use_container_width=True,
            )

            csv_data = dataset.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "Download Phase 3 Training Dataset CSV",
                data=csv_data,
                file_name="phase3_cross_sectional_training_dataset.csv",
                mime="text/csv",
            )

    else:
        st.info(
            "Configure the dataset settings and click "
            "Build Phase 3 Cross-Sectional Model."
        )


# =============================================================================
# STOCK PROBABILITY RESEARCH
# =============================================================================

else:
    st.subheader(
        "Stock Probability Research"
    )

    st.caption(
        "Use the trained Phase 3 cross-sectional model to estimate the "
        "historical probability that the current price-derived setup reaches "
        "its selected target before its selected stop within the model horizon."
    )

    model_package = st.session_state.get(
        "phase3_model_package"
    )

    model_settings = st.session_state.get(
        "phase3_settings"
    )

    if model_package is None or model_settings is None:
        st.warning(
            "Build the Phase 3 Cross-Sectional Probability Model first. "
            "Open the sidebar and select Phase 3 Model Builder."
        )
        st.stop()

    symbols = sorted(
        stock_universe["Symbol"].tolist()
    )

    default_index = (
        symbols.index("RELIANCE")
        if "RELIANCE" in symbols
        else 0
    )

    selected_symbol = st.selectbox(
        "Select Stock",
        symbols,
        index=default_index,
    )

    selected_record = stock_universe[
        stock_universe["Symbol"] == selected_symbol
    ].iloc[0]

    selected_ticker = selected_record["Ticker"]

    with st.spinner(
        f"Loading current price data for {selected_symbol}..."
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
            "Price data is unavailable for this stock."
        )
        st.stop()

    market_cap = safe_number(
        fundamentals.get("marketCap"),
        0,
    ) / 10000000

    if market_cap < minimum_market_cap:
        st.warning(
            f"{selected_symbol} is below the chosen ₹{minimum_market_cap:,.0f} "
            "crore market-cap filter. You can still view research details."
        )

    current_features = calculate_current_features(
        stock_data,
        nifty_50_data,
    )

    prediction = predict_current_probability(
        model_package,
        current_features,
    )

    current_price = float(
        stock_data["close"].iloc[-1]
    )

    daily_data = stock_data

    weekly_data = resample_ohlcv(
        stock_data,
        "Weekly",
    )

    monthly_data = resample_ohlcv(
        stock_data,
        "Monthly",
    )

    alignment = calculate_multitimeframe_alignment(
        daily_data,
        weekly_data,
        monthly_data,
    )

    rs_data = calculate_relative_strength(
        stock_data,
        nifty_50_data,
    )

    top_col1, top_col2, top_col3, top_col4, top_col5 = (
        st.columns(5)
    )

    top_col1.metric(
        "Last Close",
        format_price(current_price),
    )

    top_col2.metric(
        "Market Cap",
        format_market_cap(
            fundamentals.get("marketCap")
        ),
    )

    top_col3.metric(
        "RS Status",
        rs_data["status"],
    )

    top_col4.metric(
        "MTF Alignment",
        (
            f"{alignment['score']}/10"
            if alignment["score"] is not None
            else "Not available"
        ),
    )

    top_col5.metric(
        "Model Horizon",
        f"{model_settings['horizon']} Days",
        (
            f"+{model_settings['target']}% / "
            f"-{model_settings['stop']}%"
        ),
    )

    st.caption(
        f"{selected_record['Company Name']} • "
        f"{selected_record['Industry']}"
    )

    (
        probability_tab,
        features_tab,
        technical_tab,
        relative_strength_tab,
        model_tab,
    ) = st.tabs(
        [
            "🎯 Probability Estimate",
            "Current Model Features",
            "Technical Chart",
            "Relative Strength",
            "Model Context",
        ]
    )

    # =========================================================================
    # PROBABILITY TAB
    # =========================================================================

    with probability_tab:
        st.subheader(
            "Phase 3 Cross-Sectional Probability Estimate"
        )

        st.caption(
            "Estimate: probability of target being reached before stop "
            "within the selected model horizon, based on historical "
            "cross-sectional price-derived setups."
        )

        if prediction is None:
            st.warning(
                "The current setup has missing model features. "
                "A probability estimate cannot be calculated."
            )

        else:
            probability_percent = (
                prediction["probability"] * 100
            )

            probability_class = prediction[
                "classification"
            ]

            if probability_percent >= 70:
                card_class = "green-card"

            elif probability_percent >= 60:
                card_class = "blue-card"

            elif probability_percent >= 50:
                card_class = "orange-card"

            else:
                card_class = "red-card"

            st.markdown(
                f"""
                <div class="research-card {card_class}">
                    <h2>{probability_percent:.1f}%</h2>
                    <h3>{probability_class}</h3>
                    <p>
                        Estimated historical probability of reaching
                        +{model_settings["target"]}% before reaching
                        -{model_settings["stop"]}% within
                        {model_settings["horizon"]} trading days.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            data_col1, data_col2, data_col3, data_col4 = (
                st.columns(4)
            )

            data_col1.metric(
                "Training Events",
                model_package["training_rows"],
            )

            data_col2.metric(
                "Positive Historical Events",
                model_package["positive_rows"],
            )

            data_col3.metric(
                "Training Target Rate",
                (
                    f"{model_package['positive_rows'] / model_package['training_rows'] * 100:.1f}%"
                ),
            )

            data_col4.metric(
                "Sample Confidence",
                model_confidence_label(
                    model_package["training_rows"],
                    model_package["positive_rows"],
                ),
            )

            dataset = st.session_state.get(
                "phase3_dataset"
            )

            st.plotly_chart(
                create_probability_distribution_chart(
                    dataset,
                    prediction["probability"],
                ),
                use_container_width=True,
            )

            st.warning(
                "This is not a forecast or guaranteed probability. "
                "It is based on a model trained on historical observations "
                "and has not yet undergone Phase 4 walk-forward validation."
            )

    # =========================================================================
    # FEATURE TAB
    # =========================================================================

    with features_tab:
        st.subheader(
            "Current Price-Derived Model Features"
        )

        st.caption(
            "These are the values currently supplied to the Phase 3 model."
        )

        feature_display = pd.DataFrame(
            [
                [
                    "Daily Trend Score",
                    current_features.get(
                        "daily_trend_score"
                    ),
                    current_features.get(
                        "daily_trend_label"
                    ),
                ],
                [
                    "Weekly Trend Score",
                    current_features.get(
                        "weekly_trend_score"
                    ),
                    current_features.get(
                        "weekly_trend_label"
                    ),
                ],
                [
                    "Relative Strength, 3M",
                    current_features.get("rs_3m"),
                    current_features.get(
                        "rs_status"
                    ),
                ],
                [
                    "Distance from 52W High",
                    current_features.get(
                        "distance_52w_high"
                    ),
                    "Closer to high is generally stronger",
                ],
                [
                    "Volume vs 20D Average",
                    current_features.get(
                        "volume_change"
                    ),
                    "Positive indicates higher-than-average volume",
                ],
                [
                    "ATR %",
                    current_features.get(
                        "atr_percent"
                    ),
                    "Daily volatility measure",
                ],
                [
                    "1M Momentum",
                    current_features.get(
                        "momentum_1m"
                    ),
                    "21-trading-day price return",
                ],
                [
                    "Distance from MA50",
                    current_features.get(
                        "distance_ma50"
                    ),
                    "Positive indicates price above MA50",
                ],
            ],
            columns=[
                "Model Feature",
                "Current Value",
                "Interpretation",
            ],
        )

        st.dataframe(
            feature_display,
            hide_index=True,
            use_container_width=True,
        )

    # =========================================================================
    # TECHNICAL CHART TAB
    # =========================================================================

    with technical_tab:
        st.subheader(
            "Daily Technical Chart"
        )

        st.plotly_chart(
            create_candlestick_chart(
                stock_data,
                f"{selected_symbol} Daily Chart",
            ),
            use_container_width=True,
        )

        technical_col1, technical_col2, technical_col3 = (
            st.columns(3)
        )

        technical_col1.metric(
            "Daily Trend",
            calculate_overall_trend(
                daily_data
            ),
        )

        technical_col2.metric(
            "Weekly Trend",
            calculate_overall_trend(
                weekly_data
            ),
        )

        technical_col3.metric(
            "Monthly Trend",
            calculate_overall_trend(
                monthly_data
            ),
        )

    # =========================================================================
    # RELATIVE STRENGTH TAB
    # =========================================================================

    with relative_strength_tab:
        st.subheader(
            "Relative Strength versus Nifty 50"
        )

        rs_col1, rs_col2, rs_col3 = st.columns(3)

        rs_col1.metric(
            "RS Status",
            rs_data["status"],
        )

        rs_col2.metric(
            "RS Trend",
            rs_data["rs_trend"],
        )

        rs_col3.metric(
            "Relative 3M Return",
            format_percent(
                rs_data["relative_3m"]
            ),
        )

        st.plotly_chart(
            create_rs_chart(
                rs_data["rs_line"],
                selected_symbol,
            ),
            use_container_width=True,
        )

    # =========================================================================
    # MODEL CONTEXT TAB
    # =========================================================================

    with model_tab:
        st.subheader(
            "Phase 3 Model Context"
        )

        st.markdown(
            f"""
            | Parameter | Current Model Setting |
            |---|---|
            | Target Barrier | +{model_settings["target"]}% |
            | Stop Barrier | -{model_settings["stop"]}% |
            | Time Barrier | {model_settings["horizon"]} Trading Days |
            | Setup Spacing | {model_settings["spacing"]} Trading Days |
            | Stocks Used | {model_settings["stock_limit"]} |
            | Strong Daily Required | {model_settings["strong_daily"]} |
            | Bullish Weekly Required | {model_settings["weekly_bullish"]} |
            | Positive 3M RS Required | {model_settings["positive_rs"]} |
            | Maximum 52W-High Distance | {model_settings["max_distance"]}% |
            """
        )

        st.markdown("### Model Feature Coefficients")

        st.dataframe(
            model_package["coefficients"],
            hide_index=True,
            use_container_width=True,
        )

        st.warning(
            "Coefficient direction is not investment advice. "
            "Positive or negative coefficients can vary materially across "
            "different markets, stocks, training windows, targets, stops, "
            "and horizons. Phase 4 will test whether the model works on "
            "unseen future periods."
        )


# =============================================================================
# FOOTER
# =============================================================================

st.divider()

st.caption(
    "Data sources: Nifty Indices constituent data and Yahoo Finance. "
    "Phase 3 uses Logistic Regression on historical OHLCV-derived "
    "cross-sectional setup events. Probability estimates are model outputs, "
    "not forecasts, recommendations, or guarantees. This model currently "
    "uses in-sample diagnostics only. Phase 4 walk-forward validation is "
    "required before relying on probability estimates for decision support."
)
