import io
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
    page_title="Nifty Total Market Winner Dashboard",
    page_icon="🏆",
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
            background-color: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 16px;
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


# =============================================================================
# GENERAL UTILITIES
# =============================================================================

def safe_number(value, default=None):
    """Converts values safely to float."""

    try:
        if value is None:
            return default

        if pd.isna(value):
            return default

        return float(value)

    except Exception:
        return default


def format_price(value):
    """Formats INR price values."""

    value = safe_number(value)

    if value is None:
        return "Not available"

    return f"₹{value:,.2f}"


def format_percent(value):
    """Formats signed percentage values."""

    value = safe_number(value)

    if value is None:
        return "Not available"

    return f"{value:+.2f}%"


def format_market_cap(value):
    """Formats a market capitalization value into crore rupees."""

    value = safe_number(value)

    if value is None or value <= 0:
        return "Not available"

    return f"₹{value / 10000000:,.0f} Cr"


def calculate_percent_change(current, previous):
    """Calculates percent change safely."""

    current = safe_number(current)
    previous = safe_number(previous)

    if current is None or previous is None:
        return None

    if previous == 0:
        return None

    return (
        (current - previous)
        / abs(previous)
    ) * 100


# =============================================================================
# NIFTY TOTAL MARKET STOCK UNIVERSE
# =============================================================================

@st.cache_data(ttl=21600, show_spinner=False)
def get_nifty_total_market_members():
    """
    Loads current Nifty Total Market constituents.

    Uses a fallback list if the external CSV cannot be retrieved.
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
                "Nifty constituent CSV does not contain Symbol."
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
                "Received too few valid constituents."
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
            "Could not load the live Nifty Total Market constituent file. "
            f"Using fallback stocks. Reason: {type(error).__name__}: {error}"
        )

        return fallback, warning


# =============================================================================
# MARKET DATA
# =============================================================================

@st.cache_data(ttl=900, show_spinner=False)
def fetch_price_data(ticker, period="5y"):
    """Fetches OHLCV data from Yahoo Finance."""

    try:
        ticker_object = yf.Ticker(ticker)

        data = ticker_object.history(
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

        data = data[
            required_columns
        ].dropna()

        data.index = pd.to_datetime(
            data.index
        )

        if getattr(data.index, "tz", None) is not None:
            data.index = (
                data.index.tz_localize(None)
            )

        return data

    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=43200, show_spinner=False)
def fetch_fundamentals(ticker):
    """Fetches available basic company fundamental data."""

    fundamental_fields = [
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

        company_info = ticker_object.get_info()

        return {
            field: company_info.get(field)
            for field in fundamental_fields
        }

    except Exception:
        return {}


# =============================================================================
# PRICE RESAMPLING
# =============================================================================

def resample_ohlcv(data, timeframe):
    """Resamples daily OHLCV data to Weekly or Monthly candles."""

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


# =============================================================================
# TREND ENGINE
# =============================================================================

def calculate_overall_trend(data):
    """
    Returns the overall trend from moving-average alignment.

    Strong bullish:
    Price > MA20 > MA50 > MA200

    Bullish:
    Price > MA20 > MA50

    Bearish:
    Price < MA20 < MA50
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


def calculate_multitimeframe_alignment(
    daily_data,
    weekly_data,
    monthly_data,
):
    """
    Scores Daily, Weekly and Monthly trend alignment.

    Maximum score: 10.
    """

    daily_trend = calculate_overall_trend(
        daily_data
    )

    weekly_trend = calculate_overall_trend(
        weekly_data
    )

    monthly_trend = calculate_overall_trend(
        monthly_data
    )

    if (
        daily_trend == "Insufficient data"
        and weekly_trend == "Insufficient data"
        and monthly_trend == "Insufficient data"
    ):
        return {
            "daily": daily_trend,
            "weekly": weekly_trend,
            "monthly": monthly_trend,
            "score": None,
            "status": "Insufficient data",
            "structure": "Insufficient price history.",
        }

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
            "Daily, Weekly and Monthly trends are aligned upward."
        )

    elif bullish_count >= 2:
        status = "Bullish multi-timeframe alignment"

        structure = (
            "Most major chart timeframes are bullish."
        )

    elif bearish_count >= 2:
        status = "Bearish multi-timeframe alignment"

        structure = (
            "Most major chart timeframes are bearish."
        )

    else:
        status = "Mixed timeframe alignment"

        structure = (
            "Daily, Weekly and Monthly chart structures are mixed."
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

def calculate_period_return(data, trading_days):
    """Calculates return over an approximate number of trading days."""

    if data is None or data.empty:
        return None

    if len(data) <= trading_days:
        return None

    current_close = float(
        data["close"].iloc[-1]
    )

    previous_close = float(
        data["close"].iloc[-trading_days - 1]
    )

    if previous_close == 0:
        return None

    return (
        current_close / previous_close - 1
    ) * 100


def align_stock_and_benchmark(
    stock_data,
    benchmark_data,
):
    """Aligns stock and benchmark price data on common dates."""

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


def calculate_relative_strength(
    stock_data,
    benchmark_data,
):
    """
    Calculates stock performance relative to Nifty 50.

    Relative Return = Stock Return - Nifty 50 Return.
    """

    aligned_data = align_stock_and_benchmark(
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

    if len(aligned_data) < 30:
        return empty_result

    stock_prices = pd.DataFrame(
        {
            "close": aligned_data["stock_close"],
        }
    )

    benchmark_prices = pd.DataFrame(
        {
            "close": aligned_data["benchmark_close"],
        }
    )

    stock_1m = calculate_period_return(
        stock_prices,
        21,
    )

    stock_3m = calculate_period_return(
        stock_prices,
        63,
    )

    stock_6m = calculate_period_return(
        stock_prices,
        126,
    )

    stock_12m = calculate_period_return(
        stock_prices,
        252,
    )

    benchmark_1m = calculate_period_return(
        benchmark_prices,
        21,
    )

    benchmark_3m = calculate_period_return(
        benchmark_prices,
        63,
    )

    benchmark_6m = calculate_period_return(
        benchmark_prices,
        126,
    )

    benchmark_12m = calculate_period_return(
        benchmark_prices,
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

    rs_line = (
        aligned_data["stock_close"]
        / aligned_data["benchmark_close"]
    )

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
        rs_status = "Insufficient data"

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
# TECHNICAL PATTERN DETECTION
# =============================================================================

def find_pivots(values, pivot_type="high", order=3):
    """Finds local price pivots."""

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
    """Creates a standardised technical pattern signal."""

    latest_close = float(
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
        "Current": latest_close,
        "Return %": (
            latest_close / float(level) - 1
        ) * 100,
        "Volume %": volume_change,
        "Pattern Height": pattern_height,
        "Notes": notes,
    }


def detect_patterns(data):
    """
    Detects current technical pattern candidates.

    Pattern Height is included where possible for measured-move targets.
    """

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

    detected = []

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

            height = peak - neckline

            status = (
                "Confirmed"
                if close[-1] < neckline * 0.99
                else "In progress"
            )

            detected.append(
                build_pattern_signal(
                    "Double Top",
                    status,
                    data,
                    neckline,
                    "Bearish",
                    (
                        "Two similar highs. Confirmation requires "
                        "a close below neckline support."
                    ),
                    pattern_height=height,
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

            height = neckline - base

            status = (
                "Confirmed"
                if close[-1] > neckline * 1.01
                else "In progress"
            )

            detected.append(
                build_pattern_signal(
                    "Double Bottom",
                    status,
                    data,
                    neckline,
                    "Bullish",
                    (
                        "Two similar lows. Confirmation requires "
                        "a close above neckline resistance."
                    ),
                    pattern_height=height,
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

            detected.append(
                build_pattern_signal(
                    "Head & Shoulders",
                    status,
                    data,
                    neckline,
                    "Bearish",
                    (
                        "Confirmation requires a close below "
                        "neckline support."
                    ),
                    pattern_height=pattern_height,
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

            detected.append(
                build_pattern_signal(
                    "Inverse Head & Shoulders",
                    status,
                    data,
                    neckline,
                    "Bullish",
                    (
                        "Confirmation requires a close above "
                        "neckline resistance."
                    ),
                    pattern_height=pattern_height,
                )
            )

    # RECTANGLE AND TRIANGLE PATTERNS
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

        range_percent = (
            pattern_height
            / max(resistance, 1)
        )

        if range_percent < 0.15:
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

                detected.append(
                    build_pattern_signal(
                        pattern_name,
                        status,
                        data,
                        level,
                        direction,
                        (
                            "Confirmation requires a close outside "
                            "the pattern range."
                        ),
                        pattern_height=pattern_height,
                    )
                )

    # REVERSAL BOTTOM
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

    candle_body = abs(
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
        and candle_body / candle_range < 0.35
        and latest_close <= previous_low * 1.04
    ):
        detected.append(
            build_pattern_signal(
                "Reversal Bottom",
                "Candidate",
                data,
                latest_low,
                "Bullish",
                (
                    "Hammer-like candle near local low. "
                    "Wait for price confirmation."
                ),
                pattern_height=None,
            )
        )

    # REVERSAL TOP
    if (
        upper_shadow / candle_range > 0.55
        and candle_body / candle_range < 0.35
        and latest_close >= previous_high * 0.96
    ):
        detected.append(
            build_pattern_signal(
                "Reversal Top",
                "Candidate",
                data,
                latest_high,
                "Bearish",
                (
                    "Shooting-star-like candle near local high. "
                    "Wait for price confirmation."
                ),
                pattern_height=None,
            )
        )

    return detected


# =============================================================================
# SUPPORT, RESISTANCE, ATR AND RISK ENGINE
# =============================================================================

def calculate_support_resistance(data):
    """Calculates pivot-based nearby support and resistance."""

    if len(data) < 30:
        support = float(
            data["low"].tail(10).min()
        )

        resistance = float(
            data["high"].tail(10).max()
        )

        return support, resistance

    current_price = float(
        data["close"].iloc[-1]
    )

    low_pivots = find_pivots(
        data["low"].to_numpy(),
        "low",
        3,
    )

    high_pivots = find_pivots(
        data["high"].to_numpy(),
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


def calculate_atr(data, period=14):
    """
    Calculates Average True Range.

    ATR is a volatility measure, not a directional indicator.
    """

    if data is None or len(data) < period + 1:
        return None

    high = data["high"]
    low = data["low"]
    close = data["close"]

    previous_close = close.shift(1)

    true_range_one = high - low

    true_range_two = (
        high - previous_close
    ).abs()

    true_range_three = (
        low - previous_close
    ).abs()

    true_range = pd.concat(
        [
            true_range_one,
            true_range_two,
            true_range_three,
        ],
        axis=1,
    ).max(axis=1)

    atr = true_range.rolling(
        period
    ).mean()

    latest_atr = atr.iloc[-1]

    if pd.isna(latest_atr):
        return None

    return float(latest_atr)


def classify_entry_quality(
    current_price,
    breakout_level,
    direction,
):
    """
    Classifies whether current price is close enough to the level
    to be considered an ideal, acceptable, extended or avoid-chasing entry.

    This is a rule-based research classification, not trade advice.
    """

    current_price = safe_number(current_price)
    breakout_level = safe_number(breakout_level)

    if current_price is None or breakout_level is None:
        return {
            "status": "Insufficient data",
            "distance_percent": None,
            "notes": "Current price or breakout level is unavailable.",
        }

    if breakout_level == 0:
        return {
            "status": "Insufficient data",
            "distance_percent": None,
            "notes": "Breakout level is zero or unavailable.",
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
        status = "Below / Near Breakout Level"

        notes = (
            "Price remains at or below the selected breakout level. "
            "Confirmation should be reviewed."
        )

    elif distance_percent <= 3:
        status = "Ideal Entry Zone"

        notes = (
            "Price is within 3% of the breakout level."
        )

    elif distance_percent <= 7:
        status = "Acceptable Entry Zone"

        notes = (
            "Price is moderately above the breakout level."
        )

    elif distance_percent <= 12:
        status = "Extended"

        notes = (
            "Price has moved materially away from the breakout level."
        )

    else:
        status = "Avoid Chasing"

        notes = (
            "Price is highly extended from the breakout level."
        )

    return {
        "status": status,
        "distance_percent": distance_percent,
        "notes": notes,
    }


def calculate_pattern_target(
    pattern_signal,
):
    """
    Calculates a simple measured-move target.

    Bullish target:
    Breakout Level + Pattern Height

    Bearish target:
    Breakdown Level - Pattern Height
    """

    direction = pattern_signal.get(
        "Direction"
    )

    breakout_level = safe_number(
        pattern_signal.get("Level")
    )

    pattern_height = safe_number(
        pattern_signal.get("Pattern Height")
    )

    if breakout_level is None:
        return None

    if pattern_height is None:
        return None

    if pattern_height <= 0:
        return None

    if direction == "Bullish":
        return breakout_level + pattern_height

    if direction == "Bearish":
        return breakout_level - pattern_height

    return None


def calculate_trade_plan(
    data,
    pattern_signal,
):
    """
    Builds a rule-based risk/reward plan for a detected technical setup.

    This is educational research output only. It is not a recommendation
    to enter, exit, buy, sell, or hold a security.
    """

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
        period=14,
    )

    direction = pattern_signal.get(
        "Direction",
        "Neutral",
    )

    breakout_level = safe_number(
        pattern_signal.get("Level")
    )

    target = calculate_pattern_target(
        pattern_signal
    )

    if breakout_level is None:
        breakout_level = current_price

    entry_quality = classify_entry_quality(
        current_price,
        breakout_level,
        direction,
    )

    if direction == "Bullish":
        technical_stop = support

        atr_stop = (
            current_price - (2 * atr)
            if atr is not None
            else None
        )

        stop_candidates = [
            value
            for value in [
                technical_stop,
                atr_stop,
            ]
            if value is not None
            and value < current_price
        ]

        selected_stop = (
            min(stop_candidates)
            if stop_candidates
            else None
        )

        if target is None:
            target = resistance

            if target <= current_price and atr is not None:
                target = current_price + (3 * atr)

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
            current_price + (2 * atr)
            if atr is not None
            else None
        )

        stop_candidates = [
            value
            for value in [
                technical_stop,
                atr_stop,
            ]
            if value is not None
            and value > current_price
        ]

        selected_stop = (
            max(stop_candidates)
            if stop_candidates
            else None
        )

        if target is None:
            target = support

            if target >= current_price and atr is not None:
                target = current_price - (3 * atr)

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
            "direction": "Neutral",
            "current_price": current_price,
            "breakout_level": breakout_level,
            "support": support,
            "resistance": resistance,
            "atr": atr,
            "entry_quality": entry_quality,
            "technical_stop": None,
            "atr_stop": None,
            "selected_stop": None,
            "target": None,
            "risk_per_share": None,
            "reward_per_share": None,
            "risk_reward_ratio": None,
            "risk_reward_status": "No directional trade plan",
        }

    risk_reward_ratio = None

    if (
        risk_per_share is not None
        and reward_per_share is not None
        and risk_per_share > 0
        and reward_per_share > 0
    ):
        risk_reward_ratio = (
            reward_per_share / risk_per_share
        )

    if risk_reward_ratio is None:
        risk_reward_status = "Insufficient data"

    elif risk_reward_ratio >= 3:
        risk_reward_status = "Excellent"

    elif risk_reward_ratio >= 2:
        risk_reward_status = "Good"

    elif risk_reward_ratio >= 1.5:
        risk_reward_status = "Acceptable"

    else:
        risk_reward_status = "Weak"

    return {
        "direction": direction,
        "current_price": current_price,
        "breakout_level": breakout_level,
        "support": support,
        "resistance": resistance,
        "atr": atr,
        "entry_quality": entry_quality,
        "technical_stop": technical_stop,
        "atr_stop": atr_stop,
        "selected_stop": selected_stop,
        "target": target,
        "risk_per_share": risk_per_share,
        "reward_per_share": reward_per_share,
        "risk_reward_ratio": risk_reward_ratio,
        "risk_reward_status": risk_reward_status,
    }


def calculate_position_size(
    portfolio_value,
    risk_percent,
    entry_price,
    stop_loss,
):
    """
    Calculates maximum position size using fixed percentage risk.

    Maximum loss = Portfolio Value × Risk Percentage.

    Quantity = Maximum Loss / Risk Per Share.
    """

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

    maximum_allowed_loss = (
        portfolio_value * risk_percent / 100
    )

    maximum_quantity = int(
        maximum_allowed_loss / risk_per_share
    )

    approximate_position_value = (
        maximum_quantity * entry_price
    )

    portfolio_allocation_percent = (
        approximate_position_value
        / portfolio_value
    ) * 100

    return {
        "portfolio_value": portfolio_value,
        "risk_percent": risk_percent,
        "maximum_allowed_loss": maximum_allowed_loss,
        "entry_price": entry_price,
        "stop_loss": stop_loss,
        "risk_per_share": risk_per_share,
        "maximum_quantity": maximum_quantity,
        "position_value": approximate_position_value,
        "portfolio_allocation_percent": portfolio_allocation_percent,
    }


# =============================================================================
# WINNER SCORE ENGINE
# =============================================================================

def score_relative_strength(relative_strength):
    """Scores relative strength out of 20."""

    score = 0
    positives = []
    risks = []

    rs_status = relative_strength.get(
        "status",
        "Insufficient data",
    )

    rs_trend = relative_strength.get(
        "rs_trend",
        "Unavailable",
    )

    rs_3m = safe_number(
        relative_strength.get(
            "relative_3m"
        )
    )

    rs_6m = safe_number(
        relative_strength.get(
            "relative_6m"
        )
    )

    if rs_status == "Leader":
        score += 10
        positives.append(
            "Relative Strength status is Leader"
        )

    elif rs_status == "Strong":
        score += 7
        positives.append(
            "Relative Strength status is Strong"
        )

    elif rs_status == "Neutral":
        score += 3

    elif rs_status == "Weak":
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

    if rs_3m is not None:
        if rs_3m >= 15:
            score += 3
            positives.append(
                f"Strong 3-month RS: {rs_3m:.1f}%"
            )

        elif rs_3m >= 5:
            score += 2
            positives.append(
                f"Positive 3-month RS: {rs_3m:.1f}%"
            )

        elif rs_3m < 0:
            risks.append(
                f"Negative 3-month RS: {rs_3m:.1f}%"
            )

    if rs_6m is not None:
        if rs_6m >= 20:
            score += 3
            positives.append(
                f"Strong 6-month RS: {rs_6m:.1f}%"
            )

        elif rs_6m >= 8:
            score += 2
            positives.append(
                f"Positive 6-month RS: {rs_6m:.1f}%"
            )

        elif rs_6m < 0:
            risks.append(
                f"Negative 6-month RS: {rs_6m:.1f}%"
            )

    return {
        "score": min(score, 20),
        "positives": positives,
        "risks": risks,
    }


def score_alignment(alignment):
    """Scores multi-timeframe alignment out of 20."""

    score = 0
    positives = []
    risks = []

    status = alignment.get(
        "status",
        "Insufficient data",
    )

    daily_trend = alignment.get(
        "daily",
        "Insufficient data",
    )

    weekly_trend = alignment.get(
        "weekly",
        "Insufficient data",
    )

    monthly_trend = alignment.get(
        "monthly",
        "Insufficient data",
    )

    if status == "Strong multi-timeframe alignment":
        score += 16
        positives.append(
            "Strong multi-timeframe alignment"
        )

    elif status == "Bullish multi-timeframe alignment":
        score += 12
        positives.append(
            "Bullish multi-timeframe alignment"
        )

    elif status == "Mixed timeframe alignment":
        score += 5
        risks.append(
            "Timeframes are not fully aligned"
        )

    elif status == "Bearish multi-timeframe alignment":
        risks.append(
            "Most chart timeframes are bearish"
        )

    if monthly_trend == "Strong bullish":
        score += 2
        positives.append(
            "Monthly trend is Strong Bullish"
        )

    elif monthly_trend == "Bearish":
        risks.append(
            "Monthly trend is Bearish"
        )

    if weekly_trend == "Strong bullish":
        score += 1
        positives.append(
            "Weekly trend is Strong Bullish"
        )

    elif weekly_trend == "Bearish":
        risks.append(
            "Weekly trend is Bearish"
        )

    if daily_trend == "Strong bullish":
        score += 1
        positives.append(
            "Daily trend is Strong Bullish"
        )

    elif daily_trend == "Bearish":
        risks.append(
            "Daily trend is Bearish"
        )

    return {
        "score": min(score, 20),
        "positives": positives,
        "risks": risks,
    }


def score_fundamental_quality(fundamentals):
    """Scores basic available fundamental quality out of 20."""

    score = 0
    positives = []
    risks = []

    roe = safe_number(
        fundamentals.get(
            "returnOnEquity"
        )
    )

    roa = safe_number(
        fundamentals.get(
            "returnOnAssets"
        )
    )

    profit_margin = safe_number(
        fundamentals.get(
            "profitMargins"
        )
    )

    operating_margin = safe_number(
        fundamentals.get(
            "operatingMargins"
        )
    )

    revenue_growth = safe_number(
        fundamentals.get(
            "revenueGrowth"
        )
    )

    earnings_growth = safe_number(
        fundamentals.get(
            "earningsGrowth"
        )
    )

    debt_equity = safe_number(
        fundamentals.get(
            "debtToEquity"
        )
    )

    free_cashflow = safe_number(
        fundamentals.get(
            "freeCashflow"
        )
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
            risks.append(
                f"Low profit margin: {profit_margin * 100:.1f}%"
            )

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
            risks.append(
                "Negative revenue growth"
            )

    if earnings_growth is not None:
        if earnings_growth >= 0.20:
            score += 2
            positives.append(
                f"Strong earnings growth: {earnings_growth * 100:.1f}%"
            )

        elif earnings_growth >= 0.10:
            score += 1

        elif earnings_growth < 0:
            risks.append(
                "Negative earnings growth"
            )

    if debt_equity is not None:
        if debt_equity < 50:
            score += 2
            positives.append(
                "Low debt/equity"
            )

        elif debt_equity > 200:
            risks.append(
                "High debt/equity"
            )

    if free_cashflow is not None:
        if free_cashflow > 0:
            score += 2
            positives.append(
                "Positive free cash flow"
            )

        else:
            risks.append(
                "Negative free cash flow"
            )

    return {
        "score": min(score, 20),
        "positives": positives,
        "risks": risks,
    }


def score_technical_trend(
    daily_data,
    weekly_data,
    monthly_data,
):
    """Scores individual timeframe trends out of 15."""

    score = 0
    positives = []
    risks = []

    daily_trend = calculate_overall_trend(
        daily_data
    )

    weekly_trend = calculate_overall_trend(
        weekly_data
    )

    monthly_trend = calculate_overall_trend(
        monthly_data
    )

    if monthly_trend == "Strong bullish":
        score += 6
        positives.append(
            "Strong Bullish Monthly trend"
        )

    elif monthly_trend == "Bullish":
        score += 4
        positives.append(
            "Bullish Monthly trend"
        )

    elif monthly_trend == "Bearish":
        risks.append(
            "Bearish Monthly trend"
        )

    if weekly_trend == "Strong bullish":
        score += 5
        positives.append(
            "Strong Bullish Weekly trend"
        )

    elif weekly_trend == "Bullish":
        score += 3
        positives.append(
            "Bullish Weekly trend"
        )

    elif weekly_trend == "Bearish":
        risks.append(
            "Bearish Weekly trend"
        )

    if daily_trend == "Strong bullish":
        score += 4
        positives.append(
            "Strong Bullish Daily trend"
        )

    elif daily_trend == "Bullish":
        score += 2
        positives.append(
            "Bullish Daily trend"
        )

    elif daily_trend == "Bearish":
        risks.append(
            "Bearish Daily trend"
        )

    return {
        "score": min(score, 15),
        "positives": positives,
        "risks": risks,
    }


def score_pattern_quality(
    daily_patterns,
    weekly_patterns,
    monthly_patterns,
):
    """Scores current pattern and breakout quality out of 15."""

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
            pattern_name = pattern.get("Pattern")

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
                    f"Confirmed Bullish {pattern_name} on {timeframe}"
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
                    f"Bullish {pattern_name} forming on {timeframe}"
                )

            elif (
                status == "Confirmed"
                and direction == "Bearish"
            ):
                risks.append(
                    f"Confirmed Bearish {pattern_name} on {timeframe}"
                )

    if not confirmed_bullish:
        risks.append(
            "No confirmed bullish breakout is currently detected"
        )

    return {
        "score": min(score, 15),
        "positives": positives,
        "risks": risks,
    }


def score_volume_confirmation(
    daily_patterns,
    weekly_patterns,
):
    """Scores bullish breakout volume confirmation out of 5."""

    volume_values = []

    for pattern in daily_patterns + weekly_patterns:
        if (
            pattern.get("Status") == "Confirmed"
            and pattern.get("Direction") == "Bullish"
        ):
            volume_change = safe_number(
                pattern.get("Volume %")
            )

            if volume_change is not None:
                volume_values.append(
                    volume_change
                )

    if not volume_values:
        return {
            "score": 0,
            "positives": [],
            "risks": [],
        }

    highest_volume = max(volume_values)

    if highest_volume >= 50:
        return {
            "score": 5,
            "positives": [
                (
                    "Strong breakout volume confirmation: "
                    f"{highest_volume:.1f}% above average"
                )
            ],
            "risks": [],
        }

    if highest_volume >= 25:
        return {
            "score": 4,
            "positives": [
                (
                    "Good breakout volume confirmation: "
                    f"{highest_volume:.1f}% above average"
                )
            ],
            "risks": [],
        }

    if highest_volume > 0:
        return {
            "score": 2,
            "positives": [
                (
                    "Positive breakout volume confirmation: "
                    f"{highest_volume:.1f}% above average"
                )
            ],
            "risks": [],
        }

    return {
        "score": 0,
        "positives": [],
        "risks": [
            "Bullish breakout volume is below average"
        ],
    }


def score_valuation_risk(fundamentals):
    """Scores valuation and leverage factors out of 5."""

    score = 0
    positives = []
    risks = []

    trailing_pe = safe_number(
        fundamentals.get("trailingPE")
    )

    price_to_book = safe_number(
        fundamentals.get("priceToBook")
    )

    debt_equity = safe_number(
        fundamentals.get("debtToEquity")
    )

    if trailing_pe is not None:
        if 0 < trailing_pe <= 20:
            score += 2
            positives.append(
                f"Reasonable P/E: {trailing_pe:.1f}"
            )

        elif trailing_pe > 80:
            risks.append(
                f"Very high P/E: {trailing_pe:.1f}"
            )

    if price_to_book is not None:
        if 0 < price_to_book <= 3:
            score += 1
            positives.append(
                f"Reasonable Price/Book: {price_to_book:.2f}"
            )

        elif price_to_book > 12:
            risks.append(
                f"High Price/Book: {price_to_book:.2f}"
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
    relative_strength,
    alignment,
    daily_data,
    weekly_data,
    monthly_data,
    daily_patterns,
    weekly_patterns,
    monthly_patterns,
):
    """Calculates a transparent Winner Score out of 100."""

    relative_strength_component = (
        score_relative_strength(
            relative_strength
        )
    )

    alignment_component = score_alignment(
        alignment
    )

    fundamental_component = score_fundamental_quality(
        fundamentals
    )

    technical_component = score_technical_trend(
        daily_data,
        weekly_data,
        monthly_data,
    )

    pattern_component = score_pattern_quality(
        daily_patterns,
        weekly_patterns,
        monthly_patterns,
    )

    volume_component = score_volume_confirmation(
        daily_patterns,
        weekly_patterns,
    )

    valuation_component = score_valuation_risk(
        fundamentals
    )

    final_score = (
        relative_strength_component["score"]
        + alignment_component["score"]
        + fundamental_component["score"]
        + technical_component["score"]
        + pattern_component["score"]
        + volume_component["score"]
        + valuation_component["score"]
    )

    final_score = min(
        round(final_score, 1),
        100,
    )

    if final_score >= 80:
        category = "Elite Candidate"

    elif final_score >= 65:
        category = "High-Conviction Watchlist"

    elif final_score >= 50:
        category = "Watchlist"

    elif final_score >= 35:
        category = "Neutral / Mixed"

    else:
        category = "Avoid / Weak"

    positives = (
        relative_strength_component["positives"]
        + alignment_component["positives"]
        + fundamental_component["positives"]
        + technical_component["positives"]
        + pattern_component["positives"]
        + volume_component["positives"]
        + valuation_component["positives"]
    )

    risks = (
        relative_strength_component["risks"]
        + alignment_component["risks"]
        + fundamental_component["risks"]
        + technical_component["risks"]
        + pattern_component["risks"]
        + volume_component["risks"]
        + valuation_component["risks"]
    )

    return {
        "score": final_score,
        "category": category,
        "fundamental_score": fundamental_component["score"],
        "relative_strength_score": relative_strength_component[
            "score"
        ],
        "alignment_score": alignment_component["score"],
        "technical_score": technical_component["score"],
        "pattern_score": pattern_component["score"],
        "volume_score": volume_component["score"],
        "valuation_score": valuation_component["score"],
        "positives": positives,
        "risks": risks,
    }


# =============================================================================
# CHART FUNCTIONS
# =============================================================================

def create_candlestick_chart(
    data,
    signals,
    title,
):
    """Creates candlestick chart with MA20, MA50, volume and signal levels."""

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

    for signal in signals:
        line_color = "#16a34a"

        if signal["Direction"] == "Bearish":
            line_color = "#dc2626"

        elif signal["Direction"] == "Neutral":
            line_color = "#f59e0b"

        figure.add_hline(
            y=signal["Level"],
            line_dash="dot",
            line_color=line_color,
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
        margin=dict(
            l=10,
            r=10,
            t=50,
            b=10,
        ),
    )

    return figure


def create_relative_strength_chart(
    rs_line,
    symbol,
):
    """Creates Relative Strength line chart vs Nifty 50."""

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
        title=(
            f"{symbol} Relative Strength vs Nifty 50"
        ),
        height=360,
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Stock Price / Nifty 50",
        margin=dict(
            l=10,
            r=10,
            t=50,
            b=10,
        ),
    )

    return figure


# =============================================================================
# LOAD STOCK UNIVERSE AND BENCHMARK
# =============================================================================

with st.spinner(
    "Loading Nifty Total Market constituents..."
):
    stock_universe, universe_error = (
        get_nifty_total_market_members()
    )

with st.spinner(
    "Loading Nifty 50 benchmark data..."
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
        Technical patterns • Relative strength • Winner Score •
        Multi-timeframe alignment • Risk/Reward research plan
    </div>
    """,
    unsafe_allow_html=True,
)

if universe_error:
    st.warning(universe_error)

if stock_universe.empty:
    st.error(
        "No stock universe is available. "
        "Refresh the cached data and try again."
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

with st.sidebar:
    st.header("🔍 Dashboard Controls")

    dashboard_mode = st.radio(
        "Mode",
        [
            "Stock research",
            "Winner ranking scanner",
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
        "Price data cache: 15 minutes\n\n"
        "Fundamental data cache: 12 hours\n\n"
        "Constituent data cache: 6 hours"
    )


# =============================================================================
# STOCK RESEARCH MODE
# =============================================================================

if dashboard_mode == "Stock research":
    available_symbols = sorted(
        stock_universe["Symbol"].tolist()
    )

    default_index = (
        available_symbols.index("RELIANCE")
        if "RELIANCE" in available_symbols
        else 0
    )

    selected_symbol = st.selectbox(
        "Search a Nifty Total Market stock",
        available_symbols,
        index=default_index,
    )

    selected_record = stock_universe[
        stock_universe["Symbol"] == selected_symbol
    ].iloc[0]

    ticker = selected_record["Ticker"]
    company_name = selected_record["Company Name"]
    industry = selected_record["Industry"]

    with st.spinner(
        f"Loading market data for {selected_symbol}..."
    ):
        stock_data = fetch_price_data(
            ticker,
            "5y",
        )

        fundamentals = fetch_fundamentals(
            ticker
        )

    if stock_data.empty:
        st.error(
            "No price data is available for this symbol."
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

    relative_strength = calculate_relative_strength(
        stock_data,
        nifty_50_data,
    )

    alignment = calculate_multitimeframe_alignment(
        daily_data,
        weekly_data,
        monthly_data,
    )

    winner_score = calculate_winner_score(
        fundamentals=fundamentals,
        relative_strength=relative_strength,
        alignment=alignment,
        daily_data=daily_data,
        weekly_data=weekly_data,
        monthly_data=monthly_data,
        daily_patterns=daily_patterns,
        weekly_patterns=weekly_patterns,
        monthly_patterns=monthly_patterns,
    )

    current_price = float(
        stock_data["close"].iloc[-1]
    )

    market_cap = fundamentals.get(
        "marketCap"
    )

    market_cap_crore = (
        safe_number(market_cap, 0)
        / 10000000
    )

    (
        metric_1,
        metric_2,
        metric_3,
        metric_4,
        metric_5,
    ) = st.columns(5)

    metric_1.metric(
        "Last Close",
        format_price(current_price),
    )

    metric_2.metric(
        "Market Cap",
        format_market_cap(market_cap),
    )

    metric_3.metric(
        "RS vs Nifty 50",
        relative_strength["status"],
    )

    metric_4.metric(
        "MTF Alignment",
        (
            f"{alignment['score']}/10"
            if alignment["score"] is not None
            else "Not available"
        ),
    )

    metric_5.metric(
        "Winner Score",
        f"{winner_score['score']}/100",
        winner_score["category"],
    )

    st.caption(
        f"{company_name} • {industry}"
    )

    if (
        market_cap_crore > 0
        and market_cap_crore < minimum_market_cap
    ):
        st.warning(
            f"{selected_symbol} is below the chosen minimum "
            f"market cap of ₹{minimum_market_cap:,.0f} Cr."
        )

    (
        winner_tab,
        technical_tab,
        relative_strength_tab,
        alignment_tab,
        risk_reward_tab,
        fundamentals_tab,
    ) = st.tabs(
        [
            "🏆 Winner Score",
            "Technical Research",
            "Relative Strength",
            "MTF Alignment",
            "Risk / Reward Plan",
            "Fundamentals",
        ]
    )

    # =========================================================================
    # WINNER SCORE TAB
    # =========================================================================

    with winner_tab:
        st.subheader(
            "Winner Score and Research Ranking"
        )

        score = winner_score["score"]

        if score >= 80:
            score_card_class = "green-card"
        elif score >= 65:
            score_card_class = "blue-card"
        elif score < 35:
            score_card_class = "red-card"
        else:
            score_card_class = "orange-card"

        st.markdown(
            f"""
            <div class="research-card {score_card_class}">
                <h2>Winner Score: {score}/100</h2>
                <h3>{winner_score["category"]}</h3>
                <p>
                    The score is a research-ranking system that combines
                    available fundamentals, Relative Strength, price trends,
                    chart alignment, pattern evidence, volume confirmation
                    and valuation-risk checks.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        score_dataframe = pd.DataFrame(
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
                "Score Component",
                "Points Earned",
                "Maximum Points",
            ],
        )

        score_dataframe["Strength %"] = (
            score_dataframe["Points Earned"]
            / score_dataframe["Maximum Points"]
        ) * 100

        st.markdown("### Score Breakdown")

        st.dataframe(
            score_dataframe,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Points Earned": st.column_config.NumberColumn(
                    "Points",
                    format="%.1f",
                ),
                "Maximum Points": st.column_config.NumberColumn(
                    "Maximum",
                    format="%.0f",
                ),
                "Strength %": st.column_config.ProgressColumn(
                    "Component Strength",
                    min_value=0,
                    max_value=100,
                    format="%.0f%%",
                ),
            },
        )

        evidence_col, risk_col = st.columns(2)

        with evidence_col:
            st.markdown("### Positive Evidence")

            if winner_score["positives"]:
                for evidence in winner_score["positives"][:20]:
                    st.success(f"✅ {evidence}")
            else:
                st.info(
                    "No positive evidence is currently available."
                )

        with risk_col:
            st.markdown("### Risk Flags")

            if winner_score["risks"]:
                for risk in winner_score["risks"][:20]:
                    st.warning(f"⚠️ {risk}")
            else:
                st.success(
                    "No major rule-based risk flags were detected."
                )

    # =========================================================================
    # TECHNICAL RESEARCH TAB
    # =========================================================================

    with technical_tab:
        daily_tab, weekly_tab, monthly_tab = st.tabs(
            [
                "Daily",
                "Weekly",
                "Monthly",
            ]
        )

        timeframe_configuration = [
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
        ]

        for (
            tab,
            timeframe_name,
            timeframe_data,
            patterns,
        ) in timeframe_configuration:
            with tab:
                support, resistance = (
                    calculate_support_resistance(
                        timeframe_data
                    )
                )

                trend_col, support_col, resistance_col = (
                    st.columns(3)
                )

                trend_col.metric(
                    "Overall Trend",
                    calculate_overall_trend(
                        timeframe_data
                    ),
                )

                support_col.metric(
                    "Nearest Support",
                    format_price(support),
                )

                resistance_col.metric(
                    "Nearest Resistance",
                    format_price(resistance),
                )

                st.subheader(
                    f"{timeframe_name} Pattern Status"
                )

                if patterns:
                    patterns_dataframe = pd.DataFrame(
                        patterns
                    )

                    st.dataframe(
                        patterns_dataframe,
                        hide_index=True,
                        use_container_width=True,
                        column_config={
                            "Date": st.column_config.DatetimeColumn(
                                "Signal Date",
                                format="YYYY-MM-DD",
                            ),
                            "Level": st.column_config.NumberColumn(
                                "Breakout / Neckline",
                                format="₹%.2f",
                            ),
                            "Current": st.column_config.NumberColumn(
                                "Current Price",
                                format="₹%.2f",
                            ),
                            "Return %": st.column_config.NumberColumn(
                                "Return Since Level",
                                format="%.2f%%",
                            ),
                            "Volume %": st.column_config.NumberColumn(
                                "Volume vs Average",
                                format="%.1f%%",
                            ),
                            "Pattern Height": st.column_config.NumberColumn(
                                "Pattern Height",
                                format="₹%.2f",
                            ),
                        },
                    )

                else:
                    st.info(
                        "No supported active technical pattern was detected."
                    )

                st.plotly_chart(
                    create_candlestick_chart(
                        timeframe_data,
                        patterns,
                        f"{selected_symbol} — {timeframe_name}",
                    ),
                    use_container_width=True,
                )

    # =========================================================================
    # RELATIVE STRENGTH TAB
    # =========================================================================

    with relative_strength_tab:
        st.subheader(
            "Relative Strength versus Nifty 50"
        )

        st.caption(
            "Relative Return = Stock Return − Nifty 50 Return. "
            "Positive Relative Return indicates that the stock "
            "outperformed Nifty 50 over the same period."
        )

        rs_col_1, rs_col_2, rs_col_3 = st.columns(3)

        rs_col_1.metric(
            "RS Status",
            relative_strength["status"],
        )

        rs_col_2.metric(
            "RS Line Trend",
            relative_strength["rs_trend"],
        )

        rs_col_3.metric(
            "Benchmark",
            "Nifty 50 (^NSEI)",
        )

        rs_dataframe = pd.DataFrame(
            [
                [
                    "1 Month",
                    relative_strength["stock_1m"],
                    relative_strength["benchmark_1m"],
                    relative_strength["relative_1m"],
                ],
                [
                    "3 Months",
                    relative_strength["stock_3m"],
                    relative_strength["benchmark_3m"],
                    relative_strength["relative_3m"],
                ],
                [
                    "6 Months",
                    relative_strength["stock_6m"],
                    relative_strength["benchmark_6m"],
                    relative_strength["relative_6m"],
                ],
                [
                    "12 Months",
                    relative_strength["stock_12m"],
                    relative_strength["benchmark_12m"],
                    relative_strength["relative_12m"],
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
            rs_dataframe,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Stock Return %": st.column_config.NumberColumn(
                    "Stock Return",
                    format="%.2f%%",
                ),
                "Nifty 50 Return %": st.column_config.NumberColumn(
                    "Nifty 50 Return",
                    format="%.2f%%",
                ),
                "Relative Return %": st.column_config.NumberColumn(
                    "Relative Return",
                    format="%.2f%%",
                ),
            },
        )

        if (
            relative_strength["rs_line"] is not None
            and not relative_strength["rs_line"].empty
        ):
            st.plotly_chart(
                create_relative_strength_chart(
                    relative_strength["rs_line"],
                    selected_symbol,
                ),
                use_container_width=True,
            )

        else:
            st.info(
                "There is insufficient aligned price history "
                "to draw the Relative Strength line."
            )

    # =========================================================================
    # MULTI-TIMEFRAME ALIGNMENT TAB
    # =========================================================================

    with alignment_tab:
        st.subheader(
            "Daily, Weekly and Monthly Alignment"
        )

        daily_col, weekly_col, monthly_col = st.columns(3)

        daily_col.metric(
            "Daily Trend",
            alignment["daily"],
        )

        weekly_col.metric(
            "Weekly Trend",
            alignment["weekly"],
        )

        monthly_col.metric(
            "Monthly Trend",
            alignment["monthly"],
        )

        alignment_col, structure_col = st.columns(2)

        alignment_col.metric(
            "Alignment Score",
            (
                f"{alignment['score']}/10"
                if alignment["score"] is not None
                else "Not available"
            ),
        )

        structure_col.metric(
            "Alignment Status",
            alignment["status"],
        )

        st.markdown(
            f"""
            <div class="research-card">
                <h4>Market Structure</h4>
                <p>{alignment["structure"]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            | Timeframe | Strong Bullish | Bullish | Neutral | Bearish |
            |---|---:|---:|---:|---:|
            | Daily | 3 | 2 | 1 | 0 |
            | Weekly | 4 | 3 | 1 | 0 |
            | Monthly | 3 | 2 | 1 | 0 |
            """
        )

    # =========================================================================
    # RISK / REWARD TAB
    # =========================================================================

    with risk_reward_tab:
        st.subheader(
            "Risk / Reward and Position Sizing Research"
        )

        st.caption(
            "These levels are rule-based educational calculations. "
            "They are not an instruction to buy, sell, enter, exit, "
            "or allocate capital to any security."
        )

        timeframe_for_plan = st.selectbox(
            "Select chart timeframe for trade-plan analysis",
            [
                "Daily",
                "Weekly",
                "Monthly",
            ],
        )

        if timeframe_for_plan == "Daily":
            plan_data = daily_data
            plan_patterns = daily_patterns

        elif timeframe_for_plan == "Weekly":
            plan_data = weekly_data
            plan_patterns = weekly_patterns

        else:
            plan_data = monthly_data
            plan_patterns = monthly_patterns

        bullish_patterns = [
            pattern
            for pattern in plan_patterns
            if pattern.get("Direction") == "Bullish"
        ]

        bearish_patterns = [
            pattern
            for pattern in plan_patterns
            if pattern.get("Direction") == "Bearish"
        ]

        available_patterns = bullish_patterns + bearish_patterns

        if available_patterns:
            pattern_labels = []

            for index, pattern in enumerate(
                available_patterns
            ):
                label = (
                    f"{index + 1}. "
                    f"{pattern['Pattern']} | "
                    f"{pattern['Status']} | "
                    f"{pattern['Direction']}"
                )

                pattern_labels.append(label)

            selected_pattern_label = st.selectbox(
                "Select a detected pattern",
                pattern_labels,
            )

            selected_pattern_index = pattern_labels.index(
                selected_pattern_label
            )

            selected_pattern = available_patterns[
                selected_pattern_index
            ]

            trade_plan = calculate_trade_plan(
                plan_data,
                selected_pattern,
            )

            if trade_plan is not None:
                entry_status = trade_plan[
                    "entry_quality"
                ]["status"]

                if entry_status == "Ideal Entry Zone":
                    entry_card_class = "green-card"

                elif entry_status == "Acceptable Entry Zone":
                    entry_card_class = "blue-card"

                elif entry_status in [
                    "Extended",
                    "Avoid Chasing",
                ]:
                    entry_card_class = "orange-card"

                else:
                    entry_card_class = "red-card"

                st.markdown(
                    f"""
                    <div class="research-card {entry_card_class}">
                        <h3>{selected_pattern["Pattern"]}</h3>
                        <p>
                            Status: {selected_pattern["Status"]} |
                            Direction: {selected_pattern["Direction"]}
                        </p>
                        <h4>Entry Quality: {entry_status}</h4>
                        <p>
                            {trade_plan["entry_quality"]["notes"]}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                (
                    plan_col_1,
                    plan_col_2,
                    plan_col_3,
                    plan_col_4,
                ) = st.columns(4)

                plan_col_1.metric(
                    "Current Price",
                    format_price(
                        trade_plan["current_price"]
                    ),
                )

                plan_col_2.metric(
                    "Breakout / Neckline",
                    format_price(
                        trade_plan["breakout_level"]
                    ),
                )

                plan_col_3.metric(
                    "14-Period ATR",
                    format_price(
                        trade_plan["atr"]
                    ),
                )

                plan_col_4.metric(
                    "Entry Distance",
                    format_percent(
                        trade_plan[
                            "entry_quality"
                        ]["distance_percent"]
                    ),
                )

                (
                    stop_col_1,
                    stop_col_2,
                    stop_col_3,
                    stop_col_4,
                ) = st.columns(4)

                stop_col_1.metric(
                    "Nearest Support",
                    format_price(
                        trade_plan["support"]
                    ),
                )

                stop_col_2.metric(
                    "Technical Stop",
                    format_price(
                        trade_plan["technical_stop"]
                    ),
                )

                stop_col_3.metric(
                    "ATR Stop",
                    format_price(
                        trade_plan["atr_stop"]
                    ),
                )

                stop_col_4.metric(
                    "Selected Stop",
                    format_price(
                        trade_plan["selected_stop"]
                    ),
                )

                (
                    reward_col_1,
                    reward_col_2,
                    reward_col_3,
                    reward_col_4,
                ) = st.columns(4)

                reward_col_1.metric(
                    "Pattern Target",
                    format_price(
                        trade_plan["target"]
                    ),
                )

                reward_col_2.metric(
                    "Risk per Share",
                    format_price(
                        trade_plan["risk_per_share"]
                    ),
                )

                reward_col_3.metric(
                    "Reward per Share",
                    format_price(
                        trade_plan["reward_per_share"]
                    ),
                )

                reward_ratio = trade_plan[
                    "risk_reward_ratio"
                ]

                reward_col_4.metric(
                    "Risk / Reward",
                    (
                        f"1 : {reward_ratio:.2f}"
                        if reward_ratio is not None
                        else "Not available"
                    ),
                    trade_plan[
                        "risk_reward_status"
                    ],
                )

                st.divider()

                st.subheader(
                    "Position Sizing Calculator"
                )

                input_col_1, input_col_2 = st.columns(2)

                with input_col_1:
                    portfolio_value = st.number_input(
                        "Portfolio value (₹)",
                        min_value=10000.0,
                        value=1000000.0,
                        step=50000.0,
                    )

                with input_col_2:
                    risk_per_trade = st.slider(
                        "Maximum risk per trade (%)",
                        min_value=0.25,
                        max_value=5.00,
                        value=1.00,
                        step=0.25,
                    )

                position_size = calculate_position_size(
                    portfolio_value=portfolio_value,
                    risk_percent=risk_per_trade,
                    entry_price=trade_plan[
                        "current_price"
                    ],
                    stop_loss=trade_plan[
                        "selected_stop"
                    ],
                )

                if position_size is not None:
                    (
                        size_col_1,
                        size_col_2,
                        size_col_3,
                        size_col_4,
                    ) = st.columns(4)

                    size_col_1.metric(
                        "Maximum Allowed Loss",
                        format_price(
                            position_size[
                                "maximum_allowed_loss"
                            ]
                        ),
                    )

                    size_col_2.metric(
                        "Risk per Share",
                        format_price(
                            position_size[
                                "risk_per_share"
                            ]
                        ),
                    )

                    size_col_3.metric(
                        "Maximum Quantity",
                        f"{position_size['maximum_quantity']:,} shares",
                    )

                    size_col_4.metric(
                        "Approx. Position Value",
                        format_price(
                            position_size[
                                "position_value"
                            ]
                        ),
                    )

                    st.caption(
                        "Approximate portfolio allocation: "
                        f"{position_size['portfolio_allocation_percent']:.2f}%"
                    )

                    if (
                        position_size[
                            "portfolio_allocation_percent"
                        ] > 25
                    ):
                        st.warning(
                            "The calculated position is more than 25% "
                            "of the specified portfolio. Consider a "
                            "separate maximum-position-size rule."
                        )

                else:
                    st.info(
                        "A position-size calculation requires a valid "
                        "current price and selected stop-loss."
                    )

        else:
            st.info(
                "No bullish or bearish detected pattern is currently "
                "available for the selected timeframe. "
                "A risk/reward plan requires a directional setup."
            )

    # =========================================================================
    # FUNDAMENTALS TAB
    # =========================================================================

    with fundamentals_tab:
        st.subheader(
            "Basic Fundamental Metrics"
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

        free_cashflow_text = (
            f"₹{free_cashflow / 10000000:,.0f} Cr"
            if free_cashflow is not None
            else "Not available"
        )

        fundamentals_dataframe = pd.DataFrame(
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
                    safe_number(
                        fundamentals.get(
                            "trailingPE"
                        )
                    ),
                ],
                [
                    "Forward P/E",
                    safe_number(
                        fundamentals.get(
                            "forwardPE"
                        )
                    ),
                ],
                [
                    "Price / Book",
                    safe_number(
                        fundamentals.get(
                            "priceToBook"
                        )
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
                    safe_number(
                        fundamentals.get(
                            "debtToEquity"
                        )
                    ),
                ],
                [
                    "Current Ratio",
                    safe_number(
                        fundamentals.get(
                            "currentRatio"
                        )
                    ),
                ],
                [
                    "Free Cash Flow",
                    free_cashflow_text,
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
            fundamentals_dataframe,
            hide_index=True,
            use_container_width=True,
        )

        st.info(
            "The current Winner Score uses available Yahoo Finance "
            "fundamental data. Detailed annual/quarterly evidence, "
            "sector-specific operational KPIs, ownership trends and "
            "promoter pledge analysis should be added in later versions."
        )


# =============================================================================
# WINNER RANKING SCANNER
# =============================================================================

else:
    st.subheader(
        "Winner Ranking Scanner"
    )

    st.caption(
        "Ranks Nifty Total Market stocks using Relative Strength, "
        "multi-timeframe alignment, technical structure, available "
        "fundamental quality, volume confirmation and valuation-risk checks."
    )

    scanner_col_1, scanner_col_2, scanner_col_3 = st.columns(3)

    with scanner_col_1:
        scanner_pattern = st.selectbox(
            "Pattern",
            PATTERN_OPTIONS,
        )

        scanner_timeframe = st.selectbox(
            "Pattern Timeframe",
            [
                "Any",
                "Daily",
                "Weekly",
                "Monthly",
            ],
        )

        scanner_pattern_status = st.selectbox(
            "Pattern Status",
            [
                "Any",
                "Confirmed",
                "In progress",
                "Candidate",
            ],
        )

    with scanner_col_2:
        scanner_trend = st.selectbox(
            "Overall Trend",
            TREND_OPTIONS,
        )

        scanner_rs_status = st.selectbox(
            "Relative Strength Status",
            RS_STATUS_OPTIONS,
        )

        scanner_alignment_status = st.selectbox(
            "MTF Alignment",
            MTF_ALIGNMENT_OPTIONS,
        )

    with scanner_col_3:
        scanner_min_winner_score = st.slider(
            "Minimum Winner Score",
            min_value=0,
            max_value=100,
            value=50,
            step=5,
        )

        scanner_winner_category = st.selectbox(
            "Winner Category",
            WINNER_CATEGORY_OPTIONS,
        )

        scan_limit = st.selectbox(
            "Maximum Stocks to Scan",
            [
                50,
                100,
                250,
                500,
                750,
            ],
            index=1,
        )

    st.warning(
        "Free Streamlit Cloud may take several minutes for a wide scan. "
        "Start with 50 or 100 stocks. Use 750 only after confirming "
        "that the app runs successfully."
    )

    if st.button(
        "🏆 Run Winner Ranking Scan",
        type="primary",
    ):
        scan_results = []

        if scanner_timeframe == "Any":
            timeframes_to_scan = [
                "Daily",
                "Weekly",
                "Monthly",
            ]
        else:
            timeframes_to_scan = [
                scanner_timeframe
            ]

        scan_universe = stock_universe.head(
            scan_limit
        ).copy()

        progress = st.progress(0)
        progress_text = st.empty()

        total_stocks = len(scan_universe)

        for row_number, (_, stock_record) in enumerate(
            scan_universe.iterrows(),
            start=1,
        ):
            symbol = stock_record["Symbol"]
            ticker = stock_record["Ticker"]

            progress_text.caption(
                f"Scanning {row_number:,} of {total_stocks:,}: "
                f"{symbol}"
            )

            try:
                stock_data = fetch_price_data(
                    ticker,
                    "5y",
                )

                if stock_data.empty:
                    continue

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
                    continue

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

                relative_strength = calculate_relative_strength(
                    stock_data,
                    nifty_50_data,
                )

                alignment = (
                    calculate_multitimeframe_alignment(
                        daily_data,
                        weekly_data,
                        monthly_data,
                    )
                )

                winner_score = calculate_winner_score(
                    fundamentals=fundamentals,
                    relative_strength=relative_strength,
                    alignment=alignment,
                    daily_data=daily_data,
                    weekly_data=weekly_data,
                    monthly_data=monthly_data,
                    daily_patterns=daily_patterns,
                    weekly_patterns=weekly_patterns,
                    monthly_patterns=monthly_patterns,
                )

                if (
                    winner_score["score"]
                    < scanner_min_winner_score
                ):
                    continue

                if (
                    scanner_winner_category != "Any"
                    and winner_score["category"]
                    != scanner_winner_category
                ):
                    continue

                if (
                    scanner_rs_status != "Any"
                    and relative_strength["status"]
                    != scanner_rs_status
                ):
                    continue

                if (
                    scanner_alignment_status != "Any"
                    and alignment["status"]
                    != scanner_alignment_status
                ):
                    continue

                timeframe_mapping = {
                    "Daily": (
                        daily_data,
                        daily_patterns,
                    ),
                    "Weekly": (
                        weekly_data,
                        weekly_patterns,
                    ),
                    "Monthly": (
                        monthly_data,
                        monthly_patterns,
                    ),
                }

                matches = []

                for timeframe in timeframes_to_scan:
                    timeframe_data, timeframe_patterns = (
                        timeframe_mapping[timeframe]
                    )

                    timeframe_trend = calculate_overall_trend(
                        timeframe_data
                    )

                    if (
                        scanner_trend != "Any"
                        and timeframe_trend != scanner_trend
                    ):
                        continue

                    for signal in timeframe_patterns:
                        pattern_match = (
                            scanner_pattern == "Any"
                            or signal["Pattern"]
                            == scanner_pattern
                        )

                        status_match = (
                            scanner_pattern_status == "Any"
                            or signal["Status"]
                            == scanner_pattern_status
                        )

                        if pattern_match and status_match:
                            matches.append(
                                {
                                    "Timeframe": timeframe,
                                    "Overall Trend": timeframe_trend,
                                    **signal,
                                }
                            )

                if (
                    scanner_pattern != "Any"
                    and not matches
                ):
                    continue

                if not matches:
                    matches = [
                        {
                            "Timeframe": "Overall",
                            "Overall Trend": calculate_overall_trend(
                                daily_data
                            ),
                            "Pattern": "No active pattern",
                            "Status": "No Pattern",
                            "Direction": "Neutral",
                            "Date": stock_data.index[-1],
                            "Level": np.nan,
                            "Current": float(
                                stock_data["close"].iloc[-1]
                            ),
                            "Return %": np.nan,
                            "Volume %": np.nan,
                            "Pattern Height": np.nan,
                            "Notes": (
                                "Ranked by Winner Score. "
                                "No supported active pattern found."
                            ),
                        }
                    ]

                for match in matches:
                    scan_results.append(
                        {
                            "Stock": symbol,
                            "Company": stock_record[
                                "Company Name"
                            ],
                            "Industry": stock_record[
                                "Industry"
                            ],
                            "Market Cap (Cr)": round(
                                market_cap_crore,
                                0,
                            ),
                            "Winner Score": winner_score[
                                "score"
                            ],
                            "Winner Category": winner_score[
                                "category"
                            ],
                            "Fundamental Score": winner_score[
                                "fundamental_score"
                            ],
                            "RS Score": winner_score[
                                "relative_strength_score"
                            ],
                            "RS Status": relative_strength[
                                "status"
                            ],
                            "RS Trend": relative_strength[
                                "rs_trend"
                            ],
                            "RS 3M %": relative_strength[
                                "relative_3m"
                            ],
                            "RS 6M %": relative_strength[
                                "relative_6m"
                            ],
                            "MTF Score": alignment[
                                "score"
                            ],
                            "MTF Alignment": alignment[
                                "status"
                            ],
                            **match,
                        }
                    )

            except Exception:
                pass

            progress.progress(
                row_number / total_stocks
            )

        progress.empty()
        progress_text.empty()

        st.subheader(
            f"Winner Ranking Results: {len(scan_results)}"
        )

        if scan_results:
            results_dataframe = pd.DataFrame(
                scan_results
            )

            results_dataframe = results_dataframe.sort_values(
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

            result_metric_1, result_metric_2, result_metric_3, result_metric_4 = (
                st.columns(4)
            )

            result_metric_1.metric(
                "Ranked Results",
                len(results_dataframe),
            )

            result_metric_2.metric(
                "Elite Candidates",
                int(
                    (
                        results_dataframe[
                            "Winner Category"
                        ]
                        == "Elite Candidate"
                    ).sum()
                ),
            )

            result_metric_3.metric(
                "High-Conviction",
                int(
                    (
                        results_dataframe[
                            "Winner Category"
                        ]
                        == "High-Conviction Watchlist"
                    ).sum()
                ),
            )

            result_metric_4.metric(
                "RS Leaders",
                int(
                    (
                        results_dataframe[
                            "RS Status"
                        ]
                        == "Leader"
                    ).sum()
                ),
            )

            st.dataframe(
                results_dataframe,
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
                    "Date": st.column_config.DatetimeColumn(
                        "Signal Date",
                        format="YYYY-MM-DD",
                    ),
                    "Level": st.column_config.NumberColumn(
                        "Breakout / Neckline",
                        format="₹%.2f",
                    ),
                    "Current": st.column_config.NumberColumn(
                        "Current Price",
                        format="₹%.2f",
                    ),
                    "Return %": st.column_config.NumberColumn(
                        "Return Since Level",
                        format="%.2f%%",
                    ),
                    "Volume %": st.column_config.NumberColumn(
                        "Volume vs Average",
                        format="%.1f%%",
                    ),
                    "Market Cap (Cr)": st.column_config.NumberColumn(
                        "Market Cap",
                        format="₹%d Cr",
                    ),
                },
            )

            csv_data = results_dataframe.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Download Winner Ranking Results",
                data=csv_data,
                file_name="nifty_total_market_winner_ranking.csv",
                mime="text/csv",
            )

        else:
            st.info(
                "No stocks matched your selected Winner Score, "
                "pattern, trend, Relative Strength, alignment, "
                "market-cap, or category filters."
            )


# =============================================================================
# FOOTER
# =============================================================================

st.divider()

st.caption(
    "Data sources: Nifty Indices constituent data and Yahoo Finance. "
    "Risk/reward levels, ATR stops, pattern targets, position sizing, "
    "technical patterns, Winner Scores and rankings are rule-based "
    "research calculations. They are not investment advice, buy/sell "
    "recommendations, or guaranteed outcomes. Verify all data independently."
)
