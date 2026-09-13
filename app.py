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
# VISUAL STYLE
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

TARGET_OPTIONS = [
    5,
    10,
    15,
]

STOP_OPTIONS = [
    5,
    8,
    10,
    12,
]

HORIZON_OPTIONS = [
    40,
    60,
    90,
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
    """Format a price as INR."""

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


def format_plain_percent(value):
    """Format an unsigned percentage."""

    value = safe_number(value)

    if value is None:
        return "Not available"

    return f"{value:.2f}%"


def format_market_cap(value):
    """Format market capitalization in crore rupees."""

    value = safe_number(value)

    if value is None or value <= 0:
        return "Not available"

    return f"₹{value / 10000000:,.0f} Cr"


def calculate_percentage_change(current_value, previous_value):
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


# =============================================================================
# NIFTY TOTAL MARKET UNIVERSE
# =============================================================================

@st.cache_data(ttl=21600, show_spinner=False)
def get_nifty_total_market_members():
    """
    Load current Nifty Total Market constituents.

    Falls back to a smaller list when the Nifty Indices file cannot
    be downloaded from Streamlit Cloud.
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
                "Too few constituents received."
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
            "Could not load the live Nifty Total Market constituent list. "
            f"Using fallback stocks. Reason: {type(error).__name__}: {error}"
        )

        return fallback, warning


# =============================================================================
# PRICE AND FUNDAMENTAL DATA
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

        data.index = pd.to_datetime(data.index)

        if getattr(data.index, "tz", None) is not None:
            data.index = data.index.tz_localize(None)

        return data

    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=43200, show_spinner=False)
def fetch_fundamentals(ticker):
    """Fetch available summary fundamental data."""

    requested_fields = [
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
            for field in requested_fields
        }

    except Exception:
        return {}


# =============================================================================
# PRICE RESAMPLING
# =============================================================================

def resample_ohlcv(data, timeframe):
    """Resample daily OHLCV data to Weekly or Monthly candles."""

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
    Determine trend using 20, 50, and 200-period moving averages.

    Strong bullish:
    Close > MA20 > MA50 > MA200.

    Bullish:
    Close > MA20 > MA50.

    Bearish:
    Close < MA20 < MA50.
    """

    if data is None or len(data) < 55:
        return "Insufficient data"

    latest_close = float(data["close"].iloc[-1])

    moving_average_20 = float(
        data["close"]
        .rolling(20)
        .mean()
        .iloc[-1]
    )

    moving_average_50 = float(
        data["close"]
        .rolling(50)
        .mean()
        .iloc[-1]
    )

    if len(data) >= 200:
        moving_average_200 = float(
            data["close"]
            .rolling(200)
            .mean()
            .iloc[-1]
        )

        if (
            latest_close > moving_average_20
            > moving_average_50
            > moving_average_200
        ):
            return "Strong bullish"

    if (
        latest_close > moving_average_20
        and moving_average_20 > moving_average_50
    ):
        return "Bullish"

    if (
        latest_close < moving_average_20
        and moving_average_20 < moving_average_50
    ):
        return "Bearish"

    return "Neutral / consolidating"


def calculate_multitimeframe_alignment(
    daily_data,
    weekly_data,
    monthly_data,
):
    """Calculate Daily, Weekly and Monthly alignment score out of 10."""

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
            "structure": "Insufficient chart history.",
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
            "Chart timeframes are mixed."
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
    """Calculate historical return over a selected trading-day period."""

    if data is None or data.empty:
        return None

    if len(data) <= trading_days:
        return None

    current_close = float(data["close"].iloc[-1])

    old_close = float(
        data["close"].iloc[-trading_days - 1]
    )

    if old_close == 0:
        return None

    return (
        current_close / old_close - 1
    ) * 100


def align_stock_benchmark(stock_data, benchmark_data):
    """Align stock and benchmark close prices on common trading dates."""

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

    aligned_data = stock_close.join(
        benchmark_close,
        how="inner",
    )

    return aligned_data.dropna()


def calculate_relative_strength(
    stock_data,
    benchmark_data,
):
    """
    Calculate stock outperformance against Nifty 50.

    Relative Return = Stock Return - Nifty 50 Return.
    """

    aligned_data = align_stock_benchmark(
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
        latest_rs = float(rs_line.iloc[-1])
        old_rs = float(rs_line.iloc[-63])

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
# TECHNICAL PATTERN FUNCTIONS
# =============================================================================

def find_pivots(values, pivot_type="high", order=3):
    """Find local high/low pivot indices."""

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
    """Create a standard technical pattern result."""

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
    """Detect selected current rule-based technical patterns."""

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

    results = []

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

        difference = abs(
            high[first_top] - high[second_top]
        ) / max(high[first_top], 1)

        if (
            second_top - first_top >= 8
            and difference < 0.045
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

            results.append(
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
                    height,
                )
            )

    # DOUBLE BOTTOM
    if len(pivot_lows) >= 2:
        first_bottom = pivot_lows[-2]
        second_bottom = pivot_lows[-1]

        difference = abs(
            low[first_bottom] - low[second_bottom]
        ) / max(low[first_bottom], 1)

        if (
            second_bottom - first_bottom >= 8
            and difference < 0.045
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

            results.append(
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
                    height,
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
            height = (
                high[head] - neckline
            )

            status = (
                "Confirmed"
                if close[-1] < neckline * 0.99
                else "In progress"
            )

            results.append(
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
                    height,
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
            height = neckline - low[head]

            status = (
                "Confirmed"
                if close[-1] > neckline * 1.01
                else "In progress"
            )

            results.append(
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
                    height,
                )
            )

    # RECTANGLE AND TRIANGLE PATTERNS
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

                results.append(
                    build_pattern_signal(
                        pattern_name,
                        status,
                        data,
                        level,
                        direction,
                        (
                            "Confirmation requires a close outside "
                            "the current pattern range."
                        ),
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
        results.append(
            build_pattern_signal(
                "Reversal Bottom",
                "Candidate",
                data,
                latest_low,
                "Bullish",
                (
                    "Hammer-like candle near local low. "
                    "Wait for confirmation."
                ),
                None,
            )
        )

    if (
        upper_shadow / candle_range > 0.55
        and body / candle_range < 0.35
        and latest_close >= previous_high * 0.96
    ):
        results.append(
            build_pattern_signal(
                "Reversal Top",
                "Candidate",
                data,
                latest_high,
                "Bearish",
                (
                    "Shooting-star-like candle near local high. "
                    "Wait for confirmation."
                ),
                None,
            )
        )

    return results


# =============================================================================
# SUPPORT, RESISTANCE, ATR AND RISK/REWARD
# =============================================================================

def calculate_support_resistance(data):
    """Calculate nearest simple pivot-based support and resistance."""

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
    """Calculate Average True Range."""

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

    atr = true_range.rolling(period).mean()

    latest_atr = atr.iloc[-1]

    if pd.isna(latest_atr):
        return None

    return float(latest_atr)


def calculate_pattern_target(pattern_signal):
    """Calculate simple measured-move target from pattern height."""

    breakout_level = safe_number(
        pattern_signal.get("Level")
    )

    pattern_height = safe_number(
        pattern_signal.get("Pattern Height")
    )

    direction = pattern_signal.get(
        "Direction"
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


def classify_entry_quality(
    current_price,
    breakout_level,
    direction,
):
    """Classify distance from pattern breakout/breakdown level."""

    current_price = safe_number(current_price)
    breakout_level = safe_number(breakout_level)

    if current_price is None or breakout_level is None:
        return {
            "status": "Insufficient data",
            "distance_percent": None,
            "notes": "Price or breakout level unavailable.",
        }

    if breakout_level == 0:
        return {
            "status": "Insufficient data",
            "distance_percent": None,
            "notes": "Invalid breakout level.",
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

        notes = (
            "Price is at or below the breakout level."
        )

    elif distance_percent <= 3:
        status = "Ideal Entry Zone"

        notes = (
            "Price is within 3% of breakout level."
        )

    elif distance_percent <= 7:
        status = "Acceptable Entry Zone"

        notes = (
            "Price is moderately above breakout level."
        )

    elif distance_percent <= 12:
        status = "Extended"

        notes = (
            "Price is materially extended from breakout."
        )

    else:
        status = "Avoid Chasing"

        notes = (
            "Price is highly extended from breakout."
        )

    return {
        "status": status,
        "distance_percent": distance_percent,
        "notes": notes,
    }


def calculate_trade_plan(data, pattern_signal):
    """Build rule-based risk/reward research levels."""

    if data is None or data.empty:
        return None

    current_price = float(
        data["close"].iloc[-1]
    )

    support, resistance = calculate_support_resistance(
        data
    )

    atr = calculate_atr(data, 14)

    direction = pattern_signal.get(
        "Direction",
        "Neutral",
    )

    breakout_level = safe_number(
        pattern_signal.get("Level")
    )

    if breakout_level is None:
        breakout_level = current_price

    target = calculate_pattern_target(
        pattern_signal
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
                        current_price + 3 * atr
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
                        current_price - 3 * atr
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
            "risk_reward_status": "No directional setup",
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


# =============================================================================
# WINNER SCORE ENGINE
# =============================================================================

def score_relative_strength(relative_strength):
    """Score RS out of 20."""

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
        relative_strength.get("relative_3m")
    )

    rs_6m = safe_number(
        relative_strength.get("relative_6m")
    )

    if rs_status == "Leader":
        score += 10
        positives.append("Relative Strength status is Leader")

    elif rs_status == "Strong":
        score += 7
        positives.append("Relative Strength status is Strong")

    elif rs_status == "Neutral":
        score += 3

    elif rs_status == "Weak":
        risks.append("Stock is underperforming Nifty 50")

    if rs_trend == "Rising":
        score += 4
        positives.append("Relative Strength line is rising")

    elif rs_trend == "Falling":
        risks.append("Relative Strength line is falling")

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
    """Score Daily/Weekly/Monthly alignment out of 20."""

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
        positives.append("Strong multi-timeframe alignment")

    elif status == "Bullish multi-timeframe alignment":
        score += 12
        positives.append("Bullish multi-timeframe alignment")

    elif status == "Mixed timeframe alignment":
        score += 5
        risks.append("Daily, Weekly and Monthly trends are mixed")

    elif status == "Bearish multi-timeframe alignment":
        risks.append("Most major chart timeframes are bearish")

    if monthly_trend == "Strong bullish":
        score += 2
        positives.append("Monthly trend is Strong Bullish")

    elif monthly_trend == "Bearish":
        risks.append("Monthly trend is Bearish")

    if weekly_trend == "Strong bullish":
        score += 1
        positives.append("Weekly trend is Strong Bullish")

    elif weekly_trend == "Bearish":
        risks.append("Weekly trend is Bearish")

    if daily_trend == "Strong bullish":
        score += 1
        positives.append("Daily trend is Strong Bullish")

    elif daily_trend == "Bearish":
        risks.append("Daily trend is Bearish")

    return {
        "score": min(score, 20),
        "positives": positives,
        "risks": risks,
    }


def score_fundamentals(fundamentals):
    """Score available basic fundamental quality out of 20."""

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
    """Score technical trend quality out of 15."""

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
        positives.append("Strong Bullish Monthly trend")

    elif monthly_trend == "Bullish":
        score += 4
        positives.append("Bullish Monthly trend")

    elif monthly_trend == "Bearish":
        risks.append("Bearish Monthly trend")

    if weekly_trend == "Strong bullish":
        score += 5
        positives.append("Strong Bullish Weekly trend")

    elif weekly_trend == "Bullish":
        score += 3
        positives.append("Bullish Weekly trend")

    elif weekly_trend == "Bearish":
        risks.append("Bearish Weekly trend")

    if daily_trend == "Strong bullish":
        score += 4
        positives.append("Strong Bullish Daily trend")

    elif daily_trend == "Bullish":
        score += 2
        positives.append("Bullish Daily trend")

    elif daily_trend == "Bearish":
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
    """Score current technical pattern quality out of 15."""

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
            "No confirmed bullish breakout currently detected"
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
    """Score breakout volume confirmation out of 5."""

    bullish_volumes = []

    for pattern in daily_patterns + weekly_patterns:
        if (
            pattern.get("Status") == "Confirmed"
            and pattern.get("Direction") == "Bullish"
        ):
            volume = safe_number(
                pattern.get("Volume %")
            )

            if volume is not None:
                bullish_volumes.append(volume)

    if not bullish_volumes:
        return {
            "score": 0,
            "positives": [],
            "risks": [],
        }

    highest_volume = max(bullish_volumes)

    if highest_volume >= 50:
        return {
            "score": 5,
            "positives": [
                (
                    "Strong breakout volume: "
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
                    "Good breakout volume: "
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
                    "Positive breakout volume: "
                    f"{highest_volume:.1f}% above average"
                )
            ],
            "risks": [],
        }

    return {
        "score": 0,
        "positives": [],
        "risks": [
            "Breakout volume below average"
        ],
    }


def score_valuation_risk(fundamentals):
    """Score simple valuation/risk factors out of 5."""

    score = 0
    positives = []
    risks = []

    pe = safe_number(
        fundamentals.get("trailingPE")
    )

    price_to_book = safe_number(
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
    relative_strength,
    alignment,
    daily_data,
    weekly_data,
    monthly_data,
    daily_patterns,
    weekly_patterns,
    monthly_patterns,
):
    """Calculate transparent Winner Score out of 100."""

    rs_component = score_relative_strength(
        relative_strength
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

    total_score = (
        rs_component["score"]
        + alignment_component["score"]
        + fundamental_component["score"]
        + technical_component["score"]
        + pattern_component["score"]
        + volume_component["score"]
        + valuation_component["score"]
    )

    total_score = min(
        round(total_score, 1),
        100,
    )

    if total_score >= 80:
        category = "Elite Candidate"

    elif total_score >= 65:
        category = "High-Conviction Watchlist"

    elif total_score >= 50:
        category = "Watchlist"

    elif total_score >= 35:
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
        "score": total_score,
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
# PHASE 1: HISTORICAL SETUP EVENT STUDY
# =============================================================================

def historical_daily_trend(
    data,
    index,
):
    """
    Calculate Daily trend using only historical data available at index.

    This avoids using future candles for historical setup classification.
    """

    if index < 55:
        return "Insufficient data"

    close = data["close"]

    current_close = float(
        close.iloc[index]
    )

    ma20 = float(
        close.iloc[:index + 1]
        .rolling(20)
        .mean()
        .iloc[-1]
    )

    ma50 = float(
        close.iloc[:index + 1]
        .rolling(50)
        .mean()
        .iloc[-1]
    )

    if index >= 199:
        ma200 = float(
            close.iloc[:index + 1]
            .rolling(200)
            .mean()
            .iloc[-1]
        )

        if (
            current_close > ma20
            > ma50
            > ma200
        ):
            return "Strong bullish"

    if (
        current_close > ma20
        and ma20 > ma50
    ):
        return "Bullish"

    if (
        current_close < ma20
        and ma20 < ma50
    ):
        return "Bearish"

    return "Neutral / consolidating"


def historical_weekly_trend(
    daily_data,
    index,
):
    """
    Calculate weekly trend using only daily candles available up to index.
    """

    if index < 100:
        return "Insufficient data"

    historical_daily_data = daily_data.iloc[
        :index + 1
    ].copy()

    weekly_data = resample_ohlcv(
        historical_daily_data,
        "Weekly",
    )

    if len(weekly_data) < 55:
        return "Insufficient data"

    return calculate_overall_trend(
        weekly_data
    )


def calculate_historical_rs_3m(
    stock_data,
    benchmark_data,
    index,
):
    """
    Calculate 3-month relative return using only historical information
    available on the selected event date.
    """

    if index < 63:
        return None

    event_date = stock_data.index[index]

    benchmark_slice = benchmark_data[
        benchmark_data.index <= event_date
    ].copy()

    stock_slice = stock_data.iloc[
        :index + 1
    ].copy()

    aligned = align_stock_benchmark(
        stock_slice,
        benchmark_slice,
    )

    if len(aligned) < 64:
        return None

    stock_now = float(
        aligned["stock_close"].iloc[-1]
    )

    stock_old = float(
        aligned["stock_close"].iloc[-64]
    )

    benchmark_now = float(
        aligned["benchmark_close"].iloc[-1]
    )

    benchmark_old = float(
        aligned["benchmark_close"].iloc[-64]
    )

    if stock_old == 0 or benchmark_old == 0:
        return None

    stock_return = (
        stock_now / stock_old - 1
    ) * 100

    benchmark_return = (
        benchmark_now / benchmark_old - 1
    ) * 100

    return stock_return - benchmark_return


def calculate_historical_52w_distance(
    data,
    index,
):
    """
    Calculate percentage distance from historical 52-week high.

    Uses only historical price history up to the event date.
    """

    if index < 20:
        return None

    lookback_start = max(
        0,
        index - 251,
    )

    historical_high = float(
        data["high"]
        .iloc[lookback_start:index + 1]
        .max()
    )

    current_close = float(
        data["close"].iloc[index]
    )

    if historical_high == 0:
        return None

    return (
        (current_close - historical_high)
        / historical_high
    ) * 100


def calculate_historical_volume_ratio(
    data,
    index,
):
    """
    Calculate current volume versus prior 20-day average at historical index.
    """

    if index < 21:
        return None

    current_volume = float(
        data["volume"].iloc[index]
    )

    average_volume = float(
        data["volume"]
        .iloc[index - 20:index]
        .mean()
    )

    if average_volume == 0:
        return None

    return (
        current_volume / average_volume - 1
    ) * 100


def is_historical_bullish_setup(
    stock_data,
    benchmark_data,
    index,
    require_strong_daily=False,
    require_weekly_bullish=True,
    require_positive_rs=True,
    max_52w_high_distance=-7.0,
    minimum_volume_change=None,
):
    """
    Identify whether a historical date meets a transparent bullish setup rule.

    The rule uses only price and volume data visible up to the event date.

    Default setup:
    - Daily trend is Bullish or Strong bullish.
    - Weekly trend is Bullish or Strong bullish.
    - 3M Relative Strength versus Nifty 50 is positive.
    - Stock is within 7% of its historical 52-week high.
    - Optional volume confirmation.
    """

    daily_trend = historical_daily_trend(
        stock_data,
        index,
    )

    weekly_trend = historical_weekly_trend(
        stock_data,
        index,
    )

    rs_3m = calculate_historical_rs_3m(
        stock_data,
        benchmark_data,
        index,
    )

    distance_52w = calculate_historical_52w_distance(
        stock_data,
        index,
    )

    volume_change = calculate_historical_volume_ratio(
        stock_data,
        index,
    )

    if require_strong_daily:
        daily_condition = (
            daily_trend == "Strong bullish"
        )
    else:
        daily_condition = (
            daily_trend in [
                "Strong bullish",
                "Bullish",
            ]
        )

    if require_weekly_bullish:
        weekly_condition = (
            weekly_trend in [
                "Strong bullish",
                "Bullish",
            ]
        )
    else:
        weekly_condition = True

    if require_positive_rs:
        rs_condition = (
            rs_3m is not None
            and rs_3m > 0
        )
    else:
        rs_condition = True

    distance_condition = (
        distance_52w is not None
        and distance_52w >= max_52w_high_distance
    )

    if minimum_volume_change is None:
        volume_condition = True
    else:
        volume_condition = (
            volume_change is not None
            and volume_change >= minimum_volume_change
        )

    qualifies = (
        daily_condition
        and weekly_condition
        and rs_condition
        and distance_condition
        and volume_condition
    )

    return {
        "qualifies": qualifies,
        "daily_trend": daily_trend,
        "weekly_trend": weekly_trend,
        "rs_3m": rs_3m,
        "distance_52w": distance_52w,
        "volume_change": volume_change,
    }


def evaluate_triple_barrier(
    stock_data,
    entry_index,
    target_percent,
    stop_percent,
    horizon_days,
):
    """
    Evaluate a bullish triple-barrier event.

    Upper barrier:
    Entry Price × (1 + target %).

    Lower barrier:
    Entry Price × (1 - stop %).

    Vertical barrier:
    Horizon in trading days.

    Intraday high determines whether target is reached.
    Intraday low determines whether stop is reached.

    If target and stop are both touched on the same day, the result is
    marked Ambiguous Same Day because daily OHLC data cannot confirm
    which level was touched first intraday.
    """

    entry_price = float(
        stock_data["close"].iloc[entry_index]
    )

    entry_date = stock_data.index[entry_index]

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

    future_data = stock_data.iloc[
        entry_index + 1:last_index + 1
    ].copy()

    maximum_high = float(
        future_data["high"].max()
    )

    minimum_low = float(
        future_data["low"].min()
    )

    mfe_percent = (
        maximum_high / entry_price - 1
    ) * 100

    mae_percent = (
        minimum_low / entry_price - 1
    ) * 100

    target_hit = False
    stop_hit = False
    outcome = "Time Expired"
    outcome_date = future_data.index[-1]
    days_to_outcome = len(future_data)

    for day_number, (date, row) in enumerate(
        future_data.iterrows(),
        start=1,
    ):
        day_high = float(row["high"])
        day_low = float(row["low"])

        target_touched = (
            day_high >= target_price
        )

        stop_touched = (
            day_low <= stop_price
        )

        if target_touched and stop_touched:
            outcome = "Ambiguous Same Day"
            outcome_date = date
            days_to_outcome = day_number
            target_hit = True
            stop_hit = True
            break

        if target_touched:
            outcome = "Target Hit First"
            outcome_date = date
            days_to_outcome = day_number
            target_hit = True
            break

        if stop_touched:
            outcome = "Stop Hit First"
            outcome_date = date
            days_to_outcome = day_number
            stop_hit = True
            break

    final_close = float(
        future_data["close"].iloc[-1]
    )

    forward_return = (
        final_close / entry_price - 1
    ) * 100

    return {
        "Entry Date": entry_date,
        "Entry Price": entry_price,
        "Target %": target_percent,
        "Target Price": target_price,
        "Stop %": stop_percent,
        "Stop Price": stop_price,
        "Horizon Days": horizon_days,
        "Outcome": outcome,
        "Outcome Date": outcome_date,
        "Days to Outcome": days_to_outcome,
        "Final Close": final_close,
        "Forward Return %": forward_return,
        "Maximum Favourable Excursion %": mfe_percent,
        "Maximum Adverse Excursion %": mae_percent,
        "Target Hit": target_hit,
        "Stop Hit": stop_hit,
    }


@st.cache_data(ttl=43200, show_spinner=False)
def run_historical_setup_study(
    stock_data,
    benchmark_data,
    target_percent,
    stop_percent,
    horizon_days,
    setup_spacing_days,
    require_strong_daily,
    require_weekly_bullish,
    require_positive_rs,
    max_52w_high_distance,
    minimum_volume_change,
):
    """
    Run Phase 1 historical setup event study.

    Event dates are separated by a minimum spacing period to avoid
    counting many consecutive days of the same trend as separate trades.
    """

    if stock_data is None or stock_data.empty:
        return pd.DataFrame()

    if benchmark_data is None or benchmark_data.empty:
        return pd.DataFrame()

    minimum_history = 260

    if len(stock_data) < minimum_history + horizon_days:
        return pd.DataFrame()

    events = []

    last_event_index = -setup_spacing_days

    final_valid_index = (
        len(stock_data) - horizon_days - 1
    )

    for index in range(
        minimum_history,
        final_valid_index,
    ):
        if (
            index - last_event_index
            < setup_spacing_days
        ):
            continue

        setup = is_historical_bullish_setup(
            stock_data=stock_data,
            benchmark_data=benchmark_data,
            index=index,
            require_strong_daily=require_strong_daily,
            require_weekly_bullish=require_weekly_bullish,
            require_positive_rs=require_positive_rs,
            max_52w_high_distance=max_52w_high_distance,
            minimum_volume_change=minimum_volume_change,
        )

        if not setup["qualifies"]:
            continue

        event = evaluate_triple_barrier(
            stock_data=stock_data,
            entry_index=index,
            target_percent=target_percent,
            stop_percent=stop_percent,
            horizon_days=horizon_days,
        )

        if event is None:
            continue

        event["Daily Trend"] = setup[
            "daily_trend"
        ]

        event["Weekly Trend"] = setup[
            "weekly_trend"
        ]

        event["RS 3M %"] = setup["rs_3m"]

        event["Distance from 52W High %"] = setup[
            "distance_52w"
        ]

        event["Volume vs 20D Avg %"] = setup[
            "volume_change"
        ]

        events.append(event)

        last_event_index = index

    if not events:
        return pd.DataFrame()

    event_dataframe = pd.DataFrame(events)

    event_dataframe = event_dataframe.sort_values(
        by="Entry Date",
        ascending=False,
    ).reset_index(drop=True)

    return event_dataframe


def summarise_historical_study(event_dataframe):
    """
    Create event-study summary statistics.
    """

    if event_dataframe is None or event_dataframe.empty:
        return {
            "total_events": 0,
            "target_hit_first": 0,
            "stop_hit_first": 0,
            "time_expired": 0,
            "ambiguous_same_day": 0,
            "target_hit_rate": None,
            "stop_hit_rate": None,
            "expiry_rate": None,
            "median_return": None,
            "average_return": None,
            "median_mfe": None,
            "median_mae": None,
            "worst_mae": None,
            "best_mfe": None,
            "median_days_to_target": None,
            "confidence": "Insufficient sample",
        }

    total_events = len(event_dataframe)

    target_hit_first = int(
        (
            event_dataframe["Outcome"]
            == "Target Hit First"
        ).sum()
    )

    stop_hit_first = int(
        (
            event_dataframe["Outcome"]
            == "Stop Hit First"
        ).sum()
    )

    time_expired = int(
        (
            event_dataframe["Outcome"]
            == "Time Expired"
        ).sum()
    )

    ambiguous_same_day = int(
        (
            event_dataframe["Outcome"]
            == "Ambiguous Same Day"
        ).sum()
    )

    clean_outcomes = event_dataframe[
        event_dataframe["Outcome"]
        != "Ambiguous Same Day"
    ].copy()

    clean_sample_count = len(clean_outcomes)

    if clean_sample_count > 0:
        target_hit_rate = (
            target_hit_first / clean_sample_count
        ) * 100

        stop_hit_rate = (
            stop_hit_first / clean_sample_count
        ) * 100

        expiry_rate = (
            time_expired / clean_sample_count
        ) * 100

    else:
        target_hit_rate = None
        stop_hit_rate = None
        expiry_rate = None

    median_return = float(
        event_dataframe[
            "Forward Return %"
        ].median()
    )

    average_return = float(
        event_dataframe[
            "Forward Return %"
        ].mean()
    )

    median_mfe = float(
        event_dataframe[
            "Maximum Favourable Excursion %"
        ].median()
    )

    median_mae = float(
        event_dataframe[
            "Maximum Adverse Excursion %"
        ].median()
    )

    worst_mae = float(
        event_dataframe[
            "Maximum Adverse Excursion %"
        ].min()
    )

    best_mfe = float(
        event_dataframe[
            "Maximum Favourable Excursion %"
        ].max()
    )

    target_events = event_dataframe[
        event_dataframe["Outcome"]
        == "Target Hit First"
    ]

    if not target_events.empty:
        median_days_to_target = float(
            target_events[
                "Days to Outcome"
            ].median()
        )
    else:
        median_days_to_target = None

    if total_events >= 50:
        confidence = "High"

    elif total_events >= 25:
        confidence = "Medium"

    elif total_events >= 10:
        confidence = "Low"

    else:
        confidence = "Insufficient sample"

    return {
        "total_events": total_events,
        "target_hit_first": target_hit_first,
        "stop_hit_first": stop_hit_first,
        "time_expired": time_expired,
        "ambiguous_same_day": ambiguous_same_day,
        "target_hit_rate": target_hit_rate,
        "stop_hit_rate": stop_hit_rate,
        "expiry_rate": expiry_rate,
        "median_return": median_return,
        "average_return": average_return,
        "median_mfe": median_mfe,
        "median_mae": median_mae,
        "worst_mae": worst_mae,
        "best_mfe": best_mfe,
        "median_days_to_target": median_days_to_target,
        "confidence": confidence,
    }


def create_historical_outcomes_chart(
    event_dataframe,
    target_percent,
    stop_percent,
):
    """Create a scatter chart for historical setup outcomes."""

    figure = go.Figure()

    if event_dataframe is None or event_dataframe.empty:
        figure.update_layout(
            title="No historical events available",
            height=350,
            template="plotly_white",
        )

        return figure

    color_mapping = {
        "Target Hit First": "#16a34a",
        "Stop Hit First": "#dc2626",
        "Time Expired": "#f59e0b",
        "Ambiguous Same Day": "#6b7280",
    }

    for outcome in event_dataframe["Outcome"].unique():
        subset = event_dataframe[
            event_dataframe["Outcome"] == outcome
        ]

        figure.add_trace(
            go.Scatter(
                x=subset["Entry Date"],
                y=subset["Forward Return %"],
                mode="markers",
                name=outcome,
                marker=dict(
                    size=10,
                    color=color_mapping.get(
                        outcome,
                        "#2563eb",
                    ),
                ),
                hovertemplate=(
                    "Entry Date: %{x}<br>"
                    "Forward Return: %{y:.2f}%<br>"
                    "<extra></extra>"
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
        title="Historical Setup Outcomes",
        height=400,
        template="plotly_white",
        xaxis_title="Historical Setup Entry Date",
        yaxis_title="Return at Outcome / Horizon %",
        legend_title="Outcome",
    )

    return figure


# =============================================================================
# CHART FUNCTIONS
# =============================================================================

def create_candlestick_chart(
    data,
    patterns,
    title,
):
    """Create a candlestick chart with MA20, MA50, volume and patterns."""

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
        line_color = "#16a34a"

        if pattern["Direction"] == "Bearish":
            line_color = "#dc2626"

        elif pattern["Direction"] == "Neutral":
            line_color = "#f59e0b"

        figure.add_hline(
            y=pattern["Level"],
            line_dash="dot",
            line_color=line_color,
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
    """Create relative strength line chart versus Nifty 50."""

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
        height=350,
        template="plotly_white",
        xaxis_title="Date",
        yaxis_title="Stock Price / Nifty 50",
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
# DASHBOARD HEADER
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
        Technical analysis • Relative Strength • Winner Score •
        Risk/Reward • Historical 5% / 10% / 15% Outcome Study
    </div>
    """,
    unsafe_allow_html=True,
)

if universe_error:
    st.warning(universe_error)

if stock_universe.empty:
    st.error(
        "No stock universe is available. "
        "Refresh cached data and try again."
    )
    st.stop()

if nifty_50_data.empty:
    st.warning(
        "Nifty 50 benchmark data could not be loaded. "
        "Relative Strength and Historical Setup Study may be unavailable."
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
        "Price cache: 15 minutes\n\n"
        "Fundamental cache: 12 hours\n\n"
        "Historical event studies cache: 12 hours\n\n"
        "Universe cache: 6 hours"
    )


# =============================================================================
# STOCK RESEARCH MODE
# =============================================================================

if dashboard_mode == "Stock research":
    symbols = sorted(
        stock_universe["Symbol"].tolist()
    )

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

    selected_record = stock_universe[
        stock_universe["Symbol"] == selected_symbol
    ].iloc[0]

    selected_ticker = selected_record["Ticker"]
    company_name = selected_record["Company Name"]
    industry = selected_record["Industry"]

    with st.spinner(
        f"Loading current data for {selected_symbol}..."
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
            "Price data is not available for this stock."
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

    current_price = float(
        stock_data["close"].iloc[-1]
    )

    market_cap = fundamentals.get("marketCap")

    metric_col1, metric_col2, metric_col3, metric_col4 = (
        st.columns(4)
    )

    metric_col1.metric(
        "Last Close",
        format_price(current_price),
    )

    metric_col2.metric(
        "Market Cap",
        format_market_cap(market_cap),
    )

    metric_col3.metric(
        "RS vs Nifty 50",
        relative_strength["status"],
    )

    metric_col4.metric(
        "MTF Alignment",
        (
            f"{alignment['score']}/10"
            if alignment["score"] is not None
            else "Not available"
        ),
    )

    st.caption(
        f"{company_name} • {industry}"
    )

    (
        technical_tab,
        rs_tab,
        alignment_tab,
        risk_reward_tab,
        historical_study_tab,
        fundamentals_tab,
    ) = st.tabs(
        [
            "Technical Research",
            "Relative Strength",
            "MTF Alignment",
            "Risk / Reward Plan",
            "🎯 Historical 10% Study",
            "Fundamentals",
        ]
    )

    # =========================================================================
    # TECHNICAL RESEARCH
    # =========================================================================

    with technical_tab:
        daily_tab, weekly_tab, monthly_tab = st.tabs(
            [
                "Daily",
                "Weekly",
                "Monthly",
            ]
        )

        timeframe_views = [
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
            timeframe_tab,
            timeframe_name,
            timeframe_data,
            patterns,
        ) in timeframe_views:
            with timeframe_tab:
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
                    st.dataframe(
                        pd.DataFrame(patterns),
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
                        },
                    )

                else:
                    st.info(
                        "No supported active or confirmed pattern "
                        "is currently detected."
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
    # RELATIVE STRENGTH
    # =========================================================================

    with rs_tab:
        st.subheader(
            "Relative Strength versus Nifty 50"
        )

        st.caption(
            "Relative Return = Stock Return − Nifty 50 Return. "
            "Positive values indicate historical outperformance."
        )

        rs_col1, rs_col2, rs_col3 = st.columns(3)

        rs_col1.metric(
            "RS Status",
            relative_strength["status"],
        )

        rs_col2.metric(
            "RS Line Trend",
            relative_strength["rs_trend"],
        )

        rs_col3.metric(
            "Benchmark",
            "Nifty 50 (^NSEI)",
        )

        relative_strength_table = pd.DataFrame(
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
            relative_strength_table,
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

    # =========================================================================
    # MULTI-TIMEFRAME ALIGNMENT
    # =========================================================================

    with alignment_tab:
        st.subheader(
            "Daily, Weekly and Monthly Trend Alignment"
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

        score_col, status_col = st.columns(2)

        score_col.metric(
            "Alignment Score",
            (
                f"{alignment['score']}/10"
                if alignment["score"] is not None
                else "Not available"
            ),
        )

        status_col.metric(
            "Alignment Status",
            alignment["status"],
        )

        st.info(
            alignment["structure"]
        )

    # =========================================================================
    # RISK / REWARD PLAN
    # =========================================================================

    with risk_reward_tab:
        st.subheader(
            "Risk / Reward Research Plan"
        )

        st.caption(
            "Rule-based research levels only. This is not a recommendation "
            "to buy, sell, enter, exit, or allocate capital."
        )

        selected_timeframe = st.selectbox(
            "Timeframe for risk/reward analysis",
            [
                "Daily",
                "Weekly",
                "Monthly",
            ],
        )

        if selected_timeframe == "Daily":
            selected_data = daily_data
            selected_patterns = daily_patterns

        elif selected_timeframe == "Weekly":
            selected_data = weekly_data
            selected_patterns = weekly_patterns

        else:
            selected_data = monthly_data
            selected_patterns = monthly_patterns

        directional_patterns = [
            pattern
            for pattern in selected_patterns
            if pattern.get("Direction")
            in [
                "Bullish",
                "Bearish",
            ]
        ]

        if directional_patterns:
            labels = []

            for index, pattern in enumerate(
                directional_patterns
            ):
                labels.append(
                    (
                        f"{index + 1}. "
                        f"{pattern['Pattern']} | "
                        f"{pattern['Status']} | "
                        f"{pattern['Direction']}"
                    )
                )

            selected_pattern_label = st.selectbox(
                "Detected pattern",
                labels,
            )

            selected_pattern_index = labels.index(
                selected_pattern_label
            )

            selected_pattern = directional_patterns[
                selected_pattern_index
            ]

            trade_plan = calculate_trade_plan(
                selected_data,
                selected_pattern,
            )

            if trade_plan is not None:
                plan_col1, plan_col2, plan_col3, plan_col4 = (
                    st.columns(4)
                )

                plan_col1.metric(
                    "Current Price",
                    format_price(
                        trade_plan["current_price"]
                    ),
                )

                plan_col2.metric(
                    "Breakout / Neckline",
                    format_price(
                        trade_plan["breakout_level"]
                    ),
                )

                plan_col3.metric(
                    "14-Period ATR",
                    format_price(
                        trade_plan["atr"]
                    ),
                )

                plan_col4.metric(
                    "Entry Quality",
                    trade_plan["entry_quality"]["status"],
                )

                stop_col1, stop_col2, stop_col3, stop_col4 = (
                    st.columns(4)
                )

                stop_col1.metric(
                    "Support",
                    format_price(
                        trade_plan["support"]
                    ),
                )

                stop_col2.metric(
                    "Technical Stop",
                    format_price(
                        trade_plan["technical_stop"]
                    ),
                )

                stop_col3.metric(
                    "ATR Stop",
                    format_price(
                        trade_plan["atr_stop"]
                    ),
                )

                stop_col4.metric(
                    "Selected Stop",
                    format_price(
                        trade_plan["selected_stop"]
                    ),
                )

                reward_col1, reward_col2, reward_col3, reward_col4 = (
                    st.columns(4)
                )

                reward_col1.metric(
                    "Pattern Target",
                    format_price(
                        trade_plan["target"]
                    ),
                )

                reward_col2.metric(
                    "Risk / Share",
                    format_price(
                        trade_plan["risk_per_share"]
                    ),
                )

                reward_col3.metric(
                    "Reward / Share",
                    format_price(
                        trade_plan["reward_per_share"]
                    ),
                )

                reward_ratio = trade_plan[
                    "risk_reward_ratio"
                ]

                reward_col4.metric(
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

        else:
            st.info(
                "No current bullish or bearish pattern is available "
                "for this timeframe."
            )

    # =========================================================================
    # PHASE 1: HISTORICAL SETUP EVENT STUDY
    # =========================================================================

    with historical_study_tab:
        st.subheader(
            "🎯 Historical 5% / 10% / 15% Setup Study"
        )

        st.caption(
            "This Phase 1 event study uses historical price, volume, trend "
            "and relative-strength conditions. It does not predict the future. "
            "It measures what happened after comparable historical setups."
        )

        st.markdown(
            """
            **Triple-barrier rule**

            - **Target barrier:** selected upside percentage.
            - **Stop barrier:** selected downside percentage.
            - **Time barrier:** selected maximum holding horizon.
            - **Outcome:** whichever barrier is touched first.
            """
        )

        study_filter_col1, study_filter_col2, study_filter_col3 = (
            st.columns(3)
        )

        with study_filter_col1:
            target_percent = st.selectbox(
                "Upside Target",
                TARGET_OPTIONS,
                index=1,
                format_func=lambda value: f"+{value}%",
            )

            stop_percent = st.selectbox(
                "Downside Stop",
                STOP_OPTIONS,
                index=1,
                format_func=lambda value: f"-{value}%",
            )

        with study_filter_col2:
            horizon_days = st.selectbox(
                "Maximum Holding Horizon",
                HORIZON_OPTIONS,
                index=1,
                format_func=lambda value: (
                    f"{value} Trading Days"
                ),
            )

            setup_spacing_days = st.selectbox(
                "Minimum Gap Between Historical Setups",
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

        with study_filter_col3:
            require_strong_daily = st.checkbox(
                "Require Strong Bullish Daily Trend",
                value=False,
            )

            require_weekly_bullish = st.checkbox(
                "Require Bullish Weekly Trend",
                value=True,
            )

            require_positive_rs = st.checkbox(
                "Require Positive 3M Relative Strength",
                value=True,
            )

            maximum_distance_52w = st.selectbox(
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
            )

        use_volume_filter = st.checkbox(
            "Require Volume Confirmation",
            value=False,
        )

        if use_volume_filter:
            minimum_volume_change = st.selectbox(
                "Minimum Volume Above 20-Day Average",
                [
                    0,
                    10,
                    25,
                    50,
                    75,
                ],
                index=1,
                format_func=lambda value: (
                    f"+{value}% Above Average"
                ),
            )
        else:
            minimum_volume_change = None

        if st.button(
            "Run Historical Setup Study",
            type="primary",
        ):
            with st.spinner(
                "Evaluating historical bullish setups and triple-barrier outcomes..."
            ):
                study_events = run_historical_setup_study(
                    stock_data=stock_data,
                    benchmark_data=nifty_50_data,
                    target_percent=target_percent,
                    stop_percent=stop_percent,
                    horizon_days=horizon_days,
                    setup_spacing_days=setup_spacing_days,
                    require_strong_daily=require_strong_daily,
                    require_weekly_bullish=require_weekly_bullish,
                    require_positive_rs=require_positive_rs,
                    max_52w_high_distance=maximum_distance_52w,
                    minimum_volume_change=minimum_volume_change,
                )

            st.session_state[
                "historical_study_events"
            ] = study_events

            st.session_state[
                "historical_study_settings"
            ] = {
                "symbol": selected_symbol,
                "target_percent": target_percent,
                "stop_percent": stop_percent,
                "horizon_days": horizon_days,
                "setup_spacing_days": setup_spacing_days,
                "require_strong_daily": require_strong_daily,
                "require_weekly_bullish": require_weekly_bullish,
                "require_positive_rs": require_positive_rs,
                "maximum_distance_52w": maximum_distance_52w,
                "minimum_volume_change": minimum_volume_change,
            }

        stored_events = st.session_state.get(
            "historical_study_events"
        )

        stored_settings = st.session_state.get(
            "historical_study_settings"
        )

        settings_match = (
            stored_settings is not None
            and stored_settings.get("symbol") == selected_symbol
            and stored_settings.get("target_percent") == target_percent
            and stored_settings.get("stop_percent") == stop_percent
            and stored_settings.get("horizon_days") == horizon_days
            and stored_settings.get("setup_spacing_days")
            == setup_spacing_days
            and stored_settings.get("require_strong_daily")
            == require_strong_daily
            and stored_settings.get("require_weekly_bullish")
            == require_weekly_bullish
            and stored_settings.get("require_positive_rs")
            == require_positive_rs
            and stored_settings.get("maximum_distance_52w")
            == maximum_distance_52w
            and stored_settings.get("minimum_volume_change")
            == minimum_volume_change
        )

        if stored_events is not None and settings_match:
            study_summary = summarise_historical_study(
                stored_events
            )

            st.divider()

            st.subheader(
                "Historical Setup Summary"
            )

            summary_col1, summary_col2, summary_col3, summary_col4 = (
                st.columns(4)
            )

            summary_col1.metric(
                "Historical Setups",
                study_summary["total_events"],
            )

            summary_col2.metric(
                f"Target +{target_percent}% Hit First",
                (
                    f"{study_summary['target_hit_rate']:.1f}%"
                    if study_summary[
                        "target_hit_rate"
                    ] is not None
                    else "Not available"
                ),
                (
                    f"{study_summary['target_hit_first']}"
                    f" / {study_summary['total_events']}"
                ),
            )

            summary_col3.metric(
                f"Stop -{stop_percent}% Hit First",
                (
                    f"{study_summary['stop_hit_rate']:.1f}%"
                    if study_summary[
                        "stop_hit_rate"
                    ] is not None
                    else "Not available"
                ),
                (
                    f"{study_summary['stop_hit_first']}"
                    f" / {study_summary['total_events']}"
                ),
            )

            summary_col4.metric(
                "Confidence",
                study_summary["confidence"],
            )

            outcome_col1, outcome_col2, outcome_col3, outcome_col4 = (
                st.columns(4)
            )

            outcome_col1.metric(
                "Median Forward Return",
                format_percent(
                    study_summary["median_return"]
                ),
            )

            outcome_col2.metric(
                "Average Forward Return",
                format_percent(
                    study_summary["average_return"]
                ),
            )

            outcome_col3.metric(
                "Median Max Gain",
                format_percent(
                    study_summary["median_mfe"]
                ),
            )

            outcome_col4.metric(
                "Median Max Drawdown",
                format_percent(
                    study_summary["median_mae"]
                ),
            )

            detail_col1, detail_col2, detail_col3, detail_col4 = (
                st.columns(4)
            )

            detail_col1.metric(
                "Time Expired",
                study_summary["time_expired"],
            )

            detail_col2.metric(
                "Ambiguous Same-Day Events",
                study_summary["ambiguous_same_day"],
            )

            detail_col3.metric(
                "Best Max Gain",
                format_percent(
                    study_summary["best_mfe"]
                ),
            )

            detail_col4.metric(
                "Worst Max Drawdown",
                format_percent(
                    study_summary["worst_mae"]
                ),
            )

            st.caption(
                "Ambiguous Same-Day Events occur where the day's High "
                "reached the target and the day's Low reached the stop. "
                "Daily OHLC data cannot determine which occurred first."
            )

            st.markdown(
                f"""
                <div class="research-card blue-card">
                    <h4>Study Configuration</h4>
                    <p>
                        Target: +{target_percent}% |
                        Stop: -{stop_percent}% |
                        Horizon: {horizon_days} trading days |
                        Minimum setup spacing: {setup_spacing_days} trading days
                    </p>
                    <p>
                        Daily trend: {"Strong Bullish only" if require_strong_daily else "Bullish or Strong Bullish"} |
                        Weekly trend required: {"Yes" if require_weekly_bullish else "No"} |
                        Positive RS required: {"Yes" if require_positive_rs else "No"} |
                        52W-high distance: Within {abs(maximum_distance_52w):.0f}% |
                        Volume confirmation: {"Yes" if minimum_volume_change is not None else "No"}
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

            if study_summary["total_events"] < 10:
                st.warning(
                    "This study has fewer than 10 historical setups. "
                    "Treat the result as exploratory rather than reliable."
                )

            elif study_summary["total_events"] < 25:
                st.info(
                    "This study has a low sample size. It can provide "
                    "context but should not be treated as a robust probability."
                )

            st.plotly_chart(
                create_historical_outcomes_chart(
                    stored_events,
                    target_percent,
                    stop_percent,
                ),
                use_container_width=True,
            )

            st.subheader(
                "Historical Setup Events"
            )

            display_columns = [
                "Entry Date",
                "Entry Price",
                "Daily Trend",
                "Weekly Trend",
                "RS 3M %",
                "Distance from 52W High %",
                "Volume vs 20D Avg %",
                "Target Price",
                "Stop Price",
                "Outcome",
                "Outcome Date",
                "Days to Outcome",
                "Final Close",
                "Forward Return %",
                "Maximum Favourable Excursion %",
                "Maximum Adverse Excursion %",
            ]

            display_events = stored_events[
                display_columns
            ].copy()

            st.dataframe(
                display_events,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Entry Date": st.column_config.DatetimeColumn(
                        "Entry Date",
                        format="YYYY-MM-DD",
                    ),
                    "Outcome Date": st.column_config.DatetimeColumn(
                        "Outcome Date",
                        format="YYYY-MM-DD",
                    ),
                    "Entry Price": st.column_config.NumberColumn(
                        "Entry Price",
                        format="₹%.2f",
                    ),
                    "Target Price": st.column_config.NumberColumn(
                        "Target Price",
                        format="₹%.2f",
                    ),
                    "Stop Price": st.column_config.NumberColumn(
                        "Stop Price",
                        format="₹%.2f",
                    ),
                    "Final Close": st.column_config.NumberColumn(
                        "Final Close",
                        format="₹%.2f",
                    ),
                    "RS 3M %": st.column_config.NumberColumn(
                        "RS 3M",
                        format="%.2f%%",
                    ),
                    "Distance from 52W High %": st.column_config.NumberColumn(
                        "Distance from 52W High",
                        format="%.2f%%",
                    ),
                    "Volume vs 20D Avg %": st.column_config.NumberColumn(
                        "Volume vs 20D Avg",
                        format="%.2f%%",
                    ),
                    "Forward Return %": st.column_config.NumberColumn(
                        "Forward Return",
                        format="%.2f%%",
                    ),
                    "Maximum Favourable Excursion %": st.column_config.NumberColumn(
                        "Max Gain",
                        format="%.2f%%",
                    ),
                    "Maximum Adverse Excursion %": st.column_config.NumberColumn(
                        "Max Drawdown",
                        format="%.2f%%",
                    ),
                },
            )

            historical_csv = stored_events.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Download Historical Study CSV",
                data=historical_csv,
                file_name=(
                    f"{selected_symbol.lower()}_"
                    f"historical_setup_study.csv"
                ),
                mime="text/csv",
            )

        else:
            st.info(
                "Configure the historical setup filters and click "
                "Run Historical Setup Study."
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

        free_cashflow_text = (
            f"₹{free_cashflow / 10000000:,.0f} Cr"
            if free_cashflow is not None
            else "Not available"
        )

        fundamentals_table = pd.DataFrame(
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
            fundamentals_table,
            hide_index=True,
            use_container_width=True,
        )

        st.info(
            "Annual and quarterly point-in-time financial data should be "
            "added before using fundamentals inside a historical outcome model. "
            "Using today's revised financial information on old setup dates "
            "would create look-ahead bias."
        )


# =============================================================================
# WINNER RANKING SCANNER
# =============================================================================

else:
    st.subheader(
        "Winner Ranking Scanner"
    )

    st.info(
        "The Historical Setup Study is available in Stock Research mode "
        "for one stock at a time. A cross-sectional Nifty 750 historical "
        "event database will be introduced in Phase 2."
    )

    st.caption(
        "Use the existing technical, trend and Relative Strength filters "
        "to rank candidate stocks."
    )

    scanner_col1, scanner_col2, scanner_col3 = st.columns(3)

    with scanner_col1:
        scanner_pattern = st.selectbox(
            "Pattern",
            PATTERN_OPTIONS,
        )

        scanner_timeframe = st.selectbox(
            "Timeframe",
            [
                "Any",
                "Daily",
                "Weekly",
                "Monthly",
            ],
        )

        scanner_status = st.selectbox(
            "Pattern Status",
            [
                "Any",
                "Confirmed",
                "In progress",
                "Candidate",
            ],
        )

    with scanner_col2:
        scanner_trend = st.selectbox(
            "Overall Trend",
            TREND_OPTIONS,
        )

        scanner_rs_status = st.selectbox(
            "Relative Strength",
            RS_STATUS_OPTIONS,
        )

        scanner_alignment = st.selectbox(
            "MTF Alignment",
            MTF_ALIGNMENT_OPTIONS,
        )

    with scanner_col3:
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

        st.caption(
            "A broad 750-stock scan can be slow on free hosting."
        )

    if st.button(
        "Run Technical Ranking Scan",
        type="primary",
    ):
        if scanner_timeframe == "Any":
            timeframe_list = [
                "Daily",
                "Weekly",
                "Monthly",
            ]
        else:
            timeframe_list = [
                scanner_timeframe
            ]

        scan_rows = []

        scan_data = stock_universe.head(
            scan_limit
        ).copy()

        progress = st.progress(0)
        progress_text = st.empty()

        total_stocks = len(scan_data)

        for row_number, (_, record) in enumerate(
            scan_data.iterrows(),
            start=1,
        ):
            symbol = record["Symbol"]
            ticker = record["Ticker"]

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

                if (
                    scanner_rs_status != "Any"
                    and relative_strength["status"]
                    != scanner_rs_status
                ):
                    continue

                if (
                    scanner_alignment != "Any"
                    and alignment["status"]
                    != scanner_alignment
                ):
                    continue

                data_mapping = {
                    "Daily": daily_data,
                    "Weekly": weekly_data,
                    "Monthly": monthly_data,
                }

                for timeframe in timeframe_list:
                    timeframe_data = data_mapping[
                        timeframe
                    ]

                    timeframe_trend = calculate_overall_trend(
                        timeframe_data
                    )

                    if (
                        scanner_trend != "Any"
                        and timeframe_trend != scanner_trend
                    ):
                        continue

                    patterns = detect_patterns(
                        timeframe_data
                    )

                    for pattern in patterns:
                        if (
                            scanner_pattern != "Any"
                            and pattern["Pattern"]
                            != scanner_pattern
                        ):
                            continue

                        if (
                            scanner_status != "Any"
                            and pattern["Status"]
                            != scanner_status
                        ):
                            continue

                        scan_rows.append(
                            {
                                "Stock": symbol,
                                "Company": record[
                                    "Company Name"
                                ],
                                "Industry": record[
                                    "Industry"
                                ],
                                "Market Cap (Cr)": round(
                                    market_cap_crore,
                                    0,
                                ),
                                "Timeframe": timeframe,
                                "Overall Trend": timeframe_trend,
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
                                **pattern,
                            }
                        )

            except Exception:
                pass

            progress.progress(
                row_number / total_stocks
            )

        progress.empty()
        progress_text.empty()

        if scan_rows:
            scan_dataframe = pd.DataFrame(
                scan_rows
            )

            scan_dataframe["Status Priority"] = (
                scan_dataframe["Status"]
                .map(
                    {
                        "Confirmed": 1,
                        "In progress": 2,
                        "Candidate": 3,
                    }
                )
                .fillna(99)
            )

            scan_dataframe = (
                scan_dataframe
                .sort_values(
                    by=[
                        "Status Priority",
                        "MTF Score",
                        "RS 6M %",
                    ],
                    ascending=[
                        True,
                        False,
                        False,
                    ],
                )
                .drop(
                    columns=[
                        "Status Priority"
                    ]
                )
            )

            st.subheader(
                f"Matching Signals: {len(scan_dataframe)}"
            )

            st.dataframe(
                scan_dataframe,
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
                    "RS 3M %": st.column_config.NumberColumn(
                        "RS 3M",
                        format="%.2f%%",
                    ),
                    "RS 6M %": st.column_config.NumberColumn(
                        "RS 6M",
                        format="%.2f%%",
                    ),
                    "MTF Score": st.column_config.NumberColumn(
                        "MTF Score",
                        format="%.0f/10",
                    ),
                    "Market Cap (Cr)": st.column_config.NumberColumn(
                        "Market Cap",
                        format="₹%d Cr",
                    ),
                },
            )

            output_csv = scan_dataframe.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "Download Scanner CSV",
                data=output_csv,
                file_name="nifty_technical_scanner.csv",
                mime="text/csv",
            )

        else:
            st.info(
                "No stocks matched your selected scanner filters."
            )


# =============================================================================
# FOOTER
# =============================================================================

st.divider()

st.caption(
    "Data sources: Nifty Indices constituent list and Yahoo Finance. "
    "The Historical Setup Study is a rule-based backtest using daily OHLCV "
    "data. It does not predict future returns or provide investment advice. "
    "Historical performance, target-hit rates, and risk/reward calculations "
    "do not guarantee future outcomes. Verify data and assess risk independently."
)
