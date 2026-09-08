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
            font-size: 2.35rem;
            font-weight: 800;
            color: #111827;
            margin-bottom: 0.15rem;
        }

        .sub-title {
            font-size: 1rem;
            color: #6b7280;
            margin-bottom: 1.2rem;
        }

        .research-card {
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 16px;
            background-color: #ffffff;
            margin-bottom: 12px;
        }

        .excellent-card {
            border-left: 6px solid #16a34a;
        }

        .watch-card {
            border-left: 6px solid #2563eb;
        }

        .neutral-card {
            border-left: 6px solid #f59e0b;
        }

        .weak-card {
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
    """Converts a value safely into float."""

    try:
        if value is None:
            return default

        if pd.isna(value):
            return default

        return float(value)

    except Exception:
        return default


def format_price(value):
    """Formats a price in Indian rupees."""

    value = safe_number(value)

    if value is None:
        return "Not available"

    return f"₹{value:,.2f}"


def format_percent(value):
    """Formats a percentage value."""

    value = safe_number(value)

    if value is None:
        return "Not available"

    return f"{value:+.2f}%"


def format_plain_percent(value):
    """Formats a percentage without a plus sign."""

    value = safe_number(value)

    if value is None:
        return "Not available"

    return f"{value:.2f}%"


def format_market_cap(value):
    """Formats market capitalization in crore rupees."""

    value = safe_number(value)

    if value is None or value <= 0:
        return "Not available"

    crore = value / 10000000

    return f"₹{crore:,.0f} Cr"


def calculate_percentage_change(current_value, previous_value):
    """Calculates percentage change safely."""

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


def calculate_cagr(start_value, end_value, years):
    """Calculates CAGR safely."""

    start_value = safe_number(start_value)
    end_value = safe_number(end_value)
    years = safe_number(years)

    if start_value is None:
        return None

    if end_value is None:
        return None

    if years is None:
        return None

    if start_value <= 0:
        return None

    if end_value <= 0:
        return None

    if years <= 0:
        return None

    return (
        (end_value / start_value)
        ** (1 / years) - 1
    ) * 100


def display_value(value, decimal_places=2):
    """Displays missing values as Not available instead of zero."""

    value = safe_number(value)

    if value is None:
        return "Not available"

    return f"{value:,.{decimal_places}f}"


# =============================================================================
# NIFTY TOTAL MARKET UNIVERSE
# =============================================================================

@st.cache_data(ttl=21600, show_spinner=False)
def get_nifty_total_market_members():
    """
    Loads current Nifty Total Market constituent data.

    Uses a fallback stock list if the official constituent file cannot
    be reached from Streamlit Cloud.
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
                "Insufficient constituents loaded."
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
            "Live Nifty Total Market constituent data is unavailable. "
            f"Using fallback stocks. Reason: {type(error).__name__}: {error}"
        )

        return fallback, warning


# =============================================================================
# PRICE AND FUNDAMENTAL DATA
# =============================================================================

@st.cache_data(ttl=900, show_spinner=False)
def fetch_price_data(ticker, period="5y"):
    """Fetches stock or benchmark OHLCV data from Yahoo Finance."""

    try:
        ticker_object = yf.Ticker(ticker)

        data = ticker_object.history(
            period=period,
            auto_adjust=True,
        )

        if data is None or data.empty:
            return pd.DataFrame()

        data = data.rename(columns=str.lower)

        expected_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        data = data[
            expected_columns
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
    """Fetches basic summary fundamentals from Yahoo Finance."""

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
    """Resamples daily OHLCV data to weekly or monthly candles."""

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
# TREND AND MULTI-TIMEFRAME ALIGNMENT
# =============================================================================

def calculate_overall_trend(data):
    """
    Calculates moving-average trend.

    Strong bullish: Price > 20 MA > 50 MA > 200 MA.
    Bullish: Price > 20 MA > 50 MA.
    Bearish: Price < 20 MA < 50 MA.
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
    Scores trend alignment across Daily, Weekly and Monthly charts.

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

    score = 0

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

    score += daily_points.get(
        daily_trend,
        0,
    )

    score += weekly_points.get(
        weekly_trend,
        0,
    )

    score += monthly_points.get(
        monthly_trend,
        0,
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
            "Daily, Weekly and Monthly timeframes are not aligned."
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
    """Calculates return over selected number of trading days."""

    if data is None or data.empty:
        return None

    if len(data) <= trading_days:
        return None

    current_close = float(
        data["close"].iloc[-1]
    )

    old_close = float(
        data["close"].iloc[-trading_days - 1]
    )

    if old_close == 0:
        return None

    return (
        current_close / old_close - 1
    ) * 100


def align_stock_benchmark(
    stock_data,
    benchmark_data,
):
    """Aligns stock and Nifty 50 data using common trading dates."""

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


def calculate_relative_strength(
    stock_data,
    benchmark_data,
):
    """
    Calculates stock outperformance versus Nifty 50.

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

    stock_close = pd.DataFrame(
        {
            "close": aligned_data["stock_close"],
        }
    )

    benchmark_close = pd.DataFrame(
        {
            "close": aligned_data["benchmark_close"],
        }
    )

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

    rs_line = (
        aligned_data["stock_close"]
        / aligned_data["benchmark_close"]
    )

    rs_trend = "Unavailable"

    if len(rs_line) >= 63:
        latest_rs = float(
            rs_line.iloc[-1]
        )

        past_rs = float(
            rs_line.iloc[-63]
        )

        if latest_rs > past_rs * 1.03:
            rs_trend = "Rising"

        elif latest_rs < past_rs * 0.97:
            rs_trend = "Falling"

        else:
            rs_trend = "Flat"

    relative_values = [
        value
        for value in [
            relative_1m,
            relative_3m,
            relative_6m,
            relative_12m,
        ]
        if value is not None
    ]

    if len(relative_values) < 2:
        rs_status = "Insufficient data"

    else:
        positive_count = sum(
            value > 0
            for value in relative_values
        )

        strong_count = sum(
            value > 5
            for value in relative_values
        )

        negative_count = sum(
            value < 0
            for value in relative_values
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
    """Finds local price highs or lows."""

    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) < (2 * order) + 1:
        return []

    indexes = []

    for index in range(
        order,
        len(values) - order,
    ):
        window = values[
            index - order:index + order + 1
        ]

        if pivot_type == "high":
            if values[index] >= np.max(window):
                indexes.append(index)

        else:
            if values[index] <= np.min(window):
                indexes.append(index)

    return indexes


def build_pattern_signal(
    pattern,
    status,
    data,
    level,
    direction,
    notes,
):
    """Creates a standard pattern result record."""

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
        "Notes": notes,
    }


def detect_patterns(data):
    """
    Detects rule-based technical patterns and reversal candidates.
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
                        "a close below the neckline."
                    ),
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
                        "a close above the neckline."
                    ),
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

            detected.append(
                build_pattern_signal(
                    "Head & Shoulders",
                    status,
                    data,
                    neckline,
                    "Bearish",
                    (
                        "Confirmation requires a close below neckline support."
                    ),
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

            detected.append(
                build_pattern_signal(
                    "Inverse Head & Shoulders",
                    status,
                    data,
                    neckline,
                    "Bullish",
                    (
                        "Confirmation requires a close above neckline resistance."
                    ),
                )
            )

    # RECTANGLES AND TRIANGLES
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

        range_percent = (
            resistance - support
        ) / max(resistance, 1)

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
                            "Breakout confirmation requires a close outside "
                            "the defined pattern range."
                        ),
                    )
                )

    # REVERSAL CANDLES
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
        detected.append(
            build_pattern_signal(
                "Reversal Bottom",
                "Candidate",
                data,
                latest_low,
                "Bullish",
                (
                    "Hammer-like candle near a local low. "
                    "Wait for next-candle confirmation."
                ),
            )
        )

    if (
        upper_shadow / candle_range > 0.55
        and body / candle_range < 0.35
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
                    "Shooting-star-like candle near a local high. "
                    "Wait for next-candle confirmation."
                ),
            )
        )

    return detected


def calculate_support_resistance(data):
    """Calculates nearby support and resistance using recent pivots."""

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


# =============================================================================
# WINNER SCORE ENGINE
# =============================================================================

def score_relative_strength(relative_strength):
    """Scores relative strength out of 20."""

    score = 0
    positives = []
    risks = []

    status = relative_strength.get(
        "status",
        "Insufficient data",
    )

    rs_trend = relative_strength.get(
        "rs_trend",
        "Unavailable",
    )

    relative_3m = safe_number(
        relative_strength.get("relative_3m")
    )

    relative_6m = safe_number(
        relative_strength.get("relative_6m")
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
                f"Strong 3-month outperformance: {relative_3m:.1f}%"
            )

        elif relative_3m >= 5:
            score += 2
            positives.append(
                f"Positive 3-month outperformance: {relative_3m:.1f}%"
            )

        elif relative_3m < 0:
            risks.append(
                f"Negative 3-month relative return: {relative_3m:.1f}%"
            )

    if relative_6m is not None:
        if relative_6m >= 20:
            score += 3
            positives.append(
                f"Strong 6-month outperformance: {relative_6m:.1f}%"
            )

        elif relative_6m >= 8:
            score += 2
            positives.append(
                f"Positive 6-month outperformance: {relative_6m:.1f}%"
            )

        elif relative_6m < 0:
            risks.append(
                f"Negative 6-month relative return: {relative_6m:.1f}%"
            )

    return {
        "score": min(score, 20),
        "positives": positives,
        "risks": risks,
    }


def score_multitimeframe_alignment(alignment):
    """Scores multi-timeframe trend alignment out of 20."""

    score = 0
    positives = []
    risks = []

    alignment_status = alignment.get(
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

    if alignment_status == "Strong multi-timeframe alignment":
        score += 16
        positives.append(
            "Strong Daily, Weekly and Monthly trend alignment"
        )

    elif alignment_status == "Bullish multi-timeframe alignment":
        score += 12
        positives.append(
            "Most chart timeframes are bullish"
        )

    elif alignment_status == "Mixed timeframe alignment":
        score += 5
        risks.append(
            "Timeframes are not fully aligned"
        )

    elif alignment_status == "Bearish multi-timeframe alignment":
        risks.append(
            "Most major chart timeframes are bearish"
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
    """
    Scores basic fundamental quality out of 20.

    Missing Yahoo Finance fields do not automatically imply weakness.
    """

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

    current_ratio = safe_number(
        fundamentals.get("currentRatio")
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
            risks.append(
                "Revenue growth is negative"
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
                "Earnings growth is negative"
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

    if current_ratio is not None:
        if current_ratio >= 1.50:
            score += 1
            positives.append(
                f"Healthy current ratio: {current_ratio:.2f}"
            )

        elif current_ratio < 0.75:
            risks.append(
                f"Low current ratio: {current_ratio:.2f}"
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
    """Scores individual chart timeframe trend quality out of 15."""

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
    """
    Scores bullish patterns out of 15.

    Weekly and Monthly confirmed breakouts are given higher weight.
    """

    score = 0
    positives = []
    risks = []

    confirmed_bullish_signal = False

    pattern_sets = [
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

    for timeframe, patterns in pattern_sets:
        for pattern in patterns:
            status = pattern.get("Status")
            direction = pattern.get("Direction")
            pattern_name = pattern.get("Pattern")

            if (
                status == "Confirmed"
                and direction == "Bullish"
            ):
                confirmed_bullish_signal = True

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
                    f"Bullish {pattern_name} is forming on {timeframe}"
                )

            elif (
                status == "Confirmed"
                and direction == "Bearish"
            ):
                risks.append(
                    f"Confirmed Bearish {pattern_name} on {timeframe}"
                )

    if not confirmed_bullish_signal:
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
    """Scores breakout volume confirmation out of 5."""

    volume_changes = []

    for pattern in daily_patterns + weekly_patterns:
        if (
            pattern.get("Status") == "Confirmed"
            and pattern.get("Direction") == "Bullish"
        ):
            volume_change = safe_number(
                pattern.get("Volume %")
            )

            if volume_change is not None:
                volume_changes.append(
                    volume_change
                )

    if not volume_changes:
        return {
            "score": 0,
            "positives": [],
            "risks": [],
        }

    highest_volume_change = max(
        volume_changes
    )

    if highest_volume_change >= 50:
        return {
            "score": 5,
            "positives": [
                (
                    "Strong breakout volume confirmation: "
                    f"{highest_volume_change:.1f}% above average"
                )
            ],
            "risks": [],
        }

    if highest_volume_change >= 25:
        return {
            "score": 4,
            "positives": [
                (
                    "Good breakout volume confirmation: "
                    f"{highest_volume_change:.1f}% above average"
                )
            ],
            "risks": [],
        }

    if highest_volume_change > 0:
        return {
            "score": 2,
            "positives": [
                (
                    "Positive breakout volume confirmation: "
                    f"{highest_volume_change:.1f}% above average"
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


def score_valuation_and_risk(fundamentals):
    """
    Scores valuation and leverage conditions out of 5.

    This is intentionally low weight because valuation needs
    sector-specific comparisons in a later version.
    """

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
    """
    Creates a transparent Winner Score out of 100.

    Score structure:
    - Fundamental Quality: 20
    - Relative Strength: 20
    - Multi-Timeframe Alignment: 20
    - Technical Trend: 15
    - Pattern Quality: 15
    - Volume Confirmation: 5
    - Valuation and Risk: 5
    """

    fundamental_component = score_fundamental_quality(
        fundamentals
    )

    relative_strength_component = score_relative_strength(
        relative_strength
    )

    alignment_component = score_multitimeframe_alignment(
        alignment
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

    valuation_component = score_valuation_and_risk(
        fundamentals
    )

    final_score = (
        fundamental_component["score"]
        + relative_strength_component["score"]
        + alignment_component["score"]
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
        fundamental_component["positives"]
        + relative_strength_component["positives"]
        + alignment_component["positives"]
        + technical_component["positives"]
        + pattern_component["positives"]
        + volume_component["positives"]
        + valuation_component["positives"]
    )

    risks = (
        fundamental_component["risks"]
        + relative_strength_component["risks"]
        + alignment_component["risks"]
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
# SUPPORT AND RESISTANCE
# =============================================================================

def calculate_support_resistance(data):
    """Calculates simple recent pivot support and resistance."""

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


# =============================================================================
# CHART FUNCTIONS
# =============================================================================

def create_candlestick_chart(
    data,
    signals,
    title,
):
    """Creates a candlestick chart with moving averages and volume."""

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
    relative_strength_line,
    symbol,
):
    """Creates a Relative Strength line chart versus Nifty 50."""

    figure = go.Figure()

    if (
        relative_strength_line is not None
        and not relative_strength_line.empty
    ):
        figure.add_trace(
            go.Scatter(
                x=relative_strength_line.index,
                y=relative_strength_line,
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
            f"{symbol} Relative Strength Line "
            "vs Nifty 50"
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
# APP HEADER
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
        Pattern analysis • Relative Strength • Multi-Timeframe Alignment •
        Winner Score • Fundamental quality • Support and resistance
    </div>
    """,
    unsafe_allow_html=True,
)

if universe_error:
    st.warning(universe_error)

if stock_universe.empty:
    st.error(
        "No stock universe is available. "
        "Please refresh cached data and try again."
    )
    st.stop()

if nifty_50_data.empty:
    st.warning(
        "Nifty 50 benchmark data is unavailable. "
        "Relative Strength data may be incomplete."
    )


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.header("🔍 Dashboard Controls")

    dashboard_mode = st.radio(
        "Research Mode",
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
        "Constituent cache: 6 hours"
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

    stock_record = stock_universe[
        stock_universe["Symbol"] == selected_symbol
    ].iloc[0]

    ticker = stock_record["Ticker"]
    company_name = stock_record["Company Name"]
    industry = stock_record["Industry"]

    with st.spinner(
        f"Loading data for {selected_symbol}..."
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
            "No price data is currently available "
            "for this stock."
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

    timeframe_alignment = (
        calculate_multitimeframe_alignment(
            daily_data,
            weekly_data,
            monthly_data,
        )
    )

    winner_score = calculate_winner_score(
        fundamentals=fundamentals,
        relative_strength=relative_strength,
        alignment=timeframe_alignment,
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

    market_cap = fundamentals.get("marketCap")

    market_cap_crore = (
        safe_number(market_cap, 0)
        / 10000000
    )

    header_col1, header_col2, header_col3, header_col4, header_col5 = (
        st.columns(5)
    )

    header_col1.metric(
        "Last Close",
        format_price(current_price),
    )

    header_col2.metric(
        "Market Cap",
        format_market_cap(market_cap),
    )

    header_col3.metric(
        "RS vs Nifty 50",
        relative_strength["status"],
    )

    header_col4.metric(
        "MTF Alignment",
        (
            f"{timeframe_alignment['score']}/10"
            if timeframe_alignment["score"] is not None
            else "Not available"
        ),
    )

    header_col5.metric(
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
            f"{selected_symbol} is below the selected "
            f"minimum market cap of ₹{minimum_market_cap:,.0f} Cr."
        )

    (
        winner_tab,
        technical_tab,
        relative_strength_tab,
        alignment_tab,
        fundamentals_tab,
    ) = st.tabs(
        [
            "🏆 Winner Score",
            "Technical Research",
            "Relative Strength",
            "MTF Alignment",
            "Fundamentals",
        ]
    )

    # =========================================================================
    # WINNER SCORE TAB
    # =========================================================================

    with winner_tab:
        st.subheader(
            "Winner Score and Ranking"
        )

        st.caption(
            "The score ranks available evidence. It is not a prediction "
            "and does not constitute a recommendation to buy, sell or hold."
        )

        score_value = winner_score["score"]
        score_category = winner_score["category"]

        card_class = "neutral-card"

        if score_value >= 80:
            card_class = "excellent-card"

        elif score_value >= 65:
            card_class = "watch-card"

        elif score_value < 35:
            card_class = "weak-card"

        st.markdown(
            f"""
            <div class="research-card {card_class}">
                <h2>Winner Score: {score_value}/100</h2>
                <h3>{score_category}</h3>
                <p>
                    Ranking combines relative strength, trend alignment,
                    technical breakout quality, available fundamentals,
                    volume confirmation and valuation-risk factors.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        score_breakdown = pd.DataFrame(
            [
                [
                    "Fundamental Quality",
                    winner_score["fundamental_score"],
                    20,
                ],
                [
                    "Relative Strength",
                    winner_score["relative_strength_score"],
                    20,
                ],
                [
                    "Multi-Timeframe Alignment",
                    winner_score["alignment_score"],
                    20,
                ],
                [
                    "Technical Trend",
                    winner_score["technical_score"],
                    15,
                ],
                [
                    "Pattern / Breakout Quality",
                    winner_score["pattern_score"],
                    15,
                ],
                [
                    "Volume Confirmation",
                    winner_score["volume_score"],
                    5,
                ],
                [
                    "Valuation / Risk",
                    winner_score["valuation_score"],
                    5,
                ],
            ],
            columns=[
                "Component",
                "Points",
                "Maximum",
            ],
        )

        score_breakdown["Strength %"] = (
            score_breakdown["Points"]
            / score_breakdown["Maximum"]
        ) * 100

        st.markdown("### Score Breakdown")

        st.dataframe(
            score_breakdown,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Points": st.column_config.NumberColumn(
                    "Points Earned",
                    format="%.1f",
                ),
                "Maximum": st.column_config.NumberColumn(
                    "Maximum Points",
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

        positive_col, risk_col = st.columns(2)

        with positive_col:
            st.markdown("### Positive Evidence")

            if winner_score["positives"]:
                for reason in winner_score["positives"][:18]:
                    st.success(
                        f"✅ {reason}"
                    )
            else:
                st.info(
                    "No strong positive rule-based evidence "
                    "is available at this time."
                )

        with risk_col:
            st.markdown("### Risk Flags")

            if winner_score["risks"]:
                for risk in winner_score["risks"][:18]:
                    st.warning(
                        f"⚠️ {risk}"
                    )
            else:
                st.success(
                    "No material rule-based risk flags were detected."
                )

        st.markdown("### Winner Score Categories")

        winner_categories = pd.DataFrame(
            [
                [
                    "80–100",
                    "Elite Candidate",
                    (
                        "Strong alignment of leadership, trends, "
                        "technical structure and available business quality."
                    ),
                ],
                [
                    "65–79",
                    "High-Conviction Watchlist",
                    (
                        "Good overall setup; review the remaining risks "
                        "and entry/stop-loss conditions."
                    ),
                ],
                [
                    "50–64",
                    "Watchlist",
                    (
                        "Some quality signals exist, but the complete "
                        "evidence is not fully aligned."
                    ),
                ],
                [
                    "35–49",
                    "Neutral / Mixed",
                    (
                        "Mixed signals; not currently a high-quality "
                        "winner-screen candidate."
                    ),
                ],
                [
                    "0–34",
                    "Avoid / Weak",
                    (
                        "Technical or fundamental evidence is weak, "
                        "missing, or materially misaligned."
                    ),
                ],
            ],
            columns=[
                "Score Range",
                "Classification",
                "Interpretation",
            ],
        )

        st.dataframe(
            winner_categories,
            hide_index=True,
            use_container_width=True,
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

        timeframe_items = [
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

        for tab, timeframe, timeframe_data, signals in timeframe_items:
            with tab:
                support, resistance = (
                    calculate_support_resistance(
                        timeframe_data
                    )
                )

                metric_one, metric_two, metric_three = (
                    st.columns(3)
                )

                metric_one.metric(
                    "Overall Trend",
                    calculate_overall_trend(
                        timeframe_data
                    ),
                )

                metric_two.metric(
                    "Nearest Support",
                    format_price(support),
                )

                metric_three.metric(
                    "Nearest Resistance",
                    format_price(resistance),
                )

                st.subheader(
                    f"{timeframe} Pattern Status"
                )

                if signals:
                    signals_table = pd.DataFrame(
                        signals
                    )

                    st.dataframe(
                        signals_table,
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
                        "No supported active pattern is currently detected."
                    )

                st.plotly_chart(
                    create_candlestick_chart(
                        timeframe_data,
                        signals,
                        f"{selected_symbol} — {timeframe}",
                    ),
                    use_container_width=True,
                )

    # =========================================================================
    # RELATIVE STRENGTH TAB
    # =========================================================================

    with relative_strength_tab:
        st.subheader(
            "Relative Strength vs Nifty 50"
        )

        st.caption(
            "Relative Return is the stock return minus the Nifty 50 return "
            "for the same period. Positive values indicate outperformance."
        )

        rs_metric_one, rs_metric_two, rs_metric_three = (
            st.columns(3)
        )

        rs_metric_one.metric(
            "RS Status",
            relative_strength["status"],
        )

        rs_metric_two.metric(
            "RS Line Trend",
            relative_strength["rs_trend"],
        )

        rs_metric_three.metric(
            "Benchmark",
            "Nifty 50 (^NSEI)",
        )

        rs_table = pd.DataFrame(
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
            rs_table,
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
                "Insufficient price data is available to draw the RS line."
            )

    # =========================================================================
    # MULTI-TIMEFRAME ALIGNMENT TAB
    # =========================================================================

    with alignment_tab:
        st.subheader(
            "Multi-Timeframe Trend Alignment"
        )

        daily_col, weekly_col, monthly_col = st.columns(3)

        daily_col.metric(
            "Daily Trend",
            timeframe_alignment["daily"],
        )

        weekly_col.metric(
            "Weekly Trend",
            timeframe_alignment["weekly"],
        )

        monthly_col.metric(
            "Monthly Trend",
            timeframe_alignment["monthly"],
        )

        alignment_col, status_col = st.columns(2)

        alignment_col.metric(
            "Alignment Score",
            (
                f"{timeframe_alignment['score']}/10"
                if timeframe_alignment["score"] is not None
                else "Not available"
            ),
        )

        status_col.metric(
            "Alignment Status",
            timeframe_alignment["status"],
        )

        st.markdown(
            f"""
            <div class="research-card">
                <h4>Market Structure</h4>
                <p>{timeframe_alignment["structure"]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # =========================================================================
    # FUNDAMENTALS TAB
    # =========================================================================

    with fundamentals_tab:
        st.subheader(
            "Basic Fundamental Quality"
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
                    display_value(
                        fundamentals.get("trailingPE")
                    ),
                ],
                [
                    "Forward P/E",
                    display_value(
                        fundamentals.get("forwardPE")
                    ),
                ],
                [
                    "Price / Book",
                    display_value(
                        fundamentals.get("priceToBook")
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
                    display_value(
                        fundamentals.get("debtToEquity")
                    ),
                ],
                [
                    "Current Ratio",
                    display_value(
                        fundamentals.get("currentRatio")
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
            "The Winner Score uses currently available Yahoo Finance "
            "fundamental fields. Detailed annual/quarterly Indian-company "
            "fundamentals and sector KPIs will be added later through "
            "NSE filing, XBRL, annual-report, or specialist data ingestion."
        )


# =============================================================================
# WINNER RANKING SCANNER
# =============================================================================

else:
    st.subheader(
        "Winner Ranking Scanner"
    )

    st.caption(
        "Ranks Nifty Total Market stocks using technical trend, "
        "relative strength, multi-timeframe alignment, available "
        "fundamentals, patterns, volume confirmation and valuation checks."
    )

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
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

        scanner_status = st.selectbox(
            "Pattern Status",
            [
                "Any",
                "Confirmed",
                "In progress",
                "Candidate",
            ],
        )

    with filter_col2:
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

    with filter_col3:
        scanner_min_score = st.slider(
            "Minimum Winner Score",
            min_value=0,
            max_value=100,
            value=50,
            step=5,
        )

        scanner_category = st.selectbox(
            "Winner Category",
            WINNER_CATEGORY_OPTIONS,
        )

        scan_limit = st.selectbox(
            "Maximum stocks to scan",
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
        "A complete Nifty 750 scan can be slow on free Streamlit Cloud. "
        "Start with 50 or 100 stocks to test the dashboard. "
        "Use 750 only when you are comfortable waiting several minutes."
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

        progress_bar = st.progress(0)
        status_label = st.empty()

        total_count = len(scan_universe)

        for row_number, (_, record) in enumerate(
            scan_universe.iterrows(),
            start=1,
        ):
            symbol = record["Symbol"]
            ticker = record["Ticker"]

            status_label.caption(
                f"Scanning {row_number:,} of {total_count:,}: {symbol}"
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
                    < scanner_min_score
                ):
                    continue

                if (
                    scanner_category != "Any"
                    and winner_score["category"]
                    != scanner_category
                ):
                    continue

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

                timeframe_data_map = {
                    "Daily": daily_data,
                    "Weekly": weekly_data,
                    "Monthly": monthly_data,
                }

                pattern_matches = []

                for timeframe in timeframes_to_scan:
                    timeframe_data = timeframe_data_map[
                        timeframe
                    ]

                    trend = calculate_overall_trend(
                        timeframe_data
                    )

                    if (
                        scanner_trend != "Any"
                        and trend != scanner_trend
                    ):
                        continue

                    timeframe_patterns = detect_patterns(
                        timeframe_data
                    )

                    for pattern_signal in timeframe_patterns:
                        matches_pattern = (
                            scanner_pattern == "Any"
                            or pattern_signal["Pattern"]
                            == scanner_pattern
                        )

                        matches_status = (
                            scanner_status == "Any"
                            or pattern_signal["Status"]
                            == scanner_status
                        )

                        if matches_pattern and matches_status:
                            pattern_matches.append(
                                {
                                    "Timeframe": timeframe,
                                    "Overall Trend": trend,
                                    **pattern_signal,
                                }
                            )

                # If a specific pattern filter is chosen, require it.
                if scanner_pattern != "Any":
                    if not pattern_matches:
                        continue

                # If no specific pattern is chosen, return top-ranked stocks
                # even if no current pattern is active.
                if not pattern_matches:
                    pattern_matches = [
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
                            "Notes": (
                                "Ranked using Winner Score; "
                                "no supported active pattern detected."
                            ),
                        }
                    ]

                for match in pattern_matches:
                    scan_results.append(
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

            progress_bar.progress(
                row_number / total_count
            )

        progress_bar.empty()
        status_label.empty()

        st.subheader(
            f"Winner Ranking Results: {len(scan_results)}"
        )

        if scan_results:
            scan_dataframe = pd.DataFrame(
                scan_results
            )

            scan_dataframe = scan_dataframe.sort_values(
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

            metric_col1, metric_col2, metric_col3, metric_col4 = (
                st.columns(4)
            )

            metric_col1.metric(
                "Ranked Stocks",
                len(scan_dataframe),
            )

            metric_col2.metric(
                "Elite Candidates",
                int(
                    (
                        scan_dataframe[
                            "Winner Category"
                        ]
                        == "Elite Candidate"
                    ).sum()
                ),
            )

            metric_col3.metric(
                "High-Conviction",
                int(
                    (
                        scan_dataframe[
                            "Winner Category"
                        ]
                        == "High-Conviction Watchlist"
                    ).sum()
                ),
            )

            metric_col4.metric(
                "RS Leaders",
                int(
                    (
                        scan_dataframe[
                            "RS Status"
                        ]
                        == "Leader"
                    ).sum()
                ),
            )

            st.dataframe(
                scan_dataframe,
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

            csv_output = scan_dataframe.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Download Winner Ranking CSV",
                data=csv_output,
                file_name="nifty_total_market_winner_ranking.csv",
                mime="text/csv",
            )

        else:
            st.info(
                "No stocks matched the selected Winner Score, Relative Strength, "
                "Alignment, Pattern, Trend and Market Cap filters."
            )


# =============================================================================
# FOOTER
# =============================================================================

st.divider()

st.caption(
    "Data sources: Nifty Indices constituent data and Yahoo Finance. "
    "Winner Score, technical patterns, relative strength and trend alignment "
    "are rule-based research tools. They may contain incomplete data, "
    "false positives, or stale figures. Verify all information independently. "
    "This dashboard is for educational and research purposes only; "
    "it is not investment advice."
)
