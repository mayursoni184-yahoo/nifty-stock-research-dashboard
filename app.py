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
    page_title="Nifty Total Market Command Center",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# STYLING
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
            background: #ffffff;
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

        .small-note {
            color: #6b7280;
            font-size: 0.9rem;
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
    """Safely converts a value to float."""

    try:
        if value is None:
            return default

        if pd.isna(value):
            return default

        return float(value)

    except Exception:
        return default


def format_price(value):
    """Formats an INR value safely."""

    value = safe_number(value)

    if value is None:
        return "Not available"

    return f"₹{value:,.2f}"


def format_percent(value):
    """Formats a signed percentage safely."""

    value = safe_number(value)

    if value is None:
        return "Not available"

    return f"{value:+.2f}%"


def format_market_cap(value):
    """Formats market cap to crore rupees."""

    value = safe_number(value)

    if value is None or value <= 0:
        return "Not available"

    return f"₹{value / 10000000:,.0f} Cr"


def get_status_priority(status):
    """Gives confirmed patterns the highest sort priority."""

    priorities = {
        "Confirmed": 1,
        "In progress": 2,
        "Candidate": 3,
        "No Pattern": 4,
    }

    return priorities.get(status, 99)


# =============================================================================
# NIFTY TOTAL MARKET UNIVERSE
# =============================================================================

@st.cache_data(ttl=21600, show_spinner=False)
def get_nifty_total_market_members():
    """
    Loads current Nifty Total Market constituents.

    Uses a fallback list if the official CSV is unavailable.
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
                "Constituent CSV does not contain Symbol."
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
                "Too few Nifty Total Market constituents loaded."
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
            "Live Nifty Total Market constituent data could not be loaded. "
            f"Using fallback universe. Reason: {type(error).__name__}: {error}"
        )

        return fallback, warning


# =============================================================================
# MARKET DATA
# =============================================================================

@st.cache_data(ttl=900, show_spinner=False)
def fetch_price_data(ticker, period="5y"):
    """Fetches OHLCV price history from Yahoo Finance."""

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
            data.index = data.index.tz_localize(None)

        return data

    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=43200, show_spinner=False)
def fetch_fundamentals(ticker):
    """Fetches available Yahoo Finance fundamental data."""

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
        data = ticker_object.get_info()

        return {
            field: data.get(field)
            for field in fields
        }

    except Exception:
        return {}


# =============================================================================
# PRICE RESAMPLING
# =============================================================================

def resample_ohlcv(data, timeframe):
    """Resamples daily data into Weekly or Monthly OHLCV candles."""

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
    Classifies overall moving-average trend.

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


def calculate_multitimeframe_alignment(
    daily_data,
    weekly_data,
    monthly_data,
):
    """Scores Daily, Weekly and Monthly alignment out of 10."""

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
        value in [
            "Strong bullish",
            "Bullish",
        ]
        for value in [
            daily_trend,
            weekly_trend,
            monthly_trend,
        ]
    )

    bearish_count = sum(
        value == "Bearish"
        for value in [
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
            "Timeframes are mixed; trend conviction is lower."
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
    """Calculates return over selected approximate trading-day periods."""

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


def align_stock_benchmark(stock_data, benchmark_data):
    """Aligns stock and benchmark prices to common dates."""

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
    Calculates relative return and RS-line condition vs Nifty 50.

    Relative Return = Stock Return - Benchmark Return.
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

    stock_prices = pd.DataFrame(
        {
            "close": aligned["stock_close"],
        }
    )

    benchmark_prices = pd.DataFrame(
        {
            "close": aligned["benchmark_close"],
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
        aligned["stock_close"]
        / aligned["benchmark_close"]
    )

    rs_trend = "Unavailable"

    if len(rs_line) >= 63:
        latest_value = float(
            rs_line.iloc[-1]
        )

        old_value = float(
            rs_line.iloc[-63]
        )

        if latest_value > old_value * 1.03:
            rs_trend = "Rising"

        elif latest_value < old_value * 0.97:
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
# PATTERN DETECTION
# =============================================================================

def find_pivots(values, pivot_type="high", order=3):
    """Finds local high and local low pivot points."""

    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) < (order * 2) + 1:
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


def build_pattern_signal(
    pattern,
    status,
    data,
    level,
    direction,
    notes,
    pattern_height=None,
):
    """Creates one standard technical signal row."""

    current_price = float(
        data["close"].iloc[-1]
    )

    volume_average = float(
        data["volume"]
        .tail(21)
        .iloc[:-1]
        .mean()
    )

    latest_volume = float(
        data["volume"].iloc[-1]
    )

    volume_change = (
        latest_volume / max(volume_average, 1) - 1
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
    """
    Detects basic active technical structures.

    These are rule-based research signals and do not replace
    discretionary chart review.
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

    signals = []

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
        top_one = pivot_highs[-2]
        top_two = pivot_highs[-1]

        difference = abs(
            high[top_one] - high[top_two]
        ) / max(high[top_one], 1)

        if (
            top_two - top_one >= 8
            and difference < 0.045
        ):
            neckline = float(
                np.min(
                    low[top_one:top_two + 1]
                )
            )

            peak = max(
                high[top_one],
                high[top_two],
            )

            pattern_height = (
                peak - neckline
            )

            status = (
                "Confirmed"
                if close[-1] < neckline * 0.99
                else "In progress"
            )

            signals.append(
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
                    pattern_height,
                )
            )

    # DOUBLE BOTTOM
    if len(pivot_lows) >= 2:
        bottom_one = pivot_lows[-2]
        bottom_two = pivot_lows[-1]

        difference = abs(
            low[bottom_one] - low[bottom_two]
        ) / max(low[bottom_one], 1)

        if (
            bottom_two - bottom_one >= 8
            and difference < 0.045
        ):
            neckline = float(
                np.max(
                    high[bottom_one:bottom_two + 1]
                )
            )

            base = min(
                low[bottom_one],
                low[bottom_two],
            )

            pattern_height = (
                neckline - base
            )

            status = (
                "Confirmed"
                if close[-1] > neckline * 1.01
                else "In progress"
            )

            signals.append(
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

            signals.append(
                build_pattern_signal(
                    "Head & Shoulders",
                    status,
                    data,
                    neckline,
                    "Bearish",
                    (
                        "Confirmation requires a close below neckline support."
                    ),
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

            signals.append(
                build_pattern_signal(
                    "Inverse Head & Shoulders",
                    status,
                    data,
                    neckline,
                    "Bullish",
                    (
                        "Confirmation requires a close above neckline resistance."
                    ),
                    pattern_height,
                )
            )

    # RECTANGLES AND TRIANGLES
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

        channel_size = (
            pattern_height
            / max(resistance, 1)
        )

        if channel_size < 0.15:
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

                signals.append(
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

    prior_low = float(
        np.min(low[-11:-1])
    )

    prior_high = float(
        np.max(high[-11:-1])
    )

    if (
        lower_shadow / candle_range > 0.55
        and candle_body / candle_range < 0.35
        and latest_close <= prior_low * 1.04
    ):
        signals.append(
            build_pattern_signal(
                "Reversal Bottom",
                "Candidate",
                data,
                latest_low,
                "Bullish",
                (
                    "Hammer-like candle near a local low. "
                    "Wait for confirmation."
                ),
                None,
            )
        )

    if (
        upper_shadow / candle_range > 0.55
        and candle_body / candle_range < 0.35
        and latest_close >= prior_high * 0.96
    ):
        signals.append(
            build_pattern_signal(
                "Reversal Top",
                "Candidate",
                data,
                latest_high,
                "Bearish",
                (
                    "Shooting-star-like candle near a local high. "
                    "Wait for confirmation."
                ),
                None,
            )
        )

    return signals


# =============================================================================
# SUPPORT, RESISTANCE, ATR AND TRADE-PLAN ENGINE
# =============================================================================

def calculate_support_resistance(data):
    """Calculates pivot-based nearby support and resistance."""

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
    """Calculates 14-period Average True Range."""

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

    atr = true_range.rolling(
        period
    ).mean()

    atr_value = atr.iloc[-1]

    if pd.isna(atr_value):
        return None

    return float(atr_value)


def calculate_pattern_target(pattern_signal):
    """
    Calculates basic measured-move target.

    Bullish: breakout level + pattern height.
    Bearish: breakdown level - pattern height.
    """

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
    """Classifies entry distance from breakout level."""

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
            "notes": "Price or breakout level is unavailable.",
        }

    if breakout_level == 0:
        return {
            "status": "Insufficient data",
            "distance_percent": None,
            "notes": "Breakout level is not valid.",
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

        notes = (
            "Price is at or below the breakout level. "
            "Confirmation should be checked."
        )

    elif distance <= 3:
        status = "Ideal Entry Zone"

        notes = (
            "Price is within 3% of the breakout level."
        )

    elif distance <= 7:
        status = "Acceptable Entry Zone"

        notes = (
            "Price is moderately above breakout level."
        )

    elif distance <= 12:
        status = "Extended"

        notes = (
            "Price has moved materially away from breakout."
        )

    else:
        status = "Avoid Chasing"

        notes = (
            "Price is highly extended from breakout level."
        )

    return {
        "status": status,
        "distance_percent": distance,
        "notes": notes,
    }


def calculate_trade_plan(data, pattern_signal):
    """
    Builds a rule-based research trade plan.

    Uses pivot support/resistance and ATR-based stop levels.
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
        14,
    )

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

        possible_stops = [
            stop
            for stop in [
                technical_stop,
                atr_stop,
            ]
            if stop is not None
            and stop < current_price
        ]

        selected_stop = (
            min(possible_stops)
            if possible_stops
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

        possible_stops = [
            stop
            for stop in [
                technical_stop,
                atr_stop,
            ]
            if stop is not None
            and stop > current_price
        ]

        selected_stop = (
            max(possible_stops)
            if possible_stops
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
            "risk_reward_status": "No directional plan",
        }

    risk_reward_ratio = None

    if (
        risk_per_share is not None
        and reward_per_share is not None
        and risk_per_share > 0
        and reward_per_share > 0
    ):
        risk_reward_ratio = (
            reward_per_share
            / risk_per_share
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
            "Daily, Weekly and Monthly trends are mixed"
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


def score_fundamentals(fundamentals):
    """Scores available basic fundamental quality out of 20."""

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


def score_technical_trends(
    daily_data,
    weekly_data,
    monthly_data,
):
    """Scores individual chart trend quality out of 15."""

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


def score_patterns(
    daily_patterns,
    weekly_patterns,
    monthly_patterns,
):
    """Scores current technical breakout quality out of 15."""

    score = 0
    positives = []
    risks = []

    confirmed_bullish = False

    all_pattern_sets = [
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

    for timeframe, patterns in all_pattern_sets:
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
            "No confirmed bullish breakout is currently detected"
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
    """Scores breakout volume confirmation out of 5."""

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
                bullish_volumes.append(
                    volume
                )

    if not bullish_volumes:
        return {
            "score": 0,
            "positives": [],
            "risks": [],
        }

    highest_volume = max(
        bullish_volumes
    )

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
            "Breakout volume is below average"
        ],
    }


def score_valuation_risk(fundamentals):
    """Scores basic valuation and leverage factors out of 5."""

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
    """Calculates transparent Winner Score out of 100."""

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
# COMMAND CENTER MARKET REGIME
# =============================================================================

def calculate_market_regime(
    nifty_50_data,
    breadth_data,
):
    """
    Calculates market regime from Nifty 50 trend and Nifty-universe breadth.
    """

    nifty_trend = calculate_overall_trend(
        nifty_50_data
    )

    breadth_above_50 = safe_number(
        breadth_data.get("above_50dma_percent")
    )

    breadth_above_200 = safe_number(
        breadth_data.get("above_200dma_percent")
    )

    if (
        nifty_trend == "Strong bullish"
        and breadth_above_50 is not None
        and breadth_above_50 >= 60
    ):
        regime = "Strong Bullish"

        description = (
            "Nifty 50 trend and market breadth are supportive."
        )

    elif (
        nifty_trend in [
            "Strong bullish",
            "Bullish",
        ]
        and breadth_above_50 is not None
        and breadth_above_50 >= 50
    ):
        regime = "Bullish"

        description = (
            "Market conditions are generally constructive."
        )

    elif (
        nifty_trend == "Bearish"
        or (
            breadth_above_50 is not None
            and breadth_above_50 < 35
        )
    ):
        regime = "Defensive"

        description = (
            "Market participation is weak; reduce breakout expectations."
        )

    else:
        regime = "Neutral"

        description = (
            "Market conditions are mixed."
        )

    return {
        "regime": regime,
        "nifty_trend": nifty_trend,
        "breadth_50": breadth_above_50,
        "breadth_200": breadth_above_200,
        "description": description,
    }


def calculate_market_breadth(scan_records):
    """
    Calculates breadth from scanned stocks.

    Breadth:
    Percentage of scanned stocks above 50 DMA and 200 DMA.
    """

    above_50_count = 0
    above_200_count = 0
    valid_count = 0

    for record in scan_records:
        close = safe_number(
            record.get("Current Price")
        )

        ma50 = safe_number(
            record.get("MA50")
        )

        ma200 = safe_number(
            record.get("MA200")
        )

        if close is None:
            continue

        valid_count += 1

        if ma50 is not None and close > ma50:
            above_50_count += 1

        if ma200 is not None and close > ma200:
            above_200_count += 1

    if valid_count == 0:
        return {
            "above_50dma_percent": None,
            "above_200dma_percent": None,
            "valid_count": 0,
        }

    return {
        "above_50dma_percent": (
            above_50_count / valid_count
        ) * 100,
        "above_200dma_percent": (
            above_200_count / valid_count
        ) * 100,
        "valid_count": valid_count,
    }


def get_moving_average_values(data):
    """Gets latest MA20, MA50 and MA200 values where available."""

    if data.empty:
        return {
            "MA20": None,
            "MA50": None,
            "MA200": None,
        }

    ma20 = (
        float(
            data["close"]
            .rolling(20)
            .mean()
            .iloc[-1]
        )
        if len(data) >= 20
        else None
    )

    ma50 = (
        float(
            data["close"]
            .rolling(50)
            .mean()
            .iloc[-1]
        )
        if len(data) >= 50
        else None
    )

    ma200 = (
        float(
            data["close"]
            .rolling(200)
            .mean()
            .iloc[-1]
        )
        if len(data) >= 200
        else None
    )

    return {
        "MA20": ma20,
        "MA50": ma50,
        "MA200": ma200,
    }


# =============================================================================
# COMMAND CENTER SCAN
# =============================================================================

def analyse_stock_for_command_center(
    symbol,
    ticker,
    company_name,
    industry,
    nifty_data,
    minimum_market_cap,
):
    """
    Builds a complete command-center record for one stock.
    """

    stock_data = fetch_price_data(
        ticker,
        "5y",
    )

    if stock_data.empty:
        return None

    fundamentals = fetch_fundamentals(
        ticker
    )

    market_cap = safe_number(
        fundamentals.get("marketCap"),
        0,
    )

    market_cap_crore = market_cap / 10000000

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

    relative_strength = calculate_relative_strength(
        stock_data,
        nifty_data,
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

    moving_averages = get_moving_average_values(
        daily_data
    )

    current_price = float(
        daily_data["close"].iloc[-1]
    )

    latest_date = daily_data.index[-1]

    all_patterns = []

    for timeframe_name, patterns in [
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
            all_patterns.append(
                {
                    "Timeframe": timeframe_name,
                    **pattern,
                }
            )

    bullish_confirmed_patterns = [
        pattern
        for pattern in all_patterns
        if (
            pattern.get("Status") == "Confirmed"
            and pattern.get("Direction") == "Bullish"
        )
    ]

    bearish_confirmed_patterns = [
        pattern
        for pattern in all_patterns
        if (
            pattern.get("Status") == "Confirmed"
            and pattern.get("Direction") == "Bearish"
        )
    ]

    selected_pattern = None

    if bullish_confirmed_patterns:
        priority = {
            "Monthly": 3,
            "Weekly": 2,
            "Daily": 1,
        }

        bullish_confirmed_patterns = sorted(
            bullish_confirmed_patterns,
            key=lambda item: priority.get(
                item["Timeframe"],
                0,
            ),
            reverse=True,
        )

        selected_pattern = bullish_confirmed_patterns[0]

    elif all_patterns:
        selected_pattern = all_patterns[0]

    trade_plan = None

    if selected_pattern is not None:
        selected_timeframe = selected_pattern[
            "Timeframe"
        ]

        if selected_timeframe == "Daily":
            plan_data = daily_data

        elif selected_timeframe == "Weekly":
            plan_data = weekly_data

        else:
            plan_data = monthly_data

        trade_plan = calculate_trade_plan(
            plan_data,
            selected_pattern,
        )

    support, resistance = calculate_support_resistance(
        daily_data
    )

    return {
        "Stock": symbol,
        "Company": company_name,
        "Industry": industry,
        "Ticker": ticker,
        "Date": latest_date,
        "Current Price": current_price,
        "Market Cap (Cr)": market_cap_crore,
        "Winner Score": winner_score["score"],
        "Winner Category": winner_score["category"],
        "RS Status": relative_strength["status"],
        "RS Trend": relative_strength["rs_trend"],
        "RS 3M %": relative_strength["relative_3m"],
        "RS 6M %": relative_strength["relative_6m"],
        "Daily Trend": alignment["daily"],
        "Weekly Trend": alignment["weekly"],
        "Monthly Trend": alignment["monthly"],
        "MTF Score": alignment["score"],
        "MTF Alignment": alignment["status"],
        "Current Pattern": (
            selected_pattern.get("Pattern")
            if selected_pattern is not None
            else "No active pattern"
        ),
        "Pattern Status": (
            selected_pattern.get("Status")
            if selected_pattern is not None
            else "No Pattern"
        ),
        "Pattern Direction": (
            selected_pattern.get("Direction")
            if selected_pattern is not None
            else "Neutral"
        ),
        "Pattern Timeframe": (
            selected_pattern.get("Timeframe")
            if selected_pattern is not None
            else "Overall"
        ),
        "Breakout Level": (
            selected_pattern.get("Level")
            if selected_pattern is not None
            else None
        ),
        "Pattern Volume %": (
            selected_pattern.get("Volume %")
            if selected_pattern is not None
            else None
        ),
        "Support": support,
        "Resistance": resistance,
        "MA20": moving_averages["MA20"],
        "MA50": moving_averages["MA50"],
        "MA200": moving_averages["MA200"],
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
            trade_plan["risk_reward_ratio"]
            if trade_plan is not None
            else None
        ),
        "Risk Reward Status": (
            trade_plan["risk_reward_status"]
            if trade_plan is not None
            else "No directional plan"
        ),
        "Bullish Confirmed Count": len(
            bullish_confirmed_patterns
        ),
        "Bearish Confirmed Count": len(
            bearish_confirmed_patterns
        ),
        "Risk Flags": " | ".join(
            winner_score["risks"][:5]
        ),
        "Positive Evidence": " | ".join(
            winner_score["positives"][:5]
        ),
    }


@st.cache_data(ttl=900, show_spinner=False)
def run_command_center_scan(
    universe_data,
    benchmark_data,
    minimum_market_cap,
    scan_limit,
):
    """
    Runs the command-center scan.

    The result is cached for 15 minutes.
    """

    scan_rows = []

    limited_universe = universe_data.head(
        scan_limit
    ).copy()

    for _, record in limited_universe.iterrows():
        result = analyse_stock_for_command_center(
            symbol=record["Symbol"],
            ticker=record["Ticker"],
            company_name=record["Company Name"],
            industry=record["Industry"],
            nifty_data=benchmark_data,
            minimum_market_cap=minimum_market_cap,
        )

        if result is not None:
            scan_rows.append(result)

    if not scan_rows:
        return pd.DataFrame()

    return pd.DataFrame(scan_rows)


# =============================================================================
# CHART FUNCTIONS
# =============================================================================

def create_candlestick_chart(
    data,
    patterns,
    title,
):
    """Creates stock candlestick chart with moving averages and volume."""

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
    """Creates RS line chart versus Nifty 50."""

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
# PAGE HEADER
# =============================================================================

st.markdown(
    """
    <div class="main-title">
        🏆 Nifty Total Market Daily Command Center
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="sub-title">
        Find stronger stocks, confirmed breakouts, ideal entry zones,
        extended setups and risk alerts from the Nifty Total Market universe.
    </div>
    """,
    unsafe_allow_html=True,
)

if universe_error:
    st.warning(universe_error)

if stock_universe.empty:
    st.error(
        "No stock universe is currently available."
    )
    st.stop()

if nifty_50_data.empty:
    st.warning(
        "Nifty 50 benchmark data could not be loaded. "
        "Relative strength results may be unavailable."
    )


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.header("🔍 Dashboard Controls")

    dashboard_mode = st.radio(
        "Dashboard View",
        [
            "Daily Command Center",
            "Stock Research",
            "Winner Ranking Scanner",
        ],
    )

    minimum_market_cap = st.number_input(
        "Minimum market cap (₹ crore)",
        min_value=0,
        value=3000,
        step=1000,
    )

    command_center_scan_limit = st.selectbox(
        "Command Center scan size",
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
        f"Nifty universe loaded: {len(stock_universe)} stocks"
    )

    if st.button("🔄 Refresh all cached data"):
        st.cache_data.clear()
        st.rerun()

    st.divider()

    st.caption(
        "Prices and command-center scans cache for 15 minutes.\n\n"
        "Fundamental data caches for 12 hours.\n\n"
        "The constituent list caches for 6 hours."
    )


# =============================================================================
# DAILY COMMAND CENTER
# =============================================================================

if dashboard_mode == "Daily Command Center":
    st.subheader(
        "Daily Market Command Center"
    )

    st.caption(
        "Run the scan once, then use the sections below to identify "
        "market regime, strongest candidates, breakouts, ideal entries, "
        "extended stocks, and risk signals."
    )

    if st.button(
        "▶ Run / Refresh Command Center Scan",
        type="primary",
    ):
        with st.spinner(
            "Scanning selected Nifty Total Market stocks..."
        ):
            command_center_data = run_command_center_scan(
                universe_data=stock_universe,
                benchmark_data=nifty_50_data,
                minimum_market_cap=minimum_market_cap,
                scan_limit=command_center_scan_limit,
            )

        st.session_state[
            "command_center_data"
        ] = command_center_data

    if "command_center_data" not in st.session_state:
        st.info(
            "Click Run / Refresh Command Center Scan to build "
            "today's market dashboard."
        )

    else:
        command_center_data = st.session_state[
            "command_center_data"
        ].copy()

        if command_center_data.empty:
            st.warning(
                "No eligible stocks were found in the selected scan range. "
                "Try reducing the minimum market-cap filter or increasing "
                "the scan size."
            )

        else:
            breadth = calculate_market_breadth(
                command_center_data.to_dict(
                    "records"
                )
            )

            market_regime = calculate_market_regime(
                nifty_50_data,
                breadth,
            )

            regime_card = "orange-card"

            if market_regime["regime"] in [
                "Strong Bullish",
                "Bullish",
            ]:
                regime_card = "green-card"

            elif market_regime["regime"] == "Defensive":
                regime_card = "red-card"

            st.markdown(
                f"""
                <div class="research-card {regime_card}">
                    <h3>Market Regime: {market_regime["regime"]}</h3>
                    <p>{market_regime["description"]}</p>
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
                "Stocks Above 50 DMA",
                (
                    f"{market_regime['breadth_50']:.1f}%"
                    if market_regime["breadth_50"] is not None
                    else "Not available"
                ),
            )

            regime_col3.metric(
                "Stocks Above 200 DMA",
                (
                    f"{market_regime['breadth_200']:.1f}%"
                    if market_regime["breadth_200"] is not None
                    else "Not available"
                ),
            )

            regime_col4.metric(
                "Scanned Eligible Stocks",
                market_regime["breadth_50"]
                and breadth["valid_count"],
            )

            st.divider()

            # -----------------------------------------------------------------
            # TOP WINNER CANDIDATES
            # -----------------------------------------------------------------

            st.subheader(
                "🏆 Top Winner Candidates"
            )

            top_candidates = (
                command_center_data
                .sort_values(
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
                .head(15)
                .copy()
            )

            st.dataframe(
                top_candidates[
                    [
                        "Stock",
                        "Company",
                        "Winner Score",
                        "Winner Category",
                        "RS Status",
                        "RS Trend",
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

            st.divider()

            # -----------------------------------------------------------------
            # NEW CONFIRMED BREAKOUTS
            # -----------------------------------------------------------------

            st.subheader(
                "📈 New Confirmed Breakouts"
            )

            confirmed_breakouts = command_center_data[
                (
                    command_center_data[
                        "Pattern Status"
                    ]
                    == "Confirmed"
                )
                & (
                    command_center_data[
                        "Pattern Direction"
                    ]
                    == "Bullish"
                )
            ].copy()

            confirmed_breakouts = (
                confirmed_breakouts
                .sort_values(
                    by=[
                        "Winner Score",
                        "Pattern Volume %",
                    ],
                    ascending=[
                        False,
                        False,
                    ],
                )
                .head(25)
            )

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
                    "No confirmed bullish breakouts were detected "
                    "in the scanned universe."
                )

            st.divider()

            # -----------------------------------------------------------------
            # IDEAL ENTRY ZONES
            # -----------------------------------------------------------------

            st.subheader(
                "🎯 Near Ideal Entry Zone"
            )

            ideal_entries = command_center_data[
                (
                    command_center_data[
                        "Entry Quality"
                    ]
                    .isin(
                        [
                            "Ideal Entry Zone",
                            "Acceptable Entry Zone",
                        ]
                    )
                )
                & (
                    command_center_data[
                        "Pattern Direction"
                    ]
                    == "Bullish"
                )
                & (
                    command_center_data[
                        "Winner Score"
                    ]
                    >= 50
                )
            ].copy()

            ideal_entries = (
                ideal_entries
                .sort_values(
                    by=[
                        "Winner Score",
                        "Risk Reward",
                    ],
                    ascending=[
                        False,
                        False,
                    ],
                )
                .head(25)
            )

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
                    "No bullish setups near an ideal or acceptable "
                    "entry zone were found."
                )

            st.divider()

            # -----------------------------------------------------------------
            # EXTENDED / AVOID CHASING
            # -----------------------------------------------------------------

            st.subheader(
                "⚠️ Extended / Avoid Chasing"
            )

            extended_setups = command_center_data[
                (
                    command_center_data[
                        "Entry Quality"
                    ]
                    .isin(
                        [
                            "Extended",
                            "Avoid Chasing",
                        ]
                    )
                )
                & (
                    command_center_data[
                        "Pattern Direction"
                    ]
                    == "Bullish"
                )
            ].copy()

            extended_setups = (
                extended_setups
                .sort_values(
                    by=[
                        "Distance From Breakout %",
                        "Winner Score",
                    ],
                    ascending=[
                        False,
                        False,
                    ],
                )
                .head(25)
            )

            if not extended_setups.empty:
                st.dataframe(
                    extended_setups[
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

            # -----------------------------------------------------------------
            # RISK ALERTS
            # -----------------------------------------------------------------

            st.subheader(
                "🚨 Risk Alerts"
            )

            risk_alerts = []

            for _, row in command_center_data.iterrows():
                stock_name = row["Stock"]
                current_price = safe_number(
                    row["Current Price"]
                )

                ma20 = safe_number(
                    row["MA20"]
                )

                ma50 = safe_number(
                    row["MA50"]
                )

                weekly_trend = row["Weekly Trend"]
                monthly_trend = row["Monthly Trend"]
                pattern_direction = row[
                    "Pattern Direction"
                ]
                pattern_status = row[
                    "Pattern Status"
                ]
                risk_reward = safe_number(
                    row["Risk Reward"]
                )
                rs_status = row["RS Status"]
                winner_score = safe_number(
                    row["Winner Score"]
                )

                if (
                    current_price is not None
                    and ma20 is not None
                    and current_price < ma20
                ):
                    risk_alerts.append(
                        {
                            "Stock": stock_name,
                            "Alert Type": "Below Daily 20 DMA",
                            "Details": (
                                f"Current price {format_price(current_price)} "
                                f"is below Daily MA20 {format_price(ma20)}."
                            ),
                            "Winner Score": winner_score,
                        }
                    )

                if weekly_trend == "Bearish":
                    risk_alerts.append(
                        {
                            "Stock": stock_name,
                            "Alert Type": "Weekly Bearish Trend",
                            "Details": (
                                "Weekly trend is Bearish."
                            ),
                            "Winner Score": winner_score,
                        }
                    )

                if monthly_trend == "Bearish":
                    risk_alerts.append(
                        {
                            "Stock": stock_name,
                            "Alert Type": "Monthly Bearish Trend",
                            "Details": (
                                "Monthly trend is Bearish."
                            ),
                            "Winner Score": winner_score,
                        }
                    )

                if (
                    pattern_status == "Confirmed"
                    and pattern_direction == "Bearish"
                ):
                    risk_alerts.append(
                        {
                            "Stock": stock_name,
                            "Alert Type": "Confirmed Bearish Pattern",
                            "Details": (
                                f"{row['Current Pattern']} is confirmed "
                                f"on {row['Pattern Timeframe']}."
                            ),
                            "Winner Score": winner_score,
                        }
                    )

                if rs_status == "Weak":
                    risk_alerts.append(
                        {
                            "Stock": stock_name,
                            "Alert Type": "Weak Relative Strength",
                            "Details": (
                                "Stock is underperforming Nifty 50 "
                                "over multiple measured periods."
                            ),
                            "Winner Score": winner_score,
                        }
                    )

                if (
                    risk_reward is not None
                    and risk_reward < 1.5
                    and row["Pattern Direction"] == "Bullish"
                ):
                    risk_alerts.append(
                        {
                            "Stock": stock_name,
                            "Alert Type": "Weak Risk / Reward",
                            "Details": (
                                f"Current rule-based risk/reward is "
                                f"1 : {risk_reward:.2f}."
                            ),
                            "Winner Score": winner_score,
                        }
                    )

            if risk_alerts:
                risk_alert_dataframe = pd.DataFrame(
                    risk_alerts
                )

                risk_alert_dataframe = (
                    risk_alert_dataframe
                    .sort_values(
                        by="Winner Score",
                        ascending=False,
                    )
                    .head(50)
                )

                st.dataframe(
                    risk_alert_dataframe,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Winner Score": st.column_config.ProgressColumn(
                            "Winner Score",
                            min_value=0,
                            max_value=100,
                            format="%.1f",
                        ),
                    },
                )

            else:
                st.success(
                    "No major rule-based risk alerts were found "
                    "in the current scanned universe."
                )

            st.divider()

            # -----------------------------------------------------------------
            # EXPORT
            # -----------------------------------------------------------------

            st.subheader(
                "⬇️ Export Daily Command Center Results"
            )

            export_data = command_center_data.sort_values(
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

            csv_data = export_data.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "Download Complete Command Center CSV",
                data=csv_data,
                file_name="nifty_daily_command_center.csv",
                mime="text/csv",
            )


# =============================================================================
# STOCK RESEARCH
# =============================================================================

elif dashboard_mode == "Stock Research":
    st.subheader(
        "Stock Research"
    )

    symbols = sorted(
        stock_universe["Symbol"].tolist()
    )

    default_symbol_index = (
        symbols.index("RELIANCE")
        if "RELIANCE" in symbols
        else 0
    )

    selected_symbol = st.selectbox(
        "Select a Nifty Total Market stock",
        symbols,
        index=default_symbol_index,
    )

    selected_record = stock_universe[
        stock_universe["Symbol"] == selected_symbol
    ].iloc[0]

    ticker = selected_record["Ticker"]

    with st.spinner(
        f"Loading research data for {selected_symbol}..."
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
            "Price data is unavailable for this stock."
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
        "RS Status",
        relative_strength["status"],
    )

    header_col4.metric(
        "MTF Alignment",
        (
            f"{alignment['score']}/10"
            if alignment["score"] is not None
            else "Not available"
        ),
    )

    header_col5.metric(
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

    with winner_tab:
        st.subheader(
            "Winner Score Breakdown"
        )

        breakdown = pd.DataFrame(
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
                    winner_score["alignment_score"],
                    20,
                ],
                [
                    "Technical Trend",
                    winner_score["technical_score"],
                    15,
                ],
                [
                    "Pattern Quality",
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
                    "Component Strength",
                    min_value=0,
                    max_value=100,
                    format="%.0f%%",
                ),
            },
        )

        reason_col, risk_col = st.columns(2)

        with reason_col:
            st.markdown("### Positive Evidence")

            if winner_score["positives"]:
                for item in winner_score["positives"][:15]:
                    st.success(f"✅ {item}")

        with risk_col:
            st.markdown("### Risk Flags")

            if winner_score["risks"]:
                for item in winner_score["risks"][:15]:
                    st.warning(f"⚠️ {item}")

    with technical_tab:
        daily_tab, weekly_tab, monthly_tab = st.tabs(
            [
                "Daily",
                "Weekly",
                "Monthly",
            ]
        )

        technical_views = [
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

        for tab, name, data, patterns in technical_views:
            with tab:
                support, resistance = (
                    calculate_support_resistance(
                        data
                    )
                )

                trend_col, support_col, resistance_col = (
                    st.columns(3)
                )

                trend_col.metric(
                    "Trend",
                    calculate_overall_trend(data),
                )

                support_col.metric(
                    "Support",
                    format_price(support),
                )

                resistance_col.metric(
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
                        "No supported technical patterns currently detected."
                    )

                st.plotly_chart(
                    create_candlestick_chart(
                        data,
                        patterns,
                        f"{selected_symbol} — {name}",
                    ),
                    use_container_width=True,
                )

    with rs_tab:
        st.subheader(
            "Relative Strength vs Nifty 50"
        )

        rs_col1, rs_col2, rs_col3 = st.columns(3)

        rs_col1.metric(
            "RS Status",
            relative_strength["status"],
        )

        rs_col2.metric(
            "RS Trend",
            relative_strength["rs_trend"],
        )

        rs_col3.metric(
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
        )

        st.plotly_chart(
            create_relative_strength_chart(
                relative_strength["rs_line"],
                selected_symbol,
            ),
            use_container_width=True,
        )

    with alignment_tab:
        st.subheader(
            "Multi-Timeframe Alignment"
        )

        daily_col, weekly_col, monthly_col = st.columns(3)

        daily_col.metric(
            "Daily",
            alignment["daily"],
        )

        weekly_col.metric(
            "Weekly",
            alignment["weekly"],
        )

        monthly_col.metric(
            "Monthly",
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

        financial_table = pd.DataFrame(
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
            financial_table,
            hide_index=True,
            use_container_width=True,
        )


# =============================================================================
# WINNER RANKING SCANNER
# =============================================================================

else:
    st.subheader(
        "Winner Ranking Scanner"
    )

    st.caption(
        "Rank stocks using Winner Score, Relative Strength, trend alignment, "
        "technical pattern status, breakout quality and market-cap filters."
    )

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        selected_pattern = st.selectbox(
            "Pattern",
            PATTERN_OPTIONS,
        )

        selected_timeframe = st.selectbox(
            "Pattern Timeframe",
            [
                "Any",
                "Daily",
                "Weekly",
                "Monthly",
            ],
        )

        selected_pattern_status = st.selectbox(
            "Pattern Status",
            [
                "Any",
                "Confirmed",
                "In progress",
                "Candidate",
            ],
        )

    with filter_col2:
        selected_trend = st.selectbox(
            "Overall Trend",
            TREND_OPTIONS,
        )

        selected_rs_status = st.selectbox(
            "Relative Strength",
            RS_STATUS_OPTIONS,
        )

        selected_alignment = st.selectbox(
            "MTF Alignment",
            MTF_ALIGNMENT_OPTIONS,
        )

    with filter_col3:
        minimum_winner_score = st.slider(
            "Minimum Winner Score",
            min_value=0,
            max_value=100,
            value=50,
            step=5,
        )

        selected_category = st.selectbox(
            "Winner Category",
            WINNER_CATEGORY_OPTIONS,
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
        )

    st.warning(
        "Start with 50 or 100 stocks on free Streamlit Cloud. "
        "A 750-stock scan can take several minutes."
    )

    if st.button(
        "🏆 Run Winner Ranking Scan",
        type="primary",
    ):
        with st.spinner(
            "Running Winner Ranking scan..."
        ):
            scanner_data = run_command_center_scan(
                universe_data=stock_universe,
                benchmark_data=nifty_50_data,
                minimum_market_cap=minimum_market_cap,
                scan_limit=scanner_limit,
            )

        if scanner_data.empty:
            st.warning(
                "No eligible stocks were found."
            )

        else:
            filtered_data = scanner_data.copy()

            filtered_data = filtered_data[
                filtered_data["Winner Score"]
                >= minimum_winner_score
            ]

            if selected_category != "Any":
                filtered_data = filtered_data[
                    filtered_data["Winner Category"]
                    == selected_category
                ]

            if selected_rs_status != "Any":
                filtered_data = filtered_data[
                    filtered_data["RS Status"]
                    == selected_rs_status
                ]

            if selected_alignment != "Any":
                filtered_data = filtered_data[
                    filtered_data["MTF Alignment"]
                    == selected_alignment
                ]

            if selected_pattern != "Any":
                filtered_data = filtered_data[
                    filtered_data["Current Pattern"]
                    == selected_pattern
                ]

            if selected_pattern_status != "Any":
                filtered_data = filtered_data[
                    filtered_data["Pattern Status"]
                    == selected_pattern_status
                ]

            if selected_timeframe != "Any":
                filtered_data = filtered_data[
                    filtered_data["Pattern Timeframe"]
                    == selected_timeframe
                ]

            if selected_trend != "Any":
                filtered_data = filtered_data[
                    (
                        filtered_data["Daily Trend"]
                        == selected_trend
                    )
                    | (
                        filtered_data["Weekly Trend"]
                        == selected_trend
                    )
                    | (
                        filtered_data["Monthly Trend"]
                        == selected_trend
                    )
                ]

            filtered_data = filtered_data.sort_values(
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
                f"Ranked Results: {len(filtered_data)}"
            )

            st.dataframe(
                filtered_data[
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

            csv_data = filtered_data.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Download Winner Ranking CSV",
                data=csv_data,
                file_name="nifty_winner_ranking.csv",
                mime="text/csv",
            )


# =============================================================================
# FOOTER
# =============================================================================

st.divider()

st.caption(
    "Data sources: Nifty Indices constituent list and Yahoo Finance. "
    "The Command Center, Winner Score, technical patterns, breadth, "
    "relative strength, risk/reward levels and risk alerts are rule-based "
    "research tools. They may be incomplete, delayed, or produce false "
    "positives. This dashboard is for education and research only and is "
    "not investment advice."
)
