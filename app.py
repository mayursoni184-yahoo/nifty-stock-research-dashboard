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

TARGET_OPTIONS = [5, 10, 15]
STOP_OPTIONS = [5, 8, 10, 12]
HORIZON_OPTIONS = [40, 60, 90]


# =============================================================================
# GENERAL UTILITIES
# =============================================================================

def safe_number(value, default=None):
    """Convert a value safely into float."""

    try:
        if value is None:
            return default

        if pd.isna(value):
            return default

        return float(value)

    except Exception:
        return default


def format_price(value):
    """Format a numeric value as INR price."""

    value = safe_number(value)

    if value is None:
        return "Not available"

    return f"₹{value:,.2f}"


def format_percent(value):
    """Format signed percentage values."""

    value = safe_number(value)

    if value is None:
        return "Not available"

    return f"{value:+.2f}%"


def format_market_cap(value):
    """Format market capitalization in crore rupees."""

    value = safe_number(value)

    if value is None or value <= 0:
        return "Not available"

    return f"₹{value / 10000000:,.0f} Cr"


def calculate_percent_change(current_value, old_value):
    """Calculate percentage change safely."""

    current_value = safe_number(current_value)
    old_value = safe_number(old_value)

    if current_value is None:
        return None

    if old_value is None:
        return None

    if old_value == 0:
        return None

    return (
        (current_value - old_value)
        / abs(old_value)
    ) * 100


def clip_score(value, lower=0, upper=100):
    """Limit a score to a selected range."""

    value = safe_number(value, 0)

    return max(
        lower,
        min(value, upper),
    )


# =============================================================================
# NIFTY TOTAL MARKET UNIVERSE
# =============================================================================

@st.cache_data(ttl=21600, show_spinner=False)
def get_nifty_total_market_members():
    """
    Load Nifty Total Market members with a safe fallback list.
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
                "Too few constituents loaded."
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
            "Live Nifty Total Market list unavailable. "
            f"Using fallback symbols. Reason: {type(error).__name__}: {error}"
        )

        return fallback, warning


# =============================================================================
# PRICE AND FUNDAMENTAL DATA
# =============================================================================

@st.cache_data(ttl=900, show_spinner=False)
def fetch_price_data(ticker, period="5y"):
    """Fetch OHLCV price data from Yahoo Finance."""

    try:
        ticker_object = yf.Ticker(ticker)

        data = ticker_object.history(
            period=period,
            auto_adjust=True,
        )

        if data is None or data.empty:
            return pd.DataFrame()

        data = data.rename(columns=str.lower)

        data = data[
            [
                "open",
                "high",
                "low",
                "close",
                "volume",
            ]
        ].dropna()

        data.index = pd.to_datetime(data.index)

        if getattr(data.index, "tz", None) is not None:
            data.index = data.index.tz_localize(None)

        return data

    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=43200, show_spinner=False)
def fetch_fundamentals(ticker):
    """Fetch available fundamental summary data."""

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
# OHLCV RESAMPLING
# =============================================================================

def resample_ohlcv(data, timeframe):
    """Resample daily OHLCV data into Weekly or Monthly data."""

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
    """Calculate trend from MA20, MA50 and MA200."""

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
    """Calculate Daily, Weekly, Monthly alignment score."""

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
        structure = "Daily, Weekly and Monthly trends align upward."

    elif bullish_count >= 2:
        status = "Bullish multi-timeframe alignment"
        structure = "Most chart timeframes are bullish."

    elif bearish_count >= 2:
        status = "Bearish multi-timeframe alignment"
        structure = "Most chart timeframes are bearish."

    else:
        status = "Mixed timeframe alignment"
        structure = "Daily, Weekly and Monthly trends disagree."

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

def calculate_period_return(data, days):
    """Calculate historical return over a number of trading days."""

    if data is None or data.empty:
        return None

    if len(data) <= days:
        return None

    current_price = float(
        data["close"].iloc[-1]
    )

    earlier_price = float(
        data["close"].iloc[-days - 1]
    )

    if earlier_price == 0:
        return None

    return (
        current_price / earlier_price - 1
    ) * 100


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

    aligned = stock_close.join(
        benchmark_close,
        how="inner",
    )

    return aligned.dropna()


def calculate_relative_strength(
    stock_data,
    benchmark_data,
):
    """Calculate relative returns and RS status vs Nifty 50."""

    aligned = align_stock_benchmark(
        stock_data,
        benchmark_data,
    )

    empty = {
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
        return empty

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
        current_rs = float(rs_line.iloc[-1])
        old_rs = float(rs_line.iloc[-63])

        if current_rs > old_rs * 1.03:
            rs_trend = "Rising"

        elif current_rs < old_rs * 0.97:
            rs_trend = "Falling"

        else:
            rs_trend = "Flat"

    all_relative_returns = [
        value
        for value in [
            relative_1m,
            relative_3m,
            relative_6m,
            relative_12m,
        ]
        if value is not None
    ]

    if len(all_relative_returns) < 2:
        status = "Insufficient data"

    else:
        positive_count = sum(
            value > 0
            for value in all_relative_returns
        )

        strong_count = sum(
            value > 5
            for value in all_relative_returns
        )

        negative_count = sum(
            value < 0
            for value in all_relative_returns
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
# TECHNICAL PATTERN DETECTION
# =============================================================================

def find_pivots(values, pivot_type="high", order=3):
    """Find local high and low pivot indexes."""

    values = np.asarray(
        values,
        dtype=float,
    )

    if len(values) < (order * 2) + 1:
        return []

    indexes = []

    for index in range(order, len(values) - order):
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
    pattern_height=None,
):
    """Build a standard technical pattern signal."""

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
    """Detect active rule-based price patterns."""

    if data is None or data.empty:
        return []

    data = data.dropna().copy()

    if len(data) < 40:
        return []

    high = data["high"].to_numpy(dtype=float)
    low = data["low"].to_numpy(dtype=float)
    close = data["close"].to_numpy(dtype=float)
    open_price = data["open"].to_numpy(dtype=float)

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

            signals.append(
                build_pattern_signal(
                    "Double Top",
                    status,
                    data,
                    neckline,
                    "Bearish",
                    "Two similar highs; confirmation is below neckline.",
                    height,
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

            signals.append(
                build_pattern_signal(
                    "Double Bottom",
                    status,
                    data,
                    neckline,
                    "Bullish",
                    "Two similar lows; confirmation is above neckline.",
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

        neckline = min(left_trough, right_trough)

        shoulder_difference = abs(
            high[left_shoulder]
            - high[right_shoulder]
        ) / max(shoulder_average, 1)

        if (
            high[head] > shoulder_average * 1.04
            and shoulder_difference < 0.10
        ):
            height = high[head] - neckline

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
                    "Confirmation is below neckline support.",
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

        neckline = max(left_peak, right_peak)

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

            signals.append(
                build_pattern_signal(
                    "Inverse Head & Shoulders",
                    status,
                    data,
                    neckline,
                    "Bullish",
                    "Confirmation is above neckline resistance.",
                    height,
                )
            )

    # RECTANGLE / TRIANGLE
    formation_window = 30

    if len(data) >= formation_window:
        recent_highs = high[-formation_window:]
        recent_lows = low[-formation_window:]

        x_axis = np.arange(formation_window)

        high_slope = (
            np.polyfit(
                x_axis,
                recent_highs,
                1,
            )[0]
            / max(np.mean(recent_highs), 1)
        )

        low_slope = (
            np.polyfit(
                x_axis,
                recent_lows,
                1,
            )[0]
            / max(np.mean(recent_lows), 1)
        )

        resistance = float(np.max(recent_highs))
        support = float(np.min(recent_lows))

        height = resistance - support

        width_ratio = (
            height / max(resistance, 1)
        )

        if width_ratio < 0.15:
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
                        "Breakout confirmation requires close outside range.",
                        height,
                    )
                )

    # REVERSAL BOTTOM / TOP
    latest_open = float(open_price[-1])
    latest_close = float(close[-1])
    latest_high = float(high[-1])
    latest_low = float(low[-1])

    body = abs(latest_close - latest_open)
    candle_range = max(latest_high - latest_low, 0.000001)

    lower_shadow = (
        min(latest_open, latest_close)
        - latest_low
    )

    upper_shadow = (
        latest_high
        - max(latest_open, latest_close)
    )

    previous_low = float(np.min(low[-11:-1]))
    previous_high = float(np.max(high[-11:-1]))

    if (
        lower_shadow / candle_range > 0.55
        and body / candle_range < 0.35
        and latest_close <= previous_low * 1.04
    ):
        signals.append(
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
        signals.append(
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

    return signals


# =============================================================================
# VOLATILITY, SUPPORT AND RISK ENGINE
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

    result = atr.iloc[-1]

    if pd.isna(result):
        return None

    return float(result)


def calculate_support_resistance(data):
    """Calculate nearby pivot support and resistance."""

    if data is None or data.empty:
        return None, None

    if len(data) < 30:
        return (
            float(data["low"].tail(10).min()),
            float(data["high"].tail(10).max()),
        )

    current_price = float(data["close"].iloc[-1])

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
# PHASE 1 HISTORICAL SETUP STUDY
# =============================================================================

def historical_daily_trend(data, index):
    """Calculate daily trend using only data available at historical index."""

    if index < 55:
        return "Insufficient data"

    historical_close = data["close"].iloc[
        :index + 1
    ]

    current_close = float(
        historical_close.iloc[-1]
    )

    ma20 = float(
        historical_close
        .rolling(20)
        .mean()
        .iloc[-1]
    )

    ma50 = float(
        historical_close
        .rolling(50)
        .mean()
        .iloc[-1]
    )

    if index >= 199:
        ma200 = float(
            historical_close
            .rolling(200)
            .mean()
            .iloc[-1]
        )

        if current_close > ma20 > ma50 > ma200:
            return "Strong bullish"

    if current_close > ma20 > ma50:
        return "Bullish"

    if current_close < ma20 < ma50:
        return "Bearish"

    return "Neutral / consolidating"


def historical_weekly_trend(daily_data, index):
    """Calculate weekly trend using daily data only up to historical index."""

    if index < 100:
        return "Insufficient data"

    available_data = daily_data.iloc[
        :index + 1
    ].copy()

    weekly_data = resample_ohlcv(
        available_data,
        "Weekly",
    )

    if len(weekly_data) < 55:
        return "Insufficient data"

    return calculate_overall_trend(
        weekly_data
    )


def historical_relative_strength_3m(
    stock_data,
    benchmark_data,
    index,
):
    """Calculate 3-month RS using information only up to historical date."""

    if index < 63:
        return None

    event_date = stock_data.index[index]

    stock_slice = stock_data.iloc[
        :index + 1
    ].copy()

    benchmark_slice = benchmark_data[
        benchmark_data.index <= event_date
    ].copy()

    aligned = align_stock_benchmark(
        stock_slice,
        benchmark_slice,
    )

    if len(aligned) < 64:
        return None

    stock_current = float(
        aligned["stock_close"].iloc[-1]
    )

    stock_old = float(
        aligned["stock_close"].iloc[-64]
    )

    benchmark_current = float(
        aligned["benchmark_close"].iloc[-1]
    )

    benchmark_old = float(
        aligned["benchmark_close"].iloc[-64]
    )

    if stock_old == 0 or benchmark_old == 0:
        return None

    stock_return = (
        stock_current / stock_old - 1
    ) * 100

    benchmark_return = (
        benchmark_current / benchmark_old - 1
    ) * 100

    return stock_return - benchmark_return


def historical_52w_high_distance(data, index):
    """Calculate historical distance from 52-week high."""

    if index < 20:
        return None

    lookback_start = max(0, index - 251)

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


def historical_volume_change(data, index):
    """Calculate historical volume versus prior 20-day average."""

    if index < 21:
        return None

    current_volume = float(
        data["volume"].iloc[index]
    )

    prior_average_volume = float(
        data["volume"]
        .iloc[index - 20:index]
        .mean()
    )

    if prior_average_volume == 0:
        return None

    return (
        current_volume / prior_average_volume - 1
    ) * 100


def historical_atr_percent(data, index, period=14):
    """Calculate historical ATR as percentage of price."""

    if index < period + 1:
        return None

    historical_data = data.iloc[
        :index + 1
    ].copy()

    high = historical_data["high"]
    low = historical_data["low"]
    close = historical_data["close"]

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = true_range.rolling(period).mean().iloc[-1]

    if pd.isna(atr):
        return None

    current_close = float(
        historical_data["close"].iloc[-1]
    )

    if current_close == 0:
        return None

    return (
        float(atr) / current_close
    ) * 100


def historical_momentum_1m(data, index):
    """Calculate 1-month momentum at a historical event date."""

    if index < 21:
        return None

    current_close = float(
        data["close"].iloc[index]
    )

    old_close = float(
        data["close"].iloc[index - 21]
    )

    if old_close == 0:
        return None

    return (
        current_close / old_close - 1
    ) * 100


def historical_ma50_distance(data, index):
    """Calculate price distance from 50 DMA at historical setup date."""

    if index < 50:
        return None

    current_close = float(
        data["close"].iloc[index]
    )

    ma50 = float(
        data["close"]
        .iloc[:index + 1]
        .rolling(50)
        .mean()
        .iloc[-1]
    )

    if ma50 == 0:
        return None

    return (
        current_close / ma50 - 1
    ) * 100


def is_historical_bullish_setup(
    stock_data,
    benchmark_data,
    index,
    require_strong_daily,
    require_weekly_bullish,
    require_positive_rs,
    max_distance_52w,
    minimum_volume_change,
):
    """
    Defines transparent Phase 1 bullish historical setup condition.

    It uses no future data for setup classification.
    """

    daily_trend = historical_daily_trend(
        stock_data,
        index,
    )

    weekly_trend = historical_weekly_trend(
        stock_data,
        index,
    )

    rs_3m = historical_relative_strength_3m(
        stock_data,
        benchmark_data,
        index,
    )

    distance_52w = historical_52w_high_distance(
        stock_data,
        index,
    )

    volume_change = historical_volume_change(
        stock_data,
        index,
    )

    atr_percent = historical_atr_percent(
        stock_data,
        index,
    )

    momentum_1m = historical_momentum_1m(
        stock_data,
        index,
    )

    ma50_distance = historical_ma50_distance(
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

    high_condition = (
        distance_52w is not None
        and distance_52w >= max_distance_52w
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
        and high_condition
        and volume_condition
    )

    return {
        "qualifies": qualifies,
        "Daily Trend": daily_trend,
        "Weekly Trend": weekly_trend,
        "RS 3M %": rs_3m,
        "Distance from 52W High %": distance_52w,
        "Volume vs 20D Avg %": volume_change,
        "ATR %": atr_percent,
        "Momentum 1M %": momentum_1m,
        "Distance from MA50 %": ma50_distance,
    }


def evaluate_triple_barrier(
    stock_data,
    entry_index,
    target_percent,
    stop_percent,
    horizon_days,
):
    """
    Evaluate target, stop, and time barriers for a bullish historical setup.
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

    final_index = min(
        entry_index + horizon_days,
        len(stock_data) - 1,
    )

    if final_index <= entry_index:
        return None

    future_data = stock_data.iloc[
        entry_index + 1:final_index + 1
    ].copy()

    max_high = float(
        future_data["high"].max()
    )

    min_low = float(
        future_data["low"].min()
    )

    mfe = (
        max_high / entry_price - 1
    ) * 100

    mae = (
        min_low / entry_price - 1
    ) * 100

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

    forward_return = (
        final_close / entry_price - 1
    ) * 100

    return {
        "Entry Date": entry_date,
        "Entry Price": entry_price,
        "Target Price": target_price,
        "Stop Price": stop_price,
        "Outcome": outcome,
        "Outcome Date": outcome_date,
        "Days to Outcome": days_to_outcome,
        "Final Close": final_close,
        "Forward Return %": forward_return,
        "Maximum Favourable Excursion %": mfe,
        "Maximum Adverse Excursion %": mae,
    }


@st.cache_data(ttl=43200, show_spinner=False)
def run_historical_study(
    stock_data,
    benchmark_data,
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
    Run Phase 1 historical event study and retain feature snapshots
    needed for Phase 2 similarity analysis.
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

    final_index = (
        len(stock_data) - horizon_days - 1
    )

    for index in range(
        minimum_history,
        final_index,
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
            max_distance_52w=max_distance_52w,
            minimum_volume_change=minimum_volume_change,
        )

        if not setup["qualifies"]:
            continue

        barrier_result = evaluate_triple_barrier(
            stock_data=stock_data,
            entry_index=index,
            target_percent=target_percent,
            stop_percent=stop_percent,
            horizon_days=horizon_days,
        )

        if barrier_result is None:
            continue

        barrier_result.update(setup)

        events.append(barrier_result)

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


def summarise_study(events):
    """Create Phase 1 summary metrics."""

    if events is None or events.empty:
        return {
            "events": 0,
            "target_hit_rate": None,
            "stop_hit_rate": None,
            "median_return": None,
            "average_return": None,
            "median_mfe": None,
            "median_mae": None,
            "confidence": "Insufficient sample",
        }

    valid = events[
        events["Outcome"]
        != "Ambiguous Same Day"
    ].copy()

    if valid.empty:
        target_hit_rate = None
        stop_hit_rate = None
    else:
        target_hit_rate = (
            (
                valid["Outcome"]
                == "Target Hit First"
            ).mean()
            * 100
        )

        stop_hit_rate = (
            (
                valid["Outcome"]
                == "Stop Hit First"
            ).mean()
            * 100
        )

    count = len(events)

    if count >= 50:
        confidence = "High"
    elif count >= 25:
        confidence = "Medium"
    elif count >= 10:
        confidence = "Low"
    else:
        confidence = "Insufficient sample"

    return {
        "events": count,
        "target_hit_rate": target_hit_rate,
        "stop_hit_rate": stop_hit_rate,
        "median_return": float(
            events["Forward Return %"].median()
        ),
        "average_return": float(
            events["Forward Return %"].mean()
        ),
        "median_mfe": float(
            events[
                "Maximum Favourable Excursion %"
            ].median()
        ),
        "median_mae": float(
            events[
                "Maximum Adverse Excursion %"
            ].median()
        ),
        "confidence": confidence,
    }


# =============================================================================
# PHASE 2 SIMILARITY ENGINE
# =============================================================================

def calculate_current_setup_features(
    stock_data,
    benchmark_data,
):
    """
    Build a current-state feature vector for similarity matching.
    """

    if stock_data is None or stock_data.empty:
        return {}

    index = len(stock_data) - 1

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

    rs_data = calculate_relative_strength(
        stock_data,
        benchmark_data,
    )

    current_close = float(
        stock_data["close"].iloc[-1]
    )

    if len(stock_data) >= 252:
        high_52w = float(
            stock_data["high"]
            .tail(252)
            .max()
        )
    else:
        high_52w = float(
            stock_data["high"].max()
        )

    distance_52w = (
        (current_close - high_52w)
        / high_52w
    ) * 100

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

        volume_change = (
            latest_volume
            / max(average_volume, 1)
            - 1
        ) * 100
    else:
        volume_change = None

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
        stock_data,
        21,
    )

    if len(stock_data) >= 50:
        ma50 = float(
            stock_data["close"]
            .rolling(50)
            .mean()
            .iloc[-1]
        )

        ma50_distance = (
            current_close / ma50 - 1
        ) * 100
    else:
        ma50_distance = None

    return {
        "Daily Trend": daily_trend,
        "Weekly Trend": weekly_trend,
        "RS 3M %": rs_data.get(
            "relative_3m"
        ),
        "Distance from 52W High %": distance_52w,
        "Volume vs 20D Avg %": volume_change,
        "ATR %": atr_percent,
        "Momentum 1M %": momentum_1m,
        "Distance from MA50 %": ma50_distance,
    }


def categorical_similarity(
    current_value,
    historical_value,
    full_points,
):
    """Score categorical similarity."""

    if current_value is None or historical_value is None:
        return full_points * 0.50

    if current_value == historical_value:
        return full_points

    bullish_values = [
        "Strong bullish",
        "Bullish",
    ]

    if (
        current_value in bullish_values
        and historical_value in bullish_values
    ):
        return full_points * 0.70

    if (
        current_value == "Neutral / consolidating"
        and historical_value == "Neutral / consolidating"
    ):
        return full_points

    return 0


def numeric_similarity(
    current_value,
    historical_value,
    tolerance,
    full_points,
):
    """
    Score numerical feature similarity.

    Full points when the historical value is equal to current value.
    Score declines linearly toward zero at 2 × tolerance.
    """

    current_value = safe_number(current_value)
    historical_value = safe_number(historical_value)

    if current_value is None or historical_value is None:
        return full_points * 0.50

    difference = abs(
        current_value - historical_value
    )

    if difference <= tolerance:
        return full_points

    if difference >= tolerance * 2:
        return 0

    proportional_score = 1 - (
        (difference - tolerance)
        / tolerance
    )

    return full_points * proportional_score


def calculate_similarity_score(
    current_features,
    historical_event,
):
    """
    Calculate similarity score out of 100.

    Components:
    Daily Trend:                 15 points
    Weekly Trend:                15 points
    Relative Strength 3M:        15 points
    Distance from 52W High:      15 points
    Volume condition:            10 points
    ATR % volatility:            10 points
    1M momentum:                 10 points
    Distance from MA50:          10 points
    """

    components = {}

    components["Daily Trend"] = categorical_similarity(
        current_features.get("Daily Trend"),
        historical_event.get("Daily Trend"),
        15,
    )

    components["Weekly Trend"] = categorical_similarity(
        current_features.get("Weekly Trend"),
        historical_event.get("Weekly Trend"),
        15,
    )

    components["RS 3M"] = numeric_similarity(
        current_features.get("RS 3M %"),
        historical_event.get("RS 3M %"),
        tolerance=8,
        full_points=15,
    )

    components["52W High Distance"] = numeric_similarity(
        current_features.get(
            "Distance from 52W High %"
        ),
        historical_event.get(
            "Distance from 52W High %"
        ),
        tolerance=4,
        full_points=15,
    )

    components["Volume"] = numeric_similarity(
        current_features.get(
            "Volume vs 20D Avg %"
        ),
        historical_event.get(
            "Volume vs 20D Avg %"
        ),
        tolerance=35,
        full_points=10,
    )

    components["ATR %"] = numeric_similarity(
        current_features.get("ATR %"),
        historical_event.get("ATR %"),
        tolerance=1.5,
        full_points=10,
    )

    components["Momentum 1M"] = numeric_similarity(
        current_features.get("Momentum 1M %"),
        historical_event.get("Momentum 1M %"),
        tolerance=6,
        full_points=10,
    )

    components["MA50 Distance"] = numeric_similarity(
        current_features.get(
            "Distance from MA50 %"
        ),
        historical_event.get(
            "Distance from MA50 %"
        ),
        tolerance=5,
        full_points=10,
    )

    total_score = sum(
        components.values()
    )

    return {
        "similarity_score": round(
            clip_score(total_score),
            1,
        ),
        "components": components,
    }


def add_similarity_scores(
    historical_events,
    current_features,
):
    """
    Add Phase 2 similarity scores to every historical event.
    """

    if historical_events is None or historical_events.empty:
        return pd.DataFrame()

    scored_events = historical_events.copy()

    similarity_scores = []
    daily_scores = []
    weekly_scores = []
    rs_scores = []
    high_distance_scores = []
    volume_scores = []
    atr_scores = []
    momentum_scores = []
    ma50_scores = []

    for _, event in scored_events.iterrows():
        similarity = calculate_similarity_score(
            current_features,
            event,
        )

        similarity_scores.append(
            similarity["similarity_score"]
        )

        daily_scores.append(
            similarity["components"]["Daily Trend"]
        )

        weekly_scores.append(
            similarity["components"]["Weekly Trend"]
        )

        rs_scores.append(
            similarity["components"]["RS 3M"]
        )

        high_distance_scores.append(
            similarity["components"][
                "52W High Distance"
            ]
        )

        volume_scores.append(
            similarity["components"]["Volume"]
        )

        atr_scores.append(
            similarity["components"]["ATR %"]
        )

        momentum_scores.append(
            similarity["components"]["Momentum 1M"]
        )

        ma50_scores.append(
            similarity["components"][
                "MA50 Distance"
            ]
        )

    scored_events["Similarity Score"] = (
        similarity_scores
    )

    scored_events["Similarity: Daily Trend"] = (
        daily_scores
    )

    scored_events["Similarity: Weekly Trend"] = (
        weekly_scores
    )

    scored_events["Similarity: RS 3M"] = rs_scores

    scored_events["Similarity: 52W Distance"] = (
        high_distance_scores
    )

    scored_events["Similarity: Volume"] = (
        volume_scores
    )

    scored_events["Similarity: ATR"] = atr_scores

    scored_events["Similarity: Momentum"] = (
        momentum_scores
    )

    scored_events["Similarity: MA50 Distance"] = (
        ma50_scores
    )

    return (
        scored_events
        .sort_values(
            by="Similarity Score",
            ascending=False,
        )
        .reset_index(drop=True)
    )


def summarise_similar_events(
    scored_events,
    minimum_similarity,
):
    """
    Build a Phase 2 summary using only sufficiently similar events.
    """

    if scored_events is None or scored_events.empty:
        return {
            "count": 0,
            "target_rate": None,
            "stop_rate": None,
            "expiry_rate": None,
            "median_return": None,
            "average_return": None,
            "median_mfe": None,
            "median_mae": None,
            "median_days_target": None,
            "confidence": "Insufficient sample",
        }

    similar_events = scored_events[
        scored_events["Similarity Score"]
        >= minimum_similarity
    ].copy()

    if similar_events.empty:
        return {
            "count": 0,
            "target_rate": None,
            "stop_rate": None,
            "expiry_rate": None,
            "median_return": None,
            "average_return": None,
            "median_mfe": None,
            "median_mae": None,
            "median_days_target": None,
            "confidence": "Insufficient sample",
        }

    non_ambiguous = similar_events[
        similar_events["Outcome"]
        != "Ambiguous Same Day"
    ].copy()

    if non_ambiguous.empty:
        target_rate = None
        stop_rate = None
        expiry_rate = None
    else:
        target_rate = (
            (
                non_ambiguous["Outcome"]
                == "Target Hit First"
            ).mean()
            * 100
        )

        stop_rate = (
            (
                non_ambiguous["Outcome"]
                == "Stop Hit First"
            ).mean()
            * 100
        )

        expiry_rate = (
            (
                non_ambiguous["Outcome"]
                == "Time Expired"
            ).mean()
            * 100
        )

    target_events = similar_events[
        similar_events["Outcome"]
        == "Target Hit First"
    ]

    median_days_target = (
        float(
            target_events["Days to Outcome"]
            .median()
        )
        if not target_events.empty
        else None
    )

    event_count = len(similar_events)

    if event_count >= 30:
        confidence = "Medium"

    elif event_count >= 15:
        confidence = "Low"

    else:
        confidence = "Insufficient sample"

    return {
        "count": event_count,
        "target_rate": target_rate,
        "stop_rate": stop_rate,
        "expiry_rate": expiry_rate,
        "median_return": float(
            similar_events["Forward Return %"].median()
        ),
        "average_return": float(
            similar_events["Forward Return %"].mean()
        ),
        "median_mfe": float(
            similar_events[
                "Maximum Favourable Excursion %"
            ].median()
        ),
        "median_mae": float(
            similar_events[
                "Maximum Adverse Excursion %"
            ].median()
        ),
        "median_days_target": median_days_target,
        "confidence": confidence,
    }


# =============================================================================
# CHART FUNCTIONS
# =============================================================================

def create_candlestick_chart(
    data,
    patterns,
    title,
):
    """Create candlestick, MA and volume chart."""

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
    """Create RS-line chart versus Nifty 50."""

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


def create_similarity_outcomes_chart(
    similar_events,
    target_percent,
    stop_percent,
):
    """Create chart of similarity score versus forward outcome."""

    figure = go.Figure()

    if similar_events is None or similar_events.empty:
        figure.update_layout(
            title="No similar historical events",
            height=350,
            template="plotly_white",
        )

        return figure

    colors = {
        "Target Hit First": "#16a34a",
        "Stop Hit First": "#dc2626",
        "Time Expired": "#f59e0b",
        "Ambiguous Same Day": "#6b7280",
    }

    for outcome in similar_events["Outcome"].unique():
        subset = similar_events[
            similar_events["Outcome"] == outcome
        ]

        figure.add_trace(
            go.Scatter(
                x=subset["Similarity Score"],
                y=subset["Forward Return %"],
                mode="markers",
                name=outcome,
                marker=dict(
                    size=11,
                    color=colors.get(
                        outcome,
                        "#2563eb",
                    ),
                ),
                hovertemplate=(
                    "Similarity: %{x:.1f}/100<br>"
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
        title="Similarity Score vs Historical Forward Outcome",
        height=420,
        template="plotly_white",
        xaxis_title="Similarity Score",
        yaxis_title="Forward Return %",
        legend_title="Outcome",
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
        🏆 Nifty Total Market Winner Research Dashboard
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="sub-title">
        Technical research • Relative Strength • Historical event study •
        Similarity scoring • Evidence-based probability research
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
        "Relative Strength and similarity calculations may be incomplete."
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
        "Historical studies cache: 12 hours\n\n"
        "Universe cache: 6 hours"
    )


# =============================================================================
# STOCK RESEARCH MODE
# =============================================================================

if dashboard_mode == "Stock research":
    symbols = sorted(
        stock_universe["Symbol"].tolist()
    )

    selected_symbol = st.selectbox(
        "Search a Nifty Total Market stock",
        symbols,
        index=(
            symbols.index("RELIANCE")
            if "RELIANCE" in symbols
            else 0
        ),
    )

    selected_record = stock_universe[
        stock_universe["Symbol"] == selected_symbol
    ].iloc[0]

    ticker = selected_record["Ticker"]

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
            "Price data is unavailable for this stock."
        )
        st.stop()

    daily_data = stock_data
    weekly_data = resample_ohlcv(stock_data, "Weekly")
    monthly_data = resample_ohlcv(stock_data, "Monthly")

    relative_strength = calculate_relative_strength(
        stock_data,
        nifty_50_data,
    )

    alignment = calculate_multitimeframe_alignment(
        daily_data,
        weekly_data,
        monthly_data,
    )

    current_features = calculate_current_setup_features(
        stock_data,
        nifty_50_data,
    )

    current_price = float(
        stock_data["close"].iloc[-1]
    )

    header_col1, header_col2, header_col3, header_col4 = (
        st.columns(4)
    )

    header_col1.metric(
        "Last Close",
        format_price(current_price),
    )

    header_col2.metric(
        "Market Cap",
        format_market_cap(
            fundamentals.get("marketCap")
        ),
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

    st.caption(
        f"{selected_record['Company Name']} • "
        f"{selected_record['Industry']}"
    )

    (
        technical_tab,
        relative_strength_tab,
        alignment_tab,
        historical_study_tab,
        similarity_tab,
        fundamentals_tab,
    ) = st.tabs(
        [
            "Technical Research",
            "Relative Strength",
            "MTF Alignment",
            "🎯 Phase 1 Historical Study",
            "🧩 Phase 2 Similarity Analysis",
            "Fundamentals",
        ]
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

        for tab, name, data in [
            (
                daily_tab,
                "Daily",
                daily_data,
            ),
            (
                weekly_tab,
                "Weekly",
                weekly_data,
            ),
            (
                monthly_tab,
                "Monthly",
                monthly_data,
            ),
        ]:
            with tab:
                patterns = detect_patterns(data)
                support, resistance = calculate_support_resistance(data)

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Trend",
                    calculate_overall_trend(data),
                )

                col2.metric(
                    "Support",
                    format_price(support),
                )

                col3.metric(
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
                        "No supported active pattern is currently detected."
                    )

                st.plotly_chart(
                    create_candlestick_chart(
                        data,
                        patterns,
                        f"{selected_symbol} — {name}",
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

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "RS Status",
            relative_strength["status"],
        )

        col2.metric(
            "RS Line Trend",
            relative_strength["rs_trend"],
        )

        col3.metric(
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
        )

        st.plotly_chart(
            create_relative_strength_chart(
                relative_strength["rs_line"],
                selected_symbol,
            ),
            use_container_width=True,
        )

    # =========================================================================
    # MULTI-TIMEFRAME ALIGNMENT TAB
    # =========================================================================

    with alignment_tab:
        st.subheader(
            "Multi-Timeframe Alignment"
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Daily Trend",
            alignment["daily"],
        )

        col2.metric(
            "Weekly Trend",
            alignment["weekly"],
        )

        col3.metric(
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
    # PHASE 1 HISTORICAL STUDY TAB
    # =========================================================================

    with historical_study_tab:
        st.subheader(
            "Phase 1: Historical Triple-Barrier Study"
        )

        st.caption(
            "Historical studies evaluate what happened after previous "
            "price-based bullish setups. They do not predict future returns."
        )

        col1, col2, col3 = st.columns(3)

        with col1:
            target_percent = st.selectbox(
                "Upside Target",
                TARGET_OPTIONS,
                index=1,
                format_func=lambda value: f"+{value}%",
                key="phase1_target",
            )

            stop_percent = st.selectbox(
                "Downside Stop",
                STOP_OPTIONS,
                index=1,
                format_func=lambda value: f"-{value}%",
                key="phase1_stop",
            )

        with col2:
            horizon_days = st.selectbox(
                "Horizon",
                HORIZON_OPTIONS,
                index=1,
                format_func=lambda value: (
                    f"{value} Trading Days"
                ),
                key="phase1_horizon",
            )

            setup_spacing = st.selectbox(
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
                key="phase1_spacing",
            )

        with col3:
            require_strong_daily = st.checkbox(
                "Require Strong Bullish Daily",
                value=False,
                key="phase1_strong_daily",
            )

            require_weekly_bullish = st.checkbox(
                "Require Bullish Weekly",
                value=True,
                key="phase1_weekly",
            )

            require_positive_rs = st.checkbox(
                "Require Positive 3M RS",
                value=True,
                key="phase1_rs",
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
            key="phase1_52w",
        )

        use_volume_filter = st.checkbox(
            "Require Volume Confirmation",
            value=False,
            key="phase1_volume_toggle",
        )

        if use_volume_filter:
            minimum_volume_change = st.selectbox(
                "Minimum Volume Above 20D Average",
                [
                    0,
                    10,
                    25,
                    50,
                ],
                index=1,
                format_func=lambda value: (
                    f"+{value}% Above Average"
                ),
                key="phase1_volume",
            )
        else:
            minimum_volume_change = None

        if st.button(
            "Run Phase 1 Historical Study",
            type="primary",
        ):
            with st.spinner(
                "Running historical triple-barrier analysis..."
            ):
                historical_events = run_historical_study(
                    stock_data=stock_data,
                    benchmark_data=nifty_50_data,
                    target_percent=target_percent,
                    stop_percent=stop_percent,
                    horizon_days=horizon_days,
                    setup_spacing_days=setup_spacing,
                    require_strong_daily=require_strong_daily,
                    require_weekly_bullish=require_weekly_bullish,
                    require_positive_rs=require_positive_rs,
                    max_distance_52w=maximum_distance_52w,
                    minimum_volume_change=minimum_volume_change,
                )

            st.session_state[
                "phase1_events"
            ] = historical_events

            st.session_state[
                "phase1_symbol"
            ] = selected_symbol

            st.session_state[
                "phase1_settings"
            ] = {
                "target": target_percent,
                "stop": stop_percent,
                "horizon": horizon_days,
                "spacing": setup_spacing,
                "strong_daily": require_strong_daily,
                "weekly_bullish": require_weekly_bullish,
                "positive_rs": require_positive_rs,
                "max_distance_52w": maximum_distance_52w,
                "minimum_volume": minimum_volume_change,
            }

        stored_phase1_events = st.session_state.get(
            "phase1_events"
        )

        stored_phase1_symbol = st.session_state.get(
            "phase1_symbol"
        )

        if (
            stored_phase1_events is not None
            and stored_phase1_symbol == selected_symbol
        ):
            summary = summarise_study(
                stored_phase1_events
            )

            st.divider()

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Historical Setups",
                summary["events"],
            )

            c2.metric(
                "Target Hit Rate",
                (
                    f"{summary['target_hit_rate']:.1f}%"
                    if summary["target_hit_rate"] is not None
                    else "Not available"
                ),
            )

            c3.metric(
                "Stop Hit Rate",
                (
                    f"{summary['stop_hit_rate']:.1f}%"
                    if summary["stop_hit_rate"] is not None
                    else "Not available"
                ),
            )

            c4.metric(
                "Confidence",
                summary["confidence"],
            )

            st.dataframe(
                stored_phase1_events,
                hide_index=True,
                use_container_width=True,
            )

    # =========================================================================
    # PHASE 2 SIMILARITY TAB
    # =========================================================================

    with similarity_tab:
        st.subheader(
            "Phase 2: Historical Setup Similarity Scoring"
        )

        st.caption(
            "This feature ranks past Phase 1 events by similarity to the "
            "current setup. It does not generate a guaranteed forecast."
        )

        st.markdown(
            """
            **Similarity components**

            | Component | Maximum Points |
            |---|---:|
            | Daily trend similarity | 15 |
            | Weekly trend similarity | 15 |
            | 3-month Relative Strength similarity | 15 |
            | Distance from 52-week high similarity | 15 |
            | Volume condition similarity | 10 |
            | ATR volatility similarity | 10 |
            | One-month momentum similarity | 10 |
            | Distance from 50 DMA similarity | 10 |
            | Total | 100 |
            """
        )

        current_features_dataframe = pd.DataFrame(
            [
                [
                    "Daily Trend",
                    current_features.get(
                        "Daily Trend"
                    ),
                ],
                [
                    "Weekly Trend",
                    current_features.get(
                        "Weekly Trend"
                    ),
                ],
                [
                    "Relative Strength, 3M",
                    current_features.get(
                        "RS 3M %"
                    ),
                ],
                [
                    "Distance from 52W High",
                    current_features.get(
                        "Distance from 52W High %"
                    ),
                ],
                [
                    "Volume vs 20D Average",
                    current_features.get(
                        "Volume vs 20D Avg %"
                    ),
                ],
                [
                    "ATR %",
                    current_features.get(
                        "ATR %"
                    ),
                ],
                [
                    "1M Momentum",
                    current_features.get(
                        "Momentum 1M %"
                    ),
                ],
                [
                    "Distance from 50 DMA",
                    current_features.get(
                        "Distance from MA50 %"
                    ),
                ],
            ],
            columns=[
                "Current Feature",
                "Value",
            ],
        )

        st.markdown("### Current Setup Features")

        st.dataframe(
            current_features_dataframe,
            hide_index=True,
            use_container_width=True,
        )

        stored_events = st.session_state.get(
            "phase1_events"
        )

        stored_symbol = st.session_state.get(
            "phase1_symbol"
        )

        stored_settings = st.session_state.get(
            "phase1_settings"
        )

        if (
            stored_events is None
            or stored_events.empty
            or stored_symbol != selected_symbol
        ):
            st.info(
                "Run Phase 1 Historical Study first. Phase 2 uses its "
                "historical event set as the similarity comparison library."
            )

        else:
            min_similarity_col, top_cases_col = st.columns(2)

            with min_similarity_col:
                minimum_similarity = st.slider(
                    "Minimum Similarity Score",
                    min_value=40,
                    max_value=95,
                    value=70,
                    step=5,
                )

            with top_cases_col:
                top_case_count = st.selectbox(
                    "Display Top Similar Cases",
                    [
                        5,
                        10,
                        15,
                        20,
                    ],
                    index=1,
                )

            if st.button(
                "Calculate Similar Historical Setups",
                type="primary",
            ):
                scored_events = add_similarity_scores(
                    historical_events=stored_events,
                    current_features=current_features,
                )

                st.session_state[
                    "phase2_scored_events"
                ] = scored_events

                st.session_state[
                    "phase2_symbol"
                ] = selected_symbol

                st.session_state[
                    "phase2_min_similarity"
                ] = minimum_similarity

            scored_events = st.session_state.get(
                "phase2_scored_events"
            )

            phase2_symbol = st.session_state.get(
                "phase2_symbol"
            )

            if (
                scored_events is not None
                and not scored_events.empty
                and phase2_symbol == selected_symbol
            ):
                similar_summary = summarise_similar_events(
                    scored_events,
                    minimum_similarity,
                )

                highly_similar_events = scored_events[
                    scored_events["Similarity Score"]
                    >= minimum_similarity
                ].copy()

                st.divider()

                st.subheader(
                    "Similarity-Based Historical Evidence"
                )

                result_col1, result_col2, result_col3, result_col4 = (
                    st.columns(4)
                )

                result_col1.metric(
                    "Similar Historical Cases",
                    similar_summary["count"],
                )

                result_col2.metric(
                    "Target Hit First",
                    (
                        f"{similar_summary['target_rate']:.1f}%"
                        if similar_summary[
                            "target_rate"
                        ] is not None
                        else "Not available"
                    ),
                )

                result_col3.metric(
                    "Stop Hit First",
                    (
                        f"{similar_summary['stop_rate']:.1f}%"
                        if similar_summary[
                            "stop_rate"
                        ] is not None
                        else "Not available"
                    ),
                )

                result_col4.metric(
                    "Similarity Confidence",
                    similar_summary["confidence"],
                )

                return_col1, return_col2, return_col3, return_col4 = (
                    st.columns(4)
                )

                return_col1.metric(
                    "Median Forward Return",
                    format_percent(
                        similar_summary[
                            "median_return"
                        ]
                    ),
                )

                return_col2.metric(
                    "Average Forward Return",
                    format_percent(
                        similar_summary[
                            "average_return"
                        ]
                    ),
                )

                return_col3.metric(
                    "Median Max Gain",
                    format_percent(
                        similar_summary[
                            "median_mfe"
                        ]
                    ),
                )

                return_col4.metric(
                    "Median Max Drawdown",
                    format_percent(
                        similar_summary[
                            "median_mae"
                        ]
                    ),
                )

                st.caption(
                    "The hit rates apply only to Phase 1 events whose "
                    f"similarity score is at least {minimum_similarity}/100."
                )

                if similar_summary["count"] < 10:
                    st.warning(
                        "Fewer than 10 sufficiently similar historical cases "
                        "were found. Treat the result as exploratory."
                    )

                st.plotly_chart(
                    create_similarity_outcomes_chart(
                        highly_similar_events,
                        stored_settings["target"],
                        stored_settings["stop"],
                    ),
                    use_container_width=True,
                )

                st.subheader(
                    "Most Similar Historical Cases"
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
                    "Days to Outcome",
                    "Forward Return %",
                    "Maximum Favourable Excursion %",
                    "Maximum Adverse Excursion %",
                ]

                top_cases = scored_events[
                    display_columns
                ].head(top_case_count)

                st.dataframe(
                    top_cases,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Similarity Score": st.column_config.ProgressColumn(
                            "Similarity",
                            min_value=0,
                            max_value=100,
                            format="%.1f",
                        ),
                        "Entry Date": st.column_config.DatetimeColumn(
                            "Historical Entry Date",
                            format="YYYY-MM-DD",
                        ),
                        "Entry Price": st.column_config.NumberColumn(
                            "Entry Price",
                            format="₹%.2f",
                        ),
                        "RS 3M %": st.column_config.NumberColumn(
                            "RS 3M",
                            format="%.2f%%",
                        ),
                        "Distance from 52W High %": st.column_config.NumberColumn(
                            "52W High Distance",
                            format="%.2f%%",
                        ),
                        "Volume vs 20D Avg %": st.column_config.NumberColumn(
                            "Volume vs Average",
                            format="%.2f%%",
                        ),
                        "ATR %": st.column_config.NumberColumn(
                            "ATR %",
                            format="%.2f%%",
                        ),
                        "Momentum 1M %": st.column_config.NumberColumn(
                            "1M Momentum",
                            format="%.2f%%",
                        ),
                        "Distance from MA50 %": st.column_config.NumberColumn(
                            "Distance from MA50",
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

                st.subheader(
                    "Similarity Score Explanation"
                )

                top_similarity_components = [
                    "Similarity: Daily Trend",
                    "Similarity: Weekly Trend",
                    "Similarity: RS 3M",
                    "Similarity: 52W Distance",
                    "Similarity: Volume",
                    "Similarity: ATR",
                    "Similarity: Momentum",
                    "Similarity: MA50 Distance",
                ]

                score_explanation = scored_events[
                    [
                        "Similarity Score",
                        "Entry Date",
                    ]
                    + top_similarity_components
                ].head(1)

                st.caption(
                    "The row below explains how the highest-ranked "
                    "historical analogue received its similarity score."
                )

                st.dataframe(
                    score_explanation,
                    hide_index=True,
                    use_container_width=True,
                )

                similarity_csv = scored_events.to_csv(
                    index=False
                ).encode("utf-8")

                st.download_button(
                    "⬇️ Download Similarity Study CSV",
                    data=similarity_csv,
                    file_name=(
                        f"{selected_symbol.lower()}_"
                        f"phase2_similarity_study.csv"
                    ),
                    mime="text/csv",
                )

            else:
                st.info(
                    "Set a minimum similarity score and click "
                    "Calculate Similar Historical Setups."
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
                        fundamentals.get("trailingPE")
                    ),
                ],
                [
                    "Forward P/E",
                    safe_number(
                        fundamentals.get("forwardPE")
                    ),
                ],
                [
                    "Price / Book",
                    safe_number(
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
                    safe_number(
                        fundamentals.get("debtToEquity")
                    ),
                ],
                [
                    "Current Ratio",
                    safe_number(
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
            "Phase 2 similarity currently uses point-in-time price, trend, "
            "volume, relative strength and volatility data. Fundamental "
            "features should be added only after a point-in-time annual and "
            "quarterly filing database is implemented."
        )


# =============================================================================
# WINNER RANKING SCANNER
# =============================================================================

else:
    st.subheader(
        "Winner Ranking Scanner"
    )

    st.info(
        "Phase 2 is currently designed for one selected stock at a time "
        "inside Stock Research. Cross-sectional similarity across Nifty 750 "
        "will be part of the later Phase 3/4 database workflow."
    )

    st.caption(
        "Use Stock Research → Phase 1 Historical Study → "
        "Phase 2 Similarity Analysis."
    )


# =============================================================================
# FOOTER
# =============================================================================

st.divider()

st.caption(
    "Data sources: Nifty Indices constituent list and Yahoo Finance. "
    "Phase 1 and Phase 2 use historical OHLCV data and rule-based setup "
    "definitions. Similarity scores, target-hit rates, and forward outcomes "
    "are research statistics, not predictions or investment advice. "
    "Historical behaviour may not repeat, and daily OHLC data cannot always "
    "identify intraday barrier order when both target and stop are reached "
    "on the same day."
)
