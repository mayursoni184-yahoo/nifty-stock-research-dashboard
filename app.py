import io
import requests
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Nifty Total Market Research Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# STYLING
# =============================================================================

st.markdown(
    """
    <style>
        .title-main {
            font-size: 2.3rem;
            font-weight: 800;
            color: #111827;
            margin-bottom: 0.2rem;
        }

        .title-sub {
            font-size: 1rem;
            color: #6b7280;
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
            border-left: 5px solid #16a34a;
        }

        .orange-card {
            border-left: 5px solid #f59e0b;
        }

        .red-card {
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

ALIGNMENT_OPTIONS = [
    "Any",
    "Strong multi-timeframe alignment",
    "Bullish multi-timeframe alignment",
    "Mixed timeframe alignment",
    "Bearish multi-timeframe alignment",
    "Insufficient data",
]


# =============================================================================
# GENERAL UTILITY FUNCTIONS
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


def format_percentage(value):
    """Formats a percentage safely."""

    value = safe_number(value)

    if value is None:
        return "Not available"

    return f"{value:+.2f}%"


def format_price(value):
    """Formats an INR price safely."""

    value = safe_number(value)

    if value is None:
        return "Not available"

    return f"₹{value:,.2f}"


def format_market_cap(value):
    """Formats market capitalization in crore rupees."""

    value = safe_number(value)

    if value is None or value <= 0:
        return "Not available"

    market_cap_crore = value / 10000000

    return f"₹{market_cap_crore:,.0f} Cr"


# =============================================================================
# UNIVERSE DATA
# =============================================================================

@st.cache_data(ttl=21600, show_spinner=False)
def get_nifty_total_market_members():
    """
    Loads official Nifty Total Market constituents.

    Uses a fallback stock list if the external Nifty Indices file is
    unavailable or does not respond in 15 seconds.
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
                "The constituent file has no Symbol column."
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
                "Too few Nifty Total Market constituents received."
            )

        return members, None

    except Exception as error:
        fallback_members = pd.DataFrame(
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
            "Could not retrieve the live Nifty Total Market constituent list. "
            f"Using fallback stocks. Reason: {type(error).__name__}: {error}"
        )

        return fallback_members, warning


# =============================================================================
# PRICE AND FUNDAMENTAL DATA
# =============================================================================

@st.cache_data(ttl=900, show_spinner=False)
def fetch_price_data(ticker, period="5y"):
    """Fetches daily OHLCV data from Yahoo Finance."""

    try:
        stock = yf.Ticker(ticker)

        data = stock.history(
            period=period,
            auto_adjust=True,
        )

        if data is None or data.empty:
            return pd.DataFrame()

        data = data.rename(columns=str.lower)

        columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        data = data[columns].dropna()

        data.index = pd.to_datetime(data.index)

        if getattr(data.index, "tz", None) is not None:
            data.index = data.index.tz_localize(None)

        return data

    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=43200, show_spinner=False)
def fetch_fundamentals(ticker):
    """Fetches basic available Yahoo Finance company information."""

    required_fields = [
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
        "dividendYield",
        "fiftyTwoWeekHigh",
        "fiftyTwoWeekLow",
    ]

    try:
        stock = yf.Ticker(ticker)
        stock_info = stock.get_info()

        return {
            field: stock_info.get(field)
            for field in required_fields
        }

    except Exception:
        return {}


# =============================================================================
# RESAMPLING AND TREND ANALYSIS
# =============================================================================

def resample_ohlcv(data, timeframe):
    """Resamples daily OHLCV data into weekly or monthly candles."""

    if data.empty:
        return data

    if timeframe == "Daily":
        return data.copy()

    if timeframe == "Weekly":
        rule = "W-FRI"
    else:
        rule = "ME"

    return (
        data
        .resample(rule)
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
    Calculates trend using 20, 50 and 200 moving averages.

    Strong bullish:
    Price > 20 MA > 50 MA > 200 MA

    Bullish:
    Price > 20 MA > 50 MA

    Bearish:
    Price < 20 MA < 50 MA
    """

    if data is None or len(data) < 55:
        return "Insufficient data"

    close = float(data["close"].iloc[-1])

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
            close > moving_average_20
            > moving_average_50
            > moving_average_200
        ):
            return "Strong bullish"

    if (
        close > moving_average_20
        and moving_average_20 > moving_average_50
    ):
        return "Bullish"

    if (
        close < moving_average_20
        and moving_average_20 < moving_average_50
    ):
        return "Bearish"

    return "Neutral / consolidating"


# =============================================================================
# RELATIVE STRENGTH ENGINE
# =============================================================================

def calculate_return(data, trading_days):
    """
    Calculates stock return for a selected lookback.

    Approximate lookbacks:
    21 trading days = 1 month
    63 trading days = 3 months
    126 trading days = 6 months
    252 trading days = 12 months
    """

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
    """
    Aligns stock and benchmark close prices by common trading dates.
    """

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
    Calculates returns, relative returns and an RS status.

    Relative Return = Stock Return - Benchmark Return.

    RS line = Stock Close / Benchmark Close.
    """

    aligned_data = align_stock_and_benchmark(
        stock_data,
        benchmark_data,
    )

    if len(aligned_data) < 30:
        return {
            "status": "Insufficient data",
            "rs_line": pd.Series(dtype=float),
            "rs_trend": "Unavailable",
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

    stock_1m = calculate_return(
        stock_prices,
        21,
    )

    stock_3m = calculate_return(
        stock_prices,
        63,
    )

    stock_6m = calculate_return(
        stock_prices,
        126,
    )

    stock_12m = calculate_return(
        stock_prices,
        252,
    )

    benchmark_1m = calculate_return(
        benchmark_prices,
        21,
    )

    benchmark_3m = calculate_return(
        benchmark_prices,
        63,
    )

    benchmark_6m = calculate_return(
        benchmark_prices,
        126,
    )

    benchmark_12m = calculate_return(
        benchmark_prices,
        252,
    )

    relative_1m = None
    relative_3m = None
    relative_6m = None
    relative_12m = None

    if stock_1m is not None and benchmark_1m is not None:
        relative_1m = stock_1m - benchmark_1m

    if stock_3m is not None and benchmark_3m is not None:
        relative_3m = stock_3m - benchmark_3m

    if stock_6m is not None and benchmark_6m is not None:
        relative_6m = stock_6m - benchmark_6m

    if stock_12m is not None and benchmark_12m is not None:
        relative_12m = stock_12m - benchmark_12m

    rs_line = (
        aligned_data["stock_close"]
        / aligned_data["benchmark_close"]
    )

    rs_trend = "Unavailable"

    if len(rs_line) >= 63:
        rs_now = float(rs_line.iloc[-1])
        rs_3_months_ago = float(rs_line.iloc[-63])

        if rs_now > rs_3_months_ago * 1.03:
            rs_trend = "Rising"

        elif rs_now < rs_3_months_ago * 0.97:
            rs_trend = "Falling"

        else:
            rs_trend = "Flat"

    available_relative_returns = [
        value
        for value in [
            relative_1m,
            relative_3m,
            relative_6m,
            relative_12m,
        ]
        if value is not None
    ]

    if len(available_relative_returns) < 2:
        rs_status = "Insufficient data"

    else:
        positive_count = sum(
            value > 0
            for value in available_relative_returns
        )

        strong_count = sum(
            value > 5
            for value in available_relative_returns
        )

        weak_count = sum(
            value < 0
            for value in available_relative_returns
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

        elif weak_count >= 3:
            rs_status = "Weak"

        else:
            rs_status = "Neutral"

    return {
        "status": rs_status,
        "rs_line": rs_line,
        "rs_trend": rs_trend,
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


def create_relative_strength_chart(
    rs_line,
    symbol,
):
    """Creates an RS-line chart versus Nifty 50."""

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
            f"{symbol} Relative Strength Line "
            "vs Nifty 50"
        ),
        height=350,
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
# MULTI-TIMEFRAME ALIGNMENT ENGINE
# =============================================================================

def trend_points(trend):
    """Converts a trend classification into alignment points."""

    if trend == "Strong bullish":
        return 4

    if trend == "Bullish":
        return 3

    if trend == "Neutral / consolidating":
        return 1

    if trend == "Bearish":
        return 0

    return 0


def calculate_multitimeframe_alignment(
    daily_data,
    weekly_data,
    monthly_data,
):
    """
    Calculates Daily / Weekly / Monthly trend alignment.

    Maximum score: 10
    Daily:   maximum 3 points
    Weekly:  maximum 4 points
    Monthly: maximum 3 points
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
            "structure": "Insufficient chart history",
        }

    score = 0

    if daily_trend == "Strong bullish":
        score += 3

    elif daily_trend == "Bullish":
        score += 2

    elif daily_trend == "Neutral / consolidating":
        score += 1

    if weekly_trend == "Strong bullish":
        score += 4

    elif weekly_trend == "Bullish":
        score += 3

    elif weekly_trend == "Neutral / consolidating":
        score += 1

    if monthly_trend == "Strong bullish":
        score += 3

    elif monthly_trend == "Bullish":
        score += 2

    elif monthly_trend == "Neutral / consolidating":
        score += 1

    bullish_trends = sum(
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

    bearish_trends = sum(
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
        alignment_status = (
            "Strong multi-timeframe alignment"
        )

        market_structure = (
            "Daily, Weekly and Monthly trends are aligned upward."
        )

    elif bullish_trends >= 2:
        alignment_status = (
            "Bullish multi-timeframe alignment"
        )

        market_structure = (
            "Most major chart timeframes are bullish."
        )

    elif bearish_trends >= 2:
        alignment_status = (
            "Bearish multi-timeframe alignment"
        )

        market_structure = (
            "Most major chart timeframes are bearish."
        )

    else:
        alignment_status = (
            "Mixed timeframe alignment"
        )

        market_structure = (
            "Timeframes disagree; avoid treating this as a high-conviction trend."
        )

    return {
        "daily": daily_trend,
        "weekly": weekly_trend,
        "monthly": monthly_trend,
        "score": score,
        "status": alignment_status,
        "structure": market_structure,
    }


# =============================================================================
# TECHNICAL PATTERN FUNCTIONS
# =============================================================================

def find_pivots(values, pivot_type="high", order=3):
    """Finds local high or low pivot indexes."""

    values = np.asarray(values, dtype=float)

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
):
    """Creates a standardized technical signal."""

    latest_price = float(data["close"].iloc[-1])

    previous_volumes = (
        data["volume"]
        .tail(21)
        .iloc[:-1]
    )

    average_volume = float(
        previous_volumes.mean()
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
        "Current": latest_price,
        "Return %": (
            latest_price / float(level) - 1
        ) * 100,
        "Volume %": volume_change,
        "Notes": notes,
    }


def detect_patterns(data):
    """
    Detects selected rule-based price patterns.

    These patterns are educational signals, not trade recommendations.
    """

    if data is None or data.empty:
        return []

    data = data.dropna().copy()

    if len(data) < 40:
        return []

    high = data["high"].to_numpy(dtype=float)
    low = data["low"].to_numpy(dtype=float)
    close = data["close"].to_numpy(dtype=float)
    open_price = data["open"].to_numpy(dtype=float)

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
                        "Two similar peaks. Confirmation requires "
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

            results.append(
                build_pattern_signal(
                    "Double Bottom",
                    status,
                    data,
                    neckline,
                    "Bullish",
                    (
                        "Two similar troughs. Confirmation requires "
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

            results.append(
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

            results.append(
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
    window_size = 30

    if len(data) >= window_size:
        recent_high = high[-window_size:]
        recent_low = low[-window_size:]

        x_axis = np.arange(window_size)

        high_slope = (
            np.polyfit(
                x_axis,
                recent_high,
                1,
            )[0]
            / max(np.mean(recent_high), 1)
        )

        low_slope = (
            np.polyfit(
                x_axis,
                recent_low,
                1,
            )[0]
            / max(np.mean(recent_low), 1)
        )

        resistance = float(
            np.max(recent_high)
        )

        support = float(
            np.min(recent_low)
        )

        range_size = (
            resistance - support
        ) / max(resistance, 1)

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
                            "Breakout confirmation requires a close outside "
                            "the pattern range."
                        ),
                    )
                )

    # REVERSAL BOTTOM
    latest_open = float(open_price[-1])
    latest_close = float(close[-1])
    latest_high = float(high[-1])
    latest_low = float(low[-1])

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

    recent_low = float(
        np.min(low[-11:-1])
    )

    recent_high = float(
        np.max(high[-11:-1])
    )

    if (
        lower_shadow / candle_range > 0.55
        and candle_body / candle_range < 0.35
        and latest_close <= recent_low * 1.04
    ):
        results.append(
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
            )
        )

    # REVERSAL TOP
    if (
        upper_shadow / candle_range > 0.55
        and candle_body / candle_range < 0.35
        and latest_close >= recent_high * 0.96
    ):
        results.append(
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
            )
        )

    return results


def calculate_support_resistance(data):
    """Calculates simple pivot-based support and resistance."""

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
    """Creates the price, moving-average and volume chart."""

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
        signal_color = "#16a34a"

        if signal["Direction"] == "Bearish":
            signal_color = "#dc2626"

        elif signal["Direction"] == "Neutral":
            signal_color = "#f59e0b"

        figure.add_hline(
            y=signal["Level"],
            line_dash="dot",
            line_color=signal_color,
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


# =============================================================================
# LOAD MARKET UNIVERSE AND BENCHMARK
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
    nifty_50_prices = fetch_price_data(
        NIFTY_50_BENCHMARK,
        "5y",
    )


# =============================================================================
# HEADER
# =============================================================================

st.markdown(
    """
    <div class="title-main">
        📊 Nifty Total Market (750) Research Dashboard
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="title-sub">
        Technical patterns • Fundamentals • Relative Strength •
        Multi-timeframe alignment • Support and Resistance
    </div>
    """,
    unsafe_allow_html=True,
)

if universe_error:
    st.warning(universe_error)

if stock_universe.empty:
    st.error(
        "No stock universe is available. "
        "Please refresh the cached data."
    )
    st.stop()

if nifty_50_prices.empty:
    st.warning(
        "Nifty 50 benchmark data is currently unavailable. "
        "Relative-strength calculations will show insufficient data."
    )


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.header("🔍 Research Controls")

    dashboard_mode = st.radio(
        "Dashboard Mode",
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
        f"Loaded universe: {len(stock_universe)} stocks"
    )

    if st.button("🔄 Refresh cached data"):
        st.cache_data.clear()
        st.rerun()

    st.divider()

    st.caption(
        "Price cache: 15 minutes\n\n"
        "Fundamental cache: 12 hours\n\n"
        "Constituent-list cache: 6 hours"
    )


# =============================================================================
# STOCK RESEARCH MODE
# =============================================================================

if dashboard_mode == "Stock research":
    symbols = sorted(
        stock_universe["Symbol"].tolist()
    )

    default_symbol_index = 0

    if "RELIANCE" in symbols:
        default_symbol_index = symbols.index(
            "RELIANCE"
        )

    selected_symbol = st.selectbox(
        "Search a Nifty Total Market stock",
        symbols,
        index=default_symbol_index,
    )

    selected_stock = stock_universe[
        stock_universe["Symbol"] == selected_symbol
    ].iloc[0]

    selected_ticker = selected_stock["Ticker"]
    selected_company = selected_stock["Company Name"]
    selected_industry = selected_stock["Industry"]

    with st.spinner(
        f"Downloading research data for {selected_symbol}..."
    ):
        stock_prices = fetch_price_data(
            selected_ticker,
            "5y",
        )

        stock_fundamentals = fetch_fundamentals(
            selected_ticker
        )

    if stock_prices.empty:
        st.error(
            "No price data is currently available for this stock."
        )
        st.stop()

    daily_prices = stock_prices

    weekly_prices = resample_ohlcv(
        stock_prices,
        "Weekly",
    )

    monthly_prices = resample_ohlcv(
        stock_prices,
        "Monthly",
    )

    relative_strength = calculate_relative_strength(
        stock_prices,
        nifty_50_prices,
    )

    timeframe_alignment = (
        calculate_multitimeframe_alignment(
            daily_prices,
            weekly_prices,
            monthly_prices,
        )
    )

    current_price = float(
        stock_prices["close"].iloc[-1]
    )

    market_cap = stock_fundamentals.get(
        "marketCap"
    )

    market_cap_crore = (
        safe_number(market_cap, 0)
        / 10000000
    )

    top_col1, top_col2, top_col3, top_col4 = (
        st.columns(4)
    )

    top_col1.metric(
        "Last Close",
        format_price(current_price),
    )

    top_col2.metric(
        "Market Cap",
        format_market_cap(market_cap),
    )

    top_col3.metric(
        "Relative Strength",
        relative_strength["status"],
    )

    top_col4.metric(
        "MTF Alignment",
        (
            f"{timeframe_alignment['score']}/10"
            if timeframe_alignment["score"] is not None
            else "Not available"
        ),
    )

    st.caption(
        f"{selected_company} • {selected_industry}"
    )

    if (
        market_cap_crore > 0
        and market_cap_crore < minimum_market_cap
    ):
        st.warning(
            f"{selected_symbol} is below your "
            f"₹{minimum_market_cap:,.0f} crore market-cap filter."
        )

    (
        technical_tab,
        relative_strength_tab,
        alignment_tab,
        fundamentals_tab,
    ) = st.tabs(
        [
            "Technical Research",
            "Relative Strength",
            "Multi-Timeframe Alignment",
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

        for tab, timeframe_name, timeframe_data in [
            (
                daily_tab,
                "Daily",
                daily_prices,
            ),
            (
                weekly_tab,
                "Weekly",
                weekly_prices,
            ),
            (
                monthly_tab,
                "Monthly",
                monthly_prices,
            ),
        ]:
            with tab:
                signals = detect_patterns(
                    timeframe_data
                )

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

                if signals:
                    signal_dataframe = pd.DataFrame(
                        signals
                    )

                    st.dataframe(
                        signal_dataframe,
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
                        "is detected on this timeframe."
                    )

                st.plotly_chart(
                    create_candlestick_chart(
                        timeframe_data,
                        signals,
                        f"{selected_symbol} — {timeframe_name}",
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
            "Relative Return = Stock Return − Nifty 50 Return. "
            "Positive relative return means the stock outperformed "
            "Nifty 50 over the same period."
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

        else:
            st.info(
                "Insufficient common price history is available to "
                "calculate the Relative Strength line."
            )

        st.markdown("### Relative Strength Classification")

        st.markdown(
            """
            | RS Status | Meaning |
            |---|---|
            | Leader | Outperforming Nifty 50 across most periods with a rising RS line |
            | Strong | Positive performance versus Nifty 50 on multiple timeframes |
            | Neutral | Mixed outperformance and underperformance |
            | Weak | Underperforming Nifty 50 across most timeframes |
            | Insufficient data | There is not enough aligned historical data |
            """
        )

    # =========================================================================
    # MULTI-TIMEFRAME ALIGNMENT TAB
    # =========================================================================

    with alignment_tab:
        st.subheader(
            "Daily, Weekly and Monthly Trend Alignment"
        )

        alignment_col1, alignment_col2, alignment_col3 = (
            st.columns(3)
        )

        alignment_col1.metric(
            "Daily Trend",
            timeframe_alignment["daily"],
        )

        alignment_col2.metric(
            "Weekly Trend",
            timeframe_alignment["weekly"],
        )

        alignment_col3.metric(
            "Monthly Trend",
            timeframe_alignment["monthly"],
        )

        alignment_score = timeframe_alignment["score"]

        if alignment_score is None:
            alignment_score_text = "Not available"
        else:
            alignment_score_text = (
                f"{alignment_score}/10"
            )

        alignment_summary_col1, alignment_summary_col2 = (
            st.columns(2)
        )

        alignment_summary_col1.metric(
            "Alignment Score",
            alignment_score_text,
        )

        alignment_summary_col2.metric(
            "Alignment Status",
            timeframe_alignment["status"],
        )

        card_class = "orange-card"

        if (
            timeframe_alignment["status"]
            == "Strong multi-timeframe alignment"
        ):
            card_class = "green-card"

        elif (
            timeframe_alignment["status"]
            == "Bearish multi-timeframe alignment"
        ):
            card_class = "red-card"

        st.markdown(
            f"""
            <div class="research-card {card_class}">
                <h4>Market Structure</h4>
                <p>{timeframe_alignment["structure"]}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            ### Alignment Score Rules

            | Timeframe | Strong Bullish | Bullish | Neutral | Bearish |
            |---|---:|---:|---:|---:|
            | Daily | 3 | 2 | 1 | 0 |
            | Weekly | 4 | 3 | 1 | 0 |
            | Monthly | 3 | 2 | 1 | 0 |

            Maximum alignment score: 10.
            """
        )

    # =========================================================================
    # FUNDAMENTALS TAB
    # =========================================================================

    with fundamentals_tab:
        st.subheader("Basic Fundamental Metrics")

        free_cashflow = stock_fundamentals.get(
            "freeCashflow"
        )

        if isinstance(free_cashflow, (int, float)):
            free_cashflow_text = (
                f"₹{free_cashflow / 10000000:,.0f} Cr"
            )
        else:
            free_cashflow_text = "Not available"

        roe_value = safe_number(
            stock_fundamentals.get(
                "returnOnEquity"
            )
        )

        roa_value = safe_number(
            stock_fundamentals.get(
                "returnOnAssets"
            )
        )

        profit_margin_value = safe_number(
            stock_fundamentals.get(
                "profitMargins"
            )
        )

        operating_margin_value = safe_number(
            stock_fundamentals.get(
                "operatingMargins"
            )
        )

        revenue_growth_value = safe_number(
            stock_fundamentals.get(
                "revenueGrowth"
            )
        )

        earnings_growth_value = safe_number(
            stock_fundamentals.get(
                "earningsGrowth"
            )
        )

        dividend_yield_value = safe_number(
            stock_fundamentals.get(
                "dividendYield"
            )
        )

        fundamentals_dataframe = pd.DataFrame(
            [
                [
                    "Sector",
                    stock_fundamentals.get(
                        "sector",
                        "Not available",
                    ),
                ],
                [
                    "Industry",
                    stock_fundamentals.get(
                        "industry",
                        "Not available",
                    ),
                ],
                [
                    "Trailing P/E",
                    stock_fundamentals.get(
                        "trailingPE",
                        "Not available",
                    ),
                ],
                [
                    "Forward P/E",
                    stock_fundamentals.get(
                        "forwardPE",
                        "Not available",
                    ),
                ],
                [
                    "Price / Book",
                    stock_fundamentals.get(
                        "priceToBook",
                        "Not available",
                    ),
                ],
                [
                    "ROE",
                    (
                        f"{roe_value * 100:.2f}%"
                        if roe_value is not None
                        else "Not available"
                    ),
                ],
                [
                    "ROA",
                    (
                        f"{roa_value * 100:.2f}%"
                        if roa_value is not None
                        else "Not available"
                    ),
                ],
                [
                    "Profit Margin",
                    (
                        f"{profit_margin_value * 100:.2f}%"
                        if profit_margin_value is not None
                        else "Not available"
                    ),
                ],
                [
                    "Operating Margin",
                    (
                        f"{operating_margin_value * 100:.2f}%"
                        if operating_margin_value is not None
                        else "Not available"
                    ),
                ],
                [
                    "Revenue Growth",
                    (
                        f"{revenue_growth_value * 100:.2f}%"
                        if revenue_growth_value is not None
                        else "Not available"
                    ),
                ],
                [
                    "Earnings Growth",
                    (
                        f"{earnings_growth_value * 100:.2f}%"
                        if earnings_growth_value is not None
                        else "Not available"
                    ),
                ],
                [
                    "Debt / Equity",
                    stock_fundamentals.get(
                        "debtToEquity",
                        "Not available",
                    ),
                ],
                [
                    "Current Ratio",
                    stock_fundamentals.get(
                        "currentRatio",
                        "Not available",
                    ),
                ],
                [
                    "Free Cash Flow",
                    free_cashflow_text,
                ],
                [
                    "Dividend Yield",
                    (
                        f"{dividend_yield_value * 100:.2f}%"
                        if dividend_yield_value is not None
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
            "This is the basic fundamental layer. The next stage can add "
            "annual/quarterly earnings acceleration, ownership trends, "
            "sector strength, risk/reward, and the composite Winner Score."
        )


# =============================================================================
# PATTERN SCANNER MODE
# =============================================================================

else:
    st.subheader(
        "Nifty Total Market Pattern Scanner"
    )

    st.caption(
        "Filter the Nifty Total Market universe by price pattern, "
        "timeframe, overall trend, relative strength and multi-timeframe alignment."
    )

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
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

    with filter_col2:
        scanner_trend = st.selectbox(
            "Overall Trend",
            TREND_OPTIONS,
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

    with filter_col3:
        scanner_rs_status = st.selectbox(
            "Relative Strength Status",
            RS_STATUS_OPTIONS,
        )

        scanner_alignment = st.selectbox(
            "MTF Alignment",
            ALIGNMENT_OPTIONS,
        )

    st.info(
        "Timeframe = Any scans Daily, Weekly and Monthly signals. "
        "Relative Strength compares each stock with Nifty 50. "
        "MTF Alignment combines Daily, Weekly and Monthly trend status."
    )

    st.warning(
        "A full 750-stock scan may take several minutes on Streamlit Cloud. "
        "For faster scans, use one timeframe instead of Any and use "
        "specific pattern/trend filters."
    )

    if st.button(
        "🔎 Scan Nifty Total Market",
        type="primary",
    ):
        results = []

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

        progress_bar = st.progress(0)
        progress_label = st.empty()

        total_stocks = len(stock_universe)

        for index, record in stock_universe.iterrows():
            symbol = record["Symbol"]
            ticker = record["Ticker"]

            progress_label.caption(
                f"Scanning {index + 1:,} of {total_stocks:,}: {symbol}"
            )

            try:
                stock_prices = fetch_price_data(
                    ticker,
                    "5y",
                )

                if stock_prices.empty:
                    continue

                daily_data = stock_prices
                weekly_data = resample_ohlcv(
                    stock_prices,
                    "Weekly",
                )

                monthly_data = resample_ohlcv(
                    stock_prices,
                    "Monthly",
                )

                stock_rs = calculate_relative_strength(
                    stock_prices,
                    nifty_50_prices,
                )

                stock_alignment = (
                    calculate_multitimeframe_alignment(
                        daily_data,
                        weekly_data,
                        monthly_data,
                    )
                )

                if (
                    scanner_rs_status != "Any"
                    and stock_rs["status"] != scanner_rs_status
                ):
                    continue

                if (
                    scanner_alignment != "Any"
                    and stock_alignment["status"]
                    != scanner_alignment
                ):
                    continue

                matching_signals = []

                timeframe_map = {
                    "Daily": daily_data,
                    "Weekly": weekly_data,
                    "Monthly": monthly_data,
                }

                for timeframe in timeframes_to_scan:
                    timeframe_data = timeframe_map[
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

                    detected_patterns = detect_patterns(
                        timeframe_data
                    )

                    for signal in detected_patterns:
                        pattern_matches = (
                            scanner_pattern == "Any"
                            or signal["Pattern"]
                            == scanner_pattern
                        )

                        status_matches = (
                            scanner_status == "Any"
                            or signal["Status"]
                            == scanner_status
                        )

                        if pattern_matches and status_matches:
                            matching_signals.append(
                                {
                                    "Timeframe": timeframe,
                                    "Overall Trend": timeframe_trend,
                                    "RS Status": stock_rs["status"],
                                    "RS Trend": stock_rs["rs_trend"],
                                    "RS 3M %": stock_rs["relative_3m"],
                                    "RS 6M %": stock_rs["relative_6m"],
                                    "MTF Score": stock_alignment[
                                        "score"
                                    ],
                                    "MTF Alignment": stock_alignment[
                                        "status"
                                    ],
                                    **signal,
                                }
                            )

                if matching_signals:
                    stock_fundamentals = fetch_fundamentals(
                        ticker
                    )

                    market_cap = (
                        safe_number(
                            stock_fundamentals.get(
                                "marketCap"
                            ),
                            0,
                        )
                        / 10000000
                    )

                    if market_cap < minimum_market_cap:
                        continue

                    for signal in matching_signals:
                        results.append(
                            {
                                "Stock": symbol,
                                "Company": record[
                                    "Company Name"
                                ],
                                "Industry": record[
                                    "Industry"
                                ],
                                "Market Cap (Cr)": round(
                                    market_cap,
                                    0,
                                ),
                                **signal,
                            }
                        )

            except Exception:
                pass

            progress_bar.progress(
                min(
                    (index + 1) / total_stocks,
                    1.0,
                )
            )

        progress_bar.empty()
        progress_label.empty()

        st.subheader(
            f"Matching Signals: {len(results)}"
        )

        if results:
            results_dataframe = pd.DataFrame(results)

            signal_priority = {
                "Confirmed": 1,
                "In progress": 2,
                "Candidate": 3,
            }

            results_dataframe["Status Priority"] = (
                results_dataframe["Status"]
                .map(signal_priority)
                .fillna(99)
            )

            results_dataframe = (
                results_dataframe
                .sort_values(
                    by=[
                        "Status Priority",
                        "MTF Score",
                        "RS 6M %",
                        "Return %",
                    ],
                    ascending=[
                        True,
                        False,
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

            result_col1, result_col2, result_col3, result_col4 = (
                st.columns(4)
            )

            result_col1.metric(
                "Total Matches",
                len(results_dataframe),
            )

            result_col2.metric(
                "Confirmed",
                int(
                    (
                        results_dataframe["Status"]
                        == "Confirmed"
                    ).sum()
                ),
            )

            result_col3.metric(
                "RS Leaders",
                int(
                    (
                        results_dataframe["RS Status"]
                        == "Leader"
                    ).sum()
                ),
            )

            result_col4.metric(
                "Strong MTF Alignment",
                int(
                    (
                        results_dataframe[
                            "MTF Alignment"
                        ]
                        == "Strong multi-timeframe alignment"
                    ).sum()
                ),
            )

            st.dataframe(
                results_dataframe,
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
                        "RS vs Nifty 50: 3M",
                        format="%.2f%%",
                    ),
                    "RS 6M %": st.column_config.NumberColumn(
                        "RS vs Nifty 50: 6M",
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

            csv_data = results_dataframe.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Download Results as CSV",
                data=csv_data,
                file_name="nifty_total_market_rs_pattern_scan.csv",
                mime="text/csv",
            )

        else:
            st.info(
                "No stocks matched all selected filters. "
                "Try selecting Any for one or more filters."
            )


# =============================================================================
# FOOTER
# =============================================================================

st.divider()

st.caption(
    "Data sources: Nifty Indices constituent list and Yahoo Finance. "
    "Relative Strength compares stocks to Nifty 50 (^NSEI). "
    "Technical patterns and alignment ratings are rule-based research aids. "
    "Verify data independently before making any investment decision. "
    "This dashboard is for research and education only, not investment advice."
)
