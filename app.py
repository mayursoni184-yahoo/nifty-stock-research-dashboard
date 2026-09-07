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
# CSS
# =============================================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 800;
            color: #111827;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            color: #6b7280;
            font-size: 1rem;
            margin-bottom: 1rem;
        }

        .research-card {
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 16px;
            background-color: #ffffff;
            margin-bottom: 12px;
        }

        .strong-card {
            border-left: 5px solid #16a34a;
        }

        .warning-card {
            border-left: 5px solid #f59e0b;
        }

        .weak-card {
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

TIMEFRAME_OPTIONS = [
    "Daily",
    "Weekly",
    "Monthly",
]

TREND_OPTIONS = [
    "Any",
    "Strong bullish",
    "Bullish",
    "Neutral / consolidating",
    "Bearish",
    "Insufficient data",
]

STATUS_OPTIONS = [
    "Any",
    "Confirmed",
    "In progress",
    "Candidate",
]


# =============================================================================
# NIFTY TOTAL MARKET UNIVERSE
# =============================================================================

@st.cache_data(ttl=21600, show_spinner=False)
def get_nifty_total_market_members():
    """
    Fetches current Nifty Total Market constituents.

    If the official Nifty Indices file cannot be downloaded,
    a smaller fallback universe is returned so the application
    remains usable.
    """

    request_headers = {
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
            headers=request_headers,
            timeout=15,
        )

        response.raise_for_status()

        constituent_data = pd.read_csv(
            io.BytesIO(response.content)
        )

        constituent_data.columns = [
            str(column).strip()
            for column in constituent_data.columns
        ]

        if "Symbol" not in constituent_data.columns:
            raise ValueError(
                "Constituent file does not contain a Symbol column."
            )

        if "Series" in constituent_data.columns:
            constituent_data = constituent_data[
                constituent_data["Series"]
                .astype(str)
                .str.upper()
                .eq("EQ")
            ].copy()

        constituent_data["Symbol"] = (
            constituent_data["Symbol"]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        constituent_data["Ticker"] = (
            constituent_data["Symbol"] + ".NS"
        )

        if "Company Name" not in constituent_data.columns:
            constituent_data["Company Name"] = (
                constituent_data["Symbol"]
            )

        if "Industry" not in constituent_data.columns:
            constituent_data["Industry"] = "Unknown"

        constituent_data = constituent_data[
            [
                "Company Name",
                "Industry",
                "Symbol",
                "Ticker",
            ]
        ]

        constituent_data = (
            constituent_data
            .drop_duplicates("Symbol")
            .reset_index(drop=True)
        )

        if len(constituent_data) < 100:
            raise ValueError(
                "Too few valid constituents were received."
            )

        return constituent_data, None

    except Exception as error:
        fallback_data = pd.DataFrame(
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

        fallback_warning = (
            "Could not load the live Nifty Total Market constituent list. "
            f"Using {len(fallback_data)} fallback stocks. "
            f"Technical reason: {type(error).__name__}: {error}"
        )

        return fallback_data, fallback_warning


# =============================================================================
# PRICE AND FUNDAMENTAL DATA
# =============================================================================

@st.cache_data(ttl=900, show_spinner=False)
def fetch_price_data(ticker, period="5y"):
    """
    Fetches OHLCV data from Yahoo Finance.

    The result is cached for 15 minutes.
    """

    try:
        stock = yf.Ticker(ticker)

        price_data = stock.history(
            period=period,
            auto_adjust=True,
        )

        if price_data is None or price_data.empty:
            return pd.DataFrame()

        price_data = price_data.rename(
            columns=str.lower
        )

        required_columns = [
            "open",
            "high",
            "low",
            "close",
            "volume",
        ]

        price_data = price_data[
            required_columns
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
def fetch_fundamental_data(ticker):
    """
    Fetches available basic company fundamentals.

    Yahoo Finance does not reliably provide all India-specific
    quarterly and operational KPIs, so those are handled separately.
    """

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
        company = yf.Ticker(ticker)
        company_info = company.get_info()

        return {
            field: company_info.get(field)
            for field in requested_fields
        }

    except Exception:
        return {}


# =============================================================================
# TECHNICAL ANALYSIS UTILITY FUNCTIONS
# =============================================================================

def resample_ohlcv(price_data, timeframe):
    """
    Resamples daily OHLCV data into weekly or monthly candles.
    """

    if price_data.empty:
        return price_data

    if timeframe == "Daily":
        return price_data.copy()

    if timeframe == "Weekly":
        frequency = "W-FRI"
    else:
        frequency = "ME"

    return (
        price_data
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


def find_pivots(values, pivot_type="high", order=3):
    """
    Finds local price highs or lows.

    This is a lightweight rule-based pivot algorithm.
    """

    values = np.asarray(values, dtype=float)

    if len(values) < (2 * order) + 1:
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
    price_data,
    level,
    direction,
    notes,
):
    """
    Creates one standard signal object for display in the dashboard.
    """

    current_price = float(
        price_data["close"].iloc[-1]
    )

    recent_volumes = (
        price_data["volume"]
        .tail(21)
        .iloc[:-1]
    )

    average_volume = float(
        recent_volumes.mean()
    )

    latest_volume = float(
        price_data["volume"].iloc[-1]
    )

    volume_difference = (
        latest_volume / max(average_volume, 1) - 1
    ) * 100

    return {
        "Pattern": pattern,
        "Status": status,
        "Direction": direction,
        "Date": price_data.index[-1],
        "Level": float(level),
        "Current": current_price,
        "Return %": (
            current_price / float(level) - 1
        ) * 100,
        "Volume %": volume_difference,
        "Notes": notes,
    }


def detect_patterns(price_data):
    """
    Detects supported technical patterns.

    The rules are intentionally conservative. A Confirmed label
    requires a close through the defined neckline, support or
    resistance threshold.
    """

    if price_data is None or price_data.empty:
        return []

    price_data = price_data.dropna().copy()

    if len(price_data) < 40:
        return []

    high = price_data["high"].to_numpy(dtype=float)
    low = price_data["low"].to_numpy(dtype=float)
    close = price_data["close"].to_numpy(dtype=float)
    open_price = price_data["open"].to_numpy(dtype=float)

    detected = []

    pivot_highs = find_pivots(
        high,
        pivot_type="high",
        order=3,
    )

    pivot_lows = find_pivots(
        low,
        pivot_type="low",
        order=3,
    )

    # -------------------------------------------------------------------------
    # DOUBLE TOP
    # -------------------------------------------------------------------------

    if len(pivot_highs) >= 2:
        first_top = pivot_highs[-2]
        second_top = pivot_highs[-1]

        price_difference = abs(
            high[first_top] - high[second_top]
        ) / max(high[first_top], 1)

        if (
            second_top - first_top >= 8
            and price_difference < 0.045
        ):
            neckline = float(
                np.min(
                    low[first_top:second_top + 1]
                )
            )

            signal_status = (
                "Confirmed"
                if close[-1] < neckline * 0.99
                else "In progress"
            )

            detected.append(
                build_pattern_signal(
                    pattern="Double Top",
                    status=signal_status,
                    price_data=price_data,
                    level=neckline,
                    direction="Bearish",
                    notes=(
                        "Two similar peaks. Confirmation requires "
                        "a close below the neckline."
                    ),
                )
            )

    # -------------------------------------------------------------------------
    # DOUBLE BOTTOM
    # -------------------------------------------------------------------------

    if len(pivot_lows) >= 2:
        first_bottom = pivot_lows[-2]
        second_bottom = pivot_lows[-1]

        price_difference = abs(
            low[first_bottom] - low[second_bottom]
        ) / max(low[first_bottom], 1)

        if (
            second_bottom - first_bottom >= 8
            and price_difference < 0.045
        ):
            neckline = float(
                np.max(
                    high[first_bottom:second_bottom + 1]
                )
            )

            signal_status = (
                "Confirmed"
                if close[-1] > neckline * 1.01
                else "In progress"
            )

            detected.append(
                build_pattern_signal(
                    pattern="Double Bottom",
                    status=signal_status,
                    price_data=price_data,
                    level=neckline,
                    direction="Bullish",
                    notes=(
                        "Two similar troughs. Confirmation requires "
                        "a close above the neckline."
                    ),
                )
            )

    # -------------------------------------------------------------------------
    # HEAD AND SHOULDERS
    # -------------------------------------------------------------------------

    if len(pivot_highs) >= 3:
        left_shoulder = pivot_highs[-3]
        head = pivot_highs[-2]
        right_shoulder = pivot_highs[-1]

        shoulder_average = (
            high[left_shoulder] +
            high[right_shoulder]
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
            high[left_shoulder] -
            high[right_shoulder]
        ) / max(shoulder_average, 1)

        if (
            high[head] > shoulder_average * 1.04
            and shoulder_difference < 0.10
        ):
            signal_status = (
                "Confirmed"
                if close[-1] < neckline * 0.99
                else "In progress"
            )

            detected.append(
                build_pattern_signal(
                    pattern="Head & Shoulders",
                    status=signal_status,
                    price_data=price_data,
                    level=neckline,
                    direction="Bearish",
                    notes=(
                        "Confirmation requires a close below "
                        "neckline support."
                    ),
                )
            )

    # -------------------------------------------------------------------------
    # INVERSE HEAD AND SHOULDERS
    # -------------------------------------------------------------------------

    if len(pivot_lows) >= 3:
        left_shoulder = pivot_lows[-3]
        head = pivot_lows[-2]
        right_shoulder = pivot_lows[-1]

        shoulder_average = (
            low[left_shoulder] +
            low[right_shoulder]
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
            low[left_shoulder] -
            low[right_shoulder]
        ) / max(shoulder_average, 1)

        if (
            low[head] < shoulder_average * 0.96
            and shoulder_difference < 0.10
        ):
            signal_status = (
                "Confirmed"
                if close[-1] > neckline * 1.01
                else "In progress"
            )

            detected.append(
                build_pattern_signal(
                    pattern="Inverse Head & Shoulders",
                    status=signal_status,
                    price_data=price_data,
                    level=neckline,
                    direction="Bullish",
                    notes=(
                        "Confirmation requires a close above "
                        "neckline resistance."
                    ),
                )
            )

    # -------------------------------------------------------------------------
    # RECTANGLES AND TRIANGLES
    # -------------------------------------------------------------------------

    pattern_window = 30

    if len(price_data) >= pattern_window:
        recent_highs = high[-pattern_window:]
        recent_lows = low[-pattern_window:]

        x_axis = np.arange(pattern_window)

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

        resistance = float(
            np.max(recent_highs)
        )

        support = float(
            np.min(recent_lows)
        )

        price_range = (
            resistance - support
        ) / max(resistance, 1)

        if price_range < 0.15:
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

            if pattern_name is not None:
                if close[-1] > resistance * 1.01:
                    signal_status = "Confirmed"
                    direction = "Bullish"
                    breakout_level = resistance

                elif close[-1] < support * 0.99:
                    signal_status = "Confirmed"
                    direction = "Bearish"
                    breakout_level = support

                else:
                    signal_status = "In progress"
                    direction = "Neutral"

                    if pattern_name == "Descending Triangle":
                        breakout_level = support
                    else:
                        breakout_level = resistance

                detected.append(
                    build_pattern_signal(
                        pattern=pattern_name,
                        status=signal_status,
                        price_data=price_data,
                        level=breakout_level,
                        direction=direction,
                        notes=(
                            "Breakout confirmation requires a close "
                            "outside the defined pattern range."
                        ),
                    )
                )

    # -------------------------------------------------------------------------
    # REVERSAL CANDLE CANDIDATES
    # -------------------------------------------------------------------------

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
        min(latest_open, latest_close) -
        latest_low
    )

    upper_shadow = (
        latest_high -
        max(latest_open, latest_close)
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
                pattern="Reversal Bottom",
                status="Candidate",
                price_data=price_data,
                level=latest_low,
                direction="Bullish",
                notes=(
                    "Hammer-like candle near a local low. "
                    "Wait for next-candle confirmation."
                ),
            )
        )

    if (
        upper_shadow / candle_range > 0.55
        and candle_body / candle_range < 0.35
        and latest_close >= previous_high * 0.96
    ):
        detected.append(
            build_pattern_signal(
                pattern="Reversal Top",
                status="Candidate",
                price_data=price_data,
                level=latest_high,
                direction="Bearish",
                notes=(
                    "Shooting-star-like candle near a local high. "
                    "Wait for next-candle confirmation."
                ),
            )
        )

    return detected


def calculate_support_resistance(price_data):
    """
    Calculates pivot-based nearby support and resistance levels.
    """

    if len(price_data) < 30:
        support = float(
            price_data["low"].tail(10).min()
        )

        resistance = float(
            price_data["high"].tail(10).max()
        )

        return support, resistance

    current_price = float(
        price_data["close"].iloc[-1]
    )

    pivot_lows = find_pivots(
        price_data["low"].to_numpy(),
        pivot_type="low",
    )

    pivot_highs = find_pivots(
        price_data["high"].to_numpy(),
        pivot_type="high",
    )

    possible_supports = [
        float(price_data["low"].iloc[index])
        for index in pivot_lows
        if price_data["low"].iloc[index] < current_price
    ]

    possible_resistances = [
        float(price_data["high"].iloc[index])
        for index in pivot_highs
        if price_data["high"].iloc[index] > current_price
    ]

    if possible_supports:
        support = max(
            possible_supports[-8:]
        )
    else:
        support = float(
            price_data["low"].tail(20).min()
        )

    if possible_resistances:
        resistance = min(
            possible_resistances[-8:]
        )
    else:
        resistance = float(
            price_data["high"].tail(20).max()
        )

    return support, resistance


def calculate_overall_trend(price_data):
    """
    Uses 20, 50, and 200 moving averages to classify overall trend.
    """

    if len(price_data) < 55:
        return "Insufficient data"

    current_price = float(
        price_data["close"].iloc[-1]
    )

    moving_average_20 = float(
        price_data["close"]
        .rolling(20)
        .mean()
        .iloc[-1]
    )

    moving_average_50 = float(
        price_data["close"]
        .rolling(50)
        .mean()
        .iloc[-1]
    )

    if len(price_data) >= 200:
        moving_average_200 = float(
            price_data["close"]
            .rolling(200)
            .mean()
            .iloc[-1]
        )

        if (
            current_price > moving_average_20
            > moving_average_50
            > moving_average_200
        ):
            return "Strong bullish"

    if (
        current_price > moving_average_20
        and moving_average_20 > moving_average_50
    ):
        return "Bullish"

    if (
        current_price < moving_average_20
        and moving_average_20 < moving_average_50
    ):
        return "Bearish"

    return "Neutral / consolidating"


# =============================================================================
# FUNDAMENTAL UTILITY FUNCTIONS
# =============================================================================

def safe_number(value, default=None):
    """Returns a float when possible, otherwise a safe default."""

    try:
        if value is None:
            return default

        if pd.isna(value):
            return default

        return float(value)

    except Exception:
        return default


def format_value(value, is_percentage=False):
    """Formats numbers safely for dashboard display."""

    value = safe_number(value)

    if value is None:
        return "—"

    if is_percentage:
        return f"{value * 100:.1f}%"

    return f"{value:.2f}"


def calculate_basic_fundamental_score(fundamentals):
    """
    Scores Yahoo Finance summary data.

    Maximum score: 9.
    """

    score = 0
    strengths = []
    concerns = []

    roe = safe_number(
        fundamentals.get("returnOnEquity")
    )

    debt_equity = safe_number(
        fundamentals.get("debtToEquity")
    )

    profit_margin = safe_number(
        fundamentals.get("profitMargins")
    )

    revenue_growth = safe_number(
        fundamentals.get("revenueGrowth")
    )

    free_cash_flow = safe_number(
        fundamentals.get("freeCashflow")
    )

    if roe is not None:
        if roe >= 0.20:
            score += 2
            strengths.append("ROE is at least 20%")

        elif roe >= 0.15:
            score += 1
            strengths.append("ROE is at least 15%")

        elif roe < 0.10:
            concerns.append("ROE is below 10%")

    if debt_equity is not None:
        if debt_equity < 50:
            score += 2
            strengths.append("Low debt/equity")

        elif debt_equity < 100:
            score += 1
            strengths.append("Moderate debt/equity")

        elif debt_equity > 200:
            concerns.append("High debt/equity")

    if profit_margin is not None:
        if profit_margin >= 0.15:
            score += 2
            strengths.append("Profit margin is at least 15%")

        elif profit_margin >= 0.08:
            score += 1
            strengths.append("Profit margin is at least 8%")

        elif profit_margin < 0.05:
            concerns.append("Low profit margin")

    if revenue_growth is not None:
        if revenue_growth >= 0.12:
            score += 1
            strengths.append("Revenue growth is at least 12%")

        elif revenue_growth < 0:
            concerns.append("Revenue growth is negative")

    if free_cash_flow is not None:
        if free_cash_flow > 0:
            score += 2
            strengths.append("Positive free cash flow")

        else:
            concerns.append("Negative free cash flow")

    if score >= 7:
        rating = "Excellent"

    elif score >= 5:
        rating = "Good"

    elif score >= 3:
        rating = "Average"

    else:
        rating = "Needs review"

    return score, rating, strengths, concerns


# =============================================================================
# ANNUAL AND QUARTERLY FUNDAMENTAL ENGINE
# =============================================================================

def assess_annual_fundamentals(annual_data):
    """
    Assesses annual business quality.

    Future NSE/XBRL, annual-report, or paid API data can populate
    annual_data with real historical financial information.
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

    free_cash_flow_positive = annual_data.get(
        "free_cash_flow_positive"
    )

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
                f"Negative revenue CAGR: {revenue_cagr:.1f}%"
            )

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
                f"Negative profit CAGR: {profit_cagr:.1f}%"
            )

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

    if free_cash_flow_positive is True:
        score += 1
        positives.append("Positive free cash flow")

    elif free_cash_flow_positive is False:
        concerns.append("Negative free cash flow")

    if score >= 9:
        rating = "Strong"

    elif score >= 6:
        rating = "Strong / Improving"

    elif score >= 4:
        rating = "Stable"

    elif score >= 2:
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
    Assesses quarterly financial and sector-KPI momentum.
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

    ebitda_yoy = safe_number(
        quarterly_data.get("ebitda_yoy")
    )

    margin_change_bps = safe_number(
        quarterly_data.get("margin_change_bps_yoy")
    )

    operating_kpi_trend = quarterly_data.get(
        "operating_kpi_trend"
    )

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
                f"Revenue declined: {revenue_yoy:.1f}% YoY"
            )

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
                f"Profit declined: {profit_yoy:.1f}% YoY"
            )

    if ebitda_yoy is not None:
        if ebitda_yoy >= 15:
            score += 1
            positives.append(
                f"Strong EBITDA growth: {ebitda_yoy:.1f}% YoY"
            )

        elif ebitda_yoy < 0:
            concerns.append(
                f"EBITDA declined: {ebitda_yoy:.1f}% YoY"
            )

    if margin_change_bps is not None:
        if margin_change_bps >= 100:
            score += 2
            positives.append(
                f"Margin expansion: {margin_change_bps:.0f} bps YoY"
            )

        elif margin_change_bps >= 25:
            score += 1
            positives.append(
                f"Margin improvement: {margin_change_bps:.0f} bps YoY"
            )

        elif margin_change_bps <= -100:
            concerns.append(
                f"Margin contraction: {margin_change_bps:.0f} bps YoY"
            )

    if operating_kpi_trend == "Improving":
        score += 2
        positives.append(
            "Sector-specific operational KPIs are improving"
        )

    elif operating_kpi_trend == "Stable":
        score += 1
        positives.append(
            "Sector-specific operational KPIs are stable"
        )

    elif operating_kpi_trend == "Deteriorating":
        concerns.append(
            "Sector-specific operational KPIs are deteriorating"
        )

    if score >= 7:
        rating = "Strong / Strengthening"

    elif score >= 5:
        rating = "Strong"

    elif score >= 3:
        rating = "Early Recovery"

    elif score >= 1:
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
    """
    Combines annual quality and quarterly momentum.
    """

    annual_rating = annual_assessment["rating"]
    quarterly_rating = quarterly_assessment["rating"]

    annual_score = annual_assessment["score"]
    quarterly_score = quarterly_assessment["score"]

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

    elif combined_score >= 6:
        combined_rating = "Strong"

    elif combined_score >= 3:
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
    """
    Determines the long-term to short-term fundamental transition.
    """

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
# SECTOR KPI LIBRARY
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
        "Provision Coverage Ratio",
        "Cost to Income Ratio",
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
        "Working Capital Days",
        "EBITDA Margin",
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
    ],
    "Oil & Gas / Refining": [
        "GRM",
        "Throughput",
        "Crude Production",
        "Gas Production",
        "Realization",
        "Refining Margin",
        "Reserve Replacement Ratio",
    ],
    "Retail / FMCG": [
        "Same Store Sales Growth",
        "Volume Growth",
        "Value Growth",
        "Gross Margin",
        "Store Additions",
        "Store Productivity",
        "Private Label Mix",
    ],
    "Pharma": [
        "US Sales Growth",
        "India Sales Growth",
        "ANDA Filings",
        "ANDA Approvals",
        "R&D as Percentage of Sales",
        "Product Concentration",
        "Complex Generics Mix",
    ],
}


# =============================================================================
# CHART FUNCTION
# =============================================================================

def create_stock_chart(
    price_data,
    pattern_signals,
    title,
):
    """
    Creates a candlestick chart with volume, moving averages,
    and detected pattern levels.
    """

    chart_data = price_data.tail(260).copy()

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

    for signal in pattern_signals:
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


# =============================================================================
# LOAD THE STOCK UNIVERSE
# =============================================================================

with st.spinner(
    "Loading Nifty Total Market constituents..."
):
    universe, universe_error = (
        get_nifty_total_market_members()
    )


# =============================================================================
# DASHBOARD HEADER
# =============================================================================

st.markdown(
    """
    <div class="main-title">
        📊 Nifty Total Market (750) Research Dashboard
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Daily, weekly and monthly pattern analysis • support and resistance •
        trend filters • annual and quarterly fundamental momentum
    </div>
    """,
    unsafe_allow_html=True,
)

if universe_error:
    st.warning(universe_error)

if universe.empty:
    st.error(
        "No stock universe is available. "
        "Please click Refresh cached data and try again."
    )
    st.stop()


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.header("🔍 Research Controls")

    app_mode = st.radio(
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
        "Prices cache for 15 minutes.\n\n"
        "Fundamentals cache for 12 hours.\n\n"
        "Nifty constituent list cache for 6 hours."
    )


# =============================================================================
# STOCK RESEARCH MODE
# =============================================================================

if app_mode == "Stock research":
    available_symbols = sorted(
        universe["Symbol"].tolist()
    )

    default_symbol_index = 0

    if "RELIANCE" in available_symbols:
        default_symbol_index = (
            available_symbols.index("RELIANCE")
        )

    selected_symbol = st.selectbox(
        "Search a Nifty Total Market stock",
        available_symbols,
        index=default_symbol_index,
    )

    selected_stock = universe[
        universe["Symbol"] == selected_symbol
    ].iloc[0]

    selected_ticker = selected_stock["Ticker"]
    selected_company_name = selected_stock[
        "Company Name"
    ]
    selected_industry = selected_stock["Industry"]

    with st.spinner(
        f"Fetching latest data for {selected_symbol}..."
    ):
        daily_prices = fetch_price_data(
            selected_ticker,
            "5y",
        )

        fundamentals = fetch_fundamental_data(
            selected_ticker
        )

    if daily_prices.empty:
        st.error(
            "No price data was available for this symbol. "
            "Please try again later."
        )
        st.stop()

    market_cap_crore = (
        fundamentals.get("marketCap") or 0
    ) / 10000000

    basic_score, basic_rating, basic_strengths, basic_concerns = (
        calculate_basic_fundamental_score(
            fundamentals
        )
    )

    current_close = float(
        daily_prices["close"].iloc[-1]
    )

    header_col1, header_col2, header_col3, header_col4 = (
        st.columns(4)
    )

    header_col1.metric(
        "Last close",
        f"₹{current_close:,.2f}",
    )

    header_col2.metric(
        "Market cap",
        (
            f"₹{market_cap_crore:,.0f} Cr"
            if market_cap_crore > 0
            else "Unavailable"
        ),
    )

    header_col3.metric(
        "Daily trend",
        calculate_overall_trend(daily_prices),
    )

    header_col4.metric(
        "Basic score",
        f"{basic_score}/9 · {basic_rating}",
    )

    st.caption(
        f"{selected_company_name} • {selected_industry}"
    )

    if (
        market_cap_crore > 0
        and market_cap_crore < minimum_market_cap
    ):
        st.warning(
            f"{selected_symbol} is below your selected market-cap "
            f"threshold of ₹{minimum_market_cap:,.0f} crore."
        )

    (
        daily_tab,
        weekly_tab,
        monthly_tab,
        fundamentals_tab,
    ) = st.tabs(
        [
            "Daily",
            "Weekly",
            "Monthly",
            "Fundamentals",
        ]
    )

    timeframe_tab_pairs = [
        (daily_tab, "Daily"),
        (weekly_tab, "Weekly"),
        (monthly_tab, "Monthly"),
    ]

    for tab, timeframe in timeframe_tab_pairs:
        with tab:
            timeframe_prices = resample_ohlcv(
                daily_prices,
                timeframe,
            )

            timeframe_signals = detect_patterns(
                timeframe_prices
            )

            support_level, resistance_level = (
                calculate_support_resistance(
                    timeframe_prices
                )
            )

            trend_col, support_col, resistance_col = (
                st.columns(3)
            )

            trend_col.metric(
                "Overall trend",
                calculate_overall_trend(
                    timeframe_prices
                ),
            )

            support_col.metric(
                "Nearest support",
                f"₹{support_level:,.2f}",
            )

            resistance_col.metric(
                "Nearest resistance",
                f"₹{resistance_level:,.2f}",
            )

            st.subheader(
                f"{timeframe} pattern research"
            )

            if timeframe_signals:
                signals_dataframe = pd.DataFrame(
                    timeframe_signals
                )

                st.dataframe(
                    signals_dataframe,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Date": st.column_config.DatetimeColumn(
                            "Signal date",
                            format="YYYY-MM-DD",
                        ),
                        "Level": st.column_config.NumberColumn(
                            "Breakout / neckline",
                            format="₹%.2f",
                        ),
                        "Current": st.column_config.NumberColumn(
                            "Current price",
                            format="₹%.2f",
                        ),
                        "Return %": st.column_config.NumberColumn(
                            "Return since breakout",
                            format="%.2f%%",
                        ),
                        "Volume %": st.column_config.NumberColumn(
                            "Volume vs 20-period average",
                            format="%.1f%%",
                        ),
                    },
                )

            else:
                st.info(
                    "No supported active, confirmed, or candidate pattern "
                    "was detected for this timeframe."
                )

            st.plotly_chart(
                create_stock_chart(
                    timeframe_prices,
                    timeframe_signals,
                    f"{selected_symbol} — {timeframe}",
                ),
                use_container_width=True,
            )

    # =========================================================================
    # FUNDAMENTALS TAB
    # =========================================================================

    with fundamentals_tab:
        st.subheader(
            f"Basic Fundamental Health: {basic_rating}"
        )

        basic_col1, basic_col2 = st.columns(2)

        with basic_col1:
            st.metric(
                "Basic fundamental score",
                f"{basic_score}/9",
            )

            if basic_strengths:
                st.markdown("#### Positive indicators")

                for strength in basic_strengths:
                    st.success(f"✅ {strength}")

        with basic_col2:
            if basic_concerns:
                st.markdown("#### Risk indicators")

                for concern in basic_concerns:
                    st.warning(f"⚠️ {concern}")

        free_cash_flow = fundamentals.get(
            "freeCashflow"
        )

        if isinstance(free_cash_flow, (int, float)):
            free_cash_flow_display = (
                f"₹{free_cash_flow / 10000000:,.0f} Cr"
            )
        else:
            free_cash_flow_display = "—"

        summary_fundamental_table = pd.DataFrame(
            [
                [
                    "Sector",
                    fundamentals.get("sector", "—"),
                ],
                [
                    "Industry",
                    fundamentals.get("industry", "—"),
                ],
                [
                    "Trailing P/E",
                    format_value(
                        fundamentals.get("trailingPE")
                    ),
                ],
                [
                    "Forward P/E",
                    format_value(
                        fundamentals.get("forwardPE")
                    ),
                ],
                [
                    "Price / Book",
                    format_value(
                        fundamentals.get("priceToBook")
                    ),
                ],
                [
                    "ROE",
                    format_value(
                        fundamentals.get("returnOnEquity"),
                        is_percentage=True,
                    ),
                ],
                [
                    "ROA",
                    format_value(
                        fundamentals.get("returnOnAssets"),
                        is_percentage=True,
                    ),
                ],
                [
                    "Profit margin",
                    format_value(
                        fundamentals.get("profitMargins"),
                        is_percentage=True,
                    ),
                ],
                [
                    "Operating margin",
                    format_value(
                        fundamentals.get("operatingMargins"),
                        is_percentage=True,
                    ),
                ],
                [
                    "Revenue growth",
                    format_value(
                        fundamentals.get("revenueGrowth"),
                        is_percentage=True,
                    ),
                ],
                [
                    "Earnings growth",
                    format_value(
                        fundamentals.get("earningsGrowth"),
                        is_percentage=True,
                    ),
                ],
                [
                    "Debt / Equity",
                    format_value(
                        fundamentals.get("debtToEquity")
                    ),
                ],
                [
                    "Current ratio",
                    format_value(
                        fundamentals.get("currentRatio")
                    ),
                ],
                [
                    "Free cash flow",
                    free_cash_flow_display,
                ],
                [
                    "Dividend yield",
                    format_value(
                        fundamentals.get("dividendYield"),
                        is_percentage=True,
                    ),
                ],
            ],
            columns=[
                "Metric",
                "Value",
            ],
        )

        st.markdown("### Basic fundamental metrics")

        st.dataframe(
            summary_fundamental_table,
            hide_index=True,
            use_container_width=True,
        )

        st.divider()

        # ---------------------------------------------------------------------
        # ANNUAL / QUARTERLY FUNDAMENTAL MOMENTUM
        # ---------------------------------------------------------------------

        st.subheader(
            "📈 Annual and Quarterly Fundamental Momentum"
        )

        st.caption(
            "The scoring engine is active. Detailed annual, quarterly, "
            "and sector-specific operational figures require an NSE XBRL, "
            "company filing, annual-report, or specialist India-market "
            "data integration. Missing fields are therefore not treated "
            "as positive signals."
        )

        # ---------------------------------------------------------------------
        # CURRENT PLACEHOLDER DATA
        #
        # These values are intentionally empty until an NSE/XBRL or database
        # integration is connected. The scoring logic is ready to evaluate
        # real values when captured.
        # ---------------------------------------------------------------------

        annual_financial_data = {
            "revenue_cagr_3y": None,
            "profit_cagr_3y": None,
            "roe": None,
            "roce": None,
            "operating_margin": None,
            "margin_trend_bps": None,
            "debt_to_equity": None,
            "free_cash_flow_positive": None,
        }

        quarterly_financial_data = {
            "revenue_yoy": None,
            "profit_yoy": None,
            "ebitda_yoy": None,
            "margin_change_bps_yoy": None,
            "operating_kpi_trend": None,
        }

        annual_assessment = assess_annual_fundamentals(
            annual_financial_data
        )

        quarterly_assessment = (
            assess_quarterly_fundamentals(
                quarterly_financial_data
            )
        )

        combined_assessment = (
            calculate_combined_fundamental_status(
                annual_assessment,
                quarterly_assessment,
            )
        )

        transition_label = (
            determine_fundamental_transition(
                annual_assessment["rating"],
                quarterly_assessment["rating"],
            )
        )

        annual_col, quarterly_col = st.columns(2)

        with annual_col:
            st.markdown(
                f"""
                <div class="research-card">
                    <h4>Annual Fundamental Status</h4>
                    <h2>{annual_assessment["rating"]}</h2>
                    <p>Annual score: {annual_assessment["score"]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with quarterly_col:
            st.markdown(
                f"""
                <div class="research-card">
                    <h4>Quarterly Fundamental Status</h4>
                    <h2>{quarterly_assessment["rating"]}</h2>
                    <p>Quarterly score: {quarterly_assessment["score"]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        combined_col, transition_col = st.columns(2)

        with combined_col:
            st.markdown(
                f"""
                <div class="research-card strong-card">
                    <h4>Combined Fundamental Status</h4>
                    <h2>{combined_assessment["rating"]}</h2>
                    <p>Combined score: {combined_assessment["score"]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with transition_col:
            st.markdown(
                f"""
                <div class="research-card warning-card">
                    <h4>Fundamental Transition</h4>
                    <h2>{transition_label}</h2>
                    <p>Annual quality compared with quarterly momentum</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("### Annual evidence")

        if annual_assessment["positives"]:
            for item in annual_assessment["positives"]:
                st.success(f"✅ {item}")
        else:
            st.info(
                "Annual historical data has not yet been loaded from "
                "NSE/XBRL, annual reports, or a dedicated data provider."
            )

        if annual_assessment["concerns"]:
            for item in annual_assessment["concerns"]:
                st.warning(f"⚠️ {item}")

        st.markdown("### Quarterly evidence")

        if quarterly_assessment["positives"]:
            for item in quarterly_assessment["positives"]:
                st.success(f"✅ {item}")
        else:
            st.info(
                "Quarterly financial and operational KPI data has not "
                "yet been loaded for this stock."
            )

        if quarterly_assessment["concerns"]:
            for item in quarterly_assessment["concerns"]:
                st.warning(f"⚠️ {item}")

        st.divider()

        st.subheader(
            "🏭 Sector-specific metrics framework"
        )

        selected_sector_kpi_group = st.selectbox(
            "Select a sector KPI template",
            sorted(
                SECTOR_KPI_LIBRARY.keys()
            ),
        )

        sector_kpis = SECTOR_KPI_LIBRARY[
            selected_sector_kpi_group
        ]

        sector_kpi_dataframe = pd.DataFrame(
            {
                "Sector KPI": sector_kpis,
                "Latest value": ["Not captured"] * len(
                    sector_kpis
                ),
                "YoY change": ["Not captured"] * len(
                    sector_kpis
                ),
                "QoQ change": ["Not captured"] * len(
                    sector_kpis
                ),
                "Source": ["NSE filing / company result presentation"] * len(
                    sector_kpis
                ),
                "Type": ["Reported or calculated"] * len(
                    sector_kpis
                ),
            }
        )

        st.dataframe(
            sector_kpi_dataframe,
            hide_index=True,
            use_container_width=True,
        )

        st.info(
            "Examples of reported KPIs: order book, ARPU, CASA, VNB, "
            "TCV, attrition, utilization, pre-sales, GRM and volume growth. "
            "Examples of calculated KPIs: revenue growth, margin change, "
            "book-to-bill ratio, cash conversion, debt reduction and "
            "quarter-on-quarter performance."
        )

        st.warning(
            "For banks and NBFCs, the general debt/equity and free-cash-flow "
            "framework should not be used alone. Use NIM, GNPA, NNPA, credit "
            "cost, CASA, loan growth, deposit growth, ROA, ROE, capital "
            "adequacy, provision coverage and cost-to-income ratio."
        )


# =============================================================================
# PATTERN SCANNER MODE
# =============================================================================

else:
    st.subheader(
        "Nifty Total Market Pattern Scanner"
    )

    st.caption(
        "Filter current pattern signals by timeframe, overall trend, "
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
        selected_trend = st.selectbox(
            "Overall trend",
            TREND_OPTIONS,
        )

        selected_status = st.selectbox(
            "Pattern status",
            STATUS_OPTIONS,
        )

    st.info(
        "Timeframe = Any means that the app evaluates Daily, Weekly, "
        "and Monthly charts. A stock can appear more than once if it "
        "matches on multiple chart timeframes."
    )

    st.warning(
        "A full Nifty Total Market scan may take several minutes on "
        "free Streamlit Cloud hosting. Timeframe = Any requires "
        "Daily, Weekly and Monthly technical evaluations for each stock."
    )

    if st.button(
        "🔎 Scan Nifty Total Market",
        type="primary",
    ):
        scan_results = []

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

        scan_progress = st.progress(0)
        scan_status = st.empty()

        universe_size = len(universe)

        for index, stock_record in universe.iterrows():
            stock_symbol = stock_record["Symbol"]
            stock_ticker = stock_record["Ticker"]

            scan_status.caption(
                f"Scanning {index + 1:,} of {universe_size:,}: "
                f"{stock_symbol}"
            )

            try:
                stock_price_data = fetch_price_data(
                    stock_ticker,
                    "5y",
                )

                if stock_price_data.empty:
                    continue

                stock_matches = []

                for scan_timeframe in timeframes_to_scan:
                    timeframe_price_data = resample_ohlcv(
                        stock_price_data,
                        scan_timeframe,
                    )

                    if timeframe_price_data.empty:
                        continue

                    timeframe_trend = calculate_overall_trend(
                        timeframe_price_data
                    )

                    trend_matches = (
                        selected_trend == "Any"
                        or timeframe_trend == selected_trend
                    )

                    if not trend_matches:
                        continue

                    timeframe_signals = detect_patterns(
                        timeframe_price_data
                    )

                    for signal in timeframe_signals:
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
                                    "Timeframe": scan_timeframe,
                                    "Overall Trend": timeframe_trend,
                                    **signal,
                                }
                            )

                if stock_matches:
                    stock_fundamentals = (
                        fetch_fundamental_data(
                            stock_ticker
                        )
                    )

                    stock_market_cap_crore = (
                        stock_fundamentals.get("marketCap") or 0
                    ) / 10000000

                    if (
                        stock_market_cap_crore
                        < minimum_market_cap
                    ):
                        continue

                    for match in stock_matches:
                        scan_results.append(
                            {
                                "Stock": stock_symbol,
                                "Company": stock_record[
                                    "Company Name"
                                ],
                                "Industry": stock_record[
                                    "Industry"
                                ],
                                "Market Cap (Cr)": round(
                                    stock_market_cap_crore,
                                    0,
                                ),
                                **match,
                            }
                        )

            except Exception:
                pass

            scan_progress.progress(
                min(
                    (index + 1) / universe_size,
                    1.0,
                )
            )

        scan_progress.empty()
        scan_status.empty()

        st.subheader(
            f"Matching signals: {len(scan_results)}"
        )

        if scan_results:
            results_dataframe = pd.DataFrame(
                scan_results
            )

            status_priority = {
                "Confirmed": 1,
                "In progress": 2,
                "Candidate": 3,
            }

            results_dataframe["Status Priority"] = (
                results_dataframe["Status"]
                .map(status_priority)
                .fillna(99)
            )

            results_dataframe = (
                results_dataframe
                .sort_values(
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
                "Total matches",
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
                "Strong bullish trend",
                int(
                    (
                        results_dataframe[
                            "Overall Trend"
                        ]
                        == "Strong bullish"
                    ).sum()
                ),
            )

            result_col4.metric(
                "Bullish direction",
                int(
                    (
                        results_dataframe[
                            "Direction"
                        ]
                        == "Bullish"
                    ).sum()
                ),
            )

            st.dataframe(
                results_dataframe,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Date": st.column_config.DatetimeColumn(
                        "Signal date",
                        format="YYYY-MM-DD",
                    ),
                    "Level": st.column_config.NumberColumn(
                        "Breakout / neckline",
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

            result_csv = results_dataframe.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Download scanner results as CSV",
                data=result_csv,
                file_name=(
                    "nifty_total_market_pattern_scan.csv"
                ),
                mime="text/csv",
            )

        else:
            st.info(
                "No stocks matched your selected Pattern, Timeframe, "
                "Overall Trend, Status, and Market Cap filters. "
                "Try setting one or more filters to Any."
            )


# =============================================================================
# FOOTER
# =============================================================================

st.divider()

st.caption(
    "Data sources: Nifty Indices constituent data and Yahoo Finance. "
    "Technical pattern recognition is rule-based and can generate false "
    "positives or miss valid structures. Verify corporate actions, prices, "
    "financial results, operational metrics and investment suitability "
    "independently. This dashboard is for education and research only; "
    "it is not investment advice."
)
