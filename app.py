import io
import time
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
    page_title="Nifty Total Market Research Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }

        .sub-title {
            color: #6b7280;
            font-size: 1rem;
            margin-bottom: 1.2rem;
        }

        .card {
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 14px;
            background: #ffffff;
            margin-bottom: 10px;
        }

        .success-card {
            border-left: 5px solid #16a34a;
        }

        .warning-card {
            border-left: 5px solid #f59e0b;
        }

        .danger-card {
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

TIMEFRAMES = [
    "Daily",
    "Weekly",
    "Monthly",
]


# =============================================================================
# DATA FUNCTIONS
# =============================================================================

@st.cache_data(ttl=21600, show_spinner=False)
def get_nifty_total_market_members():
    """
    Downloads the official Nifty Total Market constituent list.

    If the official file is unavailable, blocked, slow, or malformed,
    the dashboard remains usable with a fallback stock list.
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

        data = pd.read_csv(io.BytesIO(response.content))
        data.columns = [str(column).strip() for column in data.columns]

        if "Symbol" not in data.columns:
            raise ValueError("Downloaded constituent file has no Symbol column.")

        if "Series" in data.columns:
            data = data[
                data["Series"]
                .astype(str)
                .str.upper()
                .eq("EQ")
            ].copy()

        data["Symbol"] = (
            data["Symbol"]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        data["Ticker"] = data["Symbol"] + ".NS"

        if "Company Name" not in data.columns:
            data["Company Name"] = data["Symbol"]

        if "Industry" not in data.columns:
            data["Industry"] = "Unknown"

        data = data[
            [
                "Company Name",
                "Industry",
                "Symbol",
                "Ticker",
            ]
        ]

        data = data.drop_duplicates("Symbol").reset_index(drop=True)

        if len(data) < 100:
            raise ValueError(
                f"Only {len(data)} stocks received; constituent data is incomplete."
            )

        return data, None

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

        error_message = (
            f"Could not load the live Nifty Total Market constituent list. "
            f"Using a fallback list of {len(fallback)} large stocks. "
            f"Technical reason: {type(error).__name__}: {error}"
        )

        return fallback, error_message


@st.cache_data(ttl=900, show_spinner=False)
def fetch_price_data(ticker, period="5y"):
    """
    Fetches OHLCV data from Yahoo Finance.

    Cached for 15 minutes to reduce repeated requests.
    """

    try:
        stock = yf.Ticker(ticker)

        data = stock.history(
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

        data = data[required_columns].dropna()

        data.index = pd.to_datetime(data.index)

        if getattr(data.index, "tz", None) is not None:
            data.index = data.index.tz_localize(None)

        return data

    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=43200, show_spinner=False)
def fetch_fundamentals(ticker):
    """
    Fetches available fundamental data from Yahoo Finance.

    Cached for 12 hours because fundamentals do not change intraday.
    """

    keys = [
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
        stock = yf.Ticker(ticker)
        stock_info = stock.get_info()

        return {
            key: stock_info.get(key)
            for key in keys
        }

    except Exception:
        return {}


# =============================================================================
# TECHNICAL ANALYSIS FUNCTIONS
# =============================================================================

def resample_prices(data, timeframe):
    """Converts daily OHLCV prices into weekly or monthly candles."""

    if data.empty:
        return data

    if timeframe == "Daily":
        return data.copy()

    if timeframe == "Weekly":
        rule = "W-FRI"
    else:
        rule = "ME"

    return (
        data.resample(rule)
        .agg(
            open=("open", "first"),
            high=("high", "max"),
            low=("low", "min"),
            close=("close", "last"),
            volume=("volume", "sum"),
        )
        .dropna()
    )


def get_pivots(values, pivot_type="high", order=3):
    """
    Finds simple local high or low pivot points.

    This is rule-based analysis, not machine-learning pattern recognition.
    """

    values = np.asarray(values, dtype=float)

    if len(values) < (2 * order) + 1:
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


def make_signal(
    pattern,
    status,
    data,
    level,
    direction,
    notes,
):
    """Builds a standard pattern record."""

    current_price = float(data["close"].iloc[-1])

    average_volume = (
        data["volume"]
        .tail(21)
        .iloc[:-1]
        .mean()
    )

    latest_volume = float(data["volume"].iloc[-1])

    volume_change = (
        (latest_volume / max(average_volume, 1)) - 1
    ) * 100

    return {
        "Pattern": pattern,
        "Status": status,
        "Direction": direction,
        "Date": data.index[-1],
        "Level": float(level),
        "Current": current_price,
        "Return %": (
            (current_price / float(level)) - 1
        ) * 100,
        "Volume %": volume_change,
        "Notes": notes,
    }


def detect_patterns(data):
    """
    Detects simplified active price structures.

    Confirmed means price has closed through a defined breakout /
    breakdown level. In progress means the geometry is visible but
    the confirmation level has not been crossed.
    """

    if data is None or len(data) < 40:
        return []

    data = data.copy().dropna()

    if len(data) < 40:
        return []

    high = data["high"].to_numpy(dtype=float)
    low = data["low"].to_numpy(dtype=float)
    close = data["close"].to_numpy(dtype=float)
    open_price = data["open"].to_numpy(dtype=float)

    results = []

    pivot_highs = get_pivots(high, "high")
    pivot_lows = get_pivots(low, "low")

    # -------------------------------------------------------------------------
    # DOUBLE TOP
    # -------------------------------------------------------------------------

    if len(pivot_highs) >= 2:
        first_top, second_top = pivot_highs[-2:]

        top_similarity = abs(
            high[first_top] - high[second_top]
        ) / max(high[first_top], 1)

        if second_top - first_top >= 8 and top_similarity < 0.045:
            neckline = float(
                np.min(low[first_top:second_top + 1])
            )

            status = (
                "Confirmed"
                if close[-1] < neckline * 0.99
                else "In progress"
            )

            results.append(
                make_signal(
                    "Double Top",
                    status,
                    data,
                    neckline,
                    "Bearish",
                    "Two comparable highs. Confirmation requires a close below the neckline.",
                )
            )

    # -------------------------------------------------------------------------
    # DOUBLE BOTTOM
    # -------------------------------------------------------------------------

    if len(pivot_lows) >= 2:
        first_bottom, second_bottom = pivot_lows[-2:]

        bottom_similarity = abs(
            low[first_bottom] - low[second_bottom]
        ) / max(low[first_bottom], 1)

        if second_bottom - first_bottom >= 8 and bottom_similarity < 0.045:
            neckline = float(
                np.max(high[first_bottom:second_bottom + 1])
            )

            status = (
                "Confirmed"
                if close[-1] > neckline * 1.01
                else "In progress"
            )

            results.append(
                make_signal(
                    "Double Bottom",
                    status,
                    data,
                    neckline,
                    "Bullish",
                    "Two comparable lows. Confirmation requires a close above the neckline.",
                )
            )

    # -------------------------------------------------------------------------
    # HEAD AND SHOULDERS
    # -------------------------------------------------------------------------

    if len(pivot_highs) >= 3:
        left_shoulder, head, right_shoulder = pivot_highs[-3:]

        shoulder_average = (
            high[left_shoulder] +
            high[right_shoulder]
        ) / 2

        left_trough = np.min(
            low[left_shoulder:head + 1]
        )

        right_trough = np.min(
            low[head:right_shoulder + 1]
        )

        neckline = float(
            min(left_trough, right_trough)
        )

        shoulders_similar = abs(
            high[left_shoulder] -
            high[right_shoulder]
        ) / max(shoulder_average, 1)

        if (
            high[head] > shoulder_average * 1.04
            and shoulders_similar < 0.10
        ):
            status = (
                "Confirmed"
                if close[-1] < neckline * 0.99
                else "In progress"
            )

            results.append(
                make_signal(
                    "Head & Shoulders",
                    status,
                    data,
                    neckline,
                    "Bearish",
                    "Confirmation requires a close below neckline support.",
                )
            )

    # -------------------------------------------------------------------------
    # INVERSE HEAD AND SHOULDERS
    # -------------------------------------------------------------------------

    if len(pivot_lows) >= 3:
        left_shoulder, head, right_shoulder = pivot_lows[-3:]

        shoulder_average = (
            low[left_shoulder] +
            low[right_shoulder]
        ) / 2

        left_peak = np.max(
            high[left_shoulder:head + 1]
        )

        right_peak = np.max(
            high[head:right_shoulder + 1]
        )

        neckline = float(
            max(left_peak, right_peak)
        )

        shoulders_similar = abs(
            low[left_shoulder] -
            low[right_shoulder]
        ) / max(shoulder_average, 1)

        if (
            low[head] < shoulder_average * 0.96
            and shoulders_similar < 0.10
        ):
            status = (
                "Confirmed"
                if close[-1] > neckline * 1.01
                else "In progress"
            )

            results.append(
                make_signal(
                    "Inverse Head & Shoulders",
                    status,
                    data,
                    neckline,
                    "Bullish",
                    "Confirmation requires a close above neckline resistance.",
                )
            )

    # -------------------------------------------------------------------------
    # RECTANGLES AND TRIANGLES
    # -------------------------------------------------------------------------

    window_size = 30

    if len(data) >= window_size:
        recent_high = high[-window_size:]
        recent_low = low[-window_size:]

        x = np.arange(window_size)

        high_slope = (
            np.polyfit(x, recent_high, 1)[0]
            / max(np.mean(recent_high), 1)
        )

        low_slope = (
            np.polyfit(x, recent_low, 1)[0]
            / max(np.mean(recent_low), 1)
        )

        resistance = float(np.max(recent_high))
        support = float(np.min(recent_low))

        channel_width = (
            resistance - support
        ) / max(resistance, 1)

        if channel_width < 0.15:
            pattern_name = None

            if abs(high_slope) < 0.001 and abs(low_slope) < 0.001:
                pattern_name = "Rectangle"

            elif abs(high_slope) < 0.0007 and low_slope > 0.0007:
                pattern_name = "Ascending Triangle"

            elif high_slope < -0.0007 and abs(low_slope) < 0.0007:
                pattern_name = "Descending Triangle"

            elif high_slope < -0.0007 and low_slope > 0.0007:
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
                    make_signal(
                        pattern_name,
                        status,
                        data,
                        level,
                        direction,
                        "Breakout confirmation requires a close outside the pattern range.",
                    )
                )

    # -------------------------------------------------------------------------
    # REVERSAL CANDLE CANDIDATES
    # -------------------------------------------------------------------------

    latest_open = open_price[-1]
    latest_close = close[-1]
    latest_high = high[-1]
    latest_low = low[-1]

    body = abs(latest_close - latest_open)
    candle_range = max(latest_high - latest_low, 0.000001)

    lower_shadow = (
        min(latest_open, latest_close) -
        latest_low
    )

    upper_shadow = (
        latest_high -
        max(latest_open, latest_close)
    )

    prior_low = np.min(low[-11:-1])
    prior_high = np.max(high[-11:-1])

    if (
        lower_shadow / candle_range > 0.55
        and body / candle_range < 0.35
        and latest_close <= prior_low * 1.04
    ):
        results.append(
            make_signal(
                "Reversal Bottom",
                "Candidate",
                data,
                latest_low,
                "Bullish",
                "Hammer-like candle near local low. Await next-candle confirmation.",
            )
        )

    if (
        upper_shadow / candle_range > 0.55
        and body / candle_range < 0.35
        and latest_close >= prior_high * 0.96
    ):
        results.append(
            make_signal(
                "Reversal Top",
                "Candidate",
                data,
                latest_high,
                "Bearish",
                "Shooting-star-like candle near local high. Await next-candle confirmation.",
            )
        )

    return results


def calculate_support_resistance(data):
    """Returns nearest simple pivot-based support and resistance."""

    if len(data) < 30:
        return (
            float(data["low"].tail(10).min()),
            float(data["high"].tail(10).max()),
        )

    current_price = float(data["close"].iloc[-1])

    pivot_lows = get_pivots(
        data["low"].to_numpy(),
        "low",
    )

    pivot_highs = get_pivots(
        data["high"].to_numpy(),
        "high",
    )

    supports = [
        float(data["low"].iloc[index])
        for index in pivot_lows
        if data["low"].iloc[index] < current_price
    ]

    resistances = [
        float(data["high"].iloc[index])
        for index in pivot_highs
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


def calculate_trend(data):
    """Returns a simple moving-average trend classification."""

    if len(data) < 55:
        return "Insufficient data"

    close = float(data["close"].iloc[-1])
    ma20 = float(data["close"].rolling(20).mean().iloc[-1])
    ma50 = float(data["close"].rolling(50).mean().iloc[-1])

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


# =============================================================================
# FUNDAMENTAL FUNCTIONS
# =============================================================================

def calculate_fundamental_score(fundamentals):
    """
    Produces a simple, transparent 0-9 fundamental score.

    This score is not suitable for banks/NBFCs without sector-specific
    metrics such as NIM, CASA, GNPA, NNPA and capital adequacy.
    """

    score = 0
    strengths = []

    roe = fundamentals.get("returnOnEquity")
    debt_equity = fundamentals.get("debtToEquity")
    profit_margin = fundamentals.get("profitMargins")
    revenue_growth = fundamentals.get("revenueGrowth")
    free_cash_flow = fundamentals.get("freeCashflow")

    if isinstance(roe, (int, float)):
        if roe >= 0.20:
            score += 2
            strengths.append("ROE is at least 20%")

        elif roe >= 0.15:
            score += 1
            strengths.append("ROE is at least 15%")

    if isinstance(debt_equity, (int, float)):
        if debt_equity < 50:
            score += 2
            strengths.append("Low debt/equity")

        elif debt_equity < 100:
            score += 1
            strengths.append("Moderate debt/equity")

    if isinstance(profit_margin, (int, float)):
        if profit_margin >= 0.15:
            score += 2
            strengths.append("Profit margin is at least 15%")

        elif profit_margin >= 0.08:
            score += 1
            strengths.append("Profit margin is at least 8%")

    if isinstance(revenue_growth, (int, float)):
        if revenue_growth >= 0.12:
            score += 1
            strengths.append("Revenue growth is at least 12%")

    if isinstance(free_cash_flow, (int, float)):
        if free_cash_flow > 0:
            score += 2
            strengths.append("Positive free cash flow")

    if score >= 7:
        label = "Excellent"

    elif score >= 5:
        label = "Good"

    elif score >= 3:
        label = "Average"

    else:
        label = "Needs review"

    return score, label, strengths


def format_number(value, percentage=False):
    """Formats fundamental values safely."""

    if not isinstance(value, (int, float)):
        return "—"

    if pd.isna(value):
        return "—"

    if percentage:
        return f"{value * 100:.1f}%"

    return f"{value:.2f}"


# =============================================================================
# CHART FUNCTION
# =============================================================================

def create_chart(data, signals, title):
    """Creates a candlestick chart with moving averages and signal levels."""

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
            line=dict(color="#2563eb", width=1.5),
        ),
        row=1,
        col=1,
    )

    figure.add_trace(
        go.Scatter(
            x=chart_data.index,
            y=chart_data["close"].rolling(50).mean(),
            name="50 MA",
            line=dict(color="#f59e0b", width=1.5),
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
        color = "#16a34a"

        if signal["Direction"] == "Bearish":
            color = "#dc2626"

        elif signal["Direction"] == "Neutral":
            color = "#f59e0b"

        figure.add_hline(
            y=signal["Level"],
            line_dash="dot",
            line_color=color,
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
        margin=dict(l=10, r=10, t=50, b=10),
    )

    return figure


# =============================================================================
# LOAD UNIVERSE ONCE
# =============================================================================

with st.spinner("Loading Nifty Total Market constituents..."):
    universe, universe_error = get_nifty_total_market_members()


# =============================================================================
# PAGE HEADER
# =============================================================================

st.markdown(
    '<div class="main-title">📊 Nifty Total Market (750) Research Dashboard</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-title">'
    'Daily, weekly and monthly technical research • '
    'fundamentals • support/resistance • pattern scanning'
    '</div>',
    unsafe_allow_html=True,
)

if universe_error:
    st.warning(universe_error)

if len(universe) < 100:
    st.error(
        "The live Nifty Total Market list is unavailable. "
        "The app is operating with the fallback stock universe."
    )


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    st.header("🔍 Research controls")

    mode = st.radio(
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
        "Price cache: 15 minutes\n\n"
        "Fundamental cache: 12 hours\n\n"
        "Universe cache: 6 hours"
    )


# =============================================================================
# STOCK RESEARCH MODE
# =============================================================================

if mode == "Stock research":
    symbols = sorted(universe["Symbol"].tolist())

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

    selected_record = universe[
        universe["Symbol"] == selected_symbol
    ].iloc[0]

    ticker = selected_record["Ticker"]
    company_name = selected_record["Company Name"]
    industry = selected_record["Industry"]

    with st.spinner(f"Fetching latest data for {selected_symbol}..."):
        daily_data = fetch_price_data(ticker, "5y")
        fundamental_data = fetch_fundamentals(ticker)

    if daily_data.empty:
        st.error(
            "No price data was returned for this stock. "
            "Try again later or click Refresh cached data."
        )
        st.stop()

    market_cap_crore = (
        fundamental_data.get("marketCap") or 0
    ) / 10000000

    score, fundamental_label, strengths = (
        calculate_fundamental_score(fundamental_data)
    )

    current_price = float(daily_data["close"].iloc[-1])

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Last close",
        f"₹{current_price:,.2f}",
    )

    col2.metric(
        "Market cap",
        (
            f"₹{market_cap_crore:,.0f} Cr"
            if market_cap_crore > 0
            else "Unavailable"
        ),
    )

    col3.metric(
        "Daily trend",
        calculate_trend(daily_data),
    )

    col4.metric(
        "Fundamental score",
        f"{score}/9 · {fundamental_label}",
    )

    st.caption(
        f"{company_name} • {industry}"
    )

    if (
        market_cap_crore > 0
        and market_cap_crore < minimum_market_cap
    ):
        st.warning(
            f"{selected_symbol} is below your current minimum market-cap "
            f"filter of ₹{minimum_market_cap:,.0f} crore."
        )

    daily_tab, weekly_tab, monthly_tab, fundamental_tab = st.tabs(
        [
            "Daily",
            "Weekly",
            "Monthly",
            "Fundamentals",
        ]
    )

    for tab, timeframe in zip(
        [daily_tab, weekly_tab, monthly_tab],
        TIMEFRAMES,
    ):
        with tab:
            timeframe_data = resample_prices(
                daily_data,
                timeframe,
            )

            detected_signals = detect_patterns(
                timeframe_data
            )

            support, resistance = calculate_support_resistance(
                timeframe_data
            )

            metric1, metric2, metric3 = st.columns(3)

            metric1.metric(
                "Overall trend",
                calculate_trend(timeframe_data),
            )

            metric2.metric(
                "Nearest support",
                f"₹{support:,.2f}",
            )

            metric3.metric(
                "Nearest resistance",
                f"₹{resistance:,.2f}",
            )

            st.subheader(
                f"{timeframe} pattern status"
            )

            if detected_signals:
                signal_table = pd.DataFrame(
                    detected_signals
                )

                st.dataframe(
                    signal_table,
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Date": st.column_config.DatetimeColumn(
                            "Date",
                            format="YYYY-MM-DD",
                        ),
                        "Level": st.column_config.NumberColumn(
                            "Breakout / Neckline",
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
                    },
                )

            else:
                st.info(
                    "No supported active or confirmed pattern is currently "
                    "detected on this timeframe."
                )

            st.plotly_chart(
                create_chart(
                    timeframe_data,
                    detected_signals,
                    f"{selected_symbol} — {timeframe}",
                ),
                use_container_width=True,
            )

    with fundamental_tab:
        st.subheader(
            f"Fundamental health: {fundamental_label} ({score}/9)"
        )

        if strengths:
            st.success(
                "Positive signals: " +
                " • ".join(strengths)
            )

        else:
            st.info(
                "There is insufficient comparable Yahoo Finance data "
                "to identify strong fundamental signals."
            )

        free_cash_flow = fundamental_data.get(
            "freeCashflow"
        )

        if isinstance(free_cash_flow, (int, float)):
            free_cash_flow_value = (
                f"₹{free_cash_flow / 10000000:,.0f} Cr"
            )
        else:
            free_cash_flow_value = "—"

        fundamentals_table = pd.DataFrame(
            [
                ["Sector", fundamental_data.get("sector", "—")],
                ["Industry", fundamental_data.get("industry", "—")],
                ["Trailing P/E", format_number(fundamental_data.get("trailingPE"))],
                ["Forward P/E", format_number(fundamental_data.get("forwardPE"))],
                ["Price / Book", format_number(fundamental_data.get("priceToBook"))],
                ["ROE", format_number(fundamental_data.get("returnOnEquity"), True)],
                ["ROA", format_number(fundamental_data.get("returnOnAssets"), True)],
                ["Profit margin", format_number(fundamental_data.get("profitMargins"), True)],
                ["Operating margin", format_number(fundamental_data.get("operatingMargins"), True)],
                ["Revenue growth", format_number(fundamental_data.get("revenueGrowth"), True)],
                ["Earnings growth", format_number(fundamental_data.get("earningsGrowth"), True)],
                ["Debt / Equity", format_number(fundamental_data.get("debtToEquity"))],
                ["Current ratio", format_number(fundamental_data.get("currentRatio"))],
                ["Free cash flow", free_cash_flow_value],
                ["Dividend yield", format_number(fundamental_data.get("dividendYield"), True)],
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
            "For banks and NBFCs, Debt/Equity and free-cash-flow comparisons "
            "are less meaningful. Evaluate CASA, NIM, GNPA, NNPA, provision "
            "coverage, capital adequacy, loan growth and deposit growth."
        )


# =============================================================================
# PATTERN SCANNER MODE
# =============================================================================

else:
    st.subheader("Nifty Total Market pattern scanner")

    selected_pattern = st.selectbox(
        "Pattern",
        PATTERN_OPTIONS,
    )

    selected_timeframe = st.selectbox(
        "Timeframe",
        TIMEFRAMES,
    )

    selected_status = st.selectbox(
        "Status",
        [
            "Any",
            "Confirmed",
            "In progress",
            "Candidate",
        ],
    )

    st.warning(
        "A first full 750-stock scan can take several minutes on free "
        "Streamlit hosting. Price data is cached for 15 minutes, so later "
        "scans should be faster."
    )

    if st.button(
        "🔎 Scan current Nifty Total Market universe",
        type="primary",
    ):
        rows = []

        progress = st.progress(0)
        status_text = st.empty()

        total_stocks = len(universe)

        for index, record in universe.iterrows():
            symbol = record["Symbol"]
            ticker = record["Ticker"]

            status_text.caption(
                f"Scanning {index + 1:,} of {total_stocks:,}: {symbol}"
            )

            try:
                stock_data = fetch_price_data(
                    ticker,
                    "5y",
                )

                if stock_data.empty:
                    continue

                timeframe_data = resample_prices(
                    stock_data,
                    selected_timeframe,
                )

                stock_signals = detect_patterns(
                    timeframe_data
                )

                matching_signals = []

                for signal in stock_signals:
                    pattern_match = (
                        selected_pattern == "Any"
                        or signal["Pattern"] == selected_pattern
                    )

                    status_match = (
                        selected_status == "Any"
                        or signal["Status"] == selected_status
                    )

                    if pattern_match and status_match:
                        matching_signals.append(signal)

                if matching_signals:
                    stock_fundamentals = fetch_fundamentals(
                        ticker
                    )

                    market_cap_crore = (
                        stock_fundamentals.get("marketCap") or 0
                    ) / 10000000

                    # Strict filter: exclude stocks if market-cap data is missing
                    # or if it is below the selected threshold.
                    if market_cap_crore < minimum_market_cap:
                        continue

                    for signal in matching_signals:
                        rows.append(
                            {
                                "Stock": symbol,
                                "Company": record["Company Name"],
                                "Industry": record["Industry"],
                                "Market Cap (Cr)": round(
                                    market_cap_crore,
                                    0,
                                ),
                                **signal,
                            }
                        )

            except Exception:
                pass

            progress.progress(
                min(
                    (index + 1) / total_stocks,
                    1.0,
                )
            )

        progress.empty()
        status_text.empty()

        st.subheader(
            f"Matching signals: {len(rows)}"
        )

        if rows:
            results = pd.DataFrame(rows)

            results = results.sort_values(
                by=[
                    "Status",
                    "Return %",
                ],
                ascending=[
                    True,
                    False,
                ],
            )

            st.dataframe(
                results,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Date": st.column_config.DatetimeColumn(
                        "Date",
                        format="YYYY-MM-DD",
                    ),
                    "Level": st.column_config.NumberColumn(
                        "Breakout / Neckline",
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
                },
            )

            csv_data = results.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Download scanner results as CSV",
                data=csv_data,
                file_name="nifty_total_market_pattern_scan.csv",
                mime="text/csv",
            )

        else:
            st.info(
                "No stocks matched the selected pattern, timeframe, status, "
                "and market-cap filters."
            )


# =============================================================================
# FOOTER
# =============================================================================

st.divider()

st.caption(
    "Data sources: Nifty Indices constituent CSV and Yahoo Finance. "
    "Pattern detection is rule-based and may generate false positives or "
    "miss valid patterns. Verify data, corporate actions, financial results, "
    "prices and investment suitability independently. "
    "This dashboard is for research and education only, not investment advice."
)