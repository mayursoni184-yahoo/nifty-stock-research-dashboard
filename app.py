import io
import requests
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# =============================================================================
# APP CONFIGURATION
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
        .title-main {
            font-size: 2.3rem;
            font-weight: 800;
            color: #111827;
            margin-bottom: 0.15rem;
        }

        .title-sub {
            color: #6b7280;
            font-size: 1rem;
            margin-bottom: 1.2rem;
        }

        .research-card {
            border: 1px solid #e5e7eb;
            border-radius: 12px;
            padding: 15px;
            background-color: white;
            margin-bottom: 12px;
        }

        .positive-card {
            border-left: 5px solid #16a34a;
        }

        .neutral-card {
            border-left: 5px solid #f59e0b;
        }

        .negative-card {
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

TREND_OPTIONS = [
    "Any",
    "Strong bullish",
    "Bullish",
    "Neutral / consolidating",
    "Bearish",
    "Insufficient data",
]

PATTERN_STATUS_OPTIONS = [
    "Any",
    "Confirmed",
    "In progress",
    "Candidate",
]

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
# GENERAL UTILITY FUNCTIONS
# =============================================================================

def safe_number(value, default=None):
    """Safely converts a value into float."""

    try:
        if value is None:
            return default

        if pd.isna(value):
            return default

        return float(value)

    except Exception:
        return default


def format_number(value, percentage=False, currency=False):
    """Formats values safely for dashboard display."""

    value = safe_number(value)

    if value is None:
        return "Not available"

    if currency:
        return f"₹{value:,.2f}"

    if percentage:
        return f"{value:.2f}%"

    return f"{value:.2f}"


def calculate_percentage_change(current_value, previous_value):
    """Calculates a percentage change without converting missing values to zero."""

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
    """Calculates CAGR where valid positive values are available."""

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
        (
            end_value / start_value
        ) ** (1 / years) - 1
    ) * 100


# =============================================================================
# NIFTY TOTAL MARKET CONSTITUENTS
# =============================================================================

@st.cache_data(ttl=21600, show_spinner=False)
def get_nifty_total_market_members():
    """
    Retrieves the Nifty Total Market constituent list.

    If external constituent data is unavailable, the dashboard uses
    a smaller fallback list instead of failing to load.
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

        universe = pd.read_csv(
            io.BytesIO(response.content)
        )

        universe.columns = [
            str(column).strip()
            for column in universe.columns
        ]

        if "Symbol" not in universe.columns:
            raise ValueError(
                "Constituent data does not contain Symbol."
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
                "Too few valid Nifty Total Market constituents loaded."
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

        warning_message = (
            "Live Nifty Total Market constituents could not be loaded. "
            f"Using {len(fallback_universe)} fallback stocks. "
            f"Reason: {type(error).__name__}: {error}"
        )

        return fallback_universe, warning_message


# =============================================================================
# MARKET DATA FUNCTIONS
# =============================================================================

@st.cache_data(ttl=900, show_spinner=False)
def fetch_price_data(ticker, period="5y"):
    """Fetches price history from Yahoo Finance."""

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
def fetch_basic_fundamentals(ticker):
    """Fetches basic Yahoo Finance summary information."""

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
        company = yf.Ticker(ticker)
        company_info = company.get_info()

        return {
            field: company_info.get(field)
            for field in fields
        }

    except Exception:
        return {}


@st.cache_data(ttl=43200, show_spinner=False)
def fetch_financial_statements(ticker):
    """
    Fetches annual and quarterly financial statements.

    Data is not forced to zero. Missing statement data remains empty
    and is handled as insufficient evidence in the dashboard.
    """

    empty_dataframe = pd.DataFrame()

    try:
        company = yf.Ticker(ticker)

        annual_income = company.income_stmt
        quarterly_income = company.quarterly_income_stmt

        annual_balance_sheet = company.balance_sheet
        quarterly_balance_sheet = company.quarterly_balance_sheet

        annual_cashflow = company.cashflow
        quarterly_cashflow = company.quarterly_cashflow

        return {
            "annual_income": (
                annual_income
                if annual_income is not None
                else empty_dataframe
            ),
            "quarterly_income": (
                quarterly_income
                if quarterly_income is not None
                else empty_dataframe
            ),
            "annual_balance_sheet": (
                annual_balance_sheet
                if annual_balance_sheet is not None
                else empty_dataframe
            ),
            "quarterly_balance_sheet": (
                quarterly_balance_sheet
                if quarterly_balance_sheet is not None
                else empty_dataframe
            ),
            "annual_cashflow": (
                annual_cashflow
                if annual_cashflow is not None
                else empty_dataframe
            ),
            "quarterly_cashflow": (
                quarterly_cashflow
                if quarterly_cashflow is not None
                else empty_dataframe
            ),
        }

    except Exception:
        return {
            "annual_income": empty_dataframe,
            "quarterly_income": empty_dataframe,
            "annual_balance_sheet": empty_dataframe,
            "quarterly_balance_sheet": empty_dataframe,
            "annual_cashflow": empty_dataframe,
            "quarterly_cashflow": empty_dataframe,
        }


# =============================================================================
# FINANCIAL STATEMENT EXTRACTION
# =============================================================================

def find_statement_row(statement, candidate_names):
    """
    Finds a financial statement row using multiple possible Yahoo labels.
    """

    if statement is None or statement.empty:
        return None

    for name in candidate_names:
        if name in statement.index:
            row = statement.loc[name]

            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]

            return pd.to_numeric(
                row,
                errors="coerce",
            )

    return None


def get_statement_value(
    statement,
    candidate_names,
    position=0,
):
    """
    Gets a value from a statement row.

    position=0 is the latest reporting period.
    position=1 is the previous period.
    position=4 is generally the same quarter one year earlier,
    if enough quarterly statement history is available.
    """

    row = find_statement_row(
        statement,
        candidate_names,
    )

    if row is None:
        return None

    clean_values = row.dropna()

    if len(clean_values) <= position:
        return None

    try:
        return float(clean_values.iloc[position])

    except Exception:
        return None


def get_statement_dates(statement):
    """Returns readable reporting dates from a financial statement."""

    if statement is None or statement.empty:
        return []

    reporting_dates = []

    for column in statement.columns:
        try:
            reporting_dates.append(
                pd.to_datetime(column).strftime(
                    "%d-%b-%Y"
                )
            )

        except Exception:
            reporting_dates.append(str(column))

    return reporting_dates


def extract_annual_financial_data(
    statements,
    yahoo_fundamentals,
):
    """Extracts reported and calculated annual metrics."""

    annual_income = statements["annual_income"]
    annual_balance = statements["annual_balance_sheet"]
    annual_cashflow = statements["annual_cashflow"]

    revenue_rows = [
        "Total Revenue",
        "Operating Revenue",
        "Revenue",
    ]

    profit_rows = [
        "Net Income",
        "Net Income Common Stockholders",
        "Net Income Applicable To Common Shares",
    ]

    ebitda_rows = [
        "EBITDA",
        "Normalized EBITDA",
    ]

    operating_income_rows = [
        "Operating Income",
        "EBIT",
    ]

    revenue_latest = get_statement_value(
        annual_income,
        revenue_rows,
        0,
    )

    revenue_previous = get_statement_value(
        annual_income,
        revenue_rows,
        1,
    )

    revenue_three_years_ago = get_statement_value(
        annual_income,
        revenue_rows,
        3,
    )

    profit_latest = get_statement_value(
        annual_income,
        profit_rows,
        0,
    )

    profit_previous = get_statement_value(
        annual_income,
        profit_rows,
        1,
    )

    profit_three_years_ago = get_statement_value(
        annual_income,
        profit_rows,
        3,
    )

    ebitda_latest = get_statement_value(
        annual_income,
        ebitda_rows,
        0,
    )

    ebitda_previous = get_statement_value(
        annual_income,
        ebitda_rows,
        1,
    )

    operating_income_latest = get_statement_value(
        annual_income,
        operating_income_rows,
        0,
    )

    operating_income_previous = get_statement_value(
        annual_income,
        operating_income_rows,
        1,
    )

    free_cashflow_latest = get_statement_value(
        annual_cashflow,
        [
            "Free Cash Flow",
        ],
        0,
    )

    operating_cashflow_latest = get_statement_value(
        annual_cashflow,
        [
            "Operating Cash Flow",
            "Total Cash From Operating Activities",
        ],
        0,
    )

    total_debt_latest = get_statement_value(
        annual_balance,
        [
            "Total Debt",
            "Long Term Debt",
        ],
        0,
    )

    total_equity_latest = get_statement_value(
        annual_balance,
        [
            "Stockholders Equity",
            "Common Stock Equity",
            "Total Equity Gross Minority Interest",
        ],
        0,
    )

    operating_margin_latest = None
    operating_margin_previous = None

    if (
        operating_income_latest is not None
        and revenue_latest is not None
        and revenue_latest != 0
    ):
        operating_margin_latest = (
            operating_income_latest
            / revenue_latest
        ) * 100

    if (
        operating_income_previous is not None
        and revenue_previous is not None
        and revenue_previous != 0
    ):
        operating_margin_previous = (
            operating_income_previous
            / revenue_previous
        ) * 100

    margin_change_bps = None

    if (
        operating_margin_latest is not None
        and operating_margin_previous is not None
    ):
        margin_change_bps = (
            operating_margin_latest
            - operating_margin_previous
        ) * 100

    debt_to_equity = None

    if (
        total_debt_latest is not None
        and total_equity_latest is not None
        and total_equity_latest != 0
    ):
        debt_to_equity = (
            total_debt_latest
            / total_equity_latest
        )

    roe = None

    yahoo_roe = safe_number(
        yahoo_fundamentals.get("returnOnEquity")
    )

    if yahoo_roe is not None:
        roe = yahoo_roe * 100

    elif (
        profit_latest is not None
        and total_equity_latest is not None
        and total_equity_latest != 0
    ):
        roe = (
            profit_latest
            / total_equity_latest
        ) * 100

    return {
        "revenue_latest": revenue_latest,
        "revenue_previous": revenue_previous,
        "profit_latest": profit_latest,
        "profit_previous": profit_previous,
        "ebitda_latest": ebitda_latest,
        "ebitda_previous": ebitda_previous,
        "revenue_yoy": calculate_percentage_change(
            revenue_latest,
            revenue_previous,
        ),
        "profit_yoy": calculate_percentage_change(
            profit_latest,
            profit_previous,
        ),
        "ebitda_yoy": calculate_percentage_change(
            ebitda_latest,
            ebitda_previous,
        ),
        "revenue_cagr_3y": calculate_cagr(
            revenue_three_years_ago,
            revenue_latest,
            3,
        ),
        "profit_cagr_3y": calculate_cagr(
            profit_three_years_ago,
            profit_latest,
            3,
        ),
        "roe": roe,
        "roce": None,
        "operating_margin": operating_margin_latest,
        "margin_trend_bps": margin_change_bps,
        "debt_to_equity": debt_to_equity,
        "free_cash_flow_positive": (
            free_cashflow_latest > 0
            if free_cashflow_latest is not None
            else None
        ),
        "free_cashflow": free_cashflow_latest,
        "operating_cashflow": operating_cashflow_latest,
        "report_dates": get_statement_dates(
            annual_income
        ),
    }


def extract_quarterly_financial_data(statements):
    """
    Extracts quarterly reported and calculated metrics.

    Quarterly YoY is only calculated when Yahoo Finance contains at least
    five available quarterly observations for a row.
    """

    quarterly_income = statements[
        "quarterly_income"
    ]

    revenue_rows = [
        "Total Revenue",
        "Operating Revenue",
        "Revenue",
    ]

    profit_rows = [
        "Net Income",
        "Net Income Common Stockholders",
        "Net Income Applicable To Common Shares",
    ]

    ebitda_rows = [
        "EBITDA",
        "Normalized EBITDA",
    ]

    operating_income_rows = [
        "Operating Income",
        "EBIT",
    ]

    revenue_latest = get_statement_value(
        quarterly_income,
        revenue_rows,
        0,
    )

    revenue_previous_quarter = get_statement_value(
        quarterly_income,
        revenue_rows,
        1,
    )

    revenue_year_ago = get_statement_value(
        quarterly_income,
        revenue_rows,
        4,
    )

    profit_latest = get_statement_value(
        quarterly_income,
        profit_rows,
        0,
    )

    profit_previous_quarter = get_statement_value(
        quarterly_income,
        profit_rows,
        1,
    )

    profit_year_ago = get_statement_value(
        quarterly_income,
        profit_rows,
        4,
    )

    ebitda_latest = get_statement_value(
        quarterly_income,
        ebitda_rows,
        0,
    )

    ebitda_previous_quarter = get_statement_value(
        quarterly_income,
        ebitda_rows,
        1,
    )

    ebitda_year_ago = get_statement_value(
        quarterly_income,
        ebitda_rows,
        4,
    )

    operating_income_latest = get_statement_value(
        quarterly_income,
        operating_income_rows,
        0,
    )

    operating_income_year_ago = get_statement_value(
        quarterly_income,
        operating_income_rows,
        4,
    )

    operating_margin_latest = None
    operating_margin_year_ago = None

    if (
        operating_income_latest is not None
        and revenue_latest is not None
        and revenue_latest != 0
    ):
        operating_margin_latest = (
            operating_income_latest
            / revenue_latest
        ) * 100

    if (
        operating_income_year_ago is not None
        and revenue_year_ago is not None
        and revenue_year_ago != 0
    ):
        operating_margin_year_ago = (
            operating_income_year_ago
            / revenue_year_ago
        ) * 100

    margin_change_bps_yoy = None

    if (
        operating_margin_latest is not None
        and operating_margin_year_ago is not None
    ):
        margin_change_bps_yoy = (
            operating_margin_latest
            - operating_margin_year_ago
        ) * 100

    return {
        "revenue_latest": revenue_latest,
        "revenue_previous_quarter": revenue_previous_quarter,
        "revenue_year_ago": revenue_year_ago,
        "profit_latest": profit_latest,
        "profit_previous_quarter": profit_previous_quarter,
        "profit_year_ago": profit_year_ago,
        "ebitda_latest": ebitda_latest,
        "ebitda_previous_quarter": ebitda_previous_quarter,
        "ebitda_year_ago": ebitda_year_ago,
        "revenue_yoy": calculate_percentage_change(
            revenue_latest,
            revenue_year_ago,
        ),
        "profit_yoy": calculate_percentage_change(
            profit_latest,
            profit_year_ago,
        ),
        "ebitda_yoy": calculate_percentage_change(
            ebitda_latest,
            ebitda_year_ago,
        ),
        "revenue_qoq": calculate_percentage_change(
            revenue_latest,
            revenue_previous_quarter,
        ),
        "profit_qoq": calculate_percentage_change(
            profit_latest,
            profit_previous_quarter,
        ),
        "ebitda_qoq": calculate_percentage_change(
            ebitda_latest,
            ebitda_previous_quarter,
        ),
        "operating_margin": operating_margin_latest,
        "margin_change_bps_yoy": margin_change_bps_yoy,
        "report_dates": get_statement_dates(
            quarterly_income
        ),
    }


# =============================================================================
# FUNDAMENTAL SCORING
# =============================================================================

def calculate_basic_fundamental_score(fundamentals):
    """Calculates a basic 0–9 Yahoo Finance quality score."""

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

    free_cashflow = safe_number(
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

    if free_cashflow is not None:
        if free_cashflow > 0:
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


def assess_annual_fundamentals(annual_data):
    """
    Assesses annual financial quality.

    Crucially, it returns Insufficient data when less than three real
    annual metrics are available. It does not call missing data weak.
    """

    score = 0
    data_points = 0
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
        data_points += 1

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
                f"Negative 3-year revenue CAGR: {revenue_cagr:.1f}%"
            )

    if profit_cagr is not None:
        data_points += 1

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
                f"Negative 3-year profit CAGR: {profit_cagr:.1f}%"
            )

    if roe is not None:
        data_points += 1

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

    if operating_margin is not None:
        data_points += 1

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
        data_points += 1

        if margin_trend_bps >= 100:
            score += 1
            positives.append(
                f"Margin expanded by {margin_trend_bps:.0f} bps"
            )

        elif margin_trend_bps <= -100:
            concerns.append(
                f"Margin contracted by {margin_trend_bps:.0f} bps"
            )

    if debt_to_equity is not None:
        data_points += 1

        if debt_to_equity <= 0.50:
            score += 1
            positives.append(
                f"Low debt/equity: {debt_to_equity:.2f}"
            )

        elif debt_to_equity >= 2:
            concerns.append(
                f"High debt/equity: {debt_to_equity:.2f}"
            )

    if free_cash_flow_positive is not None:
        data_points += 1

        if free_cash_flow_positive:
            score += 1
            positives.append(
                "Positive free cash flow"
            )

        else:
            concerns.append(
                "Negative free cash flow"
            )

    if data_points < 3:
        rating = "Insufficient data"

    elif score >= 8:
        rating = "Strong"

    elif score >= 5:
        rating = "Strong / Improving"

    elif score >= 3:
        rating = "Stable"

    elif score >= 1:
        rating = "Mixed"

    else:
        rating = "Weak"

    return {
        "score": score,
        "data_points": data_points,
        "rating": rating,
        "positives": positives,
        "concerns": concerns,
    }


def assess_quarterly_fundamentals(quarterly_data):
    """
    Assesses quarterly business momentum.

    Missing data returns Insufficient data instead of Weak.
    """

    score = 0
    data_points = 0
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

    margin_change_bps_yoy = safe_number(
        quarterly_data.get("margin_change_bps_yoy")
    )

    revenue_qoq = safe_number(
        quarterly_data.get("revenue_qoq")
    )

    profit_qoq = safe_number(
        quarterly_data.get("profit_qoq")
    )

    if revenue_yoy is not None:
        data_points += 1

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
        data_points += 1

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
        data_points += 1

        if ebitda_yoy >= 15:
            score += 1
            positives.append(
                f"Strong EBITDA growth: {ebitda_yoy:.1f}% YoY"
            )

        elif ebitda_yoy < 0:
            concerns.append(
                f"EBITDA declined: {ebitda_yoy:.1f}% YoY"
            )

    if margin_change_bps_yoy is not None:
        data_points += 1

        if margin_change_bps_yoy >= 100:
            score += 2
            positives.append(
                f"Margin expanded by {margin_change_bps_yoy:.0f} bps YoY"
            )

        elif margin_change_bps_yoy >= 25:
            score += 1
            positives.append(
                f"Margin improved by {margin_change_bps_yoy:.0f} bps YoY"
            )

        elif margin_change_bps_yoy <= -100:
            concerns.append(
                f"Margin contracted by {margin_change_bps_yoy:.0f} bps YoY"
            )

    if revenue_qoq is not None:
        data_points += 1

        if revenue_qoq > 0:
            score += 1
            positives.append(
                f"Positive sequential revenue trend: {revenue_qoq:.1f}% QoQ"
            )

        elif revenue_qoq < 0:
            concerns.append(
                f"Negative sequential revenue trend: {revenue_qoq:.1f}% QoQ"
            )

    if profit_qoq is not None:
        data_points += 1

        if profit_qoq > 0:
            score += 1
            positives.append(
                f"Positive sequential profit trend: {profit_qoq:.1f}% QoQ"
            )

        elif profit_qoq < 0:
            concerns.append(
                f"Negative sequential profit trend: {profit_qoq:.1f}% QoQ"
            )

    if data_points < 3:
        rating = "Insufficient data"

    elif score >= 7:
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
        "data_points": data_points,
        "rating": rating,
        "positives": positives,
        "concerns": concerns,
    }


def calculate_combined_fundamental_status(
    annual_assessment,
    quarterly_assessment,
):
    """
    Combines annual quality and quarterly momentum safely.
    """

    annual_rating = annual_assessment.get(
        "rating",
        "Insufficient data",
    )

    quarterly_rating = quarterly_assessment.get(
        "rating",
        "Insufficient data",
    )

    annual_score = annual_assessment.get(
        "score",
        0,
    )

    quarterly_score = quarterly_assessment.get(
        "score",
        0,
    )

    if (
        annual_rating == "Insufficient data"
        and quarterly_rating == "Insufficient data"
    ):
        return {
            "score": None,
            "rating": "Insufficient data",
        }

    if annual_rating == "Insufficient data":
        return {
            "score": quarterly_score,
            "rating": (
                "Quarterly view only / "
                f"{quarterly_rating}"
            ),
        }

    if quarterly_rating == "Insufficient data":
        return {
            "score": annual_score,
            "rating": (
                "Annual view only / "
                f"{annual_rating}"
            ),
        }

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
    """Creates an annual-to-quarterly business transition label."""

    if annual_rating == "Insufficient data":
        return "Annual evidence unavailable"

    if quarterly_rating == "Insufficient data":
        return "Quarterly evidence unavailable"

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
# TECHNICAL ANALYSIS FUNCTIONS
# =============================================================================

def resample_ohlcv(price_data, timeframe):
    """Converts daily candles into weekly or monthly candles."""

    if price_data.empty:
        return price_data

    if timeframe == "Daily":
        return price_data.copy()

    if timeframe == "Weekly":
        resample_frequency = "W-FRI"
    else:
        resample_frequency = "ME"

    return (
        price_data
        .resample(resample_frequency)
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
    """Finds local high or low pivot indexes."""

    values = np.asarray(values, dtype=float)

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
    price_data,
    level,
    direction,
    notes,
):
    """Creates a normalized technical pattern signal."""

    current_price = float(
        price_data["close"].iloc[-1]
    )

    average_volume = float(
        price_data["volume"]
        .tail(21)
        .iloc[:-1]
        .mean()
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
    Detects rule-based Double Top, Double Bottom, Head & Shoulders,
    triangles, rectangles and basic reversal-candle candidates.
    """

    if price_data is None or price_data.empty:
        return []

    price_data = price_data.dropna().copy()

    if len(price_data) < 40:
        return []

    high = price_data["high"].to_numpy(
        dtype=float
    )

    low = price_data["low"].to_numpy(
        dtype=float
    )

    close = price_data["close"].to_numpy(
        dtype=float
    )

    open_price = price_data["open"].to_numpy(
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

            pattern_status = (
                "Confirmed"
                if close[-1] < neckline * 0.99
                else "In progress"
            )

            signals.append(
                build_pattern_signal(
                    "Double Top",
                    pattern_status,
                    price_data,
                    neckline,
                    "Bearish",
                    (
                        "Two similar peaks. Confirmation needs "
                        "a close below the neckline."
                    ),
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

            pattern_status = (
                "Confirmed"
                if close[-1] > neckline * 1.01
                else "In progress"
            )

            signals.append(
                build_pattern_signal(
                    "Double Bottom",
                    pattern_status,
                    price_data,
                    neckline,
                    "Bullish",
                    (
                        "Two similar troughs. Confirmation needs "
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
            pattern_status = (
                "Confirmed"
                if close[-1] < neckline * 0.99
                else "In progress"
            )

            signals.append(
                build_pattern_signal(
                    "Head & Shoulders",
                    pattern_status,
                    price_data,
                    neckline,
                    "Bearish",
                    (
                        "Confirmation needs a close below "
                        "neckline support."
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
            pattern_status = (
                "Confirmed"
                if close[-1] > neckline * 1.01
                else "In progress"
            )

            signals.append(
                build_pattern_signal(
                    "Inverse Head & Shoulders",
                    pattern_status,
                    price_data,
                    neckline,
                    "Bullish",
                    (
                        "Confirmation needs a close above "
                        "neckline resistance."
                    ),
                )
            )

    # RECTANGLES AND TRIANGLES
    formation_window = 30

    if len(price_data) >= formation_window:
        recent_highs = high[-formation_window:]
        recent_lows = low[-formation_window:]

        x_values = np.arange(
            formation_window
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
                    pattern_status = "Confirmed"
                    direction = "Bullish"
                    breakout_level = resistance

                elif close[-1] < support * 0.99:
                    pattern_status = "Confirmed"
                    direction = "Bearish"
                    breakout_level = support

                else:
                    pattern_status = "In progress"
                    direction = "Neutral"

                    if pattern_name == "Descending Triangle":
                        breakout_level = support
                    else:
                        breakout_level = resistance

                signals.append(
                    build_pattern_signal(
                        pattern_name,
                        pattern_status,
                        price_data,
                        breakout_level,
                        direction,
                        (
                            "Confirmation needs a close outside "
                            "the defined price range."
                        ),
                    )
                )

    # REVERSAL BOTTOM
    latest_open = float(open_price[-1])
    latest_close = float(close[-1])
    latest_high = float(high[-1])
    latest_low = float(low[-1])

    body = abs(
        latest_close - latest_open
    )

    full_range = max(
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
        lower_shadow / full_range > 0.55
        and body / full_range < 0.35
        and latest_close <= previous_low * 1.04
    ):
        signals.append(
            build_pattern_signal(
                "Reversal Bottom",
                "Candidate",
                price_data,
                latest_low,
                "Bullish",
                (
                    "Hammer-like candle near a local low. "
                    "Wait for next-candle confirmation."
                ),
            )
        )

    # REVERSAL TOP
    if (
        upper_shadow / full_range > 0.55
        and body / full_range < 0.35
        and latest_close >= previous_high * 0.96
    ):
        signals.append(
            build_pattern_signal(
                "Reversal Top",
                "Candidate",
                price_data,
                latest_high,
                "Bearish",
                (
                    "Shooting-star-like candle near a local high. "
                    "Wait for next-candle confirmation."
                ),
            )
        )

    return signals


def calculate_support_resistance(price_data):
    """Calculates approximate nearest pivot-based support and resistance."""

    if len(price_data) < 30:
        return (
            float(price_data["low"].tail(10).min()),
            float(price_data["high"].tail(10).max()),
        )

    current_price = float(
        price_data["close"].iloc[-1]
    )

    low_pivots = find_pivots(
        price_data["low"].to_numpy(),
        "low",
        3,
    )

    high_pivots = find_pivots(
        price_data["high"].to_numpy(),
        "high",
        3,
    )

    support_candidates = [
        float(price_data["low"].iloc[index])
        for index in low_pivots
        if price_data["low"].iloc[index] < current_price
    ]

    resistance_candidates = [
        float(price_data["high"].iloc[index])
        for index in high_pivots
        if price_data["high"].iloc[index] > current_price
    ]

    if support_candidates:
        support = max(
            support_candidates[-8:]
        )
    else:
        support = float(
            price_data["low"].tail(20).min()
        )

    if resistance_candidates:
        resistance = min(
            resistance_candidates[-8:]
        )
    else:
        resistance = float(
            price_data["high"].tail(20).max()
        )

    return support, resistance


def calculate_overall_trend(price_data):
    """Calculates a moving-average-based overall trend."""

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
# CHART CREATION
# =============================================================================

def create_stock_chart(
    price_data,
    pattern_signals,
    chart_title,
):
    """Creates candlestick, moving-average, volume, and pattern-level chart."""

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
        title=chart_title,
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
# LOAD UNIVERSE
# =============================================================================

with st.spinner(
    "Loading Nifty Total Market constituents..."
):
    stock_universe, universe_error = (
        get_nifty_total_market_members()
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
        Technical patterns • Daily / Weekly / Monthly charts •
        Annual and Quarterly financial evidence • Sector KPI framework
    </div>
    """,
    unsafe_allow_html=True,
)

if universe_error:
    st.warning(universe_error)

if stock_universe.empty:
    st.error(
        "No stock universe is currently available. "
        "Click Refresh cached data and try again."
    )
    st.stop()


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
        f"Loaded stock universe: {len(stock_universe)} stocks"
    )

    if st.button("🔄 Refresh cached data"):
        st.cache_data.clear()
        st.rerun()

    st.divider()

    st.caption(
        "Price cache: 15 minutes\n\n"
        "Basic fundamentals cache: 12 hours\n\n"
        "Financial statements cache: 12 hours\n\n"
        "Constituents cache: 6 hours"
    )


# =============================================================================
# STOCK RESEARCH MODE
# =============================================================================

if dashboard_mode == "Stock research":
    available_symbols = sorted(
        stock_universe["Symbol"].tolist()
    )

    default_index = 0

    if "RELIANCE" in available_symbols:
        default_index = available_symbols.index(
            "RELIANCE"
        )

    selected_symbol = st.selectbox(
        "Search a Nifty Total Market stock",
        available_symbols,
        index=default_index,
    )

    selected_record = stock_universe[
        stock_universe["Symbol"] == selected_symbol
    ].iloc[0]

    selected_ticker = selected_record["Ticker"]
    company_name = selected_record["Company Name"]
    industry = selected_record["Industry"]

    with st.spinner(
        f"Fetching latest research data for {selected_symbol}..."
    ):
        daily_prices = fetch_price_data(
            selected_ticker,
            "5y",
        )

        basic_fundamentals = fetch_basic_fundamentals(
            selected_ticker
        )

        financial_statements = fetch_financial_statements(
            selected_ticker
        )

    if daily_prices.empty:
        st.error(
            "Price data was not available for this stock. "
            "Please try again later."
        )
        st.stop()

    market_cap_crore = (
        basic_fundamentals.get("marketCap") or 0
    ) / 10000000

    basic_score, basic_rating, basic_strengths, basic_concerns = (
        calculate_basic_fundamental_score(
            basic_fundamentals
        )
    )

    annual_data = extract_annual_financial_data(
        financial_statements,
        basic_fundamentals,
    )

    quarterly_data = extract_quarterly_financial_data(
        financial_statements
    )

    annual_assessment = assess_annual_fundamentals(
        annual_data
    )

    quarterly_assessment = assess_quarterly_fundamentals(
        quarterly_data
    )

    combined_assessment = calculate_combined_fundamental_status(
        annual_assessment,
        quarterly_assessment,
    )

    transition = determine_fundamental_transition(
        annual_assessment["rating"],
        quarterly_assessment["rating"],
    )

    current_close = float(
        daily_prices["close"].iloc[-1]
    )

    metric_col1, metric_col2, metric_col3, metric_col4 = (
        st.columns(4)
    )

    metric_col1.metric(
        "Last Close",
        f"₹{current_close:,.2f}",
    )

    metric_col2.metric(
        "Market Cap",
        (
            f"₹{market_cap_crore:,.0f} Cr"
            if market_cap_crore > 0
            else "Not available"
        ),
    )

    metric_col3.metric(
        "Daily Trend",
        calculate_overall_trend(daily_prices),
    )

    metric_col4.metric(
        "Basic Fundamental Score",
        f"{basic_score}/9 · {basic_rating}",
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
            f"₹{minimum_market_cap:,.0f} crore market-cap filter."
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

    for tab, timeframe in [
        (daily_tab, "Daily"),
        (weekly_tab, "Weekly"),
        (monthly_tab, "Monthly"),
    ]:
        with tab:
            timeframe_prices = resample_ohlcv(
                daily_prices,
                timeframe,
            )

            signals = detect_patterns(
                timeframe_prices
            )

            support, resistance = calculate_support_resistance(
                timeframe_prices
            )

            trend_metric, support_metric, resistance_metric = (
                st.columns(3)
            )

            trend_metric.metric(
                "Overall Trend",
                calculate_overall_trend(
                    timeframe_prices
                ),
            )

            support_metric.metric(
                "Nearest Support",
                f"₹{support:,.2f}",
            )

            resistance_metric.metric(
                "Nearest Resistance",
                f"₹{resistance:,.2f}",
            )

            st.subheader(
                f"{timeframe} Pattern Status"
            )

            if signals:
                signals_dataframe = pd.DataFrame(
                    signals
                )

                st.dataframe(
                    signals_dataframe,
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
                    "No supported active pattern was found "
                    "on this timeframe."
                )

            st.plotly_chart(
                create_stock_chart(
                    timeframe_prices,
                    signals,
                    f"{selected_symbol} — {timeframe}",
                ),
                use_container_width=True,
            )

    # =========================================================================
    # FUNDAMENTALS TAB
    # =========================================================================

    with fundamentals_tab:
        st.subheader("Basic Fundamental Summary")

        basic_left, basic_right = st.columns(2)

        with basic_left:
            st.metric(
                "Basic Fundamental Score",
                f"{basic_score}/9",
            )

            if basic_strengths:
                st.markdown("#### Positive indicators")

                for strength in basic_strengths:
                    st.success(f"✅ {strength}")

        with basic_right:
            if basic_concerns:
                st.markdown("#### Risk indicators")

                for concern in basic_concerns:
                    st.warning(f"⚠️ {concern}")

        free_cashflow = basic_fundamentals.get(
            "freeCashflow"
        )

        free_cashflow_display = (
            f"₹{free_cashflow / 10000000:,.0f} Cr"
            if isinstance(free_cashflow, (int, float))
            else "Not available"
        )

        basic_metrics_dataframe = pd.DataFrame(
            [
                [
                    "Sector",
                    basic_fundamentals.get(
                        "sector",
                        "Not available",
                    ),
                ],
                [
                    "Industry",
                    basic_fundamentals.get(
                        "industry",
                        "Not available",
                    ),
                ],
                [
                    "Trailing P/E",
                    format_number(
                        basic_fundamentals.get(
                            "trailingPE"
                        )
                    ),
                ],
                [
                    "Forward P/E",
                    format_number(
                        basic_fundamentals.get(
                            "forwardPE"
                        )
                    ),
                ],
                [
                    "Price / Book",
                    format_number(
                        basic_fundamentals.get(
                            "priceToBook"
                        )
                    ),
                ],
                [
                    "ROE",
                    format_number(
                        safe_number(
                            basic_fundamentals.get(
                                "returnOnEquity"
                            )
                        )
                        * 100
                        if safe_number(
                            basic_fundamentals.get(
                                "returnOnEquity"
                            )
                        )
                        is not None
                        else None,
                        percentage=True,
                    ),
                ],
                [
                    "ROA",
                    format_number(
                        safe_number(
                            basic_fundamentals.get(
                                "returnOnAssets"
                            )
                        )
                        * 100
                        if safe_number(
                            basic_fundamentals.get(
                                "returnOnAssets"
                            )
                        )
                        is not None
                        else None,
                        percentage=True,
                    ),
                ],
                [
                    "Profit Margin",
                    format_number(
                        safe_number(
                            basic_fundamentals.get(
                                "profitMargins"
                            )
                        )
                        * 100
                        if safe_number(
                            basic_fundamentals.get(
                                "profitMargins"
                            )
                        )
                        is not None
                        else None,
                        percentage=True,
                    ),
                ],
                [
                    "Operating Margin",
                    format_number(
                        safe_number(
                            basic_fundamentals.get(
                                "operatingMargins"
                            )
                        )
                        * 100
                        if safe_number(
                            basic_fundamentals.get(
                                "operatingMargins"
                            )
                        )
                        is not None
                        else None,
                        percentage=True,
                    ),
                ],
                [
                    "Revenue Growth",
                    format_number(
                        safe_number(
                            basic_fundamentals.get(
                                "revenueGrowth"
                            )
                        )
                        * 100
                        if safe_number(
                            basic_fundamentals.get(
                                "revenueGrowth"
                            )
                        )
                        is not None
                        else None,
                        percentage=True,
                    ),
                ],
                [
                    "Earnings Growth",
                    format_number(
                        safe_number(
                            basic_fundamentals.get(
                                "earningsGrowth"
                            )
                        )
                        * 100
                        if safe_number(
                            basic_fundamentals.get(
                                "earningsGrowth"
                            )
                        )
                        is not None
                        else None,
                        percentage=True,
                    ),
                ],
                [
                    "Debt / Equity",
                    format_number(
                        basic_fundamentals.get(
                            "debtToEquity"
                        )
                    ),
                ],
                [
                    "Current Ratio",
                    format_number(
                        basic_fundamentals.get(
                            "currentRatio"
                        )
                    ),
                ],
                [
                    "Free Cash Flow",
                    free_cashflow_display,
                ],
                [
                    "Dividend Yield",
                    format_number(
                        safe_number(
                            basic_fundamentals.get(
                                "dividendYield"
                            )
                        )
                        * 100
                        if safe_number(
                            basic_fundamentals.get(
                                "dividendYield"
                            )
                        )
                        is not None
                        else None,
                        percentage=True,
                    ),
                ],
            ],
            columns=[
                "Metric",
                "Value",
            ],
        )

        st.markdown("### Basic Fundamental Metrics")

        st.dataframe(
            basic_metrics_dataframe,
            hide_index=True,
            use_container_width=True,
        )

        st.divider()

        # ---------------------------------------------------------------------
        # FUNDAMENTAL MOMENTUM CARDS
        # ---------------------------------------------------------------------

        st.subheader(
            "📈 Annual and Quarterly Fundamental Momentum"
        )

        annual_col, quarterly_col = st.columns(2)

        with annual_col:
            st.markdown(
                f"""
                <div class="research-card">
                    <h4>Annual</h4>
                    <h2>{annual_assessment["rating"]}</h2>
                    <p>Evidence points: {annual_assessment["data_points"]}</p>
                    <p>Score: {annual_assessment["score"]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with quarterly_col:
            st.markdown(
                f"""
                <div class="research-card">
                    <h4>Quarterly</h4>
                    <h2>{quarterly_assessment["rating"]}</h2>
                    <p>Evidence points: {quarterly_assessment["data_points"]}</p>
                    <p>Score: {quarterly_assessment["score"]}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        combined_col, transition_col = st.columns(2)

        with combined_col:
            st.markdown(
                f"""
                <div class="research-card positive-card">
                    <h4>Combined</h4>
                    <h2>{combined_assessment["rating"]}</h2>
                    <p>
                        Combined score:
                        {
                            combined_assessment["score"]
                            if combined_assessment["score"] is not None
                            else "Not available"
                        }
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with transition_col:
            st.markdown(
                f"""
                <div class="research-card neutral-card">
                    <h4>Transition</h4>
                    <h2>{transition}</h2>
                    <p>Annual quality compared with current quarterly momentum.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # ---------------------------------------------------------------------
        # ANNUAL EVIDENCE
        # ---------------------------------------------------------------------

        st.markdown("### Annual Evidence")

        annual_evidence = pd.DataFrame(
            [
                [
                    "Latest annual revenue",
                    annual_data.get("revenue_latest"),
                    "Reported",
                ],
                [
                    "Previous annual revenue",
                    annual_data.get("revenue_previous"),
                    "Reported",
                ],
                [
                    "Annual revenue growth",
                    annual_data.get("revenue_yoy"),
                    "Calculated",
                ],
                [
                    "3-year revenue CAGR",
                    annual_data.get("revenue_cagr_3y"),
                    "Calculated",
                ],
                [
                    "Latest annual profit",
                    annual_data.get("profit_latest"),
                    "Reported",
                ],
                [
                    "Previous annual profit",
                    annual_data.get("profit_previous"),
                    "Reported",
                ],
                [
                    "Annual profit growth",
                    annual_data.get("profit_yoy"),
                    "Calculated",
                ],
                [
                    "3-year profit CAGR",
                    annual_data.get("profit_cagr_3y"),
                    "Calculated",
                ],
                [
                    "Latest annual EBITDA",
                    annual_data.get("ebitda_latest"),
                    "Reported",
                ],
                [
                    "Annual EBITDA growth",
                    annual_data.get("ebitda_yoy"),
                    "Calculated",
                ],
                [
                    "ROE",
                    annual_data.get("roe"),
                    "Reported / Calculated",
                ],
                [
                    "Operating margin",
                    annual_data.get("operating_margin"),
                    "Calculated",
                ],
                [
                    "Operating margin change",
                    annual_data.get("margin_trend_bps"),
                    "Calculated",
                ],
                [
                    "Debt / Equity",
                    annual_data.get("debt_to_equity"),
                    "Calculated",
                ],
                [
                    "Free cash flow positive",
                    annual_data.get("free_cash_flow_positive"),
                    "Reported",
                ],
            ],
            columns=[
                "Metric",
                "Value",
                "Data Type",
            ],
        )

        annual_evidence["Value"] = (
            annual_evidence["Value"]
            .apply(
                lambda value: (
                    "Not available"
                    if value is None or pd.isna(value)
                    else value
                )
            )
        )

        st.dataframe(
            annual_evidence,
            hide_index=True,
            use_container_width=True,
        )

        annual_dates = annual_data.get(
            "report_dates",
            [],
        )

        if annual_dates:
            st.caption(
                "Annual reporting periods available: " +
                " | ".join(annual_dates)
            )

        if annual_assessment["positives"]:
            st.markdown("#### Annual positives")

            for positive in annual_assessment["positives"]:
                st.success(f"✅ {positive}")

        if annual_assessment["concerns"]:
            st.markdown("#### Annual concerns")

            for concern in annual_assessment["concerns"]:
                st.warning(f"⚠️ {concern}")

        if annual_assessment["rating"] == "Insufficient data":
            st.info(
                "The available annual statement data is incomplete. "
                "The dashboard will not label the company weak until "
                "enough real financial evidence is available."
            )

        # ---------------------------------------------------------------------
        # QUARTERLY EVIDENCE
        # ---------------------------------------------------------------------

        st.markdown("### Quarterly Evidence")

        quarterly_evidence = pd.DataFrame(
            [
                [
                    "Latest quarterly revenue",
                    quarterly_data.get("revenue_latest"),
                    "Reported",
                ],
                [
                    "Revenue growth YoY",
                    quarterly_data.get("revenue_yoy"),
                    "Calculated",
                ],
                [
                    "Revenue growth QoQ",
                    quarterly_data.get("revenue_qoq"),
                    "Calculated",
                ],
                [
                    "Latest quarterly profit",
                    quarterly_data.get("profit_latest"),
                    "Reported",
                ],
                [
                    "Profit growth YoY",
                    quarterly_data.get("profit_yoy"),
                    "Calculated",
                ],
                [
                    "Profit growth QoQ",
                    quarterly_data.get("profit_qoq"),
                    "Calculated",
                ],
                [
                    "Latest quarterly EBITDA",
                    quarterly_data.get("ebitda_latest"),
                    "Reported",
                ],
                [
                    "EBITDA growth YoY",
                    quarterly_data.get("ebitda_yoy"),
                    "Calculated",
                ],
                [
                    "EBITDA growth QoQ",
                    quarterly_data.get("ebitda_qoq"),
                    "Calculated",
                ],
                [
                    "Operating margin",
                    quarterly_data.get("operating_margin"),
                    "Calculated",
                ],
                [
                    "Operating margin change YoY",
                    quarterly_data.get(
                        "margin_change_bps_yoy"
                    ),
                    "Calculated",
                ],
            ],
            columns=[
                "Metric",
                "Value",
                "Data Type",
            ],
        )

        quarterly_evidence["Value"] = (
            quarterly_evidence["Value"]
            .apply(
                lambda value: (
                    "Not available"
                    if value is None or pd.isna(value)
                    else value
                )
            )
        )

        st.dataframe(
            quarterly_evidence,
            hide_index=True,
            use_container_width=True,
        )

        quarterly_dates = quarterly_data.get(
            "report_dates",
            [],
        )

        if quarterly_dates:
            st.caption(
                "Quarterly reporting periods available: " +
                " | ".join(quarterly_dates)
            )

        if quarterly_assessment["positives"]:
            st.markdown("#### Quarterly positives")

            for positive in quarterly_assessment["positives"]:
                st.success(f"✅ {positive}")

        if quarterly_assessment["concerns"]:
            st.markdown("#### Quarterly concerns")

            for concern in quarterly_assessment["concerns"]:
                st.warning(f"⚠️ {concern}")

        if quarterly_assessment["rating"] == "Insufficient data":
            st.info(
                "Quarterly history is incomplete or unavailable from the "
                "current data source. This is not treated as weak performance."
            )

        # ---------------------------------------------------------------------
        # SECTOR KPI FRAMEWORK
        # ---------------------------------------------------------------------

        st.divider()

        st.subheader("🏭 Sector-Specific KPI Framework")

        selected_kpi_template = st.selectbox(
            "Select a sector KPI template",
            sorted(
                SECTOR_KPI_LIBRARY.keys()
            ),
        )

        selected_kpis = SECTOR_KPI_LIBRARY[
            selected_kpi_template
        ]

        sector_kpi_dataframe = pd.DataFrame(
            {
                "Sector KPI": selected_kpis,
                "Latest Value": [
                    "Not captured"
                ] * len(selected_kpis),
                "YoY Change": [
                    "Not captured"
                ] * len(selected_kpis),
                "QoQ Change": [
                    "Not captured"
                ] * len(selected_kpis),
                "Type": [
                    "Reported or Calculated"
                ] * len(selected_kpis),
                "Expected Source": [
                    "NSE filing / company presentation"
                ] * len(selected_kpis),
            }
        )

        st.dataframe(
            sector_kpi_dataframe,
            hide_index=True,
            use_container_width=True,
        )

        st.info(
            "Reported KPIs include order book, ARPU, CASA, VNB, "
            "TCV, attrition, utilization, pre-sales, GRM and volumes. "
            "Calculated KPIs include growth rates, margin changes, "
            "book-to-bill ratio, cash conversion and debt reduction."
        )

        st.warning(
            "Banks and NBFCs must be assessed with sector-specific metrics "
            "such as NIM, GNPA, NNPA, credit cost, CASA, loan growth, "
            "deposit growth, ROA, ROE, capital adequacy and provision coverage."
        )


# =============================================================================
# PATTERN SCANNER MODE
# =============================================================================

else:
    st.subheader("Nifty Total Market Pattern Scanner")

    st.caption(
        "Scan the Nifty Total Market universe by pattern, timeframe, "
        "overall trend, signal status, and minimum market cap."
    )

    scanner_col1, scanner_col2 = st.columns(2)

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

    with scanner_col2:
        scanner_trend = st.selectbox(
            "Overall Trend",
            TREND_OPTIONS,
        )

        scanner_status = st.selectbox(
            "Pattern Status",
            PATTERN_STATUS_OPTIONS,
        )

    st.info(
        "When Timeframe is Any, the scanner checks Daily, Weekly and "
        "Monthly charts. A stock can appear multiple times if it matches "
        "your criteria on more than one chart timeframe."
    )

    st.warning(
        "A full 750-stock scan can take several minutes on free Streamlit "
        "Cloud hosting. Any timeframe requires Daily, Weekly and Monthly "
        "analysis for each stock."
    )

    if st.button(
        "🔎 Scan Nifty Total Market",
        type="primary",
    ):
        results = []

        if scanner_timeframe == "Any":
            selected_timeframes = [
                "Daily",
                "Weekly",
                "Monthly",
            ]
        else:
            selected_timeframes = [
                scanner_timeframe
            ]

        progress_bar = st.progress(0)
        progress_text = st.empty()

        stock_count = len(stock_universe)

        for index, record in stock_universe.iterrows():
            symbol = record["Symbol"]
            ticker = record["Ticker"]

            progress_text.caption(
                f"Scanning {index + 1:,} of {stock_count:,}: "
                f"{symbol}"
            )

            try:
                stock_prices = fetch_price_data(
                    ticker,
                    "5y",
                )

                if stock_prices.empty:
                    continue

                technical_matches = []

                for selected_timeframe in selected_timeframes:
                    timeframe_prices = resample_ohlcv(
                        stock_prices,
                        selected_timeframe,
                    )

                    if timeframe_prices.empty:
                        continue

                    overall_trend = calculate_overall_trend(
                        timeframe_prices
                    )

                    if (
                        scanner_trend != "Any"
                        and overall_trend != scanner_trend
                    ):
                        continue

                    detected_patterns = detect_patterns(
                        timeframe_prices
                    )

                    for pattern_signal in detected_patterns:
                        pattern_matches = (
                            scanner_pattern == "Any"
                            or pattern_signal["Pattern"]
                            == scanner_pattern
                        )

                        status_matches = (
                            scanner_status == "Any"
                            or pattern_signal["Status"]
                            == scanner_status
                        )

                        if pattern_matches and status_matches:
                            technical_matches.append(
                                {
                                    "Timeframe": selected_timeframe,
                                    "Overall Trend": overall_trend,
                                    **pattern_signal,
                                }
                            )

                if technical_matches:
                    scanner_fundamentals = fetch_basic_fundamentals(
                        ticker
                    )

                    scanner_market_cap = (
                        scanner_fundamentals.get(
                            "marketCap"
                        ) or 0
                    ) / 10000000

                    if scanner_market_cap < minimum_market_cap:
                        continue

                    for match in technical_matches:
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
                                    scanner_market_cap,
                                    0,
                                ),
                                **match,
                            }
                        )

            except Exception:
                pass

            progress_bar.progress(
                min(
                    (index + 1) / stock_count,
                    1.0,
                )
            )

        progress_bar.empty()
        progress_text.empty()

        st.subheader(
            f"Matching Signals: {len(results)}"
        )

        if results:
            result_dataframe = pd.DataFrame(results)

            signal_priority = {
                "Confirmed": 1,
                "In progress": 2,
                "Candidate": 3,
            }

            result_dataframe["Priority"] = (
                result_dataframe["Status"]
                .map(signal_priority)
                .fillna(99)
            )

            result_dataframe = (
                result_dataframe
                .sort_values(
                    by=[
                        "Priority",
                        "Return %",
                    ],
                    ascending=[
                        True,
                        False,
                    ],
                )
                .drop(columns=["Priority"])
            )

            total_col, confirmed_col, trend_col, bullish_col = (
                st.columns(4)
            )

            total_col.metric(
                "Total Matches",
                len(result_dataframe),
            )

            confirmed_col.metric(
                "Confirmed Signals",
                int(
                    (
                        result_dataframe["Status"]
                        == "Confirmed"
                    ).sum()
                ),
            )

            trend_col.metric(
                "Strong Bullish",
                int(
                    (
                        result_dataframe["Overall Trend"]
                        == "Strong bullish"
                    ).sum()
                ),
            )

            bullish_col.metric(
                "Bullish Direction",
                int(
                    (
                        result_dataframe["Direction"]
                        == "Bullish"
                    ).sum()
                ),
            )

            st.dataframe(
                result_dataframe,
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
                    "Market Cap (Cr)": st.column_config.NumberColumn(
                        "Market Cap",
                        format="₹%d Cr",
                    ),
                },
            )

            csv_export = result_dataframe.to_csv(
                index=False
            ).encode("utf-8")

            st.download_button(
                "⬇️ Download Scan Results as CSV",
                data=csv_export,
                file_name="nifty_total_market_pattern_scan.csv",
                mime="text/csv",
            )

        else:
            st.info(
                "No stocks matched the selected Pattern, Timeframe, "
                "Trend, Status and Market Cap filters."
            )


# =============================================================================
# FOOTER
# =============================================================================

st.divider()

st.caption(
    "Data sources: Nifty Indices constituent list and Yahoo Finance. "
    "Financial statement availability varies by stock. Sector-specific "
    "KPIs require NSE filings, company quarterly presentations, annual "
    "reports, or a dedicated Indian-market data provider. Technical "
    "patterns are rule-based and may produce false positives. "
    "This dashboard is for research and educational purposes only, "
    "not investment advice."
)
