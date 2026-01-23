"""Enhanced Streamlit app for FOMC trading predictions with clear action signals."""

from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from fomc_analysis.dashboard import DashboardRepository, fetch_live_prices_for_predictions
from fomc_analysis.dashboard.prediction_verifier import PredictionVerifier, export_verification_csv
from fomc_analysis.db.session import ensure_database_schema


st.set_page_config(
    page_title="FOMC Trading Predictions",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for better styling
st.markdown("""
<style>
    .big-metric { font-size: 24px; font-weight: bold; }
    .buy-yes { background-color: #d4edda; padding: 5px 10px; border-radius: 5px; color: #155724; }
    .buy-no { background-color: #f8d7da; padding: 5px 10px; border-radius: 5px; color: #721c24; }
    .hold { background-color: #f8f9fa; padding: 5px 10px; border-radius: 5px; color: #6c757d; }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 18px;
    }

    /* Enhanced styling for better UX */
    .opportunity-card {
        border: 2px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        background-color: #ffffff;
    }
    .opportunity-card-yes {
        border-left: 5px solid #28a745;
        background-color: #f0f9f4;
    }
    .opportunity-card-no {
        border-left: 5px solid #dc3545;
        background-color: #fef5f5;
    }
    .recommendation-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 14px;
    }
    .badge-buy-yes {
        background-color: #28a745;
        color: white;
    }
    .badge-buy-no {
        background-color: #dc3545;
        color: white;
    }
    .badge-hold {
        background-color: #6c757d;
        color: white;
    }
    .edge-positive {
        color: #28a745;
        font-weight: bold;
    }
    .edge-negative {
        color: #dc3545;
        font-weight: bold;
    }
    .urgency-badge {
        display: inline-block;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: bold;
        background-color: #ffc107;
        color: #000;
    }
    .hero-metric {
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin: 10px 0;
    }
    .metric-value {
        font-size: 36px;
        font-weight: bold;
        margin: 10px 0;
    }
    .metric-label {
        font-size: 14px;
        opacity: 0.9;
    }
</style>
""", unsafe_allow_html=True)


ensure_database_schema(os.getenv("DATABASE_URL"))


RESULTS_DIR = Path("results/backtest_v3")
WORD_FREQUENCY_TIMESERIES_PATH = RESULTS_DIR / "word_frequency_timeseries.csv"
WORD_FREQUENCY_SUMMARY_PATH = RESULTS_DIR / "word_frequency_summary.csv"
CONTRACT_WORDS_PATH = Path("data/kalshi_analysis/contract_words.json")


@st.cache_data(show_spinner=False)
def load_word_frequency_artifacts(
    timeseries_path: str = str(WORD_FREQUENCY_TIMESERIES_PATH),
    summary_path: str = str(WORD_FREQUENCY_SUMMARY_PATH),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load cached mention frequency CSV outputs."""
    ts_path = Path(timeseries_path)
    summary_path = Path(summary_path)

    mention_df = pd.DataFrame()
    summary_df = pd.DataFrame()

    if ts_path.exists():
        try:
            mention_df = pd.read_csv(ts_path)
            if "meeting_date" in mention_df.columns:
                mention_df["meeting_date"] = pd.to_datetime(
                    mention_df["meeting_date"], errors="coerce"
                )
                mention_df = mention_df.dropna(subset=["meeting_date"])
        except Exception:
            mention_df = pd.DataFrame()

    if summary_path.exists():
        try:
            summary_df = pd.read_csv(summary_path)
        except Exception:
            summary_df = pd.DataFrame()

    return mention_df, summary_df


@st.cache_data(show_spinner=False)
def load_upcoming_contract_words(
    contract_words_path: str = str(CONTRACT_WORDS_PATH),
) -> list[str]:
    """Load words with at least one unresolved/active market."""
    path = Path(contract_words_path)
    if not path.exists():
        return []

    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return []

    resolved_statuses = {
        "finalized",
        "settled",
        "resolved",
        "closed",
        "expired",
        "inactive",
        "cancelled",
        "canceled",
    }

    upcoming = []
    for entry in data:
        word = entry.get("word")
        if not word:
            continue
        for market in entry.get("markets", []) or []:
            status = str(market.get("status", "")).lower()
            if not status or status not in resolved_statuses:
                upcoming.append(word)
                break

    return sorted(set(upcoming))


def render_word_frequency_section(section_key: str, predictions_df: pd.DataFrame | None = None) -> None:
    """Display interactive word frequency chart + summary."""
    mentions_df, summary_df = load_word_frequency_artifacts()
    if mentions_df.empty:
        st.info(
            "Run the backtest visualization step to generate "
            "`word_frequency_timeseries.csv` before viewing trends."
        )
        return

    available_words = [col for col in mentions_df.columns if col != "meeting_date"]
    if not available_words:
        st.info("No mention columns available in the exported CSV.")
        return

    upcoming_words = [
        word for word in load_upcoming_contract_words() if word in available_words
    ]

    st.markdown("### 🔡 Word Mention Trends")
    st.caption(
        "Historical mention counts for traded markets plus the latest upcoming Kalshi words."
    )

    default_selection = (
        list(upcoming_words[:1]) if upcoming_words else list(available_words[:3])
    )
    selector_key = f"{section_key}_word_frequency_selector"
    force_flag_key = f"{selector_key}_force_upcoming"

    if upcoming_words:
        filter_cols = st.columns([1, 3])
        with filter_cols[0]:
            if st.button(
                "Quick filter: upcoming words",
                key=f"{section_key}_show_upcoming",
                help="Display only unresolved Kalshi contracts in the chart.",
            ):
                st.session_state.pop(selector_key, None)
                st.session_state[force_flag_key] = True

    if st.session_state.pop(force_flag_key, False):
        default_selection = list(upcoming_words)

    selection = st.multiselect(
        "Select contract words to visualize",
        options=available_words,
        default=default_selection,
        help="Choose words to compare. Upcoming markets are pre-selected when available.",
        key=selector_key,
    )

    focus_word = None
    if upcoming_words:
        focus_word = st.selectbox(
            "Highlight upcoming market",
            options=upcoming_words,
            index=0,
            help="Always include this upcoming market in the chart/table.",
            key=f"{section_key}_upcoming_word_focus",
        )
        if focus_word and focus_word not in selection:
            selection = selection + [focus_word]

    selection = [word for word in dict.fromkeys(selection)]

    if not selection:
        st.warning("Select at least one contract word to plot.")
        return

    min_meeting = mentions_df["meeting_date"].min().date()
    max_meeting = mentions_df["meeting_date"].max().date()
    today = pd.Timestamp.utcnow().date()
    max_picker = max(max_meeting, today)
    date_key = f"{section_key}_date_filter"

    date_range = st.date_input(
        "Meeting date range",
        value=(min_meeting, max_meeting),
        min_value=min_meeting,
        max_value=max_picker,
        key=date_key,
        help="Restrict the timeline shown in the chart and stats table.",
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_meeting, max_meeting

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    filtered_mentions = mentions_df[
        (mentions_df["meeting_date"] >= start_ts) & (mentions_df["meeting_date"] <= end_ts)
    ].copy()

    if filtered_mentions.empty:
        st.info("No meetings fall within the selected date range.")
        return

    plot_df = filtered_mentions[["meeting_date"] + selection].copy().sort_values("meeting_date")
    chart_df = plot_df.set_index("meeting_date")[selection]

    st.line_chart(chart_df, height=360)

    stats_rows = []
    total_meetings = len(chart_df.index)
    for word in selection:
        series = chart_df[word]
        meetings_with = int((series > 0).sum())
        stats_rows.append({
            "Contract": word,
            "Meetings Evaluated": total_meetings,
            "Meetings Mentioned": meetings_with,
            "Mention Frequency (%)": (meetings_with / total_meetings * 100) if total_meetings else 0.0,
            "Mean Mentions": series.mean(),
            "Median Mentions": series.median(),
            "Std Dev": series.std(),
            "Min": int(series.min()) if not series.empty else 0,
            "Max": int(series.max()) if not series.empty else 0,
            "Total Mentions": int(series.sum()),
        })

    if stats_rows:
        stats_df = pd.DataFrame(stats_rows)
        st.markdown("#### 📋 Mention Stats")
        st.dataframe(stats_df, hide_index=True, width='stretch', height=300)

        if predictions_df is not None and not predictions_df.empty:
            preds = predictions_df.copy()
            if "contract" in preds.columns:
                preds = preds[preds["contract"].isin(selection)]
            else:
                preds = pd.DataFrame()

            if not preds.empty:
                if "prediction_date" in preds.columns:
                    preds["prediction_sort"] = pd.to_datetime(
                        preds["prediction_date"], errors="coerce"
                    )
                elif "meeting_date" in preds.columns:
                    preds["prediction_sort"] = pd.to_datetime(
                        preds["meeting_date"], errors="coerce"
                    )
                else:
                    preds["prediction_sort"] = pd.Timestamp.utcnow()

                preds = (
                    preds.sort_values("prediction_sort")
                    .groupby("contract", as_index=False)
                    .tail(1)
                )

                keep_cols = [
                    "contract",
                    "meeting_date",
                    "prediction_date",
                    "predicted_probability",
                    "market_price",
                    "edge",
                    "days_until_meeting",
                    "ticker",
                ]
                existing_cols = [col for col in keep_cols if col in preds.columns]
                pred_summary = preds[existing_cols]

                combo_df = stats_df.merge(
                    pred_summary,
                    left_on="Contract",
                    right_on="contract",
                    how="left",
                )
                combo_df = combo_df.drop(columns=["contract"], errors="ignore")

                percent_cols = ["Predicted Probability", "Market Price", "Edge"]
                rename_map = {
                    "meeting_date": "Meeting Date",
                    "prediction_date": "Prediction Date",
                    "predicted_probability": "Predicted Probability",
                    "market_price": "Market Price",
                    "edge": "Edge",
                    "days_until_meeting": "Days Until Meeting",
                    "ticker": "Ticker",
                }
                combo_df = combo_df.rename(columns=rename_map)
                combo_df["Edge Raw"] = combo_df["Edge"]

                for col in percent_cols:
                    if col in combo_df.columns:
                        if col == "Edge":
                            combo_df[col] = combo_df[col].apply(
                                lambda x: f"{x*100:+.1f}%" if pd.notna(x) else "–"
                            )
                        else:
                            combo_df[col] = combo_df[col].apply(
                                lambda x: f"{x*100:.1f}%" if pd.notna(x) else "–"
                            )

                def _trade_recommendation(edge_raw: float | None) -> str:
                    if edge_raw is None or pd.isna(edge_raw):
                        return "HOLD"
                    threshold = 0.05
                    if edge_raw >= threshold:
                        return "BUY YES"
                    if edge_raw <= -threshold:
                        return "BUY NO"
                    return "HOLD"

                combo_df["Recommendation"] = combo_df["Edge Raw"].apply(_trade_recommendation)

                def highlight_trade(row):
                    rec = row.get("Recommendation", "")
                    if rec == "BUY YES":
                        return ["background-color: #d4edda"] * len(row)
                    if rec == "BUY NO":
                        return ["background-color: #f8d7da"] * len(row)
                    return ["" for _ in row]

                display_cols = [col for col in combo_df.columns if col not in {"Recommendation", "Edge Raw"}]
                styled_combo = combo_df[display_cols].style.apply(highlight_trade, axis=1)

                st.markdown("#### 🧭 Contract Snapshot (Mentions + Predictions)")
                st.dataframe(styled_combo, hide_index=True, width='stretch', height=400)

    st.caption(
        "Source: results/backtest_v3/word_frequency_timeseries.csv "
        "and word_frequency_summary.csv"
    )


@st.cache_resource(show_spinner=False)
def get_repository() -> DashboardRepository:
    return DashboardRepository(database_url=os.getenv("DATABASE_URL"))


def format_metric(value: float | None, pct: bool = False) -> str:
    """Format numerical metrics for display."""
    if value is None:
        return "–"
    if pct:
        return f"{value * 100:.2f}%"
    return f"{value:,.2f}"


def format_edge_with_color(edge: float | None) -> str:
    """Format edge value with color coding."""
    if edge is None:
        return "–"

    edge_pct = edge * 100
    color_class = "edge-positive" if edge >= 0 else "edge-negative"
    return f'<span class="{color_class}">{edge_pct:+.1f}%</span>'


def render_recommendation_badge(recommendation: str) -> str:
    """Render a styled recommendation badge."""
    badge_classes = {
        "BUY YES": "badge-buy-yes",
        "BUY NO": "badge-buy-no",
        "HOLD": "badge-hold"
    }
    badge_class = badge_classes.get(recommendation, "badge-hold")
    return f'<span class="recommendation-badge {badge_class}">{recommendation}</span>'


def get_urgency_label(days_until: float | None) -> str | None:
    """Get urgency label based on days until meeting."""
    if days_until is None or pd.isna(days_until):
        return None

    if days_until <= 1:
        return "🔥 URGENT: Less than 1 day"
    elif days_until <= 3:
        return "⚡ Soon: 2-3 days"
    elif days_until <= 7:
        return "📅 This week"
    return None


def get_last_n_meetings_for_contract(
    contract: str,
    n_meetings: int,
    mentions_df: pd.DataFrame
) -> pd.Series:
    """Get last N meetings' mention counts for a specific contract.

    Args:
        contract: The contract/word to look up
        n_meetings: Number of recent meetings to retrieve
        mentions_df: DataFrame with meeting_date and contract columns

    Returns:
        Series with meeting dates as index and mention counts as values
    """
    if contract not in mentions_df.columns:
        return pd.Series(dtype=float)

    # Sort by date and get last N meetings
    sorted_df = mentions_df.sort_values('meeting_date', ascending=True)
    last_n = sorted_df[['meeting_date', contract]].tail(n_meetings)

    return last_n.set_index('meeting_date')[contract]


def calculate_trend(historical_mentions: pd.Series) -> tuple[str, str]:
    """Calculate trend from historical mention data.

    Args:
        historical_mentions: Series of mention counts over time

    Returns:
        Tuple of (trend_label, trend_emoji)
    """
    if len(historical_mentions) < 3:
        return "Insufficient data", "⚪"

    # Calculate simple moving average trend
    values = historical_mentions.values
    first_half_avg = values[:len(values)//2].mean()
    second_half_avg = values[len(values)//2:].mean()

    # Calculate percentage change
    if first_half_avg > 0:
        pct_change = (second_half_avg - first_half_avg) / first_half_avg
    else:
        pct_change = 1.0 if second_half_avg > 0 else 0.0

    # Determine trend based on change
    if pct_change > 0.3:
        return "Increasing", "📈"
    elif pct_change < -0.3:
        return "Decreasing", "📉"
    else:
        return "Stable", "➡️"


def get_historical_frequency(
    contract: str,
    mentions_df: pd.DataFrame,
    n_meetings: int = 6
) -> tuple[str, int, int]:
    """Get frequency string and counts for historical mentions.

    Args:
        contract: The contract/word to look up
        mentions_df: DataFrame with meeting_date and contract columns
        n_meetings: Number of recent meetings to check (default 6)

    Returns:
        Tuple of (frequency_string, mentioned_count, total_count)
        e.g., ("5/6", 5, 6)
    """
    mentions = get_last_n_meetings_for_contract(contract, n_meetings, mentions_df)

    if mentions.empty:
        return "N/A", 0, 0

    mentioned_count = int((mentions > 0).sum())
    total_count = len(mentions)

    return f"{mentioned_count}/{total_count}", mentioned_count, total_count


def calculate_confidence_score(
    row: pd.Series,
    historical_mentions: pd.Series | None = None
) -> float:
    """Calculate 0-100 confidence score for a trade recommendation.

    Score is based on:
    - Edge magnitude (40%)
    - Historical consistency (30%)
    - Model confidence interval width (20%)
    - Days until meeting (10%)

    Args:
        row: Prediction row with edge, confidence bounds, days_until_meeting
        historical_mentions: Optional series of historical mention counts

    Returns:
        Confidence score from 0-100
    """
    score = 0.0

    # Edge score (40 points max)
    edge = row.get('edge', 0)
    if pd.notna(edge):
        edge_score = min(abs(edge) / 0.3, 1.0) * 40
        score += edge_score

    # Historical consistency score (30 points max)
    if historical_mentions is not None and not historical_mentions.empty:
        mention_rate = (historical_mentions > 0).sum() / len(historical_mentions)
        score += mention_rate * 30

    # Confidence interval score (20 points max)
    # Narrower CI = higher confidence
    conf_lower = row.get('confidence_lower', 0)
    conf_upper = row.get('confidence_upper', 1)
    if pd.notna(conf_lower) and pd.notna(conf_upper):
        ci_width = conf_upper - conf_lower
        ci_score = max(0, (1 - ci_width / 0.4)) * 20
        score += ci_score

    # Urgency score (10 points max)
    days_until = row.get('days_until_meeting')
    if pd.notna(days_until):
        if days_until <= 3:
            score += 10
        elif days_until <= 7:
            score += 7
        else:
            score += 5

    return min(score, 100.0)


def get_trade_recommendation(
    edge: float | None,
    predicted_prob: float | None,
    yes_threshold: float = 0.15,
    no_threshold: float = 0.12,
    min_yes_prob: float = 0.60,
    max_no_prob: float = 0.40,
) -> tuple[str, str]:
    """
    Determine trade recommendation based on edge and probability.

    Returns:
        (recommendation, css_class) tuple
    """
    if edge is None or predicted_prob is None:
        return "HOLD", "hold"

    # BUY YES: High predicted probability + positive edge
    if edge >= yes_threshold and predicted_prob >= min_yes_prob:
        return "BUY YES", "buy-yes"

    # BUY NO: Low predicted probability + negative edge
    if edge <= -no_threshold and predicted_prob <= max_no_prob:
        return "BUY NO", "buy-no"

    return "HOLD", "hold"


def display_live_price_card(ticker: str, live_price_data: dict, compact: bool = False) -> None:
    """Display a live price card with bid/ask spread and volume."""
    if ticker not in live_price_data:
        st.warning(f"⚠️ No live price data available for {ticker}")
        return

    price = live_price_data[ticker]

    if compact:
        # Compact view - just key metrics
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "💰 Last Price",
                f"{price.last_price*100:.1f}¢" if price.last_price else "–"
            )

        with col2:
            bid_ask_spread = None
            if price.yes_bid is not None and price.yes_ask is not None:
                bid_ask_spread = (price.yes_ask - price.yes_bid) * 100
            st.metric(
                "📊 Spread",
                f"{bid_ask_spread:.1f}¢" if bid_ask_spread else "–",
                help="Bid/Ask spread"
            )

        with col3:
            st.metric(
                "📈 Volume (24h)",
                f"{price.volume_24h:,}" if price.volume_24h else "–"
            )
    else:
        # Full view
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "💰 Last Price",
                f"{price.last_price*100:.1f}¢" if price.last_price else "–"
            )

        with col2:
            bid_ask_spread = None
            if price.yes_bid is not None and price.yes_ask is not None:
                bid_ask_spread = (price.yes_ask - price.yes_bid) * 100
            spread_color = "🟢" if bid_ask_spread and bid_ask_spread < 3 else "🟡" if bid_ask_spread and bid_ask_spread < 5 else "🔴"
            st.metric(
                f"{spread_color} Spread",
                f"{bid_ask_spread:.1f}¢" if bid_ask_spread else "–",
                help="Lower spread = better liquidity"
            )

        with col3:
            st.metric(
                "📈 24h Volume",
                f"{price.volume_24h:,}" if price.volume_24h else "–"
            )

        with col4:
            st.metric(
                "🎯 Open Interest",
                f"{price.open_interest:,}" if price.open_interest else "–"
            )

        # Display bid/ask details in expandable section
        with st.expander("📋 Detailed Prices", expanded=False):
            price_cols = st.columns(2)
            with price_cols[0]:
                st.markdown("**YES Side**")
                st.markdown(f"• Bid: {price.yes_bid*100:.1f}¢" if price.yes_bid else "• Bid: –")
                st.markdown(f"• Ask: {price.yes_ask*100:.1f}¢" if price.yes_ask else "• Ask: –")

            with price_cols[1]:
                st.markdown("**NO Side**")
                st.markdown(f"• Bid: {price.no_bid*100:.1f}¢" if price.no_bid else "• Bid: –")
                st.markdown(f"• Ask: {price.no_ask*100:.1f}¢" if price.no_ask else "• Ask: –")


def display_live_prices_section(predictions_df: pd.DataFrame) -> None:
    """Display live prices for all predictions in an expandable section."""
    st.markdown("**Live Market Data from Kalshi**")

    with st.spinner("Fetching live price data from Kalshi..."):
        try:
            live_prices = fetch_live_prices_for_predictions(predictions_df)

            if not live_prices:
                st.warning("⚠️ Could not fetch live price data. Check your Kalshi API credentials.")
                return

            st.success(f"✅ Loaded live prices for {len(live_prices)} markets")

            # Show compact view option
            view_mode = st.radio(
                "View mode",
                ["Compact", "Detailed"],
                horizontal=True,
                help="Compact view shows only key metrics"
            )

            compact = view_mode == "Compact"

            # Group by meeting date
            if "meeting_date" in predictions_df.columns:
                meeting_dates = sorted(predictions_df["meeting_date"].dropna().unique())

                for meeting_date in meeting_dates:
                    meeting_predictions = predictions_df[
                        predictions_df["meeting_date"] == meeting_date
                    ]

                    with st.expander(f"📅 {meeting_date} ({len(meeting_predictions)} markets)", expanded=False):
                        for idx, row in meeting_predictions.iterrows():
                            ticker = row.get("ticker")
                            contract = row.get("contract", ticker)

                            st.markdown(f"**{contract}**")
                            display_live_price_card(ticker, live_prices, compact=compact)
                            if idx < len(meeting_predictions) - 1:
                                st.markdown("---")
            else:
                # If no meeting date, just display all
                for idx, row in predictions_df.iterrows():
                    ticker = row.get("ticker")
                    contract = row.get("contract", ticker)

                    with st.expander(f"{contract}", expanded=False):
                        display_live_price_card(ticker, live_prices, compact=compact)

        except Exception as e:
            st.error(f"❌ Error fetching live prices: {str(e)}")
            st.info("💡 Make sure your Kalshi API credentials are configured in the .env file.")


def style_prediction_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add styling and recommendation column to predictions."""
    if df.empty:
        return df

    df = df.copy()

    # Add recommendation
    df["recommendation"] = df.apply(
        lambda row: get_trade_recommendation(
            row.get("edge"),
            row.get("predicted_probability")
        )[0],
        axis=1
    )

    # Format percentages
    for col in ["predicted_probability", "confidence_lower", "confidence_upper", "market_price"]:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "–")

    if "edge" in df.columns:
        df["edge"] = df["edge"].apply(lambda x: f"{x*100:+.1f}%" if pd.notna(x) else "–")

    return df


def refresh_predictions():
    """Run prediction generation script to fetch latest data from Kalshi."""
    with st.spinner("Refreshing predictions from Kalshi... This may take a minute."):
        try:
            # Run the prediction generation
            result = subprocess.run(
                ["python", "-m", "fomc_analysis.cli", "predict-upcoming",
                 "--contract-words", "data/kalshi_analysis/contract_words.json",
                 "--output", "results/upcoming_predictions"],
                capture_output=True,
                text=True,
                timeout=120
            )

            if result.returncode == 0:
                # Load predictions into database
                load_result = subprocess.run(
                    ["python", "scripts/load_upcoming_predictions.py"],
                    capture_output=True,
                    text=True,
                    timeout=60
                )

                if load_result.returncode == 0:
                    st.success("✅ Predictions refreshed successfully!")
                    st.rerun()
                else:
                    st.error(f"Failed to load predictions: {load_result.stderr}")
            else:
                st.error(f"Failed to generate predictions: {result.stderr}")
        except subprocess.TimeoutExpired:
            st.error("Prediction refresh timed out. Please try again.")
        except Exception as e:
            st.error(f"Error refreshing predictions: {str(e)}")


# Initialize repository
repo = get_repository()

# Sidebar configuration
st.sidebar.title("⚙️ Configuration")

# Trading thresholds
st.sidebar.markdown("### 📊 Trade Thresholds")
st.sidebar.caption("Adjust these to control recommendation sensitivity")

yes_edge_threshold = st.sidebar.slider(
    "📈 BUY YES Edge Threshold",
    min_value=0.05,
    max_value=0.30,
    value=0.15,
    step=0.01,
    help="Minimum edge required to recommend buying YES. Higher = more conservative."
)

no_edge_threshold = st.sidebar.slider(
    "📉 BUY NO Edge Threshold",
    min_value=0.05,
    max_value=0.30,
    value=0.12,
    step=0.01,
    help="Minimum edge required to recommend buying NO. Higher = more conservative."
)

min_yes_prob = st.sidebar.slider(
    "🎯 Min Probability for YES",
    min_value=0.50,
    max_value=0.80,
    value=0.60,
    step=0.05,
    help="Minimum predicted probability to consider BUY YES. Higher = more conservative."
)

max_no_prob = st.sidebar.slider(
    "🎯 Max Probability for NO",
    min_value=0.20,
    max_value=0.50,
    value=0.40,
    step=0.05,
    help="Maximum predicted probability to consider BUY NO. Lower = more conservative."
)

# Show current logic
with st.sidebar.expander("ℹ️ How recommendations work", expanded=False):
    st.markdown(f"""
    **BUY YES** when:
    - Edge ≥ {yes_edge_threshold:.0%}
    - Probability ≥ {min_yes_prob:.0%}

    **BUY NO** when:
    - Edge ≤ -{no_edge_threshold:.0%}
    - Probability ≤ {max_no_prob:.0%}

    **HOLD** otherwise
    """)

st.sidebar.divider()

# Dataset selection
st.sidebar.markdown("### 📁 Data Selection")
dataset_types = repo.list_dataset_types()

if dataset_types:
    selected_type = st.sidebar.selectbox(
        "Dataset type",
        dataset_types,
        index=0,
        help="Select the type of dataset to analyze"
    )
else:
    selected_type = None

runs_df = repo.list_dataset_runs(dataset_type=selected_type)

if runs_df.empty:
    st.warning("⚠️ No dataset runs found. Load data via scripts/load_* before using the dashboard.")
    st.info("Run: `python scripts/load_upcoming_predictions.py` to load predictions.")
    st.stop()

runs_df["label"] = runs_df.apply(
    lambda row: f"{row['dataset_slug']} – {row['run_timestamp']}", axis=1
)

selected_run_label = st.sidebar.selectbox(
    "Dataset run",
    runs_df["label"].tolist(),
    index=0,
    help="Select a specific dataset run to view"
)
selected_run_id = runs_df.loc[runs_df["label"] == selected_run_label, "dataset_run_id"].iloc[0]
metadata = repo.get_dataset_run(selected_run_id)

# Show data freshness indicator
if metadata and metadata.run_timestamp:
    st.sidebar.caption(f"📅 Last updated: {metadata.run_timestamp}")

# Main header
st.title("📊 Word Mention Prediction Markets")
st.markdown("**AI-powered word mention predictions for earnings calls and FOMC meetings**")
st.caption("Identify mispriced contracts and find trading opportunities with machine learning")

# Contract type selector (top-level tabs)
contract_type_tabs = st.tabs(["💼 FOMC Speaker Words", "📞 Earnings Call Words (Coming Soon)"])

# Store the active contract type for later use
active_contract_type = "FOMC"  # Will be "Earnings" when that tab is active

# Load data
predictions_df = repo.get_predictions(selected_run_id)
trades_df = repo.get_trades(selected_run_id)
grid_df = repo.get_grid_search_results(selected_run_id)

# Determine if this is a live prediction run
is_live_run = bool(
    metadata
    and metadata.dataset_slug
    and "upcoming" in metadata.dataset_slug
) or (not predictions_df.empty and predictions_df["prediction_kind"].eq("live").all())

# FOMC Contract Type Tab (Active)
with contract_type_tabs[0]:
    st.subheader("Federal Reserve FOMC Press Conference Word Mentions")
    st.markdown("""
    Track and predict word mentions in Federal Reserve FOMC press conferences.
    Our AI model analyzes historical transcripts to predict which words Jerome Powell
    and other Fed speakers are likely to mention.
    """)

    # Create main tabs for FOMC
    if is_live_run:
        main_tabs = st.tabs(["🎯 Predictions", "📈 Training Data", "⚙️ Settings"])
    else:
        main_tabs = st.tabs(["📈 Backtest Results", "🎯 Predictions", "💼 Trades", "🔍 Grid Search"])

# PREDICTIONS TAB (Main view for live runs)
if is_live_run:
    with main_tabs[0]:
        # Header with refresh button
        header_cols = st.columns([3, 1])
        with header_cols[0]:
            st.header("🎯 Live Trading Predictions")
        with header_cols[1]:
            if st.button("🔄 Refresh", width='stretch', type="primary"):
                refresh_predictions()

        if predictions_df.empty:
            st.info("📭 No live predictions available. Click 'Refresh' to generate new ones.")
        else:
            # Load historical mention data for context
            mentions_df, summary_df = load_word_frequency_artifacts()
            # Get predictions with recommendations
            df = predictions_df.copy()

            # Calculate recommendations
            df["recommendation"] = df.apply(
                lambda row: get_trade_recommendation(
                    row.get("edge"),
                    row.get("predicted_probability"),
                    yes_edge_threshold,
                    no_edge_threshold,
                    min_yes_prob,
                    max_no_prob
                )[0],
                axis=1
            )

            # Calculate stats
            total_opportunities = len(df[df["recommendation"] != "HOLD"])
            buy_yes_count = len(df[df["recommendation"] == "BUY YES"])
            buy_no_count = len(df[df["recommendation"] == "BUY NO"])

            # Get next meeting date
            next_meeting = None
            if "meeting_date" in df.columns:
                next_meeting = df["meeting_date"].min()

            # Get days until meeting
            days_until = None
            if "days_until_meeting" in df.columns:
                days_until = df["days_until_meeting"].min()

            # Top summary metrics in colored boxes
            st.markdown("### 📊 Summary")
            metric_cols = st.columns(5)

            with metric_cols[0]:
                st.metric(
                    "💼 Total Markets",
                    len(df),
                    help="Total number of prediction markets"
                )

            with metric_cols[1]:
                st.metric(
                    "🎯 Action Items",
                    total_opportunities,
                    delta=f"{buy_yes_count} YES, {buy_no_count} NO" if total_opportunities > 0 else None,
                    delta_color="normal",
                    help="Trading opportunities based on current thresholds"
                )

            with metric_cols[2]:
                st.metric(
                    "📅 Next Meeting",
                    str(next_meeting) if next_meeting else "–",
                    help="Next FOMC meeting date"
                )

            with metric_cols[3]:
                best_edge = df["edge"].abs().max() if "edge" in df.columns else None
                st.metric(
                    "📈 Best Edge",
                    f"{best_edge*100:.1f}%" if pd.notna(best_edge) else "–",
                    help="Highest absolute edge among all predictions"
                )

            with metric_cols[4]:
                st.metric(
                    "⏰ Days Until",
                    int(days_until) if pd.notna(days_until) else "–",
                    help="Days until next FOMC meeting"
                )

            st.divider()

            # Quick Filters at the top
            st.markdown("### 🔍 Filters")

            # Smart filter presets
            st.markdown("**Quick Presets:**")
            preset_cols = st.columns(4)

            with preset_cols[0]:
                if st.button("⭐ High Confidence", help="Edge >15% & mentioned in 4+ of last 6 meetings"):
                    st.session_state['preset_filter'] = 'high_confidence'
                    st.rerun()

            with preset_cols[1]:
                if st.button("📈 Trending Up", help="Increasing mention frequency"):
                    st.session_state['preset_filter'] = 'trending_up'
                    st.rerun()

            with preset_cols[2]:
                if st.button("🔥 Urgent & Verified", help="<3 days & strong historical pattern"):
                    st.session_state['preset_filter'] = 'urgent_verified'
                    st.rerun()

            with preset_cols[3]:
                if st.button("🔄 Clear Filters", help="Reset all filters"):
                    st.session_state['preset_filter'] = None
                    st.rerun()

            st.markdown("")

            # Apply preset filters if set
            preset_filter = st.session_state.get('preset_filter', None)

            # Set default values based on preset
            default_edge = 0.15 if preset_filter == 'high_confidence' else 0.0
            default_freq = 4 if preset_filter == 'high_confidence' else (3 if preset_filter in ['trending_up', 'urgent_verified'] else 0)
            default_urgent = preset_filter == 'urgent_verified'

            filter_cols = st.columns([2, 2, 2, 2, 2])

            with filter_cols[0]:
                meeting_dates = ["All"] + sorted(df["meeting_date"].dropna().astype(str).unique())
                meeting_filter = st.selectbox("📅 Meeting Date", meeting_dates)

            with filter_cols[1]:
                recommendation_filter = st.selectbox(
                    "💡 Recommendation",
                    ["All", "BUY YES", "BUY NO", "HOLD"],
                    help="Filter by trade recommendation"
                )

            with filter_cols[2]:
                min_edge_filter = st.slider(
                    "📊 Min Absolute Edge",
                    0.0, 0.5, default_edge, 0.01,
                    help="Minimum edge threshold for filtering"
                )

            with filter_cols[3]:
                min_historical_frequency = st.slider(
                    "📖 Min Historical Freq",
                    0, 6, default_freq, 1,
                    help="Minimum times mentioned in last 6 meetings"
                )

            with filter_cols[4]:
                show_only_urgent = st.checkbox("⚡ Only urgent (<3 days)", value=default_urgent)

            # Apply filters
            filtered_df = df.copy()

            if meeting_filter != "All":
                filtered_df = filtered_df[filtered_df["meeting_date"].astype(str) == meeting_filter]

            if recommendation_filter != "All":
                filtered_df = filtered_df[filtered_df["recommendation"] == recommendation_filter]

            if min_edge_filter > 0:
                filtered_df = filtered_df[filtered_df["edge"].abs() >= min_edge_filter]

            # Historical frequency filter
            if min_historical_frequency > 0 and not mentions_df.empty:
                def meets_frequency_threshold(contract):
                    _, mentioned_count, _ = get_historical_frequency(contract, mentions_df, 6)
                    return mentioned_count >= min_historical_frequency

                filtered_df = filtered_df[filtered_df['contract'].apply(meets_frequency_threshold)]

            # Trending up filter (for preset)
            if preset_filter == 'trending_up' and not mentions_df.empty:
                def is_trending_up(contract):
                    mentions = get_last_n_meetings_for_contract(contract, 6, mentions_df)
                    if mentions.empty or len(mentions) < 3:
                        return False
                    trend_label, _ = calculate_trend(mentions)
                    return trend_label == "Increasing"

                filtered_df = filtered_df[filtered_df['contract'].apply(is_trending_up)]

            if show_only_urgent and "days_until_meeting" in filtered_df.columns:
                filtered_df = filtered_df[filtered_df["days_until_meeting"] <= 3]

            # Sort by absolute edge
            filtered_df["edge_abs"] = filtered_df["edge"].abs()
            filtered_df = filtered_df.sort_values("edge_abs", ascending=False)

            st.divider()

            # TOP OPPORTUNITIES FIRST (Most important content)
            st.markdown("### 🎯 Top Opportunities")
            top_opportunities = filtered_df[filtered_df["recommendation"] != "HOLD"].head(10)

            if top_opportunities.empty:
                st.info("💡 No strong trading opportunities at current thresholds. Try adjusting the filters or threshold settings in the sidebar.")
            else:
                # Display top opportunities in a clean card layout
                for idx, row in top_opportunities.iterrows():
                    # Determine card styling
                    card_class = "opportunity-card"
                    if row["recommendation"] == "BUY YES":
                        card_class += " opportunity-card-yes"
                        icon = "📈"
                    elif row["recommendation"] == "BUY NO":
                        card_class += " opportunity-card-no"
                        icon = "📉"
                    else:
                        icon = "➖"

                    # Get urgency label
                    urgency = get_urgency_label(row.get("days_until_meeting"))

                    # Get historical context for this contract
                    contract_name = row['contract']
                    historical_mentions = get_last_n_meetings_for_contract(
                        contract_name, 6, mentions_df
                    ) if not mentions_df.empty else pd.Series(dtype=float)

                    # Calculate trend and confidence
                    trend_label, trend_emoji = calculate_trend(historical_mentions) if not historical_mentions.empty else ("N/A", "⚪")
                    confidence_score = calculate_confidence_score(row, historical_mentions)

                    # Build the expander title with trend and confidence
                    edge_color = "edge-positive" if row["edge"] >= 0 else "edge-negative"
                    confidence_color = "🟢" if confidence_score >= 75 else "🟡" if confidence_score >= 50 else "🔴"
                    expander_title = f"{icon} **{row['recommendation']}** - {row['contract']}   {trend_emoji} {trend_label}   {confidence_color} {confidence_score:.0f}/100"

                    with st.expander(expander_title, expanded=False):
                        # Urgency badge if applicable
                        if urgency:
                            st.markdown(f'<span class="urgency-badge">{urgency}</span>', unsafe_allow_html=True)
                            st.markdown("")

                        # Key metrics in columns
                        detail_cols = st.columns(4)

                        with detail_cols[0]:
                            st.metric(
                                "Predicted Probability",
                                f"{row['predicted_probability']*100:.1f}%",
                                help="Model's predicted probability"
                            )

                        with detail_cols[1]:
                            market_price = row.get('market_price')
                            st.metric(
                                "Market Price",
                                f"{market_price*100:.1f}%" if pd.notna(market_price) else "–",
                                help="Current market price"
                            )

                        with detail_cols[2]:
                            edge_val = row['edge']
                            st.metric(
                                "Edge",
                                f"{edge_val*100:+.1f}%",
                                delta=None,
                                help="Difference between predicted probability and market price"
                            )

                        with detail_cols[3]:
                            days = row.get('days_until_meeting')
                            st.metric(
                                "Days Until",
                                int(days) if pd.notna(days) else "–",
                                help="Days until FOMC meeting"
                            )

                        # Historical Verification Panel
                        st.markdown("---")
                        st.markdown("**📊 Historical Verification (Last 6 Meetings)**")

                        if not historical_mentions.empty:
                            # Create visual timeline
                            hist_cols = st.columns(6)
                            for col_idx, (meeting_date, mention_count) in enumerate(historical_mentions.items()):
                                with hist_cols[col_idx]:
                                    # Visual indicator
                                    color_emoji = "🟢" if mention_count > 0 else "⚪"
                                    date_str = meeting_date.strftime('%m/%Y') if hasattr(meeting_date, 'strftime') else str(meeting_date)[:7]
                                    st.markdown(f"{color_emoji} **{date_str}**")
                                    st.caption(f"{int(mention_count)} times")

                            # Summary statistics
                            freq_str, mentioned_count, total_count = get_historical_frequency(
                                contract_name, mentions_df, 6
                            )
                            avg_mentions = historical_mentions.mean()

                            st.markdown("")
                            summary_cols = st.columns(3)
                            with summary_cols[0]:
                                st.markdown(f"**Frequency:** {freq_str} meetings ({mentioned_count/total_count*100:.0f}%)")
                            with summary_cols[1]:
                                st.markdown(f"**Avg Mentions:** {avg_mentions:.1f} per meeting")
                            with summary_cols[2]:
                                st.markdown(f"**Trend:** {trend_emoji} {trend_label}")
                        else:
                            st.info("No historical data available for this contract")

                        # Additional details
                        st.markdown("---")
                        detail_info_cols = st.columns(2)

                        with detail_info_cols[0]:
                            st.markdown(f"**📅 Meeting Date:** {row.get('meeting_date', '–')}")
                            conf_lower = row.get('confidence_lower', 0)
                            conf_upper = row.get('confidence_upper', 0)
                            st.markdown(f"**📊 Confidence Interval:** {conf_lower*100:.1f}% - {conf_upper*100:.1f}%")

                        with detail_info_cols[1]:
                            st.markdown(f"**🎫 Ticker:** {row.get('ticker', '–')}")
                            st.markdown(f"**📍 Market Status:** {row.get('market_status', '–')}")

            st.divider()

            # FULL PREDICTIONS TABLE
            st.markdown(f"### 📋 All Predictions ({len(filtered_df)} markets)")

            if filtered_df.empty:
                st.info("No predictions match your filters.")
            else:
                # Format for display
                display_df = filtered_df.copy()

                # Add historical context columns if data available
                if not mentions_df.empty:
                    display_df['historical_freq'] = display_df['contract'].apply(
                        lambda x: get_historical_frequency(x, mentions_df, 6)[0]
                    )
                    display_df['trend'] = display_df['contract'].apply(
                        lambda x: calculate_trend(
                            get_last_n_meetings_for_contract(x, 6, mentions_df)
                        )[1] if not get_last_n_meetings_for_contract(x, 6, mentions_df).empty else "⚪"
                    )
                    display_df['confidence'] = display_df.apply(
                        lambda row: f"{calculate_confidence_score(row, get_last_n_meetings_for_contract(row['contract'], 6, mentions_df)):.0f}",
                        axis=1
                    )

                # Select and order columns - streamlined with new columns
                display_cols = [
                    "recommendation",
                    "contract",
                    "predicted_probability",
                    "market_price",
                    "edge",
                ]

                if not mentions_df.empty:
                    display_cols.extend(["historical_freq", "trend", "confidence"])

                display_cols.extend([
                    "days_until_meeting",
                    "meeting_date",
                ])

                display_cols = [col for col in display_cols if col in display_df.columns]
                display_df = display_df[display_cols]

                # Format percentages
                for col in ["predicted_probability", "market_price"]:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(
                            lambda x: f"{x*100:.1f}%" if pd.notna(x) else "–"
                        )

                if "edge" in display_df.columns:
                    display_df["edge"] = display_df["edge"].apply(
                        lambda x: f"{x*100:+.1f}%" if pd.notna(x) else "–"
                    )

                # Rename columns for better readability
                column_renames = {
                    "recommendation": "Action",
                    "contract": "Contract",
                    "predicted_probability": "Predicted",
                    "market_price": "Market",
                    "edge": "Edge",
                    "historical_freq": "Hist Freq",
                    "trend": "Trend",
                    "confidence": "Confidence",
                    "days_until_meeting": "Days",
                    "meeting_date": "Meeting"
                }
                display_df = display_df.rename(columns=column_renames)

                # Color code recommendations
                def highlight_recommendation(row):
                    if "Action" in row:
                        if row["Action"] == "BUY YES":
                            return ["background-color: #d4edda"] * len(row)
                        elif row["Action"] == "BUY NO":
                            return ["background-color: #f8d7da"] * len(row)
                    return [""] * len(row)

                styled_df = display_df.style.apply(highlight_recommendation, axis=1)
                st.dataframe(styled_df, hide_index=True, width='stretch', height=400)

            st.divider()

            # OPENAI VERIFICATION SECTION
            st.markdown("### 🤖 AI Prediction Verification")
            st.markdown("""
            Use OpenAI to verify predictions by searching for relevant news, economic trends,
            and market signals. The AI will analyze current events and either confirm or cast doubt on the predictions.
            """)

            verify_cols = st.columns([2, 1, 1])

            with verify_cols[0]:
                openai_api_key = st.text_input(
                    "OpenAI API Key",
                    type="password",
                    help="Your OpenAI API key. Required for verification.",
                    placeholder="sk-..."
                )

            with verify_cols[1]:
                export_csv = st.checkbox(
                    "📥 Export CSV",
                    value=True,
                    help="Export predictions and word frequency data to CSV"
                )

            with verify_cols[2]:
                verify_button = st.button(
                    "🔍 Verify Predictions",
                    type="primary",
                    disabled=not openai_api_key,
                    help="Send predictions to OpenAI for verification with web search"
                )

            # Handle verification
            if verify_button and openai_api_key:
                with st.spinner("🔍 Verifying predictions with OpenAI..."):
                    try:
                        # Initialize verifier
                        verifier = PredictionVerifier(api_key=openai_api_key)

                        # Export CSV if requested
                        csv_path = None
                        if export_csv:
                            csv_path = export_verification_csv(
                                filtered_df if not filtered_df.empty else predictions_df,
                                mentions_df,
                                output_path="verification_data.csv"
                            )
                            st.success(f"✅ Exported data to {csv_path}")

                        # Prepare verification data
                        verification_data = verifier.prepare_verification_data(
                            filtered_df if not filtered_df.empty else predictions_df,
                            mentions_df,
                            summary_df
                        )

                        # Get next meeting date
                        meeting_date = str(next_meeting) if next_meeting else None

                        # Verify predictions
                        result = verifier.verify_predictions(
                            verification_data,
                            meeting_date=meeting_date
                        )

                        # Display results
                        st.markdown("#### 📊 Verification Results")

                        # Show overall assessment in an info box with confidence color
                        confidence_colors = {
                            "High": "🟢",
                            "Medium": "🟡",
                            "Low": "🔴"
                        }
                        confidence_icon = confidence_colors.get(result["confidence_level"], "⚪")

                        result_cols = st.columns([1, 1, 1])

                        with result_cols[0]:
                            st.metric(
                                "Confidence Level",
                                f"{confidence_icon} {result['confidence_level']}",
                                help="AI's confidence in the predictions"
                            )

                        with result_cols[1]:
                            st.metric(
                                "Recommendation",
                                result["recommendation"],
                                help="AI's overall recommendation"
                            )

                        with result_cols[2]:
                            st.metric(
                                "Verified At",
                                datetime.fromisoformat(result["timestamp"]).strftime("%H:%M:%S"),
                                help="When the verification was performed"
                            )

                        # Full assessment in an expander
                        with st.expander("📝 Full Analysis", expanded=True):
                            st.markdown(result["overall_assessment"])

                        # Show concerns and confirmations if available
                        if result.get("concerns"):
                            with st.expander("⚠️ Concerns & Risks"):
                                for concern in result["concerns"]:
                                    st.markdown(f"- {concern}")

                        if result.get("confirmations"):
                            with st.expander("✅ Confirmations & Support"):
                                for confirmation in result["confirmations"]:
                                    st.markdown(f"- {confirmation}")

                    except Exception as e:
                        st.error(f"❌ Verification failed: {str(e)}")
                        st.info("💡 Make sure your OpenAI API key is valid and has sufficient credits.")

            st.divider()

            render_word_frequency_section(
                section_key="live_predictions",
                predictions_df=filtered_df if not filtered_df.empty else predictions_df,
            )

            st.divider()

            # LIVE PRICES SECTION - Moved to bottom, collapsed by default
            with st.expander("💹 Live Market Prices (Click to expand)", expanded=False):
                display_live_prices_section(filtered_df if not filtered_df.empty else predictions_df)

    # Training Data Tab
    with main_tabs[1]:
        st.header("📚 Training Data & Model Info")

        if metadata:
            st.markdown("### 🔧 Model Configuration")
            meta_cols = st.columns(3)
            with meta_cols[0]:
                st.metric("Dataset", metadata.dataset_slug or "–")
            with meta_cols[1]:
                st.metric("Type", metadata.dataset_type or "–")
            with meta_cols[2]:
                st.metric("Last Updated", str(metadata.run_timestamp) if metadata.run_timestamp else "–")

            st.divider()

            if metadata.hyperparameters:
                st.markdown("### ⚙️ Hyperparameters")
                st.json(metadata.hyperparameters, expanded=False)

        st.divider()
        st.info("📖 This tab shows the model configuration and training metadata. The **Predictions** tab is where you'll find actionable trading signals.")

    # Settings Tab
    with main_tabs[2]:
        st.header("⚙️ Settings & Information")

        # Trade recommendation logic
        st.markdown("### 🎯 Trade Recommendation Logic")

        logic_cols = st.columns(2)

        with logic_cols[0]:
            st.markdown("**🟢 BUY YES** recommendations require:")
            st.markdown(f"- ✅ Predicted probability ≥ **{min_yes_prob:.0%}**")
            st.markdown(f"- ✅ Edge ≥ **{yes_edge_threshold:.0%}**")
            st.success("Buy when model predicts YES higher than market")

        with logic_cols[1]:
            st.markdown("**🔴 BUY NO** recommendations require:")
            st.markdown(f"- ✅ Predicted probability ≤ **{max_no_prob:.0%}**")
            st.markdown(f"- ✅ Edge ≤ **-{no_edge_threshold:.0%}**")
            st.error("Buy when model predicts NO higher than market")

        st.info("**ℹ️ HOLD** for everything else - when edge is insufficient or probability doesn't meet thresholds")

        st.divider()

        # About section
        st.markdown("### 📖 About This Dashboard")
        st.markdown("""
        This dashboard analyzes **FOMC press conference transcripts** to predict mention probabilities
        for Kalshi prediction market contracts. The predictions use historical data and statistical
        models to identify potential trading opportunities.

        #### 🌟 Key Features:

        - **🎯 Clear Recommendations**: BUY YES / BUY NO / HOLD signals based on edge
        - **📊 Confidence Intervals**: Uncertainty quantification for every prediction
        - **🔄 Live Data**: Real-time refresh from Kalshi API
        - **⚙️ Configurable Thresholds**: Adjust risk/reward in sidebar
        - **📈 Backtesting**: Historical validation of model performance
        - **💹 Live Prices**: Real-time market data integration

        #### 🧠 How It Works:

        1. **Historical Analysis**: Model trains on past FOMC transcripts
        2. **Probability Estimation**: Predicts word mention likelihood
        3. **Market Comparison**: Compares predictions to Kalshi prices
        4. **Edge Calculation**: Identifies mispriced contracts
        5. **Recommendations**: Suggests trades when edge exceeds thresholds

        #### 🎓 Understanding Edge:

        **Edge** = Predicted Probability - Market Price

        - **Positive Edge**: Model thinks YES is underpriced → BUY YES
        - **Negative Edge**: Model thinks NO is underpriced → BUY NO
        - **Near Zero**: Market fairly priced → HOLD
        """)

else:
    # BACKTEST VIEW
    with main_tabs[0]:
        st.header("📈 Backtest Results")

        overall = repo.get_overall_metrics(selected_run_id)
        if overall:
            st.markdown("### 📊 Overall Performance")
            metric_cols = st.columns(4)

            with metric_cols[0]:
                roi = overall.get("roi")
                st.metric(
                    "💰 ROI",
                    format_metric(roi, pct=True),
                    delta=None,
                    help="Return on Investment"
                )

            with metric_cols[1]:
                sharpe = overall.get("sharpe")
                st.metric(
                    "📈 Sharpe Ratio",
                    format_metric(sharpe),
                    delta=None,
                    help="Risk-adjusted returns"
                )

            with metric_cols[2]:
                win_rate = overall.get("win_rate")
                st.metric(
                    "🎯 Win Rate",
                    format_metric(win_rate, pct=True),
                    delta=None,
                    help="Percentage of profitable trades"
                )

            with metric_cols[3]:
                pnl = overall.get("total_pnl")
                st.metric(
                    "💵 Total PnL",
                    f"${format_metric(pnl)}",
                    delta=None,
                    help="Total profit/loss"
                )

        st.divider()

        horizon_df = repo.get_horizon_metrics(selected_run_id)
        if not horizon_df.empty:
            st.markdown("### 📅 Performance by Horizon")
            st.dataframe(horizon_df, hide_index=True, width='stretch', height=300)

    with main_tabs[1]:
        st.header("🎯 Predictions")

        filter_col1, filter_col2 = st.columns([3, 1])
        with filter_col1:
            min_edge = st.slider(
                "📊 Minimum absolute edge",
                0.0, 0.5, 0.0, 0.01,
                help="Filter predictions by minimum edge threshold"
            )

        if not predictions_df.empty:
            df = predictions_df.copy()
            if "edge" in df.columns:
                df["edge"] = pd.to_numeric(df["edge"], errors="coerce")
            else:
                df["edge"] = 0.0
            df["edge_abs"] = df["edge"].fillna(0.0).abs()
            df = df[df["edge_abs"] >= min_edge]
            df = df.sort_values("edge_abs", ascending=False).drop(columns=["edge_abs"])

            st.markdown(f"**Showing {len(df)} predictions** (filtered by edge ≥ {min_edge:.1%})")
            st.dataframe(df, hide_index=True, width='stretch', height=400)
        else:
            st.info("📭 No predictions available for this run.")

        st.divider()
        render_word_frequency_section(
            section_key="backtest_predictions",
            predictions_df=df if not df.empty else predictions_df,
        )

    with main_tabs[2]:
        st.header("💼 Trades")

        if not trades_df.empty:
            # Summary metrics first
            st.markdown("### 📊 Trade Summary")
            trade_cols = st.columns(4)

            total_pnl = trades_df["pnl"].fillna(0).sum()
            winning_trades = len(trades_df[trades_df["pnl"] > 0])
            losing_trades = len(trades_df[trades_df["pnl"] < 0])
            total_trades = len(trades_df)
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

            with trade_cols[0]:
                st.metric(
                    "💵 Total PnL",
                    f"${format_metric(total_pnl)}",
                    delta=None,
                    help="Total profit/loss across all trades"
                )

            with trade_cols[1]:
                st.metric(
                    "📈 Total Trades",
                    total_trades,
                    help="Number of trades executed"
                )

            with trade_cols[2]:
                st.metric(
                    "✅ Winning Trades",
                    f"{winning_trades}",
                    delta=f"{win_rate:.1f}% win rate",
                    help="Number of profitable trades"
                )

            with trade_cols[3]:
                st.metric(
                    "❌ Losing Trades",
                    losing_trades,
                    help="Number of unprofitable trades"
                )

            st.divider()

            # Trade details
            st.markdown("### 📋 Trade Details")
            st.dataframe(trades_df, hide_index=True, width='stretch', height=400)
        else:
            st.info("📭 No trades executed for this run.")

    with main_tabs[3]:
        st.header("🔍 Grid Search Results")

        if not grid_df.empty:
            st.markdown("### 📊 Hyperparameter Search Results")
            st.caption("Results from systematic hyperparameter optimization")
            st.dataframe(grid_df, hide_index=True, width='stretch', height=400)

            # Show best performing config if available
            if "roi" in grid_df.columns or "sharpe" in grid_df.columns:
                st.divider()
                st.markdown("### 🏆 Best Configuration")

                sort_col = "roi" if "roi" in grid_df.columns else "sharpe"
                best_row = grid_df.sort_values(sort_col, ascending=False).iloc[0]

                best_cols = st.columns(3)
                for i, (key, value) in enumerate(best_row.items()):
                    if i % 3 == 0 and i > 0:
                        best_cols = st.columns(3)
                    with best_cols[i % 3]:
                        st.metric(key, value)
        else:
            st.info("📭 No grid search results for this dataset run.")

# Earnings Call Words Tab (Coming Soon)
with contract_type_tabs[1]:
    st.subheader("📞 Earnings Call Word Mention Predictions")

    # Coming soon banner
    st.info("🚧 **Feature Under Development** 🚧")

    st.markdown("""
    ### Coming Soon: Earnings Call Word Predictions

    We're expanding our word mention prediction capabilities to include **corporate earnings calls**.
    This will allow you to predict and trade on word mentions during quarterly earnings calls
    from major public companies.

    #### 📊 How It Will Work

    Similar to our FOMC predictions, we'll analyze historical earnings call transcripts to predict:
    - **CEO & CFO Keywords**: Track mentions of strategic terms (AI, growth, innovation, etc.)
    - **Financial Terminology**: Revenue, profit, guidance, headwinds, tailwinds
    - **Industry-Specific Terms**: Sector-relevant buzzwords and metrics
    - **Sentiment Indicators**: Cautious, optimistic, challenging language patterns

    #### 🔬 Model Architecture & Differences

    While the core prediction methodology is similar to FOMC analysis, earnings call predictions
    will incorporate several unique variables:

    **Similarities to FOMC:**
    - 📝 **Transcript-based training**: Both analyze speaker transcripts
    - 🎯 **Word/phrase counting**: Same fundamental counting methodology
    - 📊 **Bayesian modeling**: Beta-binomial or similar statistical approach
    - 🔄 **Recency weighting**: Recent calls matter more than older ones

    **Key Differences:**

    1. **📅 Seasonality Effects**
       - Q1, Q2, Q3, Q4 patterns differ significantly
       - Holiday quarter (Q4) typically has different language patterns
       - Year-over-year comparisons are more relevant than sequential quarters

    2. **📈 Quarter-Specific Variables**
       - Guidance language differs between Q1-Q3 vs Q4
       - End-of-year calls include more forward-looking statements
       - Tax season (Q1) has unique terminology

    3. **🏢 Company-Specific Patterns**
       - Each company has unique communication styles
       - CEO/CFO changes affect language patterns
       - Industry context matters (tech vs retail vs finance)

    4. **📊 Performance-Dependent Language**
       - Word usage correlates with earnings beats/misses
       - Defensive language appears during downturns
       - Bullish terms increase with strong performance

    5. **🌍 Macro Events**
       - Economic conditions affect earnings language
       - Regulatory changes drive specific terminology
       - Industry disruption creates new buzzwords

    #### 🎯 Prediction Model Adaptations

    To account for these differences, the earnings call prediction model will include:

    - **Quarter indicators**: One-hot encoding for Q1/Q2/Q3/Q4
    - **Year-over-year features**: Compare same quarter across years
    - **Company embeddings**: Learn company-specific patterns
    - **Performance indicators**: Incorporate stock price movements
    - **Macro sentiment**: External economic indicator integration
    - **Sector context**: Industry-specific normalization

    #### 🛠️ Data Requirements

    - **Transcript Sources**: Public earnings call transcripts (10-Q related)
    - **Historical Data**: At least 2-3 years of quarterly calls per company
    - **Market Coverage**: Initially focusing on S&P 500 companies
    - **Kalshi Markets**: New earnings-related prediction markets

    #### 📅 Timeline

    - **Phase 1**: Data collection & preprocessing *(4-6 weeks)*
    - **Phase 2**: Model development & backtesting *(6-8 weeks)*
    - **Phase 3**: Kalshi market integration *(2-4 weeks)*
    - **Phase 4**: Dashboard integration & live predictions *(2-3 weeks)*

    #### 💡 Why This Matters

    Earnings calls are **highly predictable** in many ways, yet markets often misprice
    the probability of specific word mentions. By combining:

    - Historical transcript analysis
    - Company-specific patterns
    - Seasonal adjustments
    - Performance correlations

    We can identify mispriced contracts and generate alpha, similar to our FOMC strategy.

    #### 🔔 Stay Updated

    This feature is actively in development. Check back soon for updates, or reach out
    if you'd like to contribute to the earnings call prediction model development.

    ---

    **Questions or suggestions?** Open an issue on our GitHub repository!
    """)

    # Visual mockup section
    st.divider()
    st.subheader("📸 Preview: What the Earnings Tab Will Look Like")

    # Mock example
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Next Earnings Date", "Jan 28, 2026")
    with col2:
        st.metric("Companies Covered", "150+")
    with col3:
        st.metric("Active Markets", "Coming Soon")

    # Example prediction table (mock data)
    st.markdown("**Example: Upcoming Apple (AAPL) Q1 2026 Earnings Call Predictions**")

    mock_data = pd.DataFrame({
        "Word/Phrase": ["AI", "Services Growth", "iPhone", "Vision Pro", "China"],
        "Predicted Probability": ["92%", "78%", "95%", "45%", "62%"],
        "Market Price": ["85%", "72%", "90%", "55%", "60%"],
        "Edge": ["+7%", "+6%", "+5%", "-10%", "+2%"],
        "Recommendation": ["BUY YES", "BUY YES", "BUY YES", "BUY NO", "HOLD"],
    })

    st.dataframe(mock_data, hide_index=True, width='stretch')
