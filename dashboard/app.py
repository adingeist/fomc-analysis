"""Enhanced Streamlit app for FOMC trading predictions with clear action signals."""

from __future__ import annotations

import os
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from fomc_analysis.dashboard import DashboardRepository, fetch_live_prices_for_predictions


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
</style>
""", unsafe_allow_html=True)


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


def display_live_price_card(ticker: str, live_price_data: dict) -> None:
    """Display a live price card with bid/ask spread and volume."""
    if ticker not in live_price_data:
        st.warning(f"⚠️ No live price data available for {ticker}")
        return

    price = live_price_data[ticker]

    # Create columns for price display
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Last Price",
            f"{price.last_price*100:.1f}¢" if price.last_price else "–",
            delta=None,
        )

    with col2:
        bid_ask_spread = None
        if price.yes_bid is not None and price.yes_ask is not None:
            bid_ask_spread = (price.yes_ask - price.yes_bid) * 100
        st.metric(
            "Bid/Ask Spread",
            f"{bid_ask_spread:.1f}¢" if bid_ask_spread else "–",
            delta=None,
        )

    with col3:
        st.metric(
            "24h Volume",
            f"{price.volume_24h:,}" if price.volume_24h else "–",
            delta=None,
        )

    with col4:
        st.metric(
            "Open Interest",
            f"{price.open_interest:,}" if price.open_interest else "–",
            delta=None,
        )

    # Display bid/ask details
    price_cols = st.columns(2)
    with price_cols[0]:
        st.markdown(f"**YES Bid:** {price.yes_bid*100:.1f}¢" if price.yes_bid else "**YES Bid:** –")
        st.markdown(f"**YES Ask:** {price.yes_ask*100:.1f}¢" if price.yes_ask else "**YES Ask:** –")

    with price_cols[1]:
        st.markdown(f"**NO Bid:** {price.no_bid*100:.1f}¢" if price.no_bid else "**NO Bid:** –")
        st.markdown(f"**NO Ask:** {price.no_ask*100:.1f}¢" if price.no_ask else "**NO Ask:** –")


def display_live_prices_section(predictions_df: pd.DataFrame) -> None:
    """Display live prices for all predictions in an expandable section."""
    st.subheader("💹 Live Market Prices")

    with st.spinner("Fetching live price data from Kalshi..."):
        try:
            live_prices = fetch_live_prices_for_predictions(predictions_df)

            if not live_prices:
                st.warning("⚠️ Could not fetch live price data. Check your Kalshi API credentials.")
                return

            st.success(f"✓ Loaded live prices for {len(live_prices)} markets")

            # Group by meeting date
            if "meeting_date" in predictions_df.columns:
                meeting_dates = sorted(predictions_df["meeting_date"].dropna().unique())

                for meeting_date in meeting_dates:
                    meeting_predictions = predictions_df[
                        predictions_df["meeting_date"] == meeting_date
                    ]

                    with st.expander(f"📅 {meeting_date} ({len(meeting_predictions)} markets)", expanded=True):
                        for idx, row in meeting_predictions.iterrows():
                            ticker = row.get("ticker")
                            contract = row.get("contract", ticker)

                            st.markdown(f"### {contract}")
                            display_live_price_card(ticker, live_prices)
                            st.divider()
            else:
                # If no meeting date, just display all
                for idx, row in predictions_df.iterrows():
                    ticker = row.get("ticker")
                    contract = row.get("contract", ticker)

                    with st.expander(f"{contract}", expanded=False):
                        display_live_price_card(ticker, live_prices)

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
st.sidebar.subheader("Trade Thresholds")
yes_edge_threshold = st.sidebar.slider(
    "BUY YES Edge Threshold",
    min_value=0.05,
    max_value=0.30,
    value=0.15,
    step=0.01,
    help="Minimum edge required to recommend buying YES"
)

no_edge_threshold = st.sidebar.slider(
    "BUY NO Edge Threshold",
    min_value=0.05,
    max_value=0.30,
    value=0.12,
    step=0.01,
    help="Minimum edge required to recommend buying NO"
)

min_yes_prob = st.sidebar.slider(
    "Min Probability for YES",
    min_value=0.50,
    max_value=0.80,
    value=0.60,
    step=0.05,
    help="Minimum predicted probability to consider BUY YES"
)

max_no_prob = st.sidebar.slider(
    "Max Probability for NO",
    min_value=0.20,
    max_value=0.50,
    value=0.40,
    step=0.05,
    help="Maximum predicted probability to consider BUY NO"
)

st.sidebar.divider()

# Dataset selection
st.sidebar.subheader("Data Selection")
dataset_types = repo.list_dataset_types()

if dataset_types:
    selected_type = st.sidebar.selectbox("Dataset type", dataset_types, index=0)
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

selected_run_label = st.sidebar.selectbox("Dataset run", runs_df["label"].tolist(), index=0)
selected_run_id = runs_df.loc[runs_df["label"] == selected_run_label, "dataset_run_id"].iloc[0]
metadata = repo.get_dataset_run(selected_run_id)

# Main header
st.title("📊 Word Mention Prediction Markets")
st.caption("AI-powered word mention predictions for earnings calls and FOMC meetings")

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
        st.header("🎯 Live Trading Predictions")

        # Refresh button
        col1, col2, col3 = st.columns([2, 1, 1])
        with col3:
            if st.button("🔄 Refresh Predictions", width='stretch'):
                refresh_predictions()

        if predictions_df.empty:
            st.info("📭 No live predictions available. Click 'Refresh Predictions' to generate new ones.")
        else:
            # Add live prices section at the top
            st.divider()
            display_live_prices_section(predictions_df)
            st.divider()

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

            # Top metrics
            metric_cols = st.columns(5)
            metric_cols[0].metric(
                "Total Predictions",
                len(df)
            )
            metric_cols[1].metric(
                "Action Items",
                total_opportunities,
                delta=f"{buy_yes_count} YES, {buy_no_count} NO"
            )
            metric_cols[2].metric(
                "Next Meeting",
                str(next_meeting) if next_meeting else "–"
            )

            if "edge" in df.columns:
                best_edge = df["edge"].abs().max()
                metric_cols[3].metric(
                    "Best Edge",
                    f"{best_edge*100:.1f}%" if pd.notna(best_edge) else "–"
                )

            if "days_until_meeting" in df.columns:
                days_until = df["days_until_meeting"].min()
                metric_cols[4].metric(
                    "Days Until Meeting",
                    int(days_until) if pd.notna(days_until) else "–"
                )

            st.divider()

            # Filters
            filter_cols = st.columns(3)

            with filter_cols[0]:
                meeting_dates = ["All"] + sorted(df["meeting_date"].dropna().astype(str).unique())
                meeting_filter = st.selectbox("Meeting Date", meeting_dates)

            with filter_cols[1]:
                recommendation_filter = st.selectbox(
                    "Recommendation",
                    ["All", "BUY YES", "BUY NO", "HOLD"]
                )

            with filter_cols[2]:
                min_edge_filter = st.slider(
                    "Min Absolute Edge",
                    0.0, 0.5, 0.0, 0.01
                )

            # Apply filters
            filtered_df = df.copy()

            if meeting_filter != "All":
                filtered_df = filtered_df[filtered_df["meeting_date"].astype(str) == meeting_filter]

            if recommendation_filter != "All":
                filtered_df = filtered_df[filtered_df["recommendation"] == recommendation_filter]

            if min_edge_filter > 0:
                filtered_df = filtered_df[filtered_df["edge"].abs() >= min_edge_filter]

            # Sort by absolute edge
            filtered_df["edge_abs"] = filtered_df["edge"].abs()
            filtered_df = filtered_df.sort_values("edge_abs", ascending=False)

            # Display predictions
            st.subheader(f"📋 {len(filtered_df)} Predictions")

            if filtered_df.empty:
                st.info("No predictions match your filters.")
            else:
                # Format for display
                display_df = filtered_df.copy()

                # Select and order columns
                display_cols = [
                    "recommendation",
                    "contract",
                    "meeting_date",
                    "predicted_probability",
                    "market_price",
                    "edge",
                    "confidence_lower",
                    "confidence_upper",
                    "days_until_meeting",
                    "market_status",
                ]

                display_cols = [col for col in display_cols if col in display_df.columns]
                display_df = display_df[display_cols]

                # Format percentages
                for col in ["predicted_probability", "confidence_lower", "confidence_upper", "market_price"]:
                    if col in display_df.columns:
                        display_df[col] = display_df[col].apply(
                            lambda x: f"{x*100:.1f}%" if pd.notna(x) else "–"
                        )

                if "edge" in display_df.columns:
                    display_df["edge"] = display_df["edge"].apply(
                        lambda x: f"{x*100:+.1f}%" if pd.notna(x) else "–"
                    )

                # Color code recommendations
                def highlight_recommendation(row):
                    if row["recommendation"] == "BUY YES":
                        return ["background-color: #d4edda"] * len(row)
                    elif row["recommendation"] == "BUY NO":
                        return ["background-color: #f8d7da"] * len(row)
                    else:
                        return [""] * len(row)

                styled_df = display_df.style.apply(highlight_recommendation, axis=1)
                st.dataframe(styled_df, hide_index=True, width='stretch')

                # Show detailed view for top opportunities
                st.subheader("🎯 Top Opportunities")
                top_opportunities = filtered_df[filtered_df["recommendation"] != "HOLD"].head(5)

                if top_opportunities.empty:
                    st.info("No strong trading opportunities at current thresholds.")
                else:
                    for idx, row in top_opportunities.iterrows():
                        with st.expander(
                            f"{row['recommendation']} - {row['contract']} (Edge: {row['edge']*100:+.1f}%)"
                        ):
                            detail_cols = st.columns(4)
                            detail_cols[0].metric("Predicted Probability", f"{row['predicted_probability']*100:.1f}%")
                            detail_cols[1].metric("Market Price", f"{row['market_price']*100:.1f}%" if pd.notna(row.get('market_price')) else "–")
                            detail_cols[2].metric("Edge", f"{row['edge']*100:+.1f}%")
                            detail_cols[3].metric("Days Until", int(row.get('days_until_meeting', 0)) if pd.notna(row.get('days_until_meeting')) else "–")

                            st.markdown(f"**Meeting Date:** {row.get('meeting_date', '–')}")
                            st.markdown(f"**Confidence Interval:** {row.get('confidence_lower', 0)*100:.1f}% - {row.get('confidence_upper', 0)*100:.1f}%")
                            st.markdown(f"**Market Status:** {row.get('market_status', '–')}")

    # Training Data Tab
    with main_tabs[1]:
        st.header("📚 Training Data & Model Info")

        if metadata:
            st.subheader("Model Configuration")
            meta_cols = st.columns(3)
            meta_cols[0].markdown(f"**Dataset**: `{metadata.dataset_slug}`")
            meta_cols[1].markdown(f"**Type**: `{metadata.dataset_type}`")
            meta_cols[2].markdown(f"**Updated**: {metadata.run_timestamp}")

            if metadata.hyperparameters:
                st.json(metadata.hyperparameters, expanded=False)

        st.info("📖 This tab shows the model configuration and training metadata. The Predictions tab is where you'll find actionable trading signals.")

    # Settings Tab
    with main_tabs[2]:
        st.header("⚙️ Settings & Information")

        st.subheader("Trade Recommendation Logic")
        st.markdown("""
        **BUY YES** recommendations require:
        - Predicted probability ≥ {:.0%}
        - Edge ≥ {:.0%}

        **BUY NO** recommendations require:
        - Predicted probability ≤ {:.0%}
        - Edge ≤ -{:.0%}

        **HOLD** for everything else.
        """.format(min_yes_prob, yes_edge_threshold, max_no_prob, no_edge_threshold))

        st.subheader("About This Dashboard")
        st.markdown("""
        This dashboard analyzes FOMC press conference transcripts to predict mention probabilities
        for Kalshi prediction market contracts. The predictions use historical data and statistical
        models to identify potential trading opportunities.

        **Key Features:**
        - 🎯 Clear BUY YES / BUY NO / HOLD recommendations
        - 📊 Confidence intervals for uncertainty quantification
        - 🔄 Easy refresh from live Kalshi data
        - ⚙️ Configurable trade thresholds
        - 📈 Backtest results for model validation
        """)

else:
    # BACKTEST VIEW
    with main_tabs[0]:
        st.header("📈 Backtest Results")

        overall = repo.get_overall_metrics(selected_run_id)
        if overall:
            st.subheader("Overall Performance")
            metric_cols = st.columns(4)
            metric_cols[0].metric("ROI", format_metric(overall.get("roi"), pct=True))
            metric_cols[1].metric("Sharpe Ratio", format_metric(overall.get("sharpe")))
            metric_cols[2].metric("Win Rate", format_metric(overall.get("win_rate"), pct=True))
            metric_cols[3].metric("Total PnL", f"${format_metric(overall.get('total_pnl'))}")

        horizon_df = repo.get_horizon_metrics(selected_run_id)
        if not horizon_df.empty:
            st.subheader("Performance by Horizon")
            st.dataframe(horizon_df, hide_index=True, width='stretch')

    with main_tabs[1]:
        st.header("🎯 Predictions")
        min_edge = st.slider("Minimum absolute edge", 0.0, 0.5, 0.0, 0.01)

        if not predictions_df.empty:
            df = predictions_df.copy()
            if "edge" in df.columns:
                df["edge"] = pd.to_numeric(df["edge"], errors="coerce")
            else:
                df["edge"] = 0.0
            df["edge_abs"] = df["edge"].fillna(0.0).abs()
            df = df[df["edge_abs"] >= min_edge]
            df = df.sort_values("edge_abs", ascending=False).drop(columns=["edge_abs"])
            st.dataframe(df, hide_index=True, width='stretch')
        else:
            st.info("No predictions available for this run.")

    with main_tabs[2]:
        st.header("💼 Trades")

        if not trades_df.empty:
            st.dataframe(trades_df, hide_index=True, width='stretch')

            trade_cols = st.columns(3)
            total_pnl = trades_df["pnl"].fillna(0).sum()
            winning_trades = len(trades_df[trades_df["pnl"] > 0])
            total_trades = len(trades_df)

            trade_cols[0].metric("Total PnL", f"${format_metric(total_pnl)}")
            trade_cols[1].metric("Total Trades", total_trades)
            trade_cols[2].metric("Winning Trades", winning_trades)
        else:
            st.info("No trades executed for this run.")

    with main_tabs[3]:
        st.header("🔍 Grid Search Results")

        if not grid_df.empty:
            st.dataframe(grid_df, hide_index=True, width='stretch')
        else:
            st.info("No grid search results for this dataset run.")

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
