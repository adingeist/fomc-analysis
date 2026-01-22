"""Enhanced Streamlit app for FOMC trading predictions with clear action signals."""

from __future__ import annotations

import os
import subprocess
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from fomc_analysis.dashboard import DashboardRepository


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
st.title("📊 FOMC Trading Predictions")
st.caption("AI-powered FOMC mention predictions with actionable trading signals")

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

# Create main tabs
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
            if st.button("🔄 Refresh Predictions", use_container_width=True):
                refresh_predictions()

        if predictions_df.empty:
            st.info("📭 No live predictions available. Click 'Refresh Predictions' to generate new ones.")
        else:
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
                st.dataframe(styled_df, hide_index=True, use_container_width=True)

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
            st.dataframe(horizon_df, hide_index=True, use_container_width=True)

    with main_tabs[1]:
        st.header("🎯 Predictions")
        min_edge = st.slider("Minimum absolute edge", 0.0, 0.5, 0.0, 0.01)

        if not predictions_df.empty:
            df = predictions_df.copy()
            df["edge_abs"] = df["edge"].fillna(0).abs()
            df = df[df["edge_abs"] >= min_edge]
            df = df.sort_values("edge_abs", ascending=False).drop(columns=["edge_abs"])
            st.dataframe(df, hide_index=True, use_container_width=True)
        else:
            st.info("No predictions available for this run.")

    with main_tabs[2]:
        st.header("💼 Trades")

        if not trades_df.empty:
            st.dataframe(trades_df, hide_index=True, use_container_width=True)

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
            st.dataframe(grid_df, hide_index=True, use_container_width=True)
        else:
            st.info("No grid search results for this dataset run.")
