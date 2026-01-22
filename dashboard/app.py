"""Streamlit app for exploring FOMC analytics outputs."""

from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from fomc_analysis.dashboard import DashboardRepository


st.set_page_config(page_title="FOMC Analytics Dashboard", layout="wide")


@st.cache_resource(show_spinner=False)
def get_repository() -> DashboardRepository:
    return DashboardRepository(database_url=os.getenv("DATABASE_URL"))


def format_metric(value: float | None, pct: bool = False) -> str:
    if value is None:
        return "–"
    if pct:
        return f"{value * 100:.2f}%"
    return f"{value:,.2f}"


repo = get_repository()
dataset_types = repo.list_dataset_types()

st.sidebar.title("Filters")
if dataset_types:
    selected_type = st.sidebar.selectbox("Dataset type", dataset_types, index=0)
else:
    selected_type = None

runs_df = repo.list_dataset_runs(dataset_type=selected_type)
if runs_df.empty:
    st.warning("No dataset runs found. Load data via scripts/load_* before using the dashboard.")
    st.stop()


if "run_identifier" in runs_df.columns:
    runs_df["label"] = runs_df.apply(
        lambda row: f"{row['dataset_slug']} – {row['run_timestamp']}", axis=1
    )
else:
    runs_df["label"] = runs_df.apply(
        lambda row: f"{row['dataset_slug']} – {row['run_timestamp']}", axis=1
    )

selected_run_label = st.sidebar.selectbox("Dataset run", runs_df["label"].tolist(), index=0)
selected_run_id = runs_df.loc[runs_df["label"] == selected_run_label, "dataset_run_id"].iloc[0]
metadata = repo.get_dataset_run(selected_run_id)

st.title("FOMC Analytics Dashboard")
st.caption("Interactive view of backtests, grid searches, and live predictions")

if metadata:
    st.subheader("Run Metadata")
    meta_cols = st.columns(3)
    meta_cols[0].markdown(f"**Dataset**: `{metadata.dataset_slug}`")
    meta_cols[1].markdown(f"**Type**: `{metadata.dataset_type}`")
    meta_cols[2].markdown(f"**Timestamp**: {metadata.run_timestamp}")
    st.json(metadata.hyperparameters or {}, expanded=False)

predictions_df = repo.get_predictions(selected_run_id)
trades_df = repo.get_trades(selected_run_id)
grid_df = repo.get_grid_search_results(selected_run_id)

is_live_run = bool(
    metadata
    and metadata.dataset_slug
    and "upcoming" in metadata.dataset_slug
) or (not predictions_df.empty and predictions_df["prediction_kind"].eq("live").all())

if not is_live_run:
    overall = repo.get_overall_metrics(selected_run_id)
    if overall:
        st.subheader("Overall Metrics")
        metric_cols = st.columns(4)
        metric_cols[0].metric("ROI", format_metric(overall.get("roi"), pct=True))
        metric_cols[1].metric("Sharpe", format_metric(overall.get("sharpe")))
        metric_cols[2].metric("Win Rate", format_metric(overall.get("win_rate"), pct=True))
        metric_cols[3].metric("Total PnL", format_metric(overall.get("total_pnl")))

    horizon_df = repo.get_horizon_metrics(selected_run_id)
    if not horizon_df.empty:
        st.subheader("Horizon Metrics")
        st.dataframe(horizon_df, hide_index=True)

    tabs = st.tabs(["Predictions", "Trades", "Grid Search"])

    with tabs[0]:
        st.subheader("Predictions")
        min_edge = st.slider("Minimum absolute edge", 0.0, 0.5, 0.0, 0.01)
        if not predictions_df.empty:
            df = predictions_df.copy()
            df["edge_abs"] = df["edge"].fillna(0).abs()
            df = df[df["edge_abs"] >= min_edge]
            df = df.drop(columns=["edge_abs"])
            st.dataframe(df, hide_index=True)
        else:
            st.info("No predictions available for this run.")

    with tabs[1]:
        st.subheader("Trades")
        if not trades_df.empty:
            st.dataframe(trades_df, hide_index=True)
            total_pnl = trades_df["pnl"].fillna(0).sum()
            st.metric("Aggregate PnL", format_metric(total_pnl))
        else:
            st.info("No trades executed for this run.")

    with tabs[2]:
        st.subheader("Grid Search Results")
        if not grid_df.empty:
            st.dataframe(grid_df, hide_index=True)
        else:
            st.info("No grid search results for this dataset run.")
else:
    st.subheader("Upcoming Predictions")
    if predictions_df.empty:
        st.info("No live predictions have been loaded yet.")
    else:
        meeting_dates = sorted(predictions_df["meeting_date"].dropna().astype(str).unique())
        if meeting_dates:
            meeting_filter = st.selectbox(
                "Meeting date",
                options=["All"] + meeting_dates,
                index=0,
            )
        else:
            meeting_filter = "All"

        min_edge = st.slider("Minimum absolute edge", 0.0, 0.5, 0.0, 0.01)
        df = predictions_df.copy()
        df["edge_abs"] = df["edge"].fillna(0).abs()
        if meeting_filter != "All":
            df = df[df["meeting_date"].astype(str) == meeting_filter]
        df = df[df["edge_abs"] >= min_edge]
        live_cols = [
            "meeting_date",
            "contract",
            "ticker",
            "market_status",
            "predicted_probability",
            "confidence_lower",
            "confidence_upper",
            "market_price",
            "edge",
            "days_until_meeting",
        ]
        display_cols = [col for col in live_cols if col in df.columns]
        if df.empty:
            st.info("No predictions match the current filters.")
        else:
            df = df.sort_values(by="edge_abs", ascending=False)
            df = df.drop(columns=["edge_abs"])
            st.dataframe(df[display_cols], hide_index=True)
