"""Initial database schema."""

from __future__ import annotations

from alembic import op
import sqlalchemy as sa


revision = "20240201_000001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "dataset_runs",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("run_id", sa.String(length=64), nullable=False, unique=True),
        sa.Column("dataset_slug", sa.String(length=128), nullable=False),
        sa.Column("dataset_type", sa.String(length=64), nullable=False),
        sa.Column("run_timestamp", sa.DateTime(timezone=True), nullable=False),
        sa.Column("source_file", sa.String(length=512), nullable=True),
        sa.Column("source_hash", sa.String(length=128), nullable=True),
        sa.Column("hyperparameters", sa.JSON(), nullable=True),
        sa.Column("metadata_json", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.UniqueConstraint("dataset_slug", "run_timestamp", name="uq_dataset_slug_run_ts"),
    )

    op.create_table(
        "overall_metrics",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("dataset_run_id", sa.String(length=36), nullable=False, unique=True),
        sa.Column("total_trades", sa.Float(), nullable=True),
        sa.Column("total_pnl", sa.Float(), nullable=True),
        sa.Column("roi", sa.Float(), nullable=True),
        sa.Column("win_rate", sa.Float(), nullable=True),
        sa.Column("sharpe", sa.Float(), nullable=True),
        sa.Column("sortino", sa.Float(), nullable=True),
        sa.Column("avg_pnl_per_trade", sa.Float(), nullable=True),
        sa.Column("final_capital", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(["dataset_run_id"], ["dataset_runs.id"], ondelete="CASCADE"),
    )

    op.create_table(
        "horizon_metrics",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("dataset_run_id", sa.String(length=36), nullable=False),
        sa.Column("horizon_days", sa.Integer(), nullable=False),
        sa.Column("total_predictions", sa.Integer(), nullable=True),
        sa.Column("correct_predictions", sa.Integer(), nullable=True),
        sa.Column("accuracy", sa.Float(), nullable=True),
        sa.Column("total_trades", sa.Integer(), nullable=True),
        sa.Column("winning_trades", sa.Integer(), nullable=True),
        sa.Column("win_rate", sa.Float(), nullable=True),
        sa.Column("total_pnl", sa.Float(), nullable=True),
        sa.Column("avg_pnl_per_trade", sa.Float(), nullable=True),
        sa.Column("roi", sa.Float(), nullable=True),
        sa.Column("sharpe_ratio", sa.Float(), nullable=True),
        sa.Column("brier_score", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(["dataset_run_id"], ["dataset_runs.id"], ondelete="CASCADE"),
        sa.UniqueConstraint("dataset_run_id", "horizon_days", name="uq_run_horizon"),
    )

    op.create_table(
        "predictions",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("dataset_run_id", sa.String(length=36), nullable=False),
        sa.Column("meeting_date", sa.Date(), nullable=True),
        sa.Column("prediction_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("contract", sa.String(length=256), nullable=True),
        sa.Column("ticker", sa.String(length=128), nullable=True),
        sa.Column("event_ticker", sa.String(length=128), nullable=True),
        sa.Column("market_status", sa.String(length=64), nullable=True),
        sa.Column("prediction_kind", sa.String(length=32), nullable=True),
        sa.Column("days_before_meeting", sa.Integer(), nullable=True),
        sa.Column("days_until_meeting", sa.Integer(), nullable=True),
        sa.Column("predicted_probability", sa.Float(), nullable=True),
        sa.Column("confidence_lower", sa.Float(), nullable=True),
        sa.Column("confidence_upper", sa.Float(), nullable=True),
        sa.Column("market_price", sa.Float(), nullable=True),
        sa.Column("edge", sa.Float(), nullable=True),
        sa.Column("actual_outcome", sa.Integer(), nullable=True),
        sa.Column("correct", sa.Boolean(), nullable=True),
        sa.ForeignKeyConstraint(["dataset_run_id"], ["dataset_runs.id"], ondelete="CASCADE"),
    )

    op.create_table(
        "trades",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("dataset_run_id", sa.String(length=36), nullable=False),
        sa.Column("meeting_date", sa.Date(), nullable=True),
        sa.Column("prediction_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("contract", sa.String(length=256), nullable=True),
        sa.Column("ticker", sa.String(length=128), nullable=True),
        sa.Column("days_before_meeting", sa.Integer(), nullable=True),
        sa.Column("side", sa.String(length=16), nullable=True),
        sa.Column("position_size", sa.Float(), nullable=True),
        sa.Column("entry_price", sa.Float(), nullable=True),
        sa.Column("predicted_probability", sa.Float(), nullable=True),
        sa.Column("edge", sa.Float(), nullable=True),
        sa.Column("actual_outcome", sa.Integer(), nullable=True),
        sa.Column("pnl", sa.Float(), nullable=True),
        sa.Column("roi", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(["dataset_run_id"], ["dataset_runs.id"], ondelete="CASCADE"),
    )

    op.create_table(
        "grid_search_results",
        sa.Column("id", sa.String(length=36), primary_key=True),
        sa.Column("dataset_run_id", sa.String(length=36), nullable=False),
        sa.Column("total_pnl", sa.Float(), nullable=True),
        sa.Column("roi", sa.Float(), nullable=True),
        sa.Column("sharpe", sa.Float(), nullable=True),
        sa.Column("trades", sa.Integer(), nullable=True),
        sa.Column("win_rate", sa.Float(), nullable=True),
        sa.Column("slippage", sa.Float(), nullable=True),
        sa.Column("transaction_cost_rate", sa.Float(), nullable=True),
        sa.Column("max_position_size", sa.Float(), nullable=True),
        sa.Column("train_window_size", sa.Integer(), nullable=True),
        sa.Column("test_start_date", sa.String(length=32), nullable=True),
        sa.Column("yes_position_size_pct", sa.Float(), nullable=True),
        sa.Column("no_position_size_pct", sa.Float(), nullable=True),
        sa.Column("yes_edge_threshold", sa.Float(), nullable=True),
        sa.Column("no_edge_threshold", sa.Float(), nullable=True),
        sa.ForeignKeyConstraint(["dataset_run_id"], ["dataset_runs.id"], ondelete="CASCADE"),
    )


def downgrade() -> None:
    op.drop_table("grid_search_results")
    op.drop_table("trades")
    op.drop_table("predictions")
    op.drop_table("horizon_metrics")
    op.drop_table("overall_metrics")
    op.drop_table("dataset_runs")
