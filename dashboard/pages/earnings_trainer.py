"""
Streamlit page for training earnings call prediction models.

This page allows users to:
1. Select a ticker and fetch Kalshi contracts
2. View and manage transcript data
3. Analyze word mention frequency statistics
4. Train Beta-Binomial prediction models
5. Run backtests and view performance
6. Generate predictions for upcoming earnings
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

# Configure page
st.set_page_config(
    page_title="Earnings Call Trainer",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stat-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin: 10px 0;
    }
    .stat-value {
        font-size: 32px;
        font-weight: bold;
    }
    .stat-label {
        font-size: 14px;
        opacity: 0.9;
    }
    .success-card {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .warning-card {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .error-card {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .metric-row {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid #eee;
    }
    .training-progress {
        background-color: #e9ecef;
        border-radius: 10px;
        padding: 3px;
    }
    .training-progress-bar {
        background: linear-gradient(90deg, #28a745 0%, #20c997 100%);
        border-radius: 8px;
        height: 20px;
        transition: width 0.3s ease;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session State Initialization
# ---------------------------------------------------------------------------
def init_session_state():
    """Initialize session state variables."""
    defaults = {
        "selected_ticker": None,
        "contracts": [],
        "transcripts": [],
        "features_df": None,
        "outcomes_df": None,
        "model_trained": False,
        "backtest_result": None,
        "predictions": [],
        "training_log": [],
        "mention_stats": {},
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


init_session_state()


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def get_kalshi_client():
    """Get Kalshi API client."""
    try:
        from fomc_analysis.kalshi_client_factory import get_kalshi_client
        return get_kalshi_client()
    except Exception as e:
        st.error(f"Failed to initialize Kalshi client: {e}")
        return None


def run_async(coro):
    """Run async coroutine in sync context."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():
        # Create a new loop for this thread
        new_loop = asyncio.new_event_loop()
        try:
            return new_loop.run_until_complete(coro)
        finally:
            new_loop.close()
    else:
        return loop.run_until_complete(coro)


@st.cache_data(ttl=300)
def fetch_contracts_for_ticker(_client, ticker: str) -> List[Dict]:
    """Fetch Kalshi contracts for a ticker (cached for 5 min)."""
    from earnings_analysis.kalshi.contract_analyzer import EarningsContractAnalyzer

    analyzer = EarningsContractAnalyzer(_client, ticker)
    contracts = run_async(analyzer.fetch_contracts(market_status="all"))
    return [c.to_dict() for c in contracts]


def load_transcript_data(ticker: str, data_dir: Path) -> pd.DataFrame:
    """Load transcript segment data for a ticker."""
    segments_dir = data_dir / "segments" / ticker

    if not segments_dir.exists():
        return pd.DataFrame()

    all_segments = []
    for segment_file in sorted(segments_dir.glob("*.jsonl")):
        call_date = segment_file.stem.replace(f"{ticker}_", "")
        with open(segment_file) as f:
            for line in f:
                seg = json.loads(line)
                seg["call_date"] = call_date
                seg["file"] = segment_file.name
                all_segments.append(seg)

    return pd.DataFrame(all_segments)


def count_word_mentions(text: str, word: str) -> int:
    """Count mentions of a word (handling multi-word phrases)."""
    import re

    # Handle "word1 / word2" format (either counts)
    variants = [v.strip() for v in word.split("/")]

    total = 0
    for variant in variants:
        # Case-insensitive, word boundary matching
        pattern = r"\b" + re.escape(variant) + r"\b"
        total += len(re.findall(pattern, text, re.IGNORECASE))

    return total


def build_features_and_outcomes(
    segments_df: pd.DataFrame,
    contracts: List[Dict],
    speaker_mode: str = "executives_only"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build features and outcomes DataFrames from segments and contracts."""
    if segments_df.empty or not contracts:
        return pd.DataFrame(), pd.DataFrame()

    # Filter by speaker mode
    if speaker_mode == "executives_only":
        filtered = segments_df[segments_df["role"].isin(["ceo", "cfo", "executive"])]
    elif speaker_mode == "ceo_only":
        filtered = segments_df[segments_df["role"] == "ceo"]
    elif speaker_mode == "cfo_only":
        filtered = segments_df[segments_df["role"] == "cfo"]
    else:
        filtered = segments_df

    # Group by call date
    call_dates = sorted(filtered["call_date"].unique())

    # Get words from contracts
    words = [c["word"] for c in contracts]

    # Build features (mention counts)
    features_data = {}
    outcomes_data = {}

    for call_date in call_dates:
        call_segments = filtered[filtered["call_date"] == call_date]
        combined_text = " ".join(call_segments["text"].fillna(""))

        for word in words:
            if word not in features_data:
                features_data[word] = {}
                outcomes_data[word] = {}

            count = count_word_mentions(combined_text, word)
            features_data[word][call_date] = count
            # Binary outcome: mentioned at least once
            outcomes_data[word][call_date] = 1 if count > 0 else 0

    features_df = pd.DataFrame(features_data)
    features_df.index.name = "call_date"

    outcomes_df = pd.DataFrame(outcomes_data)
    outcomes_df.index.name = "call_date"

    return features_df, outcomes_df


def calculate_mention_statistics(
    features_df: pd.DataFrame,
    outcomes_df: pd.DataFrame
) -> Dict[str, Dict[str, Any]]:
    """Calculate detailed mention statistics for each word."""
    stats = {}

    for word in features_df.columns:
        counts = features_df[word]
        outcomes = outcomes_df[word]

        stats[word] = {
            "total_calls": len(counts),
            "calls_mentioned": int(outcomes.sum()),
            "mention_frequency": float(outcomes.mean()),
            "total_mentions": int(counts.sum()),
            "avg_mentions_per_call": float(counts.mean()),
            "max_mentions": int(counts.max()),
            "min_mentions": int(counts.min()),
            "std_mentions": float(counts.std()),
            "recent_3_avg": float(counts.tail(3).mean()) if len(counts) >= 3 else float(counts.mean()),
            "trend": "increasing" if len(counts) >= 3 and counts.tail(3).mean() > counts.head(3).mean() else "decreasing" if len(counts) >= 3 and counts.tail(3).mean() < counts.head(3).mean() else "stable",
        }

    return stats


def train_model_for_word(
    features_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    word: str,
    model_params: Dict
) -> Dict:
    """Train a Beta-Binomial model for a specific word."""
    from earnings_analysis.models.beta_binomial import BetaBinomialEarningsModel

    model = BetaBinomialEarningsModel(**model_params)
    model.fit(features_df, outcomes_df[word])
    prediction = model.predict()

    return {
        "word": word,
        "alpha_posterior": model.alpha_post,
        "beta_posterior": model.beta_post,
        "n_observations": model.n_observations,
        "predicted_probability": float(prediction.iloc[0]["probability"]),
        "confidence_lower": float(prediction.iloc[0]["lower_bound"]),
        "confidence_upper": float(prediction.iloc[0]["upper_bound"]),
        "uncertainty": float(prediction.iloc[0]["uncertainty"]),
    }


def run_backtest(
    ticker: str,
    features_df: pd.DataFrame,
    outcomes_df: pd.DataFrame,
    model_params: Dict,
    backtest_params: Dict
) -> Dict:
    """Run a backtest and return results."""
    from earnings_analysis.kalshi.backtester import EarningsKalshiBacktester, BacktestResult
    from earnings_analysis.models.beta_binomial import BetaBinomialEarningsModel

    backtester = EarningsKalshiBacktester(
        features=features_df,
        outcomes=outcomes_df,
        model_class=BetaBinomialEarningsModel,
        model_params=model_params,
        **backtest_params
    )

    result = backtester.run(ticker=ticker)

    return {
        "predictions": [asdict(p) for p in result.predictions],
        "trades": [asdict(t) for t in result.trades],
        "metrics": asdict(result.metrics),
        "metadata": result.metadata,
    }


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------
def render_sidebar():
    """Render sidebar with configuration options."""
    st.sidebar.title("📞 Earnings Trainer")
    st.sidebar.markdown("Train models for Kalshi earnings mention contracts")

    st.sidebar.divider()

    # Ticker selection
    st.sidebar.markdown("### 📊 Ticker Selection")

    # Common tickers with Kalshi contracts
    common_tickers = ["META", "TSLA", "NVDA", "AAPL", "GOOGL", "AMZN", "MSFT", "COIN", "NFLX", "AMD"]

    ticker_input = st.sidebar.text_input(
        "Enter ticker symbol",
        value=st.session_state.selected_ticker or "",
        placeholder="e.g., META",
        help="Enter the stock ticker symbol for the company"
    ).upper()

    st.sidebar.caption("Quick select:")
    cols = st.sidebar.columns(5)
    for i, ticker in enumerate(common_tickers[:5]):
        if cols[i].button(ticker, key=f"quick_{ticker}"):
            st.session_state.selected_ticker = ticker
            st.rerun()

    cols2 = st.sidebar.columns(5)
    for i, ticker in enumerate(common_tickers[5:]):
        if cols2[i].button(ticker, key=f"quick2_{ticker}"):
            st.session_state.selected_ticker = ticker
            st.rerun()

    if ticker_input and ticker_input != st.session_state.selected_ticker:
        st.session_state.selected_ticker = ticker_input
        # Clear cached data when ticker changes
        st.session_state.contracts = []
        st.session_state.features_df = None
        st.session_state.outcomes_df = None
        st.session_state.model_trained = False
        st.rerun()

    st.sidebar.divider()

    # Model parameters
    st.sidebar.markdown("### ⚙️ Model Parameters")

    alpha_prior = st.sidebar.slider(
        "Alpha Prior",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Beta distribution alpha parameter (higher = more confident prior for YES)"
    )

    beta_prior = st.sidebar.slider(
        "Beta Prior",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Beta distribution beta parameter (higher = more confident prior for NO)"
    )

    half_life = st.sidebar.slider(
        "Half-life (quarters)",
        min_value=1.0,
        max_value=20.0,
        value=8.0,
        step=1.0,
        help="Exponential decay half-life for recency weighting"
    )

    st.sidebar.divider()

    # Backtest parameters
    st.sidebar.markdown("### 📈 Backtest Parameters")

    edge_threshold = st.sidebar.slider(
        "Edge Threshold",
        min_value=0.05,
        max_value=0.30,
        value=0.12,
        step=0.01,
        help="Minimum edge required to execute a trade"
    )

    position_size = st.sidebar.slider(
        "Position Size (%)",
        min_value=1,
        max_value=10,
        value=3,
        step=1,
        help="Percentage of capital per trade"
    )

    min_train_window = st.sidebar.slider(
        "Min Training Window",
        min_value=2,
        max_value=8,
        value=4,
        step=1,
        help="Minimum number of historical calls before making predictions"
    )

    return {
        "model_params": {
            "alpha_prior": alpha_prior,
            "beta_prior": beta_prior,
            "half_life": half_life,
        },
        "backtest_params": {
            "edge_threshold": edge_threshold,
            "position_size_pct": position_size / 100,
            "min_train_window": min_train_window,
        }
    }


def render_contracts_section(ticker: str, client):
    """Render Kalshi contracts section."""
    st.markdown("### 📜 Kalshi Contracts")
    st.markdown(f"Available word mention contracts for **{ticker}** earnings calls")

    col1, col2 = st.columns([3, 1])

    with col2:
        if st.button("🔄 Fetch Contracts", type="primary"):
            with st.spinner(f"Fetching contracts for {ticker}..."):
                try:
                    contracts = fetch_contracts_for_ticker(client, ticker)
                    st.session_state.contracts = contracts
                    st.success(f"Found {len(contracts)} contracts!")
                except Exception as e:
                    st.error(f"Error fetching contracts: {e}")

    contracts = st.session_state.contracts

    if not contracts:
        st.info("Click 'Fetch Contracts' to load available Kalshi contracts for this ticker.")
        return

    # Display contracts in a table
    contracts_df = pd.DataFrame([
        {
            "Word": c["word"],
            "Threshold": c["threshold"],
            "Markets": len(c.get("markets", [])),
            "Market Ticker": c["market_ticker"],
        }
        for c in contracts
    ])

    st.dataframe(contracts_df, hide_index=True, use_container_width=True)

    # Show active markets
    with st.expander("📊 Active Markets Details", expanded=False):
        for contract in contracts:
            markets = contract.get("markets", [])
            active_markets = [m for m in markets if m.get("status", "").lower() in ("active", "open")]

            if active_markets:
                st.markdown(f"**{contract['word']}** - {len(active_markets)} active markets")
                for market in active_markets[:3]:  # Show first 3
                    price = market.get("last_price", 0)
                    if isinstance(price, (int, float)):
                        price_display = f"{price:.0f}¢" if price < 1 else f"${price:.2f}"
                    else:
                        price_display = str(price)
                    st.caption(f"  • {market.get('ticker', 'N/A')}: {price_display}")


def render_data_section(ticker: str):
    """Render transcript data management section."""
    st.markdown("### 📚 Training Data")
    st.markdown("Manage earnings call transcripts for model training")

    data_dir = Path("data/earnings")
    segments_dir = data_dir / "segments" / ticker

    # Check for existing data
    if segments_dir.exists():
        segment_files = list(segments_dir.glob("*.jsonl"))
        st.success(f"Found {len(segment_files)} transcript files for {ticker}")

        # Load and display summary
        segments_df = load_transcript_data(ticker, data_dir)

        if not segments_df.empty:
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Total Segments", len(segments_df))
            with col2:
                st.metric("Unique Calls", segments_df["call_date"].nunique())
            with col3:
                st.metric("CEO Segments", len(segments_df[segments_df["role"] == "ceo"]))
            with col4:
                st.metric("CFO Segments", len(segments_df[segments_df["role"] == "cfo"]))

            with st.expander("📋 View Transcript Data", expanded=False):
                st.dataframe(
                    segments_df[["call_date", "speaker", "role", "text"]].head(50),
                    hide_index=True,
                    use_container_width=True
                )
    else:
        st.warning(f"No transcript data found for {ticker}")
        st.info("Transcript data should be in: `data/earnings/segments/{ticker}/`")

    # Data generation options
    st.markdown("#### 📥 Data Options")

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔧 Generate Mock Data", help="Create synthetic data for testing"):
            with st.spinner("Generating mock data..."):
                generate_mock_data(ticker, data_dir)
                st.success("Mock data generated!")
                st.rerun()

    with col2:
        if st.button("📡 Fetch from SEC", help="Attempt to fetch transcripts from SEC EDGAR"):
            st.info("SEC transcript fetching requires manual setup. See CLAUDE.md for details.")

    with col3:
        if st.button("📂 Upload Transcripts", help="Upload your own transcript files"):
            st.info("Use the file uploader below to add transcript JSONL files.")

    # File uploader
    uploaded_file = st.file_uploader(
        "Upload transcript JSONL file",
        type=["jsonl"],
        help="Upload a JSONL file with segments in format: {speaker, role, text}"
    )

    if uploaded_file:
        try:
            # Save uploaded file
            segments_dir.mkdir(parents=True, exist_ok=True)
            file_path = segments_dir / uploaded_file.name
            file_path.write_bytes(uploaded_file.getvalue())
            st.success(f"Uploaded {uploaded_file.name}")
            st.rerun()
        except Exception as e:
            st.error(f"Error saving file: {e}")


def generate_mock_data(ticker: str, data_dir: Path):
    """Generate mock transcript data for testing."""
    import random

    segments_dir = data_dir / "segments" / ticker
    segments_dir.mkdir(parents=True, exist_ok=True)

    # Common words that might be tracked
    tracked_words = ["AI", "cloud", "revenue", "growth", "innovation", "metaverse", "efficiency"]

    # Generate 8 quarters of mock data
    for q in range(8):
        year = 2024 - (q // 4)
        quarter = 4 - (q % 4)
        call_date = f"{year}-Q{quarter}"

        segments = []

        # CEO prepared remarks
        ceo_text = f"Welcome to our Q{quarter} {year} earnings call. "
        for word in random.sample(tracked_words, k=random.randint(2, 5)):
            ceo_text += f"We are seeing strong {word} momentum. "
        ceo_text += "Let me now turn it over to our CFO."

        segments.append({
            "speaker": "CEO",
            "role": "ceo",
            "text": ceo_text,
            "segment_idx": 0
        })

        # CFO remarks
        cfo_text = "Thank you. Looking at our financials, "
        for word in random.sample(tracked_words, k=random.randint(2, 4)):
            cfo_text += f"Our {word} initiatives delivered results. "
        cfo_text += "Now let's open for questions."

        segments.append({
            "speaker": "CFO",
            "role": "cfo",
            "text": cfo_text,
            "segment_idx": 1
        })

        # Q&A segments
        for i in range(3):
            analyst_text = f"Can you comment on your {random.choice(tracked_words)} strategy?"
            segments.append({
                "speaker": f"Analyst {i+1}",
                "role": "analyst",
                "text": analyst_text,
                "segment_idx": 2 + i * 2
            })

            exec_response = f"Great question. Our {random.choice(tracked_words)} approach is focused on "
            exec_response += f"delivering {random.choice(tracked_words)} to customers."
            segments.append({
                "speaker": "CEO" if i % 2 == 0 else "CFO",
                "role": "ceo" if i % 2 == 0 else "cfo",
                "text": exec_response,
                "segment_idx": 3 + i * 2
            })

        # Save to file
        file_path = segments_dir / f"{ticker}_{call_date}.jsonl"
        with open(file_path, "w") as f:
            for seg in segments:
                f.write(json.dumps(seg) + "\n")


def render_stats_section(ticker: str, params: Dict):
    """Render mention frequency statistics section."""
    st.markdown("### 📊 Mention Frequency Statistics")
    st.markdown("Analyze historical word mention patterns")

    data_dir = Path("data/earnings")
    contracts = st.session_state.contracts

    if not contracts:
        st.warning("Fetch contracts first to see statistics.")
        return

    # Load transcript data
    segments_df = load_transcript_data(ticker, data_dir)

    if segments_df.empty:
        st.warning("No transcript data available. Generate mock data or upload transcripts.")
        return

    # Speaker mode selection
    speaker_mode = st.selectbox(
        "Speaker Filter",
        ["executives_only", "ceo_only", "cfo_only", "full_transcript"],
        index=0,
        help="Which speakers to include in the analysis"
    )

    # Build features and outcomes
    with st.spinner("Analyzing transcripts..."):
        features_df, outcomes_df = build_features_and_outcomes(
            segments_df, contracts, speaker_mode
        )

    if features_df.empty:
        st.warning("Could not build features from transcript data.")
        return

    # Save to session state
    st.session_state.features_df = features_df
    st.session_state.outcomes_df = outcomes_df

    # Calculate statistics
    stats = calculate_mention_statistics(features_df, outcomes_df)
    st.session_state.mention_stats = stats

    # Display summary stats
    st.markdown("#### 📈 Summary Statistics")

    stats_df = pd.DataFrame([
        {
            "Word": word,
            "Mention Rate": f"{s['mention_frequency']:.0%}",
            "Avg Mentions": f"{s['avg_mentions_per_call']:.1f}",
            "Total": s['total_mentions'],
            "Max": s['max_mentions'],
            "Trend": "📈" if s['trend'] == "increasing" else "📉" if s['trend'] == "decreasing" else "➡️",
            "Recent Avg": f"{s['recent_3_avg']:.1f}",
        }
        for word, s in stats.items()
    ]).sort_values("Mention Rate", ascending=False)

    st.dataframe(stats_df, hide_index=True, use_container_width=True)

    # Visualization
    st.markdown("#### 📉 Mention Trends Over Time")

    # Select words to visualize
    available_words = list(features_df.columns)
    selected_words = st.multiselect(
        "Select words to visualize",
        available_words,
        default=available_words[:5] if len(available_words) > 5 else available_words,
        help="Choose which words to show in the chart"
    )

    if selected_words:
        chart_df = features_df[selected_words].copy()
        chart_df.index = pd.to_datetime(chart_df.index, errors='coerce')
        chart_df = chart_df.sort_index()

        st.line_chart(chart_df, height=350)

    # Binary outcomes chart
    st.markdown("#### ✅ Mention Frequency (Binary)")

    if selected_words:
        outcomes_chart = outcomes_df[selected_words].copy()
        outcomes_chart.index = pd.to_datetime(outcomes_chart.index, errors='coerce')
        outcomes_chart = outcomes_chart.sort_index()

        # Calculate rolling average for visualization
        rolling_avg = outcomes_chart.rolling(window=3, min_periods=1).mean()
        st.area_chart(rolling_avg, height=250)


def render_training_section(ticker: str, params: Dict):
    """Render model training section."""
    st.markdown("### 🧠 Model Training")
    st.markdown("Train Beta-Binomial prediction models on historical data")

    features_df = st.session_state.features_df
    outcomes_df = st.session_state.outcomes_df

    if features_df is None or features_df.empty:
        st.warning("Build statistics first (in the Statistics tab) before training.")
        return

    # Training info
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Training Samples", len(features_df))
    with col2:
        st.metric("Words to Train", len(features_df.columns))
    with col3:
        st.metric("Model Type", "Beta-Binomial")

    # Display current parameters
    with st.expander("⚙️ Training Parameters", expanded=True):
        st.json(params["model_params"])

    # Train button
    if st.button("🚀 Train Models", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()

        trained_models = []
        words = list(features_df.columns)

        for i, word in enumerate(words):
            status_text.text(f"Training model for '{word}'...")

            try:
                result = train_model_for_word(
                    features_df, outcomes_df, word, params["model_params"]
                )
                trained_models.append(result)
            except Exception as e:
                st.warning(f"Failed to train model for '{word}': {e}")

            progress_bar.progress((i + 1) / len(words))

        status_text.text("Training complete!")
        st.session_state.model_trained = True
        st.session_state.predictions = trained_models
        st.success(f"Successfully trained {len(trained_models)} models!")

    # Display trained model results
    if st.session_state.model_trained and st.session_state.predictions:
        st.markdown("#### 📊 Model Predictions")

        predictions = st.session_state.predictions
        pred_df = pd.DataFrame([
            {
                "Word": p["word"],
                "Predicted Prob": f"{p['predicted_probability']:.1%}",
                "95% CI Lower": f"{p['confidence_lower']:.1%}",
                "95% CI Upper": f"{p['confidence_upper']:.1%}",
                "Uncertainty": f"{p['uncertainty']:.3f}",
                "Observations": p["n_observations"],
            }
            for p in predictions
        ]).sort_values("Predicted Prob", ascending=False)

        st.dataframe(pred_df, hide_index=True, use_container_width=True)

        # Visualization
        st.markdown("#### 📈 Prediction Distribution")

        probs = [p["predicted_probability"] for p in predictions]
        lowers = [p["confidence_lower"] for p in predictions]
        uppers = [p["confidence_upper"] for p in predictions]
        words = [p["word"] for p in predictions]

        # Create a simple bar chart with error ranges
        chart_data = pd.DataFrame({
            "word": words,
            "probability": probs,
        }).set_index("word")

        st.bar_chart(chart_data, height=300)


def render_backtest_section(ticker: str, params: Dict):
    """Render backtesting section."""
    st.markdown("### 📈 Backtesting")
    st.markdown("Test trading strategy on historical data")

    features_df = st.session_state.features_df
    outcomes_df = st.session_state.outcomes_df

    if features_df is None or features_df.empty:
        st.warning("Build statistics first (in the Statistics tab) before backtesting.")
        return

    # Backtest parameters display
    with st.expander("⚙️ Backtest Parameters", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Model Parameters:**")
            st.json(params["model_params"])
        with col2:
            st.markdown("**Trading Parameters:**")
            st.json(params["backtest_params"])

    # Run backtest button
    if st.button("▶️ Run Backtest", type="primary"):
        with st.spinner("Running backtest..."):
            try:
                result = run_backtest(
                    ticker,
                    features_df,
                    outcomes_df,
                    params["model_params"],
                    params["backtest_params"]
                )
                st.session_state.backtest_result = result
                st.success("Backtest complete!")
            except Exception as e:
                st.error(f"Backtest failed: {e}")
                return

    # Display results
    result = st.session_state.backtest_result

    if not result:
        st.info("Click 'Run Backtest' to test your trading strategy.")
        return

    # Key metrics
    st.markdown("#### 📊 Performance Metrics")

    metrics = result["metrics"]

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        roi_color = "green" if metrics["roi"] > 0 else "red"
        st.metric(
            "💰 ROI",
            f"{metrics['roi']:.1%}",
            delta=f"{metrics['total_pnl']:.2f} P&L"
        )

    with col2:
        st.metric(
            "🎯 Win Rate",
            f"{metrics['win_rate']:.1%}",
            delta=f"{metrics['winning_trades']}/{metrics['total_trades']} trades"
        )

    with col3:
        st.metric(
            "📈 Sharpe Ratio",
            f"{metrics['sharpe_ratio']:.2f}"
        )

    with col4:
        st.metric(
            "📊 Accuracy",
            f"{metrics['accuracy']:.1%}",
            delta=f"{metrics['correct_predictions']}/{metrics['total_predictions']}"
        )

    # Additional metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Trades", metrics["total_trades"])
    with col2:
        st.metric("Avg P&L/Trade", f"${metrics['avg_pnl_per_trade']:.2f}")
    with col3:
        st.metric("Brier Score", f"{metrics['brier_score']:.4f}")
    with col4:
        final_cap = result["metadata"].get("final_capital", 10000)
        st.metric("Final Capital", f"${final_cap:,.2f}")

    # Trades table
    if result["trades"]:
        st.markdown("#### 💼 Trade History")

        trades_df = pd.DataFrame(result["trades"])
        trades_df["entry_price"] = trades_df["entry_price"].apply(lambda x: f"{x:.2%}")
        trades_df["predicted_probability"] = trades_df["predicted_probability"].apply(lambda x: f"{x:.1%}")
        trades_df["edge"] = trades_df["edge"].apply(lambda x: f"{x:+.1%}")
        trades_df["pnl"] = trades_df["pnl"].apply(lambda x: f"${x:+.2f}")
        trades_df["roi"] = trades_df["roi"].apply(lambda x: f"{x:+.1%}")

        st.dataframe(
            trades_df[["call_date", "contract", "side", "entry_price", "edge", "actual_outcome", "pnl", "roi"]],
            hide_index=True,
            use_container_width=True
        )

    # Predictions table
    with st.expander("📋 All Predictions", expanded=False):
        if result["predictions"]:
            pred_df = pd.DataFrame(result["predictions"])
            pred_df["predicted_probability"] = pred_df["predicted_probability"].apply(lambda x: f"{x:.1%}")
            pred_df["market_price"] = pred_df["market_price"].apply(lambda x: f"{x:.1%}" if x else "N/A")
            pred_df["edge"] = pred_df["edge"].apply(lambda x: f"{x:+.1%}" if x else "N/A")

            st.dataframe(
                pred_df[["call_date", "contract", "predicted_probability", "market_price", "edge", "actual_outcome", "correct"]],
                hide_index=True,
                use_container_width=True
            )


def render_predictions_section(ticker: str, params: Dict):
    """Render live predictions section."""
    st.markdown("### 🎯 Live Predictions")
    st.markdown("Generate predictions for upcoming earnings calls")

    if not st.session_state.model_trained:
        st.warning("Train models first (in the Training tab) before generating predictions.")
        return

    predictions = st.session_state.predictions
    contracts = st.session_state.contracts
    stats = st.session_state.mention_stats

    if not predictions:
        st.warning("No trained models available.")
        return

    st.markdown("#### 📊 Current Predictions")

    # Build predictions with market comparison
    pred_data = []

    for pred in predictions:
        word = pred["word"]
        prob = pred["predicted_probability"]

        # Find matching contract
        contract = next((c for c in contracts if c["word"] == word), None)

        # Get current market price if available
        market_price = None
        if contract:
            markets = contract.get("markets", [])
            active_markets = [m for m in markets if m.get("status", "").lower() in ("active", "open")]
            if active_markets:
                last_price = active_markets[0].get("last_price", 0)
                if isinstance(last_price, (int, float)):
                    market_price = last_price / 100 if last_price > 1 else last_price

        # Calculate edge
        edge = prob - market_price if market_price else None

        # Get recommendation
        if edge is not None:
            if edge >= params["backtest_params"]["edge_threshold"] and prob >= 0.60:
                recommendation = "BUY YES"
                rec_color = "🟢"
            elif edge <= -params["backtest_params"]["edge_threshold"] and prob <= 0.40:
                recommendation = "BUY NO"
                rec_color = "🔴"
            else:
                recommendation = "HOLD"
                rec_color = "⚪"
        else:
            recommendation = "NO DATA"
            rec_color = "⚪"

        # Get historical stats
        word_stats = stats.get(word, {})

        pred_data.append({
            "Rec": rec_color,
            "Word": word,
            "Predicted": f"{prob:.1%}",
            "Market": f"{market_price:.1%}" if market_price else "N/A",
            "Edge": f"{edge:+.1%}" if edge else "N/A",
            "Action": recommendation,
            "Hist Freq": f"{word_stats.get('mention_frequency', 0):.0%}",
            "Trend": "📈" if word_stats.get("trend") == "increasing" else "📉" if word_stats.get("trend") == "decreasing" else "➡️",
            "CI": f"[{pred['confidence_lower']:.0%}, {pred['confidence_upper']:.0%}]",
        })

    pred_df = pd.DataFrame(pred_data)

    # Color code by recommendation
    def highlight_recommendations(row):
        if row["Action"] == "BUY YES":
            return ["background-color: #d4edda"] * len(row)
        elif row["Action"] == "BUY NO":
            return ["background-color: #f8d7da"] * len(row)
        return [""] * len(row)

    styled_df = pred_df.style.apply(highlight_recommendations, axis=1)
    st.dataframe(styled_df, hide_index=True, use_container_width=True, height=400)

    # Export options
    st.markdown("#### 📥 Export")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("💾 Save Predictions to JSON"):
            output_dir = Path("data/earnings/predictions")
            output_dir.mkdir(parents=True, exist_ok=True)

            output_file = output_dir / f"{ticker}_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            with open(output_file, "w") as f:
                json.dump({
                    "ticker": ticker,
                    "timestamp": datetime.now().isoformat(),
                    "predictions": predictions,
                    "model_params": params["model_params"],
                }, f, indent=2)

            st.success(f"Saved to {output_file}")

    with col2:
        if st.button("📊 Load to Dashboard"):
            st.info("This will load predictions into the main dashboard for monitoring.")
            # TODO: Implement database loading


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------
def main():
    """Main application entry point."""
    # Render sidebar and get parameters
    params = render_sidebar()

    ticker = st.session_state.selected_ticker

    # Main header
    st.title("📞 Earnings Call Prediction Trainer")

    if not ticker:
        st.markdown("""
        ### Welcome to the Earnings Call Trainer

        This tool helps you train machine learning models to predict word mentions
        in corporate earnings calls for Kalshi prediction market trading.

        **Getting Started:**
        1. Select a ticker from the sidebar (e.g., META, TSLA, NVDA)
        2. Fetch available Kalshi contracts
        3. Load or generate transcript training data
        4. Analyze mention frequency statistics
        5. Train prediction models
        6. Run backtests to validate performance
        7. Generate predictions for upcoming earnings

        **Select a ticker in the sidebar to begin.**
        """)
        return

    # Get Kalshi client
    client = get_kalshi_client()

    if not client:
        st.error("Could not connect to Kalshi API. Check your credentials.")
        return

    # Display ticker header
    st.markdown(f"## {ticker} - Earnings Call Analysis")

    # Create tabs for different sections
    tabs = st.tabs([
        "📜 Contracts",
        "📚 Data",
        "📊 Statistics",
        "🧠 Training",
        "📈 Backtest",
        "🎯 Predictions"
    ])

    with tabs[0]:
        render_contracts_section(ticker, client)

    with tabs[1]:
        render_data_section(ticker)

    with tabs[2]:
        render_stats_section(ticker, params)

    with tabs[3]:
        render_training_section(ticker, params)

    with tabs[4]:
        render_backtest_section(ticker, params)

    with tabs[5]:
        render_predictions_section(ticker, params)


if __name__ == "__main__":
    main()
