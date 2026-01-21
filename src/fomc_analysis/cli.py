"""
Comprehensive CLI for FOMC analysis pipeline.

This module provides all required CLI commands:
- fetch-transcripts: Download PDFs from Fed website
- parse: Extract and segment transcripts
- build-variants: Generate phrase variants with OpenAI
- featurize: Extract features from segments
- train: Train probability models
- backtest: Run walk-forward backtest
- report: Generate mispricing reports
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import click
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from .parsing.pdf_extractor import extract_pdf_to_text, clean_text
from .parsing.speaker_segmenter import (
    segment_speakers,
    save_segments_jsonl,
    load_segments_jsonl,
)
from .variants.generator import generate_variants
from .featurizer import (
    FeatureConfig,
    build_feature_matrix,
    load_contract_phrases,
)
from .models import EWMAModel, BetaBinomialModel
from .backtester_v2 import WalkForwardBacktester, save_backtest_result
from .fetcher import fetch_transcripts as fetch_transcripts_impl


# Load environment variables
load_dotenv()


@click.group()
def cli():
    """FOMC Press Conference Analysis Pipeline."""
    pass


@cli.command()
@click.option(
    "--start-year",
    type=int,
    default=2011,
    help="Start year for transcript fetching.",
)
@click.option(
    "--end-year",
    type=int,
    default=2025,
    help="End year for transcript fetching.",
)
@click.option(
    "--out-dir",
    type=click.Path(path_type=Path),
    default=Path("data/raw_pdf"),
    help="Output directory for PDFs.",
)
def fetch_transcripts(start_year: int, end_year: int, out_dir: Path):
    """
    Fetch FOMC press conference PDFs from Fed website.

    This command scrapes the Federal Reserve website to download all
    press conference transcript PDFs for the specified year range.
    """
    click.echo(f"Fetching transcripts for years {start_year}..{end_year}")
    click.echo(f"Output directory: {out_dir}")
    
    fetch_transcripts_impl(
        start_year=start_year,
        end_year=end_year,
        out_dir=out_dir,
        workers=8,
        overwrite=False,
        dry_run=False,
    )
    
    click.echo(f"✓ Done. Transcripts saved to {out_dir}")


@cli.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing PDF transcripts.",
)
@click.option(
    "--mode",
    type=click.Choice(["deterministic", "ai"]),
    default="deterministic",
    help="Parsing mode: deterministic (regex) or ai (OpenAI-assisted).",
)
@click.option(
    "--raw-text-dir",
    type=click.Path(path_type=Path),
    default=Path("data/raw_text"),
    help="Output directory for raw text.",
)
@click.option(
    "--clean-text-dir",
    type=click.Path(path_type=Path),
    default=Path("data/clean_text"),
    help="Output directory for cleaned text.",
)
@click.option(
    "--segments-dir",
    type=click.Path(path_type=Path),
    default=Path("data/segments"),
    help="Output directory for speaker segments (JSONL).",
)
def parse(
    input_dir: Path,
    mode: str,
    raw_text_dir: Path,
    clean_text_dir: Path,
    segments_dir: Path,
):
    """
    Parse PDF transcripts into structured speaker segments.

    Two-stage pipeline:
    1. Deterministic PDF extraction → raw text → clean text
    2. Speaker segmentation (deterministic or AI-assisted)
    """
    click.echo(f"Parsing transcripts from {input_dir}")

    # Create output directories
    raw_text_dir.mkdir(parents=True, exist_ok=True)
    clean_text_dir.mkdir(parents=True, exist_ok=True)
    segments_dir.mkdir(parents=True, exist_ok=True)

    # Initialize OpenAI client if using AI mode
    openai_client = None
    if mode == "ai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            click.echo("Error: OPENAI_API_KEY not found in environment")
            return
        openai_client = OpenAI(api_key=api_key)

    # Process each PDF
    pdf_files = sorted(input_dir.glob("*.pdf"))
    click.echo(f"Found {len(pdf_files)} PDF files")

    with click.progressbar(pdf_files, label="Processing PDFs") as bar:
        for pdf_path in bar:
            # Extract date from filename
            date_str = pdf_path.stem.replace("FOMCpresconf", "")

            # Stage A: Extract text
            raw_text = extract_pdf_to_text(pdf_path, include_page_markers=True)
            (raw_text_dir / f"{date_str}.txt").write_text(raw_text, encoding="utf-8")

            cleaned = clean_text(raw_text)
            (clean_text_dir / f"{date_str}.txt").write_text(cleaned, encoding="utf-8")

            # Stage B: Segment speakers
            use_ai = mode == "ai"
            segments = segment_speakers(
                cleaned,
                use_ai=use_ai,
                openai_client=openai_client,
            )

            # Save segments
            save_segments_jsonl(segments, segments_dir / f"{date_str}.jsonl")

    click.echo(f"✓ Parsed {len(pdf_files)} transcripts")


@cli.command()
@click.option(
    "--contracts",
    type=click.Path(exists=True, path_type=Path),
    default=Path("configs/contract_mapping.yaml"),
    help="Contract mapping YAML file.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("data/variants"),
    help="Output directory for variant cache files.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Force regeneration even if cached.",
)
def build_variants(contracts: Path, output_dir: Path, force: bool):
    """
    Generate phrase variants using OpenAI API.

    This command reads the contract mapping file and uses OpenAI to
    generate comprehensive phrase variants (plurals, possessives, etc.)
    for each contract. Results are cached to avoid redundant API calls.
    """
    import yaml

    # Load OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        click.echo("Error: OPENAI_API_KEY not found in environment")
        return

    client = OpenAI(api_key=api_key)

    # Load contract mapping
    data = yaml.safe_load(contracts.read_text())

    click.echo(f"Generating variants for {len(data)} contracts")

    output_dir.mkdir(parents=True, exist_ok=True)

    with click.progressbar(data.items(), label="Generating variants") as bar:
        for contract, entry in bar:
            base_phrases = entry.get("synonyms", [])

            result = generate_variants(
                contract=contract,
                base_phrases=base_phrases,
                openai_client=client,
                cache_dir=output_dir,
                force_regenerate=force,
            )

            click.echo(
                f"\n  {contract}: {len(result.variants)} variants "
                f"(from {len(base_phrases)} base phrases)"
            )

    click.echo(f"✓ Variants saved to {output_dir}")


@cli.command()
@click.option(
    "--segments-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing segment JSONL files.",
)
@click.option(
    "--contracts",
    type=click.Path(exists=True, path_type=Path),
    default=Path("configs/contract_mapping.yaml"),
    help="Contract mapping file.",
)
@click.option(
    "--variants-dir",
    type=click.Path(path_type=Path),
    default=Path("data/variants"),
    help="Variants cache directory.",
)
@click.option(
    "--speaker-mode",
    type=click.Choice(["powell_only", "full_transcript"]),
    default="powell_only",
    help="Which speakers to include.",
)
@click.option(
    "--phrase-mode",
    type=click.Choice(["strict", "variants"]),
    default="strict",
    help="Use strict base phrases or AI-generated variants.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    required=True,
    help="Output parquet file for features.",
)
def featurize(
    segments_dir: Path,
    contracts: Path,
    variants_dir: Path,
    speaker_mode: str,
    phrase_mode: str,
    output: Path,
):
    """
    Extract features from parsed transcripts.

    This command processes speaker segments and counts contract mentions
    according to the specified resolution mode (powell_only vs full,
    strict vs variants).
    """
    click.echo(f"Featurizing transcripts from {segments_dir}")

    # Load contract phrases
    click.echo(f"Loading contract phrases (mode: {phrase_mode})...")
    use_variants = phrase_mode == "variants"
    contract_phrases = load_contract_phrases(
        contracts,
        variants_dir=variants_dir if use_variants else None,
        use_variants=use_variants,
    )
    click.echo(f"  Loaded {len(contract_phrases)} contracts")

    # Count segment files
    segment_files = sorted(segments_dir.glob("*.jsonl"))
    click.echo(f"Found {len(segment_files)} transcript files to process")

    # Configure featurization
    config = FeatureConfig(
        speaker_mode=speaker_mode,
        phrase_mode=phrase_mode,
        case_sensitive=False,
        word_boundaries=True,
    )

    # Build feature matrix with progress bar
    from .featurizer import extract_features_from_segments
    rows = []

    with click.progressbar(segment_files, label="Extracting features") as bar:
        for segment_file in bar:
            # Extract date from filename
            date_str = segment_file.stem

            # Load segments
            segments = load_segments_jsonl(segment_file)

            # Extract features
            features = extract_features_from_segments(segments, contract_phrases, config)
            features["date"] = date_str
            rows.append(features)

    # Create DataFrame
    features = pd.DataFrame(rows)
    if "date" in features.columns:
        features = features.set_index("date")

    # Save to parquet
    output.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output)

    click.echo(f"✓ Features saved to {output}")
    click.echo(f"  Shape: {features.shape}")


@cli.command()
@click.option(
    "--features",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Feature matrix parquet file.",
)
@click.option(
    "--model",
    type=click.Choice(["ewma", "beta"]),
    default="beta",
    help="Model type to train.",
)
@click.option(
    "--alpha",
    type=float,
    default=0.5,
    help="Alpha parameter (EWMA smoothing or Beta prior).",
)
@click.option(
    "--beta-prior",
    type=float,
    default=1.0,
    help="Beta prior parameter (Beta-Binomial only).",
)
@click.option(
    "--half-life",
    type=int,
    default=None,
    help="Half-life for exponential decay (Beta-Binomial only).",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    required=True,
    help="Output JSON file for trained model.",
)
def train(
    features: Path,
    model: str,
    alpha: float,
    beta_prior: float,
    half_life: Optional[int],
    output: Path,
):
    """
    Train a mention probability model.

    This command trains a baseline model (EWMA or Beta-Binomial) on
    historical features and saves it for backtesting or prediction.
    """
    click.echo(f"Training {model} model on {features}")

    # Load features
    click.echo("Loading feature matrix...")
    df = pd.read_parquet(features)
    click.echo(f"  Loaded {df.shape[0]} transcripts with {df.shape[1]} features")

    # Convert to binary events (extract _mentioned columns)
    event_cols = [col for col in df.columns if col.endswith("_mentioned")]
    events = df[event_cols].copy()

    # Rename columns to remove _mentioned suffix
    events.columns = [col.replace("_mentioned", "") for col in events.columns]
    click.echo(f"  Processing {len(events.columns)} contracts")

    # Train model
    click.echo(f"Training {model.upper()} model...")
    if model == "ewma":
        model_obj = EWMAModel(alpha=alpha)
        click.echo(f"  Parameters: alpha={alpha}")
    else:  # beta
        model_obj = BetaBinomialModel(
            alpha_prior=alpha,
            beta_prior=beta_prior,
            half_life=half_life,
        )
        click.echo(f"  Parameters: alpha={alpha}, beta={beta_prior}, half_life={half_life}")

    model_obj.fit(events)
    click.echo("  Model fitting complete")

    # Save model
    output.parent.mkdir(parents=True, exist_ok=True)
    model_obj.save(output)

    click.echo(f"✓ Model saved to {output}")

    # Show prediction summary
    preds = model_obj.predict()
    click.echo(f"\nPrediction summary:")
    click.echo(preds.to_string(index=False))


@cli.command()
@click.option(
    "--features",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Feature matrix parquet file.",
)
@click.option(
    "--prices",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Historical prices CSV/parquet (optional).",
)
@click.option(
    "--model",
    type=click.Choice(["ewma", "beta"]),
    default="beta",
    help="Model type.",
)
@click.option(
    "--edge-threshold",
    type=float,
    default=0.05,
    help="Minimum edge to trade.",
)
@click.option(
    "--initial-capital",
    type=float,
    default=1000.0,
    help="Starting capital.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for backtest results.",
)
def backtest(
    features: Path,
    prices: Optional[Path],
    model: str,
    edge_threshold: float,
    initial_capital: float,
    output: Path,
):
    """
    Run walk-forward backtest with no-lookahead guarantees.

    This command simulates realistic trading using historical features
    and prices. At each time step, the model trains on past data and
    predicts the next event.
    """
    click.echo(f"Running backtest with {model.upper()} model")

    # Load features
    click.echo("Loading feature matrix...")
    df = pd.read_parquet(features)
    click.echo(f"  Loaded {df.shape[0]} dates")

    # Convert to binary events
    event_cols = [col for col in df.columns if col.endswith("_mentioned")]
    events = df[event_cols].copy()
    events.columns = [col.replace("_mentioned", "") for col in events.columns]
    click.echo(f"  Processing {len(events.columns)} contracts")

    # Load prices if provided
    prices_df = None
    if prices:
        click.echo(f"Loading market prices from {prices}...")
        if prices.suffix == ".parquet":
            prices_df = pd.read_parquet(prices)
        else:
            prices_df = pd.read_csv(prices, index_col=0, parse_dates=True)
        click.echo(f"  Loaded prices for {prices_df.shape[0]} dates")
    else:
        click.echo("No prices provided - will skip trade execution")

    # Create backtester
    click.echo(f"Initializing backtester with edge threshold {edge_threshold}...")
    backtester = WalkForwardBacktester(
        events=events,
        prices=prices_df,
        edge_threshold=edge_threshold,
    )

    # Select model class
    if model == "ewma":
        model_class = EWMAModel
        model_params = {"alpha": 0.5}
        click.echo(f"  Using EWMA model (alpha=0.5)")
    else:  # beta
        model_class = BetaBinomialModel
        model_params = {"alpha_prior": 1.0, "beta_prior": 1.0}
        click.echo(f"  Using Beta-Binomial model (alpha=1.0, beta=1.0)")

    # Run backtest
    min_train = 3
    num_steps = len(events) - min_train
    click.echo(f"\nRunning walk-forward backtest ({num_steps} time steps)...")
    click.echo("  This may take a while - training model at each step...")

    result = backtester.run(
        model_class=model_class,
        model_params=model_params,
        initial_capital=initial_capital,
    )

    click.echo(f"  Backtest execution complete - processed {len(result.trades)} trades")

    # Save results
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    click.echo(f"\nSaving results to {output}...")
    save_backtest_result(result, output / "backtest_results.json")

    # Save equity curve
    result.equity_curve.to_csv(output / "equity_curve.csv")

    # Print metrics
    click.echo(f"\n✓ Backtest complete")
    click.echo(f"\nMetrics:")
    for key, value in result.metrics.items():
        click.echo(f"  {key}: {value:.4f}")


@cli.command()
@click.option(
    "--results",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Backtest results directory.",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    required=True,
    help="Output CSV for mispricing table.",
)
def report(results: Path, output: Path):
    """
    Generate mispricing report from backtest results.

    This command analyzes backtest results and produces a table showing
    which contracts were mispriced and by how much.
    """
    import json

    click.echo(f"Generating report from {results}")

    # Load backtest results
    click.echo("Loading backtest results...")
    results_file = Path(results) / "backtest_results.json"
    data = json.loads(results_file.read_text())

    trades = data["trades"]
    click.echo(f"  Found {len(trades)} trades to analyze")

    # Aggregate by contract
    click.echo("Aggregating statistics by contract...")
    contract_stats = {}

    with click.progressbar(trades, label="Processing trades") as bar:
        for trade in bar:
            contract = trade["contract"]
            if contract not in contract_stats:
                contract_stats[contract] = {
                    "trades": 0,
                    "wins": 0,
                    "total_pnl": 0.0,
                    "avg_edge": 0.0,
                }

            stats = contract_stats[contract]
            stats["trades"] += 1
            if trade["pnl"] > 0:
                stats["wins"] += 1
            stats["total_pnl"] += trade["pnl"]
            stats["avg_edge"] += trade["edge"]

    click.echo(f"  Analyzed {len(contract_stats)} unique contracts")

    # Compute averages and create report
    click.echo("Computing final statistics...")
    report_rows = []
    for contract, stats in contract_stats.items():
        report_rows.append({
            "contract": contract,
            "trades": stats["trades"],
            "win_rate": stats["wins"] / stats["trades"],
            "total_pnl": stats["total_pnl"],
            "avg_edge": stats["avg_edge"] / stats["trades"],
        })

    report_df = pd.DataFrame(report_rows)
    report_df = report_df.sort_values("total_pnl", ascending=False)

    # Save report
    output.parent.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(output, index=False)

    click.echo(f"\n✓ Report saved to {output}")
    click.echo(f"\nTop mispriced contracts:")
    click.echo(report_df.head(10).to_string(index=False))


@cli.command()
@click.option(
    "--series-ticker",
    type=str,
    default="KXFEDMENTION",
    help="Kalshi series ticker (e.g., KXFEDMENTION).",
)
@click.option(
    "--event-ticker",
    type=str,
    default=None,
    help="Specific event ticker (e.g., kxfedmention-26jan). If not provided, fetches all markets in series.",
)
@click.option(
    "--segments-dir",
    type=click.Path(exists=True, path_type=Path),
    default=Path("data/segments"),
    help="Directory containing segmented transcripts.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("data/kalshi_analysis"),
    help="Output directory for analysis results.",
)
@click.option(
    "--scope",
    type=click.Choice(["powell_only", "full_transcript"]),
    default="powell_only",
    help="Search scope: powell_only or full_transcript.",
)
def analyze_kalshi_contracts(
    series_ticker: str,
    event_ticker: Optional[str],
    segments_dir: Path,
    output_dir: Path,
    scope: str,
):
    """
    Analyze Kalshi mention contracts against historical FOMC transcripts.

    This command:
    1. Fetches mention contracts from Kalshi API (e.g., KXFEDMENTION series)
    2. Extracts tracked words from market titles
    3. Generates word variants using OpenAI (plurals, possessives, compounds)
    4. Scans all FOMC transcripts for matches
    5. Builds statistical analysis of historical mention frequencies

    Requires KALSHI_API_KEY, KALSHI_API_SECRET, and OPENAI_API_KEY in environment.
    """
    from .kalshi_api import KalshiClient
    from .kalshi_contract_analyzer import run_kalshi_analysis

    # Load API clients
    click.echo("Loading API clients...")

    # Kalshi client
    try:
        kalshi_client = KalshiClient()
    except ValueError as e:
        click.echo(f"Error: {e}")
        click.echo("Please set KALSHI_API_KEY and KALSHI_API_SECRET in environment or .env file")
        return

    # OpenAI client
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        click.echo("Error: OPENAI_API_KEY not found in environment")
        return

    openai_client = OpenAI(api_key=api_key)

    # Run analysis
    try:
        run_kalshi_analysis(
            kalshi_client=kalshi_client,
            openai_client=openai_client,
            series_ticker=series_ticker,
            event_ticker=event_ticker,
            segments_dir=segments_dir,
            output_dir=output_dir,
            scope=scope,
        )
    except Exception as e:
        click.echo(f"Error during analysis: {e}")
        raise


if __name__ == "__main__":
    cli()
