"""
CLI for earnings call analysis.

Commands:
- fetch-transcripts: Fetch earnings call transcripts
- fetch-prices: Fetch stock price outcomes
- parse: Parse and segment transcripts
- featurize: Extract features from transcripts
- backtest: Run backtest on historical data
- predict: Make predictions for upcoming earnings
"""

from __future__ import annotations

import json
from pathlib import Path

import click
import pandas as pd

from ..config import get_config
from ..fetchers import (
    fetch_earnings_transcripts,
    fetch_price_outcomes,
    fetch_earnings_data,
)
from ..parsing import segment_earnings_transcript, parse_transcript
from ..features import featurize_earnings_calls
from ..models import BetaBinomialEarningsModel, SentimentBasedModel
from ..backtester import EarningsBacktester, save_backtest_result


@click.group()
def cli():
    """Earnings call analysis CLI."""
    pass


@cli.command()
@click.option("--ticker", required=True, help="Stock ticker symbol (e.g., COIN, GOOGL)")
@click.option("--num-quarters", default=8, help="Number of recent quarters to fetch")
@click.option("--source", default="auto", help="Data source: auto, sec, alpha_vantage")
@click.option("--output-dir", type=Path, default=None, help="Output directory")
@click.option("--alpha-vantage-key", default=None, help="Alpha Vantage API key")
def fetch_transcripts(ticker, num_quarters, source, output_dir, alpha_vantage_key):
    """Fetch earnings call transcripts for a ticker."""
    config = get_config()

    if output_dir is None:
        output_dir = config.transcripts_dir

    if alpha_vantage_key is None:
        alpha_vantage_key = config.alpha_vantage_api_key

    click.echo(f"Fetching {num_quarters} quarters of transcripts for {ticker}...")

    metadata_list = fetch_earnings_transcripts(
        ticker=ticker,
        output_dir=output_dir,
        num_quarters=num_quarters,
        source=source,
        alpha_vantage_key=alpha_vantage_key,
    )

    click.echo(f"Fetched {len(metadata_list)} transcripts")

    # Save metadata
    metadata_file = output_dir / ticker / "metadata.json"
    metadata_file.parent.mkdir(parents=True, exist_ok=True)

    with open(metadata_file, "w") as f:
        json.dump(
            [
                {
                    "ticker": m.ticker,
                    "fiscal_quarter": m.fiscal_quarter,
                    "call_date": m.call_date.isoformat() if m.call_date else None,
                    "source": m.source,
                    "has_transcript": m.has_transcript,
                }
                for m in metadata_list
            ],
            f,
            indent=2,
        )

    click.echo(f"Metadata saved to {metadata_file}")


@cli.command()
@click.option("--ticker", required=True, help="Stock ticker symbol")
@click.option("--metadata-file", type=Path, required=True, help="Transcript metadata JSON file")
@click.option("--horizons", default="1,5,10", help="Days after earnings to measure (comma-separated)")
@click.option("--output-file", type=Path, default=None, help="Output CSV file")
def fetch_prices(ticker, metadata_file, horizons, output_file):
    """Fetch stock price outcomes for earnings dates."""
    config = get_config()

    if output_file is None:
        output_file = config.outcomes_dir / f"{ticker}_outcomes.csv"

    # Load metadata
    with open(metadata_file, "r") as f:
        metadata = json.load(f)

    # Extract earnings dates
    from datetime import datetime
    earnings_dates = [
        datetime.fromisoformat(m["call_date"])
        for m in metadata
        if m.get("call_date")
    ]

    if not earnings_dates:
        click.echo("No earnings dates found in metadata")
        return

    # Parse horizons
    horizon_list = [int(h.strip()) for h in horizons.split(",")]

    click.echo(f"Fetching price outcomes for {len(earnings_dates)} earnings dates...")

    outcomes = fetch_price_outcomes(
        ticker=ticker,
        earnings_dates=earnings_dates,
        horizons=horizon_list,
    )

    if outcomes.empty:
        click.echo("No price data found")
        return

    # Save outcomes
    output_file.parent.mkdir(parents=True, exist_ok=True)
    outcomes.to_csv(output_file, index=False)

    click.echo(f"Price outcomes saved to {output_file}")
    click.echo(f"Shape: {outcomes.shape}")


@cli.command()
@click.option("--ticker", required=True, help="Stock ticker symbol")
@click.option("--input-dir", type=Path, required=True, help="Directory with transcript files")
@click.option("--output-dir", type=Path, default=None, help="Output directory for segments")
@click.option("--use-ai", is_flag=True, help="Use AI for improved segmentation")
def parse(ticker, input_dir, output_dir, use_ai):
    """Parse and segment earnings call transcripts."""
    config = get_config()

    if output_dir is None:
        output_dir = config.segments_dir / ticker

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find transcript files
    transcript_files = list(input_dir.glob("*.txt"))

    if not transcript_files:
        click.echo(f"No transcript files found in {input_dir}")
        return

    click.echo(f"Parsing {len(transcript_files)} transcripts...")

    for transcript_file in transcript_files:
        click.echo(f"Processing {transcript_file.name}...")

        # Parse transcript
        text = parse_transcript(transcript_file, clean=True)

        # Segment by speaker
        segments = segment_earnings_transcript(
            text,
            ticker=ticker,
            use_ai=use_ai,
            openai_api_key=config.openai_api_key if use_ai else None,
        )

        # Save segments
        output_file = output_dir / f"{transcript_file.stem}.jsonl"

        with open(output_file, "w") as f:
            for segment in segments:
                f.write(
                    json.dumps({
                        "speaker": segment.speaker,
                        "role": segment.role,
                        "text": segment.text,
                        "confidence": segment.confidence,
                        "company": segment.company,
                    }) + "\n"
                )

        click.echo(f"  Saved {len(segments)} segments to {output_file}")


@cli.command()
@click.option("--ticker", required=True, help="Stock ticker symbol")
@click.option("--segments-dir", type=Path, required=True, help="Directory with segment files")
@click.option("--keywords-config", type=Path, default=None, help="Keywords YAML config")
@click.option("--speaker-mode", default="executives_only", help="Speaker filter mode")
@click.option("--phrase-mode", default="variants", help="Phrase matching mode")
@click.option("--output-file", type=Path, default=None, help="Output parquet file")
def featurize(ticker, segments_dir, keywords_config, speaker_mode, phrase_mode, output_file):
    """Extract features from segmented transcripts."""
    config = get_config()

    if keywords_config is None:
        keywords_config = Path("configs/earnings/default_keywords.yaml")

    if output_file is None:
        output_file = config.features_dir / f"{ticker}_features.parquet"

    click.echo(f"Featurizing transcripts for {ticker}...")

    features = featurize_earnings_calls(
        segments_dir=segments_dir,
        ticker=ticker,
        keywords_config=keywords_config,
        speaker_mode=speaker_mode,
        phrase_mode=phrase_mode,
    )

    if features.empty:
        click.echo("No features extracted")
        return

    # Save features
    output_file.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_file)

    click.echo(f"Features saved to {output_file}")
    click.echo(f"Shape: {features.shape}")


@cli.command()
@click.option("--ticker", required=True, help="Stock ticker symbol")
@click.option("--features-file", type=Path, required=True, help="Features parquet file")
@click.option("--outcomes-file", type=Path, required=True, help="Outcomes CSV file")
@click.option("--model", default="beta", help="Model type: beta, sentiment")
@click.option("--edge-threshold", default=0.1, type=float, help="Minimum edge to trade")
@click.option("--initial-capital", default=10000, type=float, help="Starting capital")
@click.option("--output-dir", type=Path, default=None, help="Output directory")
def backtest(ticker, features_file, outcomes_file, model, edge_threshold, initial_capital, output_dir):
    """Run backtest on historical earnings data."""
    config = get_config()

    if output_dir is None:
        output_dir = Path("results/earnings") / ticker

    click.echo(f"Loading data for {ticker}...")

    # Load features
    features = pd.read_parquet(features_file)
    click.echo(f"Features: {features.shape}")

    # Load outcomes
    outcomes = pd.read_csv(outcomes_file)
    outcomes["call_date"] = pd.to_datetime(outcomes["earnings_date"])
    outcomes = outcomes.set_index("call_date")
    click.echo(f"Outcomes: {outcomes.shape}")

    # Select model
    if model == "beta":
        model_class = BetaBinomialEarningsModel
        model_params = {"alpha_prior": 1.0, "beta_prior": 1.0, "half_life": 4}
    elif model == "sentiment":
        model_class = SentimentBasedModel
        model_params = {}
    else:
        raise ValueError(f"Unknown model: {model}")

    click.echo(f"Running backtest with {model} model...")

    backtester = EarningsBacktester(
        features=features,
        outcomes=outcomes,
        model_class=model_class,
        model_params=model_params,
        edge_threshold=edge_threshold,
    )

    result = backtester.run(initial_capital=initial_capital)

    # Save results
    save_backtest_result(result, output_dir)

    click.echo(f"\nResults saved to {output_dir}")
    click.echo("\nMetrics:")
    for key, value in result.metrics.items():
        click.echo(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    cli()


@cli.command()
@click.option("--ticker", required=True, help="Stock ticker symbol (e.g., META, COIN)")
@click.option("--segments-dir", type=Path, required=True, help="Directory with transcript segments")
@click.option("--output-dir", type=Path, default=None, help="Output directory")
@click.option("--market-status", default="all", help="Filter by market status: open, resolved, all")
@click.option("--speaker-mode", default="executives_only", help="Speaker filter mode")
def analyze_kalshi_contracts(ticker, segments_dir, output_dir, market_status, speaker_mode):
    """Analyze Kalshi earnings mention contracts for a ticker."""
    import asyncio
    from fomc_analysis.kalshi_client_factory import get_kalshi_client
    from ..kalshi import analyze_earnings_kalshi_contracts
    
    config = get_config()
    
    if output_dir is None:
        output_dir = config.data_dir / "kalshi_analysis" / ticker
    
    client = get_kalshi_client()
    
    click.echo(f"Analyzing Kalshi contracts for {ticker}...")
    
    async def run_analysis():
        contracts, analyses = await analyze_earnings_kalshi_contracts(
            kalshi_client=client,
            ticker=ticker,
            segments_dir=segments_dir,
            output_dir=output_dir,
            market_status=market_status,
            speaker_mode=speaker_mode,
        )
        return contracts, analyses
    
    contracts, analyses = asyncio.run(run_analysis())
    
    click.echo(f"\nAnalysis complete!")
    click.echo(f"  Contracts found: {len(contracts)}")
    click.echo(f"  Analysis results: {len(analyses)}")
    click.echo(f"  Output: {output_dir}")


@cli.command()
@click.option("--ticker", required=True, help="Stock ticker symbol")
@click.option("--features-file", type=Path, required=True, help="Features parquet file")
@click.option("--contract-words-file", type=Path, required=True, help="Contract words JSON file")
@click.option("--model", default="beta", help="Model type: beta")
@click.option("--edge-threshold", default=0.12, type=float, help="Minimum edge to trade")
@click.option("--initial-capital", default=10000, type=float, help="Starting capital")
@click.option("--output-dir", type=Path, default=None, help="Output directory")
def backtest_kalshi(ticker, features_file, contract_words_file, model, edge_threshold, initial_capital, output_dir):
    """Run backtest on Kalshi earnings mention contracts."""
    from ..kalshi.backtester import EarningsKalshiBacktester, save_earnings_backtest_result
    from ..models import BetaBinomialEarningsModel
    
    config = get_config()
    
    if output_dir is None:
        output_dir = Path("results/earnings_kalshi") / ticker
    
    click.echo(f"Loading data for {ticker}...")
    
    # Load features
    features = pd.read_parquet(features_file)
    click.echo(f"Features: {features.shape}")
    
    # Load contract words
    with open(contract_words_file, "r") as f:
        contract_words_data = json.load(f)
    
    # Build outcomes dataframe from contract words
    # This would typically come from Kalshi historical data
    # For now, we'll need to fetch this separately
    click.echo("Note: You need to provide actual Kalshi contract outcomes")
    click.echo("This requires fetching resolved contract data from Kalshi API")
    
    # For demonstration, create a dummy outcomes frame
    # In real usage, this should be fetched from Kalshi
    click.echo("\nWARNING: Using placeholder outcomes. Integrate with Kalshi API for real data.")
    
    # Select model
    if model == "beta":
        model_class = BetaBinomialEarningsModel
        model_params = {"alpha_prior": 1.0, "beta_prior": 1.0, "half_life": 4}
    else:
        raise ValueError(f"Unknown model: {model}")
    
    click.echo(f"\nBacktest framework ready. Connect to Kalshi API to fetch actual outcomes.")
    click.echo(f"See FOMC framework's kalshi integration for reference.")


if __name__ == "__main__":
    cli()
