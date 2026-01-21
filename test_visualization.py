#!/usr/bin/env python3
"""
Test script to verify visualization functionality works.

This creates mock data and tests the chart generation without
needing to run the full pipeline.
"""

import json
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


def create_mock_data():
    """Create mock data for testing."""
    print("Creating mock data...")

    # Create directories
    Path("test_data/segments").mkdir(parents=True, exist_ok=True)
    Path("test_results").mkdir(parents=True, exist_ok=True)

    # Create mock segments
    dates = pd.date_range('2024-01-01', periods=6, freq='2ME')  # ME = Month End

    for date in dates:
        date_str = date.strftime('%Y%m%d')
        segment_file = Path(f"test_data/segments/{date_str}.jsonl")

        segments = [
            {
                "speaker": "CHAIR POWELL",
                "role": "powell",
                "text": "Inflation inflation inflation inflation remains elevated. Unemployment is stable. Volatility has decreased.",
                "confidence": 1.0
            },
            {
                "speaker": "REPORTER",
                "role": "reporter",
                "text": "Question about inflation and unemployment?",
                "confidence": 1.0
            }
        ]

        with segment_file.open('w') as f:
            for seg in segments:
                f.write(json.dumps(seg) + '\n')

    # Create mock backtest results
    results = {
        "predictions": [],
        "trades": [
            {"contract": "Inflation 40+", "meeting_date": "2024-03-01", "days_before_meeting": 7, "pnl": 100},
            {"contract": "Unemployment", "meeting_date": "2024-03-01", "days_before_meeting": 7, "pnl": 50},
            {"contract": "Volatility", "meeting_date": "2024-05-01", "days_before_meeting": 14, "pnl": -30},
            {"contract": "Inflation 40+", "meeting_date": "2024-07-01", "days_before_meeting": 7, "pnl": 75},
        ],
        "horizon_metrics": {},
        "overall_metrics": {
            "total_trades": 4,
            "total_pnl": 195,
            "roi": 0.195,
            "win_rate": 0.75,
            "sharpe": 1.2,
            "avg_pnl_per_trade": 48.75,
            "final_capital": 11950,
        },
        "metadata": {}
    }

    results_file = Path("test_results/backtest_results.json")
    results_file.write_text(json.dumps(results, indent=2))

    # Create mock contract mapping
    contract_mapping = {
        "Inflation 40+": {
            "synonyms": ["inflation"],
            "threshold": 40,
        },
        "Unemployment": {
            "synonyms": ["unemployment"],
            "threshold": 1,
        },
        "Volatility": {
            "synonyms": ["volatility"],
            "threshold": 1,
        },
    }

    mapping_file = Path("test_data/contract_mapping.yaml")
    import yaml
    mapping_file.write_text(yaml.dump(contract_mapping))

    print("✓ Mock data created")
    return results_file, mapping_file


def test_frequency_chart():
    """Test the frequency chart generation."""
    print("\n" + "=" * 80)
    print("TESTING VISUALIZATION")
    print("=" * 80)

    # Create mock data
    results_file, mapping_file = create_mock_data()

    # Load results
    results = json.loads(results_file.read_text())

    # Get traded contracts
    traded_contracts = list(set(trade["contract"] for trade in results["trades"]))
    print(f"\nTraded contracts: {traded_contracts}")

    # Load contract mapping
    import yaml
    contract_words = yaml.safe_load(mapping_file.read_text())

    # Load segment data
    from fomc_analysis.parsing.speaker_segmenter import load_segments_jsonl
    from fomc_analysis.featurizer import count_phrase_mentions

    segments_dir = Path("test_data/segments")
    records = []

    for segment_file in sorted(segments_dir.glob("*.jsonl")):
        date_str = segment_file.stem
        segments = load_segments_jsonl(segment_file)

        powell_segments = [s for s in segments if s.role == "powell"]
        powell_text = " ".join(seg.text for seg in powell_segments)

        records.append({
            "date": pd.to_datetime(date_str, format="%Y%m%d"),
            "text": powell_text,
        })

    segments_df = pd.DataFrame(records).sort_values("date")
    print(f"\nLoaded {len(segments_df)} meeting transcripts")

    # Count mentions
    mention_data = []

    for _, row in segments_df.iterrows():
        text = row["text"]
        date = row["date"]

        for contract_name, contract_def in contract_words.items():
            synonyms = contract_def.get("synonyms", [])
            if not synonyms:
                continue

            count = count_phrase_mentions(
                text=text,
                phrases=synonyms,
                case_sensitive=False,
                word_boundaries=True,
            )

            mention_data.append({
                "date": date,
                "contract": contract_name,
                "count": count,
            })

    mentions_df = pd.DataFrame(mention_data)
    mentions_pivot = mentions_df.pivot_table(
        index="date",
        columns="contract",
        values="count",
        aggfunc="sum",
        fill_value=0
    )

    print("\nMention counts:")
    print(mentions_pivot)

    # Generate chart
    available_contracts = [c for c in traded_contracts if c in mentions_pivot.columns]

    if not available_contracts:
        print("\n✗ No traded contracts found in mentions data")
        return False

    plot_data = mentions_pivot[available_contracts]

    fig, ax = plt.subplots(figsize=(14, 8))

    for contract in available_contracts:
        ax.plot(
            plot_data.index,
            plot_data[contract],
            marker='o',
            linewidth=2,
            markersize=6,
            label=contract,
            alpha=0.7,
        )

    ax.set_xlabel("FOMC Meeting Date", fontsize=12, fontweight='bold')
    ax.set_ylabel("Mention Count (Powell Only)", fontsize=12, fontweight='bold')
    ax.set_title(
        "Word Frequencies for Traded Contracts Across FOMC Meetings",
        fontsize=14,
        fontweight='bold',
        pad=20
    )
    ax.legend(
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        fontsize=10,
        frameon=True,
        shadow=True
    )
    ax.grid(True, alpha=0.3, linestyle='--')

    import matplotlib.dates as mdates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    # Save
    output_path = Path("test_results/word_frequencies.pdf")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Chart saved to: {output_path}")

    png_path = output_path.with_suffix('.png')
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"✓ Chart saved to: {png_path}")

    plt.close()

    print("\n" + "=" * 80)
    print("✓ VISUALIZATION TEST PASSED")
    print("=" * 80)

    # Cleanup
    print("\nCleaning up test data...")
    import shutil
    if Path("test_data").exists():
        shutil.rmtree("test_data")
    if Path("test_results").exists():
        shutil.rmtree("test_results")
    print("✓ Cleanup complete")

    return True


if __name__ == "__main__":
    try:
        success = test_frequency_chart()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
