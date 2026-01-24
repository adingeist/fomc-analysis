"""
Validate Word Detection Logic.

This script validates that our word counting/detection logic matches
Kalshi's settlement decisions by:
1. Loading earnings transcripts
2. Running our detection logic
3. Comparing against known outcomes
4. Calculating accuracy metrics

This ensures our framework will correctly predict contract outcomes.
"""

from pathlib import Path
from typing import List, Dict, Any, Tuple
import json
import pandas as pd
from collections import defaultdict


class WordDetectionValidator:
    """Validate word detection accuracy against known outcomes."""

    def __init__(self, transcripts_dir: Path = Path("data/transcripts")):
        """Initialize validator."""
        self.transcripts_dir = transcripts_dir

        if not transcripts_dir.exists():
            raise FileNotFoundError(
                f"Transcripts directory not found: {transcripts_dir}\n"
                "Run 'python scripts/fetch_earnings_transcripts.py --create-sample' first"
            )

    def count_word_mentions(
        self,
        text: str,
        word: str,
        case_sensitive: bool = False
    ) -> int:
        """
        Count mentions of a word or phrase in text.

        Handles multi-word phrases (e.g., "VR / Virtual Reality") by checking
        each variant separated by '/'.

        Parameters:
            text: Text to search in
            word: Word or phrase to search for
            case_sensitive: Whether to match case

        Returns:
            Number of mentions found
        """
        if not case_sensitive:
            text = text.lower()
            word = word.lower()

        # Handle multi-word phrases like "VR / Virtual Reality"
        if '/' in word:
            # Split on '/' and check each variant
            variants = [v.strip() for v in word.split('/')]
            total_count = 0

            for variant in variants:
                # Count occurrences of this variant
                # Use word boundaries to avoid partial matches
                import re
                # Escape special regex characters
                escaped = re.escape(variant)
                # Create word boundary pattern
                pattern = r'\b' + escaped + r'\b'

                try:
                    matches = re.findall(pattern, text, re.IGNORECASE if not case_sensitive else 0)
                    total_count += len(matches)
                except:
                    # Fallback to simple substring count
                    total_count += text.count(variant)

            return total_count

        else:
            # Single word/phrase
            import re
            escaped = re.escape(word)
            pattern = r'\b' + escaped + r'\b'

            try:
                matches = re.findall(pattern, text, re.IGNORECASE if not case_sensitive else 0)
                return len(matches)
            except:
                # Fallback
                return text.count(word)

    def detect_word_in_transcript(
        self,
        transcript_segments: List[Dict[str, Any]],
        word: str,
        speaker_filter: List[str] = ['ceo', 'cfo', 'executive']
    ) -> Tuple[bool, int, Dict[str, int]]:
        """
        Detect if word is mentioned in transcript.

        Parameters:
            transcript_segments: List of segment dicts with speaker, role, text
            word: Word to detect
            speaker_filter: Only count mentions from these roles

        Returns:
            (mentioned: bool, total_count: int, counts_by_role: dict)
        """
        total_count = 0
        counts_by_role = defaultdict(int)

        for segment in transcript_segments:
            role = segment.get('role', 'unknown')
            text = segment.get('text', '')

            # Count in this segment
            count = self.count_word_mentions(text, word)

            if count > 0:
                counts_by_role[role] += count
                total_count += count

        # Filter to executive roles only
        executive_count = sum(
            counts_by_role[role]
            for role in speaker_filter
            if role in counts_by_role
        )

        # Mentioned = at least 1 mention from executives
        mentioned = executive_count > 0

        return mentioned, total_count, dict(counts_by_role)

    def validate_transcript(
        self,
        transcript_path: Path,
        expected_outcomes_path: Path
    ) -> pd.DataFrame:
        """
        Validate word detection on a single transcript.

        Parameters:
            transcript_path: Path to transcript JSONL
            expected_outcomes_path: Path to expected outcomes JSON

        Returns:
            DataFrame with validation results
        """
        # Load transcript
        segments = []
        with open(transcript_path, 'r') as f:
            for line in f:
                segments.append(json.loads(line))

        # Load expected outcomes
        with open(expected_outcomes_path, 'r') as f:
            expected = json.load(f)

        # Validate each word
        results = []

        for word, expected_outcome in expected.items():
            # Run detection
            detected, total_count, counts_by_role = self.detect_word_in_transcript(
                segments,
                word,
                speaker_filter=['ceo', 'cfo', 'executive']
            )

            predicted_outcome = 1 if detected else 0

            # Calculate correctness
            correct = (predicted_outcome == expected_outcome)

            results.append({
                'word': word,
                'expected': expected_outcome,
                'predicted': predicted_outcome,
                'correct': correct,
                'total_mentions': total_count,
                'counts_by_role': counts_by_role,
            })

        return pd.DataFrame(results)

    def calculate_metrics(self, results_df: pd.DataFrame) -> Dict[str, float]:
        """
        Calculate classification metrics.

        Returns:
            Dict with accuracy, precision, recall, F1
        """
        if len(results_df) == 0:
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'num_samples': 0,
            }

        # True positives, false positives, etc.
        tp = len(results_df[(results_df['predicted'] == 1) & (results_df['expected'] == 1)])
        fp = len(results_df[(results_df['predicted'] == 1) & (results_df['expected'] == 0)])
        tn = len(results_df[(results_df['predicted'] == 0) & (results_df['expected'] == 0)])
        fn = len(results_df[(results_df['predicted'] == 0) & (results_df['expected'] == 1)])

        accuracy = (tp + tn) / len(results_df)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'num_samples': len(results_df),
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn,
        }

    def print_validation_report(
        self,
        results_df: pd.DataFrame,
        metrics: Dict[str, float],
        transcript_name: str
    ):
        """Print validation report to console."""
        print("=" * 80)
        print(f"WORD DETECTION VALIDATION: {transcript_name}")
        print("=" * 80)

        print("\nOVERALL METRICS:")
        print(f"  Accuracy:  {metrics['accuracy']:.1%} ({metrics['tp'] + metrics['tn']}/{metrics['num_samples']})")
        print(f"  Precision: {metrics['precision']:.1%}")
        print(f"  Recall:    {metrics['recall']:.1%}")
        print(f"  F1 Score:  {metrics['f1']:.1%}")

        print("\nCONFUSION MATRIX:")
        print(f"  True Positives:  {metrics['tp']}")
        print(f"  False Positives: {metrics['fp']}")
        print(f"  True Negatives:  {metrics['tn']}")
        print(f"  False Negatives: {metrics['fn']}")

        # Show incorrect predictions
        incorrect = results_df[~results_df['correct']]
        if len(incorrect) > 0:
            print(f"\nINCORRECT PREDICTIONS ({len(incorrect)}):")
            for _, row in incorrect.iterrows():
                print(f"  ✗ {row['word']}")
                print(f"      Expected: {row['expected']}, Predicted: {row['predicted']}")
                print(f"      Mentions: {row['total_mentions']}, By role: {row['counts_by_role']}")
        else:
            print("\n✓ ALL PREDICTIONS CORRECT!")

        # Show all results summary
        print("\nDETAILED RESULTS:")
        print(results_df[['word', 'expected', 'predicted', 'correct', 'total_mentions']].to_string(index=False))

    def validate_all_transcripts(self) -> pd.DataFrame:
        """
        Validate all transcripts in directory.

        Returns:
            Combined DataFrame with all results
        """
        # Find all transcript files
        transcript_files = list(self.transcripts_dir.glob("*_transcript.jsonl"))

        if len(transcript_files) == 0:
            print("No transcript files found.")
            print("Run 'python scripts/fetch_earnings_transcripts.py --create-sample' first")
            return pd.DataFrame()

        all_results = []

        for transcript_path in transcript_files:
            # Find corresponding expected outcomes file
            base_name = transcript_path.stem.replace('_transcript', '')
            outcomes_path = self.transcripts_dir / f"{base_name}_expected_outcomes.json"

            if not outcomes_path.exists():
                print(f"⚠ No expected outcomes for {transcript_path.name}, skipping")
                continue

            print(f"\nValidating {transcript_path.name}...")

            # Validate
            results_df = self.validate_transcript(transcript_path, outcomes_path)

            # Calculate metrics
            metrics = self.calculate_metrics(results_df)

            # Print report
            self.print_validation_report(results_df, metrics, transcript_path.stem)

            # Add to combined results
            results_df['transcript'] = transcript_path.stem
            all_results.append(results_df)

        if all_results:
            return pd.concat(all_results, ignore_index=True)
        else:
            return pd.DataFrame()


def main():
    """Main entry point."""
    validator = WordDetectionValidator()

    # Validate all transcripts
    all_results = validator.validate_all_transcripts()

    if not all_results.empty:
        # Calculate overall metrics
        overall_metrics = validator.calculate_metrics(all_results)

        print("\n" + "=" * 80)
        print("OVERALL VALIDATION SUMMARY")
        print("=" * 80)
        print(f"\nTotal words tested: {len(all_results)}")
        print(f"Overall accuracy: {overall_metrics['accuracy']:.1%}")
        print(f"Overall precision: {overall_metrics['precision']:.1%}")
        print(f"Overall recall: {overall_metrics['recall']:.1%}")
        print(f"Overall F1: {overall_metrics['f1']:.1%}")

        # Save results
        output_path = Path("data/word_detection_validation_results.csv")
        all_results.to_csv(output_path, index=False)
        print(f"\n✓ Saved results: {output_path}")

        # Interpretation
        print("\n" + "=" * 80)
        print("INTERPRETATION")
        print("=" * 80)

        if overall_metrics['accuracy'] >= 0.95:
            print("\n✓ EXCELLENT: >95% accuracy - word detection is highly reliable")
        elif overall_metrics['accuracy'] >= 0.90:
            print("\n✓ GOOD: >90% accuracy - word detection is reliable with minor issues")
        elif overall_metrics['accuracy'] >= 0.80:
            print("\n⚠ ACCEPTABLE: >80% accuracy - some issues but usable")
        else:
            print("\n✗ POOR: <80% accuracy - significant issues, needs improvement")

        if overall_metrics['fp'] > 0:
            print(f"\n⚠ {overall_metrics['fp']} false positives - detecting words that weren't mentioned")
            print("   → May cause us to buy YES on contracts that should be NO")

        if overall_metrics['fn'] > 0:
            print(f"\n⚠ {overall_metrics['fn']} false negatives - missing words that were mentioned")
            print("   → May cause us to miss YES opportunities or buy NO incorrectly")

    return 0


if __name__ == "__main__":
    exit(main())
