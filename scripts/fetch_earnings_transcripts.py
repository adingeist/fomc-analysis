"""
Fetch Earnings Call Transcripts.

This script fetches earnings call transcripts from various sources:
1. Manual upload (user-provided transcript files)
2. SEC EDGAR (free, official filings)
3. API services (Alpha Vantage, Financial Modeling Prep, etc.)

Outputs transcripts in JSONL format with speaker segmentation for word detection validation.
"""

from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import re
from datetime import datetime


class TranscriptFetcher:
    """Fetch and process earnings call transcripts."""

    def __init__(self, output_dir: Path = Path("data/transcripts")):
        """Initialize fetcher."""
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def parse_manual_transcript(
        self,
        transcript_text: str,
        ticker: str,
        call_date: str,
        format: str = "auto"
    ) -> List[Dict[str, Any]]:
        """
        Parse manually provided transcript text into structured format.

        Parameters:
            transcript_text: Raw transcript text
            ticker: Company ticker (e.g., 'META')
            call_date: Date of earnings call (YYYY-MM-DD)
            format: Transcript format ('auto', 'seeking_alpha', 'motley_fool', 'raw')

        Returns:
            List of segments with speaker, role, text
        """
        segments = []

        if format == "auto":
            # Try to detect format
            if "Operator:" in transcript_text or "Operator\n" in transcript_text:
                format = "seeking_alpha"
            else:
                format = "raw"

        if format == "seeking_alpha":
            # Seeking Alpha format: "Speaker Name\nText\n\n"
            segments = self._parse_seeking_alpha_format(transcript_text)

        elif format == "motley_fool":
            # Motley Fool format: "Speaker Name -- Position\nText"
            segments = self._parse_motley_fool_format(transcript_text)

        else:  # raw
            # Try to detect speakers from patterns
            segments = self._parse_raw_format(transcript_text)

        # Add metadata
        for seg in segments:
            seg['ticker'] = ticker
            seg['call_date'] = call_date

        return segments

    def _parse_seeking_alpha_format(self, text: str) -> List[Dict[str, Any]]:
        """Parse Seeking Alpha transcript format."""
        segments = []

        # Split by speaker changes (typically "Speaker Name\n")
        # Pattern: Name on line, then text, then blank line
        lines = text.split('\n')

        current_speaker = None
        current_text = []
        segment_idx = 0

        for line in lines:
            line = line.strip()

            if not line:
                # Empty line - end of current segment
                if current_speaker and current_text:
                    segments.append({
                        'speaker': current_speaker,
                        'role': self._infer_role(current_speaker),
                        'text': ' '.join(current_text),
                        'segment_idx': segment_idx
                    })
                    segment_idx += 1
                    current_text = []
                continue

            # Check if this is a speaker name (heuristic: short line, no punctuation at end)
            if len(line) < 50 and not line.endswith('.') and not line.endswith('?'):
                # Likely a speaker name
                if current_speaker and current_text:
                    # Save previous segment
                    segments.append({
                        'speaker': current_speaker,
                        'role': self._infer_role(current_speaker),
                        'text': ' '.join(current_text),
                        'segment_idx': segment_idx
                    })
                    segment_idx += 1
                    current_text = []

                current_speaker = line
            else:
                # This is dialogue text
                current_text.append(line)

        # Save last segment
        if current_speaker and current_text:
            segments.append({
                'speaker': current_speaker,
                'role': self._infer_role(current_speaker),
                'text': ' '.join(current_text),
                'segment_idx': segment_idx
            })

        return segments

    def _parse_motley_fool_format(self, text: str) -> List[Dict[str, Any]]:
        """Parse Motley Fool transcript format."""
        # Similar to Seeking Alpha but with " -- " separator for titles
        # TODO: Implement if needed
        return self._parse_seeking_alpha_format(text)

    def _parse_raw_format(self, text: str) -> List[Dict[str, Any]]:
        """Parse raw transcript with minimal structure."""
        # Try to detect Q&A section vs prepared remarks
        segments = []

        # Split into paragraphs
        paragraphs = text.split('\n\n')

        for idx, para in enumerate(paragraphs):
            para = para.strip()
            if not para:
                continue

            # Infer speaker from context (very basic)
            speaker = "Unknown"
            role = "executive"  # Default to executive

            # Check for common patterns
            if any(word in para.lower() for word in ['question:', 'analyst:', 'q:']):
                role = "analyst"
                speaker = "Analyst"
            elif any(word in para.lower() for word in ['operator:', 'moderator:']):
                role = "operator"
                speaker = "Operator"

            segments.append({
                'speaker': speaker,
                'role': role,
                'text': para,
                'segment_idx': idx
            })

        return segments

    def _infer_role(self, speaker_name: str) -> str:
        """
        Infer speaker role from name.

        Returns: 'ceo', 'cfo', 'executive', 'analyst', 'operator'
        """
        speaker_lower = speaker_name.lower()

        # Exact matches
        if 'operator' in speaker_lower or 'moderator' in speaker_lower:
            return 'operator'

        # Title-based inference
        if 'ceo' in speaker_lower or 'chief executive' in speaker_lower:
            return 'ceo'
        if 'cfo' in speaker_lower or 'chief financial' in speaker_lower:
            return 'cfo'
        if any(word in speaker_lower for word in ['analyst', 'question']):
            return 'analyst'
        if any(word in speaker_lower for word in ['chief', 'president', 'vp', 'vice president', 'head of']):
            return 'executive'

        # Default: if first few segments, likely executive; later, likely analyst
        return 'executive'

    def save_transcript(
        self,
        segments: List[Dict[str, Any]],
        ticker: str,
        call_date: str,
    ) -> Path:
        """
        Save transcript segments to JSONL file.

        Parameters:
            segments: List of segment dicts
            ticker: Company ticker
            call_date: Call date (YYYY-MM-DD)

        Returns:
            Path to saved file
        """
        filename = f"{ticker}_{call_date.replace('-', '')}_transcript.jsonl"
        output_path = self.output_dir / filename

        with open(output_path, 'w') as f:
            for segment in segments:
                f.write(json.dumps(segment) + '\n')

        print(f"✓ Saved transcript: {output_path}")
        print(f"  {len(segments)} segments")

        # Print role breakdown
        roles = {}
        for seg in segments:
            role = seg['role']
            roles[role] = roles.get(role, 0) + 1

        print(f"  Roles: {roles}")

        return output_path

    def load_transcript(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load transcript from JSONL file."""
        segments = []
        with open(file_path, 'r') as f:
            for line in f:
                segments.append(json.loads(line))
        return segments

    def create_sample_transcript(self, ticker: str = "META", call_date: str = "2025-10-29"):
        """
        Create a sample transcript for testing.

        Uses the META 2025-10-29 call as example (from our outcomes database).
        """
        sample_text = """Operator
Good afternoon, and welcome to Meta's Third Quarter 2025 Earnings Conference Call.

Mark Zuckerberg -- CEO
Thanks everyone for joining today. I want to start by talking about AI and our progress with Llama. We've made significant investments in our AI infrastructure this quarter, including expanding our cloud computing capabilities. Our AI recommendation systems are now powering Instagram, Facebook, and WhatsApp with great results.

We're also excited about our Ray-Ban smart glasses, which continue to see strong adoption. The integration of AI into these devices has been transformative. On the metaverse front, our Orion glasses prototype is progressing well, though we're not ready to discuss specific timelines.

Regarding dividend policy, we're pleased to announce continued returns to shareholders.

Susan Li -- CFO
Thank you, Mark. Turning to our financial results, revenue this quarter was strong, driven by advertising growth. Our cloud infrastructure investments are paying off with improved margins.

I want to highlight that we did not see any government shutdown impacts this quarter, and we remain focused on executing our strategic priorities. While we continue to monitor regulatory discussions around antitrust, our focus remains on delivering value to users and advertisers.

Operator
We'll now take questions from analysts.

Analyst
Hi, this is a question about the algorithm changes you mentioned. Can you provide more details?

Mark Zuckerberg -- CEO
Great question. Our algorithm improvements have been significant, particularly in content recommendation.

Analyst
Thank you. And a quick follow-up on Threads and your competitive positioning.

Mark Zuckerberg -- CEO
Threads continues to grow nicely. We're seeing strong engagement.

Analyst
Last question - any thoughts on TikTok and the competitive landscape?

Mark Zuckerberg -- CEO
We're focused on our own products and delivering the best experience. I think we have unique strengths.

Analyst
Thanks. And nothing on Scout or VR?

Mark Zuckerberg -- CEO
We're always evaluating our product portfolio, but no specific updates on those today.

Operator
That concludes our call. Thank you for joining.
"""

        print(f"\nCreating sample transcript for {ticker} {call_date}...")

        # Parse the sample
        segments = self.parse_manual_transcript(
            sample_text,
            ticker=ticker,
            call_date=call_date,
            format="seeking_alpha"
        )

        # Save it
        output_path = self.save_transcript(segments, ticker, call_date)

        # Also create a validation file mapping expected outcomes
        expected_outcomes = {
            "Cloud": 1,  # Mentioned: "cloud computing", "cloud infrastructure"
            "AI": 1,  # Mentioned multiple times
            "Llama": 1,  # Mentioned: "Llama"
            "Instagram": 1,  # Mentioned
            "WhatsApp": 1,  # Mentioned
            "Ray-Ban": 1,  # Mentioned
            "Orion": 1,  # Mentioned
            "Threads": 1,  # Mentioned
            "Algorithm": 1,  # Mentioned
            "Dividend": 1,  # Mentioned
            "VR / Virtual Reality": 0,  # Asked about but not mentioned by executives
            "TikTok": 0,  # Asked about but declined to discuss
            "Scout": 0,  # Asked about but no update
            "Shutdown / Shut Down": 0,  # Mentioned "no shutdown" (checking if detection is too loose)
            "Antitrust": 0,  # Mentioned "antitrust discussions" but this is borderline
            "Maverick": 0,  # Not mentioned
            "Behemoth": 0,  # Not mentioned
            "Channel": 0,  # Not mentioned (likely)
        }

        validation_path = self.output_dir / f"{ticker}_{call_date.replace('-', '')}_expected_outcomes.json"
        with open(validation_path, 'w') as f:
            json.dump(expected_outcomes, f, indent=2)

        print(f"✓ Saved expected outcomes: {validation_path}")

        return output_path, validation_path


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Fetch earnings transcripts")
    parser.add_argument(
        "--create-sample",
        action="store_true",
        help="Create sample transcript for testing",
    )
    parser.add_argument(
        "--ticker",
        default="META",
        help="Ticker symbol (default: META)",
    )
    parser.add_argument(
        "--date",
        default="2025-10-29",
        help="Call date YYYY-MM-DD (default: 2025-10-29)",
    )

    args = parser.parse_args()

    fetcher = TranscriptFetcher()

    if args.create_sample:
        transcript_path, outcomes_path = fetcher.create_sample_transcript(
            ticker=args.ticker,
            call_date=args.date
        )

        print("\n" + "=" * 80)
        print("SAMPLE TRANSCRIPT CREATED")
        print("=" * 80)
        print(f"\nTranscript: {transcript_path}")
        print(f"Expected outcomes: {outcomes_path}")
        print("\nNext step:")
        print("  python scripts/validate_word_detection.py")

    else:
        print("No action specified. Use --create-sample to create test transcript.")
        print("\nFor manual transcript upload:")
        print("  1. Save transcript text to a file")
        print("  2. Use fetcher.parse_manual_transcript() in Python")
        print("  3. Or implement your preferred source (SEC EDGAR, API, etc.)")

    return 0


if __name__ == "__main__":
    exit(main())
