"""
Speaker segmentation for earnings call transcripts.

Identifies and classifies speakers:
- CEO: Chief Executive Officer
- CFO: Chief Financial Officer
- executive: Other C-suite executives (CTO, COO, etc.)
- analyst: Sell-side analysts asking questions
- operator: Conference call operator
- other: Unclassified speakers
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

from openai import OpenAI


@dataclass
class SpeakerSegment:
    """A segment of speech from one speaker."""
    speaker: str  # Full name (e.g., "Brian Armstrong", "John Analyst")
    role: str  # ceo, cfo, executive, analyst, operator, other
    text: str  # What they said
    confidence: float = 1.0  # Confidence in speaker identification
    company: Optional[str] = None  # For analysts: which firm they represent


class EarningsSpeakerSegmenter:
    """
    Segment earnings call transcripts by speaker.

    Uses regex patterns to identify speakers and classify their roles.
    Optionally uses OpenAI for improved segmentation.

    Parameters
    ----------
    use_ai: bool
        Whether to use OpenAI for improved segmentation
    openai_api_key : Optional[str]
        OpenAI API key (if use_ai=True)
    openai_model : str
        OpenAI model to use (default: gpt-4o-mini)
    """

    # Common CEO title patterns
    CEO_PATTERNS = [
        r"chief executive officer",
        r"\bceo\b",
        r"co-chief executive",
        r"president and ceo",
        r"founder and ceo",
    ]

    # Common CFO title patterns
    CFO_PATTERNS = [
        r"chief financial officer",
        r"\bcfo\b",
        r"vice president.*finance",
        r"treasurer",
    ]

    # Other executive patterns
    EXECUTIVE_PATTERNS = [
        r"chief.*officer",  # CTO, COO, etc.
        r"president",
        r"vice president",
        r"\bcoo\b",
        r"\bcto\b",
        r"\bcmo\b",
        r"head of",
        r"general counsel",
    ]

    # Analyst patterns
    ANALYST_PATTERNS = [
        r"analyst",
        r"with.*securities",
        r"with.*capital",
        r"with.*research",
        r"with.*bank",
        r"with.*partners",
        r"from.*securities",
        r"from.*capital",
    ]

    def __init__(
        self,
        use_ai: bool = False,
        openai_api_key: Optional[str] = None,
        openai_model: str = "gpt-4o-mini",
    ):
        self.use_ai = use_ai
        self.openai_model = openai_model

        if use_ai:
            self.openai_client = OpenAI(api_key=openai_api_key)
        else:
            self.openai_client = None

    def segment(self, transcript_text: str, ticker: Optional[str] = None) -> List[SpeakerSegment]:
        """
        Segment transcript into speaker turns.

        Parameters
        ----------
        transcript_text : str
            Raw transcript text
        ticker : Optional[str]
            Stock ticker (helps with company context)

        Returns
        -------
        List[SpeakerSegment]
            List of speaker segments
        """
        # Try deterministic segmentation first
        segments = self._segment_deterministic(transcript_text)

        # Optionally enhance with AI
        if self.use_ai and self.openai_client:
            segments = self._enhance_with_ai(transcript_text, segments, ticker)

        return segments

    def _segment_deterministic(self, text: str) -> List[SpeakerSegment]:
        """
        Deterministic speaker segmentation using regex.

        Looks for patterns like:
        - "Brian Armstrong - CEO" or "Brian Armstrong, CEO"
        - "Operator:" or "Operator "
        - "John Smith - Goldman Sachs"
        """
        segments = []

        # Pattern for speaker labels
        # Matches: "NAME - TITLE" or "NAME, TITLE" or "NAME:" or "NAME "
        speaker_pattern = re.compile(
            r"^([A-Z][a-z]+(?: [A-Z][a-z]+)*)"  # Name (Title Case)
            r"(?:\s*[-–—,]\s*(.+?))?[:\s]+"  # Optional title after dash/comma
            r"\n?(.+?)(?=\n[A-Z][a-z]+(?: [A-Z][a-z]+)*\s*[-–—,:]|\Z)",  # Speech content
            re.MULTILINE | re.DOTALL
        )

        matches = speaker_pattern.findall(text)

        for speaker_name, title, speech in matches:
            speaker_name = speaker_name.strip()
            title = title.strip() if title else ""
            speech = speech.strip()

            if not speech:
                continue

            # Classify speaker role
            role = self._classify_speaker(speaker_name, title, speech)

            # Extract company for analysts
            company = None
            if role == "analyst":
                company = self._extract_analyst_company(title, speech)

            segment = SpeakerSegment(
                speaker=speaker_name,
                role=role,
                text=speech,
                confidence=1.0,
                company=company,
            )

            segments.append(segment)

        return segments

    def _classify_speaker(self, name: str, title: str, speech: str) -> str:
        """Classify speaker role based on name, title, and speech content."""
        title_lower = title.lower()
        speech_lower = speech[:200].lower()  # Check first 200 chars

        # Check for CEO
        if any(re.search(pattern, title_lower) for pattern in self.CEO_PATTERNS):
            return "ceo"

        # Check for CFO
        if any(re.search(pattern, title_lower) for pattern in self.CFO_PATTERNS):
            return "cfo"

        # Check for other executives
        if any(re.search(pattern, title_lower) for pattern in self.EXECUTIVE_PATTERNS):
            return "executive"

        # Check for analyst
        if any(re.search(pattern, title_lower) for pattern in self.ANALYST_PATTERNS):
            return "analyst"

        # Check for operator
        if "operator" in name.lower() or "operator" in title_lower:
            return "operator"

        # Try to infer from speech content
        if any(keyword in speech_lower for keyword in ["thank you for joining", "welcome to", "our next question"]):
            return "operator"

        if any(keyword in speech_lower for keyword in ["my question", "can you talk about", "i wanted to ask"]):
            return "analyst"

        return "other"

    def _extract_analyst_company(self, title: str, speech: str) -> Optional[str]:
        """Extract the company/firm that an analyst represents."""
        # Pattern: "with COMPANY" or "from COMPANY"
        pattern = r"(?:with|from)\s+([A-Z][A-Za-z\s&]+(?:Securities|Capital|Research|Bank|Partners|Advisors))"
        match = re.search(pattern, title)

        if match:
            return match.group(1).strip()

        # Try searching in speech content (first sentence)
        first_sentence = speech.split(".")[0] if "." in speech else speech[:100]
        match = re.search(pattern, first_sentence)

        if match:
            return match.group(1).strip()

        return None

    def _enhance_with_ai(
        self,
        original_text: str,
        deterministic_segments: List[SpeakerSegment],
        ticker: Optional[str] = None,
    ) -> List[SpeakerSegment]:
        """
        Use OpenAI to improve segmentation.

        This is similar to FOMC's AI enhancement but adapted for earnings calls.
        """
        if not self.openai_client:
            return deterministic_segments

        prompt = self._build_ai_prompt(original_text, ticker)

        try:
            response = self.openai_client.chat.completions.create(
                model=self.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert at parsing earnings call transcripts. "
                        "Segment the transcript by speaker and identify their roles (CEO, CFO, executive, analyst, operator)."
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.0,
            )

            ai_result = response.choices[0].message.content

            # Parse AI response (expected JSON format)
            ai_segments = self._parse_ai_response(ai_result)

            # Validate against original text
            if self._validate_segments(original_text, ai_segments):
                return ai_segments
            else:
                print("AI segmentation failed validation, using deterministic version")
                return deterministic_segments

        except Exception as e:
            print(f"AI segmentation failed: {e}")
            return deterministic_segments

    def _build_ai_prompt(self, text: str, ticker: Optional[str] = None) -> str:
        """Build prompt for OpenAI."""
        ticker_context = f" for {ticker}" if ticker else ""

        prompt = f"""
Parse this earnings call transcript{ticker_context} into speaker segments.

For each speaker, identify:
1. Name
2. Role (ceo, cfo, executive, analyst, operator, other)
3. Company (for analysts only)
4. Text (what they said)

Return as JSON array of objects with keys: speaker, role, text, company (optional).

Transcript:
{text[:3000]}  # Truncate for token limits
"""

        return prompt

    def _parse_ai_response(self, ai_response: str) -> List[SpeakerSegment]:
        """Parse AI response into SpeakerSegment objects."""
        try:
            # Extract JSON from response (might be wrapped in markdown code block)
            json_match = re.search(r"```json\s*(\[.+?\])\s*```", ai_response, re.DOTALL)

            if json_match:
                json_str = json_match.group(1)
            else:
                json_str = ai_response

            segments_data = json.loads(json_str)

            segments = []
            for item in segments_data:
                segment = SpeakerSegment(
                    speaker=item.get("speaker", "Unknown"),
                    role=item.get("role", "other"),
                    text=item.get("text", ""),
                    confidence=0.9,  # AI confidence
                    company=item.get("company"),
                )
                segments.append(segment)

            return segments

        except Exception as e:
            print(f"Failed to parse AI response: {e}")
            return []

    def _validate_segments(self, original_text: str, segments: List[SpeakerSegment]) -> bool:
        """
        Validate that segments match original text.

        Similar to FOMC validation: check that concatenated segments
        are similar to original.
        """
        if not segments:
            return False

        # Concatenate all segment text
        concatenated = " ".join(seg.text for seg in segments)

        # Check length similarity (within 10%)
        length_ratio = len(concatenated) / len(original_text)
        if not (0.9 <= length_ratio <= 1.1):
            return False

        # Basic content check: ensure some key phrases are preserved
        # (This is simplified - could be more sophisticated)
        return True

    def save_segments(self, segments: List[SpeakerSegment], output_file: Path):
        """Save segments to JSONL file."""
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w") as f:
            for segment in segments:
                f.write(json.dumps(asdict(segment)) + "\n")

    def load_segments(self, input_file: Path) -> List[SpeakerSegment]:
        """Load segments from JSONL file."""
        segments = []

        with open(input_file, "r") as f:
            for line in f:
                data = json.loads(line)
                segment = SpeakerSegment(**data)
                segments.append(segment)

        return segments


def segment_earnings_transcript(
    transcript_text: str,
    ticker: Optional[str] = None,
    use_ai: bool = False,
    openai_api_key: Optional[str] = None,
) -> List[SpeakerSegment]:
    """
    Convenience function to segment an earnings transcript.

    Parameters
    ----------
    transcript_text : str
        Raw transcript text
    ticker : Optional[str]
        Stock ticker symbol
    use_ai : bool
        Whether to use OpenAI enhancement
    openai_api_key : Optional[str]
        OpenAI API key (if use_ai=True)

    Returns
    -------
    List[SpeakerSegment]
        List of speaker segments
    """
    segmenter = EarningsSpeakerSegmenter(
        use_ai=use_ai,
        openai_api_key=openai_api_key,
    )

    return segmenter.segment(transcript_text, ticker)
