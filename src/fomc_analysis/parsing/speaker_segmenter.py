"""
Stage B: Speaker segmentation.

This module implements speaker turn parsing with two modes:
1. Deterministic regex-based segmentation (always used as baseline)
2. Optional OpenAI-powered repair and segmentation (with validation)

The AI mode must pass strict validation:
- Concatenated segments must match cleaned text
- Coverage must be ~100% (allowing whitespace differences)
- If validation fails, fall back to deterministic segmentation
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from openai import OpenAI

from .validation import compute_text_similarity


# Speaker detection patterns (ALL CAPS + period/colon)
SPEAKER_PATTERN = re.compile(
    r"^([A-Z][A-Z\s\.\,]+?)[\:\.](?:\s|$)",
    re.MULTILINE
)

# Known speaker roles
POWELL_PATTERNS = [
    r"CHAIR\s+POWELL",
    r"CHAIRMAN\s+POWELL",
    r"POWELL",
]

REPORTER_PATTERNS = [
    r"MR\.\s+\w+",
    r"MS\.\s+\w+",
    r"REPORTER",
]

MODERATOR_PATTERNS = [
    r"MODERATOR",
    r"MR\.\s+ENGLISH",  # Common FOMC moderator
]


@dataclass
class SpeakerTurn:
    """Represents a single speaker turn in a transcript."""
    speaker: str
    role: str  # "powell", "reporter", "moderator", "other"
    text: str
    start_page: Optional[int] = None
    end_page: Optional[int] = None
    confidence: float = 1.0  # Confidence score (1.0 for deterministic)


def classify_speaker_role(speaker: str) -> str:
    """
    Classify a speaker label into a role category.

    Parameters
    ----------
    speaker : str
        Speaker label (e.g., "CHAIR POWELL", "MR. SMITH")

    Returns
    -------
    str
        One of: "powell", "reporter", "moderator", "other"
    """
    speaker_upper = speaker.upper()

    for pattern in POWELL_PATTERNS:
        if re.search(pattern, speaker_upper):
            return "powell"

    for pattern in MODERATOR_PATTERNS:
        if re.search(pattern, speaker_upper):
            return "moderator"

    for pattern in REPORTER_PATTERNS:
        if re.search(pattern, speaker_upper):
            return "reporter"

    return "other"


def segment_speakers_deterministic(text: str) -> List[SpeakerTurn]:
    """
    Segment text into speaker turns using deterministic regex patterns.

    This function looks for lines that start with ALL CAPS speaker labels
    followed by a colon or period (e.g., "CHAIR POWELL:" or "MR. SMITH.").

    Parameters
    ----------
    text : str
        Cleaned transcript text.

    Returns
    -------
    List[SpeakerTurn]
        List of speaker turns in order.
    """
    turns: List[SpeakerTurn] = []
    current_speaker = "Unknown"
    current_role = "other"
    buffer: List[str] = []

    lines = text.split("\n")

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Check if this line starts with a speaker label
        match = SPEAKER_PATTERN.match(line)
        if match:
            # Save previous turn
            if buffer:
                turn_text = " ".join(buffer).strip()
                if turn_text:
                    turns.append(SpeakerTurn(
                        speaker=current_speaker,
                        role=current_role,
                        text=turn_text,
                        confidence=1.0,
                    ))
                buffer = []

            # Extract new speaker
            speaker_label = match.group(1).strip()
            current_speaker = speaker_label
            current_role = classify_speaker_role(speaker_label)

            # Remainder after speaker label
            remainder = line[match.end():].strip()
            if remainder:
                buffer.append(remainder)
        else:
            # Continuation of current speaker
            buffer.append(line)

    # Save final turn
    if buffer:
        turn_text = " ".join(buffer).strip()
        if turn_text:
            turns.append(SpeakerTurn(
                speaker=current_speaker,
                role=current_role,
                text=turn_text,
                confidence=1.0,
            ))

    return turns


def segment_speakers_with_ai(
    text: str,
    openai_client: OpenAI,
    model: str = "gpt-4o-mini",
    max_retries: int = 2,
) -> Optional[List[SpeakerTurn]]:
    """
    Segment text into speaker turns using OpenAI API.

    This function uses GPT to identify speaker boundaries and segment the
    transcript. The output must be valid JSON with strict structure.

    Parameters
    ----------
    text : str
        Cleaned transcript text.
    openai_client : OpenAI
        Configured OpenAI client.
    model : str, default="gpt-4o-mini"
        OpenAI model to use.
    max_retries : int, default=2
        Number of retry attempts if parsing fails.

    Returns
    -------
    Optional[List[SpeakerTurn]]
        List of speaker turns, or None if parsing failed.
    """
    prompt = """You are parsing an FOMC press conference transcript.

Your task is to segment the text into speaker turns. Each turn should have:
- speaker: The speaker's name/label (e.g., "CHAIR POWELL", "MR. SMITH")
- text: The exact text spoken by that speaker

Return ONLY a JSON array of objects, each with "speaker" and "text" fields.
Do NOT add, remove, or modify any words from the original text.
Do NOT include any explanation or markdown formatting.

Example output format:
[
  {"speaker": "CHAIR POWELL", "text": "Good afternoon..."},
  {"speaker": "MR. SMITH", "text": "Thank you..."}
]

Transcript to segment:
"""

    full_prompt = prompt + "\n\n" + text[:15000]  # Limit to avoid token limits

    for attempt in range(max_retries + 1):
        try:
            response = openai_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a precise transcript parser. Output only valid JSON."},
                    {"role": "user", "content": full_prompt},
                ],
                temperature=0.0,  # Deterministic
            )

            content = response.choices[0].message.content.strip()

            # Remove markdown code blocks if present
            content = re.sub(r"^```json\s*", "", content)
            content = re.sub(r"^```\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

            # Parse JSON
            data = json.loads(content)

            # Convert to SpeakerTurn objects
            turns = []
            for item in data:
                speaker = item.get("speaker", "Unknown")
                text_content = item.get("text", "")
                role = classify_speaker_role(speaker)

                turns.append(SpeakerTurn(
                    speaker=speaker,
                    role=role,
                    text=text_content,
                    confidence=0.9,  # AI-generated
                ))

            return turns

        except (json.JSONDecodeError, KeyError) as e:
            if attempt == max_retries:
                return None
            continue

    return None


def validate_segments(
    segments: List[SpeakerTurn],
    original_text: str,
    similarity_threshold: float = 0.95,
) -> bool:
    """
    Validate that speaker segments match the original text.

    This checks that:
    1. Concatenated segment text is similar to original text
    2. Coverage is high (allowing for whitespace normalization)

    Parameters
    ----------
    segments : List[SpeakerTurn]
        Speaker turns to validate.
    original_text : str
        Original cleaned text.
    similarity_threshold : float, default=0.95
        Minimum similarity score required (0-1).

    Returns
    -------
    bool
        True if validation passed, False otherwise.
    """
    # Concatenate all segment text
    concatenated = " ".join(turn.text for turn in segments)

    # Normalize both texts for comparison
    def normalize(text: str) -> str:
        # Remove extra whitespace, lowercase, remove punctuation
        text = text.lower()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s]", "", text)
        return text.strip()

    norm_original = normalize(original_text)
    norm_concatenated = normalize(concatenated)

    # Compute similarity
    similarity = compute_text_similarity(norm_original, norm_concatenated)

    # Check coverage (ratio of lengths)
    coverage = len(norm_concatenated) / max(len(norm_original), 1)

    return similarity >= similarity_threshold and coverage >= 0.90


def segment_speakers(
    text: str,
    use_ai: bool = False,
    openai_client: Optional[OpenAI] = None,
    model: str = "gpt-4o-mini",
) -> List[SpeakerTurn]:
    """
    Segment transcript text into speaker turns.

    This function tries deterministic segmentation first. If `use_ai=True`
    and an OpenAI client is provided, it will attempt AI-powered segmentation
    with validation. If AI segmentation fails validation, it falls back to
    deterministic segmentation.

    Parameters
    ----------
    text : str
        Cleaned transcript text.
    use_ai : bool, default=False
        Whether to attempt AI-powered segmentation.
    openai_client : Optional[OpenAI]
        OpenAI client for AI segmentation (required if use_ai=True).
    model : str, default="gpt-4o-mini"
        OpenAI model to use for AI segmentation.

    Returns
    -------
    List[SpeakerTurn]
        Speaker turns (deterministic or AI-powered).
    """
    # Always compute deterministic segmentation as fallback
    deterministic_turns = segment_speakers_deterministic(text)

    if not use_ai or openai_client is None:
        return deterministic_turns

    # Attempt AI segmentation
    ai_turns = segment_speakers_with_ai(text, openai_client, model)

    if ai_turns is None:
        # AI parsing failed, use deterministic
        return deterministic_turns

    # Validate AI output
    if validate_segments(ai_turns, text):
        return ai_turns
    else:
        # Validation failed, fall back to deterministic
        return deterministic_turns


def save_segments_jsonl(segments: List[SpeakerTurn], output_path: Path) -> None:
    """
    Save speaker segments to JSONL file.

    Parameters
    ----------
    segments : List[SpeakerTurn]
        Speaker turns to save.
    output_path : Path
        Path to output JSONL file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for turn in segments:
            record = {
                "speaker": turn.speaker,
                "role": turn.role,
                "text": turn.text,
                "start_page": turn.start_page,
                "end_page": turn.end_page,
                "confidence": turn.confidence,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_segments_jsonl(input_path: Path) -> List[SpeakerTurn]:
    """
    Load speaker segments from JSONL file.

    Parameters
    ----------
    input_path : Path
        Path to JSONL file.

    Returns
    -------
    List[SpeakerTurn]
        Speaker turns loaded from file.
    """
    turns = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            turns.append(SpeakerTurn(
                speaker=record["speaker"],
                role=record["role"],
                text=record["text"],
                start_page=record.get("start_page"),
                end_page=record.get("end_page"),
                confidence=record.get("confidence", 1.0),
            ))
    return turns
