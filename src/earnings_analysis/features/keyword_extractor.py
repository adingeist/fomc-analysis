"""
Keyword extraction and sentiment scoring for earnings calls.

Provides sentiment analysis and keyword importance scoring.
"""

from __future__ import annotations

from typing import Dict, List

import re


class SentimentScorer:
    """
    Score sentiment in earnings call text.

    Uses keyword-based scoring with positive/negative word lists.
    """

    POSITIVE_KEYWORDS = [
        "strong", "growth", "increased", "momentum", "exceeded", "confident",
        "optimistic", "improvement", "success", "opportunity", "innovative",
        "strength", "solid", "accelerate", "expand", "robust", "outperform",
        "ahead", "better", "record", "milestone", "breakthrough", "efficiency",
    ]

    NEGATIVE_KEYWORDS = [
        "headwind", "challenge", "pressure", "decline", "weakness", "concern",
        "difficult", "uncertainty", "risk", "slow", "lower", "decrease",
        "missed", "shortfall", "disappoint", "volatile", "compete", "threat",
        "deteriorate", "struggle", "impact", "negatively", "worse", "below",
    ]

    HEDGING_KEYWORDS = [
        "cautious", "uncertain", "potential", "possible", "maybe", "might",
        "could", "would", "should", "expect", "believe", "anticipate",
    ]

    def __init__(self):
        pass

    def score(self, text: str) -> Dict[str, float]:
        """
        Calculate sentiment scores for text.

        Parameters
        ----------
        text : str
            Text to score

        Returns
        -------
        Dict[str, float]
            Scores: positive_score, negative_score, net_score, hedging_score
        """
        text_lower = text.lower()
        word_count = len(text.split())

        if word_count == 0:
            return {
                "positive_score": 0.0,
                "negative_score": 0.0,
                "net_score": 0.0,
                "hedging_score": 0.0,
            }

        # Count positive keywords
        positive_count = sum(
            len(re.findall(r"\b" + re.escape(kw) + r"\b", text_lower))
            for kw in self.POSITIVE_KEYWORDS
        )

        # Count negative keywords
        negative_count = sum(
            len(re.findall(r"\b" + re.escape(kw) + r"\b", text_lower))
            for kw in self.NEGATIVE_KEYWORDS
        )

        # Count hedging keywords
        hedging_count = sum(
            len(re.findall(r"\b" + re.escape(kw) + r"\b", text_lower))
            for kw in self.HEDGING_KEYWORDS
        )

        # Normalize by word count
        positive_score = positive_count / word_count
        negative_score = negative_count / word_count
        net_score = positive_score - negative_score
        hedging_score = hedging_count / word_count

        return {
            "positive_score": positive_score,
            "negative_score": negative_score,
            "net_score": net_score,
            "hedging_score": hedging_score,
        }


class KeywordExtractor:
    """
    Extract important keywords from earnings calls.

    Can identify emerging themes and compare to historical calls.
    """

    def __init__(self, stopwords: List[str] = None):
        self.stopwords = stopwords or self._default_stopwords()

    def _default_stopwords(self) -> List[str]:
        """Default stopwords for financial text."""
        return [
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "been",
            "be", "have", "has", "had", "do", "does", "did", "will", "would",
            "should", "could", "may", "might", "must", "can", "this", "that",
            "these", "those", "i", "you", "he", "she", "it", "we", "they",
            "quarter", "year", "company", "business", "thank", "thanks",
        ]

    def extract_top_keywords(
        self,
        text: str,
        n: int = 50,
        min_length: int = 4,
    ) -> Dict[str, int]:
        """
        Extract top N keywords by frequency.

        Parameters
        ----------
        text : str
            Text to analyze
        n : int
            Number of top keywords to return
        min_length : int
            Minimum keyword length

        Returns
        -------
        Dict[str, int]
            Keywords and their frequencies
        """
        text_lower = text.lower()

        # Extract words
        words = re.findall(r"\b[a-z]+\b", text_lower)

        # Filter out stopwords and short words
        words = [
            w for w in words
            if w not in self.stopwords and len(w) >= min_length
        ]

        # Count frequencies
        freq = {}
        for word in words:
            freq[word] = freq.get(word, 0) + 1

        # Sort by frequency and return top N
        sorted_keywords = sorted(freq.items(), key=lambda x: x[1], reverse=True)

        return dict(sorted_keywords[:n])

    def compare_to_baseline(
        self,
        current_text: str,
        baseline_texts: List[str],
        n: int = 20,
    ) -> Dict[str, float]:
        """
        Find keywords that are significantly more frequent in current vs baseline.

        Useful for detecting new themes or topics.

        Parameters
        ----------
        current_text : str
            Current earnings call text
        baseline_texts : List[str]
            Historical earnings call texts
        n : int
            Number of keywords to return

        Returns
        -------
        Dict[str, float]
            Keywords and their relative increase
        """
        # Get keyword frequencies for current text
        current_freq = self.extract_top_keywords(current_text, n=200)
        current_total = sum(current_freq.values())

        # Get keyword frequencies for baseline
        baseline_text = " ".join(baseline_texts)
        baseline_freq = self.extract_top_keywords(baseline_text, n=200)
        baseline_total = sum(baseline_freq.values())

        if baseline_total == 0 or current_total == 0:
            return {}

        # Calculate relative increase
        relative_increase = {}

        for keyword in current_freq:
            current_rate = current_freq[keyword] / current_total
            baseline_rate = baseline_freq.get(keyword, 0) / baseline_total

            if baseline_rate > 0:
                increase = (current_rate - baseline_rate) / baseline_rate
            else:
                increase = float("inf") if current_rate > 0 else 0

            if increase > 0.5:  # At least 50% increase
                relative_increase[keyword] = increase

        # Sort and return top N
        sorted_increases = sorted(
            relative_increase.items(),
            key=lambda x: x[1],
            reverse=True
        )

        return dict(sorted_increases[:n])
