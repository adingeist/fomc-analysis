"""OpenAI-powered prediction verification with web search capabilities."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import pandas as pd
from openai import OpenAI


class PredictionVerifier:
    """Verifies FOMC predictions using OpenAI with web search tools."""

    def __init__(self, api_key: str | None = None):
        """Initialize the verifier with OpenAI client.

        Args:
            api_key: OpenAI API key. If None, will use OPENAI_API_KEY env var.
        """
        self.client = OpenAI(api_key=api_key)

    def prepare_verification_data(
        self,
        predictions_df: pd.DataFrame,
        word_frequency_df: pd.DataFrame | None = None,
        word_frequency_summary_df: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Prepare prediction and word frequency data for verification.

        Args:
            predictions_df: DataFrame with predictions (from database)
            word_frequency_df: Optional DataFrame with word frequency timeseries
            word_frequency_summary_df: Optional DataFrame with word frequency summary stats

        Returns:
            Dictionary containing formatted data for OpenAI
        """
        # Convert predictions to a clean format
        predictions_data = []
        for _, row in predictions_df.iterrows():
            pred = {
                "contract": row.get("contract_ticker") or row.get("contract"),
                "meeting_date": str(row.get("meeting_date")),
                "predicted_probability": float(row.get("predicted_probability", 0)),
                "confidence_lower": float(row.get("confidence_lower", 0)),
                "confidence_upper": float(row.get("confidence_upper", 0)),
                "edge": float(row.get("edge", 0)),
                "recommendation": row.get("recommendation", "HOLD"),
            }

            # Add market data if available
            if "market_price" in row:
                pred["market_price"] = float(row["market_price"])
            if "last_price" in row:
                pred["last_price"] = float(row["last_price"])

            predictions_data.append(pred)

        result = {"predictions": predictions_data}

        # Add word frequency data if available
        if word_frequency_df is not None and not word_frequency_df.empty:
            # Get the most recent entries (last 5 meetings)
            recent_freq = word_frequency_df.tail(5).to_dict(orient="records")
            result["word_frequency_recent"] = recent_freq

        if word_frequency_summary_df is not None and not word_frequency_summary_df.empty:
            summary = word_frequency_summary_df.to_dict(orient="records")
            result["word_frequency_summary"] = summary

        return result

    def verify_predictions(
        self,
        verification_data: dict[str, Any],
        meeting_date: str | None = None,
    ) -> dict[str, Any]:
        """Verify predictions using OpenAI with web search capabilities.

        Args:
            verification_data: Prepared data from prepare_verification_data()
            meeting_date: Optional meeting date to focus the search on

        Returns:
            Dictionary with verification results including:
            - overall_assessment: str
            - confidence_level: str (High/Medium/Low)
            - concerns: list of str
            - confirmations: list of str
            - news_findings: list of dict
            - recommendation: str
        """
        # Build the prompt for OpenAI
        prompt = self._build_verification_prompt(verification_data, meeting_date)

        # Define the web search tool
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_web",
                    "description": "Search the web for news articles, trends, and information related to Federal Reserve, FOMC meetings, and economic terms",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query to find relevant news and information"
                            },
                            "focus": {
                                "type": "string",
                                "description": "What aspect to focus on: 'news', 'trends', or 'analysis'",
                                "enum": ["news", "trends", "analysis"]
                            }
                        },
                        "required": ["query"],
                        "additionalProperties": False
                    }
                }
            }
        ]

        # Make the initial request
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert financial analyst specializing in Federal Reserve policy and FOMC meetings. "
                    "Your task is to verify trading predictions by searching for relevant news, economic trends, "
                    "and market signals. Provide a thorough analysis that either confirms or casts doubt on the predictions. "
                    "Consider all variables including word frequency patterns, predicted probabilities, market prices, "
                    "and current events."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Simulate conversation with tool calls
        max_iterations = 10
        iteration = 0

        while iteration < max_iterations:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=tools,
                tool_choice="auto",
            )

            assistant_message = response.choices[0].message
            messages.append(assistant_message)

            # Check if the model wants to call a function
            if assistant_message.tool_calls:
                # Process each tool call
                for tool_call in assistant_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)

                    if function_name == "search_web":
                        # Simulate web search (in production, this would call a real search API)
                        search_result = self._simulate_web_search(
                            function_args.get("query", ""),
                            function_args.get("focus", "news"),
                            meeting_date
                        )

                        # Add the function result to messages
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "content": json.dumps(search_result)
                        })
            else:
                # No more tool calls, we have the final answer
                break

            iteration += 1

        # Parse the final response
        final_response = messages[-1].content if hasattr(messages[-1], 'content') else str(messages[-1])

        # Extract structured data from the response
        verification_result = self._parse_verification_response(final_response)

        return verification_result

    def _build_verification_prompt(
        self,
        verification_data: dict[str, Any],
        meeting_date: str | None
    ) -> str:
        """Build the verification prompt for OpenAI."""
        prompt_parts = [
            f"Please verify the following FOMC trading predictions for the meeting on {meeting_date or 'the upcoming date'}.",
            "",
            "# Predictions to Verify",
            json.dumps(verification_data["predictions"], indent=2),
        ]

        if "word_frequency_recent" in verification_data:
            prompt_parts.extend([
                "",
                "# Recent Word Frequency Trends (Last 5 Meetings)",
                json.dumps(verification_data["word_frequency_recent"], indent=2),
            ])

        if "word_frequency_summary" in verification_data:
            prompt_parts.extend([
                "",
                "# Word Frequency Summary Statistics",
                json.dumps(verification_data["word_frequency_summary"], indent=2),
            ])

        prompt_parts.extend([
            "",
            "# Your Task",
            "1. Use the search_web tool to find recent news about:",
            "   - Federal Reserve policy and statements",
            "   - Economic indicators and trends",
            "   - Specific terms mentioned in the contracts (e.g., 'inflation', 'recession', 'growth', etc.)",
            "   - FOMC member speeches and comments",
            "",
            "2. Analyze the predictions considering:",
            "   - Historical word frequency patterns",
            "   - Current economic context from your searches",
            "   - Market pricing vs. model predictions (edge)",
            "   - Confidence intervals and uncertainty",
            "",
            "3. For EACH prediction, provide:",
            "   - Whether you CONFIRM or have DOUBTS about it",
            "   - Specific reasons based on your research",
            "   - Relevant news or trends you discovered",
            "",
            "4. Conclude with:",
            "   - Overall confidence in the predictions (High/Medium/Low)",
            "   - Key concerns or risks",
            "   - Key confirmations or support",
            "   - Final recommendation (Proceed/Proceed with Caution/Reconsider)",
            "",
            "Be thorough in your research. Search for multiple angles and perspectives."
        ])

        return "\n".join(prompt_parts)

    def _simulate_web_search(
        self,
        query: str,
        focus: str,
        meeting_date: str | None
    ) -> dict[str, Any]:
        """Simulate a web search result.

        In production, this would call a real search API like:
        - Bing Search API
        - Google Custom Search
        - News API
        - etc.

        For now, return a structured placeholder that OpenAI can work with.
        """
        return {
            "query": query,
            "focus": focus,
            "search_performed": True,
            "note": (
                "This is a simulated search result. In production, this would connect to a real search API "
                "to fetch current news articles, economic data, and FOMC-related information. "
                "For demonstration purposes, please provide your analysis based on your training data "
                "and knowledge of Federal Reserve policy patterns."
            ),
            "suggested_approach": (
                f"Consider recent trends in '{query}' related to Federal Reserve policy. "
                f"Think about typical patterns before FOMC meetings and how '{query}' typically appears "
                "in Fed communications during different economic conditions."
            )
        }

    def _parse_verification_response(self, response_text: str) -> dict[str, Any]:
        """Parse the verification response into structured data.

        Args:
            response_text: The final response from OpenAI

        Returns:
            Structured verification results
        """
        # This is a simple parser - in production you might want to ask
        # OpenAI to return structured JSON directly

        result = {
            "overall_assessment": response_text,
            "confidence_level": "Medium",  # Default
            "concerns": [],
            "confirmations": [],
            "news_findings": [],
            "recommendation": "Proceed with Caution",
            "timestamp": datetime.now().isoformat(),
        }

        # Try to extract confidence level
        response_lower = response_text.lower()
        if "high confidence" in response_lower or "strongly confirm" in response_lower:
            result["confidence_level"] = "High"
        elif "low confidence" in response_lower or "significant doubt" in response_lower:
            result["confidence_level"] = "Low"

        # Try to extract recommendation
        if "proceed" in response_lower and "caution" not in response_lower.split("proceed")[1][:50]:
            result["recommendation"] = "Proceed"
        elif "reconsider" in response_lower or "do not proceed" in response_lower:
            result["recommendation"] = "Reconsider"

        return result


def export_verification_csv(
    predictions_df: pd.DataFrame,
    word_frequency_df: pd.DataFrame | None = None,
    output_path: str = "verification_data.csv"
) -> str:
    """Export predictions and word frequency to a single CSV for manual review.

    Args:
        predictions_df: Predictions DataFrame
        word_frequency_df: Word frequency DataFrame
        output_path: Where to save the CSV

    Returns:
        Path to the saved CSV file
    """
    # Start with predictions
    export_df = predictions_df.copy()

    # Add word frequency for each contract if available
    if word_frequency_df is not None and not word_frequency_df.empty:
        # Get the most recent word frequencies
        latest_freq = word_frequency_df.iloc[-1].to_dict()

        # Add as new columns
        for col, val in latest_freq.items():
            if col != "meeting_date":
                export_df[f"recent_freq_{col}"] = val

    # Save to CSV
    export_df.to_csv(output_path, index=False)

    return output_path
