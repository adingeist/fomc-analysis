# OpenAI Prediction Verification Feature

## Overview

This feature adds AI-powered verification of FOMC predictions using OpenAI's GPT-4 with web search capabilities. The system analyzes predictions, word frequency data, and searches for relevant news to either confirm or cast doubt on the model's predictions.

## Features

- **AI Analysis**: Uses OpenAI GPT-4 to analyze predictions in context
- **Web Search Tools**: AI can search for relevant news and economic trends
- **Historical Context**: Incorporates word frequency patterns from past meetings
- **CSV Export**: Export predictions and word frequency data for manual review
- **Confidence Scoring**: Provides confidence levels (High/Medium/Low) and recommendations

## How to Use

### 1. In the Dashboard

1. Navigate to the **🎯 Predictions** tab
2. Scroll down to the **🤖 AI Prediction Verification** section
3. Enter your OpenAI API key
4. (Optional) Check "Export CSV" to save data locally
5. Click **🔍 Verify Predictions**
6. Review the verification results:
   - Confidence Level
   - Overall Recommendation
   - Full Analysis
   - Concerns & Risks
   - Confirmations & Support

### 2. Required Setup

**OpenAI API Key:**
- Get your API key from https://platform.openai.com/api-keys
- The feature uses GPT-4, which requires API credits
- Alternatively, set the `OPENAI_API_KEY` environment variable

## Implementation Details

### Files Added/Modified

1. **`src/fomc_analysis/dashboard/prediction_verifier.py`** (NEW)
   - `PredictionVerifier` class: Main verification logic
   - `prepare_verification_data()`: Formats predictions and word frequency
   - `verify_predictions()`: Calls OpenAI with web search tools
   - `export_verification_csv()`: Exports data to CSV

2. **`dashboard/app.py`** (MODIFIED)
   - Added verification section in Predictions tab
   - Integrated with existing prediction and word frequency data
   - Results display with confidence levels and recommendations

### How It Works

1. **Data Preparation**:
   - Collects current predictions (probability, edge, recommendations)
   - Gathers recent word frequency trends (last 5 meetings)
   - Includes summary statistics for each contract

2. **OpenAI Analysis**:
   - Sends data to GPT-4 with a specialized prompt
   - Provides a `search_web` tool for the AI to use
   - AI searches for relevant news, Fed speeches, economic indicators
   - AI analyzes predictions considering all variables

3. **Results**:
   - Overall confidence level (High/Medium/Low)
   - Specific confirmations or doubts for each prediction
   - Final recommendation (Proceed/Proceed with Caution/Reconsider)
   - Relevant news findings

## Enhancing with Real Web Search

Currently, the `search_web` tool is simulated. To enable real web search, integrate one of these APIs:

### Option 1: Bing Search API

```python
import os
import requests

def _real_web_search(self, query: str, focus: str, meeting_date: str | None) -> dict:
    """Real web search using Bing Search API."""
    subscription_key = os.getenv("BING_SEARCH_API_KEY")
    endpoint = "https://api.bing.microsoft.com/v7.0/search"

    # Enhance query with focus
    if focus == "news":
        query += " FOMC Federal Reserve news"
    elif focus == "trends":
        query += " economic trend analysis"

    headers = {"Ocp-Apim-Subscription-Key": subscription_key}
    params = {
        "q": query,
        "mkt": "en-US",
        "count": 10,
        "freshness": "Month"  # Recent articles
    }

    response = requests.get(endpoint, headers=headers, params=params)
    response.raise_for_status()

    results = response.json()
    articles = []

    for page in results.get("webPages", {}).get("value", [])[:5]:
        articles.append({
            "title": page.get("name"),
            "url": page.get("url"),
            "snippet": page.get("snippet"),
            "date": page.get("datePublished")
        })

    return {
        "query": query,
        "focus": focus,
        "articles": articles,
        "search_performed": True
    }
```

### Option 2: Google Custom Search

```python
import os
import requests

def _real_web_search(self, query: str, focus: str, meeting_date: str | None) -> dict:
    """Real web search using Google Custom Search API."""
    api_key = os.getenv("GOOGLE_SEARCH_API_KEY")
    search_engine_id = os.getenv("GOOGLE_SEARCH_ENGINE_ID")
    endpoint = "https://www.googleapis.com/customsearch/v1"

    params = {
        "key": api_key,
        "cx": search_engine_id,
        "q": query + " FOMC Federal Reserve",
        "num": 10
    }

    response = requests.get(endpoint, params=params)
    response.raise_for_status()

    results = response.json()
    articles = []

    for item in results.get("items", [])[:5]:
        articles.append({
            "title": item.get("title"),
            "url": item.get("link"),
            "snippet": item.get("snippet")
        })

    return {
        "query": query,
        "articles": articles,
        "search_performed": True
    }
```

### Option 3: News API

```python
import os
import requests
from datetime import datetime, timedelta

def _real_web_search(self, query: str, focus: str, meeting_date: str | None) -> dict:
    """Real web search using News API."""
    api_key = os.getenv("NEWS_API_KEY")
    endpoint = "https://newsapi.org/v2/everything"

    # Search last 30 days
    from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    params = {
        "apiKey": api_key,
        "q": query + " AND (Federal Reserve OR FOMC OR Jerome Powell)",
        "from": from_date,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 10
    }

    response = requests.get(endpoint, params=params)
    response.raise_for_status()

    results = response.json()
    articles = []

    for article in results.get("articles", [])[:5]:
        articles.append({
            "title": article.get("title"),
            "url": article.get("url"),
            "snippet": article.get("description"),
            "source": article.get("source", {}).get("name"),
            "published": article.get("publishedAt")
        })

    return {
        "query": query,
        "articles": articles,
        "search_performed": True
    }
```

## To Integrate Real Search:

1. Choose a search API provider
2. Sign up and get API keys
3. Replace `_simulate_web_search()` in `prediction_verifier.py` with real implementation
4. Add API key to environment variables
5. Update the tool result format to include actual article data

## Cost Considerations

- **OpenAI API**: GPT-4 costs vary by usage (~$0.01-0.03 per verification)
- **Search APIs**:
  - Bing: Free tier available (1000 queries/month)
  - Google: Free tier available (100 queries/day)
  - News API: Free tier available (100 requests/day)

## Security Notes

- Never commit API keys to version control
- Use environment variables or Streamlit secrets
- Consider rate limiting for production use
- Monitor API usage and costs

## Future Enhancements

- [ ] Add caching for verification results
- [ ] Support batch verification of multiple meetings
- [ ] Historical verification tracking (compare AI vs actual outcomes)
- [ ] Integration with additional data sources (Fed minutes, speeches)
- [ ] Customizable verification prompts
- [ ] Export verification reports to PDF
