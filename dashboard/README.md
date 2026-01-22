# FOMC Trading Predictions Dashboard

An enhanced Streamlit dashboard for FOMC mention predictions with clear, actionable trading signals.

## Features

### 🎯 Predictions Tab (Main View)
- **Clear Recommendations**: BUY YES, BUY NO, or HOLD for each prediction
- **Color-Coded Display**: Green for BUY YES, red for BUY NO, white for HOLD
- **Top Opportunities**: Detailed view of best trading signals
- **Smart Filtering**: Filter by meeting date, recommendation type, and edge
- **Key Metrics**: Total predictions, action items, next meeting, best edge, days until meeting
- **Refresh Button**: One-click refresh to fetch latest data from Kalshi

### 📈 Backtest Results
- Overall performance metrics (ROI, Sharpe ratio, Win rate, Total PnL)
- Performance breakdown by prediction horizon (7, 14, 30 days)
- Historical predictions and trades
- Grid search parameter optimization results

### ⚙️ Configurable Settings
- **Trade Thresholds**: Customize BUY YES and BUY NO edge requirements
- **Probability Filters**: Set minimum probability for YES trades, maximum for NO trades
- **Real-time Adjustments**: Change thresholds and see recommendations update instantly

## Quick Start

1. **Install dependencies** (if not already installed):
   ```bash
   uv pip install -e .
   ```

2. **Ensure you have prediction data loaded**:
   ```bash
   # Generate predictions from Kalshi
   python -m fomc_analysis.cli predict-upcoming \
     --contract-words data/kalshi_analysis/contract_words.json \
     --output results/upcoming_predictions

   # Load predictions into database
   python scripts/load_upcoming_predictions.py
   ```

3. **Run the dashboard**:
   ```bash
   streamlit run dashboard/app.py
   ```

4. **Open in browser**:
   The dashboard will automatically open at `http://localhost:8501`

## Trade Recommendation Logic

The dashboard uses the following logic to generate recommendations:

### BUY YES
Recommended when:
- Predicted probability ≥ 60% (configurable)
- Edge ≥ 15% (configurable)
- Meaning: Model predicts YES outcome with high confidence AND market is underpricing it

### BUY NO
Recommended when:
- Predicted probability ≤ 40% (configurable)
- Edge ≤ -12% (configurable)
- Meaning: Model predicts NO outcome with high confidence AND market is overpricing YES

### HOLD
All other cases where the edge is not significant enough or probability is uncertain.

## Configuration Options

### In Sidebar
- **BUY YES Edge Threshold**: Minimum edge required for BUY YES (default: 15%)
- **BUY NO Edge Threshold**: Minimum edge required for BUY NO (default: 12%)
- **Min Probability for YES**: Minimum predicted probability for BUY YES (default: 60%)
- **Max Probability for NO**: Maximum predicted probability for BUY NO (default: 40%)
- **Dataset Selection**: Choose between different prediction runs

### In Predictions Tab
- **Meeting Date Filter**: View predictions for specific FOMC meetings
- **Recommendation Filter**: Show only BUY YES, BUY NO, HOLD, or All
- **Min Absolute Edge**: Filter out predictions below a certain edge threshold

## Data Flow

```
Kalshi API → predict-upcoming CLI → predictions.csv → Database → Streamlit Dashboard
     ↓                                                                    ↓
Contract Data                                                    User-Friendly
Market Prices                                                  Trade Recommendations
```

## Refreshing Data

The dashboard includes a **🔄 Refresh Predictions** button that:
1. Fetches latest contract data from Kalshi
2. Runs the prediction model on updated data
3. Loads new predictions into the database
4. Automatically refreshes the dashboard display

This ensures you're always working with the most current market data and predictions.

## Color Coding

- 🟢 **Green rows**: BUY YES recommendations (strong positive edge)
- 🔴 **Red rows**: BUY NO recommendations (strong negative edge)
- ⚪ **White rows**: HOLD (insufficient edge or uncertain probabilities)

## Understanding the Metrics

- **Predicted Probability**: Model's estimate of mention likelihood (0-100%)
- **Market Price**: Current Kalshi market price (0-100%)
- **Edge**: Difference between prediction and market price (positive = underpriced, negative = overpriced)
- **Confidence Interval**: Range of uncertainty (lower - upper bounds)
- **Days Until Meeting**: Time remaining before FOMC press conference

## Tips for Best Results

1. **Review Top Opportunities**: Start with the expandable "Top Opportunities" section
2. **Check Confidence Intervals**: Wider intervals = more uncertainty
3. **Adjust Thresholds**: Increase edge thresholds for more conservative trades
4. **Filter by Time**: Focus on upcoming meetings with sufficient time to place trades
5. **Refresh Regularly**: Use the refresh button before each FOMC meeting to get latest data

## Troubleshooting

**No predictions showing?**
- Ensure you've loaded prediction data: `python scripts/load_upcoming_predictions.py`
- Check that `data/kalshi_analysis/contract_words.json` exists
- Run the refresh button to generate new predictions

**Refresh button not working?**
- Ensure Kalshi API credentials are configured in `.env`
- Check that all dependencies are installed
- Verify `data/kalshi_analysis/contract_words.json` is present

**Database errors?**
- Run database migrations: `alembic upgrade head`
- Check `DATABASE_URL` environment variable is set correctly

## Architecture

The dashboard uses:
- **Streamlit**: Frontend framework for the web interface
- **SQLAlchemy**: Database ORM for data persistence
- **Pandas**: Data manipulation and analysis
- **subprocess**: Calling CLI commands for data refresh

The prediction model is a **Bayesian Beta-Binomial** model that:
- Trains on historical FOMC transcript mentions
- Incorporates uncertainty quantification
- Uses exponential decay weighting for recency bias
- Provides credible intervals for each prediction
