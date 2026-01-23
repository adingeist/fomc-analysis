# Streamlit Dashboard UX Improvement Plan

## Executive Summary

Based on user feedback emphasizing the need to verify trading opportunities against historical meeting data (especially the last 6 meetings), this document outlines specific UX enhancements to make trade verification faster and more intuitive.

## Current Strengths (Keep These!)

1. ✅ **Color-coded trade recommendations** - Green (BUY YES) / Red (BUY NO) system is intuitive
2. ✅ **Clear opportunity cards** - Expandable cards with key metrics are well-designed
3. ✅ **Word Mention Trends section** - Historical chart is valuable for verification

## Priority 1: Historical Context Integration

### 1.1 Add "Historical Context" to Each Opportunity Card

**Problem**: Users need to scroll down to the Word Mention Trends section to verify each trade, breaking their workflow.

**Solution**: Embed historical context directly within each opportunity card.

**Implementation**:

```python
# Within each opportunity expander, add a new section:

with st.expander(f"{icon} **{row['recommendation']}** - {row['contract']}", expanded=False):
    # ... existing metrics ...

    st.markdown("---")
    st.markdown("**📊 Historical Verification (Last 6 Meetings)**")

    # Get historical data for this specific contract
    historical_mentions = get_last_n_meetings_for_contract(
        contract=row['contract'],
        n_meetings=6,
        mentions_df=mentions_df
    )

    if not historical_mentions.empty:
        # Create visual timeline
        hist_cols = st.columns(6)
        for idx, (meeting_date, mention_count) in enumerate(historical_mentions.items()):
            with hist_cols[idx]:
                # Visual indicator
                color = "🟢" if mention_count > 0 else "⚪"
                st.markdown(f"{color} **{meeting_date.strftime('%m/%Y')}**")
                st.caption(f"{int(mention_count)} mentions")

        # Summary statistics
        st.markdown(f"""
        - **Mention Frequency**: {(historical_mentions > 0).sum()}/6 meetings
        - **Avg Mentions**: {historical_mentions.mean():.1f} per meeting
        - **Trend**: {calculate_trend(historical_mentions)}
        """)

        # Quick link to detailed view
        if st.button(f"📖 View Full History", key=f"history_{row['contract']}"):
            st.session_state['jump_to_contract'] = row['contract']
            # This would scroll to Word Mention Trends section
    else:
        st.info("No historical data available for this contract")
```

### 1.2 Add Trend Indicators

**Visual indicators for each opportunity**:
- 📈 **Increasing**: Mentions have been trending up
- 📉 **Decreasing**: Mentions have been trending down
- ➡️ **Stable**: Consistent mention pattern
- ⚠️ **Volatile**: Highly variable mention pattern
- 🆕 **New**: No significant historical data

**Display in opportunity card header**:
```
📈 BUY YES - AI / Artificial Intelligence ⬆️ Increasing
```

## Priority 2: Quick Verification Modal

### 2.1 "Verify This Trade" Button

Add a prominent verification workflow for each opportunity:

```python
# Inside each opportunity card
verify_col1, verify_col2 = st.columns([1, 3])

with verify_col1:
    if st.button("🔍 Verify Trade", key=f"verify_{row['contract']}", type="primary"):
        st.session_state['verify_modal_open'] = row['contract']

# Display verification modal if triggered
if st.session_state.get('verify_modal_open') == row['contract']:
    with st.expander("✅ Trade Verification Panel", expanded=True):
        st.markdown(f"### Verifying: {row['contract']}")

        # Tabs for different verification aspects
        verify_tabs = st.tabs([
            "📊 Historical Mentions",
            "📈 Trend Analysis",
            "💡 Model Confidence",
            "📰 Meeting Context"
        ])

        with verify_tabs[0]:
            # Detailed historical chart for this contract
            # Show last 12 meetings with highlight on last 6
            display_historical_mentions_chart(row['contract'])

        with verify_tabs[1]:
            # Trend analysis
            display_trend_analysis(row['contract'])

        with verify_tabs[2]:
            # Model performance on this word historically
            display_model_accuracy_for_word(row['contract'])

        with verify_tabs[3]:
            # Meeting context (if available from transcripts)
            display_meeting_context(row['contract'])

        # Close button
        if st.button("Close Verification Panel"):
            st.session_state['verify_modal_open'] = None
            st.rerun()
```

## Priority 3: Enhanced Summary Metrics

### 3.1 Add Historical Performance Metrics to Summary

Update the summary section to include:

```python
with metric_cols[5]:  # New 6th column
    # Calculate average historical mention rate
    avg_mention_rate = calculate_avg_mention_rate(filtered_df, mentions_df)
    st.metric(
        "📖 Avg Historical Rate",
        f"{avg_mention_rate:.1f}%",
        help="Average mention rate in last 6 meetings for action items"
    )
```

### 3.2 Confidence Score Badge

Add a composite confidence score to each opportunity:

```python
def calculate_confidence_score(row, historical_data):
    """
    Calculate 0-100 confidence score based on:
    - Edge magnitude (40%)
    - Historical consistency (30%)
    - Model confidence interval width (20%)
    - Days until meeting (10%)
    """
    edge_score = min(abs(row['edge']) / 0.3, 1.0) * 40

    # Historical consistency: how often was word mentioned?
    hist_consistency = (historical_data > 0).sum() / len(historical_data) * 30

    # Confidence interval: narrower = better
    ci_width = row['confidence_upper'] - row['confidence_lower']
    ci_score = max(0, (1 - ci_width / 0.4)) * 20

    # Urgency: closer = higher confidence
    if row['days_until_meeting'] <= 3:
        urgency_score = 10
    elif row['days_until_meeting'] <= 7:
        urgency_score = 7
    else:
        urgency_score = 5

    return edge_score + hist_consistency + ci_score + urgency_score

# Display in opportunity card
confidence_score = calculate_confidence_score(row, historical_mentions)
confidence_color = "🟢" if confidence_score >= 75 else "🟡" if confidence_score >= 50 else "🔴"

st.markdown(f"""
### {confidence_color} Confidence Score: {confidence_score:.0f}/100
""")
```

## Priority 4: Improved Word Mention Trends Section

### 4.1 Smart Default Selection

When viewing the Word Mention Trends section from an opportunity:

```python
# If user clicked from an opportunity card, auto-select that contract
if 'jump_to_contract' in st.session_state:
    default_selection = [st.session_state['jump_to_contract']]
    st.info(f"📍 Showing history for: **{st.session_state['jump_to_contract']}**")
    # Clear the session state after use
    del st.session_state['jump_to_contract']
```

### 4.2 Highlight Recent Meetings

Add visual emphasis to the most recent 6 meetings in the chart:

```python
# Add shaded region for last 6 meetings
fig = go.Figure()

# Plot historical data
fig.add_trace(go.Scatter(
    x=chart_df.index,
    y=chart_df[word],
    name=word,
    line=dict(width=2)
))

# Add shaded region for last 6 meetings
last_6_meetings = chart_df.index[-6:]
fig.add_vrect(
    x0=last_6_meetings[0],
    x1=last_6_meetings[-1],
    fillcolor="rgba(255, 200, 0, 0.1)",
    layer="below",
    line_width=0,
    annotation_text="Last 6 Meetings",
    annotation_position="top left"
)

st.plotly_chart(fig)
```

## Priority 5: Quick Filters Enhancement

### 5.1 Add Smart Filter Presets

```python
st.markdown("### 🎯 Quick Filters")

# Add preset buttons
preset_cols = st.columns(4)

with preset_cols[0]:
    if st.button("⭐ High Confidence", help="Edge >15% & frequent historical mentions"):
        apply_high_confidence_filter()

with preset_cols[1]:
    if st.button("📈 Trending Up", help="Increasing mention frequency"):
        apply_trending_filter("up")

with preset_cols[2]:
    if st.button("📉 Contrarian", help="Historically rare, now predicted"):
        apply_contrarian_filter()

with preset_cols[3]:
    if st.button("🔥 Urgent & Verified", help="<3 days & strong history"):
        apply_urgent_verified_filter()
```

### 5.2 Historical Frequency Filter

```python
with filter_cols[4]:  # New filter
    min_historical_frequency = st.slider(
        "📖 Min Historical Frequency",
        0, 6, 0,
        help="Minimum times mentioned in last 6 meetings"
    )

    # Apply filter
    if min_historical_frequency > 0:
        filtered_df = filter_by_historical_frequency(
            filtered_df,
            mentions_df,
            min_frequency=min_historical_frequency
        )
```

## Priority 6: Visual Enhancements

### 6.1 Meeting Timeline Visualization

Add a visual timeline at the top of each opportunity card:

```
Last 6 FOMC Meetings:
2023-12  2024-01  2024-03  2024-05  2024-06  2024-07
   🟢      🟢       ⚪       🟢       🟢       🟢

Pattern: 5/6 meetings (83% mention rate) ⬆️ Increasing
```

### 6.2 Sparkline Charts

Add mini sparkline charts in the opportunities table:

```python
# In the opportunities table, add a "Trend" column with sparklines
def generate_sparkline_svg(values):
    """Generate a simple SVG sparkline"""
    # Simple line chart showing trend
    ...

display_df['Trend'] = display_df['contract'].apply(
    lambda x: generate_sparkline_for_contract(x, mentions_df)
)
```

## Priority 7: Enhanced Table View

### 7.1 Add Historical Context Columns

Update the main predictions table:

```python
display_cols = [
    "recommendation",
    "contract",
    "predicted_probability",
    "market_price",
    "edge",
    "historical_frequency",  # NEW: e.g., "5/6"
    "trend",                  # NEW: "↑", "↓", "→"
    "confidence_score",       # NEW: 0-100 score
    "days_until_meeting",
    "meeting_date",
]
```

### 7.2 Sortable by Confidence

Allow sorting by the new confidence score:

```python
sort_option = st.selectbox(
    "Sort by",
    ["Edge (Absolute)", "Confidence Score", "Historical Frequency", "Days Until"],
    key="opportunities_sort"
)
```

## Implementation Priority

### Phase 1 (Immediate - Highest Impact):
1. ✅ Add historical context panel to opportunity cards (section 1.1)
2. ✅ Add trend indicators to opportunity headers (section 1.2)
3. ✅ Add confidence score badge (section 3.2)
4. ✅ Add historical frequency filter (section 5.2)

### Phase 2 (Short-term):
5. ✅ Implement "Verify This Trade" modal (section 2.1)
6. ✅ Add smart filter presets (section 5.1)
7. ✅ Enhance Word Mention Trends with highlights (section 4.2)

### Phase 3 (Medium-term):
8. ✅ Add meeting timeline visualization (section 6.1)
9. ✅ Enhanced table view with historical columns (section 7.1)
10. ✅ Sparkline charts (section 6.2)

## Technical Requirements

### New Helper Functions Needed:

```python
def get_last_n_meetings_for_contract(contract: str, n_meetings: int, mentions_df: pd.DataFrame) -> pd.Series:
    """Get last N meetings' mention counts for a specific contract"""
    if contract not in mentions_df.columns:
        return pd.Series()

    return mentions_df[contract].tail(n_meetings)

def calculate_trend(historical_mentions: pd.Series) -> str:
    """Calculate trend from historical mention data"""
    if len(historical_mentions) < 3:
        return "Insufficient data"

    # Simple linear regression slope
    from scipy.stats import linregress
    x = range(len(historical_mentions))
    slope, _, _, _, _ = linregress(x, historical_mentions)

    if slope > 0.5:
        return "📈 Increasing"
    elif slope < -0.5:
        return "📉 Decreasing"
    else:
        return "➡️ Stable"

def get_historical_frequency(contract: str, mentions_df: pd.DataFrame, n_meetings: int = 6) -> str:
    """Get frequency string like '5/6'"""
    mentions = get_last_n_meetings_for_contract(contract, n_meetings, mentions_df)
    if mentions.empty:
        return "N/A"

    mentioned_count = (mentions > 0).sum()
    return f"{mentioned_count}/{n_meetings}"

def filter_by_historical_frequency(df: pd.DataFrame, mentions_df: pd.DataFrame, min_frequency: int) -> pd.DataFrame:
    """Filter predictions by minimum historical mention frequency"""
    def meets_threshold(contract):
        mentions = get_last_n_meetings_for_contract(contract, 6, mentions_df)
        return (mentions > 0).sum() >= min_frequency

    return df[df['contract'].apply(meets_threshold)]
```

### Data Requirements:

1. **Word Frequency Timeseries** (✅ Already available)
   - Path: `results/backtest_v3/word_frequency_timeseries.csv`
   - Contains: meeting dates and mention counts per word

2. **Meeting Metadata** (⚠️ May need to add)
   - Meeting dates
   - Associated news headlines
   - Economic context
   - Key themes discussed

3. **Historical Model Performance** (⚠️ May need to add)
   - Track model accuracy per word over time
   - Store in database as new table

## Success Metrics

After implementation, measure:

1. **Time to Verify Trade**: Should decrease from ~2-3 minutes to <30 seconds
2. **User Confidence**: Survey users on confidence in trade decisions
3. **Trade Quality**: Track if trades with high confidence scores perform better
4. **Engagement**: Measure how often users click "Verify Trade" button
5. **Error Rate**: Reduce incorrect trades due to insufficient verification

## Future Enhancements (Beyond Initial Scope)

1. **Transcript Snippets**: Show actual quotes from transcripts where word was mentioned
2. **Correlation Analysis**: Show which other words tend to appear together
3. **Economic Indicator Integration**: Overlay economic data on mention trends
4. **Saved Watchlists**: Allow users to save and monitor specific contracts
5. **Alert System**: Notify users when high-confidence opportunities emerge
6. **Mobile Optimization**: Responsive design for mobile verification workflows

---

## Quick Reference: Key User Flows

### Flow 1: Reviewing Top Opportunities (Current)
1. View Top Opportunities section
2. Click expander to see details
3. **❌ Need to scroll down to Word Mention Trends**
4. **❌ Need to manually select contract**
5. **❌ Need to visually count last 6 meetings**
6. **❌ Scroll back up to opportunity**
7. Make decision

**Time: ~2-3 minutes per opportunity**

### Flow 2: Reviewing Top Opportunities (Proposed)
1. View Top Opportunities section
2. Click expander to see details
3. **✅ See "Last 6 Meetings" timeline directly in card**
4. **✅ See trend indicator and frequency**
5. **✅ View confidence score**
6. *(Optional)* Click "Verify Trade" for deep dive
7. Make decision

**Time: ~20-30 seconds per opportunity**

### Flow 3: Deep Verification (New)
1. Click "🔍 Verify Trade" button
2. View tabbed verification panel:
   - Historical chart (12 meetings, last 6 highlighted)
   - Trend analysis with statistics
   - Model confidence and past accuracy
   - Meeting context/notes if available
3. Make informed decision
4. Close panel

**Time: ~1-2 minutes for thorough verification**

---

## Mockup: Enhanced Opportunity Card

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ 📈 BUY YES - AI / Artificial Intelligence   📈 Increasing   🟢 85/100  ┃
┣━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┫
┃                                                                        ┃
┃  ⚡ URGENT: Less than 3 days                                          ┃
┃                                                                        ┃
┃  Predicted: 78.5%    Market: 64.0%    Edge: +14.5%    Days: 2        ┃
┃                                                                        ┃
┃  ─────────────────────────────────────────────────────────────────    ┃
┃                                                                        ┃
┃  📊 Historical Verification (Last 6 Meetings)                         ┃
┃                                                                        ┃
┃   2023-12   2024-01   2024-03   2024-05   2024-06   2024-07          ┃
┃     🟢        🟢         ⚪         🟢        🟢         🟢             ┃
┃   15 times  12 times  0 times  8 times  18 times  22 times           ┃
┃                                                                        ┃
┃   • Mention Frequency: 5/6 meetings (83%)                             ┃
┃   • Avg Mentions: 12.5 per meeting                                    ┃
┃   • Trend: 📈 Increasing (mentions growing)                           ┃
┃                                                                        ┃
┃   [🔍 Verify Trade]  [📖 View Full History]                           ┃
┃                                                                        ┃
┗━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┛
```

## Questions for User

Before implementing, please confirm:

1. **Historical Window**: Is 6 meetings the right default, or would you prefer 8-10?
2. **Trend Calculation**: Should we weight recent meetings more heavily?
3. **Confidence Score**: Are the weightings appropriate (Edge 40%, History 30%, CI 20%, Urgency 10%)?
4. **Transcript Context**: Would you like us to integrate actual transcript snippets if available?
5. **Alert System**: Would push notifications or email alerts for high-confidence trades be useful?

## Conclusion

These improvements will:
- ✅ Keep what works (color-coded system, clear metrics)
- ✅ Add historical context directly to opportunity cards
- ✅ Reduce verification time from minutes to seconds
- ✅ Increase user confidence in trading decisions
- ✅ Maintain clean, intuitive UI

**Estimated Development Time**: 2-3 weeks for Phase 1, 1-2 weeks each for Phases 2-3
