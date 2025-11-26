import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import io
import numpy as np
import os
import glob
from scipy import stats

# Set page configuration
st.set_page_config(
    page_title="Search Query Performance Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_and_combine_uploaded_csvs(uploaded_files):
    """
    Load and combine all uploaded CSV files for Tab 1
    """
    if not uploaded_files:
        return None
    
    files = []
    for uploaded_file in uploaded_files:
        try:
            # Read the uploaded file
            df_temp = pd.read_csv(uploaded_file, header=1)  # Skip first row, use second as header
            files.append(df_temp)
            st.success(f"✅ Loaded: {uploaded_file.name}")
        except Exception as e:
            st.warning(f"⚠️ Could not load {uploaded_file.name}: {str(e)}")
            continue
    
    if not files:
        st.error("No CSV files could be loaded successfully")
        return None
    
    combined_df = pd.concat(files, ignore_index=True)
    st.success(f"✅ Successfully combined {len(files)} CSV files with {len(combined_df)} total records")
    return combined_df

@st.cache_data
def auto_detect_uploaded_files(uploaded_files):
    """
    Auto-detect and categorize uploaded files based on naming patterns for Tab 2
    """
    if not uploaded_files:
        return None, None, None, None, None, None
    
    # Initialize variables for detected files
    excel_file = None
    csv_product_file = None
    csv_brand_file = None
    csv_top_search_term_file = None
    csv_targeting_report_file = None
    csv_business_report_file = None
    
    detected_files = []
    
    for uploaded_file in uploaded_files:
        filename = uploaded_file.name.lower()
        
        # SP_Ad_product + Excel format
        if 'sp_ad_product' in filename and (filename.endswith('.xlsx') or filename.endswith('.xls')):
            excel_file = uploaded_file
            detected_files.append(f"📊 Ad Product Report: {uploaded_file.name}")
        
        # SP_ST_imp + CSV format
        elif 'sp_st_imp' in filename and filename.endswith('.csv'):
            csv_product_file = uploaded_file
            detected_files.append(f"📈 Product Search Terms: {uploaded_file.name}")
        
        # SB_ST_imp + CSV format
        elif 'sb_st_imp' in filename and filename.endswith('.csv'):
            csv_brand_file = uploaded_file
            detected_files.append(f"🏷️ Brand Search Terms: {uploaded_file.name}")
        
        # Top_search_terms + CSV format
        elif 'top_search_terms' in filename and filename.endswith('.csv'):
            csv_top_search_term_file = uploaded_file
            detected_files.append(f"🔍 Top Search Terms: {uploaded_file.name}")
        
        # SP_Targeting + Excel format
        elif 'sp_targeting' in filename and (filename.endswith('.xlsx') or filename.endswith('.xls')):
            csv_targeting_report_file = uploaded_file
            detected_files.append(f"🎯 Targeting Report: {uploaded_file.name}")
        
        # BusinessReport + CSV format
        elif 'businessreport' in filename and filename.endswith('.csv'):
            csv_business_report_file = uploaded_file
            detected_files.append(f"📋 Business Report: {uploaded_file.name}")
    
    # Show detected files
    if detected_files:
        st.success(f"📁 Auto-detected {len(detected_files)} files:")
        for file_info in detected_files:
            st.info(file_info)
    else:
        st.warning("⚠️ No files matching the expected patterns were found")
        st.caption("Expected patterns: SP_Ad_product.xlsx, SP_ST_imp.csv, SB_ST_imp.csv, Top_search_terms.csv, SP_Targeting.xlsx, BusinessReport.csv")
    
    return excel_file, csv_product_file, csv_brand_file, csv_top_search_term_file, csv_targeting_report_file, csv_business_report_file

def fill_missing_dates(data: pd.DataFrame, date_column: str, freq: str) -> pd.DataFrame:
    """
    Fills missing dates in a time series DataFrame for each unique product,
    with a frequency of 'ME' (monthly) or 'W' (weekly).
    
    For weekly data: Normalizes dates to week periods regardless of which day of the week
    the data falls on. All dates in the same week are grouped together and represented
    by the week start date (Monday).
    
    For monthly data: Normalizes to month-end dates.
    """
    # Validate the frequency input
    if freq not in ['ME', 'W']:
        raise ValueError("Invalid frequency. Please use 'ME' for monthly or 'W' for weekly.")

    data = data.rename(columns={'Reporting Date': 'Date'})
    # Ensure the 'Date' column is in the correct format
    data['Date'] = pd.to_datetime(data['Date'])

    if freq == 'W':
        # For weekly data: Normalize all dates to the start of their week (Monday)
        # This handles cases where different products have data on different days of the week
        data['Week_Period'] = data['Date'].dt.to_period('W-SUN')  # Week ending on Sunday
        data['Date'] = data['Week_Period'].dt.start_time  # Convert to Monday (week start)
        data = data.drop(columns=['Week_Period'])
        
        # Group by Date and Search Query, aggregating numeric columns
        # This combines data that falls in the same week for the same search query
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = ['Date', 'Search Query']
        
        if len(numeric_cols) > 0:
            # Aggregate by taking the sum for numeric columns (since we're grouping by week)
            agg_dict = {col: 'sum' for col in numeric_cols}
            data = data.groupby(['Date', 'Search Query'], as_index=False).agg(agg_dict)
    
    elif freq == 'ME':
        # For monthly data: Normalize to month-end
        data['Month_Period'] = data['Date'].dt.to_period('M')
        data['Date'] = data['Month_Period'].dt.end_time  # Convert to month-end
        data = data.drop(columns=['Month_Period'])
        
        # Group by Date and Search Query, aggregating numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) > 0:
            agg_dict = {col: 'sum' for col in numeric_cols}
            data = data.groupby(['Date', 'Search Query'], as_index=False).agg(agg_dict)

    # Get all unique dates that exist after normalization
    unique_dates = sorted(data['Date'].unique())
    
    # Get a list of all unique products
    products = data['Search Query'].unique()

    # Create an empty list to store the processed data for each product
    filled_data_list = []

    # Iterate over each product
    for product in products:
        # Filter the original data for the current product
        product_data = data[data['Search Query'] == product].copy()
        
        # Create a DataFrame with all unique dates from the dataset
        full_product_df = pd.DataFrame({
            'Date': unique_dates,
            'Search Query': product
        })

        # Merge with the product's actual data
        merged_df = pd.merge(full_product_df, product_data, on=['Date', 'Search Query'], how='left')

        # Append the merged DataFrame to our list
        filled_data_list.append(merged_df)

    # Concatenate all the individual product DataFrames into one
    filled_df = pd.concat(filled_data_list, ignore_index=True)

    # Sort the final DataFrame by Product and then by Date for clean display
    filled_df.sort_values(by=['Search Query', 'Date'], inplace=True)

    return filled_df

def analyze_search_term_trends(filtered_df):
    """
    Analyze trends for a search term over time with correlations and insights
    Returns a dictionary with trend analysis
    """
    # Ensure Date is datetime and sort
    df = filtered_df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Remove rows where volume is 0 (gaps in data)
    df_valid = df[df['Search Query Volume'] > 0].copy()
    
    if len(df_valid) < 2:
        return None
    
    # Create numeric index for regression
    df_valid['time_index'] = range(len(df_valid))
    
    def calculate_trend(series, time_index):
        """Calculate trend direction and slope"""
        if len(series) < 2 or series.isna().all():
            return {'direction': 'insufficient_data', 'slope': 0, 'change_pct': 0, 'r_squared': 0}
        
        # Remove NaN values
        valid_mask = ~(series.isna() | time_index.isna())
        clean_series = series[valid_mask]
        clean_time = time_index[valid_mask]
        
        if len(clean_series) < 2:
            return {'direction': 'insufficient_data', 'slope': 0, 'change_pct': 0, 'r_squared': 0}
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(clean_time, clean_series)
        
        # Calculate percentage change
        first_val = clean_series.iloc[0]
        last_val = clean_series.iloc[-1]
        change_pct = ((last_val - first_val) / first_val * 100) if first_val != 0 else 0
        
        # Determine direction
        if abs(slope) < 0.01 and abs(change_pct) < 5:
            direction = 'flat'
        elif slope > 0:
            direction = 'up'
        elif slope < 0:
            direction = 'down'
        else:
            direction = 'flat'
        
        # Check for fluctuation (high variance relative to mean)
        cv = clean_series.std() / clean_series.mean() if clean_series.mean() != 0 else 0
        if cv > 0.3:  # Coefficient of variation > 30%
            direction = 'fluctuating'
        
        return {
            'direction': direction,
            'slope': slope,
            'change_pct': change_pct,
            'r_squared': r_value**2,
            'p_value': p_value
        }
    
    # Analyze trends for each metric
    analysis = {
        'search_volume': calculate_trend(df_valid['Search Query Volume'], df_valid['time_index']),
        'impression_share': calculate_trend(df_valid['Brand Impressions Share'], df_valid['time_index']),
        'click_share': calculate_trend(df_valid['Brand Click Share'], df_valid['time_index']),
        'purchase_share': calculate_trend(df_valid['Brand Purchase Share'], df_valid['time_index']),
        'total_purchases': calculate_trend(df_valid['Total Purchase Count'], df_valid['time_index'])
    }
    
    # Calculate correlations
    correlations = {}
    try:
        # Volume vs Impression Share
        if len(df_valid) >= 3:
            correlations['volume_vs_impression'] = df_valid[['Search Query Volume', 'Brand Impressions Share']].corr().iloc[0, 1]
            correlations['impression_vs_click'] = df_valid[['Brand Impressions Share', 'Brand Click Share']].corr().iloc[0, 1]
            correlations['click_vs_purchase'] = df_valid[['Brand Click Share', 'Brand Purchase Share']].corr().iloc[0, 1]
            correlations['volume_vs_purchases'] = df_valid[['Search Query Volume', 'Total Purchase Count']].corr().iloc[0, 1]
    except:
        pass
    
    analysis['correlations'] = correlations
    
    # Month-over-month absolute purchases
    mom_purchases = []
    if len(df_valid) >= 2:
        for i in range(1, len(df_valid)):
            prev_purchases = df_valid.iloc[i-1]['Total Purchase Count']
            curr_purchases = df_valid.iloc[i]['Total Purchase Count']
            change = curr_purchases - prev_purchases
            change_pct = (change / prev_purchases * 100) if prev_purchases != 0 else 0
            mom_purchases.append({
                'period': f"{df_valid.iloc[i-1]['Date'].strftime('%Y-%m-%d')} → {df_valid.iloc[i]['Date'].strftime('%Y-%m-%d')}",
                'prev': prev_purchases,
                'curr': curr_purchases,
                'change': change,
                'change_pct': change_pct
            })
    
    analysis['mom_purchases'] = mom_purchases
    
    # Generate insights
    insights = []
    
    # Search Volume insights
    vol_trend = analysis['search_volume']
    if vol_trend['direction'] == 'up':
        insights.append(f"📈 **Search Volume Trending Up**: {vol_trend['change_pct']:.1f}% increase over period")
    elif vol_trend['direction'] == 'down':
        insights.append(f"📉 **Search Volume Declining**: {vol_trend['change_pct']:.1f}% decrease over period")
    elif vol_trend['direction'] == 'fluctuating':
        insights.append(f"📊 **Search Volume Fluctuating**: High variability in search demand")
    else:
        insights.append(f"➡️ **Search Volume Stable**: Consistent search demand")
    
    # Impression Share insights
    imp_trend = analysis['impression_share']
    if imp_trend['direction'] == 'up':
        insights.append(f"✅ **Impression Share Growing**: {imp_trend['change_pct']:.1f}% increase - good visibility")
    elif imp_trend['direction'] == 'down':
        insights.append(f"⚠️ **Impression Share Declining**: {imp_trend['change_pct']:.1f}% decrease - losing visibility")
    
    # Click Share insights
    click_trend = analysis['click_share']
    if click_trend['direction'] == 'up':
        insights.append(f"👆 **Click Share Improving**: {click_trend['change_pct']:.1f}% increase")
    elif click_trend['direction'] == 'down':
        insights.append(f"👇 **Click Share Declining**: {click_trend['change_pct']:.1f}% decrease")
    
    # Purchase Share insights
    purch_trend = analysis['purchase_share']
    if purch_trend['direction'] == 'up':
        insights.append(f"💰 **Purchase Share Growing**: {purch_trend['change_pct']:.1f}% increase - winning conversions")
    elif purch_trend['direction'] == 'down':
        insights.append(f"⚠️ **Purchase Share Declining**: {purch_trend['change_pct']:.1f}% decrease")
    
    # Correlation insights
    if correlations:
        # Comprehensive combination analysis of Impression, Click, and Purchase trends
        imp_dir = imp_trend['direction']
        click_dir = click_trend['direction']
        purch_dir = purch_trend['direction']
        
        # === IDEAL SCENARIOS ===
        
        # All three growing - Perfect execution
        if imp_dir == 'up' and click_dir == 'up' and purch_dir == 'up':
            insights.append(f"🎯 **Excellent Performance**: All metrics trending up - impression, click & purchase share growing. Continue current strategy!")
        
        # Efficiency gains - Doing more with same/less impressions
        elif imp_dir in ['flat', 'down'] and click_dir == 'up' and purch_dir == 'up':
            insights.append(f"✨ **High Efficiency Gains**: Click & purchase share growing despite {'stable' if imp_dir == 'flat' else 'declining'} impressions - excellent ad relevance and conversion optimization!")
        
        # Improving conversion funnel
        elif imp_dir == 'flat' and click_dir == 'flat' and purch_dir == 'up':
            insights.append(f"💎 **Conversion Rate Improvement**: Purchase share growing with stable impressions & clicks - better at closing sales!")
        
        # === WARNING SCENARIOS ===
        
        # Impression up but not converting to clicks/purchases
        elif imp_dir == 'up' and click_dir in ['flat', 'down'] and purch_dir in ['flat', 'down']:
            insights.append(f"⚠️ **Low CTR Alert**: Impression share up but clicks {'flat' if click_dir == 'flat' else 'declining'} - check ad copy, images, pricing, and relevance. May be showing for less relevant searches.")
        
        # Clicks up but not converting
        elif imp_dir == 'up' and click_dir == 'up' and purch_dir in ['flat', 'down']:
            insights.append(f"⚠️ **Conversion Problem**: Getting impressions & clicks but purchase share {'not growing' if purch_dir == 'flat' else 'declining'} - check product page, pricing, reviews, A+ content, and fulfillment options.")
        
        # Stable impressions but losing clicks and purchases
        elif imp_dir == 'flat' and click_dir == 'down' and purch_dir == 'down':
            insights.append(f"⚠️ **Declining Engagement**: Visibility stable but click & purchase share declining - competitors may have better offers, or product/price competitiveness declining.")
        
        # Only impressions growing
        elif imp_dir == 'up' and click_dir == 'flat' and purch_dir == 'flat':
            insights.append(f"📊 **Mixed Visibility**: Impression share growing but click & purchase share flat - gaining visibility but not engagement. Review ad quality and relevance.")
        
        # Losing impressions while clicks/purchases stable
        elif imp_dir == 'down' and click_dir in ['flat'] and purch_dir in ['flat']:
            insights.append(f"🎯 **Efficiency Maintained**: Impression share declining but maintaining click & purchase rates - may need to increase bids to regain visibility, but conversion quality is good.")
        
        # === DECLINING SCENARIOS ===
        
        # Everything declining
        elif imp_dir == 'down' and click_dir == 'down' and purch_dir == 'down':
            insights.append(f"🚨 **Full Decline Alert**: All metrics declining - immediate action needed. Check: 1) Bid competitiveness, 2) Budget constraints, 3) Competitor activity, 4) Product reviews/ratings, 5) Seasonality.")
        
        # Impressions down, but conversion holding
        elif imp_dir == 'down' and click_dir == 'down' and purch_dir in ['flat', 'up']:
            insights.append(f"💡 **Quality over Quantity**: Impression & click share down but purchase share {'stable' if purch_dir == 'flat' else 'growing'} - higher conversion rate but lower volume. Consider increasing bids to scale winning traffic.")
        
        # Clicks declining faster than impressions
        elif imp_dir in ['flat', 'up'] and click_dir == 'down' and purch_dir == 'down':
            insights.append(f"⚠️ **CTR Deterioration**: Click & purchase share declining despite {'stable' if imp_dir == 'flat' else 'growing'} impressions - ad fatigue or competitive pressure. Refresh creative and check pricing.")
        
        # === MIXED/COMPLEX SCENARIOS ===
        
        # Impressions up, clicks down, purchases up (unusual but possible)
        elif imp_dir == 'up' and click_dir == 'down' and purch_dir == 'up':
            insights.append(f"🔍 **Interesting Pattern**: Impression up, clicks down, but purchases up - getting higher quality/converting clicks despite lower CTR. May indicate better search term targeting or improved product appeal.")
        
        # Stable impressions, clicks up, purchases down
        elif imp_dir == 'flat' and click_dir == 'up' and purch_dir == 'down':
            insights.append(f"⚠️ **Conversion Drop**: Click share growing but purchase share declining - traffic quality may be declining or product page has issues. Review: product content, pricing competitiveness, inventory status.")
        
        # Clicks stable but diverging impression and purchase
        elif imp_dir == 'up' and click_dir == 'flat' and purch_dir == 'down':
            insights.append(f"🔍 **Funnel Breakdown**: Impressions up, clicks stable, purchases down - either CTR is declining (getting irrelevant impressions) or conversion rate dropping. Audit search term quality.")
        
        elif imp_dir == 'down' and click_dir == 'flat' and purch_dir == 'up':
            insights.append(f"✅ **Optimization Success**: Fewer impressions but better click quality leading to higher purchase share - efficient targeting. Can scale with increased budget.")
        
        # Fluctuating scenarios
        if 'fluctuating' in [imp_dir, click_dir, purch_dir]:
            fluctuating_metrics = []
            if imp_dir == 'fluctuating':
                fluctuating_metrics.append('impression share')
            if click_dir == 'fluctuating':
                fluctuating_metrics.append('click share')
            if purch_dir == 'fluctuating':
                fluctuating_metrics.append('purchase share')
            
            insights.append(f"📊 **High Variability**: {', '.join(fluctuating_metrics)} showing high fluctuation - may indicate seasonality, promotional cycles, or inconsistent campaign management. Consider stabilizing bids and budgets.")
        
        # Volume up with stable share
        if vol_trend['direction'] == 'up' and imp_trend['direction'] in ['flat']:
            insights.append(f"📈 **Market Growth Opportunity**: Search volume increasing - absolute impressions growing even with stable share %. Consider increasing bids to capture more of growing market.")
        
        # Volume down scenarios
        elif vol_trend['direction'] == 'down':
            if imp_trend['direction'] == 'up':
                insights.append(f"🎯 **Market Share Gain**: Gaining impression share despite declining search volume - winning against competitors in shrinking market.")
            elif imp_trend['direction'] == 'down':
                insights.append(f"⚠️ **Market Decline**: Both search volume and impression share declining - market may be shrinking or seasonal. Evaluate long-term viability of this search term.")
        
        # Strong correlation between volume and purchases
        if correlations.get('volume_vs_purchases', 0) > 0.7:
            insights.append(f"✅ **Strong Volume-Purchase Correlation**: {correlations['volume_vs_purchases']:.2f} - effectively capturing search demand growth. Performance scales with market size.")
        elif 0.3 <= correlations.get('volume_vs_purchases', 0) <= 0.7:
            insights.append(f"📊 **Moderate Volume-Purchase Correlation**: {correlations['volume_vs_purchases']:.2f} - capturing some search demand but room for optimization.")
        elif correlations.get('volume_vs_purchases', 0) < 0.3:
            insights.append(f"⚠️ **Weak Volume-Purchase Correlation**: {correlations['volume_vs_purchases']:.2f} - not converting search demand efficiently. Major optimization opportunity.")
        
        # Impression-Click correlation
        if correlations.get('impression_vs_click', 0) > 0.7:
            insights.append(f"✅ **Strong Impression-Click Correlation**: {correlations['impression_vs_click']:.2f} - visibility translating well to engagement.")
        elif correlations.get('impression_vs_click', 0) < 0.3:
            insights.append(f"⚠️ **Weak Impression-Click Correlation**: {correlations['impression_vs_click']:.2f} - getting impressions but poor engagement. Review ad creative and positioning.")
        
        # Click-Purchase correlation
        if correlations.get('click_vs_purchase', 0) > 0.7:
            insights.append(f"✅ **Strong Click-Purchase Correlation**: {correlations['click_vs_purchase']:.2f} - clicks converting well to sales.")
        elif correlations.get('click_vs_purchase', 0) < 0.3:
            insights.append(f"⚠️ **Weak Click-Purchase Correlation**: {correlations['click_vs_purchase']:.2f} - getting clicks but poor conversion. Optimize product page, pricing, and fulfillment.")
    
    # Total purchases trend
    total_purch_trend = analysis['total_purchases']
    if total_purch_trend['direction'] == 'up':
        insights.append(f"💵 **Absolute Purchases Growing**: {total_purch_trend['change_pct']:.1f}% increase in actual sales")
    elif total_purch_trend['direction'] == 'down':
        insights.append(f"⚠️ **Absolute Purchases Declining**: {total_purch_trend['change_pct']:.1f}% decrease in actual sales")
    
    analysis['insights'] = insights
    analysis['data_points'] = len(df_valid)
    
    return analysis

def create_interactive_plots(filtered_df):
    """
    Creates four interactive plots from a pandas DataFrame
    """
    # Ensure the 'Date' column is in datetime format for proper plotting
    filtered_df['Date'] = pd.to_datetime(filtered_df['Date'])

    plots = []

    # --- Plot 1: Brand Purchase Share, Brand Click Share, and Search Query Volume ---
    fig1 = go.Figure()
    fig1.add_trace(
        go.Bar(
            x=filtered_df['Date'],
            y=filtered_df['Search Query Volume'],
            name='Search Query Volume',
            marker_color='turquoise'
        )
    )
    fig1.add_trace(
        go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df['Brand Purchase Share'],
            mode='lines+markers',
            name='Brand Purchase Share',
            line=dict(color='darkgreen'),
            yaxis='y2'
        )
    )
    fig1.add_trace(
        go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df['Brand Click Share'],
            mode='lines+markers',
            name='Brand Click Share',
            line=dict(color='saddlebrown'),
            yaxis='y2'
        )
    )
    fig1.update_layout(
        title_text='Brand Purchase Share, Brand Click Share, and Search Query Volume',
        title_x=0.5,
        xaxis_title='Date',
        yaxis_title='Search Query Volume',
        yaxis2=dict(
            title='Share %',
            overlaying='y',
            side='right'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500
    )
    plots.append(("Purchase & Click Share vs Volume", fig1))

    # --- Plot 2: Brand Cart Adds Share and Brand Click Share ---
    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df['Brand Cart Adds Share'],
            mode='lines+markers',
            name='Brand Cart Adds Share',
            line=dict(color='saddlebrown')
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df['Brand Click Share'],
            mode='lines+markers',
            name='Brand Click Share',
            line=dict(color='royalblue')
        )
    )
    fig2.update_layout(
        title_text='Brand Cart Adds Share and Brand Click Share',
        title_x=0.5,
        xaxis_title='Date',
        yaxis_title='Share %',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500
    )
    plots.append(("Cart Adds vs Click Share", fig2))

    # --- Plot 3: Brand Impressions Share and Search Query Volume ---
    fig3 = go.Figure()
    fig3.add_trace(
        go.Bar(
            x=filtered_df['Date'],
            y=filtered_df['Search Query Volume'],
            name='Search Query Volume',
            marker_color='turquoise'
        )
    )
    fig3.add_trace(
        go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df['Brand Impressions Share'],
            mode='lines+markers',
            name='Brand Impressions Share',
            line=dict(color='saddlebrown'),
            yaxis='y2'
        )
    )
    fig3.update_layout(
        title_text='Brand Impressions Share and Search Query Volume',
        title_x=0.5,
        xaxis_title='Date',
        yaxis_title='Search Query Volume',
        yaxis2=dict(
            title='Share %',
            overlaying='y',
            side='right'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500
    )
    plots.append(("Impressions Share vs Volume", fig3))

    # --- Plot 4: Brand Purchase Share and Brand Cart Adds Share ---
    fig4 = go.Figure()
    fig4.add_trace(
        go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df['Brand Purchase Share'],
            mode='lines+markers',
            name='Brand Purchase Share',
            line=dict(color='saddlebrown')
        )
    )
    fig4.add_trace(
        go.Scatter(
            x=filtered_df['Date'],
            y=filtered_df['Brand Cart Adds Share'],
            mode='lines+markers',
            name='Brand Cart Adds Share',
            line=dict(color='royalblue')
        )
    )
    fig4.update_layout(
        title_text='Brand Purchase Share and Brand Cart Adds Share',
        title_x=0.5,
        xaxis_title='Date',
        yaxis_title='Share %',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=500
    )
    plots.append(("Purchase vs Cart Adds Share", fig4))

    return plots

@st.cache_data
def load_sponsored_product_data_from_uploads(excel_file, csv_product_file, csv_brand_file, csv_top_search_term_file, csv_targeting_report_file, csv_business_report_file):
    """
    Load and process sponsored product data from uploaded files
    """
    if not excel_file:
        return None, None, None, None, None, None
    
    try:
        # Read Excel file - get available sheets first
        all_sheets = pd.read_excel(excel_file, sheet_name=None)
        # Find the sheet that contains the actual data (not 'Sheet1')
        data_sheet = None
        for sheet_name in all_sheets.keys():
            if sheet_name != 'Sheet1':
                data_sheet = sheet_name
                break
        
        if data_sheet:
            df_ad_product = pd.read_excel(excel_file, sheet_name=data_sheet)
        else:
            df_ad_product = pd.read_excel(excel_file)
            
        # Read CSV files
        st_imp_product_df = pd.read_csv(csv_product_file) if csv_product_file else None
        st_imp_brand_df = pd.read_csv(csv_brand_file) if csv_brand_file else None
        st_imp_top_search_term_df = pd.read_csv(csv_top_search_term_file, header=1) if csv_top_search_term_file else None
        df_targeting_report_final = pd.read_excel(csv_targeting_report_file) if csv_targeting_report_file else None
        df_business_report = pd.read_csv(csv_business_report_file) if csv_business_report_file else None
        
        return df_ad_product, st_imp_product_df, st_imp_brand_df, st_imp_top_search_term_df, df_targeting_report_final, df_business_report
    except Exception as e:
        st.error(f"Error loading files: {str(e)}")
        return None, None, None, None, None, None

def process_impression_share_analysis(df_ad_product, st_imp_df, selected_asin, df_business_report, df_targeting_report_final, df_top_search_term_final, st_imp_top_search_term_df, st_imp_brand_df=None, sqp_df=None):
    """
    Process impression share analysis for sponsored product search terms with business report benchmarking
    Includes both SP (Sponsored Product) and SB (Sponsored Brand) match type targeting
    
    Parameters:
    - sqp_df: Optional DataFrame with Search Query Performance data for trend analysis (from Tab 1)
    """
    # Get campaigns for the selected ASIN
    campaigns = df_ad_product[df_ad_product['Advertised ASIN'] == selected_asin]['Campaign Name'].unique()
    
    # Filter search term impression data for these campaigns
    filtered_st_imp_df = st_imp_df[st_imp_df['Campaign Name'].isin(campaigns)].copy()
    
    if len(filtered_st_imp_df) == 0:
        return None, None
    
    # Get baseline Unit Session Percentage from Business Report
    baseline_unit_session_percentage = None
    if df_business_report is not None:
        try:
            business_asin_filter = df_business_report[df_business_report['(Child) ASIN'] == selected_asin]
            if not business_asin_filter.empty:
                raw_baseline = business_asin_filter['Unit Session Percentage'].iloc[0]
                if isinstance(raw_baseline, str):
                    baseline_unit_session_percentage = float(raw_baseline.replace('%', ''))
                else:
                    baseline_unit_session_percentage = float(raw_baseline)
        except:
            baseline_unit_session_percentage = None
    
    # Ensure numeric columns are properly formatted
    numeric_columns = ['Search Term Impression Rank', 'Search Term Impression Share', 
                      'Impressions', 'Clicks', 'Spend', '7 Day Total Orders (#)', 
                      '7 Day Total Sales ($)']
    
    for col in numeric_columns:
        if col in filtered_st_imp_df.columns:
            filtered_st_imp_df[col] = filtered_st_imp_df[col].astype(str).str.replace('$', '', regex=False)
            filtered_st_imp_df[col] = filtered_st_imp_df[col].astype(str).str.replace('%', '', regex=False)
            filtered_st_imp_df[col] = pd.to_numeric(filtered_st_imp_df[col], errors='coerce')
    
    # Group by Customer Search Term
    grouped_df = filtered_st_imp_df.groupby(['Customer Search Term']).agg({
        'Search Term Impression Rank': 'mean',
        'Search Term Impression Share': 'mean',
        'Impressions': 'sum',
        'Clicks': 'sum',
        'Spend': 'sum',
        '7 Day Total Orders (#)': 'sum',
        '7 Day Total Sales ($)': 'sum'
    }).reset_index()
    
    # Store unfiltered data before applying order filter
    all_grouped_df = grouped_df.copy()
    
    # Filter for search terms with ≥ 3 orders for main analysis
    grouped_df = grouped_df[grouped_df['7 Day Total Orders (#)'] >= 3].copy()
    
    if len(grouped_df) == 0:
        # Return None for main df but still return unfiltered data
        return None, None, all_grouped_df
    
    # Convert impression share to decimal if it's in percentage
    if grouped_df['Search Term Impression Share'].max() > 1:
        grouped_df['Search Term Impression Share'] = grouped_df['Search Term Impression Share'] / 100
    
    # Calculate ACR (Ad Conversion Rate)
    grouped_df['ACR'] = np.where(
        grouped_df['Clicks'] != 0,
        grouped_df['7 Day Total Orders (#)'] * 100 / grouped_df['Clicks'],
        None
    )
    
    # Convert impression share back to percentage for display
    grouped_df['Search Term Impression Share'] = grouped_df['Search Term Impression Share'] * 100
    
    # Select final columns
    final_df = grouped_df[['Customer Search Term', 'Search Term Impression Rank', 
                          'Search Term Impression Share', 'Impressions', 'Clicks',
                          '7 Day Total Orders (#)', 'ACR']].copy()
    
    final_df.columns = ['Search Term', 'Impression Rank', 'Impression Share %', 
                       'Impressions', 'Clicks', 'Orders', 'ACR %']
    
    # Clean ACR column to remove % and convert to numeric for calculations
    if 'ACR %' in final_df.columns:
        # Store original ACR for display
        final_df['ACR_display'] = final_df['ACR %'].astype(str)
        
        # Clean ACR for calculations
        final_df['ACR_numeric'] = final_df['ACR %'].astype(str).str.replace('%', '', regex=False)
        final_df['ACR_numeric'] = pd.to_numeric(final_df['ACR_numeric'], errors='coerce')
        
        # Replace the ACR % column with cleaned numeric values temporarily for calculations
        final_df['ACR %'] = final_df['ACR_numeric']
    
    # Add baseline comparison if available
    if baseline_unit_session_percentage is not None:
        final_df['Baseline USP %'] = f"{baseline_unit_session_percentage}%"
        final_df['ACR vs Baseline'] = (final_df['ACR %'] - baseline_unit_session_percentage).round(2)
        
        # Categorize search terms based on scenarios
        def categorize_search_term(row):
            acr = row['ACR %']
            rank = row['Impression Rank']
            baseline = baseline_unit_session_percentage
            
            if pd.isna(acr) or pd.isna(baseline):
                return "Insufficient Data"
            
            # Scenario A: Conversion Rate Comparable/Higher than Baseline
            if acr >= baseline:
                if rank <= 2:
                    return "High Performing - Top Rank"
                else:
                    return "High Performing - Improve Impression Share"
            elif acr >= (baseline * 0.75):  # More than 75% of USP
                return "Promising - Scale Impression Share"
            elif acr < (baseline * 0.75):  # Scenario B: More than 25% below baseline
                return "Gray Zone - Expensive to Scale"
            else:
                return "Below Baseline - Needs Optimization"
        
        final_df['Category'] = final_df.apply(categorize_search_term, axis=1)
        
        # Add recommendations
        def get_recommendations(row):
            category = row['Category']
            rank = row['Impression Rank']
            
            if category == "High Performing - Top Rank":
                return "✅ Maintain current strategy, monitor performance"
            elif category == "High Performing - Improve Impression Share":
                return "🚀 Increase bids to improve impression share rank"
            elif category == "Promising - Scale Impression Share":
                return "📈 Scale advertising efforts, check match types and targeting"
            elif category == "Gray Zone - Expensive to Scale":
                return "⚠️ Avoid increasing bids unless ranking strategy. Consider sponsored brand video ads if budget allows"
            elif category == "Below Baseline - Needs Optimization":
                return "❌ Review targeting strategy"
            else:
                return "📊 Collect more performance data"
        
        final_df['Recommendations'] = final_df.apply(get_recommendations, axis=1)
    
    # Restore ACR column to display format (with % signs) for final display
    if 'ACR_display' in final_df.columns:
        final_df['ACR %'] = final_df['ACR_display']
        # Clean up temporary columns
        final_df = final_df.drop(['ACR_display', 'ACR_numeric'], axis=1)
    
    # Process targeting report data for match type columns (same as process_search_term_analysis)
    targeting_data = {'EXACT': set(), 'PHRASE': set(), 'BROAD': set()}
    
    if df_targeting_report_final is not None:
        try:
            filtered_targeting_report_df = df_targeting_report_final[df_targeting_report_final['Campaign Name'].isin(campaigns)].reset_index(drop=True)
            
            # Create sets of targeting terms for each match type
            for match_type in ['EXACT', 'PHRASE', 'BROAD']:
                match_data = filtered_targeting_report_df[filtered_targeting_report_df['Match Type'] == match_type]
                targeting_data[match_type] = set(match_data['Targeting'].dropna().unique())
        except Exception as e:
            # If there's an error processing targeting data, continue with empty sets
            pass
    
    # Process brand targeting data for SB match types
    brand_targeting_data = {'EXACT': set(), 'PHRASE': set(), 'BROAD': set()}
    
    if st_imp_brand_df is not None:
        try:
            # Extract match type information from brand search term data
            # The brand data typically has 'Customer Search Term' and 'Match Type' columns
            brand_df_copy = st_imp_brand_df.copy()
            
            # Create sets of brand targeting terms for each match type
            for match_type in ['EXACT', 'PHRASE', 'BROAD']:
                if 'Match Type' in brand_df_copy.columns and 'Customer Search Term' in brand_df_copy.columns:
                    match_data = brand_df_copy[brand_df_copy['Match Type'] == match_type]
                    brand_targeting_data[match_type] = set(match_data['Customer Search Term'].dropna().unique())
        except Exception as e:
            # If there's an error processing brand targeting data, continue with empty sets
            pass
    
    # Add match type columns for SP (Sponsored Product)
    if df_targeting_report_final is not None:
        try:
            # Add columns for each match type
            final_df['SP_EXACT_Match'] = final_df['Search Term'].apply(
                lambda x: 'Targeted' if x in targeting_data['EXACT'] else 'Not Targeted'
            )
            final_df['SP_PHRASE_Match'] = final_df['Search Term'].apply(
                lambda x: 'Targeted' if x in targeting_data['PHRASE'] else 'Not Targeted'
            )
            final_df['SP_BROAD_Match'] = final_df['Search Term'].apply(
                lambda x: 'Targeted' if x in targeting_data['BROAD'] else 'Not Targeted'
            )
        except Exception as e:
            # If there's an error, add empty columns
            final_df['SP_EXACT_Match'] = 'Not Targeted'
            final_df['SP_PHRASE_Match'] = 'Not Targeted'
            final_df['SP_BROAD_Match'] = 'Not Targeted'
    
    # Add match type columns for SB (Sponsored Brand)
    if st_imp_brand_df is not None:
        try:
            # Add columns for each brand match type
            final_df['SB_EXACT_Match'] = final_df['Search Term'].apply(
                lambda x: 'Targeted' if x in brand_targeting_data['EXACT'] else 'Not Targeted'
            )
            final_df['SB_PHRASE_Match'] = final_df['Search Term'].apply(
                lambda x: 'Targeted' if x in brand_targeting_data['PHRASE'] else 'Not Targeted'
            )
            final_df['SB_BROAD_Match'] = final_df['Search Term'].apply(
                lambda x: 'Targeted' if x in brand_targeting_data['BROAD'] else 'Not Targeted'
            )
        except Exception as e:
            # If there's an error, add empty columns
            final_df['SB_EXACT_Match'] = 'Not Targeted'
            final_df['SB_PHRASE_Match'] = 'Not Targeted'
            final_df['SB_BROAD_Match'] = 'Not Targeted'
    
    # Add Top Search Term data if available (same as process_search_term_analysis)
    if df_top_search_term_final is not None:
        try:
            final_df = pd.merge(
                final_df, 
                df_top_search_term_final[['Search Term', 'top_3_click_share', 'top_3_conversion_share']], 
                on='Search Term', 
                how='left'
            )
            
            # Fill NaN values with 0 for the top search term columns
            final_df['top_3_click_share'] = final_df['top_3_click_share'].fillna(0)
            final_df['top_3_conversion_share'] = final_df['top_3_conversion_share'].fillna(0)
            
            # Calculate competitive intensity if we have access to individual product shares
            # We need to merge with the original top search terms data to get individual shares
            try:
                # Get individual product shares for competitive intensity calculation
                individual_shares_df = st_imp_top_search_term_df[['Search Term', 
                                                                # 'Top Clicked Product #1: Click Share',
                                                                # 'Top Clicked Product #2: Click Share', 
                                                                'Top Clicked Product #3: Click Share',
                                                                # 'Top Clicked Product #1: Conversion Share',
                                                                # 'Top Clicked Product #2: Conversion Share',
                                                                'Top Clicked Product #3: Conversion Share']].copy()
                
                # Merge individual shares
                final_df = pd.merge(final_df, individual_shares_df, on='Search Term', how='left')
                
                # Calculate competitive intensity
                def calculate_competitive_intensity(row):
                    # Click competitive intensity
                    remaining_click_share = 100 - row['top_3_click_share']
                    third_click_share = row['Top Clicked Product #3: Click Share']
                    click_intensity = remaining_click_share / third_click_share if third_click_share > 0 else 0
                    
                    # Conversion competitive intensity  
                    remaining_conversion_share = 100 - row['top_3_conversion_share']
                    third_conversion_share = row['Top Clicked Product #3: Conversion Share']
                    conversion_intensity = remaining_conversion_share / third_conversion_share if third_conversion_share > 0 else 0
                    
                    return pd.Series({
                        'Remaining Click Share': remaining_click_share,
                        'Remaining Conversion Share': remaining_conversion_share,
                        'Click Competitive Intensity': round(click_intensity, 2),
                        'Conversion Competitive Intensity': round(conversion_intensity, 2)
                    })
                
                # Apply competitive intensity calculation
                competitive_metrics = final_df.apply(calculate_competitive_intensity, axis=1)
                final_df = pd.concat([final_df, competitive_metrics], axis=1)
                
                # Clean up individual share columns (keep them for reference but move to end)
                # individual_cols = ['Top Clicked Product #1: Click Share', 'Top Clicked Product #2: Click Share', 
                #                  'Top Clicked Product #3: Click Share', 'Top Clicked Product #1: Conversion Share',
                #                  'Top Clicked Product #2: Conversion Share', 'Top Clicked Product #3: Conversion Share']
                
                individual_cols = []
                # Reorder columns to put competitive intensity metrics after top_3 shares
                main_cols = [col for col in final_df.columns if col not in individual_cols]
                final_df = final_df[main_cols + individual_cols]
                
            except Exception as e:
                # If competitive intensity calculation fails, continue without it
                pass
                
        except Exception as e:
            # If there's an error merging top search terms data, continue without it
            pass
    
    # Add trend analysis from Search Query Performance data (Tab 1)
    if sqp_df is not None and len(sqp_df) > 0:
        try:
            # Check which date column exists
            date_col = 'Date' if 'Date' in sqp_df.columns else 'Reporting Date'
            
            # Check for required columns and map them if needed
            required_cols = {
                'Search Query': 'Search Query',
                'Search Query Volume': 'Search Query Volume',
                'Impressions: ASIN Share %': 'Brand Impressions Share' if 'Brand Impressions Share' in sqp_df.columns else 'Impressions: ASIN Share %',
                'Clicks: ASIN Share %': 'Brand Click Share' if 'Brand Click Share' in sqp_df.columns else 'Clicks: ASIN Share %',
                'Purchases: ASIN Share %': 'Brand Purchase Share' if 'Brand Purchase Share' in sqp_df.columns else 'Purchases: ASIN Share %'
            }
            
            # Verify all columns exist
            missing_cols = []
            for orig_col, actual_col in required_cols.items():
                if actual_col not in sqp_df.columns:
                    missing_cols.append(f"{orig_col} (looking for {actual_col})")
            
            if missing_cols:
                st.warning(f"⚠️ Cannot add trends - missing columns: {', '.join(missing_cols)}")
                st.info(f"Available columns: {', '.join(sqp_df.columns.tolist()[:10])}...")
            else:
                # Perform aggregation on SQP data to get trend metrics for each search term
                # Group by Search Query and calculate trend metrics
                sqp_aggregated = sqp_df.groupby('Search Query').agg({
                    required_cols['Search Query Volume']: ['first', 'last', lambda x: ((x.iloc[-1] - x.iloc[0]) / x.iloc[0] * 100) if len(x) >= 2 and x.iloc[0] != 0 else 0],
                    required_cols['Impressions: ASIN Share %']: ['first', 'last', lambda x: ((x.iloc[-1] - x.iloc[0]) / x.iloc[0] * 100) if len(x) >= 2 and x.iloc[0] != 0 else 0],
                    required_cols['Clicks: ASIN Share %']: ['first', 'last', lambda x: ((x.iloc[-1] - x.iloc[0]) / x.iloc[0] * 100) if len(x) >= 2 and x.iloc[0] != 0 else 0],
                    required_cols['Purchases: ASIN Share %']: ['first', 'last', lambda x: ((x.iloc[-1] - x.iloc[0]) / x.iloc[0] * 100) if len(x) >= 2 and x.iloc[0] != 0 else 0],
                    date_col: 'count'
                }).reset_index()
                
                # Flatten column names
                sqp_aggregated.columns = [
                    'Search Query', 
                    'Vol_First', 'Vol_Last', 'Vol_Change_Pct',
                    'Imp_First', 'Imp_Last', 'Imp_Change_Pct',
                    'Click_First', 'Click_Last', 'Click_Change_Pct',
                    'Purch_First', 'Purch_Last', 'Purch_Change_Pct',
                    'Data_Points'
                ]
                
                # Function to format trend
                def format_trend(change_pct):
                    if abs(change_pct) < 5:
                        return f"→ {change_pct:.1f}%"
                    elif change_pct > 0:
                        return f"↑ {change_pct:.1f}%"
                    else:
                        return f"↓ {change_pct:.1f}%"
                
                # Add formatted trend columns
                sqp_aggregated['Vol Trend'] = sqp_aggregated['Vol_Change_Pct'].apply(format_trend)
                sqp_aggregated['Imp Share Trend'] = sqp_aggregated['Imp_Change_Pct'].apply(format_trend)
                sqp_aggregated['Click Share Trend'] = sqp_aggregated['Click_Change_Pct'].apply(format_trend)
                sqp_aggregated['Purch Share Trend'] = sqp_aggregated['Purch_Change_Pct'].apply(format_trend)
                
                # Select only the columns we need for the join
                trend_cols = sqp_aggregated[['Search Query', 'Vol Trend', 'Imp Share Trend', 'Click Share Trend', 'Purch Share Trend', 'Data_Points']]
                
                # Left join with final_df on Search Term = Search Query
                final_df = pd.merge(
                    final_df,
                    trend_cols,
                    left_on='Search Term',
                    right_on='Search Query',
                    how='left'
                )
                
                # Drop the duplicate Search Query column if it exists
                if 'Search Query' in final_df.columns:
                    final_df = final_df.drop(columns=['Search Query'])
                
                # Fill NaN values in trend columns with 'N/A'
                for col in ['Vol Trend', 'Imp Share Trend', 'Click Share Trend', 'Purch Share Trend']:
                    if col in final_df.columns:
                        final_df[col] = final_df[col].fillna('N/A')
                
                if 'Data_Points' in final_df.columns:
                    final_df['Data_Points'] = final_df['Data_Points'].fillna(0).astype(int)
                
                st.success(f"✅ Added trend analysis for {len(trend_cols)} search terms from Tab 1 data")
            
        except Exception as e:
            # If trend analysis fails, continue without it
            st.warning(f"⚠️ Could not add trend analysis: {str(e)}")
    
    # Sort by orders (highest to lowest)
    final_df = final_df.sort_values('Orders', ascending=False)
    
    # Process the unfiltered data for low-order queries analysis
    all_queries_df = None
    if all_grouped_df is not None and len(all_grouped_df) > 0:
        # Convert impression share to decimal if it's in percentage
        if all_grouped_df['Search Term Impression Share'].max() > 1:
            all_grouped_df['Search Term Impression Share'] = all_grouped_df['Search Term Impression Share'] / 100
        
        # Calculate ACR (Ad Conversion Rate)
        all_grouped_df['ACR'] = np.where(
            all_grouped_df['Clicks'] != 0,
            all_grouped_df['7 Day Total Orders (#)'] * 100 / all_grouped_df['Clicks'],
            None
        )
        
        # Convert impression share back to percentage for display
        all_grouped_df['Search Term Impression Share'] = all_grouped_df['Search Term Impression Share'] * 100
        
        # Select final columns
        all_queries_df = all_grouped_df[['Customer Search Term', 'Search Term Impression Rank', 
                              'Search Term Impression Share', 'Impressions', 'Clicks',
                              '7 Day Total Orders (#)', 'ACR']].copy()
        
        all_queries_df.columns = ['Search Term', 'Impression Rank', 'Impression Share %', 
                           'Impressions', 'Clicks', 'Orders', 'ACR %']
        
        # Clean ACR column
        if 'ACR %' in all_queries_df.columns:
            all_queries_df['ACR_display'] = all_queries_df['ACR %'].astype(str)
            all_queries_df['ACR_numeric'] = all_queries_df['ACR %'].astype(str).str.replace('%', '', regex=False)
            all_queries_df['ACR_numeric'] = pd.to_numeric(all_queries_df['ACR_numeric'], errors='coerce')
            all_queries_df['ACR %'] = all_queries_df['ACR_numeric']
        
        # Add the same match type columns and other processing
        # Process targeting report data for match type columns (SP)
        if df_targeting_report_final is not None:
            try:
                for match_type in ['EXACT', 'PHRASE', 'BROAD']:
                    all_queries_df[f'SP_{match_type}_Match'] = all_queries_df['Search Term'].apply(
                        lambda x: 'Targeted' if x in targeting_data[match_type] else 'Not Targeted'
                    )
            except:
                pass
        
        # Process brand targeting data for match type columns (SB)
        if st_imp_brand_df is not None:
            try:
                for match_type in ['EXACT', 'PHRASE', 'BROAD']:
                    all_queries_df[f'SB_{match_type}_Match'] = all_queries_df['Search Term'].apply(
                        lambda x: 'Targeted' if x in brand_targeting_data.get(match_type, set()) else 'Not Targeted'
                    )
            except:
                pass
        
        # Add Top Search Term data if available
        if df_top_search_term_final is not None:
            try:
                all_queries_df = pd.merge(
                    all_queries_df, 
                    df_top_search_term_final[['Search Term', 'top_3_click_share', 'top_3_conversion_share']], 
                    on='Search Term', 
                    how='left'
                )
                all_queries_df['top_3_click_share'] = all_queries_df['top_3_click_share'].fillna(0)
                all_queries_df['top_3_conversion_share'] = all_queries_df['top_3_conversion_share'].fillna(0)
            except:
                pass
        
        # Add trend analysis if available
        if sqp_df is not None and len(sqp_df) > 0:
            try:
                # Reuse the trend columns if they were created
                if 'Vol Trend' in final_df.columns:
                    # Merge trend columns from final_df
                    trend_cols_to_merge = ['Search Term', 'Vol Trend', 'Imp Share Trend', 'Click Share Trend', 'Purch Share Trend', 'Data_Points']
                    available_trend_cols = [col for col in trend_cols_to_merge if col in final_df.columns]
                    
                    if len(available_trend_cols) > 1:  # At least Search Term + one trend column
                        all_queries_df = pd.merge(
                            all_queries_df,
                            final_df[available_trend_cols],
                            on='Search Term',
                            how='left'
                        )
                        
                        # Fill NaN values
                        for col in ['Vol Trend', 'Imp Share Trend', 'Click Share Trend', 'Purch Share Trend']:
                            if col in all_queries_df.columns:
                                all_queries_df[col] = all_queries_df[col].fillna('N/A')
                        
                        if 'Data_Points' in all_queries_df.columns:
                            all_queries_df['Data_Points'] = all_queries_df['Data_Points'].fillna(0).astype(int)
            except:
                pass
    
    return final_df, baseline_unit_session_percentage, all_queries_df

def process_search_term_analysis(df_ad_product, st_imp_df, selected_asin, df_top_search_term_final, df_targeting_report_final, df_business_report):
    """
    Process search term analysis for a selected ASIN
    """
    # Get campaigns for the selected ASIN
    campaigns = df_ad_product[df_ad_product['Advertised ASIN'] == selected_asin]['Campaign Name'].unique()
    
    # Filter search term impression data for these campaigns
    filtered_st_imp_df = st_imp_df[st_imp_df['Campaign Name'].isin(campaigns)].copy()

    # Process targeting report data for match type columns
    targeting_data = {'EXACT': set(), 'PHRASE': set(), 'BROAD': set()}
    
    if df_targeting_report_final is not None:
        try:
            filtered_targeting_report_df = df_targeting_report_final[df_targeting_report_final['Campaign Name'].isin(campaigns)].reset_index(drop=True)
            
            # Create sets of targeting terms for each match type
            for match_type in ['EXACT', 'PHRASE', 'BROAD']:
                match_data = filtered_targeting_report_df[filtered_targeting_report_df['Match Type'] == match_type]
                targeting_data[match_type] = set(match_data['Targeting'].dropna().unique())
        except Exception as e:
            # If there's an error processing targeting data, continue with empty sets
            pass
    
    # Ensure numeric columns are properly formatted
    numeric_columns = ['Search Term Impression Rank', 'Search Term Impression Share', 
                      'Impressions', 'Clicks', 'Spend', '7 Day Total Orders (#)', 
                      '7 Day Total Sales ($)']
    
    for col in numeric_columns:
        if col in filtered_st_imp_df.columns:
            filtered_st_imp_df[col] = filtered_st_imp_df[col].astype(str).str.replace('$', '', regex=False)
            filtered_st_imp_df[col] = filtered_st_imp_df[col].astype(str).str.replace('%', '', regex=False)
            filtered_st_imp_df[col] = pd.to_numeric(filtered_st_imp_df[col], errors='coerce')
    
    # Group by Customer Search Term
    grouped_df = filtered_st_imp_df.groupby(['Customer Search Term']).agg({
        'Search Term Impression Rank': 'mean',
        'Search Term Impression Share': 'mean',
        'Impressions': 'sum',
        'Clicks': 'sum',
        'Spend': 'sum',
        '7 Day Total Orders (#)': 'sum',
        '7 Day Total Sales ($)': 'sum'
    }).reset_index()
    
    # Convert impression share to decimal if it's in percentage
    if grouped_df['Search Term Impression Share'].max() > 1:
        grouped_df['Search Term Impression Share'] = grouped_df['Search Term Impression Share'] / 100
    
    # Calculate additional metrics
    grouped_df['Available impression'] = np.where(
        grouped_df['Search Term Impression Share'] != 0,
        grouped_df['Impressions'] / grouped_df['Search Term Impression Share'],
        None
    )
    
    grouped_df['CTR'] = np.where(
        grouped_df['Available impression'] != 0,
        grouped_df['Clicks'] * 100 / grouped_df['Available impression'],
        None
    )
    
    grouped_df['ACoS'] = np.where(
        grouped_df['7 Day Total Sales ($)'] != 0,
        grouped_df['Spend'] * 100 / grouped_df['7 Day Total Sales ($)'],
        None
    )
    
    grouped_df['ACR'] = np.where(
        grouped_df['Clicks'] != 0,
        grouped_df['7 Day Total Orders (#)'] * 100 / grouped_df['Clicks'],
        None
    )
    
    # Convert impression share back to percentage for display
    grouped_df['Search Term Impression Share'] = grouped_df['Search Term Impression Share'] * 100
    
    # Select final columns
    final_df = grouped_df[['Customer Search Term', 'Search Term Impression Rank', 
                          'Search Term Impression Share', 'Impressions', 'CTR',
                          '7 Day Total Orders (#)', 'ACoS', 'ACR']].copy()
    
    final_df.columns = ['Search Term', 'Search Term Impression Rank', 
                       'Search Term Impression Share', 'Impressions', 'CTR',
                       'Total Orders', 'ACoS', 'ACR']

    # Clean ACR column to remove % and convert to numeric for calculations
    if 'ACR' in final_df.columns:
        # Store original ACR for display
        final_df['ACR_display'] = final_df['ACR'].astype(str)
        
        # Clean ACR for calculations
        final_df['ACR_numeric'] = final_df['ACR'].astype(str).str.replace('%', '', regex=False)
        final_df['ACR_numeric'] = pd.to_numeric(final_df['ACR_numeric'], errors='coerce')
        
        # Replace the ACR column with cleaned numeric values temporarily for calculations
        final_df['ACR'] = final_df['ACR_numeric']

    # Get baseline Unit Session Percentage from Business Report if available
    baseline_unit_session_percentage = None
    if df_business_report is not None:
        try:
            # Filter business report for the selected ASIN using (Child) ASIN column
            business_asin_filter = df_business_report[df_business_report['(Child) ASIN'] == selected_asin]
            
            if not business_asin_filter.empty:
                # Get the Unit Session Percentage value and clean it
                raw_baseline = business_asin_filter['Unit Session Percentage'].iloc[0]
                
                # Remove % sign if present and convert to float
                if isinstance(raw_baseline, str):
                    baseline_unit_session_percentage = float(raw_baseline.replace('%', ''))
                else:
                    baseline_unit_session_percentage = float(raw_baseline)
                
                st.info(f"📋 Business Report Baseline: Unit Session Percentage for ASIN {selected_asin} = {baseline_unit_session_percentage}%")
                
                # Add baseline column to all search terms (display with % sign)
                final_df['Baseline Unit Session %'] = f"{baseline_unit_session_percentage}%"
                
                # Calculate difference from baseline using numeric ACR values
                final_df['ACR vs Baseline'] = (final_df['ACR'] - baseline_unit_session_percentage).round(2)
            else:
                st.warning(f"⚠️ No business report data found for ASIN: {selected_asin}")
        except KeyError as e:
            st.warning(f"⚠️ Expected column not found in Business Report: {str(e)}. Please check column names.")
        except ValueError as e:
            st.warning(f"⚠️ Could not convert Unit Session Percentage to number: {str(e)}")
        except Exception as e:
            st.warning(f"⚠️ Error processing Business Report data: {str(e)}")

    # Restore ACR column to display format (with % signs) for final display
    if 'ACR_display' in final_df.columns:
        final_df['ACR'] = final_df['ACR_display']
        # Clean up temporary columns
        final_df = final_df.drop(['ACR_display', 'ACR_numeric'], axis=1)

    if df_top_search_term_final is not None:
        try:
            final_df = pd.merge(final_df, df_top_search_term_final, on='Search Term', how='left')
        except:
            pass

    # Add match type columns
    if df_targeting_report_final is not None:
        try:
            # Add columns for each match type
            final_df['EXACT_Match'] = final_df['Search Term'].apply(
                lambda x: 'Targeted' if x in targeting_data['EXACT'] else 'Not Targeted'
            )
            final_df['PHRASE_Match'] = final_df['Search Term'].apply(
                lambda x: 'Targeted' if x in targeting_data['PHRASE'] else 'Not Targeted'
            )
            final_df['BROAD_Match'] = final_df['Search Term'].apply(
                lambda x: 'Targeted' if x in targeting_data['BROAD'] else 'Not Targeted'
            )
        except Exception as e:
            # If there's an error, add empty columns
            final_df['EXACT_Match'] = 'Not Targeted'
            final_df['PHRASE_Match'] = 'Not Targeted'
            final_df['BROAD_Match'] = 'Not Targeted'
    
    return final_df

def process_brand_search_term_analysis(st_imp_df, df_top_search_term_final, st_imp_top_search_term_df):
    """
    Process brand search term analysis for all campaigns (no ASIN filtering)
    Handles both 7 Day and 14 Day column formats
    """
    if st_imp_df is None or len(st_imp_df) == 0:
        return None
        
    # Make a copy of the dataframe
    filtered_st_imp_df = st_imp_df.copy()
    
    # Check available columns and determine the correct column names
    available_columns = filtered_st_imp_df.columns.tolist()
    
    # Determine orders and sales column names (could be 7 Day or 14 Day)
    orders_col = None
    sales_col = None
    
    if '14 Day Total Orders (#)' in available_columns:
        orders_col = '14 Day Total Orders (#)'
        sales_col = '14 Day Total Sales ($)'
    elif '7 Day Total Orders (#)' in available_columns:
        orders_col = '7 Day Total Orders (#)'
        sales_col = '7 Day Total Sales ($)'
    
    # Define numeric columns based on available data
    numeric_columns = ['Search Term Impression Rank', 'Search Term Impression Share', 
                      'Impressions', 'Clicks', 'Spend']
    
    if orders_col and orders_col in available_columns:
        numeric_columns.append(orders_col)
    if sales_col and sales_col in available_columns:
        numeric_columns.append(sales_col)
    
    # Clean and convert numeric columns
    for col in numeric_columns:
        if col in filtered_st_imp_df.columns:
            filtered_st_imp_df[col] = filtered_st_imp_df[col].astype(str).str.replace('$', '', regex=False)
            filtered_st_imp_df[col] = filtered_st_imp_df[col].astype(str).str.replace('%', '', regex=False)
            filtered_st_imp_df[col] = pd.to_numeric(filtered_st_imp_df[col], errors='coerce')
    
    # Build aggregation dictionary
    agg_dict = {
        'Search Term Impression Rank': 'mean',
        'Search Term Impression Share': 'mean',
        'Impressions': 'sum',
        'Clicks': 'sum',
        'Spend': 'sum'
    }
    
    if orders_col and orders_col in filtered_st_imp_df.columns:
        agg_dict[orders_col] = 'sum'
    if sales_col and sales_col in filtered_st_imp_df.columns:
        agg_dict[sales_col] = 'sum'
    
    # Group by Customer Search Term
    grouped_df = filtered_st_imp_df.groupby(['Customer Search Term']).agg(agg_dict).reset_index()
    
    # Convert impression share to decimal if it's in percentage
    if len(grouped_df) > 0 and grouped_df['Search Term Impression Share'].max() > 1:
        grouped_df['Search Term Impression Share'] = grouped_df['Search Term Impression Share'] / 100
    
    # Calculate additional metrics
    grouped_df['Available impression'] = np.where(
        grouped_df['Search Term Impression Share'] != 0,
        grouped_df['Impressions'] / grouped_df['Search Term Impression Share'],
        None
    )
    
    grouped_df['CTR'] = np.where(
        grouped_df['Available impression'] != 0,
        grouped_df['Clicks'] * 100 / grouped_df['Available impression'],
        None
    )
    
    # Calculate ACoS if sales column exists
    if sales_col and sales_col in grouped_df.columns:
        grouped_df['ACoS'] = np.where(
            grouped_df[sales_col] != 0,
            grouped_df['Spend'] * 100 / grouped_df[sales_col],
            None
        )
    else:
        grouped_df['ACoS'] = None
    
    # Calculate ACR if orders column exists
    if orders_col and orders_col in grouped_df.columns:
        grouped_df['ACR'] = np.where(
            grouped_df['Clicks'] != 0,
            grouped_df[orders_col] * 100 / grouped_df['Clicks'],
            None
        )
    else:
        grouped_df['ACR'] = None
    
    # Convert impression share back to percentage for display
    grouped_df['Search Term Impression Share'] = grouped_df['Search Term Impression Share'] * 100
    
    # Build final columns list
    final_columns = ['Customer Search Term', 'Search Term Impression Rank', 
                    'Search Term Impression Share', 'Impressions', 'CTR']
    
    if orders_col and orders_col in grouped_df.columns:
        final_columns.append(orders_col)
    
    final_columns.extend(['ACoS', 'ACR'])
    
    # Select final columns
    final_df = grouped_df[final_columns].copy()
    
    # Rename columns for display
    display_columns = ['Search Term', 'Search Term Impression Rank', 
                      'Search Term Impression Share', 'Impressions', 'CTR']
    
    if orders_col and orders_col in grouped_df.columns:
        display_columns.append('Total Orders')
        
    display_columns.extend(['ACoS', 'ACR'])
    
    final_df.columns = display_columns

    if df_top_search_term_final is not None:
        try:
            final_df = pd.merge(final_df, df_top_search_term_final, on='Search Term', how='left')
            
            # Calculate competitive intensity if we have access to individual product shares
            if st_imp_top_search_term_df is not None:
                try:
                    # Get individual product shares for competitive intensity calculation
                    individual_shares_df = st_imp_top_search_term_df[['Search Term', 
                                                                    'Top Clicked Product #1: Click Share',
                                                                    'Top Clicked Product #2: Click Share', 
                                                                    'Top Clicked Product #3: Click Share',
                                                                    'Top Clicked Product #1: Conversion Share',
                                                                    'Top Clicked Product #2: Conversion Share',
                                                                    'Top Clicked Product #3: Conversion Share']].copy()
                    
                    # Merge individual shares
                    final_df = pd.merge(final_df, individual_shares_df, on='Search Term', how='left')
                    
                    # Calculate competitive intensity
                    def calculate_competitive_intensity(row):
                        # Click competitive intensity
                        remaining_click_share = 100 - row['top_3_click_share'] if pd.notna(row['top_3_click_share']) else 0
                        third_click_share = row['Top Clicked Product #3: Click Share'] if pd.notna(row['Top Clicked Product #3: Click Share']) else 0
                        click_intensity = remaining_click_share / third_click_share if third_click_share > 0 else 0
                        
                        # Conversion competitive intensity  
                        remaining_conversion_share = 100 - row['top_3_conversion_share'] if pd.notna(row['top_3_conversion_share']) else 0
                        third_conversion_share = row['Top Clicked Product #3: Conversion Share'] if pd.notna(row['Top Clicked Product #3: Conversion Share']) else 0
                        conversion_intensity = remaining_conversion_share / third_conversion_share if third_conversion_share > 0 else 0
                        
                        return pd.Series({
                            'Remaining Click Share': remaining_click_share,
                            'Remaining Conversion Share': remaining_conversion_share,
                            'Click Competitive Intensity': round(click_intensity, 2),
                            'Conversion Competitive Intensity': round(conversion_intensity, 2)
                        })
                    
                    # Apply competitive intensity calculation
                    competitive_metrics = final_df.apply(calculate_competitive_intensity, axis=1)
                    final_df = pd.concat([final_df, competitive_metrics], axis=1)
                    
                    # Clean up individual share columns (keep them for reference but move to end)
                    individual_cols = ['Top Clicked Product #1: Click Share', 'Top Clicked Product #2: Click Share', 
                                     'Top Clicked Product #3: Click Share', 'Top Clicked Product #1: Conversion Share',
                                     'Top Clicked Product #2: Conversion Share', 'Top Clicked Product #3: Conversion Share']
                    
                    # Reorder columns to put competitive intensity metrics after top_3 shares
                    main_cols = [col for col in final_df.columns if col not in individual_cols]
                    final_df = final_df[main_cols + individual_cols]
                    
                except Exception as e:
                    # If competitive intensity calculation fails, continue without it
                    pass
                    
        except:
            pass
    
    return final_df

def main():
    # App title and description
    st.title("📊 Search Query Performance Analytics")
    st.markdown("**Analyze Amazon search query performance data with interactive visualizations**")
    
    # Create main tabs
    main_tab1, main_tab2 = st.tabs(["📈 Search Query Performance", "🎯 Sponsored Product Analysis"])
    
    with main_tab1:
        # Data Upload & Setup Section
        st.header("📁 Data Upload & Configuration")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("📂 Upload Search Query Performance CSV Files")
            
            uploaded_files = st.file_uploader(
                "Upload all CSV files for Search Query Performance analysis:",
                type="csv",
                accept_multiple_files=True,
                help="Upload multiple CSV files containing search query performance data. All files will be automatically combined.",
                key="csv_uploader_tab1"
            )
            
            # Show uploaded files
            if uploaded_files:
                st.success(f"📊 Uploaded {len(uploaded_files)} CSV files:")
                with st.expander("📋 Uploaded Files", expanded=False):
                    for i, uploaded_file in enumerate(uploaded_files, 1):
                        st.text(f"{i}. {uploaded_file.name}")
            
            # Add some tips
            st.caption("💡 **Tips:**")
            st.caption("• All CSV files will be automatically loaded and combined")
            st.caption("• Make sure all CSV files have the same structure")
            st.caption("• Files should contain 'Search Query', 'Reporting Date' columns")
            st.caption("• **For trend analysis**: Upload files for multiple time periods (e.g., Week 1, Week 2, Week 3, etc.)")
            st.caption("• Single file = single time period = no trends. Multiple files = time series = trend analysis! 📈")
        
        with col2:
            st.subheader("Analysis Settings")
            
            # Frequency selection
            freq_option = st.selectbox(
                "Analysis Frequency:",
                options=["Monthly", "Weekly"],
                index=0,
                help="Choose the frequency for date gap filling"
            )
            freq = "ME" if freq_option == "Monthly" else "W"
            
            # Clear data button
            if st.button("🗑️ Clear Loaded Data", type="secondary"):
                # Clear session state for tab1
                for key in list(st.session_state.keys()):
                    if key.startswith('tab1_'):
                        del st.session_state[key]
                st.cache_data.clear()
                st.success("Data cleared!")
                st.rerun()
            
            # Load data button
            load_data_tab1 = st.button("📊 Process Uploaded Files", type="primary", disabled=not uploaded_files)
        
        # Load data from uploaded files and store in session state
        if uploaded_files and load_data_tab1:
            with st.spinner("Processing uploaded CSV files..."):
                combined_df = load_and_combine_uploaded_csvs(uploaded_files)
        
                
                if combined_df is not None:
                    # Show the actual dates found in the data
                    original_dates = pd.to_datetime(combined_df['Reporting Date']).dt.strftime('%Y-%m-%d').unique()
                    st.info(f"📅 **Original dates found in uploaded files:** {len(original_dates)} unique date(s)")
                    st.caption(f"Raw dates: {', '.join(sorted(original_dates))}")
                    
                    # Process data
                    with st.spinner(f"Processing data and normalizing to {freq_option.lower()} periods..."):
                        full_df = fill_missing_dates(combined_df, 'Reporting Date', freq).fillna(0)
                    
                    # Show normalized dates
                    if freq == 'W':
                        st.success(f"✅ Normalized to **{full_df['Date'].nunique()} week(s)** (starting Mondays)")
                    else:
                        st.success(f"✅ Normalized to **{full_df['Date'].nunique()} month(s)** (month-end dates)")
                    
                    normalized_dates = full_df['Date'].dt.strftime('%Y-%m-%d').unique()
                    st.caption(f"Normalized dates: {', '.join(sorted(normalized_dates))}")
                    
                    # Store in session state
                    st.session_state.tab1_full_df = full_df
                    st.session_state.tab1_freq_option = freq_option
                    st.session_state.tab1_data_loaded = True
        # Check if data exists in session state
        if hasattr(st.session_state, 'tab1_data_loaded') and st.session_state.tab1_data_loaded:
            # Retrieve data from session state
            full_df = st.session_state.tab1_full_df
            stored_freq_option = st.session_state.tab1_freq_option
            
            # Data summary in columns
            st.subheader("📈 Data Summary")
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                st.metric("Total Records", len(full_df))
            with col2:
                st.metric("Unique Search Queries", full_df['Search Query'].nunique())
            with col3:
                st.metric("Date Range Start", full_df['Date'].min().strftime('%Y-%m-%d'))
            with col4:
                st.metric("Date Range End", full_df['Date'].max().strftime('%Y-%m-%d'))
            with col5:
                # Count unique dates to show number of time periods
                unique_dates = full_df['Date'].nunique()
                st.metric("Time Periods", unique_dates)
            
            # Show warning if only one time period (insufficient for trend analysis)
            if full_df['Date'].nunique() == 1:
                st.warning("⚠️ **Single Time Period Detected**: You've uploaded data for only one time period. Trend analysis requires at least 2 time periods. Please upload multiple CSV files (e.g., multiple weeks or months) to enable trend analysis.")
                st.info(f"💡 **Tip for {stored_freq_option} Data**: Upload CSV files for at least 2 different {stored_freq_option.lower()} periods. For example, if you selected 'Weekly', upload files for Week 1, Week 2, Week 3, etc.")
            
            # Search query selection
            st.subheader("🎯 Search Query Filter")
            search_queries = sorted(full_df['Search Query'].unique())
            selected_query = st.selectbox(
                "Select Search Query for Analysis:",
                options=search_queries,
                index=search_queries.index('three farmer') if 'three farmer' in search_queries else 0,
                help="Choose a search query to analyze in the dashboard"
            )
            
            st.success(f"✅ Data loaded successfully! Selected query: **{selected_query}**")
            
            # Filter data for analysis
            filtered_df = full_df[full_df['Search Query'] == selected_query].reset_index(drop=True)
            filtered_df = filtered_df.sort_values(by='Date', ascending=True)
            
            # Define required columns and their display names
            required_columns = [
                'Date', 'Search Query Score', 'Search Query Volume', 'Purchases: Total Count', 
                'Impressions: ASIN Share %', 'Clicks: Click Rate %', 'Clicks: ASIN Share %', 
                'Cart Adds: Cart Add Rate %', 'Cart Adds: ASIN Share %', 'Purchases: Purchase Rate %', 
                'Purchases: ASIN Share %'
            ]
            
            display_columns = [
                'Date', 'Search Query Score', 'Search Query Volume', 'Total Purchase Count', 
                'Brand Impressions Share', 'Click Rate', 'Brand Click Share', 'Cart Add Rate', 
                'Brand Cart Adds Share', 'Purchase Rate', 'Brand Purchase Share'
            ]
            
            # Check if all required columns exist
            missing_columns = [col for col in required_columns if col not in filtered_df.columns]
            if missing_columns:
                st.error(f"Missing columns in data: {missing_columns}")
                st.write("Available columns:", list(filtered_df.columns))
            else:
                # Select and rename columns
                filtered_df = filtered_df[required_columns].copy()
                filtered_df.columns = display_columns
                
                # Analytics Dashboard Section
                st.header(f"🎯 Analytics Dashboard: **{selected_query}**")
                st.caption(f"Analysis Frequency: {freq_option}")
                
                # Perform Trend Analysis
                st.subheader("📈 Trend Analysis & Insights")
                
                with st.spinner("Analyzing trends..."):
                    trend_analysis = analyze_search_term_trends(filtered_df)
                
                if trend_analysis:
                    # Display Key Insights
                    st.markdown("### 🔍 Key Insights")
                    for insight in trend_analysis['insights']:
                        st.markdown(insight)
                    
                    st.markdown("---")
                    
                    # Trend Summary
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    with col1:
                        vol_trend = trend_analysis['search_volume']
                        trend_emoji = {"up": "📈", "down": "📉", "flat": "➡️", "fluctuating": "📊"}.get(vol_trend['direction'], "❓")
                        st.metric(
                            "Search Volume Trend",
                            f"{trend_emoji} {vol_trend['direction'].title()}",
                            f"{vol_trend['change_pct']:.1f}%"
                        )
                    
                    with col2:
                        imp_trend = trend_analysis['impression_share']
                        trend_emoji = {"up": "📈", "down": "📉", "flat": "➡️", "fluctuating": "📊"}.get(imp_trend['direction'], "❓")
                        st.metric(
                            "Impression Share",
                            f"{trend_emoji} {imp_trend['direction'].title()}",
                            f"{imp_trend['change_pct']:.1f}%"
                        )
                    
                    with col3:
                        click_trend = trend_analysis['click_share']
                        trend_emoji = {"up": "📈", "down": "📉", "flat": "➡️", "fluctuating": "📊"}.get(click_trend['direction'], "❓")
                        st.metric(
                            "Click Share",
                            f"{trend_emoji} {click_trend['direction'].title()}",
                            f"{click_trend['change_pct']:.1f}%"
                        )
                    
                    with col4:
                        purch_trend = trend_analysis['purchase_share']
                        trend_emoji = {"up": "📈", "down": "📉", "flat": "➡️", "fluctuating": "📊"}.get(purch_trend['direction'], "❓")
                        st.metric(
                            "Purchase Share",
                            f"{trend_emoji} {purch_trend['direction'].title()}",
                            f"{purch_trend['change_pct']:.1f}%"
                        )
                    
                    with col5:
                        total_purch_trend = trend_analysis['total_purchases']
                        trend_emoji = {"up": "📈", "down": "📉", "flat": "➡️", "fluctuating": "📊"}.get(total_purch_trend['direction'], "❓")
                        st.metric(
                            "Total Purchases",
                            f"{trend_emoji} {total_purch_trend['direction'].title()}",
                            f"{total_purch_trend['change_pct']:.1f}%"
                        )
                    
                    # Correlations Section
                    if trend_analysis['correlations']:
                        st.markdown("### 🔗 Metric Correlations")
                        corr_col1, corr_col2, corr_col3, corr_col4 = st.columns(4)
                        
                        with corr_col1:
                            corr_val = trend_analysis['correlations'].get('volume_vs_impression', 0)
                            st.metric(
                                "Volume ↔ Impression",
                                f"{corr_val:.2f}",
                                help="Correlation between search volume and impression share"
                            )
                        
                        with corr_col2:
                            corr_val = trend_analysis['correlations'].get('impression_vs_click', 0)
                            st.metric(
                                "Impression ↔ Click",
                                f"{corr_val:.2f}",
                                help="Correlation between impression share and click share"
                            )
                        
                        with corr_col3:
                            corr_val = trend_analysis['correlations'].get('click_vs_purchase', 0)
                            st.metric(
                                "Click ↔ Purchase",
                                f"{corr_val:.2f}",
                                help="Correlation between click share and purchase share"
                            )
                        
                        with corr_col4:
                            corr_val = trend_analysis['correlations'].get('volume_vs_purchases', 0)
                            st.metric(
                                "Volume ↔ Purchases",
                                f"{corr_val:.2f}",
                                help="Correlation between search volume and total purchases"
                            )
                        
                        st.caption("📊 Correlation values range from -1 to +1. Values > 0.7 indicate strong positive correlation, < 0.3 indicate weak correlation.")
                    
                    # Period-over-Period Purchases
                    if trend_analysis['mom_purchases']:
                        st.markdown("### 💰 Period-over-Period Purchase Analysis")
                        
                        # Create a dataframe for better display
                        mom_df = pd.DataFrame(trend_analysis['mom_purchases'])
                        mom_df['Prev Purchases'] = mom_df['prev'].astype(int)
                        mom_df['Current Purchases'] = mom_df['curr'].astype(int)
                        mom_df['Change (#)'] = mom_df['change'].astype(int)
                        mom_df['Change (%)'] = mom_df['change_pct'].apply(lambda x: f"{x:.1f}%")
                        mom_df['Period'] = mom_df['period']
                        
                        display_mom_df = mom_df[['Period', 'Prev Purchases', 'Current Purchases', 'Change (#)', 'Change (%)']]
                        
                        st.dataframe(
                            display_mom_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Period": st.column_config.TextColumn("Period", width="medium"),
                                "Prev Purchases": st.column_config.NumberColumn("Previous", format="%d"),
                                "Current Purchases": st.column_config.NumberColumn("Current", format="%d"),
                                "Change (#)": st.column_config.NumberColumn("Change", format="%d"),
                                "Change (%)": st.column_config.TextColumn("Change %")
                            }
                        )
                        
                        st.caption(f"📈 Analyzing absolute purchase changes across {len(trend_analysis['mom_purchases'])} periods with {trend_analysis['data_points']} data points")
                    
                    st.markdown("---")
                else:
                    st.warning("⚠️ **Insufficient data for trend analysis**")
                    st.info(f"""
                    **Why am I seeing this?**
                    - Trend analysis requires at least 2 time periods with non-zero search volume for '{selected_query}'
                    - Currently, there is only 1 data point or all search volumes are zero
                    
                    **How to fix this:**
                    - Upload CSV files for **multiple {stored_freq_option.lower()} periods** (e.g., Week 1, Week 2, Week 3, etc.)
                    - Ensure the search query '{selected_query}' has non-zero volume in at least 2 periods
                    - Use the file uploader above to add more CSV files, then click 'Process Uploaded Files' again
                    """)
                
                # Create and display plots
                st.subheader("📊 Interactive Visualizations")
                
                # Add Top K Products Selection
                st.markdown("---")
                col_viz1, col_viz2 = st.columns([2, 1])
                
                with col_viz1:
                    st.markdown("### 🏆 Top Search Queries by Volume")
                    st.caption("Analyze multiple search queries based on their average search volume")
                
                with col_viz2:
                    # Calculate average search volume for each search query
                    avg_volumes = full_df.groupby('Search Query')['Search Query Volume'].mean().sort_values(ascending=False)
                    
                    # Top K selector
                    max_products = min(20, len(avg_volumes))
                    top_k = st.slider(
                        "Select number of top queries:",
                        min_value=1,
                        max_value=max_products,
                        value=min(5, max_products),
                        help="Select how many top search queries to display based on average search volume"
                    )
                
                # Get top K products
                top_products = avg_volumes.head(top_k).index.tolist()
                
                st.success(f"📈 Displaying top {top_k} search queries by average search volume")
                
                # Display each product's visualization
                for idx, query in enumerate(top_products, 1):
                    st.markdown(f"## {idx}. **{query}**")
                    
                    # Filter data for this specific query
                    query_filtered_df = full_df[full_df['Search Query'] == query].reset_index(drop=True)
                    query_filtered_df = query_filtered_df.sort_values(by='Date', ascending=True)
                    
                    # Select and rename columns for display
                    query_display_df = query_filtered_df[required_columns].copy()
                    query_display_df.columns = display_columns
                    
                    # Show key metrics
                    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                    
                    with metric_col1:
                        avg_volume = query_display_df['Search Query Volume'].mean()
                        st.metric("Avg Search Volume", f"{avg_volume:,.0f}")
                    
                    with metric_col2:
                        avg_purchases = query_display_df['Total Purchase Count'].mean()
                        st.metric("Avg Purchases", f"{avg_purchases:.1f}")
                    
                    with metric_col3:
                        avg_imp_share = query_display_df['Brand Impressions Share'].mean()
                        st.metric("Avg Impression Share", f"{avg_imp_share:.2f}%")
                    
                    with metric_col4:
                        avg_purchase_share = query_display_df['Brand Purchase Share'].mean()
                        st.metric("Avg Purchase Share", f"{avg_purchase_share:.2f}%")
                    
                    # Create plots for this query
                    plots = create_interactive_plots(query_display_df)
                    
                    # Display plots in sub-tabs
                    plot_tab_names = [plot[0] for plot in plots]
                    plot_tabs = st.tabs(plot_tab_names)
                    
                    for i, (plot_tab, (plot_name, fig)) in enumerate(zip(plot_tabs, plots)):
                        with plot_tab:
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Display the raw data table in a collapsible expander
                    with st.expander(f"📋 View Data Table for '{query}'", expanded=False):
                        st.dataframe(
                            query_display_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Date": st.column_config.DateColumn("Date", format="YYYY-MM-DD"),
                                "Search Query Score": st.column_config.NumberColumn("Score", format="%.2f"),
                                "Search Query Volume": st.column_config.NumberColumn("Volume", format="%d"),
                                "Total Purchase Count": st.column_config.NumberColumn("Purchases", format="%d"),
                                "Brand Impressions Share": st.column_config.NumberColumn("Impression Share %", format="%.2f"),
                                "Click Rate": st.column_config.NumberColumn("Click Rate %", format="%.2f"),
                                "Brand Click Share": st.column_config.NumberColumn("Click Share %", format="%.2f"),
                                "Cart Add Rate": st.column_config.NumberColumn("Cart Add Rate %", format="%.2f"),
                                "Brand Cart Adds Share": st.column_config.NumberColumn("Cart Add Share %", format="%.2f"),
                                "Purchase Rate": st.column_config.NumberColumn("Purchase Rate %", format="%.2f"),
                                "Brand Purchase Share": st.column_config.NumberColumn("Purchase Share %", format="%.2f")
                            }
                        )
                        
                        # Download button for this query's data
                        csv_data_query = query_display_df.to_csv(index=False)
                        st.download_button(
                            label=f"📥 Download Data for '{query}'",
                            data=csv_data_query,
                            file_name=f"search_term_data_{query.replace(' ', '_')}.csv",
                            mime="text/csv",
                            key=f"download_{idx}_{query.replace(' ', '_')}"
                        )
                        
                        st.caption(f"📊 Showing {len(query_display_df)} data points for '{query}'")
                    
                    # Add separator between products (except for the last one)
                    if idx < len(top_products):
                        st.markdown("---")
        else:
            st.info("👆 Please upload CSV files and click 'Process Uploaded Files' to begin analysis.")
    
    with main_tab2:
        st.header("🎯 Sponsored Product Analysis")
        st.markdown("**Analyze search term performance for sponsored product campaigns**")
        
        # File upload section for sponsored product analysis
        st.subheader("📁 Upload Sponsored Product Files")
        
        uploaded_files_tab2 = st.file_uploader(
            "Upload all files for Sponsored Product analysis:",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=True,
            help="Upload all your sponsored product files (CSV and Excel). The system will automatically detect which file is which based on naming patterns.",
            key="files_uploader_tab2"
        )
        
        # Expected file patterns info
        with st.expander("📋 Expected File Naming Patterns", expanded=False):
            st.markdown("""
            **The system will automatically detect files based on these patterns (case-insensitive):**
            
            - **SP_Ad_product** + `.xlsx/.xls` → Ad Product Report (Excel) ✅ Required
            - **SP_ST_imp** + `.csv` → Product Search Term Impression Share (CSV) 
            - **SB_ST_imp** + `.csv` → Brand Search Term Impression Share (CSV)
            - **Top_search_terms** + `.csv` → Top Search Terms (CSV) 🔸 Optional
            - **SP_Targeting** + `.xlsx/.xls` → Targeting Report (Excel) 🔸 Optional
            - **BusinessReport** + `.csv` → Business Report (CSV) 🔸 Optional
            
            **Examples:**
            - `SP_Ad_product_report.xlsx` ✅
            - `SP_ST_imp_data_Aug.csv` ✅  
            - `SB_ST_imp_brand_data.csv` ✅
            - `Top_search_terms_monthly.csv` ✅
            - `SP_Targeting_report.xlsx` ✅
            - `BusinessReport_Aug2024.csv` ✅
            """)
        
        # Process uploaded files
        if uploaded_files_tab2:
            st.success(f"📁 Uploaded {len(uploaded_files_tab2)} files")
            
            # Auto-detect and categorize files
            with st.spinner("Auto-detecting file types..."):
                excel_file, csv_product_file, csv_brand_file, csv_top_search_term_file, csv_targeting_report_file, csv_business_report_file = auto_detect_uploaded_files(uploaded_files_tab2)
            
            # Process button
            process_files = st.button("🔍 Process Detected Files", type="primary")
            
            if process_files and (excel_file or csv_product_file or csv_brand_file):
                # Load sponsored product data
                with st.spinner("Loading sponsored product data..."):
                    df_ad_product, st_imp_product_df, st_imp_brand_df, st_imp_top_search_term_df, df_targeting_report_final, df_business_report = load_sponsored_product_data_from_uploads(excel_file, csv_product_file, csv_brand_file, csv_top_search_term_file, csv_targeting_report_file, csv_business_report_file)
                
                # Store data in session state for persistence across reruns
                if df_ad_product is not None:
                    st.session_state.tab2_df_ad_product = df_ad_product
                    st.session_state.tab2_st_imp_product_df = st_imp_product_df
                    st.session_state.tab2_st_imp_brand_df = st_imp_brand_df
                    st.session_state.tab2_st_imp_top_search_term_df = st_imp_top_search_term_df
                    st.session_state.tab2_df_targeting_report_final = df_targeting_report_final
                    st.session_state.tab2_df_business_report = df_business_report
                    st.session_state.tab2_data_loaded = True
        
        # Check if data exists in session state (either just loaded or from previous run)
        if hasattr(st.session_state, 'tab2_data_loaded') and st.session_state.tab2_data_loaded:
            # Retrieve data from session state
            df_ad_product = st.session_state.tab2_df_ad_product
            st_imp_product_df = st.session_state.tab2_st_imp_product_df
            st_imp_brand_df = st.session_state.tab2_st_imp_brand_df
            st_imp_top_search_term_df = st.session_state.tab2_st_imp_top_search_term_df
            df_targeting_report_final = st.session_state.tab2_df_targeting_report_final
            df_business_report = st.session_state.tab2_df_business_report
            
            if df_ad_product is not None:
                st.success("✅ Sponsored product data loaded successfully!")
                
                # Process top search term data if available (only process once)
                if hasattr(st.session_state, 'tab2_df_top_search_term_final'):
                    df_top_search_term_final = st.session_state.tab2_df_top_search_term_final
                else:
                    df_top_search_term_final = None
                    if st_imp_top_search_term_df is not None:
                        try:
                            st_imp_top_search_term_df['top_3_click_share'] = st_imp_top_search_term_df['Top Clicked Product #1: Click Share'] + st_imp_top_search_term_df['Top Clicked Product #2: Click Share'] + st_imp_top_search_term_df['Top Clicked Product #3: Click Share']
                            st_imp_top_search_term_df['top_3_conversion_share'] = st_imp_top_search_term_df['Top Clicked Product #1: Conversion Share'] + st_imp_top_search_term_df['Top Clicked Product #2: Conversion Share'] + st_imp_top_search_term_df['Top Clicked Product #3: Conversion Share']
                            df_top_search_term_final = st_imp_top_search_term_df[['Search Term','top_3_click_share','top_3_conversion_share']]
                            st.success("✅ Top search terms data processed successfully! Extra columns will be added to analysis.")
                            # Store processed data in session state
                            st.session_state.tab2_df_top_search_term_final = df_top_search_term_final
                        except Exception as e:
                            st.warning(f"⚠️ Could not process top search terms data: {str(e)}")
                            df_top_search_term_final = None
                    
                    # Process targeting report data if available
                    if df_targeting_report_final is not None:
                        st.success("✅ Targeting report data loaded successfully! Match type columns will be added to analysis.")
                    
                # Process business report data if available
                if df_business_report is not None:
                    st.success("✅ Business report data loaded successfully! Business metrics will be available for analysis.")
                
                # Clear data button
                if st.button("🗑️ Clear Loaded Data", type="secondary", help="Clear all loaded data and start fresh"):
                    # Clear all tab2 session state data
                    for key in list(st.session_state.keys()):
                        if key.startswith('tab2_'):
                            del st.session_state[key]
                    st.rerun()
                
                # Display data summary
                st.subheader("📈 Data Summary")
                col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
                
                with col1:
                    st.metric("Total Ad Records", len(df_ad_product))
                with col2:
                    st.metric("Unique ASINs", df_ad_product['Advertised ASIN'].nunique())
                with col3:
                    if st_imp_product_df is not None:
                        st.metric("Product Search Terms", len(st_imp_product_df))
                    else:
                        st.metric("Product Search Terms", "N/A")
                with col4:
                    if st_imp_brand_df is not None:
                        st.metric("Brand Search Terms", len(st_imp_brand_df))
                    else:
                        st.metric("Brand Search Terms", "N/A")
                with col5:
                    if df_top_search_term_final is not None:
                        st.metric("Top Search Terms", len(df_top_search_term_final))
                    else:
                        st.metric("Top Search Terms", "Not Loaded")
                with col6:
                    if df_targeting_report_final is not None:
                        st.metric("Targeting Records", len(df_targeting_report_final))
                    else:
                        st.metric("Targeting Records", "Not Loaded")
                with col7:
                    if df_business_report is not None:
                        st.metric("Business Report", len(df_business_report))
                    else:
                        st.metric("Business Report", "Not Loaded")
                
                # Create analysis tabs for Product and Brand
                analysis_tabs = []
                if st_imp_product_df is not None:
                    analysis_tabs.append("📊 Product Analysis")
                if st_imp_brand_df is not None:
                    analysis_tabs.append("🏷️ Brand Analysis")
                # Add Impression Share Analysis tab
                if st_imp_product_df is not None:
                    analysis_tabs.append("🎯 Impression Share Analysis")
                
                if analysis_tabs:
                    analysis_tab_objects = st.tabs(analysis_tabs)
                    
                    tab_index = 0
                    
                # Product Analysis Tab
                if st_imp_product_df is not None:
                    with analysis_tab_objects[tab_index]:
                        st.subheader("🎯 ASIN Selection for Product Analysis")
                        
                        available_asins = sorted(df_ad_product['Advertised ASIN'].unique())
                        selected_asin = st.selectbox(
                            "Select ASIN for Product Analysis:",
                            options=available_asins,
                            index=available_asins.index('B0BH6G8Q94') if 'B0BH6G8Q94' in available_asins else 0,
                            help="Choose an ASIN to analyze search term performance",
                            key="product_asin_selector"
                        )
                        # Process and display search term analysis for product
                        with st.spinner("Processing product search term analysis..."):
                            product_search_term_df = process_search_term_analysis(df_ad_product, st_imp_product_df, selected_asin, df_top_search_term_final, df_targeting_report_final, df_business_report)
                        
                        if product_search_term_df is not None and len(product_search_term_df) > 0:
                                st.subheader(f"📊 Product Search Term Analysis for ASIN: **{selected_asin}**")
                                
                                # Display the full search term analysis table
                                st.subheader("📋 Complete Product Search Term Analysis")
                                
                                # Display filtered results
                                st.dataframe(
                                    product_search_term_df,
                                    use_container_width=True,
                                    hide_index=True,
                                    column_config={
                                        "Search Term Impression Share": st.column_config.NumberColumn(
                                            "Impression Share (%)",
                                            format="%.2f%%"
                                        ),
                                        "CTR": st.column_config.NumberColumn(
                                            "CTR (%)",
                                            format="%.2f%%"
                                        ),
                                        "ACoS": st.column_config.NumberColumn(
                                            "ACoS (%)",
                                            format="%.2f%%"
                                        ),
                                        "ACR": st.column_config.NumberColumn(
                                            "ACR (%)",
                                            format="%.2f%%"
                                        ),
                                        "EXACT_Match": st.column_config.TextColumn(
                                            "EXACT Match",
                                            help="Shows if this search term is targeted with EXACT match type"
                                        ),
                                        "PHRASE_Match": st.column_config.TextColumn(
                                            "PHRASE Match",
                                            help="Shows if this search term is targeted with PHRASE match type"
                                        ),
                                        "BROAD_Match": st.column_config.TextColumn(
                                            "BROAD Match",
                                            help="Shows if this search term is targeted with BROAD match type"
                                        )
                                    }
                                )
                                
                                # Download button for the processed data
                                csv_data = product_search_term_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Product Search Term Analysis",
                                    data=csv_data,
                                    file_name=f"product_search_term_analysis_{selected_asin}.csv",
                                    mime="text/csv",
                                    key="download_product_analysis"
                                )
                        else:
                            st.warning("No product search term data found for the selected ASIN.")
                        
                        tab_index += 1
                    
                    # Brand Analysis Tab
                    if st_imp_brand_df is not None:
                        with analysis_tab_objects[tab_index]:
                            st.subheader("🏷️ Brand Search Term Analysis")
                            st.caption("Analysis across all campaigns (no ASIN filtering) - 14 Day Attribution")
                            
                            # Process and display search term analysis for brand
                            with st.spinner("Processing brand search term analysis..."):
                                brand_search_term_df = process_brand_search_term_analysis(st_imp_brand_df, df_top_search_term_final, st_imp_top_search_term_df)
                            
                            if brand_search_term_df is not None and len(brand_search_term_df) > 0:
                                # Display key metrics for selected search terms
                                three_farmers_data = brand_search_term_df[brand_search_term_df['Search Term'] == 'three farmers']
                                if not three_farmers_data.empty:
                                    st.info("**🔍 'Three Farmers' Search Term Metrics (Brand - 14 Day):**")
                                    col1, col2, col3, col4 = st.columns(4)
                                    
                                    with col1:
                                        st.metric("Impressions", f"{three_farmers_data.iloc[0]['Impressions']:,.0f}")
                                    with col2:
                                        st.metric("CTR", f"{three_farmers_data.iloc[0]['CTR']:.2f}%" if pd.notnull(three_farmers_data.iloc[0]['CTR']) else "N/A")
                                    with col3:
                                        st.metric("Total Orders", f"{three_farmers_data.iloc[0]['Total Orders']:,.0f}")
                                    with col4:
                                        st.metric("ACoS", f"{three_farmers_data.iloc[0]['ACoS']:.2f}%" if pd.notnull(three_farmers_data.iloc[0]['ACoS']) else "N/A")
                                
                                # Display the full search term analysis table
                                st.subheader("📋 Complete Brand Search Term Analysis")
                                
                                # Display filtered results
                                st.dataframe(
                                    brand_search_term_df,
                                    use_container_width=True,
                                    hide_index=True,
                                    column_config={
                                        "Search Term Impression Share": st.column_config.NumberColumn(
                                            "Impression Share (%)",
                                            format="%.2f%%"
                                        ),
                                        "CTR": st.column_config.NumberColumn(
                                            "CTR (%)",
                                            format="%.2f%%"
                                        ),
                                        "ACoS": st.column_config.NumberColumn(
                                            "ACoS (%)",
                                            format="%.2f%%"
                                        ),
                                        "ACR": st.column_config.NumberColumn(
                                            "ACR (%)",
                                            format="%.2f%%"
                                        ),
                                        "top_3_click_share": st.column_config.NumberColumn(
                                            "Top 3 Click Share",
                                            help="Combined click share of top 3 clicked products for this search term",
                                            format="%.2f"
                                        ),
                                        "top_3_conversion_share": st.column_config.NumberColumn(
                                            "Top 3 Conversion Share", 
                                            help="Combined conversion share of top 3 clicked products for this search term",
                                            format="%.2f"
                                        ),
                                        "Remaining Click Share": st.column_config.NumberColumn(
                                            "Remaining Click Share",
                                            help="Market share not captured by top 3 products (100 - top_3_click_share)",
                                            format="%.2f"
                                        ),
                                        "Remaining Conversion Share": st.column_config.NumberColumn(
                                            "Remaining Conversion Share",
                                            help="Conversion share not captured by top 3 products (100 - top_3_conversion_share)",
                                            format="%.2f"
                                        ),
                                        "Click Competitive Intensity": st.column_config.NumberColumn(
                                            "Click Competitive Intensity",
                                            help="Minimum # of competing products winning clicks (Remaining / 3rd product share)",
                                            format="%.2f"
                                        ),
                                        "Conversion Competitive Intensity": st.column_config.NumberColumn(
                                            "Conversion Competitive Intensity",
                                            help="Minimum # of competing products winning conversions (Remaining / 3rd product share)",
                                            format="%.2f"
                                        )
                                        # "EXACT_Match": st.column_config.TextColumn(
                                        #     "EXACT Match",
                                        #     help="Shows if this search term is targeted with EXACT match type"
                                        # ),
                                        # "PHRASE_Match": st.column_config.TextColumn(
                                        #     "PHRASE Match",
                                        #     help="Shows if this search term is targeted with PHRASE match type"
                                        # ),
                                        # "BROAD_Match": st.column_config.TextColumn(
                                        #     "BROAD Match",
                                        #     help="Shows if this search term is targeted with BROAD match type"
                                        # )
                                    }
                                )
                                
                                # Download button for the processed data
                                csv_data = brand_search_term_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Brand Search Term Analysis",
                                    data=csv_data,
                                    file_name="brand_search_term_analysis.csv",
                                    mime="text/csv",
                                    key="download_brand_analysis"
                                )
                            else:
                                st.warning("No brand search term data found.")
                        
                        tab_index += 1
                    
                    # Impression Share Analysis Tab
                    if st_imp_product_df is not None:
                        with analysis_tab_objects[tab_index]:
                            st.subheader("🎯 Search Term Impression Share Analysis")
                            st.markdown("**Advanced analysis focusing on impression share optimization opportunities**")
                            
                            # ASIN Selection for Impression Share Analysis
                            st.subheader("📋 Analysis Configuration")
                            unique_asins = sorted(df_ad_product['Advertised ASIN'].unique())
                            selected_asin_imp = st.selectbox(
                                "Select ASIN for Impression Share Analysis:",
                                options=unique_asins,
                                help="Choose an ASIN to analyze impression share opportunities",
                                key="impression_share_asin_selector"
                            )
                            
                            # Process impression share analysis
                            with st.spinner("Processing impression share analysis..."):
                                # Check if Tab 1 SQP data is available in session state
                                sqp_data = None
                                if hasattr(st.session_state, 'tab1_full_df'):
                                    sqp_data = st.session_state.tab1_full_df
                                    st.info(f"📊 Using Search Query Performance data from Tab 1 for trend analysis ({len(sqp_data)} records)")
                                
                                impression_share_df, baseline, all_queries_df = process_impression_share_analysis(
                                    df_ad_product, st_imp_product_df, selected_asin_imp, 
                                    df_business_report, df_targeting_report_final, df_top_search_term_final, st_imp_top_search_term_df, st_imp_brand_df, sqp_data
                                )
                            
                            if impression_share_df is not None and len(impression_share_df) > 0:
                                st.subheader(f"📊 Impression Share Analysis for ASIN: **{selected_asin_imp}**")
                                
                                # Display baseline information if available
                                if baseline is not None:
                                    st.info(f"📋 **Business Report Baseline:** Unit Session Percentage = {baseline}%")
                                    st.caption("Search terms are categorized based on their ACR performance vs. this baseline")
                                
                                # Show summary metrics
                                col1, col2, col3, col4, col5 = st.columns(5)
                                
                                with col1:
                                    st.metric("Total Search Terms", len(impression_share_df))
                                with col2:
                                    high_performing = len(impression_share_df[impression_share_df['Category'].str.contains('High Performing', na=False)])
                                    st.metric("High Performing", high_performing)
                                with col3:
                                    promising = len(impression_share_df[impression_share_df['Category'].str.contains('Promising', na=False)])
                                    st.metric("Promising Terms", promising)
                                with col4:
                                    gray_zone = len(impression_share_df[impression_share_df['Category'].str.contains('Gray Zone', na=False)])
                                    st.metric("Gray Zone Terms", gray_zone)
                                with col5:
                                    # Average competitive intensity for clicks (if available)
                                    if 'Click Competitive Intensity' in impression_share_df.columns:
                                        avg_click_intensity = impression_share_df['Click Competitive Intensity'].mean()
                                        st.metric("Avg Click Intensity", f"{avg_click_intensity:.1f}")
                                    else:
                                        total_orders = impression_share_df['Orders'].sum()
                                        st.metric("Total Orders", f"{total_orders:,.0f}")
                                
                                # Category breakdown
                                if 'Category' in impression_share_df.columns:
                                    st.subheader("📈 Performance Categories")
                                    category_counts = impression_share_df['Category'].value_counts()
                                    
                                    category_cols = st.columns(len(category_counts))
                                    for idx, (category, count) in enumerate(category_counts.items()):
                                        with category_cols[idx % len(category_cols)]:
                                            if category == "High Performing - Top Rank":
                                                st.success(f"✅ **{category}**: {count}")
                                            elif category == "High Performing - Improve Impression Share":
                                                st.warning(f"🚀 **{category}**: {count}")
                                            elif category == "Promising - Scale Impression Share":
                                                st.info(f"📈 **{category}**: {count}")
                                            elif category == "Gray Zone - Expensive to Scale":
                                                st.warning(f"⚠️ **{category}**: {count}")
                                            elif category == "Below Baseline - Needs Optimization":
                                                st.error(f"❌ **{category}**: {count}")
                                            else:
                                                st.metric(category, count)
                                
                                # Filter options
                                st.subheader("🔍 Filter Options")
                                filter_col1, filter_col2 = st.columns(2)
                                
                                with filter_col1:
                                    if 'Category' in impression_share_df.columns:
                                        categories = ['All'] + list(impression_share_df['Category'].unique())
                                        selected_category = st.selectbox("Filter by Category:", categories)
                                    else:
                                        selected_category = 'All'
                                
                                with filter_col2:
                                    min_orders = st.number_input("Minimum Orders:", min_value=3, value=3, max_value=100)
                                
                                # Apply filters
                                filtered_df = impression_share_df.copy()
                                if selected_category != 'All' and 'Category' in filtered_df.columns:
                                    filtered_df = filtered_df[filtered_df['Category'] == selected_category]
                                filtered_df = filtered_df[filtered_df['Orders'] >= min_orders]
                                
                                st.subheader(f"📋 Search Terms Analysis ({len(filtered_df)} terms)")
                                st.caption("**Filtering criteria:** ≥3 orders, sorted by orders (highest → lowest)")
                                
                                # Display the analysis table
                                st.dataframe(
                                    filtered_df,
                                    use_container_width=True,
                                    hide_index=True,
                                    column_config={
                                        "Impression Rank": st.column_config.NumberColumn(
                                            "Impression Rank",
                                            help="Lower rank = better impression share position",
                                            format="%.1f"
                                        ),
                                        "Impression Share %": st.column_config.NumberColumn(
                                            "Impression Share %",
                                            help="Percentage of available impressions captured",
                                            format="%.2f%%"
                                        ),
                                        "ACR %": st.column_config.NumberColumn(
                                            "ACR %",
                                            help="Ad Conversion Rate",
                                            format="%.2f%%"
                                        ),
                                        "ACR vs Baseline": st.column_config.NumberColumn(
                                            "ACR vs Baseline",
                                            help="Difference from business report baseline",
                                            format="%.2f"
                                        ),
                                        "Category": st.column_config.TextColumn(
                                            "Performance Category",
                                            help="Categorization based on ACR vs baseline performance"
                                        ),
                                        "Recommendations": st.column_config.TextColumn(
                                            "Action Recommendations",
                                            help="Suggested actions based on performance analysis"
                                        ),
                                        "top_3_click_share": st.column_config.NumberColumn(
                                            "Top 3 Click Share",
                                            help="Combined click share of top 3 clicked products for this search term",
                                            format="%.2f"
                                        ),
                                        "top_3_conversion_share": st.column_config.NumberColumn(
                                            "Top 3 Conversion Share", 
                                            help="Combined conversion share of top 3 clicked products for this search term",
                                            format="%.2f"
                                        ),
                                        "Remaining Click Share": st.column_config.NumberColumn(
                                            "Remaining Click Share",
                                            help="Market share not captured by top 3 products (100 - top_3_click_share)",
                                            format="%.2f"
                                        ),
                                        "Remaining Conversion Share": st.column_config.NumberColumn(
                                            "Remaining Conversion Share",
                                            help="Conversion share not captured by top 3 products (100 - top_3_conversion_share)",
                                            format="%.2f"
                                        ),
                                        "Click Competitive Intensity": st.column_config.NumberColumn(
                                            "Click Competitive Intensity",
                                            help="Minimum # of competing products winning clicks (Remaining / 3rd product share)",
                                            format="%.2f"
                                        ),
                                        "Conversion Competitive Intensity": st.column_config.NumberColumn(
                                            "Conversion Competitive Intensity",
                                            help="Minimum # of competing products winning conversions (Remaining / 3rd product share)",
                                            format="%.2f"
                                        ),
                                        "Vol Trend": st.column_config.TextColumn(
                                            "Volume Trend",
                                            help="Search volume trend: direction and % change over time (from Tab 1 data)"
                                        ),
                                        "Imp Share Trend": st.column_config.TextColumn(
                                            "Imp Share Trend",
                                            help="Impression share trend: direction and % change over time (from Tab 1 data)"
                                        ),
                                        "Click Share Trend": st.column_config.TextColumn(
                                            "Click Share Trend",
                                            help="Click share trend: direction and % change over time (from Tab 1 data)"
                                        ),
                                        "Purch Share Trend": st.column_config.TextColumn(
                                            "Purchase Share Trend",
                                            help="Purchase share trend: direction and % change over time (from Tab 1 data)"
                                        ),
                                        "Data_Points": st.column_config.NumberColumn(
                                            "Data Points",
                                            help="Number of data points available for trend analysis"
                                        )
                                    }
                                )
                                
                                # Action Items Section
                                st.subheader("💡 Key Action Items")
                                
                                # Helper function to extract match type information
                                def get_match_type_info(row):
                                    """Extract SP and SB match type information from a row"""
                                    sp_targeted = []
                                    sp_non_targeted = []
                                    sb_targeted = []
                                    sb_non_targeted = []
                                    
                                    # SP Match Types
                                    if 'SP_EXACT_Match' in row:
                                        if row['SP_EXACT_Match'] == 'Targeted':
                                            sp_targeted.append('EXACT')
                                        else:
                                            sp_non_targeted.append('EXACT')
                                    
                                    if 'SP_PHRASE_Match' in row:
                                        if row['SP_PHRASE_Match'] == 'Targeted':
                                            sp_targeted.append('PHRASE')
                                        else:
                                            sp_non_targeted.append('PHRASE')
                                    
                                    if 'SP_BROAD_Match' in row:
                                        if row['SP_BROAD_Match'] == 'Targeted':
                                            sp_targeted.append('BROAD')
                                        else:
                                            sp_non_targeted.append('BROAD')
                                    
                                    # SB Match Types
                                    if 'SB_EXACT_Match' in row:
                                        if row['SB_EXACT_Match'] == 'Targeted':
                                            sb_targeted.append('EXACT')
                                        else:
                                            sb_non_targeted.append('EXACT')
                                    
                                    if 'SB_PHRASE_Match' in row:
                                        if row['SB_PHRASE_Match'] == 'Targeted':
                                            sb_targeted.append('PHRASE')
                                        else:
                                            sb_non_targeted.append('PHRASE')
                                    
                                    if 'SB_BROAD_Match' in row:
                                        if row['SB_BROAD_Match'] == 'Targeted':
                                            sb_targeted.append('BROAD')
                                        else:
                                            sb_non_targeted.append('BROAD')
                                    
                                    # Build match info string
                                    match_info = ""
                                    
                                    # SP Info
                                    if sp_targeted or sp_non_targeted:
                                        if sp_targeted:
                                            match_info += f" | SP Targeted: {', '.join(sp_targeted)}"
                                        if sp_non_targeted:
                                            match_info += f" | ⚠️ SP Not Targeted: {', '.join(sp_non_targeted)}"
                                    
                                    # SB Info
                                    if sb_targeted or sb_non_targeted:
                                        if sb_targeted:
                                            match_info += f" | SB Targeted: {', '.join(sb_targeted)}"
                                        if sb_non_targeted:
                                            match_info += f" | ⚠️ SB Not Targeted: {', '.join(sb_non_targeted)}"
                                    
                                    if not match_info:
                                        match_info = " | Match Types: Not detected"
                                    
                                    return match_info
                                
                                # Helper function to get competitive intensity insights
                                def get_competitive_intensity_info(row):
                                    """Analyze competitive intensity and provide insights"""
                                    if 'Click Competitive Intensity' not in row or 'Conversion Competitive Intensity' not in row:
                                        return ""
                                    
                                    click_intensity = row.get('Click Competitive Intensity', 0)
                                    conversion_intensity = row.get('Conversion Competitive Intensity', 0)
                                    
                                    # Skip if no data
                                    if pd.isna(click_intensity) or pd.isna(conversion_intensity):
                                        return ""
                                    
                                    intensity_info = ""
                                    
                                    # High competitive intensity (>5) = many competitors, good opportunity
                                    # Low competitive intensity (<3) = concentrated market, harder to break in
                                    
                                    # Click Intensity Analysis
                                    if click_intensity > 5:
                                        intensity_info += f" | 🎯 Click Market: Fragmented ({click_intensity:.1f}) - High opportunity to gain clicks"
                                    elif click_intensity >= 3:
                                        intensity_info += f" | 📊 Click Market: Moderate competition ({click_intensity:.1f}) - Room for growth"
                                    elif click_intensity > 0:
                                        intensity_info += f" | ⚠️ Click Market: Concentrated ({click_intensity:.1f}) - Top 3 dominate, hard to scale"
                                    
                                    # Conversion Intensity Analysis
                                    if conversion_intensity > 5:
                                        intensity_info += f" | 💰 Conversion Market: Fragmented ({conversion_intensity:.1f}) - High opportunity to capture conversions"
                                    elif conversion_intensity >= 3:
                                        intensity_info += f" | 📈 Conversion Market: Moderate competition ({conversion_intensity:.1f}) - Potential for growth"
                                    elif conversion_intensity > 0:
                                        intensity_info += f" | ⚠️ Conversion Market: Concentrated ({conversion_intensity:.1f}) - Top 3 dominate, challenging"
                                    
                                    return intensity_info
                                
                                # Helper function to get trend information
                                def get_trend_info(row):
                                    """Extract and format trend information from Tab 1 data"""
                                    trend_info = ""
                                    
                                    try:
                                        vol_trend = row.get('Vol Trend', None)
                                        imp_trend = row.get('Imp Share Trend', None)
                                        click_trend = row.get('Click Share Trend', None)
                                        purch_trend = row.get('Purch Share Trend', None)
                                        data_points = row.get('Data_Points', 0)
                                        
                                        # Only show if we have valid data (not None and not 'N/A')
                                        if vol_trend and vol_trend != 'N/A' and data_points > 0:
                                            trend_info = f" | 📊 Trends: Vol {vol_trend}, Imp {imp_trend}, Click {click_trend}, Purch {purch_trend}"
                                    except:
                                        pass
                                    
                                    return trend_info
                                
                                # Initialize session state for API key if not exists
                                if 'openai_api_key' not in st.session_state:
                                    st.session_state.openai_api_key = ""
                                
                                # API Key input section - appears once at the top
                                st.markdown("#### ⚙️ OpenAI API Configuration (Optional)")
                                st.caption("Enter your API key to enable AI-powered analysis for each search term")
                                
                                col_api1, col_api2 = st.columns([3, 1])
                                
                                with col_api1:
                                    api_key_input = st.text_input(
                                        "Enter your OpenAI API Key",
                                        type="password",
                                        value=st.session_state.openai_api_key,
                                        help="Your API key will be stored only for this session. It will be cleared when you close the browser tab.",
                                        key="openai_api_key_input_top_terms",
                                        placeholder="sk-..."
                                    )
                                    
                                    if api_key_input:
                                        st.session_state.openai_api_key = api_key_input
                                
                                with col_api2:
                                    if st.session_state.openai_api_key:
                                        st.write("")
                                        st.write("")
                                        if st.button("🗑️ Clear Key", key="clear_api_key_top_terms"):
                                            st.session_state.openai_api_key = ""
                                            st.rerun()
                                
                                st.caption("Don't have an API key? Get one at [OpenAI Platform](https://platform.openai.com/api-keys)")
                                st.markdown("---")
                                
                                openai_api_key = st.session_state.openai_api_key
                                
                                # Helper function to analyze a single term with GPT
                                def analyze_term_with_gpt(term, row_data, sqp_data, openai_api_key, baseline):
                                    """Analyze a single search term with GPT and display results"""
                                    if not openai_api_key or sqp_data is None or len(sqp_data) == 0:
                                        return False
                                    
                                    # Check if term has time series data
                                    if term not in sqp_data['Search Query'].values:
                                        return False
                                    
                                    term_count = len(sqp_data[sqp_data['Search Query'] == term])
                                    if term_count < 2:
                                        return False
                                    
                                    # Check for required columns
                                    required_sqp_cols = ['Search Query Volume', 'Impressions: ASIN Share %', 'Clicks: ASIN Share %']
                                    if not all(col in sqp_data.columns for col in required_sqp_cols):
                                        return False
                                    
                                    try:
                                        import openai
                                        
                                        # Extract time series data
                                        term_data = sqp_data[sqp_data['Search Query'] == term].copy()
                                        term_data = term_data.sort_values('Date')
                                        
                                        time_series = {
                                            'search_term': term,
                                            'dates': term_data['Date'].dt.strftime('%Y-%m').tolist(),
                                            'volume': term_data['Search Query Volume'].tolist(),
                                            'impression_share': term_data['Impressions: ASIN Share %'].tolist(),
                                            'click_share': term_data['Clicks: ASIN Share %'].tolist()
                                        }
                                        
                                        # Create month-over-month changes table
                                        mom_data = []
                                        for i in range(1, len(time_series['dates'])):
                                            prev_idx = i - 1
                                            curr_idx = i
                                            
                                            vol_change = time_series['volume'][curr_idx] - time_series['volume'][prev_idx]
                                            vol_change_pct = (vol_change / time_series['volume'][prev_idx] * 100) if time_series['volume'][prev_idx] != 0 else 0
                                            
                                            imp_change = time_series['impression_share'][curr_idx] - time_series['impression_share'][prev_idx]
                                            imp_change_pct = (imp_change / time_series['impression_share'][prev_idx] * 100) if time_series['impression_share'][prev_idx] != 0 else 0
                                            
                                            click_change = time_series['click_share'][curr_idx] - time_series['click_share'][prev_idx]
                                            click_change_pct = (click_change / time_series['click_share'][prev_idx] * 100) if time_series['click_share'][prev_idx] != 0 else 0
                                            
                                            mom_data.append({
                                                'Period': f"{time_series['dates'][prev_idx]} → {time_series['dates'][curr_idx]}",
                                                'Volume': f"{time_series['volume'][prev_idx]:,} → {time_series['volume'][curr_idx]:,}",
                                                'Vol Change': f"{vol_change:+,.0f} ({vol_change_pct:+.1f}%)",
                                                'Imp Share %': f"{time_series['impression_share'][prev_idx]:.2f}% → {time_series['impression_share'][curr_idx]:.2f}%",
                                                'Imp Change': f"{imp_change:+.2f}pp ({imp_change_pct:+.1f}%)",
                                                'Click Share %': f"{time_series['click_share'][prev_idx]:.2f}% → {time_series['click_share'][curr_idx]:.2f}%",
                                                'Click Change': f"{click_change:+.2f}pp ({click_change_pct:+.1f}%)"
                                            })
                                        
                                        # Display month-over-month table
                                        with st.expander("📊 Month-over-Month Changes", expanded=True):
                                            mom_df = pd.DataFrame(mom_data)
                                            st.dataframe(mom_df, use_container_width=True, hide_index=True)
                                        
                                        # Calculate metrics for GPT analysis
                                        vol_start = time_series['volume'][0]
                                        vol_end = time_series['volume'][-1]
                                        vol_total_change = vol_end - vol_start
                                        vol_total_change_pct = (vol_total_change / vol_start * 100) if vol_start != 0 else 0
                                        
                                        imp_start = time_series['impression_share'][0]
                                        imp_end = time_series['impression_share'][-1]
                                        imp_total_change = imp_end - imp_start
                                        imp_total_change_pct = (imp_total_change / imp_start * 100) if imp_start != 0 else 0
                                        
                                        click_start = time_series['click_share'][0]
                                        click_end = time_series['click_share'][-1]
                                        click_total_change = click_end - click_start
                                        click_total_change_pct = (click_total_change / click_start * 100) if click_start != 0 else 0
                                        
                                        # Build summary for GPT
                                        summary = f"""
Search Term: {time_series['search_term']}
Time Period: {time_series['dates'][0]} to {time_series['dates'][-1]} ({len(time_series['dates'])} months)

Overall Changes:
- Volume: {vol_start:,} → {vol_end:,} ({vol_total_change:+,} units, {vol_total_change_pct:+.1f}%)
- Impression Share: {imp_start:.2f}% → {imp_end:.2f}% ({imp_total_change:+.2f}pp, {imp_total_change_pct:+.1f}%)
- Click Share: {click_start:.2f}% → {click_end:.2f}% ({click_total_change:+.2f}pp, {click_total_change_pct:+.1f}%)

Month-by-Month Data:
{chr(10).join([f"{time_series['dates'][i]}: Vol={time_series['volume'][i]:,}, Imp={time_series['impression_share'][i]:.2f}%, Click={time_series['click_share'][i]:.2f}%" for i in range(len(time_series['dates']))])}
"""
                                        
                                        prompt = f"""You are an Amazon PPC expert. Analyze this search term's performance data and provide a concise analysis.

{summary}

Provide a brief, actionable analysis (max 350 words):

1. **Trend Story** (3-4 sentences): Describe the journey of the data. Example: "Volume started at 1,000 in January, stayed flat through April around 1,050, then jumped 13% (+150 units) in May to 1,187, remaining stable afterward."

2. **Key Highlights**: List all the changes which are different from the usual patterns which are more than 10%:
   - Format: "May: Volume +150 units (+13%) - likely due to increased market demand"
   - Include actual change + percentage + basic reason (e.g., "market shift", "increased competition", "reduced visibility")
   - DO NOT mention seasonality or seasonal patterns

3. **Stability & Performance** (3-4 sentences per metric):
   - **Volume**: Overall trend + stability (stable/volatile/erratic) + what it indicates
   - **Impression Share**: Trend + market position strength + consistency
   - **Click Share**: Trend + performance quality + whether brand is maintaining dominance

4. **Performance Assessment & Cautionary Guidance** (4-5 sentences):
   - Is the brand maintaining its dominant position? Any warning signs?
   - Identify any concerning patterns (e.g., declining shares, increasing volatility, loss of market position)
   - Provide 2-3 generic, broadly applicable recommendations with caution (e.g., "Consider monitoring competitor activity closely", "May need to review bid strategy if decline continues")
   - Use cautious language: "might consider", "could indicate need for", "worth monitoring"
   - Focus on high-level, safe recommendations - avoid specific tactical suggestions

Keep it compact and analytical. Flag warning signs clearly."""
                                        
                                        # Call OpenAI API
                                        openai.api_key = openai_api_key
                                        response = openai.chat.completions.create(
                                            model="gpt-4o-mini",
                                            messages=[
                                                {"role": "system", "content": "You are an expert Amazon PPC analyst. Be concise and actionable. Avoid verbose explanations."},
                                                {"role": "user", "content": prompt}
                                            ],
                                            temperature=0.7,
                                            max_tokens=900
                                        )
                                        
                                        # Display the analysis
                                        st.markdown("**🤖 AI-Powered Performance Analysis:**")
                                        st.markdown(response.choices[0].message.content)
                                        return True
                                        
                                    except Exception as e:
                                        if "invalid_api_key" in str(e).lower() or "incorrect api key" in str(e).lower():
                                            st.error("❌ Invalid API key. Please check your OpenAI API key and try again.")
                                        elif "insufficient_quota" in str(e).lower():
                                            st.error("❌ Insufficient quota. Please check your OpenAI account balance.")
                                        else:
                                            st.error(f"❌ Error analyzing with GPT: {str(e)}")
                                        return False
                                
                                # High performing top rank terms - Already maximized
                                if 'Category' in impression_share_df.columns:
                                    top_rank_terms = impression_share_df[
                                        impression_share_df['Category'] == "High Performing - Top Rank"
                                    ]
                                    
                                    if not top_rank_terms.empty:
                                        st.success(f"✅ **Top Performing Terms** ({len(top_rank_terms)} terms - Already maximized performance)")
                                        
                                        for idx, (_, row) in enumerate(top_rank_terms.iterrows(), 1):
                                            st.markdown(f"### 📊 Search Term {idx}: **{row['Search Term']}**")
                                            
                                            # Format ACR to 2 decimal places
                                            acr_value = str(row['ACR %']).replace('%', '')
                                            try:
                                                acr_formatted = f"{float(acr_value):.2f}%"
                                            except:
                                                acr_formatted = row['ACR %']
                                            
                                            match_info = get_match_type_info(row)
                                            intensity_info = get_competitive_intensity_info(row)
                                            trend_info = get_trend_info(row)
                                            
                                            # Display key metrics
                                            col1, col2, col3, col4 = st.columns(4)
                                            with col1:
                                                st.metric("Impression Rank", f"{row['Impression Rank']:.1f}")
                                            with col2:
                                                st.metric("Impression Share", f"{row['Impression Share %']:.2f}%")
                                            with col3:
                                                st.metric("ACR", acr_formatted)
                                            with col4:
                                                st.metric("vs Baseline", f"{baseline:.2f}%")
                                            
                                            # Display match type and other info
                                            st.caption(f"**Details:** Performance maximized{match_info}{intensity_info}{trend_info}")
                                            
                                            # GPT Analysis for this specific term
                                            if openai_api_key and sqp_data is not None:
                                                with st.spinner(f"🤖 Analyzing '{row['Search Term']}' with GPT..."):
                                                    analyzed = analyze_term_with_gpt(row['Search Term'], row, sqp_data, openai_api_key, baseline)
                                                    if not analyzed:
                                                        st.info("💡 No time series data available for GPT analysis")
                                            elif not openai_api_key:
                                                st.info("💡 Enter OpenAI API key above to enable AI-powered analysis")
                                            
                                            # Add separator between terms
                                            if idx < len(top_rank_terms.head(5)):
                                                st.markdown("---")
                                        
                                        st.markdown("---")
                                
                                # High opportunity terms
                                high_opportunity = impression_share_df[
                                    impression_share_df['Category'] == "High Performing - Improve Impression Share"
                                ]
                                
                                if not high_opportunity.empty:
                                    st.success(f"🚀 **High Opportunity Terms** ({len(high_opportunity)} terms - High ACR but poor impression rank)")
                                    
                                    for idx, (_, row) in enumerate(high_opportunity.iterrows(), 1):
                                        st.markdown(f"### 📊 Search Term {idx}: **{row['Search Term']}**")
                                        
                                        # Format ACR to 2 decimal places
                                        acr_value = str(row['ACR %']).replace('%', '')
                                        try:
                                            acr_formatted = f"{float(acr_value):.2f}%"
                                        except:
                                            acr_formatted = row['ACR %']
                                        
                                        match_info = get_match_type_info(row)
                                        intensity_info = get_competitive_intensity_info(row)
                                        trend_info = get_trend_info(row)
                                        
                                        # Display key metrics
                                        col1, col2, col3, col4 = st.columns(4)
                                        with col1:
                                            st.metric("Impression Rank", f"{row['Impression Rank']:.1f}")
                                        with col2:
                                            st.metric("Impression Share", f"{row['Impression Share %']:.2f}%")
                                        with col3:
                                            st.metric("ACR", acr_formatted)
                                        with col4:
                                            st.metric("Orders", f"{row['Orders']:.0f}")
                                        
                                        # Display match type and recommendations
                                        st.caption(f"**Action:** {row['Recommendations']}")
                                        st.caption(f"**Details:**{match_info}{intensity_info}{trend_info}")
                                        
                                        # GPT Analysis for this specific term
                                        if openai_api_key and sqp_data is not None:
                                            with st.spinner(f"🤖 Analyzing '{row['Search Term']}' with GPT..."):
                                                analyzed = analyze_term_with_gpt(row['Search Term'], row, sqp_data, openai_api_key, baseline)
                                                if not analyzed:
                                                    st.info("💡 No time series data available for GPT analysis")
                                        elif not openai_api_key:
                                            st.info("💡 Enter OpenAI API key above to enable AI-powered analysis")
                                        
                                        # Add separator between terms
                                        if idx < len(high_opportunity.head(5)):
                                            st.markdown("---")
                                    
                                    st.markdown("---")
                                
                                # Promising terms to scale
                                promising_terms = impression_share_df[
                                    impression_share_df['Category'] == "Promising - Scale Impression Share"
                                ]
                                
                                if not promising_terms.empty:
                                    st.info(f"📈 **Promising Terms to Scale** ({len(promising_terms)} terms - 75%+ of baseline performance)")
                                    
                                    for idx, (_, row) in enumerate(promising_terms.iterrows(), 1):
                                        st.markdown(f"### 📊 Search Term {idx}: **{row['Search Term']}**")
                                        
                                        # Format ACR to 2 decimal places
                                        acr_value = str(row['ACR %']).replace('%', '')
                                        try:
                                            acr_formatted = f"{float(acr_value):.2f}%"
                                        except:
                                            acr_formatted = row['ACR %']
                                        
                                        match_info = get_match_type_info(row)
                                        intensity_info = get_competitive_intensity_info(row)
                                        trend_info = get_trend_info(row)
                                        
                                        # Display key metrics
                                        col1, col2, col3, col4 = st.columns(4)
                                        with col1:
                                            st.metric("Impression Rank", f"{row['Impression Rank']:.1f}")
                                        with col2:
                                            st.metric("Impression Share", f"{row['Impression Share %']:.2f}%")
                                        with col3:
                                            st.metric("ACR", acr_formatted)
                                        with col4:
                                            st.metric("vs Baseline", f"{baseline:.2f}%")
                                        
                                        # Display match type and recommendations
                                        st.caption(f"**Action:** {row['Recommendations']}")
                                        st.caption(f"**Details:**{match_info}{intensity_info}{trend_info}")
                                        
                                        # GPT Analysis for this specific term
                                        if openai_api_key and sqp_data is not None:
                                            with st.spinner(f"🤖 Analyzing '{row['Search Term']}' with GPT..."):
                                                analyzed = analyze_term_with_gpt(row['Search Term'], row, sqp_data, openai_api_key, baseline)
                                                if not analyzed:
                                                    st.info("💡 No time series data available for GPT analysis")
                                        elif not openai_api_key:
                                            st.info("💡 Enter OpenAI API key above to enable AI-powered analysis")
                                        
                                        # Add separator between terms
                                        if idx < len(promising_terms.head(5)):
                                            st.markdown("---")
                                    
                                    st.markdown("---")
                                
                                # Gray Zone terms - Scenario B
                                gray_zone_terms = impression_share_df[
                                    impression_share_df['Category'] == "Gray Zone - Expensive to Scale"
                                ]
                                
                                if not gray_zone_terms.empty:
                                    st.warning(f"⚠️ **Gray Zone Terms** ({len(gray_zone_terms)} terms - 25%+ below baseline - expensive to scale)")
                                    
                                    for idx, (_, row) in enumerate(gray_zone_terms.iterrows(), 1):
                                        st.markdown(f"### 📊 Search Term {idx}: **{row['Search Term']}**")
                                        
                                        # Format ACR to 2 decimal places
                                        acr_value = str(row['ACR %']).replace('%', '')
                                        try:
                                            acr_formatted = f"{float(acr_value):.2f}%"
                                        except:
                                            acr_formatted = row['ACR %']
                                        
                                        match_info = get_match_type_info(row)
                                        intensity_info = get_competitive_intensity_info(row)
                                        trend_info = get_trend_info(row)
                                        
                                        # Display key metrics
                                        col1, col2, col3, col4 = st.columns(4)
                                        with col1:
                                            st.metric("Impression Rank", f"{row['Impression Rank']:.1f}")
                                        with col2:
                                            st.metric("Impression Share", f"{row['Impression Share %']:.2f}%")
                                        with col3:
                                            st.metric("ACR", acr_formatted)
                                        with col4:
                                            st.metric("vs Baseline", f"{baseline:.2f}%")
                                        
                                        # Display match type and recommendations
                                        st.caption(f"**Action:** {row['Recommendations']}")
                                        st.caption(f"**Details:**{match_info}{intensity_info}{trend_info}")
                                        
                                        # GPT Analysis for this specific term
                                        if openai_api_key and sqp_data is not None:
                                            with st.spinner(f"🤖 Analyzing '{row['Search Term']}' with GPT..."):
                                                analyzed = analyze_term_with_gpt(row['Search Term'], row, sqp_data, openai_api_key, baseline)
                                                if not analyzed:
                                                    st.info("💡 No time series data available for GPT analysis")
                                        elif not openai_api_key:
                                            st.info("💡 Enter OpenAI API key above to enable AI-powered analysis")
                                        
                                        # Add separator between terms
                                        if idx < len(gray_zone_terms.head(5)):
                                            st.markdown("---")
                                    
                                    st.markdown("---")
                                
                                # New Section: Low Order Search Queries Analysis (< 3 orders)
                                st.markdown("---")
                                st.subheader("🔍 Low Order Search Queries Analysis (< 3 Orders)")
                                st.caption("Analysis of search queries with potential but limited current ad performance")
                                
                                # Use all_queries_df which contains ALL queries (not filtered by ≥3 orders)
                                if all_queries_df is not None and len(all_queries_df) > 0:
                                    # Filter for queries with < 3 orders
                                    low_order_queries = all_queries_df[all_queries_df['Orders'] < 3].copy()
                                else:
                                    low_order_queries = pd.DataFrame()  # Empty dataframe
                                
                                if len(low_order_queries) > 0:
                                    # Check if we have SQP data with search volume
                                    if sqp_data is not None and 'Search Query Volume' in sqp_data.columns:
                                        # Get search volume for each query
                                        volume_data = sqp_data.groupby('Search Query')['Search Query Volume'].mean().reset_index()
                                        volume_data.columns = ['Search Term', 'Avg Search Volume']
                                        
                                        # Merge with low order queries
                                        low_order_queries = pd.merge(
                                            low_order_queries,
                                            volume_data,
                                            on='Search Term',
                                            how='left'
                                        )
                                        low_order_queries['Avg Search Volume'] = low_order_queries['Avg Search Volume'].fillna(0)
                                        
                                        # Split into high-volume (≥1000) and low-volume (<1000)
                                        high_vol_low_orders = low_order_queries[low_order_queries['Avg Search Volume'] >= 1000].copy()
                                        low_vol_low_orders = low_order_queries[low_order_queries['Avg Search Volume'] < 1000].copy()
                                        
                                        # High Volume Queries (≥1,000 searches/month)
                                        if len(high_vol_low_orders) > 0:
                                            st.success(f"🎯 **High-Volume, Low-Order Queries** ({len(high_vol_low_orders)} terms - ≥1,000 searches/month)")
                                            st.markdown("*These queries have high search volume but low ad sales. High organic share suggests ad potential.*")
                                            
                                            # Sort by search volume descending
                                            high_vol_low_orders = high_vol_low_orders.sort_values('Avg Search Volume', ascending=False)
                                            
                                            for _, row in high_vol_low_orders.head(10).iterrows():
                                                match_info = get_match_type_info(row)
                                                trend_info = get_trend_info(row)
                                                intensity_info = get_competitive_intensity_info(row)
                                                
                                                # Get organic share info if available
                                                organic_info = ""
                                                if 'top_3_click_share' in row and not pd.isna(row['top_3_click_share']):
                                                    organic_info = f" | 🌱 Top 3 Click Share: {row['top_3_click_share']:.1f}%"
                                                if 'top_3_conversion_share' in row and not pd.isna(row['top_3_conversion_share']):
                                                    organic_info += f", Conv Share: {row['top_3_conversion_share']:.1f}%"
                                                
                                                st.write(f"• **{row['Search Term']}** - Vol: {row['Avg Search Volume']:.0f}/mo, Orders: {row['Orders']:.0f}, Imp Share: {row['Impression Share %']:.2f}%{match_info}{organic_info}{intensity_info}{trend_info}")
                                                st.caption(f"   💡 **Opportunity**: High search volume with low ad sales - consider increasing bids or adding to campaigns")
                                            
                                            if len(high_vol_low_orders) > 10:
                                                with st.expander(f"📋 View all {len(high_vol_low_orders)} high-volume, low-order queries"):
                                                    st.dataframe(
                                                        high_vol_low_orders[['Search Term', 'Avg Search Volume', 'Orders', 'Impressions', 
                                                                            'Clicks', 'Impression Share %', 'SP_EXACT_Match', 'SP_PHRASE_Match', 
                                                                            'SP_BROAD_Match', 'SB_EXACT_Match', 'SB_PHRASE_Match', 'SB_BROAD_Match']],
                                                        use_container_width=True,
                                                        hide_index=True
                                                    )
                                        
                                        # Low Volume Queries (<1,000 searches/month)
                                        if len(low_vol_low_orders) > 0:
                                            st.info(f"📌 **Low-Volume, Low-Order Queries** ({len(low_vol_low_orders)} terms - <1,000 searches/month)")
                                            st.markdown("*These queries have low search volume and low ad sales. Manual review recommended.*")
                                            
                                            # Group by ad targeting to show where they're advertised
                                            with st.expander(f"📋 View {len(low_vol_low_orders)} low-volume queries by ad coverage"):
                                                # Sort by search volume descending
                                                low_vol_low_orders = low_vol_low_orders.sort_values('Avg Search Volume', ascending=False)
                                                
                                                # Show which campaigns/match types are targeting these
                                                display_cols = ['Search Term', 'Avg Search Volume', 'Orders', 'Impressions', 'Clicks']
                                                
                                                # Add match type columns if they exist
                                                for col in ['SP_EXACT_Match', 'SP_PHRASE_Match', 'SP_BROAD_Match', 
                                                          'SB_EXACT_Match', 'SB_PHRASE_Match', 'SB_BROAD_Match']:
                                                    if col in low_vol_low_orders.columns:
                                                        display_cols.append(col)
                                                
                                                st.dataframe(
                                                    low_vol_low_orders[display_cols],
                                                    use_container_width=True,
                                                    hide_index=True
                                                )
                                                
                                                st.caption("💡 **Recommendation**: Review these terms for relevance. Consider pausing if not relevant or increasing bids if relevant but underperforming.")
                                    
                                    else:
                                        # If no SQP data, just show the low order queries
                                        st.warning(f"⚠️ Found {len(low_order_queries)} queries with < 3 orders")
                                        st.info("💡 Upload Search Query Performance data in Tab 1 to get volume-based analysis and prioritization")
                                        
                                        with st.expander(f"📋 View all {len(low_order_queries)} low-order queries"):
                                            display_cols = ['Search Term', 'Orders', 'Impressions', 'Clicks', 'Impression Share %']
                                            
                                            # Add match type columns if they exist
                                            for col in ['SP_EXACT_Match', 'SP_PHRASE_Match', 'SP_BROAD_Match', 
                                                      'SB_EXACT_Match', 'SB_PHRASE_Match', 'SB_BROAD_Match']:
                                                if col in low_order_queries.columns:
                                                    display_cols.append(col)
                                            
                                            st.dataframe(
                                                low_order_queries[display_cols],
                                                use_container_width=True,
                                                hide_index=True
                                            )
                                else:
                                    st.success("✅ No search queries with < 3 orders found. All queries meet the minimum threshold!")
                                
                                # Download button for impression share analysis
                                csv_impression = filtered_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Impression Share Analysis as CSV",
                                    data=csv_impression,
                                    file_name=f"impression_share_analysis_{selected_asin_imp}.csv",
                                    mime="text/csv",
                                    key="download_impression_analysis"
                                )
                                
                            else:
                                if df_business_report is None:
                                    st.warning("⚠️ Business Report is required for impression share analysis. Please upload a BusinessReport CSV file.")
                                else:
                                    st.warning("No search terms found with ≥3 orders for this ASIN.")
            else:
                st.info("👆 Please upload files using the file uploader above, then click 'Process Detected Files' to begin analysis.")
        else:
            st.info("👆 Please upload your sponsored product files (CSV and Excel) to begin analysis. The system will automatically detect file types based on naming patterns.")

if __name__ == "__main__":
    main()
