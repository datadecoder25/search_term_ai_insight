import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import io
import numpy as np
import os
import glob

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
        return None, None, None, None, None
    
    # Initialize variables for detected files
    excel_file = None
    csv_product_file = None
    csv_brand_file = None
    csv_top_search_term_file = None
    csv_targeting_report_file = None
    
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
    
    # Show detected files
    if detected_files:
        st.success(f"📁 Auto-detected {len(detected_files)} files:")
        for file_info in detected_files:
            st.info(file_info)
    else:
        st.warning("⚠️ No files matching the expected patterns were found")
        st.caption("Expected patterns: SP_Ad_product.xlsx, SP_ST_imp.csv, SB_ST_imp.csv, Top_search_terms.csv, SP_Targeting.xlsx")
    
    return excel_file, csv_product_file, csv_brand_file, csv_top_search_term_file, csv_targeting_report_file

def fill_missing_dates(data: pd.DataFrame, date_column: str, freq: str) -> pd.DataFrame:
    """
    Fills missing dates in a time series DataFrame for each unique product,
    with a frequency of 'ME' (monthly) or 'W' (weekly).
    """
    # Validate the frequency input
    if freq not in ['ME', 'W']:
        raise ValueError("Invalid frequency. Please use 'ME' for monthly or 'W' for weekly.")

    data = data.rename(columns={'Reporting Date': 'Date'})
    # Ensure the 'Date' column is in the correct format
    data['Date'] = pd.to_datetime(data['Date'])

    start_date = data['Date'].min()
    end_date = data['Date'].max()

    # Get a list of all unique products
    products = data['Search Query'].unique()

    # Generate a complete date range based on the specified frequency
    full_date_range = pd.date_range(start=start_date, end=end_date, freq=freq)

    # Create an empty list to store the processed data for each product
    filled_data_list = []

    # Iterate over each product
    for product in products:
        # Filter the original data for the current product
        product_data = data[data['Search Query'] == product].copy()

        # Create a DataFrame with the full date range for this product
        full_product_df = pd.DataFrame({
            'Date': full_date_range,
            'Search Query': product
        })

        # Merge the full date range with the product's data
        merged_df = pd.merge(full_product_df, product_data, on=['Date', 'Search Query'], how='left')

        # Append the merged DataFrame to our list
        filled_data_list.append(merged_df)

    # Concatenate all the individual product DataFrames into one
    filled_df = pd.concat(filled_data_list, ignore_index=True)

    # Sort the final DataFrame by Product and then by Date for clean display
    filled_df.sort_values(by=['Search Query', 'Date'], inplace=True)

    return filled_df

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
def load_sponsored_product_data_from_uploads(excel_file, csv_product_file, csv_brand_file, csv_top_search_term_file, csv_targeting_report_file):
    """
    Load and process sponsored product data from uploaded files
    """
    if not excel_file:
        return None, None, None, None, None
    
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
        
        return df_ad_product, st_imp_product_df, st_imp_brand_df, st_imp_top_search_term_df, df_targeting_report_final
    except Exception as e:
        st.error(f"Error loading files: {str(e)}")
        return None, None, None, None, None

def process_search_term_analysis(df_ad_product, st_imp_df, selected_asin, df_top_search_term_final, df_targeting_report_final):
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

def process_brand_search_term_analysis(st_imp_df, df_top_search_term_final):
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
            
            # Clear cache button
            if st.button("🔄 Clear Cache", type="secondary"):
                st.cache_data.clear()
                st.success("Cache cleared!")
            
            # Load data button
            load_data_tab1 = st.button("📊 Process Uploaded Files", type="primary", disabled=not uploaded_files)
        
        # Load data from uploaded files
        combined_df = None
        
        if uploaded_files and load_data_tab1:
            with st.spinner("Processing uploaded CSV files..."):
                combined_df = load_and_combine_uploaded_csvs(uploaded_files)
        
        if combined_df is None:
            st.info("👆 Please upload CSV files and click 'Process Uploaded Files' to begin analysis.")
        else:
            # Process data
            with st.spinner("Processing data and filling missing dates..."):
                full_df = fill_missing_dates(combined_df, 'Reporting Date', freq).fillna(0)
            
            # Data summary in columns
            st.subheader("📈 Data Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Records", len(full_df))
            with col2:
                st.metric("Unique Search Queries", full_df['Search Query'].nunique())
            with col3:
                st.metric("Date Range Start", full_df['Date'].min().strftime('%Y-%m-%d'))
            with col4:
                st.metric("Date Range End", full_df['Date'].max().strftime('%Y-%m-%d'))
            
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
                
                # Create and display plots
                st.subheader("📊 Interactive Visualizations")
                
                plots = create_interactive_plots(filtered_df)
                
                # Display plots in sub-tabs
                plot_tab_names = [plot[0] for plot in plots]
                plot_tabs = st.tabs(plot_tab_names)
                
                for i, (plot_tab, (plot_name, fig)) in enumerate(zip(plot_tabs, plots)):
                    with plot_tab:
                        st.plotly_chart(fig, use_container_width=True)
    
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
            
            **Examples:**
            - `SP_Ad_product_report.xlsx` ✅
            - `SP_ST_imp_data_Aug.csv` ✅  
            - `SB_ST_imp_brand_data.csv` ✅
            - `Top_search_terms_monthly.csv` ✅
            - `SP_Targeting_report.xlsx` ✅
            """)
        
        # Process uploaded files
        if uploaded_files_tab2:
            st.success(f"📁 Uploaded {len(uploaded_files_tab2)} files")
            
            # Auto-detect and categorize files
            with st.spinner("Auto-detecting file types..."):
                excel_file, csv_product_file, csv_brand_file, csv_top_search_term_file, csv_targeting_report_file = auto_detect_uploaded_files(uploaded_files_tab2)
            
            # Process button
            process_files = st.button("🔍 Process Detected Files", type="primary")
            
            if process_files and (excel_file or csv_product_file or csv_brand_file):
                # Load sponsored product data
                with st.spinner("Loading sponsored product data..."):
                    df_ad_product, st_imp_product_df, st_imp_brand_df, st_imp_top_search_term_df, df_targeting_report_final = load_sponsored_product_data_from_uploads(excel_file, csv_product_file, csv_brand_file, csv_top_search_term_file, csv_targeting_report_file)
                
                if df_ad_product is not None:
                    st.success("✅ Sponsored product data loaded successfully!")
                
                    # Process top search term data if available
                    df_top_search_term_final = None
                    if st_imp_top_search_term_df is not None:
                        try:
                            st_imp_top_search_term_df['top_3_click_share'] = st_imp_top_search_term_df['Top Clicked Product #1: Click Share'] + st_imp_top_search_term_df['Top Clicked Product #2: Click Share'] + st_imp_top_search_term_df['Top Clicked Product #3: Click Share']
                            st_imp_top_search_term_df['top_3_conversion_share'] = st_imp_top_search_term_df['Top Clicked Product #1: Conversion Share'] + st_imp_top_search_term_df['Top Clicked Product #2: Conversion Share'] + st_imp_top_search_term_df['Top Clicked Product #3: Conversion Share']
                            df_top_search_term_final = st_imp_top_search_term_df[['Search Term','top_3_click_share','top_3_conversion_share']]
                            st.success("✅ Top search terms data processed successfully! Extra columns will be added to analysis.")
                        except Exception as e:
                            st.warning(f"⚠️ Could not process top search terms data: {str(e)}")
                            df_top_search_term_final = None
                    
                    # Process targeting report data if available
                    if df_targeting_report_final is not None:
                        st.success("✅ Targeting report data loaded successfully! Match type columns will be added to analysis.")
                    
                    # Display data summary
                    st.subheader("📈 Data Summary")
                    col1, col2, col3, col4, col5, col6 = st.columns(6)
                    
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
                    
                    # Create analysis tabs for Product and Brand
                    analysis_tabs = []
                    if st_imp_product_df is not None:
                        analysis_tabs.append("📊 Product Analysis")
                    if st_imp_brand_df is not None:
                        analysis_tabs.append("🏷️ Brand Analysis")
                    
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
                                product_search_term_df = process_search_term_analysis(df_ad_product, st_imp_product_df, selected_asin, df_top_search_term_final, df_targeting_report_final)
                            
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
                            brand_search_term_df = process_brand_search_term_analysis(st_imp_brand_df, df_top_search_term_final)
                            
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
            else:
                st.info("👆 Please upload files using the file uploader above, then click 'Process Detected Files' to begin analysis.")
        else:
            st.info("👆 Please upload your sponsored product files (CSV and Excel) to begin analysis. The system will automatically detect file types based on naming patterns.")

if __name__ == "__main__":
    main()
