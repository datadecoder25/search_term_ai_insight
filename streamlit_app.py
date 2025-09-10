import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import io
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Search Query Performance Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_and_combine_data(uploaded_files):
    """
    Load and combine all uploaded CSV files
    """
    if not uploaded_files:
        return None
    
    files = []
    for uploaded_file in uploaded_files:
        # Read the uploaded file
        df_temp = pd.read_csv(uploaded_file, header=1)  # Skip first row, use second as header
        files.append(df_temp)
    
    combined_df = pd.concat(files, ignore_index=True)
    return combined_df

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
def load_sponsored_product_data(excel_file, csv_file):
    """
    Load and process sponsored product data from Excel and CSV files
    """
    if not excel_file or not csv_file:
        return None, None
    
    # Read Excel file - get available sheets first
    try:
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
            
        # Read CSV file
        st_imp_df = pd.read_csv(csv_file)
        
        return df_ad_product, st_imp_df
    except Exception as e:
        st.error(f"Error loading files: {str(e)}")
        return None, None

def process_search_term_analysis(df_ad_product, st_imp_df, selected_asin):
    """
    Process search term analysis for a selected ASIN
    """
    # Get campaigns for the selected ASIN
    campaigns = df_ad_product[df_ad_product['Advertised ASIN'] == selected_asin]['Campaign Name'].unique()
    
    # Filter search term impression data for these campaigns
    filtered_st_imp_df = st_imp_df[st_imp_df['Campaign Name'].isin(campaigns)].copy()
    
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
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Upload CSV Files")
            uploaded_files = st.file_uploader(
                "Choose CSV files",
                type="csv",
                accept_multiple_files=True,
                help="Upload multiple CSV files containing search query performance data",
                key="csv_uploader"
            )
        
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
        
        # Load data from uploaded files
        combined_df = None
        
        if uploaded_files:
            with st.spinner("Processing uploaded files..."):
                combined_df = load_and_combine_data(uploaded_files)
        
        if combined_df is None:
            st.info("👆 Please upload CSV files to begin analysis.")
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
        st.subheader("📁 Upload Sponsored Product Data")
        
        col1, col2 = st.columns(2)
        
        with col1:
            excel_file = st.file_uploader(
                "Upload Ad Product Report (Excel)",
                type=["xlsx", "xls"],
                help="Upload the sponsored product ad report Excel file",
                key="excel_uploader"
            )
        
        with col2:
            csv_file = st.file_uploader(
                "Upload Search Term Impression Share (CSV)",
                type="csv",
                help="Upload the search term impression share CSV file",
                key="st_csv_uploader"
            )
        
        if excel_file and csv_file:
            # Load sponsored product data
            with st.spinner("Loading sponsored product data..."):
                df_ad_product, st_imp_df = load_sponsored_product_data(excel_file, csv_file)
            
            if df_ad_product is not None and st_imp_df is not None:
                st.success("✅ Sponsored product data loaded successfully!")
                
                # Display data summary
                st.subheader("📈 Data Summary")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Ad Records", len(df_ad_product))
                with col2:
                    st.metric("Unique ASINs", df_ad_product['Advertised ASIN'].nunique())
                with col3:
                    st.metric("Search Terms", len(st_imp_df))
                with col4:
                    st.metric("Unique Campaigns", st_imp_df['Campaign Name'].nunique())
                
                # ASIN selection
                st.subheader("🎯 ASIN Selection")
                available_asins = sorted(df_ad_product['Advertised ASIN'].unique())
                selected_asin = st.selectbox(
                    "Select ASIN for Analysis:",
                    options=available_asins,
                    index=available_asins.index('B0BH6G8Q94') if 'B0BH6G8Q94' in available_asins else 0,
                    help="Choose an ASIN to analyze search term performance"
                )
                
                # Process and display search term analysis
                with st.spinner("Processing search term analysis..."):
                    search_term_df = process_search_term_analysis(df_ad_product, st_imp_df, selected_asin)
                
                if search_term_df is not None and len(search_term_df) > 0:
                    st.subheader(f"📊 Search Term Analysis for ASIN: **{selected_asin}**")
                    
                    # Display key metrics for selected search terms
                    three_farmers_data = search_term_df[search_term_df['Search Term'] == 'three farmers']
                    if not three_farmers_data.empty:
                        st.info("**🔍 'Three Farmers' Search Term Metrics:**")
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
                    st.subheader("📋 Complete Search Term Analysis")
                    
                    
                    # Display filtered results
                    st.dataframe(
                        search_term_df,
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
                            )
                        }
                    )
                    
                    # Download button for the processed data
                    csv_data = search_term_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Search Term Analysis",
                        data=csv_data,
                        file_name=f"search_term_analysis_{selected_asin}.csv",
                        mime="text/csv"
                    )
                else:
                    st.warning("No search term data found for the selected ASIN.")
        else:
            st.info("👆 Please upload both Excel and CSV files to begin sponsored product analysis.")

if __name__ == "__main__":
    main()
