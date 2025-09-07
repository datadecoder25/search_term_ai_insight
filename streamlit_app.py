import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import io

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

def main():
    # App title and description
    st.title("📊 Search Query Performance Analytics")
    st.markdown("**Analyze Amazon search query performance data with interactive visualizations**")
    
    # Data Upload & Setup Section
    st.header("📁 Data Upload & Configuration")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("Upload CSV Files")
        uploaded_files = st.file_uploader(
            "Choose CSV files",
            type="csv",
            accept_multiple_files=True,
            help="Upload multiple CSV files containing search query performance data"
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
        st.stop()
    
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
        st.stop()
    
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

if __name__ == "__main__":
    main()
