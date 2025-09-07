# Search Query Performance Analytics

A Streamlit web application for analyzing Amazon search query performance data with interactive visualizations.

## Features

- 📁 **File Upload**: Upload multiple CSV files directly through the web interface
- ⚙️ **Frequency Analysis**: Choose between monthly or weekly analysis frequency
- 🎯 **Search Query Filtering**: Select specific search queries for detailed analysis
- 📊 **Interactive Visualizations**: Dynamic Plotly charts with multiple view options
- 📈 **KPI Dashboard**: Key performance metrics with real-time calculations
- 📋 **Data Table Viewer**: Comprehensive data exploration with filtering options
- 💾 **Advanced Export**: Download data in CSV or Excel formats with custom date ranges
- 🏷️ **Streamlined Interface**: Single-page workflow for seamless user experience
- 🔄 **Automatic Processing**: Smart data cleaning and gap filling

## Installation

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run streamlit_app.py
```

2. Open your web browser and navigate to `http://localhost:8501`

3. **Upload & Analyze Data**: In the single-page interface:
   - Upload multiple CSV files using the file uploader
   - Choose analysis frequency (Monthly/Weekly)  
   - Select a search query for analysis
   - View interactive charts and KPIs immediately below
   - Export data as needed

4. The interface flows smoothly from data upload to analysis results on one page

## Data Format

The app expects CSV files with the following structure:
- First row: Metadata (will be skipped)
- Second row: Column headers
- Remaining rows: Data

Required columns:
- `Reporting Date`
- `Search Query`
- `Search Query Score`
- `Search Query Volume`
- `Purchases: Total Count`
- `Impressions: ASIN Share %`
- `Clicks: Click Rate %`
- `Clicks: ASIN Share %`
- `Cart Adds: Cart Add Rate %`
- `Cart Adds: ASIN Share %`
- `Purchases: Purchase Rate %`
- `Purchases: ASIN Share %`

## Visualizations

The app generates four interactive plots:

1. **Purchase & Click Share vs Volume**: Shows brand purchase share, click share, and search query volume over time
2. **Cart Adds vs Click Share**: Compares brand cart add share with click share
3. **Impressions Share vs Volume**: Shows brand impression share alongside search query volume
4. **Purchase vs Cart Adds Share**: Compares brand purchase share with cart add share

## Technical Details

- Built with Streamlit for the web interface
- Uses Plotly for interactive visualizations
- Data processing with pandas
- Automatic date gap filling for time series analysis
- Responsive design with tabs and columns layout
