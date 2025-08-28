import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import time
from io import BytesIO
import base64
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from PIL import Image
import textwrap

# Configure page
st.set_page_config(
    page_title="Financial Statement Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .highlight-positive {
        color: #1f77b4;
        font-weight: bold;
    }
    .highlight-negative {
        color: #d62728;
        font-weight: bold;
    }
    .stDataFrame {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

class DataIngestor:
    """Handles data ingestion from SEC EDGAR API and caching."""
    
    def __init__(self):
        self.base_url = "https://data.sec.gov/api/xbrl/companyfacts/CIK"
        self.headers = {
            'User-Agent': 'Financial Dashboard (educational@example.com)'
        }
    
    @st.cache_data(ttl=3600)  # Cache for 1 hour
    def get_company_facts(_self, ticker: str) -> Optional[Dict]:
        """
        Fetch company facts from SEC EDGAR API.
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary containing company financial facts or None if error
        """
        try:
            # Get CIK from ticker (simplified mapping)
            cik_mapping = {
                'AAPL': '0000320193',
                'MSFT': '0000789019', 
                'GOOGL': '0001652044',
                'AMZN': '0001018724',
                'TSLA': '0001318605',
                'META': '0001326801',
                'NVDA': '0001045810',
                'JPM': '0000019617',
                'JNJ': '0000200406',
                'PG': '0000080424'
            }
            
            cik = cik_mapping.get(ticker.upper())
            if not cik:
                st.warning(f"CIK not found for {ticker}. Using sample data.")
                return _self._generate_sample_data(ticker)
            
            url = f"{_self.base_url}{cik}.json"
            response = requests.get(url, headers=_self.headers, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                st.warning(f"API request failed with status {response.status_code}. Using sample data.")
                return _self._generate_sample_data(ticker)
                
        except Exception as e:
            st.error(f"Error fetching data for {ticker}: {str(e)}")
            return _self._generate_sample_data(ticker)
    
    def _generate_sample_data(self, ticker: str) -> Dict:
        """Generate sample financial data for demonstration."""
        np.random.seed(hash(ticker) % 1000)
        
        # Base values with some company-specific variation
        base_revenue = np.random.uniform(50000, 500000)
        growth_rate = np.random.uniform(0.05, 0.25)
        margin_base = np.random.uniform(0.15, 0.35)
        
        sample_data = {
            "entityName": f"{ticker} Inc.",
            "cik": f"000{np.random.randint(100000, 999999)}",
            "facts": {
                "us-gaap": {
                    "Revenues": {
                        "units": {
                            "USD": [
                                {
                                    "end": "2023-12-31",
                                    "val": int(base_revenue * (1 + growth_rate) ** 2),
                                    "fy": 2023,
                                    "form": "10-K"
                                },
                                {
                                    "end": "2022-12-31", 
                                    "val": int(base_revenue * (1 + growth_rate)),
                                    "fy": 2022,
                                    "form": "10-K"
                                },
                                {
                                    "end": "2021-12-31",
                                    "val": int(base_revenue),
                                    "fy": 2021,
                                    "form": "10-K"
                                }
                            ]
                        }
                    },
                    "CostOfGoodsAndServicesSold": {
                        "units": {
                            "USD": [
                                {
                                    "end": "2023-12-31",
                                    "val": int(base_revenue * (1 + growth_rate) ** 2 * (1 - margin_base)),
                                    "fy": 2023,
                                    "form": "10-K"
                                },
                                {
                                    "end": "2022-12-31",
                                    "val": int(base_revenue * (1 + growth_rate) * (1 - margin_base * 0.95)),
                                    "fy": 2022,
                                    "form": "10-K"
                                },
                                {
                                    "end": "2021-12-31",
                                    "val": int(base_revenue * (1 - margin_base * 0.9)),
                                    "fy": 2021,
                                    "form": "10-K"
                                }
                            ]
                        }
                    },
                    "Assets": {
                        "units": {
                            "USD": [
                                {
                                    "end": "2023-12-31",
                                    "val": int(base_revenue * 2.5),
                                    "fy": 2023,
                                    "form": "10-K"
                                },
                                {
                                    "end": "2022-12-31",
                                    "val": int(base_revenue * 2.3),
                                    "fy": 2022,
                                    "form": "10-K"
                                },
                                {
                                    "end": "2021-12-31",
                                    "val": int(base_revenue * 2.1),
                                    "fy": 2021,
                                    "form": "10-K"
                                }
                            ]
                        }
                    },
                    "AssetsCurrent": {
                        "units": {
                            "USD": [
                                {
                                    "end": "2023-12-31",
                                    "val": int(base_revenue * 0.8),
                                    "fy": 2023,
                                    "form": "10-K"
                                },
                                {
                                    "end": "2022-12-31",
                                    "val": int(base_revenue * 0.75),
                                    "fy": 2022,
                                    "form": "10-K"
                                },
                                {
                                    "end": "2021-12-31",
                                    "val": int(base_revenue * 0.7),
                                    "fy": 2021,
                                    "form": "10-K"
                                }
                            ]
                        }
                    },
                    "LiabilitiesCurrent": {
                        "units": {
                            "USD": [
                                {
                                    "end": "2023-12-31",
                                    "val": int(base_revenue * 0.4),
                                    "fy": 2023,
                                    "form": "10-K"
                                },
                                {
                                    "end": "2022-12-31",
                                    "val": int(base_revenue * 0.38),
                                    "fy": 2022,
                                    "form": "10-K"
                                },
                                {
                                    "end": "2021-12-31",
                                    "val": int(base_revenue * 0.35),
                                    "fy": 2021,
                                    "form": "10-K"
                                }
                            ]
                        }
                    },
                    "StockholdersEquity": {
                        "units": {
                            "USD": [
                                {
                                    "end": "2023-12-31",
                                    "val": int(base_revenue * 1.2),
                                    "fy": 2023,
                                    "form": "10-K"
                                },
                                {
                                    "end": "2022-12-31",
                                    "val": int(base_revenue * 1.1),
                                    "fy": 2022,
                                    "form": "10-K"
                                },
                                {
                                    "end": "2021-12-31",
                                    "val": int(base_revenue * 1.0),
                                    "fy": 2021,
                                    "form": "10-K"
                                }
                            ]
                        }
                    },
                    "NetIncomeLoss": {
                        "units": {
                            "USD": [
                                {
                                    "end": "2023-12-31",
                                    "val": int(base_revenue * margin_base * 0.6),
                                    "fy": 2023,
                                    "form": "10-K"
                                },
                                {
                                    "end": "2022-12-31",
                                    "val": int(base_revenue * margin_base * 0.55),
                                    "fy": 2022,
                                    "form": "10-K"
                                },
                                {
                                    "end": "2021-12-31",
                                    "val": int(base_revenue * margin_base * 0.5),
                                    "fy": 2021,
                                    "form": "10-K"
                                }
                            ]
                        }
                    }
                }
            }
        }
        
        return sample_data

class FinancialParser:
    """Handles parsing and normalization of financial data."""
    
    def __init__(self):
        # Mapping of common GAAP concepts to normalized names
        self.concept_mapping = {
            'revenues': ['Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax', 'SalesRevenueNet'],
            'cost_of_sales': ['CostOfGoodsAndServicesSold', 'CostOfRevenue', 'CostOfSales'],
            'assets': ['Assets'],
            'current_assets': ['AssetsCurrent'],
            'current_liabilities': ['LiabilitiesCurrent'],
            'total_equity': ['StockholdersEquity', 'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest'],
            'net_income': ['NetIncomeLoss', 'ProfitLoss'],
            'cash': ['Cash', 'CashAndCashEquivalentsAtCarryingValue'],
            'total_debt': ['DebtCurrent', 'DebtNoncurrent', 'LongTermDebt']
        }
    
    def parse_company_data(self, company_facts: Dict) -> pd.DataFrame:
        """
        Parse and normalize company financial data.
        
        Args:
            company_facts: Raw company facts from SEC API
            
        Returns:
            Normalized DataFrame with financial metrics
        """
        if not company_facts or 'facts' not in company_facts:
            return pd.DataFrame()
        
        us_gaap = company_facts['facts'].get('us-gaap', {})
        normalized_data = []
        
        # Get company info
        entity_name = company_facts.get('entityName', 'Unknown Company')
        
        # Extract data for each fiscal year
        years = set()
        for concept_data in us_gaap.values():
            if 'units' in concept_data and 'USD' in concept_data['units']:
                for entry in concept_data['units']['USD']:
                    if 'fy' in entry and entry.get('form') in ['10-K']:
                        years.add(entry['fy'])
        
        for year in sorted(years, reverse=True)[:3]:  # Get last 3 years
            year_data = {
                'company': entity_name,
                'year': year,
                'period_end': f"{year}-12-31"
            }
            
            # Extract normalized concepts
            for normalized_name, concept_list in self.concept_mapping.items():
                value = self._get_concept_value(us_gaap, concept_list, year)
                year_data[normalized_name] = value
            
            normalized_data.append(year_data)
        
        return pd.DataFrame(normalized_data)
    
    def _get_concept_value(self, us_gaap: Dict, concept_list: List[str], year: int) -> Optional[float]:
        """Get value for a concept from the first available mapping."""
        for concept in concept_list:
            if concept in us_gaap:
                concept_data = us_gaap[concept]
                if 'units' in concept_data and 'USD' in concept_data['units']:
                    for entry in concept_data['units']['USD']:
                        if entry.get('fy') == year and entry.get('form') == '10-K':
                            return float(entry.get('val', 0))
        return None

class RatioCalculator:
    """Calculates financial ratios from normalized data."""
    
    def calculate_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate financial ratios for the given data.
        
        Args:
            df: DataFrame with normalized financial data
            
        Returns:
            DataFrame with calculated ratios
        """
        if df.empty:
            return df
        
        result_df = df.copy()
        
        # Profitability ratios
        result_df['gross_margin'] = self._safe_divide(
            df['revenues'] - df['cost_of_sales'], df['revenues']
        ) * 100
        
        result_df['net_margin'] = self._safe_divide(
            df['net_income'], df['revenues']
        ) * 100
        
        result_df['roa'] = self._safe_divide(
            df['net_income'], df['assets']
        ) * 100
        
        result_df['roe'] = self._safe_divide(
            df['net_income'], df['total_equity']
        ) * 100
        
        # Liquidity ratios
        result_df['current_ratio'] = self._safe_divide(
            df['current_assets'], df['current_liabilities']
        )
        
        # Calculate year-over-year growth
        if len(result_df) > 1:
            result_df = result_df.sort_values('year')
            result_df['revenue_growth'] = result_df['revenues'].pct_change() * 100
            result_df['net_income_growth'] = result_df['net_income'].pct_change() * 100
        else:
            result_df['revenue_growth'] = None
            result_df['net_income_growth'] = None
        
        return result_df.sort_values('year', ascending=False)
    
    def _safe_divide(self, numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        """Safely divide two series, handling division by zero."""
        return pd.Series([
            n / d if d != 0 and pd.notna(n) and pd.notna(d) else None
            for n, d in zip(numerator, denominator)
        ])

class Visualizer:
    """Handles creation of charts and visualizations."""
    
    def create_ratio_comparison(self, companies_data: Dict[str, pd.DataFrame], ratio: str) -> go.Figure:
        """Create a bar chart comparing ratios across companies."""
        fig = go.Figure()
        
        companies = []
        values = []
        colors = []
        
        for company, df in companies_data.items():
            if not df.empty and ratio in df.columns:
                latest_value = df.iloc[0][ratio]  # Most recent year
                if pd.notna(latest_value):
                    companies.append(company)
                    values.append(latest_value)
                    # Color coding based on performance
                    if latest_value > 0:
                        colors.append('#1f77b4' if latest_value > np.median([v for v in values if v is not None]) else '#ff7f0e')
                    else:
                        colors.append('#d62728')
        
        if companies:
            fig.add_trace(go.Bar(
                x=companies,
                y=values,
                marker_color=colors,
                text=[f"{v:.1f}%" if ratio.endswith(('margin', 'roa', 'roe', 'growth')) else f"{v:.2f}" for v in values],
                textposition='auto'
            ))
            
            fig.update_layout(
                title=f"{ratio.replace('_', ' ').title()} Comparison",
                xaxis_title="Companies",
                yaxis_title=f"{ratio.replace('_', ' ').title()}",
                showlegend=False,
                height=400
            )
        
        return fig
    
    def create_trend_chart(self, df: pd.DataFrame, company: str, metrics: List[str]) -> go.Figure:
        """Create a time-series chart for selected metrics."""
        fig = go.Figure()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
        for i, metric in enumerate(metrics):
            if metric in df.columns:
                fig.add_trace(go.Scatter(
                    x=df['year'],
                    y=df[metric],
                    mode='lines+markers',
                    name=metric.replace('_', ' ').title(),
                    line=dict(color=colors[i % len(colors)], width=3),
                    marker=dict(size=8)
                ))
        
        fig.update_layout(
            title=f"{company} - Financial Trends",
            xaxis_title="Year",
            yaxis_title="Value",
            height=400,
            hovermode='x unified'
        )
        
        return fig
    
    def create_peer_distribution(self, peer_data: Dict[str, pd.DataFrame], ratio: str) -> go.Figure:
        """Create a box plot showing peer group distribution for a ratio."""
        fig = go.Figure()
        
        all_values = []
        company_names = []
        
        for company, df in peer_data.items():
            if not df.empty and ratio in df.columns:
                values = df[ratio].dropna().tolist()
                all_values.extend(values)
                company_names.extend([company] * len(values))
        
        if all_values:
            fig.add_trace(go.Box(
                y=all_values,
                name=ratio.replace('_', ' ').title(),
                boxpoints='all',
                pointpos=0,
                marker_color='lightblue'
            ))
            
            fig.update_layout(
                title=f"Peer Group Distribution - {ratio.replace('_', ' ').title()}",
                yaxis_title=f"{ratio.replace('_', ' ').title()}",
                height=400
            )
        
        return fig

def main():
    """Main Streamlit application."""
    
    # Initialize components
    data_ingestor = DataIngestor()
    parser = FinancialParser()
    calculator = RatioCalculator()
    visualizer = Visualizer()
    
    # Header
    st.title("📊 Financial Statement Analysis Dashboard")
    st.markdown("---")
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Company selection
        st.subheader("Select Companies")
        default_tickers = ['AAPL', 'MSFT', 'GOOGL']
        selected_tickers = st.multiselect(
            "Choose tickers to analyze:",
            ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'JNJ', 'PG'],
            default=default_tickers,
            help="Select up to 10 companies for comparison"
        )
        
        # Analysis options
        st.subheader("Analysis Options")
        show_trends = st.checkbox("Show trend analysis", value=True)
        show_peer_comparison = st.checkbox("Show peer comparison", value=True)
        
        # Ratio selection
        st.subheader("Key Ratios")
        ratio_options = {
            'Profitability': ['gross_margin', 'net_margin', 'roa', 'roe'],
            'Liquidity': ['current_ratio'],
            'Growth': ['revenue_growth', 'net_income_growth']
        }
        
        selected_ratios = []
        for category, ratios in ratio_options.items():
            st.write(f"**{category}**")
            for ratio in ratios:
                if st.checkbox(ratio.replace('_', ' ').title(), value=True, key=f"ratio_{ratio}"):
                    selected_ratios.append(ratio)
    
    if not selected_tickers:
        st.warning("Please select at least one company to analyze.")
        return
    
    # Load and process data
    with st.spinner("Loading financial data..."):
        companies_data = {}
        
        progress_bar = st.progress(0)
        for i, ticker in enumerate(selected_tickers):
            company_facts = data_ingestor.get_company_facts(ticker)
            if company_facts:
                normalized_df = parser.parse_company_data(company_facts)
                if not normalized_df.empty:
                    ratio_df = calculator.calculate_ratios(normalized_df)
                    companies_data[ticker] = ratio_df
            
            progress_bar.progress((i + 1) / len(selected_tickers))
        
        progress_bar.empty()
    
    if not companies_data:
        st.error("No valid financial data found for the selected companies.")
        return
    
    # Main dashboard tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Overview", "🔍 Detailed Analysis", "📊 Peer Comparison", "📋 Data Export"])
    
    with tab1:
        st.header("Financial Overview")
        
        # Key metrics cards
        col1, col2, col3, col4 = st.columns(4)
        
        # Calculate aggregate metrics
        total_companies = len(companies_data)
        avg_margins = []
        avg_growth = []
        
        for ticker, df in companies_data.items():
            if not df.empty:
                if 'net_margin' in df.columns and pd.notna(df.iloc[0]['net_margin']):
                    avg_margins.append(df.iloc[0]['net_margin'])
                if 'revenue_growth' in df.columns and pd.notna(df.iloc[0]['revenue_growth']):
                    avg_growth.append(df.iloc[0]['revenue_growth'])
        
        with col1:
            st.metric(
                "Companies Analyzed",
                total_companies,
                help="Number of companies with valid financial data"
            )
        
        with col2:
            if avg_margins:
                st.metric(
                    "Avg Net Margin",
                    f"{np.mean(avg_margins):.1f}%",
                    help="Average net profit margin across selected companies"
                )
        
        with col3:
            if avg_growth:
                growth_val = np.mean(avg_growth)
                st.metric(
                    "Avg Revenue Growth",
                    f"{growth_val:.1f}%",
                    delta=f"{growth_val:.1f}%",
                    help="Average revenue growth rate"
                )
        
        with col4:
            current_ratios = []
            for ticker, df in companies_data.items():
                if not df.empty and 'current_ratio' in df.columns:
                    if pd.notna(df.iloc[0]['current_ratio']):
                        current_ratios.append(df.iloc[0]['current_ratio'])
            
            if current_ratios:
                st.metric(
                    "Avg Current Ratio",
                    f"{np.mean(current_ratios):.2f}",
                    help="Average current ratio (liquidity measure)"
                )
        
        # Ratio comparison charts
        if selected_ratios:
            st.subheader("Ratio Comparisons")
            
            # Create two columns for charts
            chart_cols = st.columns(2)
            
            for i, ratio in enumerate(selected_ratios[:4]):  # Limit to 4 charts in overview
                col_idx = i % 2
                with chart_cols[col_idx]:
                    fig = visualizer.create_ratio_comparison(companies_data, ratio)
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.header("Detailed Company Analysis")
        
        # Company selector for detailed view
        detail_ticker = st.selectbox(
            "Select company for detailed analysis:",
            list(companies_data.keys()),
            help="Choose a company to view detailed trends and metrics"
        )
        
        if detail_ticker and detail_ticker in companies_data:
            df = companies_data[detail_ticker]
            
            if not df.empty:
                # Company info
                st.subheader(f"{detail_ticker} Financial Analysis")
                
                # Latest metrics table
                st.write("**Latest Financial Metrics**")
                latest_data = df.iloc[0]  # Most recent year
                
                metrics_display = {}
                for ratio in selected_ratios:
                    if ratio in latest_data and pd.notna(latest_data[ratio]):
                        if ratio.endswith(('margin', 'roa', 'roe', 'growth')):
                            metrics_display[ratio.replace('_', ' ').title()] = f"{latest_data[ratio]:.1f}%"
                        else:
                            metrics_display[ratio.replace('_', ' ').title()] = f"{latest_data[ratio]:.2f}"
                
                if metrics_display:
                    metrics_df = pd.DataFrame(list(metrics_display.items()), 
                                            columns=['Metric', 'Value'])
                    st.dataframe(metrics_df, hide_index=True)
                
                # Trend analysis
                if show_trends and len(df) > 1:
                    st.write("**Financial Trends**")
                    trend_fig = visualizer.create_trend_chart(df, detail_ticker, selected_ratios)
                    st.plotly_chart(trend_fig, use_container_width=True)
                
                # Raw data table
                with st.expander("View Raw Financial Data"):
                    # Build display columns and remove duplicates while preserving order
                    cols = ['year'] + [col for col in df.columns
                                        if col not in ['company', 'period_end'] and df[col].notna().any()]
                    seen = set()
                    display_cols = []
                    for c in cols:
                        if c not in seen:
                            seen.add(c)
                            display_cols.append(c)

                    st.dataframe(df[display_cols], hide_index=True)
    
    with tab3:
        st.header("Peer Group Comparison")
        
        if show_peer_comparison and len(companies_data) > 1:
            # Peer comparison table
            st.subheader("Peer Ranking Table")
            
            # Create comparison DataFrame
            comparison_data = []
            for ticker, df in companies_data.items():
                if not df.empty:
                    row = {'Company': ticker}
                    latest_data = df.iloc[0]
                    
                    for ratio in selected_ratios:
                        if ratio in latest_data and pd.notna(latest_data[ratio]):
                            row[ratio.replace('_', ' ').title()] = latest_data[ratio]
                    
                    if len(row) > 1:  # Has at least company name and one metric
                        comparison_data.append(row)
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                
                # Style the dataframe with conditional formatting
                def highlight_extremes(s):
                    if s.name == 'Company':
                        return [''] * len(s)
                    
                    colors = []
                    for val in s:
                        if pd.isna(val):
                            colors.append('')
                        else:
                            # Color top quartile green, bottom quartile red
                            q75 = s.quantile(0.75)
                            q25 = s.quantile(0.25)
                            
                            if val >= q75:
                                colors.append('background-color: lightgreen')
                            elif val <= q25:
                                colors.append('background-color: lightcoral')
                            else:
                                colors.append('')
                    return colors
                
                styled_df = comparison_df.style.apply(highlight_extremes, axis=0)
                st.dataframe(styled_df, hide_index=True)
                
                st.info("🟢 Green: Top quartile performers | 🔴 Red: Bottom quartile performers")
                
                # Distribution plots
                st.subheader("Peer Group Distributions")
                
                for ratio in selected_ratios[:2]:  # Show distributions for first 2 ratios
                    dist_fig = visualizer.create_peer_distribution(companies_data, ratio)
                    st.plotly_chart(dist_fig, use_container_width=True)
        else:
            st.info("Select multiple companies to enable peer comparison.")
    
    with tab4:
        st.header("Data Export & Reports")
        
        # Export options
        export_format = st.radio(
            "Select export format:",
            ["CSV", "JSON"],
            help="Choose format for downloading financial data"
        )
        
        # Prepare export data
        export_data = {}
        for ticker, df in companies_data.items():
            if not df.empty:
                export_data[ticker] = df.to_dict('records')
        
        if export_data:
            if export_format == "CSV":
                # Combine all companies into one CSV
                all_data = []
                for ticker, df in companies_data.items():
                    df_copy = df.copy()
                    df_copy['ticker'] = ticker
                    all_data.append(df_copy)
                
                if all_data:
                    combined_df = pd.concat(all_data, ignore_index=True)
                    csv_buffer = BytesIO()
                    combined_df.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()
                    
                    st.download_button(
                        label="📥 Download CSV",
                        data=csv_data,
                        file_name=f"financial_analysis_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            
            else:  # JSON format
                json_data = json.dumps(export_data, indent=2, default=str)
                st.download_button(
                    label="📥 Download JSON",
                    data=json_data,
                    file_name=f"financial_analysis_{datetime.now().strftime('%Y%m%d')}.json",
                    mime="application/json"
                )

                # PDF export
                if export_format == "PDF":
                    try:
                        def generate_pdf_report(companies_data: Dict[str, pd.DataFrame],
                                                selected_ratios: List[str],
                                                visualizer: Visualizer,
                                                summary_text: str) -> bytes:
                            """Generate a PDF bytes blob containing summary, key metrics, and charts.

                            This function renders Plotly figures to PNG (requires kaleido) and
                            embeds them into a reportlab PDF.
                            """
                            buffer = BytesIO()
                            c = canvas.Canvas(buffer, pagesize=letter)
                            width, height = letter

                            # Title
                            c.setFont("Helvetica-Bold", 16)
                            c.drawString(40, height - 50, "Financial Analysis Report")
                            c.setFont("Helvetica", 10)

                            # Summary text (wrap lines)
                            y = height - 80
                            for line in summary_text.strip().splitlines():
                                wrapped = textwrap.wrap(line, width=100)
                                for w in wrapped:
                                    if y < 120:
                                        c.showPage()
                                        y = height - 50
                                    c.drawString(40, y, w)
                                    y -= 12

                            # Add charts: ratio comparisons (limit to first 4)
                            for ratio in selected_ratios[:4]:
                                try:
                                    fig = visualizer.create_ratio_comparison(companies_data, ratio)
                                    img_bytes = fig.to_image(format='png')
                                except Exception:
                                    # If rendering fails (e.g., kaleido missing), skip image
                                    img_bytes = None

                                if img_bytes:
                                    img = ImageReader(BytesIO(img_bytes))
                                    # Start a new page if not enough space
                                    if y < 300:
                                        c.showPage()
                                        y = height - 50
                                    img_w = width - 80
                                    # maintain aspect ratio roughly - use a height of 250
                                    img_h = 250
                                    c.drawImage(img, 40, y - img_h, width=img_w, height=img_h)
                                    y -= (img_h + 20)

                            # Add per-company trend charts (one per company up to 4 companies)
                            for i, (ticker, df) in enumerate(list(companies_data.items())[:4]):
                                try:
                                    fig = visualizer.create_trend_chart(df, ticker, selected_ratios)
                                    img_bytes = fig.to_image(format='png')
                                except Exception:
                                    img_bytes = None

                                if img_bytes:
                                    if y < 300:
                                        c.showPage()
                                        y = height - 50
                                    img = ImageReader(BytesIO(img_bytes))
                                    img_w = width - 80
                                    img_h = 250
                                    c.drawImage(img, 40, y - img_h, width=img_w, height=img_h)
                                    y -= (img_h + 20)

                            # Finalize
                            c.save()
                            buffer.seek(0)
                            return buffer.getvalue()

                        pdf_bytes = generate_pdf_report(companies_data, selected_ratios, visualizer, summary_text)
                        st.download_button(
                            label="📥 Download PDF Report",
                            data=pdf_bytes,
                            file_name=f"financial_report_{datetime.now().strftime('%Y%m%d')}.pdf",
                            mime="application/pdf"
                        )
                    except Exception as e:
                        st.error("Failed to generate PDF. Ensure 'kaleido' is installed (see requirements). Error: " + str(e))
        
        # Summary report
        st.subheader("Analysis Summary")
        
        if companies_data:
            summary_text = f"""
            ## Financial Analysis Report
            **Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            **Companies Analyzed:** {', '.join(companies_data.keys())}
            
            **Key Findings:**
            """
            
            # Add key insights
            if avg_margins:
                best_margin_company = max(companies_data.items(), 
                                        key=lambda x: x[1].iloc[0]['net_margin'] 
                                        if not x[1].empty and 'net_margin' in x[1].columns 
                                        and pd.notna(x[1].iloc[0]['net_margin']) else -999)
                
                if best_margin_company[1].iloc[0]['net_margin'] > -999:
                    summary_text += f"""
            - **Highest Net Margin:** {best_margin_company[0]} ({best_margin_company[1].iloc[0]['net_margin']:.1f}%)
            """
            
            if avg_growth:
                best_growth_company = max(companies_data.items(),
                                        key=lambda x: x[1].iloc[0]['revenue_growth']
                                        if not x[1].empty and 'revenue_growth' in x[1].columns
                                        and pd.notna(x[1].iloc[0]['revenue_growth']) else -999)
                
                if best_growth_company[1].iloc[0]['revenue_growth'] > -999:
                    summary_text += f"""
            - **Highest Revenue Growth:** {best_growth_company[0]} ({best_growth_company[1].iloc[0]['revenue_growth']:.1f}%)
            """
            
            summary_text += f"""
            
            **Ratios Analyzed:** {', '.join([r.replace('_', ' ').title() for r in selected_ratios])}
            
            **Data Sources:** SEC EDGAR API 
            
            """
            
            st.markdown(summary_text)
            
            

if __name__ == "__main__":
    main()