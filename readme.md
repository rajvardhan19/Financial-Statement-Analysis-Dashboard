# Financial Statement Analysis Dashboard

A comprehensive web-based financial analysis platform built with Streamlit that automatically ingests, normalizes, and visualizes key financial statements for public companies. Enable peer-group comparisons and highlight outliers across profitability, liquidity, and leverage ratios.

## 🚀 Features

### Core Functionality
- **Automated Data Ingestion**: Fetch filings (10-K, 10-Q) from SEC EDGAR API
- **Data Normalization**: Map GAAP line items to unified schema across companies and years
- **Comprehensive Ratio Analysis**: Calculate 20+ financial ratios across all major categories
- **Peer Group Comparison**: Compare companies within sectors with statistical analysis
- **Interactive Visualizations**: Bar charts, time-series, box plots, and distribution analysis
- **Export Capabilities**: Download CSV/JSON data and generate PDF reports

### Ratio Categories
- **Profitability**: Gross margin, operating margin, net margin, ROA, ROE
- **Liquidity**: Current ratio, quick ratio, cash ratio
- **Leverage**: Debt-to-equity, debt-to-assets, equity ratio, financial leverage
- **Efficiency**: Asset turnover, inventory turnover, receivables turnover
- **Growth**: YoY percentage changes, CAGR calculations

### Advanced Features
- **Trend Analysis**: Historical ratio progression and pattern identification
- **Outlier Detection**: Statistical identification of unusual performance
- **Industry Benchmarking**: Compare against industry averages
- **DuPont Analysis**: Decompose ROE into component drivers
- **Data Quality Validation**: Comprehensive error checking and anomaly detection

## 📁 Project Structure

```
financial_dashboard/
├── app.py                  # Main Streamlit application
├── data_ingest.py         # SEC EDGAR API integration and caching
├── parser.py              # XBRL/JSON parsing and normalization
├── ratios.py              # Financial ratio calculations
├── visualize.py           # Plotting functions (integrated in main app)
├── utils.py               # Helper functions and utilities
├── tests/                 # Test suite
│   ├── test_parser.py
│   ├── test_ratios.py
│   └── test_data_ingest.py
├── requirements.txt       # Python dependencies
├── README.md             # This file
└── .streamlit/
    └── secrets.toml      # Configuration secrets (not in repo)
```

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation Steps

1. **Clone the repository**
```bash
git clone <repository-url>
cd financial_dashboard
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure secrets (optional)**
Create `.streamlit/secrets.toml` for API configurations:
```toml
[api]
sec_user_agent = "YourApp (your@email.com)"
rate_limit_delay = 0.1

[cache]
ttl_hours = 2
max_size_mb = 100
```

5. **Run the application**
```bash
streamlit run app.py
```

The dashboard will be available at `http://localhost:8501`

## 📊 Usage Guide

### Getting Started
1. **Select Companies**: Choose from supported tickers (AAPL, MSFT, GOOGL, etc.)
2. **Configure Analysis**: Select ratios and analysis options in the sidebar
3. **View Results**: Navigate through tabs for different analysis views

### Dashboard Tabs

#### 📈 Overview
- High-level metrics and KPI cards
- Ratio comparison charts across selected companies
- Quick performance insights

#### 🔍 Detailed Analysis
- Company-specific deep dive
- Historical trend analysis
- Raw financial data tables

#### 📊 Peer Comparison
- Sortable comparison tables with conditional formatting
- Peer group distribution plots
- Statistical quartile analysis

#### 📋 Data Export
- CSV/JSON data downloads
- PDF report generation
- Analysis summary reports

### Supported Companies
Currently supports 20+ major public companies including:
- **Technology**: AAPL, MSFT, GOOGL, META, NVDA
- **E-commerce**: AMZN
- **Automotive**: TSLA  
- **Financial**: JPM, BAC
- **Healthcare**: JNJ, UNH
- **Consumer**: PG, HD, DIS

## 🔧 Technical Implementation

### Data Pipeline
1. **Ingestion**: SEC EDGAR API calls with rate limiting
2. **Parsing**: XBRL concept mapping to normalized schema
3. **Calculation**: Comprehensive ratio computation
4. **Validation**: Data quality checks and anomaly detection
5. **Visualization**: Interactive Plotly charts

### Key Design Patterns
- **Caching Strategy**: Multi-level caching with TTL for performance
- **Error Handling**: Graceful degradation with sample data fallback
- **Modular Architecture**: Separation of concerns across modules
- **Type Safety**: Comprehensive type hints throughout codebase

### Performance Optimizations
- Streamlit `@st.cache_data` for API responses
- Efficient pandas operations for ratio calculations
- Lazy loading of visualizations
- Progressive data loading with progress indicators

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v --cov=. --cov-report=html
```

### Test Coverage
- **Unit Tests**: Individual function testing
- **Integration Tests**: End-to-end data pipeline
- **Performance Tests**: Large dataset handling
- **Error Handling**: Edge case validation

## 📈 API Integration

### SEC EDGAR API
- **Rate Limiting**: 10 requests/second compliance
- **User Agent**: Required identification header
- **Data Format**: JSON response parsing
- **Error Handling**: Robust retry logic with exponential backoff

### Sample Data
When SEC API is unavailable, the system generates realistic sample data based on:
- Industry-specific financial patterns
- Historical volatility modeling
- Realistic ratio relationships

## 🔍 Financial Ratios Reference

### Profitability Ratios
- **Gross Margin**: (Revenue - COGS) / Revenue × 100
- **Operating Margin**: Operating Income / Revenue × 100
- **Net Margin**: Net Income / Revenue × 100
- **ROA**: Net Income / Total Assets × 100
- **ROE**: Net Income / Shareholders' Equity × 100

### Liquidity Ratios
- **Current Ratio**: Current Assets / Current Liabilities
- **Quick Ratio**: (Current Assets - Inventory) / Current Liabilities
- **Cash Ratio**: Cash & Equivalents / Current Liabilities

### Leverage Ratios
- **Debt-to-Equity**: Total Debt / Shareholders' Equity
- **Debt-to-Assets**: Total Debt / Total Assets
- **Equity Ratio**: Shareholders' Equity / Total Assets

### Efficiency Ratios
- **Asset Turnover**: Revenue / Total Assets
- **Inventory Turnover**: COGS / Average Inventory
- **Receivables Turnover**: Revenue / Average Receivables

## 🚀 Deployment

### Local Development
```bash
streamlit run app.py --server.port 8501
```

### Production Deployment
The application can be deployed on:
- **Streamlit Cloud**: Direct GitHub integration
- **Heroku**: Web app hosting
- **AWS/GCP**: Container deployment
- **Docker**: Containerized deployment

### Environment Variables
```bash
SEC_USER_AGENT="YourApp (email@domain.com)"
CACHE_TTL_HOURS=2
MAX_COMPANIES=10
```

## 🤝 Contributing

### Development Guidelines
1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

### Code Standards
- **Black** for code formatting
- **Flake8** for linting
- **MyPy** for type checking
- **Pytest** for testing
- **Comprehensive docstrings** for all functions

### Adding New Ratios
1. Add ratio definition to `ratios.py`
2. Implement calculation method
3. Add validation rules
4. Include in visualization options
5. Update documentation

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## ⚠️ Disclaimers

- **Educational Purpose**: This tool is for educational and research purposes only
- **No Investment Advice**: Results should not be used as the sole basis for investment decisions
- **Data Accuracy**: While we strive for accuracy, users should verify critical data independently
- **SEC Compliance**: Users must comply with SEC EDGAR API terms of service

## 📞 Support

For questions, issues, or contributions:
- **Issues**: GitHub Issues tracker
- **Documentation**: In-code docstrings and comments
- **Community**: Discussions tab for questions and ideas

## 🔄 Version History

- **v1.0.0**: Initial release with core functionality
- **v1.1.0**: Enhanced peer comparison and export features
- **v1.2.0**: Advanced ratio calculations and industry benchmarking

## 🎯 Future Enhancements

- **Real-time Market Data**: Integration with financial data providers
- **Machine Learning**: Predictive analytics and anomaly detection
- **Mobile Responsiveness**: Enhanced mobile experience
- **Custom Dashboards**: User-configurable dashboard layouts
- **API Access**: RESTful API for programmatic access
- **Multi-language Support**: International financial standards