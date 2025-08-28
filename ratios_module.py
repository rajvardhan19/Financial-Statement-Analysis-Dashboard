"""
Financial ratios calculation module.
Comprehensive ratio calculations for profitability, liquidity, leverage, and efficiency analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import logging
from dataclasses import dataclass
import warnings

logger = logging.getLogger(__name__)

@dataclass
class RatioDefinition:
    """Definition of a financial ratio with metadata."""
    name: str
    category: str
    formula: str
    interpretation: str
    good_range: Optional[Tuple[float, float]] = None
    higher_is_better: Optional[bool] = None

class FinancialRatioCalculator:
    """
    Comprehensive financial ratio calculator with industry benchmarking capabilities.
    """
    
    def __init__(self):
        """Initialize the ratio calculator with ratio definitions."""
        
        self.ratio_definitions = {
            # Profitability Ratios
            'gross_margin': RatioDefinition(
                name='Gross Margin',
                category='Profitability',
                formula='(Revenues - Cost of Sales) / Revenues * 100',
                interpretation='Percentage of revenue retained after direct costs',
                good_range=(20.0, 60.0),
                higher_is_better=True
            ),
            
            'operating_margin': RatioDefinition(
                name='Operating Margin',
                category='Profitability',
                formula='Operating Income / Revenues * 100',
                interpretation='Operating efficiency before interest and taxes',
                good_range=(10.0, 30.0),
                higher_is_better=True
            ),
            
            'net_margin': RatioDefinition(
                name='Net Margin',
                category='Profitability',
                formula='Net Income / Revenues * 100',
                interpretation='Profit margin after all expenses',
                good_range=(5.0, 25.0),
                higher_is_better=True
            ),
            
            'return_on_assets': RatioDefinition(
                name='Return on Assets (ROA)',
                category='Profitability',
                formula='Net Income / Total Assets * 100',
                interpretation='Efficiency in using assets to generate profit',
                good_range=(5.0, 20.0),
                higher_is_better=True
            ),
            
            'return_on_equity': RatioDefinition(
                name='Return on Equity (ROE)',
                category='Profitability',
                formula='Net Income / Stockholders Equity * 100',
                interpretation='Return generated on shareholders investment',
                good_range=(10.0, 25.0),
                higher_is_better=True
            ),
            
            # Liquidity Ratios
            'current_ratio': RatioDefinition(
                name='Current Ratio',
                category='Liquidity',
                formula='Current Assets / Current Liabilities',
                interpretation='Ability to pay short-term obligations',
                good_range=(1.2, 3.0),
                higher_is_better=True
            ),
            
            'quick_ratio': RatioDefinition(
                name='Quick Ratio',
                category='Liquidity',
                formula='(Current Assets - Inventory) / Current Liabilities',
                interpretation='Liquidity excluding inventory',
                good_range=(1.0, 2.0),
                higher_is_better=True
            ),
            
            'cash_ratio': RatioDefinition(
                name='Cash Ratio',
                category='Liquidity',
                formula='Cash and Equivalents / Current Liabilities',
                interpretation='Most conservative liquidity measure',
                good_range=(0.2, 1.0),
                higher_is_better=True
            ),
            
            # Leverage/Solvency Ratios
            'debt_to_equity': RatioDefinition(
                name='Debt-to-Equity',
                category='Leverage',
                formula='Total Debt / Stockholders Equity',
                interpretation='Financial leverage and risk',
                good_range=(0.0, 1.0),
                higher_is_better=False
            ),
            
            'debt_to_assets': RatioDefinition(
                name='Debt-to-Assets',
                category='Leverage',
                formula='Total Debt / Total Assets',
                interpretation='Proportion of assets financed by debt',
                good_range=(0.0, 0.6),
                higher_is_better=False
            ),
            
            'equity_ratio': RatioDefinition(
                name='Equity Ratio',
                category='Leverage',
                formula='Stockholders Equity / Total Assets',
                interpretation='Proportion of assets financed by equity',
                good_range=(0.4, 0.8),
                higher_is_better=True
            ),
            
            'interest_coverage': RatioDefinition(
                name='Interest Coverage',
                category='Leverage',
                formula='Operating Income / Interest Expense',
                interpretation='Ability to service debt interest',
                good_range=(2.5, 10.0),
                higher_is_better=True
            ),
            
            # Efficiency Ratios
            'asset_turnover': RatioDefinition(
                name='Asset Turnover',
                category='Efficiency',
                formula='Revenues / Total Assets',
                interpretation='Efficiency in using assets to generate sales',
                good_range=(0.5, 3.0),
                higher_is_better=True
            ),
            
            'inventory_turnover': RatioDefinition(
                name='Inventory Turnover',
                category='Efficiency',
                formula='Cost of Sales / Average Inventory',
                interpretation='How quickly inventory is sold',
                good_range=(4.0, 12.0),
                higher_is_better=True
            ),
            
            'receivables_turnover': RatioDefinition(
                name='Receivables Turnover',
                category='Efficiency',
                formula='Revenues / Average Accounts Receivable',
                interpretation='Efficiency in collecting receivables',
                good_range=(6.0, 15.0),
                higher_is_better=True
            )
        }
    
    def calculate_all_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all available financial ratios for the given data.
        
        Args:
            df: DataFrame with normalized financial data
            
        Returns:
            DataFrame with calculated ratios
        """
        if df.empty:
            logger.warning("Empty DataFrame provided for ratio calculation")
            return df
        
        result_df = df.copy()
        
        try:
            # Profitability Ratios
            result_df = self._calculate_profitability_ratios(result_df)
            
            # Liquidity Ratios
            result_df = self._calculate_liquidity_ratios(result_df)
            
            # Leverage Ratios
            result_df = self._calculate_leverage_ratios(result_df)
            
            # Efficiency Ratios
            result_df = self._calculate_efficiency_ratios(result_df)
            
            # Growth Ratios (requires historical data)
            if len(result_df) > 1:
                result_df = self._calculate_growth_ratios(result_df)
            
            # Valuation Ratios (if market data available)
            result_df = self._calculate_valuation_ratios(result_df)
            
            logger.info(f"Successfully calculated ratios for {len(result_df)} records")
            
        except Exception as e:
            logger.error(f"Error calculating ratios: {str(e)}")
        
        return result_df
    
    def _calculate_profitability_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate profitability ratios."""
        result_df = df.copy()
        
        # Gross Margin
        if 'revenues' in df.columns and 'cost_of_sales' in df.columns:
            result_df['gross_margin'] = self._safe_divide(
                df['revenues'] - df['cost_of_sales'].fillna(0), 
                df['revenues']
            ) * 100
        
        # Operating Margin
        if 'operating_income' in df.columns and 'revenues' in df.columns:
            result_df['operating_margin'] = self._safe_divide(
                df['operating_income'], 
                df['revenues']
            ) * 100
        
        # Net Margin
        if 'net_income' in df.columns and 'revenues' in df.columns:
            result_df['net_margin'] = self._safe_divide(
                df['net_income'], 
                df['revenues']
            ) * 100
        
        # Return on Assets (ROA)
        if 'net_income' in df.columns and 'total_assets' in df.columns:
            result_df['return_on_assets'] = self._safe_divide(
                df['net_income'], 
                df['total_assets']
            ) * 100
        
        # Return on Equity (ROE)
        if 'net_income' in df.columns and 'stockholders_equity' in df.columns:
            result_df['return_on_equity'] = self._safe_divide(
                df['net_income'], 
                df['stockholders_equity']
            ) * 100
        
        # EBITDA Margin (if operating income and depreciation available)
        if 'operating_income' in df.columns and 'revenues' in df.columns:
            # For now, approximate EBITDA as operating income
            # In real implementation, would add back depreciation and amortization
            result_df['ebitda_margin'] = result_df['operating_margin']
        
        return result_df
    
    def _calculate_liquidity_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate liquidity ratios."""
        result_df = df.copy()
        
        # Current Ratio
        if 'current_assets' in df.columns and 'current_liabilities' in df.columns:
            result_df['current_ratio'] = self._safe_divide(
                df['current_assets'], 
                df['current_liabilities']
            )
        
        # Quick Ratio (Acid-Test Ratio)
        if all(col in df.columns for col in ['current_assets', 'inventory', 'current_liabilities']):
            quick_assets = df['current_assets'] - df['inventory'].fillna(0)
            result_df['quick_ratio'] = self._safe_divide(
                quick_assets, 
                df['current_liabilities']
            )
        elif 'current_assets' in df.columns and 'current_liabilities' in df.columns:
            # If no inventory data, assume quick ratio equals current ratio
            result_df['quick_ratio'] = result_df.get('current_ratio')
        
        # Cash Ratio
        if 'cash_and_equivalents' in df.columns and 'current_liabilities' in df.columns:
            result_df['cash_ratio'] = self._safe_divide(
                df['cash_and_equivalents'], 
                df['current_liabilities']
            )
        
        # Working Capital (absolute amount, not a ratio)
        if 'current_assets' in df.columns and 'current_liabilities' in df.columns:
            result_df['working_capital'] = df['current_assets'] - df['current_liabilities'].fillna(0)
        
        return result_df
    
    def _calculate_leverage_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate leverage and solvency ratios."""
        result_df = df.copy()
        
        # Calculate total debt if not already present
        if 'total_debt' not in result_df.columns:
            short_debt = df.get('short_term_debt', pd.Series([0] * len(df))).fillna(0)
            long_debt = df.get('long_term_debt', pd.Series([0] * len(df))).fillna(0)
            result_df['total_debt'] = short_debt + long_debt
        
        # Debt-to-Equity Ratio
        if 'total_debt' in result_df.columns and 'stockholders_equity' in df.columns:
            result_df['debt_to_equity'] = self._safe_divide(
                result_df['total_debt'], 
                df['stockholders_equity']
            )
        
        # Debt-to-Assets Ratio
        if 'total_debt' in result_df.columns and 'total_assets' in df.columns:
            result_df['debt_to_assets'] = self._safe_divide(
                result_df['total_debt'], 
                df['total_assets']
            )
        
        # Equity Ratio
        if 'stockholders_equity' in df.columns and 'total_assets' in df.columns:
            result_df['equity_ratio'] = self._safe_divide(
                df['stockholders_equity'], 
                df['total_assets']
            )
        
        # Interest Coverage Ratio (requires interest expense data)
        # This would typically require interest expense from income statement
        # For demonstration, we'll skip this or estimate it
        
        # Financial Leverage Ratio
        if 'total_assets' in df.columns and 'stockholders_equity' in df.columns:
            result_df['financial_leverage'] = self._safe_divide(
                df['total_assets'], 
                df['stockholders_equity']
            )
        
        return result_df
    
    def _calculate_efficiency_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate efficiency and activity ratios."""
        result_df = df.copy()
        
        # Asset Turnover
        if 'revenues' in df.columns and 'total_assets' in df.columns:
            result_df['asset_turnover'] = self._safe_divide(
                df['revenues'], 
                df['total_assets']
            )
        
        # Inventory Turnover (requires cost of goods sold)
        if 'cost_of_sales' in df.columns and 'inventory' in df.columns:
            result_df['inventory_turnover'] = self._safe_divide(
                df['cost_of_sales'], 
                df['inventory']
            )
            
            # Days Sales in Inventory
            result_df['days_sales_inventory'] = self._safe_divide(
                365, 
                result_df['inventory_turnover']
            )
        
        # Property, Plant & Equipment Turnover
        if 'revenues' in df.columns and 'property_plant_equipment' in df.columns:
            result_df['ppe_turnover'] = self._safe_divide(
                df['revenues'], 
                df['property_plant_equipment']
            )
        
        return result_df
    
    def _calculate_growth_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate year-over-year growth ratios."""
        if len(df) < 2:
            return df
        
        result_df = df.copy().sort_values('fiscal_year')
        
        # Define metrics to calculate growth for
        growth_metrics = [
            'revenues', 'net_income', 'total_assets', 'stockholders_equity',
            'operating_income', 'gross_profit', 'operating_cash_flow'
        ]
        
        for metric in growth_metrics:
            if metric in result_df.columns:
                growth_col = f"{metric}_growth"
                result_df[growth_col] = result_df[metric].pct_change() * 100
        
        # Calculate compound annual growth rate (CAGR) if more than 2 years
        if len(result_df) >= 3:
            result_df = self._calculate_cagr(result_df, growth_metrics)
        
        return result_df.sort_values('fiscal_year', ascending=False)
    
    def _calculate_cagr(self, df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
        """Calculate Compound Annual Growth Rate."""
        result_df = df.copy()
        n_years = len(df) - 1
        
        for metric in metrics:
            if metric in result_df.columns:
                first_value = result_df[metric].iloc[0]  # Oldest year
                last_value = result_df[metric].iloc[-1]   # Most recent year
                
                if pd.notna(first_value) and pd.notna(last_value) and first_value > 0:
                    cagr = ((last_value / first_value) ** (1 / n_years) - 1) * 100
                    result_df[f"{metric}_cagr"] = cagr
        
        return result_df
    
    def _calculate_valuation_ratios(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate valuation ratios (limited without market data)."""
        result_df = df.copy()
        
        # Book Value per Share
        if 'stockholders_equity' in df.columns and 'shares_outstanding' in df.columns:
            result_df['book_value_per_share'] = self._safe_divide(
                df['stockholders_equity'], 
                df['shares_outstanding']
            )
        
        # Earnings per Share (basic)
        if 'net_income' in df.columns and 'shares_outstanding' in df.columns:
            result_df['earnings_per_share'] = self._safe_divide(
                df['net_income'], 
                df['shares_outstanding']
            )
        
        return result_df
    
    def _safe_divide(self, numerator: pd.Series, denominator: pd.Series) -> pd.Series:
        """
        Safely divide two series, handling division by zero and NaN values.
        
        Args:
            numerator: Numerator series
            denominator: Denominator series
            
        Returns:
            Result series with NaN for invalid operations
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            
            result = numerator / denominator
            
            # Set infinite values to NaN
            result = result.replace([np.inf, -np.inf], np.nan)
            
            # Handle cases where denominator is zero or NaN
            mask = (denominator == 0) | denominator.isna() | numerator.isna()
            result.loc[mask] = np.nan
            
            return result
    
    def calculate_ratio_percentiles(self, companies_data: Dict[str, pd.DataFrame], 
                                   ratio_name: str) -> Dict[str, float]:
        """
        Calculate percentiles for a specific ratio across all companies.
        
        Args:
            companies_data: Dictionary mapping tickers to DataFrames
            ratio_name: Name of the ratio to analyze
            
        Returns:
            Dictionary with percentile statistics
        """
        all_values = []
        
        for ticker, df in companies_data.items():
            if not df.empty and ratio_name in df.columns:
                # Use the most recent value for each company
                latest_value = df.iloc[0][ratio_name]
                if pd.notna(latest_value):
                    all_values.append(latest_value)
        
        if not all_values:
            return {}
        
        values_series = pd.Series(all_values)
        
        return {
            'count': len(all_values),
            'mean': values_series.mean(),
            'median': values_series.median(),
            'std': values_series.std(),
            'min': values_series.min(),
            'max': values_series.max(),
            'q25': values_series.quantile(0.25),
            'q75': values_series.quantile(0.75),
            'q10': values_series.quantile(0.10),
            'q90': values_series.quantile(0.90)
        }
    
    def rank_companies_by_ratio(self, companies_data: Dict[str, pd.DataFrame], 
                               ratio_name: str, ascending: bool = False) -> pd.DataFrame:
        """
        Rank companies by a specific ratio.
        
        Args:
            companies_data: Dictionary mapping tickers to DataFrames
            ratio_name: Name of the ratio to rank by
            ascending: Whether to rank in ascending order
            
        Returns:
            DataFrame with company rankings
        """
        rankings = []
        
        for ticker, df in companies_data.items():
            if not df.empty and ratio_name in df.columns:
                latest_value = df.iloc[0][ratio_name]
                if pd.notna(latest_value):
                    rankings.append({
                        'ticker': ticker,
                        'company_name': df.iloc[0].get('company_name', ticker),
                        ratio_name: latest_value,
                        'fiscal_year': df.iloc[0]['fiscal_year']
                    })
        
        if not rankings:
            return pd.DataFrame()
        
        ranking_df = pd.DataFrame(rankings)
        ranking_df = ranking_df.sort_values(ratio_name, ascending=ascending)
        ranking_df['rank'] = range(1, len(ranking_df) + 1)
        
        return ranking_df
    
    def identify_outliers(self, companies_data: Dict[str, pd.DataFrame], 
                         ratio_name: str, method: str = 'iqr') -> Dict[str, List[str]]:
        """
        Identify outlier companies for a specific ratio.
        
        Args:
            companies_data: Dictionary mapping tickers to DataFrames
            ratio_name: Name of the ratio to analyze
            method: Method to use ('iqr' or 'zscore')
            
        Returns:
            Dictionary with lists of outlier tickers
        """
        all_values = {}
        
        for ticker, df in companies_data.items():
            if not df.empty and ratio_name in df.columns:
                latest_value = df.iloc[0][ratio_name]
                if pd.notna(latest_value):
                    all_values[ticker] = latest_value
        
        if len(all_values) < 4:  # Need at least 4 companies for meaningful outlier detection
            return {'high_outliers': [], 'low_outliers': []}
        
        values_series = pd.Series(all_values)
        
        if method == 'iqr':
            Q1 = values_series.quantile(0.25)
            Q3 = values_series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            high_outliers = values_series[values_series > upper_bound].index.tolist()
            low_outliers = values_series[values_series < lower_bound].index.tolist()
            
        elif method == 'zscore':
            z_scores = np.abs((values_series - values_series.mean()) / values_series.std())
            outlier_threshold = 2.0
            
            outliers = z_scores[z_scores > outlier_threshold].index.tolist()
            
            # Separate into high and low outliers
            mean_val = values_series.mean()
            high_outliers = [t for t in outliers if values_series[t] > mean_val]
            low_outliers = [t for t in outliers if values_series[t] < mean_val]
        
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        return {
            'high_outliers': high_outliers,
            'low_outliers': low_outliers
        }
    
    def calculate_dupont_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate DuPont analysis components.
        ROE = Net Margin × Asset Turnover × Financial Leverage
        
        Args:
            df: DataFrame with financial data
            
        Returns:
            DataFrame with DuPont components
        """
        result_df = df.copy()
        
        required_cols = ['net_margin', 'asset_turnover', 'financial_leverage']
        
        if all(col in result_df.columns for col in required_cols):
            # DuPont ROE calculation
            result_df['dupont_roe'] = (
                (result_df['net_margin'] / 100) *  # Convert percentage to decimal
                result_df['asset_turnover'] * 
                result_df['financial_leverage']
            ) * 100  # Convert back to percentage
            
            # Calculate contribution of each component
            if 'return_on_equity' in result_df.columns:
                total_roe = result_df['return_on_equity'].fillna(result_df['dupont_roe'])
                
                result_df['margin_contribution'] = (result_df['net_margin'] / 100) / (total_roe / 100)
                result_df['efficiency_contribution'] = result_df['asset_turnover'] / (total_roe / 100)
                result_df['leverage_contribution'] = result_df['financial_leverage'] / (total_roe / 100)
        
        return result_df
    
    def generate_ratio_report(self, df: pd.DataFrame, ticker: str) -> Dict[str, Any]:
        """
        Generate a comprehensive ratio analysis report for a single company.
        
        Args:
            df: DataFrame with calculated ratios
            ticker: Company ticker symbol
            
        Returns:
            Dictionary containing ratio analysis report
        """
        if df.empty:
            return {'ticker': ticker, 'error': 'No data available'}
        
        latest_data = df.iloc[0]  # Most recent year
        report = {
            'ticker': ticker,
            'company_name': latest_data.get('company_name', ticker),
            'fiscal_year': latest_data.get('fiscal_year'),
            'analysis_date': pd.Timestamp.now().strftime('%Y-%m-%d'),
            'categories': {}
        }
        
        # Organize ratios by category
        for ratio_name, definition in self.ratio_definitions.items():
            if ratio_name in latest_data and pd.notna(latest_data[ratio_name]):
                category = definition.category
                
                if category not in report['categories']:
                    report['categories'][category] = {
                        'ratios': [],
                        'overall_assessment': 'N/A'
                    }
                
                # Assess ratio performance
                ratio_value = latest_data[ratio_name]
                assessment = self._assess_ratio_performance(ratio_value, definition)
                
                report['categories'][category]['ratios'].append({
                    'name': definition.name,
                    'value': ratio_value,
                    'assessment': assessment,
                    'interpretation': definition.interpretation
                })
        
        # Add trend analysis if historical data available
        if len(df) > 1:
            report['trends'] = self._analyze_trends(df)
        
        return report
    
    def _assess_ratio_performance(self, value: float, definition: RatioDefinition) -> str:
        """Assess the performance of a ratio value."""
        if definition.good_range is None:
            return 'Neutral'
        
        min_good, max_good = definition.good_range
        
        if min_good <= value <= max_good:
            return 'Good'
        elif definition.higher_is_better:
            if value > max_good:
                return 'Excellent'
            else:
                return 'Poor'
        else:  # lower_is_better or neutral
            if value < min_good:
                return 'Excellent'
            else:
                return 'Poor'
    
    def _analyze_trends(self, df: pd.DataFrame) -> Dict[str, str]:
        """Analyze trends in key ratios over time."""
        trends = {}
        
        key_ratios = ['net_margin', 'return_on_equity', 'current_ratio', 'debt_to_equity']
        
        for ratio in key_ratios:
            if ratio in df.columns and df[ratio].count() >= 2:
                # Calculate trend over available periods
                values = df[ratio].dropna()
                if len(values) >= 2:
                    recent_avg = values.head(2).mean()  # Last 2 years
                    older_avg = values.tail(2).mean()   # First 2 years available
                    
                    change_pct = ((recent_avg - older_avg) / older_avg) * 100
                    
                    if abs(change_pct) < 5:
                        trends[ratio] = 'Stable'
                    elif change_pct > 5:
                        trends[ratio] = 'Improving'
                    else:
                        trends[ratio] = 'Declining'
        
        return trends
    
    def calculate_industry_adjusted_ratios(self, df: pd.DataFrame, 
                                         industry_benchmarks: Dict[str, float]) -> pd.DataFrame:
        """
        Calculate industry-adjusted ratios by comparing to industry benchmarks.
        
        Args:
            df: DataFrame with calculated ratios
            industry_benchmarks: Dictionary of ratio names to industry average values
            
        Returns:
            DataFrame with industry-adjusted ratio scores
        """
        result_df = df.copy()
        
        for ratio_name, industry_avg in industry_benchmarks.items():
            if ratio_name in result_df.columns:
                # Calculate relative performance vs industry
                adjusted_col = f"{ratio_name}_vs_industry"
                
                # For ratios where higher is better
                definition = self.ratio_definitions.get(ratio_name)
                if definition and definition.higher_is_better:
                    result_df[adjusted_col] = (result_df[ratio_name] / industry_avg - 1) * 100
                elif definition and definition.higher_is_better is False:
                    # For ratios where lower is better (like debt ratios)
                    result_df[adjusted_col] = (industry_avg / result_df[ratio_name] - 1) * 100
                else:
                    # Neutral - just show difference
                    result_df[adjusted_col] = result_df[ratio_name] - industry_avg
        
        return result_df
    
    def get_ratio_definitions(self) -> Dict[str, RatioDefinition]:
        """Get all ratio definitions."""
        return self.ratio_definitions.copy()
    
    def get_ratios_by_category(self, category: str) -> List[str]:
        """Get list of ratio names for a specific category."""
        return [
            name for name, definition in self.ratio_definitions.items()
            if definition.category.lower() == category.lower()
        ]
    
    def validate_ratio_calculations(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Validate calculated ratios for potential errors or anomalies.
        
        Args:
            df: DataFrame with calculated ratios
            
        Returns:
            Dictionary with validation warnings and errors
        """
        warnings_dict = {'errors': [], 'warnings': [], 'info': []}
        
        if df.empty:
            warnings_dict['errors'].append("No data available for validation")
            return warnings_dict
        
        # Check for extreme values
        for ratio_name, definition in self.ratio_definitions.items():
            if ratio_name not in df.columns:
                continue
            
            values = df[ratio_name].dropna()
            if len(values) == 0:
                continue
            
            # Check for negative values where they shouldn't exist
            if ratio_name in ['current_ratio', 'quick_ratio', 'cash_ratio']:
                negative_count = (values < 0).sum()
                if negative_count > 0:
                    warnings_dict['warnings'].append(
                        f"{ratio_name}: Found {negative_count} negative values"
                    )
            
            # Check for extreme outliers
            if len(values) > 1:
                q99 = values.quantile(0.99)
                q01 = values.quantile(0.01)
                
                extreme_high = (values > q99 * 10).sum()
                extreme_low = (values < q01 * 0.1).sum() if q01 > 0 else 0
                
                if extreme_high > 0:
                    warnings_dict['info'].append(
                        f"{ratio_name}: {extreme_high} extremely high values detected"
                    )
                
                if extreme_low > 0:
                    warnings_dict['info'].append(
                        f"{ratio_name}: {extreme_low} extremely low values detected"
                    )
        
        # Logical consistency checks
        if all(col in df.columns for col in ['current_ratio', 'quick_ratio']):
            inconsistent = (df['quick_ratio'] > df['current_ratio']).sum()
            if inconsistent > 0:
                warnings_dict['warnings'].append(
                    f"Found {inconsistent} cases where quick ratio > current ratio"
                )
        
        return warnings_dict