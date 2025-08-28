"""
Financial data parser module for normalizing SEC EDGAR XBRL data.
Handles mapping of GAAP concepts to unified schema and data cleaning.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import re

logger = logging.getLogger(__name__)

class FinancialDataParser:
    """
    Parses and normalizes financial data from SEC EDGAR API responses.
    Maps various GAAP concepts to a unified schema for consistent analysis.
    """
    
    def __init__(self):
        """Initialize the parser with concept mappings and normalization rules."""
        
        # Comprehensive mapping of GAAP concepts to normalized field names
        self.concept_mapping = {
            # Revenue concepts
            'revenues': [
                'Revenues', 'RevenueFromContractWithCustomerExcludingAssessedTax',
                'SalesRevenueNet', 'RevenueFromContractWithCustomerIncludingAssessedTax',
                'OperatingRevenueTotal', 'TotalRevenues'
            ],
            
            # Cost concepts
            'cost_of_sales': [
                'CostOfGoodsAndServicesSold', 'CostOfRevenue', 'CostOfSales',
                'CostOfGoodsAndServicesSoldExcludingAcquisitionCosts'
            ],
            
            # Operating expenses
            'operating_expenses': [
                'OperatingExpenses', 'CostsAndExpenses', 'OperatingCostsAndExpenses',
                'ResearchAndDevelopmentExpense', 'SellingGeneralAndAdministrativeExpenses'
            ],
            
            # Income concepts
            'operating_income': [
                'OperatingIncomeLoss', 'IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest'
            ],
            
            'net_income': [
                'NetIncomeLoss', 'NetIncomeLossAvailableToCommonStockholdersBasic',
                'ProfitLoss', 'NetIncomeLossAttributableToParent'
            ],
            
            # Balance Sheet - Assets
            'total_assets': ['Assets', 'AssetsTotal'],
            
            'current_assets': [
                'AssetsCurrent', 'CurrentAssets', 'AssetsCurrentTotal'
            ],
            
            'cash_and_equivalents': [
                'CashAndCashEquivalentsAtCarryingValue', 'Cash',
                'CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents'
            ],
            
            'inventory': [
                'InventoryNet', 'Inventory', 'InventoryTotal'
            ],
            
            'property_plant_equipment': [
                'PropertyPlantAndEquipmentNet', 'PropertyPlantAndEquipmentGross',
                'PropertyAndEquipmentNet'
            ],
            
            # Balance Sheet - Liabilities
            'total_liabilities': [
                'Liabilities', 'LiabilitiesTotal', 'LiabilitiesAndStockholdersEquity'
            ],
            
            'current_liabilities': [
                'LiabilitiesCurrent', 'CurrentLiabilities', 'LiabilitiesCurrentTotal'
            ],
            
            'long_term_debt': [
                'LongTermDebt', 'LongTermDebtNoncurrent', 'DebtLongTerm',
                'LongTermDebtAndCapitalLeaseObligations'
            ],
            
            'short_term_debt': [
                'DebtCurrent', 'ShortTermBorrowings', 'CommercialPaper'
            ],
            
            # Balance Sheet - Equity
            'stockholders_equity': [
                'StockholdersEquity', 'StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest',
                'ShareholdersEquity', 'TotalEquity'
            ],
            
            # Cash Flow concepts
            'operating_cash_flow': [
                'NetCashProvidedByUsedInOperatingActivities',
                'CashProvidedByUsedInOperatingActivities'
            ],
            
            'investing_cash_flow': [
                'NetCashProvidedByUsedInInvestingActivities',
                'CashProvidedByUsedInInvestingActivities'
            ],
            
            'financing_cash_flow': [
                'NetCashProvidedByUsedInFinancingActivities',
                'CashProvidedByUsedInFinancingActivities'
            ],
            
            # Per share data
            'shares_outstanding': [
                'CommonStockSharesOutstanding', 'WeightedAverageNumberOfSharesOutstandingBasic',
                'CommonSharesOutstanding'
            ]
        }
        
        # Data validation rules
        self.validation_rules = {
            'revenues': {'min_value': 0, 'max_ratio_to_assets': 5.0},
            'total_assets': {'min_value': 0},
            'stockholders_equity': {'min_ratio_to_assets': -1.0, 'max_ratio_to_assets': 1.0},
            'current_ratio': {'min_value': 0, 'max_value': 20.0}
        }
    
    def parse_company_data(self, company_facts: Dict, ticker: str = None) -> pd.DataFrame:
        """
        Parse and normalize company financial data from SEC API response.
        
        Args:
            company_facts: Raw company facts dictionary from SEC API
            ticker: Optional ticker symbol for identification
            
        Returns:
            Normalized DataFrame with financial metrics
        """
        try:
            if not self._validate_input_data(company_facts):
                logger.warning(f"Invalid input data for {ticker}")
                return pd.DataFrame()
            
            # Extract basic company information
            entity_info = self._extract_entity_info(company_facts, ticker)
            us_gaap_data = company_facts['facts'].get('us-gaap', {})
            
            # Get all available fiscal years
            fiscal_years = self._extract_fiscal_years(us_gaap_data)
            
            if not fiscal_years:
                logger.warning(f"No fiscal years found for {ticker}")
                return pd.DataFrame()
            
            # Process data for each fiscal year
            normalized_records = []
            for year in sorted(fiscal_years, reverse=True)[:5]:  # Last 5 years
                year_data = self._process_fiscal_year(us_gaap_data, year, entity_info)
                if year_data:
                    normalized_records.append(year_data)
            
            if not normalized_records:
                logger.warning(f"No valid records found for {ticker}")
                return pd.DataFrame()
            
            # Convert to DataFrame and perform final validation
            df = pd.DataFrame(normalized_records)
            df = self._apply_data_validation(df)
            df = self._calculate_derived_metrics(df)
            
            logger.info(f"Successfully parsed data for {ticker}: {len(df)} records")
            return df
            
        except Exception as e:
            logger.error(f"Error parsing company data for {ticker}: {str(e)}")
            return pd.DataFrame()
    
    def _validate_input_data(self, company_facts: Dict) -> bool:
        """Validate the input data structure."""
        required_keys = ['facts']
        
        for key in required_keys:
            if key not in company_facts:
                return False
        
        if 'us-gaap' not in company_facts['facts']:
            return False
        
        return True
    
    def _extract_entity_info(self, company_facts: Dict, ticker: str = None) -> Dict:
        """Extract basic entity information."""
        return {
            'ticker': ticker or 'Unknown',
            'entity_name': company_facts.get('entityName', 'Unknown Company'),
            'cik': company_facts.get('cik', 'Unknown'),
            'sic': company_facts.get('sic', None),
            'sicDescription': company_facts.get('sicDescription', None)
        }
    
    def _extract_fiscal_years(self, us_gaap_data: Dict) -> List[int]:
        """Extract all available fiscal years from the data."""
        fiscal_years = set()
        
        for concept_data in us_gaap_data.values():
            if isinstance(concept_data, dict) and 'units' in concept_data:
                usd_data = concept_data['units'].get('USD', [])
                for entry in usd_data:
                    if isinstance(entry, dict) and 'fy' in entry:
                        # Only include annual filings (10-K forms)
                        if entry.get('form') in ['10-K'] and isinstance(entry.get('fy'), int):
                            fiscal_years.add(entry['fy'])
        
        return list(fiscal_years)
    
    def _process_fiscal_year(self, us_gaap_data: Dict, fiscal_year: int, entity_info: Dict) -> Optional[Dict]:
        """Process financial data for a specific fiscal year."""
        try:
            year_record = {
                'ticker': entity_info['ticker'],
                'company_name': entity_info['entity_name'],
                'cik': entity_info['cik'],
                'fiscal_year': fiscal_year,
                'period_end': f"{fiscal_year}-12-31"
            }
            
            # Extract normalized financial concepts
            for normalized_name, concept_list in self.concept_mapping.items():
                value = self._get_concept_value(us_gaap_data, concept_list, fiscal_year)
                year_record[normalized_name] = value
            
            # Only return record if it has essential data
            essential_fields = ['revenues', 'total_assets', 'stockholders_equity']
            has_essential_data = any(
                year_record.get(field) is not None and year_record.get(field) != 0 
                for field in essential_fields
            )
            
            if not has_essential_data:
                logger.debug(f"Insufficient essential data for {entity_info['ticker']} {fiscal_year}")
                return None
            
            return year_record
            
        except Exception as e:
            logger.error(f"Error processing fiscal year {fiscal_year}: {str(e)}")
            return None
    
    def _get_concept_value(self, us_gaap_data: Dict, concept_list: List[str], fiscal_year: int) -> Optional[float]:
        """
        Get the value for a financial concept from the first available mapping.
        
        Args:
            us_gaap_data: US-GAAP data dictionary
            concept_list: List of possible GAAP concept names
            fiscal_year: Target fiscal year
            
        Returns:
            Concept value as float or None if not found
        """
        for concept_name in concept_list:
            if concept_name not in us_gaap_data:
                continue
            
            concept_data = us_gaap_data[concept_name]
            if not isinstance(concept_data, dict) or 'units' not in concept_data:
                continue
            
            usd_data = concept_data['units'].get('USD', [])
            if not isinstance(usd_data, list):
                continue
            
            # Look for the exact fiscal year match with annual filing
            for entry in usd_data:
                if (isinstance(entry, dict) and 
                    entry.get('fy') == fiscal_year and 
                    entry.get('form') in ['10-K'] and
                    'val' in entry):
                    
                    try:
                        value = float(entry['val'])
                        # Basic sanity check for unrealistic values
                        if abs(value) > 1e15:  # Values larger than 1 quadrillion are likely errors
                            logger.warning(f"Unrealistic value detected: {value} for {concept_name}")
                            continue
                        return value
                    except (ValueError, TypeError):
                        continue
        
        return None
    
    def _apply_data_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply validation rules to clean and validate the data.
        
        Args:
            df: DataFrame with financial data
            
        Returns:
            Validated and cleaned DataFrame
        """
        if df.empty:
            return df
        
        validated_df = df.copy()
        
        # Apply field-specific validation rules
        for field, rules in self.validation_rules.items():
            if field not in validated_df.columns:
                continue
            
            # Apply minimum value constraints
            if 'min_value' in rules:
                mask = validated_df[field] < rules['min_value']
                if mask.any():
                    logger.warning(f"Found {mask.sum()} records with {field} below minimum")
                    validated_df.loc[mask, field] = np.nan
            
            # Apply maximum value constraints
            if 'max_value' in rules:
                mask = validated_df[field] > rules['max_value']
                if mask.any():
                    logger.warning(f"Found {mask.sum()} records with {field} above maximum")
                    validated_df.loc[mask, field] = np.nan
        
        # Apply ratio-based validation
        self._apply_ratio_validation(validated_df)
        
        return validated_df
    
    def _apply_ratio_validation(self, df: pd.DataFrame) -> None:
        """Apply ratio-based validation rules."""
        if df.empty:
            return
        
        # Validate stockholders equity to assets ratio
        if 'stockholders_equity' in df.columns and 'total_assets' in df.columns:
            mask = (df['total_assets'] > 0) & (df['stockholders_equity'].notna())
            if mask.any():
                equity_ratio = df.loc[mask, 'stockholders_equity'] / df.loc[mask, 'total_assets']
                
                # Flag unrealistic equity ratios (should be between -100% and 100% of assets)
                unrealistic_mask = (equity_ratio < -1.0) | (equity_ratio > 1.0)
                if unrealistic_mask.any():
                    logger.warning(f"Found {unrealistic_mask.sum()} records with unrealistic equity ratios")
        
        # Validate revenue to assets relationship
        if 'revenues' in df.columns and 'total_assets' in df.columns:
            mask = (df['total_assets'] > 0) & (df['revenues'] > 0)
            if mask.any():
                revenue_ratio = df.loc[mask, 'revenues'] / df.loc[mask, 'total_assets']
                
                # Flag if revenue is more than 5x assets (unusual but possible for some industries)
                high_ratio_mask = revenue_ratio > 5.0
                if high_ratio_mask.any():
                    logger.info(f"Found {high_ratio_mask.sum()} records with high revenue-to-assets ratios")
    
    def _calculate_derived_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate derived metrics from the base financial data.
        
        Args:
            df: DataFrame with normalized financial data
            
        Returns:
            DataFrame with additional derived metrics
        """
        if df.empty:
            return df
        
        result_df = df.copy()
        
        # Calculate total debt (combining short-term and long-term)
        if 'short_term_debt' in df.columns or 'long_term_debt' in df.columns:
            short_debt = df['short_term_debt'].fillna(0)
            long_debt = df['long_term_debt'].fillna(0)
            result_df['total_debt'] = short_debt + long_debt
            
            # Set to NaN if both components were NaN
            mask = df['short_term_debt'].isna() & df['long_term_debt'].isna()
            result_df.loc[mask, 'total_debt'] = np.nan
        
        # Calculate gross profit
        if 'revenues' in df.columns and 'cost_of_sales' in df.columns:
            result_df['gross_profit'] = df['revenues'] - df['cost_of_sales'].fillna(0)
        
        # Calculate working capital
        if 'current_assets' in df.columns and 'current_liabilities' in df.columns:
            result_df['working_capital'] = df['current_assets'] - df['current_liabilities'].fillna(0)
        
        # Calculate book value per share (if shares outstanding is available)
        if 'stockholders_equity' in df.columns and 'shares_outstanding' in df.columns:
            mask = df['shares_outstanding'] > 0
            result_df['book_value_per_share'] = np.where(
                mask,
                df['stockholders_equity'] / df['shares_outstanding'],
                np.nan
            )
        
        return result_df
    
    def parse_multiple_companies(self, companies_data: Dict[str, Dict]) -> Dict[str, pd.DataFrame]:
        """
        Parse financial data for multiple companies.
        
        Args:
            companies_data: Dictionary mapping tickers to company facts
            
        Returns:
            Dictionary mapping tickers to normalized DataFrames
        """
        results = {}
        
        for ticker, company_facts in companies_data.items():
            try:
                parsed_df = self.parse_company_data(company_facts, ticker)
                if not parsed_df.empty:
                    results[ticker] = parsed_df
                else:
                    logger.warning(f"No data parsed for {ticker}")
            except Exception as e:
                logger.error(f"Error parsing data for {ticker}: {str(e)}")
        
        logger.info(f"Successfully parsed {len(results)} out of {len(companies_data)} companies")
        return results
    
    def get_data_coverage_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate a data coverage report for a company's financial data.
        
        Args:
            df: DataFrame with financial data
            
        Returns:
            Dictionary with coverage statistics
        """
        if df.empty:
            return {'total_records': 0, 'coverage': {}}
        
        report = {
            'total_records': len(df),
            'years_covered': sorted(df['fiscal_year'].unique().tolist(), reverse=True),
            'coverage': {}
        }
        
        # Calculate coverage for each financial concept
        financial_fields = [col for col in df.columns if col not in [
            'ticker', 'company_name', 'cik', 'fiscal_year', 'period_end'
        ]]
        
        for field in financial_fields:
            non_null_count = df[field].count()
            coverage_pct = (non_null_count / len(df)) * 100
            
            report['coverage'][field] = {
                'available_records': non_null_count,
                'coverage_percentage': round(coverage_pct, 1),
                'missing_years': df[df[field].isna()]['fiscal_year'].tolist() if coverage_pct < 100 else []
            }
        
        return report
    
    def standardize_industry_metrics(self, df: pd.DataFrame, industry: str = None) -> pd.DataFrame:
        """
        Apply industry-specific standardizations and adjustments.
        
        Args:
            df: DataFrame with financial data
            industry: Industry classification (if available)
            
        Returns:
            DataFrame with industry-adjusted metrics
        """
        if df.empty:
            return df
        
        adjusted_df = df.copy()
        
        # Industry-specific adjustments could be added here
        # For now, we'll implement basic standardizations
        
        # For financial services companies, adjust certain metrics
        if industry and 'financial' in industry.lower():
            # For banks, interest income might be more relevant than traditional revenue
            # This is a placeholder for more sophisticated industry adjustments
            pass
        
        # For retail companies, inventory turnover might be more critical
        elif industry and 'retail' in industry.lower():
            if 'inventory' in adjusted_df.columns and 'cost_of_sales' in adjusted_df.columns:
                # Calculate inventory turnover
                mask = adjusted_df['inventory'] > 0
                adjusted_df['inventory_turnover'] = np.where(
                    mask,
                    adjusted_df['cost_of_sales'] / adjusted_df['inventory'],
                    np.nan
                )
        
        return adjusted_df
    
    def detect_data_anomalies(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect potential anomalies in the financial data.
        
        Args:
            df: DataFrame with financial data
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        if df.empty or len(df) < 2:
            return anomalies
        
        # Sort by fiscal year for time series analysis
        df_sorted = df.sort_values('fiscal_year')
        
        # Check for sudden large changes in key metrics
        key_metrics = ['revenues', 'total_assets', 'stockholders_equity', 'net_income']
        
        for metric in key_metrics:
            if metric not in df_sorted.columns:
                continue
            
            # Calculate year-over-year changes
            values = df_sorted[metric].dropna()
            if len(values) < 2:
                continue
            
            pct_changes = values.pct_change().dropna()
            
            # Flag changes greater than 100% or less than -50%
            large_increases = pct_changes > 1.0
            large_decreases = pct_changes < -0.5
            
            if large_increases.any():
                anomalies.append({
                    'type': 'large_increase',
                    'metric': metric,
                    'years': df_sorted.loc[pct_changes.index[large_increases], 'fiscal_year'].tolist(),
                    'changes': pct_changes[large_increases].tolist()
                })
            
            if large_decreases.any():
                anomalies.append({
                    'type': 'large_decrease',
                    'metric': metric,
                    'years': df_sorted.loc[pct_changes.index[large_decreases], 'fiscal_year'].tolist(),
                    'changes': pct_changes[large_decreases].tolist()
                })
        
        return anomalies
    
    def get_parsing_summary(self, results: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate a summary of the parsing results for multiple companies.
        
        Args:
            results: Dictionary mapping tickers to parsed DataFrames
            
        Returns:
            Summary statistics
        """
        if not results:
            return {'total_companies': 0, 'successful_parses': 0}
        
        summary = {
            'total_companies': len(results),
            'successful_parses': sum(1 for df in results.values() if not df.empty),
            'total_records': sum(len(df) for df in results.values()),
            'years_range': {},
            'common_fields': set()
        }
        
        # Find common fields across all companies
        all_fields = []
        year_ranges = []
        
        for ticker, df in results.items():
            if not df.empty:
                all_fields.append(set(df.columns))
                years = df['fiscal_year'].tolist()
                if years:
                    year_ranges.extend(years)
        
        if all_fields:
            summary['common_fields'] = list(set.intersection(*all_fields))
        
        if year_ranges:
            summary['years_range'] = {
                'earliest': min(year_ranges),
                'latest': max(year_ranges),
                'span': max(year_ranges) - min(year_ranges) + 1
            }
        
        return summary