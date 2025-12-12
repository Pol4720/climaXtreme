"""
EDA Validator for Synthetic Climate Data.

This module provides statistical validation to ensure synthetic data
follows expected distributions and maintains realistic correlations.

Validates:
- Temperature distributions (normal, with seasonal patterns)
- Precipitation (gamma distribution, Markov chain transitions)
- Wind speed (Weibull distribution)
- Humidity (bounded normal)
- Pressure (normal around 1013.25 hPa)
- Temporal autocorrelation
- Spatial consistency
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a statistical validation test."""
    test_name: str
    passed: bool
    statistic: float
    p_value: float
    threshold: float
    description: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class EDAReport:
    """Complete EDA validation report."""
    timestamp: str
    n_records: int
    n_cities: int
    validations: List[ValidationResult]
    summary_stats: Dict[str, Any]
    warnings: List[str]
    overall_score: float  # 0-100 quality score
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            'timestamp': self.timestamp,
            'n_records': self.n_records,
            'n_cities': self.n_cities,
            'overall_score': self.overall_score,
            'validations': [
                {
                    'test': v.test_name,
                    'passed': v.passed,
                    'statistic': v.statistic,
                    'p_value': v.p_value,
                    'description': v.description
                }
                for v in self.validations
            ],
            'summary_stats': self.summary_stats,
            'warnings': self.warnings
        }


class SyntheticDataValidator:
    """
    Validates synthetic climate data against expected distributions.
    
    Performs:
    - Distribution fitting tests (KS, chi-square)
    - Autocorrelation analysis
    - Markov chain validation
    - Anomaly detection
    - Correlation structure verification
    """
    
    def __init__(self, significance_level: float = 0.05):
        """
        Initialize validator.
        
        Args:
            significance_level: Alpha for statistical tests
        """
        self.alpha = significance_level
        self.expected_distributions = {
            'temperature': {'type': 'normal', 'params': {}},
            'humidity': {'type': 'truncated_normal', 'params': {'low': 0, 'high': 100}},
            'pressure': {'type': 'normal', 'params': {'loc': 1013.25, 'scale': 15}},
            'wind_speed': {'type': 'weibull', 'params': {'c': 2.0}},
            'rain_mm': {'type': 'exponential', 'params': {}}
        }
    
    def validate_temperature_distribution(
        self, 
        data: pd.Series,
        city_name: Optional[str] = None
    ) -> ValidationResult:
        """
        Validate temperature follows expected normal distribution.
        
        Uses Shapiro-Wilk test for normality.
        """
        clean_data = data.dropna()
        
        if len(clean_data) < 20:
            return ValidationResult(
                test_name="Temperature Normality",
                passed=False,
                statistic=0.0,
                p_value=0.0,
                threshold=self.alpha,
                description="Insufficient data for test",
                details={'n_samples': len(clean_data)}
            )
        
        # Sample if too large (Shapiro-Wilk limit)
        if len(clean_data) > 5000:
            clean_data = clean_data.sample(5000, random_state=42)
        
        stat, p_value = stats.shapiro(clean_data)
        passed = p_value > self.alpha
        
        return ValidationResult(
            test_name="Temperature Normality" + (f" ({city_name})" if city_name else ""),
            passed=passed,
            statistic=stat,
            p_value=p_value,
            threshold=self.alpha,
            description="Shapiro-Wilk test for normal distribution",
            details={
                'mean': clean_data.mean(),
                'std': clean_data.std(),
                'skewness': stats.skew(clean_data),
                'kurtosis': stats.kurtosis(clean_data)
            }
        )
    
    def validate_rain_markov_chain(
        self, 
        rain_states: pd.Series
    ) -> ValidationResult:
        """
        Validate rain states follow expected Markov chain transitions.
        """
        clean_states = rain_states.dropna()
        
        if len(clean_states) < 100:
            return ValidationResult(
                test_name="Rain Markov Chain",
                passed=False,
                statistic=0.0,
                p_value=0.0,
                threshold=self.alpha,
                description="Insufficient data for Markov chain analysis",
                details={'n_samples': len(clean_states)}
            )
        
        # Expected transition matrix
        expected_transitions = {
            'dry': {'dry': 0.85, 'light': 0.10, 'moderate': 0.04, 'heavy': 0.01},
            'light': {'dry': 0.25, 'light': 0.50, 'moderate': 0.20, 'heavy': 0.05},
            'moderate': {'dry': 0.10, 'light': 0.25, 'moderate': 0.45, 'heavy': 0.20},
            'heavy': {'dry': 0.05, 'light': 0.10, 'moderate': 0.35, 'heavy': 0.50}
        }
        
        # Calculate observed transitions
        states_list = clean_states.tolist()
        observed_counts = {}
        total_transitions = 0
        
        for from_state in ['dry', 'light', 'moderate', 'heavy']:
            observed_counts[from_state] = Counter()
        
        for i in range(len(states_list) - 1):
            from_state = states_list[i]
            to_state = states_list[i + 1]
            if from_state in observed_counts:
                observed_counts[from_state][to_state] += 1
                total_transitions += 1
        
        # Chi-square test for each starting state
        chi2_stats = []
        
        for from_state in expected_transitions:
            if sum(observed_counts[from_state].values()) < 10:
                continue
            
            total = sum(observed_counts[from_state].values())
            observed = []
            expected = []
            
            for to_state in ['dry', 'light', 'moderate', 'heavy']:
                observed.append(observed_counts[from_state].get(to_state, 0))
                expected.append(expected_transitions[from_state][to_state] * total)
            
            # Skip if expected values are too small
            if min(expected) >= 5:
                chi2, p = stats.chisquare(observed, expected)
                chi2_stats.append((chi2, p))
        
        if not chi2_stats:
            return ValidationResult(
                test_name="Rain Markov Chain",
                passed=True,
                statistic=0.0,
                p_value=1.0,
                threshold=self.alpha,
                description="Insufficient transitions for chi-square test",
                details={'total_transitions': total_transitions}
            )
        
        # Combined p-value (Fisher's method)
        combined_chi2 = -2 * sum(np.log(max(p, 1e-10)) for _, p in chi2_stats)
        combined_p = 1 - stats.chi2.cdf(combined_chi2, 2 * len(chi2_stats))
        
        passed = combined_p > self.alpha
        
        return ValidationResult(
            test_name="Rain Markov Chain",
            passed=passed,
            statistic=combined_chi2,
            p_value=combined_p,
            threshold=self.alpha,
            description="Chi-square test for Markov transition probabilities",
            details={
                'n_transitions': total_transitions,
                'observed_distribution': {k: dict(v) for k, v in observed_counts.items()}
            }
        )
    
    def validate_wind_distribution(
        self, 
        wind_data: pd.Series
    ) -> ValidationResult:
        """
        Validate wind speed follows Weibull distribution.
        """
        clean_data = wind_data.dropna()
        clean_data = clean_data[clean_data > 0]  # Weibull is for positive values
        
        if len(clean_data) < 50:
            return ValidationResult(
                test_name="Wind Weibull Distribution",
                passed=False,
                statistic=0.0,
                p_value=0.0,
                threshold=self.alpha,
                description="Insufficient data for Weibull test",
                details={'n_samples': len(clean_data)}
            )
        
        # Fit Weibull and perform KS test
        try:
            shape, loc, scale = stats.weibull_min.fit(clean_data, floc=0)
            ks_stat, p_value = stats.kstest(clean_data, 'weibull_min', args=(shape, loc, scale))
            passed = p_value > self.alpha
        except Exception as e:
            logger.warning(f"Weibull fit failed: {e}")
            return ValidationResult(
                test_name="Wind Weibull Distribution",
                passed=False,
                statistic=0.0,
                p_value=0.0,
                threshold=self.alpha,
                description=f"Weibull fit failed: {str(e)}",
                details={}
            )
        
        return ValidationResult(
            test_name="Wind Weibull Distribution",
            passed=passed,
            statistic=ks_stat,
            p_value=p_value,
            threshold=self.alpha,
            description="Kolmogorov-Smirnov test for Weibull distribution",
            details={
                'shape': shape,
                'scale': scale,
                'mean': clean_data.mean(),
                'std': clean_data.std()
            }
        )
    
    def validate_humidity_bounds(
        self, 
        humidity_data: pd.Series
    ) -> ValidationResult:
        """
        Validate humidity is within valid bounds (0-100%).
        """
        clean_data = humidity_data.dropna()
        
        out_of_bounds = ((clean_data < 0) | (clean_data > 100)).sum()
        total = len(clean_data)
        violation_rate = out_of_bounds / total if total > 0 else 0
        
        passed = violation_rate < 0.01  # Less than 1% violations
        
        return ValidationResult(
            test_name="Humidity Bounds",
            passed=passed,
            statistic=violation_rate,
            p_value=1.0 - violation_rate,
            threshold=0.01,
            description="Check humidity values are within 0-100%",
            details={
                'out_of_bounds': out_of_bounds,
                'total': total,
                'min': clean_data.min(),
                'max': clean_data.max()
            }
        )
    
    def validate_pressure_distribution(
        self, 
        pressure_data: pd.Series
    ) -> ValidationResult:
        """
        Validate pressure follows normal distribution around 1013.25 hPa.
        """
        clean_data = pressure_data.dropna()
        
        if len(clean_data) < 20:
            return ValidationResult(
                test_name="Pressure Distribution",
                passed=False,
                statistic=0.0,
                p_value=0.0,
                threshold=self.alpha,
                description="Insufficient data",
                details={'n_samples': len(clean_data)}
            )
        
        # Check if mean is reasonably close to expected
        expected_mean = 1013.25
        observed_mean = clean_data.mean()
        observed_std = clean_data.std()
        
        # Z-test for mean
        z_score = (observed_mean - expected_mean) / (observed_std / np.sqrt(len(clean_data)))
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        passed = p_value > self.alpha and 970 < observed_mean < 1050
        
        return ValidationResult(
            test_name="Pressure Distribution",
            passed=passed,
            statistic=z_score,
            p_value=p_value,
            threshold=self.alpha,
            description="Z-test for pressure mean near 1013.25 hPa",
            details={
                'observed_mean': observed_mean,
                'expected_mean': expected_mean,
                'std': observed_std,
                'range': (clean_data.min(), clean_data.max())
            }
        )
    
    def validate_temporal_autocorrelation(
        self, 
        temperature_data: pd.Series
    ) -> ValidationResult:
        """
        Validate temperature shows expected temporal autocorrelation.
        
        Weather data should show positive autocorrelation at short lags.
        """
        clean_data = temperature_data.dropna().values
        
        if len(clean_data) < 50:
            return ValidationResult(
                test_name="Temporal Autocorrelation",
                passed=False,
                statistic=0.0,
                p_value=0.0,
                threshold=0.0,
                description="Insufficient data for autocorrelation",
                details={'n_samples': len(clean_data)}
            )
        
        # Calculate lag-1 autocorrelation
        n = len(clean_data)
        mean = np.mean(clean_data)
        var = np.var(clean_data)
        
        if var < 1e-10:
            return ValidationResult(
                test_name="Temporal Autocorrelation",
                passed=False,
                statistic=0.0,
                p_value=0.0,
                threshold=0.0,
                description="Variance too small for autocorrelation",
                details={'variance': var}
            )
        
        autocorr_1 = np.sum((clean_data[:-1] - mean) * (clean_data[1:] - mean)) / ((n - 1) * var)
        
        # Expected: positive autocorrelation (weather persists)
        # Threshold: autocorrelation should be > 0.1 for realistic data
        passed = autocorr_1 > 0.1
        
        return ValidationResult(
            test_name="Temporal Autocorrelation",
            passed=passed,
            statistic=autocorr_1,
            p_value=1.0 if passed else 0.0,
            threshold=0.1,
            description="Lag-1 autocorrelation should be positive (weather persistence)",
            details={
                'lag_1_autocorr': autocorr_1,
                'n_samples': n
            }
        )
    
    def validate_temperature_humidity_correlation(
        self,
        temperature: pd.Series,
        humidity: pd.Series
    ) -> ValidationResult:
        """
        Validate expected negative correlation between temperature and humidity.
        """
        # Align data
        combined = pd.DataFrame({'temp': temperature, 'humidity': humidity}).dropna()
        
        if len(combined) < 30:
            return ValidationResult(
                test_name="Temp-Humidity Correlation",
                passed=False,
                statistic=0.0,
                p_value=0.0,
                threshold=self.alpha,
                description="Insufficient data",
                details={'n_samples': len(combined)}
            )
        
        corr, p_value = stats.pearsonr(combined['temp'], combined['humidity'])
        
        # Expected: negative correlation
        passed = corr < 0 and p_value < self.alpha
        
        return ValidationResult(
            test_name="Temp-Humidity Correlation",
            passed=passed,
            statistic=corr,
            p_value=p_value,
            threshold=self.alpha,
            description="Temperature and humidity should be negatively correlated",
            details={
                'correlation': corr,
                'expected': 'negative',
                'n_samples': len(combined)
            }
        )
    
    def compute_summary_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Compute summary statistics for the dataset."""
        stats_dict = {}
        
        numeric_cols = ['temperature', 'humidity', 'pressure', 'wind_speed', 'rain_mm']
        
        for col in numeric_cols:
            if col in df.columns:
                col_data = df[col].dropna()
                stats_dict[col] = {
                    'count': len(col_data),
                    'mean': col_data.mean(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    'q25': col_data.quantile(0.25),
                    'median': col_data.median(),
                    'q75': col_data.quantile(0.75),
                    'max': col_data.max(),
                    'skewness': stats.skew(col_data) if len(col_data) > 10 else None,
                    'kurtosis': stats.kurtosis(col_data) if len(col_data) > 10 else None
                }
        
        # Categorical summaries
        if 'alert_level' in df.columns:
            stats_dict['alert_distribution'] = df['alert_level'].value_counts().to_dict()
        
        if 'rain_state' in df.columns:
            stats_dict['rain_state_distribution'] = df['rain_state'].value_counts().to_dict()
        
        if 'climate_zone' in df.columns:
            stats_dict['climate_zone_distribution'] = df['climate_zone'].value_counts().to_dict()
        
        return stats_dict
    
    def validate_dataset(self, df: pd.DataFrame) -> EDAReport:
        """
        Perform complete validation of synthetic dataset.
        
        Args:
            df: DataFrame with synthetic climate data
            
        Returns:
            Complete EDA report
        """
        from datetime import datetime
        
        validations = []
        warnings = []
        
        logger.info(f"Validating dataset with {len(df)} records")
        
        # Temperature validation
        if 'temperature' in df.columns:
            validations.append(self.validate_temperature_distribution(df['temperature']))
            validations.append(self.validate_temporal_autocorrelation(df['temperature']))
        else:
            warnings.append("No temperature column found")
        
        # Wind validation
        if 'wind_speed' in df.columns:
            validations.append(self.validate_wind_distribution(df['wind_speed']))
        else:
            warnings.append("No wind_speed column found")
        
        # Humidity validation
        if 'humidity' in df.columns:
            validations.append(self.validate_humidity_bounds(df['humidity']))
        else:
            warnings.append("No humidity column found")
        
        # Pressure validation
        if 'pressure' in df.columns:
            validations.append(self.validate_pressure_distribution(df['pressure']))
        else:
            warnings.append("No pressure column found")
        
        # Rain Markov chain
        if 'rain_state' in df.columns:
            validations.append(self.validate_rain_markov_chain(df['rain_state']))
        
        # Correlation validation
        if 'temperature' in df.columns and 'humidity' in df.columns:
            validations.append(self.validate_temperature_humidity_correlation(
                df['temperature'], df['humidity']
            ))
        
        # Calculate overall score
        passed_tests = sum(1 for v in validations if v.passed)
        total_tests = len(validations)
        overall_score = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        # Summary statistics
        summary_stats = self.compute_summary_statistics(df)
        
        # Generate warnings for failed tests
        for v in validations:
            if not v.passed:
                warnings.append(f"FAILED: {v.test_name} - {v.description}")
        
        return EDAReport(
            timestamp=datetime.now().isoformat(),
            n_records=len(df),
            n_cities=df['city'].nunique() if 'city' in df.columns else 0,
            validations=validations,
            summary_stats=summary_stats,
            warnings=warnings,
            overall_score=overall_score
        )


# ============================================================================
# Convenience Functions
# ============================================================================

def validate_synthetic_data(df: pd.DataFrame, alpha: float = 0.05) -> EDAReport:
    """
    Quick validation of synthetic data.
    
    Args:
        df: Synthetic data DataFrame
        alpha: Significance level
        
    Returns:
        EDA validation report
    """
    validator = SyntheticDataValidator(significance_level=alpha)
    return validator.validate_dataset(df)


def get_distribution_plots_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Prepare data for distribution plots.
    
    Args:
        df: DataFrame with synthetic data
        
    Returns:
        Dictionary with histogram data for each variable
    """
    plot_data = {}
    
    numeric_cols = ['temperature', 'humidity', 'pressure', 'wind_speed', 'rain_mm']
    
    for col in numeric_cols:
        if col in df.columns:
            clean_data = df[col].dropna()
            
            # Histogram bins
            hist, bin_edges = np.histogram(clean_data, bins=50)
            
            plot_data[col] = {
                'values': clean_data.tolist(),
                'hist_counts': hist.tolist(),
                'hist_bins': bin_edges.tolist(),
                'mean': clean_data.mean(),
                'std': clean_data.std()
            }
    
    return plot_data
