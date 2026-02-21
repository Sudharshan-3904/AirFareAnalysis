"""
Analysis utilities for time series decomposition and diagnostics
"""
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class TimeSeriesAnalysis:
    """Time series analysis and decomposition utilities"""
    
    def __init__(self, time_series):
        self.ts = time_series
        self.decomposition = None
        self.acf_result = None
        self.pacf_result = None
    
    def decompose_series(self, model='additive', period=12):
        """
        Decompose time series into trend, seasonal, and residual components
        
        Parameters:
        -----------
        model : str
            'additive' or 'multiplicative'
        period : int
            Seasonal period (12 for monthly data)
        """
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            
            self.decomposition = seasonal_decompose(
                self.ts,
                model=model,
                period=period
            )
            return self.decomposition
        except Exception as e:
            print(f"Error decomposing series: {e}")
            return None
    
    def get_trend_analysis(self):
        """Analyze trend component"""
        if self.decomposition is None:
            self.decompose_series()
        
        if self.decomposition is None:
            return {
                'trend_strength': 0,
                'slope': 0,
                'direction': 'Unknown',
                'trend_component': self.ts,
                'monthly_change': 0
            }
        
        trend = self.decomposition.trend.dropna()
        
        # Calculate trend strength
        detrended = self.ts - trend
        trend_strength = 1 - (np.var(detrended) / np.var(self.ts)) if np.var(self.ts) > 0 else 0
        
        # Linear regression for trend slope
        x = np.arange(len(trend))
        y = trend.values
        slope, intercept = np.polyfit(x, y, 1)
        
        return {
            'trend_strength': trend_strength,
            'slope': slope,
            'direction': 'Increasing' if slope > 0 else 'Decreasing',
            'trend_component': trend,
            'monthly_change': slope  # Average monthly change
        }
    
    def get_seasonal_analysis(self):
        """Analyze seasonal component"""
        if self.decomposition is None:
            self.decompose_series()
        
        if self.decomposition is None:
            return {
                'seasonal_strength': 0,
                'seasonal_factors': {},
                'seasonal_component': self.ts,
                'peak_months': [],
                'strength_interpretation': 'Unknown'
            }
        
        seasonal = self.decomposition.seasonal.dropna()
        
        # Calculate seasonality strength
        seasonal_strength = 1 - (np.var(self.ts - seasonal) / np.var(self.ts)) if np.var(self.ts) > 0 else 0
        
        # Monthly seasonality factors
        seasonal_factors = {}
        for month in range(1, 13):
            month_data = self.ts[self.ts.index.month == month]
            if len(month_data) > 0:
                seasonal_factors[f'Month_{month}'] = month_data.mean()
        
        return {
            'seasonal_strength': seasonal_strength,
            'seasonal_component': seasonal,
            'seasonal_factors': seasonal_factors,
            'strength_interpretation': 'Strong' if seasonal_strength > 0.5 else 'Weak'
        }
    
    def get_residual_analysis(self):
        """Analyze residual component"""
        if self.decomposition is None:
            self.decompose_series()
        
        if self.decomposition is None:
            return {
                'mean': 0,
                'std': 0,
                'residuals': self.ts,
                'normality_test': 'Unknown',
                'normality_stat': 0,
                'normality_pvalue': 0,
                'is_normal': 'Unknown'
            }
        
        residual = self.decomposition.resid.dropna()
        
        # Statistical tests
        mean_residual = residual.mean()
        std_residual = residual.std()
        
        # Normality test (Shapiro-Wilk if enough data)
        if len(residual) > 5000:
            # For large samples, use Kolmogorov-Smirnov
            ks_stat, ks_pvalue = stats.kstest(residual, 'norm', args=(mean_residual, std_residual))
            normality_test = 'Kolmogorov-Smirnov'
            normality_stat = ks_stat
            normality_pvalue = ks_pvalue
        else:
            shapiro_stat, shapiro_pvalue = stats.shapiro(residual)
            normality_test = 'Shapiro-Wilk'
            normality_stat = shapiro_stat
            normality_pvalue = shapiro_pvalue
        
        return {
            'mean': mean_residual,
            'std': std_residual,
            'residuals': residual,
            'normality_test': normality_test,
            'normality_stat': normality_stat,
            'normality_pvalue': normality_pvalue,
            'is_normal': 'Yes' if normality_pvalue > 0.05 else 'No'
        }
    
    def stationarity_test(self):
        """Perform ADF (Augmented Dickey-Fuller) test for stationarity"""
        try:
            from statsmodels.tsa.stattools import adfuller
            
            result = adfuller(self.ts.dropna(), autolag='AIC', regression='c')
            
            return {
                'test_statistic': result[0],
                'p_value': result[1],
                'lags_used': result[2],
                'observations': result[3],
                'is_stationary': 'Yes' if result[1] < 0.05 else 'No',
                'critical_values': result[4],
                'interpretation': f"{'Series is stationary' if result[1] < 0.05 else 'Series is non-stationary'} (p={result[1]:.4f})"
            }
        except Exception as e:
            print(f"Error performing ADF test: {e}")
            return None
    
    def autocorrelation_analysis(self, nlags=40):
        """Analyze autocorrelation"""
        try:
            from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
            from statsmodels.tsa.stattools import acf, pacf
            
            acf_values = acf(self.ts.dropna(), nlags=nlags)
            pacf_values = pacf(self.ts.dropna(), nlags=nlags)
            
            return {
                'acf': acf_values,
                'pacf': pacf_values,
                'significant_lags': np.where(np.abs(acf_values[1:]) > 1.96/np.sqrt(len(self.ts)))[0] + 1
            }
        except Exception as e:
            print(f"Error analyzing autocorrelation: {e}")
            return None
    
    def get_summary_statistics(self):
        """Get comprehensive summary statistics"""
        return {
            'count': len(self.ts),
            'mean': self.ts.mean(),
            'median': self.ts.median(),
            'std_dev': self.ts.std(),
            'min': self.ts.min(),
            'max': self.ts.max(),
            'range': self.ts.max() - self.ts.min(),
            'coefficient_of_variation': (self.ts.std() / self.ts.mean()) * 100 if self.ts.mean() != 0 else 0,
            'skewness': stats.skew(self.ts),
            'kurtosis': stats.kurtosis(self.ts),
            'growth_rate': ((self.ts.iloc[-1] - self.ts.iloc[0]) / self.ts.iloc[0]) * 100 if self.ts.iloc[0] != 0 else 0
        }
    
    def get_volatility_analysis(self, window=12):
        """Analyze volatility/variability over time"""
        rolling_std = self.ts.rolling(window=window).std()
        rolling_mean = self.ts.rolling(window=window).mean()
        coefficient_of_variation = (rolling_std / rolling_mean) * 100
        
        return {
            'rolling_std': rolling_std,
            'rolling_mean': rolling_mean,
            'rolling_cv': coefficient_of_variation,
            'avg_volatility': rolling_std.mean(),
            'max_volatility': rolling_std.max(),
            'min_volatility': rolling_std.min()
        }


class ForecastingRecommendations:
    """Generate recommendations based on analysis"""
    
    def __init__(self, analysis_results):
        self.analysis = analysis_results
    
    def get_model_recommendations(self):
        """Get recommended models based on analysis"""
        recommendations = []
        
        # Check stationarity
        if self.analysis.get('stationarity'):
            if self.analysis['stationarity'].get('is_stationary') == 'Yes':
                recommendations.append(({
                    'model': 'ARIMA',
                    'reason': 'Series is stationary, ARIMA is suitable',
                    'confidence': 'High'
                }))
            else:
                recommendations.append(({
                    'model': 'Differencing + ARIMA',
                    'reason': 'Series needs differencing for stationarity',
                    'confidence': 'High'
                }))
        
        # Check seasonality
        if self.analysis.get('seasonal'):
            if self.analysis['seasonal'].get('seasonal_strength', 0) > 0.5:
                recommendations.append(({
                    'model': 'SARIMA or Holt-Winters',
                    'reason': 'Strong seasonality detected',
                    'confidence': 'High'
                }))
        
        # Check trend
        if self.analysis.get('trend'):
            if abs(self.analysis['trend'].get('slope', 0)) > 0:
                recommendations.append(({
                    'model': 'Holt-Winters (with trend)',
                    'reason': 'Clear trend detected in data',
                    'confidence': 'High'
                }))
        
        # General recommendations
        recommendations.append(({
            'model': 'Prophet',
            'reason': 'Good for capturing multiple seasonalities and trend changes',
            'confidence': 'Medium'
        }))
        
        recommendations.append(({
            'model': 'Ensemble Methods',
            'reason': 'Combining multiple models often improves forecast accuracy',
            'confidence': 'High'
        }))
        
        return recommendations
    
    def get_operational_insights(self, forecast_data):
        """Generate operational insights from forecast"""
        insights = []
        
        # Capacity planning
        mean_forecast = np.mean(forecast_data)
        std_forecast = np.std(forecast_data)
        peak_forecast = np.max(forecast_data)
        
        insights.append({
            'category': 'Capacity Planning',
            'insight': f'Plan for peak demand of {peak_forecast:.0f} passengers',
            'impact': 'High'
        })
        
        insights.append({
            'category': 'Resource Allocation',
            'insight': f'Average expected demand: {mean_forecast:.0f} passengers (±{std_forecast:.0f})',
            'impact': 'High'
        })
        
        # Seasonal patterns
        if self.analysis.get('seasonal'):
            seasonal_factors = self.analysis['seasonal'].get('seasonal_factors', {})
            if seasonal_factors:
                peak_month = max(seasonal_factors, key=seasonal_factors.get)
                insights.append({
                    'category': 'Seasonal Planning',
                    'insight': f'Peak season identified: {peak_month}',
                    'impact': 'High'
                })
        
        return insights
