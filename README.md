# ✈️ Airline Passenger Demand Forecasting System

A comprehensive Python application for analyzing and forecasting airline passenger demand using advanced time series models and Streamlit interface.

## Overview

This system leverages multiple forecasting techniques to predict future airline passenger demand, going beyond the basic scope with:

- **Advanced Time Series Analysis**: Trend, seasonal, and residual decomposition
- **Multiple Forecasting Models**:
  - Holt-Winters Exponential Smoothing
  - ARIMA/SARIMA
  - Facebook Prophet
  - Simple Exponential Smoothing
  - Ensemble Methods
- **Statistical Testing**: Stationarity tests, autocorrelation analysis, residual diagnostics
- **Interactive Dashboard**: Real-time visualization and exploration
- **Operational Insights**: Actionable recommendations for capacity planning
- **Hybrid Approaches**: Combining models for improved accuracy

## Project Structure

```Directory
AirFareAnalysis/
├── data/
│   └── US Airline Flight Routes and Fares 1993-2024.csv
├── data_loader.py          # Data loading and preprocessing
├── forecasting_models.py   # Multiple forecasting model implementations
├── utils.py                # Time series analysis utilities
├── app.py                  # Main Streamlit application
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## Installation & Setup

### 1. Prerequisites

- Python 3.8 or higher
- pip package manager

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Prepare Data

Ensure the dataset is located at: `data/US Airline Flight Routes and Fares 1993-2024.csv`

### 4. Run the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

## Features Guide

### 📊 Tab 1: Overview

- Historical data visualization
- Key statistics and metrics
- Trend analysis
- Growth rate calculations
- Data quality assessment

### 📈 Tab 2: Decomposition & Analysis

- **Time Series Decomposition**: Breaks down the series into:
  - Trend Component
  - Seasonal Component
  - Residual Component
- **Statistical Tests**:
  - ADF Stationarity Test
  - Residual Normality Test
  - Autocorrelation Analysis
- **Component Analysis**:
  - Trend strength and direction
  - Seasonal strength
  - Monthly seasonal factors

### 🔮 Tab 3: Forecasting

- Multiple model forecasting
- Comparison of different approaches
- Forecast value tables
- 6-60 period forecasting horizons
- Model selection and configuration

### 📉 Tab 4: Model Comparison

- Error metrics (MAE, RMSE, MAPE)
- Visual comparison charts
- Model performance ranking
- Best model recommendations

### 💡 Tab 5: Insights & Recommendations

- Model recommendations based on data characteristics
- Operational planning insights
- Capacity planning suggestions
- Improvement strategies:
  - Ensemble methods for combining models
  - Adaptive parameters for seasonal updates
  - Exogenous variable incorporation
  - Deep learning approaches (LSTM, Transformers)
  - Bayesian uncertainty quantification
  - Dynamic regression with external factors

### 📋 Tab 6: Export & Reports

- Download forecast data as CSV
- Export analysis summaries
- Export model metrics
- Generate reports for stakeholders

## Advanced Features

### 1. Ensemble Forecasting

Combines predictions from multiple models with configurable weights for improved accuracy.

### 2. Hybrid Approaches

The system suggests improvements including:

- **Adaptive Ensemble**: Weights that change based on recent performance
- **Multi-scale Forecasting**: Different models for different time horizons
- **Probabilistic Forecasting**: Uncertainty quantification with prediction intervals
- **Causal Analysis**: Incorporating fuel prices, economic indicators, holidays

### 3. Configuration Options

Via the sidebar:

- Time series frequency (Monthly, Quarterly, Yearly)
- Forecast horizon (6-60 periods)
- Seasonal period (customizable)
- Test set size (10-50%)
- Model selection (individual or ensemble)

## Key Metrics Explained

### Error Metrics

- **MAE (Mean Absolute Error)**: Average magnitude of errors
- **RMSE (Root Mean Squared Error)**: Penalizes larger errors more
- **MAPE (Mean Absolute Percentage Error)**: Percentage error relative to actual values

### Time Series Characteristics

- **Trend Strength**: 0-1 scale indicating how strong the trend is
- **Seasonal Strength**: 0-1 scale for seasonality prominence
- **Stationarity**: Whether the series has a constant mean and variance

## Data Requirements

The system expects a CSV file with columns including:

- `year`: Year of the observation
- `month`: Month of the observation
- `passengers`: Number of passengers
- Other optional columns (airline, route, fare, etc.)

The data should ideally have:

- At least 24 observations (2 years) for meaningful analysis
- 60+ observations for reliable seasonal patterns
- Consistent time intervals

## Interpretation Guide

### Trend Analysis

- **Positive Slope**: Increasing demand over time
- **Negative Slope**: Decreasing demand
- **Trend Strength**: How much of the variation is explained by the trend

### Seasonality

- **Strong Seasonality** (>50%): Clear repeating patterns
- **Weak Seasonality** (<50%): Limited repeating patterns
- **Monthly Factors**: Show which months have peak demand

### Model Selection

- **ARIMA**: Best for stationary series without seasonality
- **SARIMA**: For seasonal, stationary series
- **Holt-Winters**: Excellent for data with both trend and seasonality
- **Prophet**: Handles multiple seasonalities and trend changes well
- **Ensemble**: Often provides best results by combining strengths

## Operational Planning Use Cases

### Capacity Planning

- Peak demand forecasts inform aircraft assignment
- Average forecasts guide staffing levels
- Uncertainty bands provide buffer planning

### Revenue Management

- Demand forecasts support pricing strategies
- Seasonal patterns guide service scheduling
- Overbooking policies aligned with demand volatility

### Resource Allocation

- Fuel and maintenance scheduling
- Crew planning based on projected routes
- Ground handling resources

### Risk Management

- Identify demand uncertainty periods
- Plan for demand fluctuations
- Manage revenue volatility

## Hybrid Approach Suggestions

### 1. Adaptive Ensemble

```python
# Implement adaptive weights based on recent model performance
# Update weights weekly/monthly based on forecast accuracy
```

### 2. Exogenous Variables

Incorporate external factors:

- Fuel prices
- Economic indicators (GDP, unemployment)
- Holidays and special events
- Competitor schedules
- Weather patterns

### 3. Deep Learning Integration

- LSTM networks for complex temporal patterns
- Transformer models for attention mechanisms
- Neural Prophet for neural + statistical hybrid

### 4. Multi-level Forecasting

- Airline-level forecasts
- Route-specific forecasts
- Aircraft-type demand
- Cabin-class demand (economy, business, first)

### 5. Probabilistic Forecasting

- Quantile regression for percentile forecasts
- Interval forecasts for risk management
- Probabilistic predictions for scenario planning

## Troubleshooting

### Large Dataset Issues

If the dataset is too large to load:

1. Filter by date range or airline
2. Aggregate to higher frequency (quarterly/yearly)
3. Use sampling for initial exploration

### Model Fitting Errors

- Ensure sufficient data (minimum 24 observations)
- Check for missing values
- Verify data types and format
- Try different seasonal periods

### Memory Issues

- Reduce forecast horizon
- Use quarterly instead of monthly frequency
- Process data in chunks

## Performance Tips

1. **Start Simple**: Fit Holt-Winters first, then expand
2. **Monitor Metrics**: Watch MAE/RMSE for relative performance
3. **Cross-Validation**: Use test set size wisely (20% recommended)
4. **Ensemble Often Wins**: Multiple models often beat single models
5. **Update Regularly**: Refit models with new data monthly

## Future Enhancements

- Real-time data streaming integration
- Automated model selection (AutoML)
- Multivariate forecasting (multiple series)
- Anomaly detection and handling
- Interactive hypothesis testing
- Advanced uncertainty quantification
- Integration with reservation systems
- A/B testing framework for model evaluation

## References

### Key Publications

- Holt-Winters: Holt, C. C. (1957). Forecasting seasonals and trends by exponentially weighted...
- ARIMA: Box, G. E., & Jenkins, G. M. (1970). Time Series Analysis: Forecasting and Control
- Prophet: Taylor, S. J., & Letham, B. (2018). Forecasting at scale

### Tools & Libraries

- [Statsmodels](https://www.statsmodels.org/): Statistical modeling
- [Prophet](https://facebook.github.io/prophet/): Facebook's forecasting tool
- [Scikit-learn](https://scikit-learn.org/): Machine learning
- [Plotly](https://plotly.com/): Interactive visualizations
- [Streamlit](https://streamlit.io/): Web application framework
