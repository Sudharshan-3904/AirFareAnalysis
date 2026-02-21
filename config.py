"""
Configuration file for airline demand forecasting system
Customize these settings based on your needs
"""

# DATA SETTINGS
DATA_PATH = "data/US Airline Flight Routes and Fares 1993-2024.csv"
CACHE_ENABLED = True
CACHE_DIR = ".streamlit_cache"

# FORECASTING SETTINGS
DEFAULT_FORECAST_PERIODS = 12
MIN_FORECAST_PERIODS = 6
MAX_FORECAST_PERIODS = 60
DEFAULT_TEST_SIZE = 0.20  # 20%

# TIME SERIES SETTINGS
MIN_OBSERVATIONS = 24  # Minimum for meaningful analysis
RECOMMENDED_OBSERVATIONS = 60  # For reliable seasonal patterns
DEFAULT_DECOMPOSITION_PERIOD = 12  # Monthly seasonality

# MODEL PARAMETERS
# Holt-Winters
HW_SEASONAL_PERIODS = 12
HW_TREND = 'add'  # 'add' for additive, 'mul' for multiplicative
HW_SEASONAL = 'add'  # 'add' for additive, 'mul' for multiplicative

# ARIMA
ARIMA_ORDER = (1, 1, 1)  # (p, d, q)
ARIMA_SEASONAL_ORDER = (0, 0, 0, 12)  # (P, D, Q, s) for SARIMA

# Prophet
PROPHET_YEARLY_SEASONALITY = True
PROPHET_WEEKLY_SEASONALITY = False
PROPHET_DAILY_SEASONALITY = False
PROPHET_SEASONALITY_MODE = 'additive'  # 'additive' or 'multiplicative'
PROPHET_SEASONALITY_PRIOR_SCALE = 0.01
PROPHET_CHANGEPOINT_PRIOR_SCALE = 0.05

# Simple Exponential Smoothing
SES_ALPHA = 0.3  # Smoothing parameter

# ENSEMBLE SETTINGS
# Equal weights (1.0 for each model)
ENSEMBLE_WEIGHTS = {
    'holt_winters': 0.33,
    'arima': 0.33,
    'prophet': 0.34,
    'simple_exp': 0.25
}

# Display weights for ensemble
ENSEMBLE_EQUAL_WEIGHTS = True  # If True, ignore weights dict and use equal weights

# VISUALIZATION SETTINGS
PLOT_THEME = 'plotly_white'
PLOT_HEIGHT = 500
DECOMPOSITION_HEIGHT = 800

# COLOR PALETTE
COLORS = {
    'historical': '#1f77b4',
    'trend': '#ff7f0e',
    'seasonal': '#2ca02c',
    'residual': '#d62728',
    'holt_winters': '#ff7f0e',
    'arima': '#2ca02c',
    'prophet': '#d62728',
    'ensemble': '#9467bd',
    'simple_exp': '#8c564b'
}

# ANALYSIS SETTINGS
STATIONARITY_THRESHOLD = 0.05  # p-value threshold for ADF test
SEASONALITY_THRESHOLD = 0.5  # Threshold for strong seasonality (0-1)

# OPERATIONAL PLANNING
PLANNING_HORIZON_MONTHS = 12  # Planning horizon for capacity planning
BUFFER_PERCENTAGE = 0.1  # 10% buffer for capacity planning

# EXPORT SETTINGS
EXPORT_DECIMAL_PLACES = 2
EXPORT_DATE_FORMAT = '%Y-%m-%d'
DEFAULT_EXPORT_FILENAME = 'forecast_{date}.csv'

# LOGGING
LOG_LEVEL = 'INFO'
LOG_FILE = 'forecasting_app.log'

# PERFORMANCE
MAX_DATAFRAME_ROWS = 10000  # Streamlit default
CACHE_TTL = 3600  # Cache time to live in seconds

# ADVANCED FEATURES
ENABLE_PROPHET = True  # Set to False to disable Prophet if dependencies not available
ENABLE_CROSS_VALIDATION = True
CROSS_VALIDATION_FOLDS = 5

# UNCERTAINTY SETTINGS
PREDICTION_INTERVAL_WIDTH = 0.95  # 95% confidence interval
QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.95]  # For quantile forecasts

# API & INTEGRATION
API_ENABLED = False
API_HOST = '0.0.0.0'
API_PORT = 8000

print("Configuration loaded successfully")
