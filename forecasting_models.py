"""
Advanced forecasting models for airline passenger demand
Includes: Holt-Winters, ARIMA, Prophet, and Ensemble approaches
"""
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings('ignore')


class ForecastingModels:
    """Collection of forecasting models with evaluation metrics"""
    
    def __init__(self, train_data, test_data=None):
        self.train_data = train_data
        self.test_data = test_data
        self.models = {}
        self.forecasts = {}
        self.metrics = {}
    
    def fit_holt_winters(self, seasonal_periods=12, trend='add', seasonal='add'):
        """
        Fit Holt-Winters seasonal exponential smoothing model
        
        Parameters:
        -----------
        seasonal_periods : int
            Number of periods in a season (12 for monthly data)
        trend : str
            'add' for additive or 'mul' for multiplicative
        seasonal : str
            'add' for additive or 'mul' for multiplicative
        """
        try:
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            
            model = ExponentialSmoothing(
                self.train_data,
                seasonal_periods=seasonal_periods,
                trend=trend,
                seasonal=seasonal,
                initialization_method='estimated'
            )
            fitted_model = model.fit(optimized=True)
            self.models['holt_winters'] = fitted_model
            
            return fitted_model
        except Exception as e:
            print(f"Error fitting Holt-Winters: {e}")
            return None
    
    def fit_arima(self, order=(1, 1, 1)):
        """
        Fit ARIMA model
        
        Parameters:
        -----------
        order : tuple
            (p, d, q) for ARIMA(p,d,q)
        """
        try:
            from statsmodels.tsa.arima.model import ARIMA
            
            model = ARIMA(self.train_data, order=order)
            fitted_model = model.fit()
            self.models['arima'] = fitted_model
            
            return fitted_model
        except Exception as e:
            print(f"Error fitting ARIMA: {e}")
            return None
    
    def fit_prophet(self):
        """Fit Facebook Prophet model"""
        try:
            from prophet import Prophet
            
            # Prepare data for Prophet
            df_prophet = pd.DataFrame({
                'ds': self.train_data.index,
                'y': self.train_data.values
            })
            
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                seasonality_mode='additive',
                interval_width=0.95
            )
            model.fit(df_prophet)
            self.models['prophet'] = model
            
            return model
        except Exception as e:
            print(f"Error fitting Prophet: {e}")
            return None
    
    def fit_exponential_smoothing_simple(self, alpha=0.3):
        """
        Fit simple exponential smoothing
        
        Parameters:
        -----------
        alpha : float
            Smoothing parameter (0 < alpha < 1)
        """
        try:
            from statsmodels.tsa.holtwinters import SimpleExpSmoothing
            
            model = SimpleExpSmoothing(self.train_data)
            fitted_model = model.fit(smoothing_level=alpha, optimized=True)
            self.models['simple_exp'] = fitted_model
            
            return fitted_model
        except Exception as e:
            print(f"Error fitting Simple Exponential Smoothing: {e}")
            return None
    
    def forecast_holt_winters(self, periods=12):
        """Generate Holt-Winters forecast"""
        if 'holt_winters' not in self.models:
            self.fit_holt_winters()
        
        model = self.models['holt_winters']
        forecast = model.get_forecast(steps=periods)
        self.forecasts['holt_winters'] = forecast.predicted_mean
        
        return forecast.predicted_mean
    
    def forecast_arima(self, periods=12):
        """Generate ARIMA forecast"""
        if 'arima' not in self.models:
            self.fit_arima()
        
        model = self.models['arima']
        forecast = model.get_forecast(steps=periods)
        self.forecasts['arima'] = forecast.predicted_mean
        
        return forecast.predicted_mean
    
    def forecast_prophet(self, periods=12):
        """Generate Prophet forecast"""
        if 'prophet' not in self.models:
            self.fit_prophet()
        
        model = self.models['prophet']
        future = model.make_future_dataframe(periods=periods, freq='M')
        forecast_df = model.predict(future)
        
        self.forecasts['prophet'] = forecast_df['yhat'].iloc[-periods:].values
        return self.forecasts['prophet']
    
    def forecast_simple_exp(self, periods=12):
        """Generate Simple Exponential Smoothing forecast"""
        if 'simple_exp' not in self.models:
            self.fit_exponential_smoothing_simple()
        
        model = self.models['simple_exp']
        forecast = model.get_forecast(steps=periods)
        self.forecasts['simple_exp'] = forecast.predicted_mean
        
        return forecast.predicted_mean
    
    def ensemble_forecast(self, periods=12, weights=None):
        """
        Create ensemble forecast combining multiple models
        
        Parameters:
        -----------
        periods : int
            Number of periods to forecast
        weights : dict
            Weights for each model (default: equal weights)
        """
        forecasts_list = []
        model_names = []
        
        # Get forecasts from available models
        if 'holt_winters' in self.models or self.models == {}:
            try:
                hw_forecast = self.forecast_holt_winters(periods)
                forecasts_list.append(hw_forecast.values)
                model_names.append('holt_winters')
            except:
                pass
        
        if 'arima' in self.models:
            try:
                arima_forecast = self.forecast_arima(periods)
                forecasts_list.append(arima_forecast.values)
                model_names.append('arima')
            except:
                pass
        
        if 'prophet' in self.models:
            try:
                prophet_forecast = self.forecast_prophet(periods)
                forecasts_list.append(prophet_forecast)
                model_names.append('prophet')
            except:
                pass
        
        if 'simple_exp' in self.models:
            try:
                simple_forecast = self.forecast_simple_exp(periods)
                forecasts_list.append(simple_forecast.values)
                model_names.append('simple_exp')
            except:
                pass
        
        if not forecasts_list:
            # Fallback - fit HW if nothing available
            hw_forecast = self.forecast_holt_winters(periods)
            forecasts_list.append(hw_forecast.values)
            model_names.append('holt_winters')
        
        # Set default weights
        if weights is None:
            weights = {name: 1.0 / len(model_names) for name in model_names}
        
        # Weighted ensemble
        ensemble = np.zeros(periods)
        for forecast, name in zip(forecasts_list, model_names):
            ensemble += weights.get(name, 1.0 / len(model_names)) * forecast
        
        self.forecasts['ensemble'] = ensemble
        return ensemble
    
    def evaluate_models(self):
        """Evaluate all models on test data if available"""
        if self.test_data is None:
            return None
        
        metrics = {}
        
        for model_name in self.models.keys():
            try:
                if model_name == 'holt_winters':
                    forecast = self.models[model_name].fittedvalues
                elif model_name == 'arima':
                    forecast = self.models[model_name].fittedvalues
                elif model_name == 'prophet':
                    continue  # Skip prophet for this evaluation
                elif model_name == 'simple_exp':
                    forecast = self.models[model_name].fittedvalues
                else:
                    continue
                
                # Align forecast with test data
                common_length = min(len(forecast), len(self.test_data))
                actual = self.test_data.iloc[:common_length]
                pred = forecast.iloc[-common_length:]
                
                metrics[model_name] = {
                    'MAE': mean_absolute_error(actual, pred),
                    'RMSE': np.sqrt(mean_squared_error(actual, pred)),
                    'MAPE': mean_absolute_percentage_error(actual, pred)
                }
            except Exception as e:
                print(f"Error evaluating {model_name}: {e}")
        
        self.metrics = metrics
        return metrics
    
    def get_model_summary(self, model_name):
        """Get summary of a specific model"""
        if model_name not in self.models:
            return None
        
        model = self.models[model_name]
        try:
            return str(model.summary())
        except:
            return str(model)
