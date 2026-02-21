"""
Example script demonstrating how to use the forecasting system programmatically
This can be run standalone without Streamlit
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_loader import AirlineDataLoader
from forecasting_models import ForecastingModels
from utils import TimeSeriesAnalysis, ForecastingRecommendations

def main():
    """Main example workflow"""
    
    print("\n" + "="*70)
    print("  Airline Passenger Demand Forecasting - Programmatic Example")
    print("="*70 + "\n")
    
    # ========== 1. LOAD DATA ==========
    print(">>> Step 1: Loading Data...")
    data_path = "data/US Airline Flight Routes and Fares 1993-2024.csv"
    loader = AirlineDataLoader(data_path)
    
    try:
        loader.load_data()
        print(f"✓ Data loaded successfully")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return
    
    # ========== 2. CREATE TIME SERIES ==========
    print("\n>>> Step 2: Creating Time Series...")
    try:
        ts_monthly = loader.create_time_series(frequency='M')
        print(f"✓ Time series created with {len(ts_monthly)} observations")
        print(f"  Date range: {ts_monthly.index[0].date()} to {ts_monthly.index[-1].date()}")
        print(f"  Avg passengers: {ts_monthly.mean():,.0f}")
        print(f"  Range: {ts_monthly.min():,.0f} - {ts_monthly.max():,.0f}")
    except Exception as e:
        print(f"✗ Error: {e}")
        return
    
    # ========== 3. ANALYZE TIME SERIES ==========
    print("\n>>> Step 3: Analyzing Time Series...")
    analysis = TimeSeriesAnalysis(ts_monthly)
    
    # Get summary statistics
    summary_stats = analysis.get_summary_statistics()
    print("\nSummary Statistics:")
    for key, value in summary_stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:,.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Decompose series
    try:
        decomposition = analysis.decompose_series(period=12)
        trend_analysis = analysis.get_trend_analysis()
        seasonal_analysis = analysis.get_seasonal_analysis()
        residual_analysis = analysis.get_residual_analysis()
        
        print("\nDecomposition Analysis:")
        print(f"  Trend Strength: {trend_analysis['trend_strength']:.2%}")
        print(f"  Trend Direction: {trend_analysis['direction']}")
        print(f"  Trend Slope: {trend_analysis['slope']:,.0f} passengers/month")
        
        print(f"\nSeasonal Analysis:")
        print(f"  Seasonal Strength: {seasonal_analysis['seasonal_strength']:.2%}")
        print(f"  Strength Level: {seasonal_analysis['strength_interpretation']}")
        
        print(f"\nResidual Analysis:")
        print(f"  Mean: {residual_analysis['mean']:.4f}")
        print(f"  Std Dev: {residual_analysis['std']:.4f}")
        print(f"  Is Normal? {residual_analysis['is_normal']}")
    except Exception as e:
        print(f"✗ Error in analysis: {e}")
    
    # Stationarity test
    try:
        stationarity = analysis.stationarity_test()
        print(f"\nStationarity Test (ADF):")
        print(f"  Test Statistic: {stationarity['test_statistic']:.4f}")
        print(f"  P-Value: {stationarity['p_value']:.4f}")
        print(f"  Is Stationary: {stationarity['is_stationary']}")
    except Exception as e:
        print(f"  Error: {e}")
    
    # ========== 4. SPLIT DATA ==========
    print("\n>>> Step 4: Splitting Data...")
    train_data, test_data = loader.split_train_test(test_size=0.2)
    print(f"✓ Train set: {len(train_data)} observations")
    print(f"  Test set: {len(test_data)} observations")
    
    # ========== 5. FIT FORECASTING MODELS ==========
    print("\n>>> Step 5: Fitting Forecasting Models...")
    fm = ForecastingModels(train_data, test_data)
    
    models_fitted = []
    
    # Fit Holt-Winters
    try:
        print("  Fitting Holt-Winters...")
        fm.fit_holt_winters(seasonal_periods=12)
        models_fitted.append('holt_winters')
        print("    ✓ Holt-Winters fitted")
    except Exception as e:
        print(f"    ✗ Error: {e}")
    
    # Fit ARIMA
    try:
        print("  Fitting ARIMA...")
        fm.fit_arima(order=(1, 1, 1))
        models_fitted.append('arima')
        print("    ✓ ARIMA fitted")
    except Exception as e:
        print(f"    ✗ Error: {e}")
    
    # Fit Simple Exponential Smoothing
    try:
        print("  Fitting Simple Exponential Smoothing...")
        fm.fit_exponential_smoothing_simple()
        models_fitted.append('simple_exp')
        print("    ✓ Simple Exponential Smoothing fitted")
    except Exception as e:
        print(f"    ✗ Error: {e}")
    
    # Fit Prophet
    try:
        print("  Fitting Prophet...")
        fm.fit_prophet()
        models_fitted.append('prophet')
        print("    ✓ Prophet fitted")
    except Exception as e:
        print(f"    ✗ Error (Prophet may need separate installation): {e}")
    
    # ========== 6. GENERATE FORECASTS ==========
    print("\n>>> Step 6: Generating Forecasts (12 periods)...")
    forecast_periods = 12
    
    forecasts = {}
    
    if 'holt_winters' in models_fitted:
        hw_forecast = fm.forecast_holt_winters(periods=forecast_periods)
        forecasts['Holt-Winters'] = hw_forecast
        print(f"  ✓ Holt-Winters: avg forecast = {hw_forecast.mean():,.0f}")
    
    if 'arima' in models_fitted:
        arima_forecast = fm.forecast_arima(periods=forecast_periods)
        forecasts['ARIMA'] = arima_forecast
        print(f"  ✓ ARIMA: avg forecast = {arima_forecast.mean():,.0f}")
    
    if 'simple_exp' in models_fitted:
        ses_forecast = fm.forecast_simple_exp(periods=forecast_periods)
        forecasts['Simple Exp'] = ses_forecast
        print(f"  ✓ Simple Exp: avg forecast = {ses_forecast.mean():,.0f}")
    
    if 'prophet' in models_fitted:
        try:
            prophet_forecast = fm.forecast_prophet(periods=forecast_periods)
            forecasts['Prophet'] = prophet_forecast
            print(f"  ✓ Prophet: avg forecast = {np.mean(prophet_forecast):,.0f}")
        except:
            pass
    
    # Create ensemble if multiple models available
    if len(fm.models) > 1:
        try:
            ensemble_forecast = fm.ensemble_forecast(periods=forecast_periods)
            forecasts['Ensemble'] = ensemble_forecast
            print(f"  ✓ Ensemble: avg forecast = {ensemble_forecast.mean():,.0f}")
        except Exception as e:
            print(f"  ✗ Ensemble error: {e}")
    
    # ========== 7. EVALUATE MODELS ==========
    print("\n>>> Step 7: Evaluating Model Performance...")
    metrics = fm.evaluate_models()
    
    if metrics:
        print("\nModel Comparison (Lower is Better):")
        print(f"{'Model':<25} {'MAE':<15} {'RMSE':<15} {'MAPE':<15}")
        print("-" * 70)
        for model_name, model_metrics in metrics.items():
            mae = model_metrics['MAE']
            rmse = model_metrics['RMSE']
            mape = model_metrics['MAPE']
            print(f"{model_name:<25} {mae:>13,.0f} {rmse:>13,.0f} {mape:>13.2%}")
    
    # ========== 8. GET RECOMMENDATIONS ==========
    print("\n>>> Step 8: Getting Model Recommendations...")
    analysis_results = {
        'trend': trend_analysis,
        'seasonal': seasonal_analysis,
        'stationarity': stationarity
    }
    
    recommender = ForecastingRecommendations(analysis_results)
    recommendations = recommender.get_model_recommendations()
    
    print("\nRecommended Models:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec['model']}")
        print(f"     Reason: {rec['reason']}")
        print(f"     Confidence: {rec['confidence']}\n")
    
    # ========== 9. OPERATIONAL INSIGHTS ==========
    print(">>> Step 9: Operational Planning Insights...")
    if len(forecasts) > 0:
        avg_forecast = np.mean([f if isinstance(f, np.ndarray) else f.values 
                               for f in forecasts.values()], axis=0)
        insights = recommender.get_operational_insights(avg_forecast)
        
        print("\nOperational Insights:")
        for insight in insights:
            print(f"  [{insight['category']}]")
            print(f"    {insight['insight']}")
            print(f"    Impact: {insight['impact']}\n")
    
    # ========== 10. EXPORT RESULTS ==========
    print(">>> Step 10: Creating Forecast Table for Export...")
    
    forecast_dates = pd.date_range(
        start=ts_monthly.index[-1] + pd.Timedelta(days=30),
        periods=forecast_periods,
        freq='MS'
    )
    
    export_df = pd.DataFrame({'Date': forecast_dates})
    for model_name, forecast in forecasts.items():
        if isinstance(forecast, pd.Series):
            export_df[model_name] = forecast.values.round(0).astype(int)
        else:
            export_df[model_name] = np.array(forecast).round(0).astype(int)
    
    print("\nForecast Table:")
    print(export_df.to_string(index=False))
    
    # Save to CSV
    try:
        export_df.to_csv('forecast_example.csv', index=False)
        print("\n✓ Forecast saved to 'forecast_example.csv'")
    except Exception as e:
        print(f"\n✗ Error saving forecast: {e}")
    
    # Summary report
    print("\n" + "="*70)
    print("  SUMMARY REPORT")
    print("="*70)
    print(f"\nData Period: {ts_monthly.index[0].date()} to {ts_monthly.index[-1].date()}")
    print(f"Total Observations: {len(ts_monthly)}")
    print(f"\nDataset Characteristics:")
    print(f"  • Average Demand: {ts_monthly.mean():,.0f} passengers")
    print(f"  • Trend: {trend_analysis['direction']} ({trend_analysis['slope']:+,.0f} passengers/month)")
    print(f"  • Seasonality: {seasonal_analysis['strength_interpretation']}")
    print(f"  • Stationarity: {stationarity['is_stationary']}")
    print(f"\nModels Fitted: {', '.join(models_fitted)}")
    print(f"Models Successfully Trained: {len(fm.models)}")
    print(f"Forecast Horizon: {forecast_periods} months")
    
    if metrics:
        best_model = min(metrics.items(), key=lambda x: x[1]['MAPE'])
        print(f"\nBest Model by MAPE: {best_model[0]} ({best_model[1]['MAPE']:.2%})")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
