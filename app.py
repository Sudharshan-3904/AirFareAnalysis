"""
Streamlit application for airline passenger demand forecasting
Comprehensive dashboard with multiple forecasting models and analysis tools
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

from data_loader import AirlineDataLoader
from forecasting_models import ForecastingModels
from utils import TimeSeriesAnalysis, ForecastingRecommendations

# Configure page
st.set_page_config(
    page_title="Airline Passenger Demand Forecasting",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .highlight {
        background-color: #e6f2ff;
        padding: 15px;
        border-left: 4px solid #0066cc;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_data():
    """Load and cache airline data"""
    data_path = "data\\AirLineData.csv"
    loader = AirlineDataLoader(data_path)
    loader.load_data()
    return loader

def create_time_series(loader, frequency):
    """Create time series from cached loader"""
    return loader.create_time_series(frequency=frequency)

# Sidebar Configuration
st.sidebar.title("Configuration")
st.sidebar.markdown("---")

# Data loading
st.sidebar.subheader("Data Settings")
frequency = st.sidebar.selectbox(
    "Time Series Frequency",
    ["Q", "Y"],
    format_func=lambda x: {"Q": "Quarterly", "Y": "Yearly"}[x]
)

# Forecasting settings
st.sidebar.subheader("Forecast Settings")
forecast_periods = st.sidebar.slider("Forecast Periods", min_value=6, max_value=60, value=12, step=6)
decomposition_period = st.sidebar.slider("Seasonal Period", min_value=4, max_value=52, value=12, step=1)

# Model selection
st.sidebar.subheader("Model Selection")
models_to_fit = st.sidebar.multiselect(
    "Select Models to Fit",
    ["Holt-Winters", "ARIMA", "Prophet", "Simple Exponential Smoothing", "Ensemble"],
    default=["Holt-Winters", "Ensemble"]
)

# Test size
test_size = st.sidebar.slider("Test Set Size (%)", min_value=10, max_value=50, value=20, step=5)

st.sidebar.markdown("---")

# Main title and description
st.title("Airline Passenger Demand Forecasting")
st.markdown("""
This application provides comprehensive analysis and forecasting of airline passenger demand using 
advanced time series models including Holt-Winters, ARIMA, Prophet, and ensemble approaches.
""")

# Load data
try:
    with st.spinner("Loading data..."):
        loader = load_data()
        time_series = create_time_series(loader, frequency)
        
        # Get summary statistics
        summary_stats = loader.get_summary_stats()
        
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.info("Make sure the CSV file is in the `data/` directory")
    st.stop()

# TAB STRUCTURE
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Overview",
    "Decomposition & Analysis",
    "Forecasting",
    "Model Comparison",
    "Insights & Recommendations",
    "Export & Reports"
])

# ============================================================================
# TAB 1: OVERVIEW
# ============================================================================
with tab1:
    st.header("Data Overview & Summary Statistics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Periods", summary_stats['length'])
    with col2:
        st.metric("Date Range", summary_stats['date_range'].split(" to ")[0])
    with col3:
        st.metric("Average Passengers", f"{summary_stats['mean']:,.0f}")
    with col4:
        st.metric("Trend (Monthly)", f"{summary_stats['trend']:,.0f}")
    
    # Summary Statistics Table
    st.subheader("Statistical Summary")
    
    stats_df = pd.DataFrame({
        'Metric': ['Count', 'Mean', 'Std Dev', 'Min', 'Max', 'Median', 'Growth Rate'],
        'Value': [
            summary_stats['length'],
            f"{summary_stats['mean']:,.0f}",
            f"{summary_stats['std']:,.0f}",
            f"{summary_stats['min']:,.0f}",
            f"{summary_stats['max']:,.0f}",
            f"{summary_stats['median']:,.0f}",
            f"{summary_stats.get('growth_rate', 0):.2f}%"
        ]
    })
    
    st.dataframe(stats_df, use_container_width=True)
    
    # Time series plot
    st.subheader("Historical Passenger Demand")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=time_series.index,
        y=time_series.values,
        mode='lines',
        name='Passengers',
        line=dict(color='#1f77b4', width=2),
        fill='tozeroy'
    ))
    
    fig.update_layout(
        title="Passenger Demand Over Time",
        xaxis_title="Date",
        yaxis_title="Number of Passengers",
        hovermode='x unified',
        height=450,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# TAB 2: DECOMPOSITION & ANALYSIS
# ============================================================================
with tab2:
    st.header("Time Series Decomposition & Analysis")
    
    # Perform analysis
    with st.spinner("Analyzing time series..."):
        analysis = TimeSeriesAnalysis(time_series)
        decomposition = analysis.decompose_series(period=decomposition_period)
        trend_analysis = analysis.get_trend_analysis()
        seasonal_analysis = analysis.get_seasonal_analysis()
        residual_analysis = analysis.get_residual_analysis()
        stationarity = analysis.stationarity_test()
    
    # Decomposition plots
    st.subheader("Decomposition Components")
    
    col1, col2, col3, col4 = st.columns(4)
    
    if decomposition:
        # Create subplots
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.08)
        
        # Original
        fig.add_trace(
            go.Scatter(x=time_series.index, y=time_series.values, name='Original',
                      line=dict(color='#1f77b4')),
            row=1, col=1
        )
        
        # Trend
        fig.add_trace(
            go.Scatter(x=decomposition.trend.index, y=decomposition.trend.values, name='Trend',
                      line=dict(color='#ff7f0e')),
            row=2, col=1
        )
        
        # Seasonal
        fig.add_trace(
            go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal.values, name='Seasonal',
                      line=dict(color='#2ca02c')),
            row=3, col=1
        )
        
        # Residual
        fig.add_trace(
            go.Scatter(x=decomposition.resid.index, y=decomposition.resid.values, name='Residual',
                      line=dict(color='#d62728')),
            row=4, col=1
        )
        
        fig.update_yaxes(title_text="Value", row=1, col=1)
        fig.update_yaxes(title_text="Trend", row=2, col=1)
        fig.update_yaxes(title_text="Seasonal", row=3, col=1)
        fig.update_yaxes(title_text="Residual", row=4, col=1)
        fig.update_xaxes(title_text="Date", row=4, col=1)
        
        fig.update_layout(height=800, template='plotly_white', hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
    
    # Analysis Results
    st.subheader("Analysis Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### Trend Analysis")
        st.metric("Trend Strength", f"{trend_analysis['trend_strength']:.2%}")
        st.metric("Direction", trend_analysis['direction'])
        st.metric("Monthly Change", f"{trend_analysis['slope']:,.0f}")
    
    with col2:
        st.markdown("#### Seasonal Analysis")
        st.metric("Seasonal Strength", f"{seasonal_analysis['seasonal_strength']:.2%}")
        st.metric("Strength Level", seasonal_analysis['strength_interpretation'])
    
    with col3:
        st.markdown("#### Stationarity Test (ADF)")
        if stationarity:
            st.metric("Test Statistic", f"{stationarity['test_statistic']:.4f}")
            st.metric("P-Value", f"{stationarity['p_value']:.4f}")
            st.metric("Is Stationary?", stationarity['is_stationary'])
    
    # Detailed Analysis Tables
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Monthly Seasonal Factors")
        seasonal_df = pd.DataFrame(
            list(seasonal_analysis['seasonal_factors'].items()),
            columns=['Month', 'Factor']
        )
        seasonal_df['Factor'] = seasonal_df['Factor'].round(0).astype(int)
        st.dataframe(seasonal_df, use_container_width=True)
    
    with col2:
        st.markdown("#### Residual Analysis")
        residual_data = {
            'Mean': f"{residual_analysis['mean']:.4f}",
            'Std Dev': f"{residual_analysis['std']:.4f}",
            'Normality Test': residual_analysis['normality_test'],
            'Is Normal?': residual_analysis['is_normal'],
            'P-Value': f"{residual_analysis['normality_pvalue']:.4f}"
        }
        residual_df = pd.DataFrame(
            list(residual_data.items()),
            columns=['Metric', 'Value']
        )
        st.dataframe(residual_df, use_container_width=True)

# ============================================================================
# TAB 3: FORECASTING
# ============================================================================
with tab3:
    st.header("Time Series Forecasting")
    
    # Split data
    train_data, test_data = loader.split_train_test(test_size=test_size/100)
    
    st.info(f"Training set: {len(train_data)} periods | Test set: {len(test_data)} periods")
    
    # Fit models
    with st.spinner("Fitting models..."):
        fm = ForecastingModels(train_data, test_data)
        
        forecasts = {}
        forecast_errors = {}
        
        if "Holt-Winters" in models_to_fit or "Ensemble" in models_to_fit:
            try:
                fm.fit_holt_winters(seasonal_periods=decomposition_period)
                hw_forecast = fm.forecast_holt_winters(periods=forecast_periods)
                forecasts['Holt-Winters'] = hw_forecast
            except Exception as e:
                st.warning(f"Could not fit Holt-Winters: {e}")
        
        if "ARIMA" in models_to_fit:
            try:
                fm.fit_arima(order=(1, 1, 1))
                arima_forecast = fm.forecast_arima(periods=forecast_periods)
                forecasts['ARIMA'] = arima_forecast
            except Exception as e:
                st.warning(f"Could not fit ARIMA: {e}")
        
        if "Prophet" in models_to_fit:
            try:
                fm.fit_prophet()
                prophet_forecast = fm.forecast_prophet(periods=forecast_periods)
                forecasts['Prophet'] = prophet_forecast
            except Exception as e:
                st.warning(f"Could not fit Prophet: {e}")
        
        if "Simple Exponential Smoothing" in models_to_fit:
            try:
                fm.fit_exponential_smoothing_simple()
                ses_forecast = fm.forecast_simple_exp(periods=forecast_periods)
                forecasts['Simple Exponential Smoothing'] = ses_forecast
            except Exception as e:
                st.warning(f"Could not fit Simple Exponential Smoothing: {e}")
        
        if "Ensemble" in models_to_fit and len(fm.models) > 0:
            try:
                ensemble_forecast = fm.ensemble_forecast(periods=forecast_periods)
                forecasts['Ensemble'] = ensemble_forecast
            except Exception as e:
                st.warning(f"Could not create ensemble: {e}")
    
    # Plot forecasts
    st.subheader("Forecast Comparison")
    
    forecast_dates = pd.date_range(
        start=time_series.index[-1] + pd.Timedelta(days=30),
        periods=forecast_periods,
        freq='MS'
    )
    
    fig = go.Figure()
    
    # Historical data
    fig.add_trace(go.Scatter(
        x=time_series.index,
        y=time_series.values,
        name='Historical',
        line=dict(color='#1f77b4', width=2),
        mode='lines'
    ))
    
    # Add vertical line at train/test split
    fig.add_vline(x=train_data.index[-1], line_dash="dash", line_color="gray")
    
    # Forecasts from different models
    colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    for idx, (model_name, forecast) in enumerate(forecasts.items()):
        if isinstance(forecast, pd.Series):
            y_values = forecast.values
        else:
            y_values = forecast
        
        fig.add_trace(go.Scatter(
            x=forecast_dates,
            y=y_values,
            name=f'{model_name} Forecast',
            line=dict(color=colors[idx % len(colors)], dash='dash'),
            mode='lines+markers'
        ))
    
    fig.update_layout(
        title="Passenger Demand Forecasts",
        xaxis_title="Date",
        yaxis_title="Number of Passengers",
        hovermode='x unified',
        height=500,
        template='plotly_white'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Forecast values table
    st.subheader("Forecast Values")
    
    forecast_table = pd.DataFrame({'Date': forecast_dates})
    
    for model_name, forecast in forecasts.items():
        if isinstance(forecast, pd.Series):
            forecast_table[model_name] = forecast.values.round(0).astype(int)
        else:
            forecast_table[model_name] = forecast.round(0).astype(int)
    
    st.dataframe(forecast_table, use_container_width=True)

# ============================================================================
# TAB 4: MODEL COMPARISON
# ============================================================================
with tab4:
    st.header("Model Performance Comparison")
    
    # Get metrics from fitted models
    with st.spinner("Evaluating models..."):
        metrics_dict = fm.evaluate_models()
    
    if metrics_dict:
        # Create metrics dataframe
        metrics_df = pd.DataFrame([
            {
                'Model': model_name,
                'MAE': f"{metrics['MAE']:,.0f}",
                'RMSE': f"{metrics['RMSE']:,.0f}",
                'MAPE': f"{metrics['MAPE']:.2%}"
            }
            for model_name, metrics in metrics_dict.items()
        ])
        
        st.subheader("Error Metrics (Lower is Better)")
        st.dataframe(metrics_df, use_container_width=True)
        
        # Model comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            mae_data = pd.DataFrame([
                {'Model': k, 'MAE': v['MAE']}
                for k, v in metrics_dict.items()
            ])
            fig_mae = px.bar(mae_data, x='Model', y='MAE', title='Mean Absolute Error (MAE)')
            fig_mae.update_layout(template='plotly_white', height=400)
            st.plotly_chart(fig_mae, use_container_width=True)
        
        with col2:
            mape_data = pd.DataFrame([
                {'Model': k, 'MAPE': v['MAPE'] * 100}
                for k, v in metrics_dict.items()
            ])
            fig_mape = px.bar(mape_data, x='Model', y='MAPE', title='Mean Absolute Percentage Error (MAPE) %')
            fig_mape.update_layout(template='plotly_white', height=400)
            st.plotly_chart(fig_mape, use_container_width=True)
    else:
        st.info("Models need to be evaluated. Please ensure at least one model is fitted.")

# ============================================================================
# TAB 5: INSIGHTS & RECOMMENDATIONS
# ============================================================================
with tab5:
    st.header("Insights & Operational Recommendations")
    
    # Generate recommendations
    with st.spinner("Generating insights..."):
        analysis_results = {
            'trend': trend_analysis,
            'seasonal': seasonal_analysis,
            'stationarity': stationarity
        }
        
        recommender = ForecastingRecommendations(analysis_results)
        model_recs = recommender.get_model_recommendations()
        
        # Average forecast for operational insights
        avg_forecast = np.mean([f if isinstance(f, np.ndarray) else f.values 
                               for f in forecasts.values()], axis=0)
        operational_insights = recommender.get_operational_insights(avg_forecast)
    
    # Model Recommendations
    st.subheader("Recommended Forecasting Models")
    for rec in model_recs:
        with st.container():
            col1, col2, col3 = st.columns([2, 2, 1])
            with col1:
                st.write(f"**{rec['model']}**")
            with col2:
                st.write(rec['reason'])
            with col3:
                st.write(f"Confidence: {rec['confidence']}")
        st.divider()
    
    # Operational Insights
    st.subheader("Operational Planning Insights")
    
    for insight in operational_insights:
        with st.container():
            col1, col2, col3 = st.columns([2, 3, 1])
            with col1:
                st.write(f"**{insight['category']}**")
            with col2:
                st.write(insight['insight'])
            with col3:
                st.write(f"Impact: {insight['impact']}")
        st.divider()
    
    # Key Findings
    st.subheader("Key Findings Summary")
    
    findings = f"""
    - **Data Period**: {summary_stats['date_range']}
    - **Total Observations**: {summary_stats['length']}
    - **Trend**: {trend_analysis['direction']} at {abs(trend_analysis['slope']):,.0f} passengers/month
    - **Seasonality**: {seasonal_analysis['strength_interpretation']} (Strength: {seasonal_analysis['seasonal_strength']:.1%})
    - **Stationarity**: {stationarity['is_stationary']}
    - **Average Passengers**: {summary_stats['mean']:,.0f}
    - **Demand Range**: {summary_stats['min']:,.0f} to {summary_stats['max']:,.0f}
    """
    
    st.markdown(findings)
    
    # Improvement Suggestions
    st.subheader("Suggested Improvements for Hybrid Approaches")
    
    improvement_suggestions = """
    1. **Ensemble Methods**: Combine multiple models to reduce individual model biases
    2. **Adaptive Parameters**: Update model parameters seasonally to capture changing patterns
    3. **Exogenous Variables**: Include fuel prices, economic indicators, or holiday calendars
    4. **Deep Learning**: Consider LSTM or Transformer models for longer-term dependencies
    5. **Bayesian Methods**: Use probabilistic approaches for uncertainty quantification
    6. **Dynamic Regression**: Capture relationships between passengers and external factors
    7. **Multi-level Forecasting**: Disaggregate by route/airline for more granular forecasts
    """
    
    st.markdown(improvement_suggestions)

# ============================================================================
# TAB 6: EXPORT & REPORTS
# ============================================================================
with tab6:
    st.header("Export & Reports")
    
    st.subheader("Download Forecast Data")
    
    # Create downloadable forecast file
    export_df = pd.DataFrame({
        'Date': forecast_dates,
    })
    
    for model_name, forecast in forecasts.items():
        if isinstance(forecast, pd.Series):
            export_df[model_name] = forecast.values.round(2)
        else:
            export_df[model_name] = np.array(forecast).round(2)
    
    csv = export_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Forecast as CSV",
        data=csv,
        file_name="airline_forecast.csv",
        mime="text/csv"
    )
    
    st.subheader("Analysis Summary Export")
    
    # Create analysis report
    report_data = {
        'Analysis Date': pd.Timestamp.now(),
        'Data Period': summary_stats['date_range'],
        'Total Observations': summary_stats['length'],
        'Forecast Horizon': f"{forecast_periods} periods",
        'Average Passengers': f"{summary_stats['mean']:,.0f}",
        'Trend Direction': trend_analysis['direction'],
        'Trend Strength': f"{trend_analysis['trend_strength']:.2%}",
        'Seasonal Strength': f"{seasonal_analysis['seasonal_strength']:.2%}",
        'Is Stationary': stationarity['is_stationary'],
        'Models Fitted': ', '.join(forecasts.keys()),
        'Test Set Size': f"{test_size}%"
    }
    
    report_df = pd.DataFrame(list(report_data.items()), columns=['Parameter', 'Value'])
    
    csv_report = report_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Analysis Summary as CSV",
        data=csv_report,
        file_name="analysis_summary.csv",
        mime="text/csv"
    )
    
    # Display summary
    st.subheader("Quick Summary")
    st.dataframe(report_df, use_container_width=True)
    
    # Export metrics
    if metrics_dict:
        st.subheader("Model Metrics Export")
        metrics_export_df = pd.DataFrame([
            {
                'Model': model_name,
                'MAE': metrics['MAE'],
                'RMSE': metrics['RMSE'],
                'MAPE': metrics['MAPE']
            }
            for model_name, metrics in metrics_dict.items()
        ])
        
        csv_metrics = metrics_export_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Model Metrics as CSV",
            data=csv_metrics,
            file_name="model_metrics.csv",
            mime="text/csv"
        )
        
        st.dataframe(metrics_export_df, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Airline Passenger Demand Forecasting System | Powered by Streamlit & Advanced Time Series Models</p>
    <p>Data: US Airline Flight Routes and Fares (1993-2024)</p>
</div>
""", unsafe_allow_html=True)
