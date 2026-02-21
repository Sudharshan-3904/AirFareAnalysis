"""
Data loading and preprocessing module for airline passenger demand analysis
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AirlineDataLoader:
    """Load and preprocess airline passenger data"""
    
    def __init__(self, data_path):
        self.data_path = data_path
        self.df = None
        self.time_series = None
        
    def load_data(self):
        """Load CSV data with optimized parameters for large files"""
        try:
            # Load with optimized parameters
            self.df = pd.read_csv(
                self.data_path,
                dtype={
                    'passengers': 'int32',
                    'year': 'int16',
                    'month': 'int8'
                },
                low_memory=True
            )
            logger.info(f"Data loaded successfully. Shape: {self.df.shape}")
            return self.df
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def explore_data(self):
        """Get basic data exploration info"""
        if self.df is None:
            self.load_data()
        
        return {
            'shape': self.df.shape,
            'columns': self.df.columns.tolist(),
            'dtypes': self.df.dtypes.to_dict(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'head': self.df.head(),
            'description': self.df.describe()
        }
    
    def create_time_series(self, agg_column='passengers', frequency='Q'):
        """
        Create a time series from the data
        
        Parameters:
        -----------
        agg_column : str
            Column to aggregate (default: 'passengers')
        frequency : str
            Aggregation frequency - 'Q' for quarterly, 'Y' for yearly
        """
        if self.df is None:
            self.load_data()
        
        # Check available columns (case-insensitive)
        columns_lower = [col.lower() for col in self.df.columns]
        
        # Try year-quarter combination first
        if 'year' in columns_lower and 'quarter' in columns_lower:
            year_col = [col for col in self.df.columns if col.lower() == 'year'][0]
            quarter_col = [col for col in self.df.columns if col.lower() == 'quarter'][0]
            pass_col = [col for col in self.df.columns if col.lower() == 'passengers'][0]
            
            df_ts = self.df[[year_col, quarter_col, pass_col]].copy()
            df_ts.columns = ['year', 'quarter', 'passengers']
            
            # Create date column from year and quarter
            # Q1 -> month 1, Q2 -> month 4, Q3 -> month 7, Q4 -> month 10
            df_ts['month'] = df_ts['quarter'].map({1: 1, 2: 4, 3: 7, 4: 10})
            df_ts['date'] = pd.to_datetime(
                df_ts[['year', 'month']].assign(day=1)
            )
            
            # Group by date and sum passengers
            ts = df_ts.groupby('date')['passengers'].sum()
            if frequency != 'Q':
                ts = ts.resample(frequency).sum()
        
        # Try year-month combination
        elif 'year' in columns_lower and 'month' in columns_lower:
            year_col = [col for col in self.df.columns if col.lower() == 'year'][0]
            month_col = [col for col in self.df.columns if col.lower() == 'month'][0]
            pass_col = [col for col in self.df.columns if col.lower() == 'passengers'][0]
            
            df_ts = self.df[[year_col, month_col, pass_col]].copy()
            df_ts.columns = ['year', 'month', 'passengers']
            
            # Create date column
            df_ts['date'] = pd.to_datetime(
                df_ts[['year', 'month']].assign(day=1)
            )
            
            # Group by date and sum passengers
            ts = df_ts.set_index('date')['passengers'].resample(frequency).sum()
        else:
            # Look for date or time columns
            date_cols = [col for col in self.df.columns if 'date' in col.lower() or 'time' in col.lower()]
            pass_cols = [col for col in self.df.columns if 'pass' in col.lower()]
            
            if date_cols and pass_cols:
                ts = self.df.set_index(date_cols[0])[pass_cols[0]].resample(frequency).sum()
            else:
                raise ValueError(f"Cannot find date and passenger columns. Available columns: {self.df.columns.tolist()}")
        
        # Remove any NaN values
        ts = ts.dropna()
        self.time_series = ts.sort_index()
        logger.info(f"Time series created. Length: {len(self.time_series)}")
        
        return self.time_series
    
    def get_summary_stats(self):
        """Get summary statistics of the time series"""
        if self.time_series is None:
            self.create_time_series()
        
        ts = self.time_series
        return {
            'min': ts.min(),
            'max': ts.max(),
            'mean': ts.mean(),
            'std': ts.std(),
            'median': ts.median(),
            'trend': (ts.iloc[-1] - ts.iloc[0]) / len(ts),  # Average change per period
            'length': len(ts),
            'date_range': f"{ts.index[0].date()} to {ts.index[-1].date()}"
        }
    
    def split_train_test(self, test_size=0.2):
        """Split time series into train and test sets"""
        if self.time_series is None:
            self.create_time_series()
        
        split_point = int(len(self.time_series) * (1 - test_size))
        train = self.time_series[:split_point]
        test = self.time_series[split_point:]
        
        return train, test
    
    def get_daily_aggregation(self):
        """Get daily aggregated data if available"""
        if self.df is None:
            self.load_data()
        
        return self.create_time_series(frequency='D')
