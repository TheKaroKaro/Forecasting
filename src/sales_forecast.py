"""
Sales Forecasting Module for Contact Center Sales Teams
Predicts revenue, conversion rates, and pipeline outcomes
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from prophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class SalesForecaster:
    """
    Sales forecasting engine for contact center sales teams
    Handles revenue forecasting, conversion rates, and pipeline analysis
    """
    
    def __init__(self):
        self.models = {}
        self.forecast_results = {}
        self.metrics = {}
    
    def prepare_sales_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare sales data for forecasting
        
        Expected columns:
        - date: Date of sales activity
        - revenue: Revenue generated
        - deals_won: Number of deals closed
        - deals_lost: Number of deals lost
        - pipeline_value: Value of pipeline
        - opportunities: Number of opportunities
        """
        df = data.copy()
        df['ds'] = pd.to_datetime(df['date'])
        df['y'] = df['revenue'].astype(float)
        return df
    
    def forecast_revenue(
        self,
        historical_data: pd.DataFrame,
        periods: int = 30,
        method: str = 'ensemble',
        include_seasonality: bool = True
    ) -> pd.DataFrame:
        """
        Forecast future revenue
        
        Args:
            historical_data: DataFrame with revenue history
            periods: Days to forecast
            method: 'prophet', 'exponential', 'ensemble'
            include_seasonality: Include seasonal patterns
        
        Returns:
            DataFrame with forecasted revenue and confidence intervals
        """
        df = self.prepare_sales_data(historical_data)
        
        if method == 'prophet' or method == 'ensemble':
            # Use Prophet for revenue forecasting
            model = Prophet(
                yearly_seasonality=include_seasonality,
                weekly_seasonality=include_seasonality,
                daily_seasonality=False,
                changepoint_prior_scale=0.05
            )
            
            # Add custom seasonality for month-end effects
            if include_seasonality:
                model.add_seasonality(
                    name='monthly',
                    period=30.5,
                    fourier_order=5
                )
                model.add_seasonality(
                    name='quarterly',
                    period=91.25,
                    fourier_order=3
                )
            
            model.fit(df)
            future = model.make_future_dataframe(periods=periods, freq='D')
            forecast = model.predict(future)
            
            self.models['prophet'] = model
            prophet_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
        else:
            prophet_forecast = None
        
        if method == 'exponential' or method == 'ensemble':
            # Use Exponential Smoothing for revenue
            series = df.set_index('ds')['y']
            try:
                model = ExponentialSmoothing(
                    series,
                    seasonal_periods=7,
                    trend='add',
                    seasonal='add'
                )
                fitted = model.fit()
                exp_forecast = fitted.forecast(periods)
                exp_index = pd.date_range(
                    start=series.index[-1] + timedelta(days=1),
                    periods=periods,
                    freq='D'
                )
                exp_df = pd.DataFrame({
                    'ds': exp_index,
                    'yhat': exp_forecast,
                    'yhat_lower': exp_forecast * 0.85,
                    'yhat_upper': exp_forecast * 1.15
                })
                self.models['exponential'] = fitted
            except:
                exp_df = None
        else:
            exp_df = None
        
        # Ensemble: average both forecasts
        if method == 'ensemble' and prophet_forecast is not None and exp_df is not None:
            # Align the forecasts
            ensemble = prophet_forecast.copy()
            ensemble['yhat'] = (prophet_forecast['yhat'] + exp_df['yhat']) / 2
            ensemble['yhat_lower'] = (prophet_forecast['yhat_lower'] + exp_df['yhat_lower']) / 2
            ensemble['yhat_upper'] = (prophet_forecast['yhat_upper'] + exp_df['yhat_upper']) / 2
            return ensemble
        elif method == 'prophet' and prophet_forecast is not None:
            return prophet_forecast
        elif method == 'exponential' and exp_df is not None:
            return exp_df
        else:
            # Fallback
            return prophet_forecast if prophet_forecast is not None else df[['ds', 'y']]
    
    def forecast_conversion_rate(
        self,
        deals_won: pd.Series,
        deals_total: pd.Series,
        periods: int = 30
    ) -> Dict:
        """
        Forecast conversion rates using moving average and trend analysis
        """
        conversion_rates = deals_won / deals_total
        
        # Simple moving average forecast
        window = min(7, len(conversion_rates))
        ma = conversion_rates.rolling(window=window, min_periods=1).mean()
        
        # Linear trend forecast
        if len(conversion_rates) > 10:
            x = np.arange(len(conversion_rates)).reshape(-1, 1)
            y = conversion_rates.values
            model = LinearRegression()
            model.fit(x, y)
            future_x = np.arange(len(conversion_rates), len(conversion_rates) + periods).reshape(-1, 1)
            trend_forecast = model.predict(future_x)
            trend_forecast = np.clip(trend_forecast, 0, 1)
        else:
            trend_forecast = np.full(periods, ma.iloc[-1])
        
        return {
            'historical_rates': conversion_rates.tolist(),
            'moving_average': ma.tolist(),
            'forecast_rates': trend_forecast.tolist(),
            'average_rate': conversion_rates.mean(),
            'trend_direction': 'increasing' if trend_forecast[-1] > trend_forecast[0] else 'decreasing',
            'confidence': 0.8 if len(conversion_rates) > 20 else 0.5
        }
    
    def pipeline_analysis(
        self,
        pipeline_data: pd.DataFrame,
        stage_weights: Dict = None
    ) -> Dict:
        """
        Analyze sales pipeline and forecast revenue from opportunities
        
        Args:
            pipeline_data: DataFrame with columns ['stage', 'value', 'close_date']
            stage_weights: Dictionary mapping stage names to close probabilities
        
        Returns:
            Pipeline analysis with weighted forecast
        """
        if stage_weights is None:
            stage_weights = {
                'prospecting': 0.1,
                'qualification': 0.2,
                'proposal': 0.4,
                'negotiation': 0.6,
                'closed_won': 1.0
            }
        
        # Calculate weighted pipeline value
        pipeline_data['weighted_value'] = pipeline_data['stage'].map(
            lambda x: stage_weights.get(x, 0.2)
        ) * pipeline_data['value']
        
        total_pipeline = pipeline_data['value'].sum()
        weighted_pipeline = pipeline_data['weighted_value'].sum()
        
        # Forecast by close date
        pipeline_data['close_date'] = pd.to_datetime(pipeline_data['close_date'])
        monthly_forecast = pipeline_data.groupby(
            pipeline_data['close_date'].dt.to_period('M')
        )['weighted_value'].sum()
        
        return {
            'total_pipeline_value': total_pipeline,
            'weighted_pipeline_value': weighted_pipeline,
            'expected_close_rate': weighted_pipeline / total_pipeline if total_pipeline > 0 else 0,
            'monthly_forecast': monthly_forecast.to_dict(),
            'stage_distribution': pipeline_data.groupby('stage')['value'].sum().to_dict(),
            'opportunities_count': len(pipeline_data),
            'top_opportunities': pipeline_data.nlargest(5, 'value')[['stage', 'value']].to_dict('records')
        }
    
    def calculate_sales_metrics(
        self,
        historical_data: pd.DataFrame
    ) -> Dict:
        """
        Calculate key sales performance metrics
        """
        # Calculate daily, weekly, monthly metrics
        df = self.prepare_sales_data(historical_data)
        df['month'] = df['ds'].dt.to_period('M')
        df['week'] = df['ds'].dt.to_period('W')
        
        monthly_revenue = df.groupby('month')['y'].sum()
        weekly_revenue = df.groupby('week')['y'].sum()
        
        # Growth rates
        if len(monthly_revenue) >= 2:
            monthly_growth = monthly_revenue.pct_change().mean()
        else:
            monthly_growth = 0
        
        # Average deal size
        if 'deals_won' in df.columns:
            avg_deal_size = df['y'].sum() / df['deals_won'].sum()
        else:
            avg_deal_size = df['y'].mean()
        
        return {
            'monthly_revenue': monthly_revenue.to_dict(),
            'weekly_revenue': weekly_revenue.to_dict(),
            'average_monthly_growth': monthly_growth,
            'average_deal_size': avg_deal_size,
            'revenue_per_day': df['y'].mean(),
            'total_revenue': df['y'].sum(),
            'days_analyzed': len(df)
        }