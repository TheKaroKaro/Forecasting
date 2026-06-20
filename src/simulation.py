"""
Shift and SLA Simulation Engine
Combines forecasting with Erlang calculations
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from .erlang_engine import ErlangEngine
from .forecasting import VolumeForecaster

class SimulationEngine:
    """
    Comprehensive simulation engine for workforce planning
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.erlang = ErlangEngine()
        self.forecaster = None
        self.results = {}
    
    def set_forecaster(self, forecaster: VolumeForecaster):
        """Set the volume forecaster for simulations"""
        self.forecaster = forecaster
    
    def run_shift_simulation(
        self,
        num_agents: int,
        shift_start: datetime,
        shift_end: datetime,
        interval_minutes: int = 30,
        target_service_level: float = 80,
        target_answer_time: float = 20  # seconds
    ) -> Dict:
        """
        Run shift simulation to determine service levels
        
        Args:
            num_agents: Number of agents available
            shift_start: Start of shift
            shift_end: End of shift
            interval_minutes: Granularity of simulation
            target_service_level: Target SLA percentage
            target_answer_time: Target answer time in seconds
        
        Returns:
            Dictionary with simulation results
        """
        # Generate intervals
        intervals = pd.date_range(start=shift_start, end=shift_end, freq=f'{interval_minutes}T')
        
        # Get volume forecast for the period
        if self.forecaster:
            forecast = self.forecaster.forecast_volume(periods=len(intervals))
        else:
            # If no forecaster, use average from config
            avg_volume = self.config.get('default_volume_per_interval', 50)
            forecast = pd.DataFrame({
                'ds': intervals,
                'yhat': [avg_volume] * len(intervals),
                'yhat_lower': [avg_volume * 0.8] * len(intervals),
                'yhat_upper': [avg_volume * 1.2] * len(intervals)
            })
        
        # Calculate Erlang metrics for each interval
        results = []
        avg_handle_time = self.config.get('avg_handle_time_minutes', 5) / 60  # Convert to hours
        
        for i, row in forecast.iterrows():
            call_rate = row['yhat'] / 60  # Per hour
            metrics = self.erlang.calculate_service_level(
                num_agents,
                call_rate,
                avg_handle_time,
                target_answer_time / 3600  # Convert to hours
            )
            
            results.append({
                'timestamp': row['ds'],
                'forecast_volume': row['yhat'],
                'service_level': metrics['service_level'],
                'asa': metrics['average_speed_answer'],
                'occupancy': metrics['occupancy'],
                'prob_wait': metrics['probability_wait']
            })
        
        # Calculate overall metrics
        df_results = pd.DataFrame(results)
        overall = {
            'average_service_level': df_results['service_level'].mean(),
            'minimum_service_level': df_results['service_level'].min(),
            'average_asa': df_results['asa'].mean(),
            'average_occupancy': df_results['occupancy'].mean(),
            'service_level_met': (df_results['service_level'] >= target_service_level).mean() * 100,
            'intervals': results
        }
        
        self.results['shift_simulation'] = overall
        return overall
    
    def calculate_staffing_requirements(
        self,
        forecast_volume: pd.DataFrame,
        target_service_level: float = 80,
        target_answer_time: float = 20,
        avg_handle_time_minutes: float = 5
    ) -> Dict:
        """
        Calculate required staff for each interval based on forecast
        
        Returns:
            Dictionary with staffing recommendations
        """
        avg_handle_time = avg_handle_time_minutes / 60  # Convert to hours
        target_answer_time_hours = target_answer_time / 3600  # Convert to hours
        
        staffing_results = []
        
        for _, row in forecast_volume.iterrows():
            call_rate = row['yhat'] / 60  # Per hour
            
            # Find minimum agents needed
            agents_needed, metrics = self.erlang.calculate_agents_needed(
                call_rate,
                avg_handle_time,
                target_service_level,
                target_answer_time_hours
            )
            
            staffing_results.append({
                'timestamp': row['ds'],
                'forecast_volume': row['yhat'],
                'agents_needed': agents_needed,
                'predicted_service_level': metrics.get('service_level', 0),
                'predicted_asa': metrics.get('average_speed_answer', float('inf')),
                'occupancy': metrics.get('occupancy', 0)
            })
        
        df_results = pd.DataFrame(staffing_results)
        
        return {
            'interval_requirements': staffing_results,
            'total_agents_needed': df_results['agents_needed'].max(),
            'average_agents_needed': df_results['agents_needed'].mean(),
            'peak_time_requirements': df_results.nlargest(3, 'forecast_volume').to_dict('records')
        }
    
    def run_sla_scenario(
        self,
        num_agents_range: List[int],
        forecast_volume: pd.DataFrame,
        avg_handle_time_minutes: float = 5
    ) -> pd.DataFrame:
        """
        Run SLA scenarios for different agent counts
        
        Returns:
            DataFrame showing SLA for each agent count scenario
        """
        scenarios = []
        avg_handle_time = avg_handle_time_minutes / 60
        target_answer_time = self.config.get('target_answer_time', 20) / 3600
        
        total_volume = forecast_volume['yhat'].sum()
        avg_call_rate = total_volume / (len(forecast_volume) * 60)  # Average calls per hour
        
        for agents in num_agents_range:
            metrics = self.erlang.calculate_service_level(
                agents,
                avg_call_rate,
                avg_handle_time,
                target_answer_time
            )
            
            scenarios.append({
                'num_agents': agents,
                'service_level': metrics['service_level'],
                'asa_seconds': metrics['average_speed_answer'],
                'occupancy': metrics['occupancy'],
                'prob_wait': metrics['probability_wait']
            })
        
        return pd.DataFrame(scenarios)
    
    def analyze_abandonment(
        self,
        num_agents: int,
        forecast_volume: pd.DataFrame,
        avg_handle_time_minutes: float = 5,
        avg_patience_time: float = 60  # seconds
    ) -> Dict:
        """
        Analyze abandonment rates based on staffing and forecast
        """
        avg_handle_time = avg_handle_time_minutes / 60
        avg_patience_time_hours = avg_patience_time / 3600
        
        total_volume = forecast_volume['yhat'].sum()
        avg_call_rate = total_volume / (len(forecast_volume) * 60)
        
        results = self.erlang.simulate_abandonment(
            num_agents,
            avg_call_rate,
            avg_handle_time,
            avg_patience_time_hours
        )
        
        # Add context about volume
        results['total_forecast_volume'] = total_volume
        results['average_call_rate'] = avg_call_rate
        results['est_abandoned_calls'] = int(total_volume * (results['abandonment_rate'] / 100))
        
        return results