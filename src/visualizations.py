"""
Advanced Visualization Module for WFM Tool
Creates interactive charts for all metrics and scenarios
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple

class WFMLVisualizer:
    """
    Creates all visualizations for the WFM tool
    Focuses on interactive, actionable charts
    """
    
    @staticmethod
    def create_volume_forecast_chart(
        historical: pd.DataFrame,
        forecast: pd.DataFrame,
        title: str = "Volume Forecast"
    ) -> go.Figure:
        """
        Create a volume forecast chart with historical data and confidence intervals
        """
        fig = go.Figure()
        
        # Historical data
        if historical is not None and not historical.empty:
            fig.add_trace(go.Scatter(
                x=historical['ds'],
                y=historical['y'],
                mode='lines',
                name='Historical',
                line=dict(color='blue', width=2)
            ))
        
        # Forecast with confidence interval
        if forecast is not None and not forecast.empty:
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_upper'],
                mode='lines',
                name='Upper Bound',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_lower'],
                mode='lines',
                name='Confidence Interval',
                fill='tonexty',
                fillcolor='rgba(68, 68, 204, 0.2)',
                line=dict(width=0),
                showlegend=True
            ))
            
            # Forecast line
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat'],
                mode='lines+markers',
                name='Forecast',
                line=dict(color='red', width=3, dash='dash')
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date/Time',
            yaxis_title='Volume',
            hovermode='x unified',
            template='plotly_white',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    @staticmethod
    def create_staffing_requirements_chart(
        forecast: pd.DataFrame,
        staffing_results: Dict,
        title: str = "Staffing Requirements by Interval"
    ) -> go.Figure:
        """
        Create a chart showing required agents per interval
        """
        intervals = staffing_results.get('interval_requirements', [])
        if not intervals:
            return go.Figure()
        
        df = pd.DataFrame(intervals)
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Bar chart: Forecast volume
        fig.add_trace(
            go.Bar(
                x=df['timestamp'],
                y=df['forecast_volume'],
                name='Forecast Volume',
                marker_color='lightblue',
                opacity=0.6
            ),
            secondary_y=False
        )
        
        # Line chart: Agents needed
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['agents_needed'],
                name='Agents Needed',
                mode='lines+markers',
                line=dict(color='red', width=3),
                marker=dict(size=8)
            ),
            secondary_y=True
        )
        
        # Line chart: Predicted SLA
        fig.add_trace(
            go.Scatter(
                x=df['timestamp'],
                y=df['predicted_service_level'],
                name='Predicted SLA %',
                mode='lines',
                line=dict(color='green', width=2, dash='dot'),
                yaxis='y3'
            ),
            secondary_y=True
        )
        
        fig.update_layout(
            title=title,
            xaxis_title='Time Interval',
            hovermode='x unified',
            template='plotly_white',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        fig.update_yaxes(title_text="Volume", secondary_y=False)
        fig.update_yaxes(title_text="Agents / SLA %", secondary_y=True)
        
        return fig
    
    @staticmethod
    def create_sla_heatmap(
        data: pd.DataFrame,
        x_col: str = 'hour',
        y_col: str = 'day',
        value_col: str = 'service_level'
    ) -> go.Figure:
        """
        Create a heatmap showing SLA by hour and day
        """
        pivot = data.pivot_table(
            values=value_col,
            index=y_col,
            columns=x_col,
            aggfunc='mean'
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=pivot.values,
            x=pivot.columns,
            y=pivot.index,
            colorscale='RdYlGn',
            zmid=80,
            text=pivot.values.round(1),
            texttemplate='%{text:.1f}%',
            textfont={"size": 10},
            colorbar=dict(
                title="SLA %",
                titleside="right"
            )
        ))
        
        fig.update_layout(
            title='SLA Performance Heatmap (By Hour and Day)',
            xaxis_title='Hour of Day',
            yaxis_title='Day of Week',
            template='plotly_white'
        )
        
        return fig
    
    @staticmethod
    def create_multi_scenario_analysis(
        scenarios: pd.DataFrame,
        title: str = "SLA vs Staffing Scenarios"
    ) -> go.Figure:
        """
        Create a multi-scenario chart showing trade-offs
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'SLA vs Agents',
                'ASA vs Agents',
                'Occupancy vs Agents',
                'Probability of Wait vs Agents'
            )
        )
        
        # SLA vs Agents
        fig.add_trace(
            go.Scatter(
                x=scenarios['num_agents'],
                y=scenarios['service_level'],
                mode='lines+markers',
                name='SLA %',
                line=dict(color='green', width=3)
            ),
            row=1, col=1
        )
        
        # ASA vs Agents
        fig.add_trace(
            go.Scatter(
                x=scenarios['num_agents'],
                y=scenarios['asa_seconds'],
                mode='lines+markers',
                name='ASA (sec)',
                line=dict(color='orange', width=3)
            ),
            row=1, col=2
        )
        
        # Occupancy vs Agents
        fig.add_trace(
            go.Scatter(
                x=scenarios['num_agents'],
                y=scenarios['occupancy'],
                mode='lines+markers',
                name='Occupancy %',
                line=dict(color='blue', width=3)
            ),
            row=2, col=1
        )
        
        # Prob Wait vs Agents
        fig.add_trace(
            go.Scatter(
                x=scenarios['num_agents'],
                y=scenarios['prob_wait'],
                mode='lines+markers',
                name='Prob Wait %',
                line=dict(color='red', width=3)
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title=title,
            template='plotly_white',
            height=800,
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def create_abandonment_analysis(
        forecast: pd.DataFrame,
        num_agents: int,
        abandonment_data: Dict
    ) -> go.Figure:
        """
        Create chart showing abandonment rates and forecast
        """
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        # Forecast volume
        fig.add_trace(
            go.Bar(
                x=forecast['ds'],
                y=forecast['yhat'],
                name='Volume',
                marker_color='lightblue',
                opacity=0.5
            ),
            secondary_y=False
        )
        
        # Add abandonment rate (estimated)
        total_volume = forecast['yhat'].sum()
        abandoned = int(total_volume * (abandonment_data.get('abandonment_rate', 0) / 100))
        
        fig.add_trace(
            go.Scatter(
                x=forecast['ds'],
                y=[abandonment_data.get('abandonment_rate', 0)] * len(forecast),
                name=f'Abandonment Rate (Est. {abandoned} calls)',
                mode='lines',
                line=dict(color='red', width=3, dash='dash')
            ),
            secondary_y=True
        )
        
        fig.update_layout(
            title='Abandonment Analysis',
            xaxis_title='Time Interval',
            template='plotly_white'
        )
        
        fig.update_yaxes(title_text="Volume", secondary_y=False)
        fig.update_yaxes(title_text="Abandonment Rate %", secondary_y=True, range=[0, 100])
        
        return fig
    
    @staticmethod
    def create_sales_forecast_chart(
        historical_revenue: pd.Series,
        forecast: pd.DataFrame,
        sales_metrics: Dict
    ) -> go.Figure:
        """
        Create sales forecast chart with revenue projections
        """
        fig = go.Figure()
        
        # Historical revenue
        fig.add_trace(go.Scatter(
            x=historical_revenue.index,
            y=historical_revenue.values,
            mode='lines+markers',
            name='Historical Revenue',
            line=dict(color='blue', width=2),
            marker=dict(size=6)
        ))
        
        # Forecast revenue
        if forecast is not None and not forecast.empty:
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat'],
                mode='lines+markers',
                name='Forecast Revenue',
                line=dict(color='green', width=3, dash='dash'),
                marker=dict(size=8)
            ))
            
            # Confidence interval
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_upper'],
                mode='lines',
                name='Upper Bound',
                line=dict(width=0),
                showlegend=False
            ))
            
            fig.add_trace(go.Scatter(
                x=forecast['ds'],
                y=forecast['yhat_lower'],
                mode='lines',
                name='Confidence Interval',
                fill='tonexty',
                fillcolor='rgba(0, 128, 0, 0.2)',
                line=dict(width=0),
                showlegend=True
            ))
        
        # Add annotations for key metrics
        fig.add_annotation(
            x=0.98,
            y=0.98,
            xref="paper",
            yref="paper",
            text=f"Avg Daily Revenue: ${sales_metrics.get('revenue_per_day', 0):,.0f}<br>Monthly Growth: {sales_metrics.get('average_monthly_growth', 0):.1%}",
            showarrow=False,
            font=dict(size=12),
            bgcolor="rgba(255, 255, 255, 0.8)",
            borderpad=4
        )
        
        fig.update_layout(
            title='Sales Revenue Forecast',
            xaxis_title='Date',
            yaxis_title='Revenue ($)',
            hovermode='x unified',
            template='plotly_white',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    @staticmethod
    def create_pipeline_visualization(
        pipeline_analysis: Dict,
        title: str = "Sales Pipeline Analysis"
    ) -> go.Figure:
        """
        Create pipeline visualization with stage distribution
        """
        stages = pipeline_analysis.get('stage_distribution', {})
        
        if not stages:
            return go.Figure()
        
        # Donut chart
        fig = go.Figure(data=[go.Pie(
            labels=list(stages.keys()),
            values=list(stages.values()),
            hole=.4,
            marker=dict(
                colors=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']
            ),
            textinfo='label+percent',
            textposition='auto'
        )])
        
        # Add annotation for total pipeline value
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text=f"Total Pipeline<br>${pipeline_analysis.get('total_pipeline_value', 0):,.0f}<br><br>Weighted Value<br>${pipeline_analysis.get('weighted_pipeline_value', 0):,.0f}",
            showarrow=False,
            font=dict(size=14, color="#333"),
            align="center"
        )
        
        fig.update_layout(
            title=title,
            template='plotly_white',
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_comprehensive_dashboard(
        volume_forecast: pd.DataFrame,
        staffing_results: Dict,
        sla_scenarios: pd.DataFrame,
        abandonment_data: Dict,
        sales_metrics: Dict = None
    ) -> go.Figure:
        """
        Create a comprehensive dashboard with multiple charts
        """
        # Create subplot grid
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Volume Forecast',
                'Sales Revenue (if available)',
                'Staffing Requirements',
                'SLA by Hour/Day',
                'Abandonment Analysis',
                'SLA vs Staffing Scenarios'
            ),
            specs=[[{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": True}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Add charts (simplified for brevity)
        # ... (implementation would add each chart type)
        
        fig.update_layout(
            title='WFM Comprehensive Dashboard',
            template='plotly_white',
            height=1200,
            showlegend=True
        )
        
        return fig
    
    @staticmethod
    def create_what_if_chart(
        base_scenario: Dict,
        alternative_scenarios: List[Dict],
        metric: str = 'service_level'
    ) -> go.Figure:
        """
        Create a 'What-If' comparison chart for different scenarios
        """
        df = pd.DataFrame(alternative_scenarios)
        
        fig = go.Figure()
        
        # Base scenario
        fig.add_trace(go.Bar(
            x=['Base Scenario'],
            y=[base_scenario.get(metric, 0)],
            name='Base',
            marker_color='blue'
        ))
        
        # Alternative scenarios
        fig.add_trace(go.Bar(
            x=df['scenario_name'].tolist(),
            y=df[metric].tolist(),
            name=metric,
            marker_color='green'
        ))
        
        # Add target line
        if 'target' in base_scenario:
            fig.add_hline(
                y=base_scenario['target'],
                line_dash="dash",
                line_color="red",
                annotation_text=f"Target: {base_scenario['target']}%"
            )
        
        fig.update_layout(
            title='What-If Scenario Comparison',
            xaxis_title='Scenario',
            yaxis_title=metric.replace('_', ' ').title(),
            template='plotly_white'
        )
        
        return fig