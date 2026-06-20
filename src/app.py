"""
WFM Pro - Complete Contact Center & Sales Management Tool
With full visualizations and AI capabilities
All features built with free resources
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv
import os

from src.forecasting import VolumeForecaster
from src.sales_forecast import SalesForecaster
from src.erlang_engine import ErlangEngine
from src.simulation import SimulationEngine
from src.llm_interface import get_llm_interface
from src.visualizations import WFMLVisualizer

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="WFM Pro - Complete Tool",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'volume_forecaster' not in st.session_state:
    st.session_state.volume_forecaster = VolumeForecaster()
if 'sales_forecaster' not in st.session_state:
    st.session_state.sales_forecaster = SalesForecaster()
if 'simulation_engine' not in st.session_state:
    st.session_state.simulation_engine = SimulationEngine()
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = WFMLVisualizer()
if 'llm_interface' not in st.session_state:
    try:
        st.session_state.llm_interface = get_llm_interface()
    except:
        st.session_state.llm_interface = None
if 'forecast_results' not in st.session_state:
    st.session_state.forecast_results = None
if 'sales_forecast_results' not in st.session_state:
    st.session_state.sales_forecast_results = None

# Sidebar Navigation
st.sidebar.title("📊 WFM Pro")
st.sidebar.markdown("---")

pages = [
    "🏠 Dashboard",
    "📞 Contact Center Forecast",
    "💰 Sales Forecast",
    "📋 Staffing Calculator",
    "🔄 Shift Simulation",
    "📊 SLA Scenarios",
    "🎯 What-If Analysis",
    "🤖 AI Assistant",
    "📚 Documentation"
]

selected_page = st.sidebar.radio("Navigation", pages)

# Footer
st.sidebar.markdown("---")
st.sidebar.caption("Built with ❤️ using free resources")
st.sidebar.caption("Powered by OpenRouter AI (free tier)")

# Initialize engines
erlang = ErlangEngine()
sim_engine = st.session_state.simulation_engine
visualizer = st.session_state.visualizer

# =============== PAGE ROUTING ===============

if selected_page == "🏠 Dashboard":
    st.title("🎯 WFM Pro Dashboard")
    st.markdown("*Complete Workforce Management with Sales & Contact Center Analytics*")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📞 Forecast Calls", "2,847", "+12% vs last week")
    with col2:
        st.metric("💰 Revenue Forecast", "$284,700", "+8.5%")
    with col3:
        st.metric("👥 Agents Needed", "34", "at peak")
    with col4:
        st.metric("🎯 Current SLA", "78.5%", "-1.5%")
    
    st.markdown("---")
    
    # Quick action buttons
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("📞 Generate Contact Forecast", use_container_width=True):
            st.info("Navigate to Contact Center Forecast page")
    with col2:
        if st.button("💰 Generate Sales Forecast", use_container_width=True):
            st.info("Navigate to Sales Forecast page")
    with col3:
        if st.button("🔄 Run What-If Analysis", use_container_width=True):
            st.info("Navigate to What-If Analysis page")
    
    st.markdown("---")
    
    # Sample charts on dashboard
    st.subheader("Quick Overview Charts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Sample volume chart
        sample_forecast = pd.DataFrame({
            'ds': pd.date_range(start='2024-01-01', periods=30, freq='D'),
            'yhat': np.random.poisson(100, 30),
            'yhat_lower': np.random.poisson(80, 30),
            'yhat_upper': np.random.poisson(120, 30)
        })
        fig = visualizer.create_volume_forecast_chart(None, sample_forecast, "Demo: Volume Forecast")
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Sample heatmap
        sample_sla_data = pd.DataFrame({
            'hour': np.tile(range(8, 21), 7),
            'day': np.repeat(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'], 13),
            'service_level': np.random.normal(75, 15, 91)
        })
        fig = visualizer.create_sla_heatmap(sample_sla_data)
        st.plotly_chart(fig, use_container_width=True)

elif selected_page == "📞 Contact Center Forecast":
    st.title("📞 Contact Center Volume Forecast")
    
    st.markdown("""
    ### Upload Historical Data
    Upload a CSV file with your historical call volume data.
    **Required columns:** `timestamp` (datetime) and `volume` (integer)
    """)
    
    # File upload
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            historical_data = pd.read_csv(uploaded_file)
            st.success(f"✅ Data loaded: {len(historical_data)} rows")
            st.dataframe(historical_data.head())
            
            # Prepare data for forecasting
            st.session_state.volume_forecaster.historical_data = historical_data
            
            # Forecasting options
            col1, col2 = st.columns(2)
            with col1:
                periods = st.slider("Forecast Periods (hours)", 12, 168, 24)
                method = st.selectbox(
                    "Forecast Method",
                    ['ensemble', 'prophet', 'arima']
                )
            
            with col2:
                include_seasonality = st.checkbox("Include Seasonality", True)
                show_confidence = st.checkbox("Show Confidence Intervals", True)
            
            if st.button("🚀 Generate Forecast", use_container_width=True):
                with st.spinner("Generating forecast..."):
                    forecast = st.session_state.volume_forecaster.forecast_volume(
                        periods=periods,
                        method=method
                    )
                    st.session_state.forecast_results = forecast
                    st.success("✅ Forecast generated!")
            
            # Display forecast
            if st.session_state.forecast_results is not None:
                forecast = st.session_state.forecast_results
                
                # Show metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Volume", f"{forecast['yhat'].sum():,}")
                with col2:
                    st.metric("Peak Volume", f"{forecast['yhat'].max():,}")
                with col3:
                    st.metric("Average Volume", f"{forecast['yhat'].mean():.0f}")
                
                # Main forecast chart
                st.subheader("📈 Volume Forecast")
                fig = visualizer.create_volume_forecast_chart(
                    st.session_state.volume_forecaster.prepare_data(historical_data),
                    forecast
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed data
                with st.expander("📋 View Detailed Forecast Data"):
                    st.dataframe(forecast)
                
                # AI Explanation
                if st.session_state.llm_interface:
                    if st.button("🤖 Get AI Explanation"):
                        with st.spinner("Generating AI insights..."):
                            explanation = st.session_state.llm_interface.explain_forecast(forecast)
                            st.markdown("### 🤖 AI Insights")
                            st.write(explanation)
        
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
    else:
        # Provide sample data
        st.info("👆 Upload a CSV file to get started, or use sample data below")
        if st.button("📊 Load Sample Contact Center Data"):
            # Generate sample data
            dates = pd.date_range(start='2024-01-01', periods=30*24, freq='H')
            base_volume = 50 + 30*np.sin(np.arange(len(dates))/24 * 2*np.pi)
            sample_data = pd.DataFrame({
                'timestamp': dates,
                'volume': np.random.poisson(base_volume + 20)
            })
            st.session_state.volume_forecaster.historical_data = sample_data
            st.success("✅ Sample data loaded!")
            st.dataframe(sample_data.head())

elif selected_page == "💰 Sales Forecast":
    st.title("💰 Sales Revenue Forecast")
    
    st.markdown("""
    ### Upload Sales Data
    Upload a CSV file with your sales history.
    **Required columns:** `date` (datetime) and `revenue` (float)
    **Optional:** `deals_won`, `deals_lost`, `pipeline_value`
    """)
    
    sales_file = st.file_uploader("Upload Sales CSV", type=['csv'], key="sales_upload")
    
    if sales_file is not None:
        try:
            sales_data = pd.read_csv(sales_file)
            st.success(f"✅ Sales data loaded: {len(sales_data)} rows")
            st.dataframe(sales_data.head())
            
            # Forecasting options
            col1, col2 = st.columns(2)
            with col1:
                forecast_days = st.slider("Forecast Days", 7, 90, 30, key="sales_days")
                method = st.selectbox(
                    "Forecast Method",
                    ['ensemble', 'prophet', 'exponential'],
                    key="sales_method"
                )
            
            with col2:
                include_seasonality = st.checkbox("Include Seasonality", True, key="sales_season")
                show_pipeline = st.checkbox("Show Pipeline Analysis", True, key="sales_pipeline")
            
            if st.button("💰 Generate Sales Forecast", use_container_width=True, key="sales_btn"):
                with st.spinner("Generating sales forecast..."):
                    sales_forecast = st.session_state.sales_forecaster.forecast_revenue(
                        sales_data,
                        periods=forecast_days,
                        method=method,
                        include_seasonality=include_seasonality
                    )
                    st.session_state.sales_forecast_results = sales_forecast
                    st.success("✅ Sales forecast generated!")
            
            # Display sales forecast
            if st.session_state.sales_forecast_results is not None:
                forecast = st.session_state.sales_forecast_results
                metrics = st.session_state.sales_forecaster.calculate_sales_metrics(sales_data)
                
                # Metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Revenue", f"${forecast['yhat'].sum():,.0f}")
                with col2:
                    st.metric("Daily Average", f"${forecast['yhat'].mean():,.0f}")
                with col3:
                    st.metric("Monthly Growth", f"{metrics.get('average_monthly_growth', 0):.1%}")
                with col4:
                    st.metric("Avg Deal Size", f"${metrics.get('average_deal_size', 0):,.0f}")
                
                # Sales forecast chart
                st.subheader("📈 Revenue Forecast")
                fig = visualizer.create_sales_forecast_chart(
                    pd.Series(sales_data['revenue'].values, index=pd.to_datetime(sales_data['date'])),
                    forecast,
                    metrics
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Pipeline analysis
                if show_pipeline and 'pipeline_value' in sales_data.columns:
                    st.subheader("📊 Pipeline Analysis")
                    # Create pipeline data
                    pipeline_data = pd.DataFrame({
                        'stage': ['prospecting', 'qualification', 'proposal', 'negotiation', 'closed_won'],
                        'value': [50000, 75000, 120000, 80000, 45000],
                        'close_date': pd.date_range(start='2024-01-01', periods=5, freq='M')
                    })
                    pipeline_analysis = st.session_state.sales_forecaster.pipeline_analysis(pipeline_data)
                    fig = visualizer.create_pipeline_visualization(pipeline_analysis)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show pipeline details
                    with st.expander("📋 Pipeline Details"):
                        st.json(pipeline_analysis)
                
                # AI Analysis
                if st.session_state.llm_interface:
                    if st.button("🤖 Get Sales Insights", key="sales_ai"):
                        with st.spinner("Analyzing sales data..."):
                            prompt = f"""
                            Sales Forecast Summary:
                            - Total forecast revenue: ${forecast['yhat'].sum():,.0f}
                            - Daily average: ${forecast['yhat'].mean():,.0f}
                            - Monthly growth: {metrics.get('average_monthly_growth', 0):.1%}
                            - Average deal size: ${metrics.get('average_deal_size', 0):,.0f}
                            
                            Provide sales strategy insights.
                            """
                            insight = st.session_state.llm_interface.answer_operational_question(
                                prompt,
                                "Sales performance data"
                            )
                            st.markdown("### 🤖 AI Sales Insights")
                            st.write(insight)
        
        except Exception as e:
            st.error(f"Error processing sales data: {str(e)}")
    else:
        st.info("👆 Upload sales data to generate forecasts and insights")

elif selected_page == "📋 Staffing Calculator":
    st.title("📋 Staffing Requirements Calculator")
    
    st.markdown("""
    ### Calculate required agents based on call volume and SLA targets
    Enter your forecast or use the sliders to estimate staffing needs.
    """)
    
    # Input method
    input_method = st.radio(
        "Input Method",
        ["Use Forecast Data", "Manual Entry"],
        horizontal=True
    )
    
    if input_method == "Use Forecast Data":
        if st.session_state.forecast_results is not None:
            forecast = st.session_state.forecast_results
            
            # SLA targets
            col1, col2, col3 = st.columns(3)
            with col1:
                target_sla = st.slider("Target SLA %", 60, 99, 80)
            with col2:
                target_answer_time = st.slider("Target Answer Time (seconds)", 10, 120, 20)
            with col3:
                avg_handle_time = st.slider("Average Handle Time (minutes)", 2, 15, 5)
            
            if st.button("📊 Calculate Staffing", use_container_width=True):
                with st.spinner("Calculating staffing requirements..."):
                    results = sim_engine.calculate_staffing_requirements(
                        forecast,
                        target_sla,
                        target_answer_time,
                        avg_handle_time
                    )
                    st.session_state.staffing_results = results
                    st.success("✅ Staffing calculation complete!")
            
            if 'staffing_results' in st.session_state:
                results = st.session_state.staffing_results
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Peak Agents Needed", results['total_agents_needed'])
                with col2:
                    st.metric("Average Agents", f"{results['average_agents_needed']:.1f}")
                with col3:
                    st.metric("Intervals Analyzed", len(results['interval_requirements']))
                
                # Staffing requirements chart
                st.subheader("📊 Staffing Requirements by Interval")
                fig = visualizer.create_staffing_requirements_chart(forecast, results)
                st.plotly_chart(fig, use_container_width=True)
                
                # Detailed table
                with st.expander("📋 View Interval Details"):
                    df = pd.DataFrame(results['interval_requirements'])
                    st.dataframe(df)
                
                # AI Recommendation
                if st.session_state.llm_interface:
                    if st.button("🤖 Get Staffing Recommendation"):
                        with st.spinner("Generating recommendation..."):
                            recommendation = st.session_state.llm_interface.generate_staffing_recommendation(
                                results,
                                target_sla,
                                forecast
                            )
                            st.markdown("### 🤖 Staffing Recommendation")
                            st.write(recommendation)
        else:
            st.warning("⚠️ Please generate a forecast first on the Contact Center Forecast page")
    
    else:  # Manual Entry
        st.subheader("Manual Staffing Calculator")
        
        col1, col2 = st.columns(2)
        with col1:
            calls_per_hour = st.number_input("Calls per Hour", min_value=1, value=50)
            avg_handle_time = st.number_input("Average Handle Time (minutes)", min_value=1, value=5)
        
        with col2:
            target_sla = st.slider("Target SLA %", 60, 99, 80, key="manual_sla")
            target_answer = st.slider("Target Answer Time (seconds)", 10, 120, 20, key="manual_answer")
        
        if st.button("🧮 Calculate", use_container_width=True):
            # Calculate staffing
            avg_handle_time_hours = avg_handle_time / 60
            target_answer_hours = target_answer / 3600
            
            agents_needed, metrics = erlang.calculate_agents_needed(
                calls_per_hour,
                avg_handle_time_hours,
                target_sla,
                target_answer_hours
            )
            
            # Display results
            st.subheader("📊 Results")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Agents Needed", agents_needed)
            with col2:
                st.metric("Expected SLA", f"{metrics.get('service_level', 0):.1f}%")
            with col3:
                st.metric("ASA (sec)", f"{metrics.get('average_speed_answer', 0):.1f}")
            with col4:
                st.metric("Occupancy", f"{metrics.get('occupancy', 0):.1f}%")
            
            # Show what-if scenarios
            st.subheader("📈 What-If Scenarios")
            scenario_agents = list(range(agents_needed - 3, agents_needed + 6))
            scenario_agents = [a for a in scenario_agents if a > 0]
            
            scenarios = []
            for a in scenario_agents:
                m = erlang.calculate_service_level(
                    a,
                    calls_per_hour,
                    avg_handle_time_hours,
                    target_answer_hours
                )
                scenarios.append({
                    'num_agents': a,
                    'service_level': m['service_level'],
                    'asa_seconds': m['average_speed_answer'],
                    'occupancy': m['occupancy'],
                    'prob_wait': m['probability_wait']
                })
            
            scenario_df = pd.DataFrame(scenarios)
            fig = visualizer.create_multi_scenario_analysis(scenario_df)
            st.plotly_chart(fig, use_container_width=True)

elif selected_page == "🔄 Shift Simulation":
    st.title("🔄 Shift Simulation")
    
    st.markdown("""
    ### Simulate shift performance and SLA achievement
    Test different staffing levels and shift configurations
    """)
    
    # Input parameters
    col1, col2 = st.columns(2)
    
    with col1:
        shift_duration = st.slider("Shift Duration (hours)", 4, 12, 8)
        num_agents = st.number_input("Number of Agents", min_value=1, max_value=100, value=20)
        avg_handle_time = st.slider("Avg Handle Time (min)", 2, 15, 5)
    
    with col2:
        target_sla = st.slider("Target SLA %", 60, 99, 80, key="sim_sla")
        target_answer = st.slider("Target Answer Time (sec)", 10, 120, 20, key="sim_answer")
        avg_patience = st.slider("Avg Patience Time (sec)", 30, 180, 60)
    
    # Volume profile
    st.subheader("📊 Volume Profile")
    profile_type = st.selectbox(
        "Select Volume Pattern",
        ["Even Distribution", "Morning Peak", "Afternoon Peak", "Late Peak", "Custom"]
    )
    
    if profile_type != "Custom":
        # Generate profile based on selection
        hours = list(range(shift_duration))
        if profile_type == "Morning Peak":
            profile = np.exp(-((np.array(hours) - 2) ** 2) / 2) * 100 + 30
        elif profile_type == "Afternoon Peak":
            profile = np.exp(-((np.array(hours) - shift_duration/2) ** 2) / 2) * 100 + 30
        elif profile_type == "Late Peak":
            profile = np.exp(-((np.array(hours) - (shift_duration-2)) ** 2) / 2) * 100 + 30
        else:
            profile = np.full(shift_duration, 50)
        
        # Display profile
        profile_df = pd.DataFrame({
            'Hour': hours,
            'Volume': profile.astype(int)
        })
        st.line_chart(profile_df.set_index('Hour'))
    
    if st.button("🔄 Run Simulation", use_container_width=True):
        with st.spinner("Running simulation..."):
            # Generate forecast data
            start_time = datetime.now().replace(hour=8, minute=0, second=0)
            intervals = pd.date_range(
                start=start_time,
                periods=shift_duration,
                freq='H'
            )
            
            if profile_type == "Custom":
                # Allow manual entry
                volumes = []
                for i in range(shift_duration):
                    vol = st.number_input(f"Hour {i+1} Volume", min_value=0, value=50, key=f"vol_{i}")
                    volumes.append(vol)
                forecast_df = pd.DataFrame({
                    'ds': intervals,
                    'yhat': volumes
                })
            else:
                forecast_df = pd.DataFrame({
                    'ds': intervals,
                    'yhat': profile.astype(int)
                })
            
            # Run simulation
            sim_engine.set_forecaster(st.session_state.volume_forecaster)
            results = sim_engine.run_shift_simulation(
                num_agents,
                start_time,
                start_time + timedelta(hours=shift_duration),
                interval_minutes=60,
                target_service_level=target_sla,
                target_answer_time=target_answer
            )
            
            # Display results
            st.subheader("📊 Simulation Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Avg SLA", f"{results['average_service_level']:.1f}%")
            with col2:
                st.metric("SLA Met", f"{results['service_level_met']:.1f}%")
            with col3:
                st.metric("Avg ASA", f"{results['average_asa']:.1f}s")
            
            # Detailed interval results
            interval_df = pd.DataFrame(results['intervals'])
            st.dataframe(interval_df)
            
            # Plot results
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=interval_df['timestamp'],
                y=interval_df['forecast_volume'],
                name='Volume',
                marker_color='lightblue'
            ))
            fig.add_trace(go.Scatter(
                x=interval_df['timestamp'],
                y=interval_df['service_level'],
                name='SLA %',
                mode='lines+markers',
                yaxis='y2',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title='Shift Performance Simulation',
                xaxis_title='Time',
                yaxis_title='Volume',
                yaxis2=dict(
                    title='SLA %',
                    overlaying='y',
                    side='right',
                    range=[0, 100]
                ),
                template='plotly_white'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Abandonment analysis
            st.subheader("📊 Abandonment Analysis")
            abandonment_results = sim_engine.analyze_abandonment(
                num_agents,
                forecast_df,
                avg_handle_time,
                avg_patience
            )
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Estimated Abandon Rate", f"{abandonment_results['abandonment_rate']:.1f}%")
            with col2:
                st.metric("Est. Abandoned Calls", abandonment_results['est_abandoned_calls'])
            with col3:
                st.metric("Est. Wait Time", f"{abandonment_results['estimated_wait_time']:.1f}s")
            
            # AI Insights
            if st.session_state.llm_interface:
                if st.button("🤖 Get Simulation Insights"):
                    with st.spinner("Analyzing simulation..."):
                        insight = st.session_state.llm_interface.explain_simulation_results(
                            results,
                            "Shift Simulation"
                        )
                        st.markdown("### 🤖 Simulation Insights")
                        st.write(insight)

elif selected_page == "📊 SLA Scenarios":
    st.title("📊 SLA Scenario Analysis")
    
    st.markdown("""
    ### Compare different staffing scenarios and their impact on SLA
    Visualize the trade-offs between staffing levels and service levels
    """)
    
    # Input parameters
    col1, col2 = st.columns(2)
    with col1:
        calls_per_hour = st.number_input("Calls per Hour", min_value=10, value=100, key="scenario_calls")
        avg_handle = st.slider("Avg Handle Time (min)", 2, 15, 5, key="scenario_handle")
        target_answer = st.slider("Target Answer (sec)", 10, 120, 20, key="scenario_answer")
    
    with col2:
        min_agents = st.number_input("Min Agents", min_value=1, value=5)
        max_agents = st.number_input("Max Agents", min_value=2, value=30)
        step_size = st.number_input("Step Size", min_value=1, value=1)
    
    if st.button("📊 Generate Scenarios", use_container_width=True):
        with st.spinner("Generating scenarios..."):
            # Generate agent range
            agent_range = list(range(min_agents, max_agents + 1, step_size))
            
            # Create scenario data
            scenarios = []
            for agents in agent_range:
                metrics = erlang.calculate_service_level(
                    agents,
                    calls_per_hour,
                    avg_handle / 60,
                    target_answer / 3600
                )
                scenarios.append({
                    'num_agents': agents,
                    'service_level': metrics['service_level'],
                    'asa_seconds': metrics['average_speed_answer'],
                    'occupancy': metrics['occupancy'],
                    'prob_wait': metrics['probability_wait']
                })
            
            scenario_df = pd.DataFrame(scenarios)
            
            # Display comprehensive scenario charts
            st.subheader("📊 Scenario Analysis Dashboard")
            
            # Multi-panel chart
            fig = visualizer.create_multi_scenario_analysis(scenario_df)
            st.plotly_chart(fig, use_container_width=True)
            
            # Add target SLA line
            st.subheader("🎯 Target Analysis")
            target_sla = st.slider("Select Target SLA %", 60, 99, 80, key="target_sla")
            
            # Find agents needed for target
            target_agents = None
            for _, row in scenario_df.iterrows():
                if row['service_level'] >= target_sla:
                    target_agents = row['num_agents']
                    break
            
            if target_agents:
                st.success(f"✅ To achieve {target_sla}% SLA, you need {target_agents} agents")
                
                # Show specific metrics at target
                target_row = scenario_df[scenario_df['num_agents'] == target_agents].iloc[0]
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("SLA", f"{target_row['service_level']:.1f}%")
                with col2:
                    st.metric("ASA", f"{target_row['asa_seconds']:.1f}s")
                with col3:
                    st.metric("Occupancy", f"{target_row['occupancy']:.1f}%")
            
            # Data table
            with st.expander("📋 Detailed Scenario Data"):
                st.dataframe(scenario_df)
            
            # Download option
            csv = scenario_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Scenario Data (CSV)",
                data=csv,
                file_name="sla_scenarios.csv",
                mime="text/csv"
            )

elif selected_page == "🎯 What-If Analysis":
    st.title("🎯 What-If Scenario Analysis")
    
    st.markdown("""
    ### Compare different what-if scenarios
    Test how changes in volume, staffing, or handle time affect performance
    """)
    
    # Base scenario
    st.subheader("Base Scenario")
    col1, col2, col3 = st.columns(3)
    with col1:
        base_volume = st.number_input("Base Volume (calls/hour)", value=100)
    with col2:
        base_agents = st.number_input("Base Agents", value=20)
    with col3:
        base_handle = st.slider("Base Handle Time (min)", 2, 15, 5, key="base_handle")
    
    # Alternative scenarios
    st.subheader("Alternative Scenarios")
    
    num_scenarios = st.number_input("Number of Scenarios to Compare", 1, 5, 3)
    
    scenarios = []
    for i in range(num_scenarios):
        st.markdown(f"**Scenario {i+1}**")
        col1, col2, col3 = st.columns(3)
        with col1:
            vol = st.number_input(f"Volume {i+1}", value=base_volume + (i+1)*20, key=f"vol_{i}")
        with col2:
            agents = st.number_input(f"Agents {i+1}", value=base_agents + (i+1)*2, key=f"agents_{i}")
        with col3:
            handle = st.slider(f"Handle Time {i+1}", 2, 15, base_handle + i, key=f"handle_{i}")
        
        # Calculate metrics
        metrics = erlang.calculate_service_level(
            agents,
            vol,
            handle / 60,
            20 / 3600  # 20 second target
        )
        scenarios.append({
            'scenario_name': f'Scenario {i+1}',
            'volume': vol,
            'agents': agents,
            'handle_time': handle,
            'service_level': metrics['service_level'],
            'asa': metrics['average_speed_answer'],
            'occupancy': metrics['occupancy']
        })
    
    if st.button("📊 Analyze Scenarios", use_container_width=True):
        # Create comparison chart
        fig = go.Figure()
        
        # Prepare data
        df = pd.DataFrame(scenarios)
        
        # Add traces for each metric
        fig.add_trace(go.Bar(
            x=df['scenario_name'],
            y=df['service_level'],
            name='SLA %',
            marker_color='green'
        ))
        
        fig.add_trace(go.Scatter(
            x=df['scenario_name'],
            y=df['asa'],
            name='ASA (seconds)',
            yaxis='y2',
            mode='lines+markers',
            line=dict(color='orange', width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=df['scenario_name'],
            y=df['occupancy'],
            name='Occupancy %',
            yaxis='y2',
            mode='lines+markers',
            line=dict(color='blue', width=3, dash='dash')
        ))
        
        fig.update_layout(
            title='Scenario Comparison',
            xaxis_title='Scenario',
            yaxis_title='SLA %',
            yaxis2=dict(
                title='ASA / Occupancy',
                overlaying='y',
                side='right'
            ),
            template='plotly_white',
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display scenario table
        st.dataframe(df)
        
        # AI Recommendation
        if st.session_state.llm_interface:
            if st.button("🤖 Get AI Recommendation"):
                with st.spinner("Analyzing scenarios..."):
                    best_scenario = df.loc[df['service_level'].idxmax()]
                    prompt = f"""
                    Based on scenario analysis:
                    - Best scenario: {best_scenario['scenario_name']}
                    - With {best_scenario['agents']} agents handling {best_scenario['volume']} calls/hour
                    - Achieves {best_scenario['service_level']:.1f}% SLA
                    - ASA: {best_scenario['asa']:.1f}s
                    
                    Provide recommendation for optimal staffing.
                    """
                    recommendation = st.session_state.llm_interface.answer_operational_question(
                        prompt,
                        "What-if scenario analysis results"
                    )
                    st.markdown("### 🤖 Optimal Scenario Recommendation")
                    st.write(recommendation)

elif selected_page == "🤖 AI Assistant":
    st.title("🤖 AI Assistant")
    
    st.markdown("""
    ### Ask questions about your data and get AI-powered insights
    The AI can help with forecasting, staffing, and operational questions.
    """)
    
    # Check LLM status
    if st.session_state.llm_interface is None:
        st.warning("⚠️ AI Assistant is not configured. Please set your OpenRouter API key.")
        st.markdown("""
        ### How to set up AI Assistant:
        1. Go to [OpenRouter.ai](https://openrouter.ai)
        2. Create a free account (no credit card needed)
        3. Get your API key
        4. Add it to a `.env` file: `OPENROUTER_API_KEY=your_key_here`
        """)
    else:
        st.success("✅ AI Assistant is ready!")
        
        # Chat interface
        question = st.text_area("Ask a question about your data:", height=100)
        
        if question:
            # Provide context based on available data
            context = "Available data: "
            if st.session_state.forecast_results is not None:
                context += f"Contact center forecast with {len(st.session_state.forecast_results)} intervals. "
            if st.session_state.sales_forecast_results is not None:
                context += f"Sales forecast available. "
            if hasattr(st.session_state, 'staffing_results'):
                context += "Staffing calculations available."
            
            if st.button("🤖 Get Answer", use_container_width=True):
                with st.spinner("Getting AI response..."):
                    response = st.session_state.llm_interface.answer_operational_question(
                        question,
                        context
                    )
                    st.markdown("### 🤖 AI Response")
                    st.write(response)
        
        # Quick question templates
        st.subheader("Quick Questions")
        col1, col2 = st.columns(2)
        
        questions = [
            "What is the peak call volume and when?",
            "How many agents do I need for 80% SLA?",
            "What's the impact of reducing handle time by 1 minute?",
            "When should I schedule more agents?",
            "How does sales volume affect staffing needs?",
            "What's the optimal shift schedule?"
        ]
        
        for i, q in enumerate(questions):
            if i % 2 == 0:
                with col1:
                    if st.button(q, key=f"q_{i}"):
                        st.session_state.ai_question = q
            else:
                with col2:
                    if st.button(q, key=f"q_{i}"):
                        st.session_state.ai_question = q

elif selected_page == "📚 Documentation":
    st.title("📚 Documentation & User Guide")
    
    st.markdown("""
    ## Welcome to WFM Pro!
    
    ### Overview
    This tool provides comprehensive workforce management capabilities for contact centers,
    including volume forecasting, staffing calculations, shift simulation, and sales forecasting.
    
    ### Features
    1. **Contact Center Forecast**: Predict call volumes using Prophet and ARIMA
    2. **Sales Forecast**: Forecast revenue and analyze sales pipeline
    3. **Staffing Calculator**: Determine optimal agent count for SLA targets
    4. **Shift Simulation**: Test different staffing scenarios
    5. **SLA Scenarios**: Compare staffing impacts on service levels
    6. **What-If Analysis**: Compare multiple scenarios side-by-side
    7. **AI Assistant**: Get intelligent insights from your data
    
    ### How to Use
    1. Start by uploading historical data on the respective forecast pages
    2. Generate forecasts to understand future demand
    3. Use the staffing calculator to determine requirements
    4. Run simulations to test different scenarios
    5. Ask the AI Assistant for insights and recommendations
    
    ### Technical Details
    - Built with Python, Streamlit, and Plotly
    - Uses free open-source libraries (Prophet, Statsmodels)
    - Powered by OpenRouter's free AI models
    - All calculations use standard Erlang C formulas
    
    ### Free Resources Used
    - **Forecasting**: Prophet (Meta), Statsmodels
    - **Calculations**: Erlang C formulas (open source)
    - **Visualization**: Plotly (free tier)
    - **AI**: OpenRouter free models (gpt-oss-120b:free)
    - **Hosting**: Streamlit Community Cloud
    
    ### Getting Started
    1. Configure your API keys in `.env` file
    2. Upload your historical data
    3. Explore the different features
    4. Generate forecasts and insights
    
    ### Support
    For questions or issues, refer to the documentation or use the AI Assistant.
    """)