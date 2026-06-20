"""
Free LLM Integration using OpenRouter's free models
No credit card required for free tier models
"""

import os
import json
from typing import Dict, Any, Optional
import requests
from datetime import datetime
import pandas as pd

class LLMInterface:
    """
    Interface with free LLM models via OpenRouter API
    Recommended free models: openai/gpt-oss-120b, meta-llama/llama-3.2-3b-instruct:free
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize LLM interface
        
        Args:
            api_key: OpenRouter API key (get free from openrouter.ai)
        """
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"
        
        # Free model options (as of 2024)
        self.free_models = [
            "openai/gpt-oss-120b:free",  # Best quality, free
            "meta-llama/llama-3.2-3b-instruct:free",  # Fast, lightweight
            "mistralai/mistral-7b-instruct:free",  # Good balance
            "google/gemini-flash-1.5:free"  # New, capable
        ]
        
        # Default to best free model
        self.default_model = self.free_models[0]
    
    def chat_completion(
        self,
        messages: list,
        model: str = None,
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> Dict:
        """
        Send chat completion request to OpenRouter
        
        Args:
            messages: List of message objects [{"role": "user", "content": "..."}]
            model: Model to use (default: best free model)
            temperature: Randomness (0-1)
            max_tokens: Max response length
        """
        if not self.api_key:
            raise ValueError("OpenRouter API key required. Get free from openrouter.ai")
        
        model = model or self.default_model
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "http://localhost:8501",  # Streamlit app
            "X-Title": "WFM Assistant",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        try:
            response = requests.post(
                self.base_url,
                headers=headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            return {"error": "Request timed out. Please try again."}
        except requests.exceptions.RequestException as e:
            return {"error": f"API request failed: {str(e)}"}
    
    def explain_forecast(self, forecast_data: pd.DataFrame) -> str:
        """
        Generate human-readable explanation of forecast
        
        Args:
            forecast_data: DataFrame with 'ds' and 'yhat' columns
        """
        # Prepare summary statistics
        total_volume = forecast_data['yhat'].sum()
        peak_volume = forecast_data['yhat'].max()
        avg_volume = forecast_data['yhat'].mean()
        peak_time = forecast_data.loc[forecast_data['yhat'].idxmax(), 'ds']
        
        # Create prompt
        prompt = f"""
        You are a WFM Assistant explaining a contact center volume forecast.
        
        Forecast Summary:
        - Total forecasted volume: {total_volume:,} calls
        - Average volume per interval: {avg_volume:.0f} calls
        - Peak volume: {peak_volume:,} calls at {peak_time}
        - Number of intervals: {len(forecast_data)}
        
        Please provide:
        1. A brief executive summary of the forecast
        2. Key insights about patterns or trends
        3. One actionable recommendation for staffing
        Keep the response professional and concise.
        """
        
        messages = [
            {"role": "system", "content": "You are a helpful WFM assistant for a contact center."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.chat_completion(messages)
        
        if 'error' in response:
            return f"AI explanation unavailable: {response['error']}\n\nSummary statistics:\n{self._generate_fallback_summary(forecast_data)}"
        
        return response.get('choices', [{}])[0].get('message', {}).get('content', '')
    
    def generate_staffing_recommendation(
        self,
        staffing_results: Dict,
        target_sla: float,
        forecast_volume: pd.DataFrame
    ) -> str:
        """
        Generate staffing recommendations using LLM
        """
        total_agents = staffing_results.get('total_agents_needed', 0)
        avg_agents = staffing_results.get('average_agents_needed', 0)
        peak_requirements = staffing_results.get('peak_time_requirements', [])
        
        prompt = f"""
        You are a WFM Assistant providing staffing recommendations.
        
        Requirements:
        - Target SLA: {target_sla}%
        - Total agents needed (peak): {total_agents}
        - Average agents needed: {avg_agents:.1f}
        - Peak times: {json.dumps(peak_requirements[:3], default=str)}
        - Total forecast volume: {forecast_volume['yhat'].sum():,}
        
        Provide:
        1. Recommended staffing strategy
        2. How to handle peak times efficiently
        3. A balanced recommendation for the schedule
        """
        
        messages = [
            {"role": "system", "content": "You are a senior WFM consultant."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.chat_completion(messages, temperature=0.5)
        
        if 'error' in response:
            return f"Recommendation unavailable: {response['error']}\n\nSummary:\n- Peak staffing: {total_agents} agents\n- Average staffing: {avg_agents:.1f} agents"
        
        return response.get('choices', [{}])[0].get('message', {}).get('content', '')
    
    def explain_simulation_results(self, results: Dict, scenario_type: str) -> str:
        """
        Explain simulation results in plain language
        """
        prompt = f"""
        Explain these {scenario_type} simulation results:
        {json.dumps(results, default=str, indent=2)}
        
        Provide:
        1. Clear summary of the results
        2. What the numbers mean for operations
        3. A recommendation based on the results
        """
        
        messages = [
            {"role": "system", "content": "You are a WFM Analyst explaining simulation results."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.chat_completion(messages, temperature=0.5)
        
        if 'error' in response:
            return f"Explanation unavailable: {response['error']}"
        
        return response.get('choices', [{}])[0].get('message', {}).get('content', '')
    
    def answer_operational_question(self, question: str, context: str) -> str:
        """
        Answer general operational questions about the data
        """
        prompt = f"""
        Context: {context}
        
        Question: {question}
        
        Answer as a WFM expert.
        """
        
        messages = [
            {"role": "system", "content": "You are an experienced WFM professional."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.chat_completion(messages, temperature=0.7)
        
        if 'error' in response:
            return f"Unable to answer: {response['error']}"
        
        return response.get('choices', [{}])[0].get('message', {}).get('content', '')
    
    def _generate_fallback_summary(self, forecast_data: pd.DataFrame) -> str:
        """Generate summary without LLM"""
        return f"""
        Total Volume: {forecast_data['yhat'].sum():,}
        Average: {forecast_data['yhat'].mean():.0f}
        Peak: {forecast_data['yhat'].max():,}
        Min: {forecast_data['yhat'].min():,}
        """

# Singleton instance
_llm_instance = None

def get_llm_interface() -> LLMInterface:
    """Get or create LLM interface instance"""
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMInterface()
    return _llm_instance