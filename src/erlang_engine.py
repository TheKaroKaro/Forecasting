"""
Erlang C Engine for Contact Center Staffing Calculations
All calculations are free and open-source
"""

import math
from typing import Tuple, Dict, List
import numpy as np

class ErlangEngine:
    """
    Implements Erlang C calculations for workforce management
    All formulas are standard and royalty-free
    """
    
    @staticmethod
    def factorial(n: int) -> float:
        """Calculate factorial, using approximation for large numbers"""
        if n == 0:
            return 1
        if n > 170:  # Use Stirling's approximation for large numbers
            return math.sqrt(2 * math.pi * n) * (n / math.e) ** n
        return math.factorial(n)
    
    @staticmethod
    def erlang_c(num_agents: int, call_rate: float, avg_handle_time: float) -> float:
        """
        Calculate Erlang C probability that a call waits
        
        Args:
            num_agents: Number of agents available
            call_rate: Calls per time unit (e.g., per hour)
            avg_handle_time: Average handling time in same time unit
        
        Returns:
            Probability that a call has to wait (0 to 1)
        """
        traffic_intensity = call_rate * avg_handle_time
        if num_agents <= traffic_intensity:
            return 1.0  # System overloaded, all calls wait
        
        erlang_b = ErlangEngine.erlang_b(num_agents, traffic_intensity)
        result = (num_agents * erlang_b) / (num_agents - traffic_intensity * (1 - erlang_b))
        return min(result, 1.0)  # Cap at 1.0
    
    @staticmethod
    def erlang_b(num_lines: int, traffic_intensity: float) -> float:
        """
        Calculate Erlang B probability that a call is blocked
        
        Args:
            num_lines: Number of lines/agents
            traffic_intensity: Offered traffic in Erlangs
        
        Returns:
            Blocking probability
        """
        if traffic_intensity == 0:
            return 0
        inv_b = 1.0
        for i in range(1, num_lines + 1):
            inv_b = 1.0 + (i / traffic_intensity) * inv_b
        return 1.0 / inv_b
    
    @staticmethod
    def calculate_service_level(
        num_agents: int,
        call_rate: float,
        avg_handle_time: float,
        target_answer_time: float
    ) -> Dict[str, float]:
        """
        Calculate service level based on Erlang C
        
        Returns:
            Dictionary with service level metrics
        """
        traffic_intensity = call_rate * avg_handle_time
        if num_agents <= traffic_intensity:
            return {
                'service_level': 0,
                'average_speed_answer': float('inf'),
                'probability_wait': 1.0,
                'occupancy': 1.0
            }
        
        prob_wait = ErlangEngine.erlang_c(num_agents, call_rate, avg_handle_time)
        
        # Calculate service level (percent answered within target time)
        service_level = 1 - prob_wait * math.exp(
            -(num_agents - traffic_intensity) * target_answer_time / avg_handle_time
        )
        
        # Calculate Average Speed of Answer (ASA)
        asa = prob_wait * avg_handle_time / (num_agents - traffic_intensity)
        
        # Calculate agent occupancy
        occupancy = traffic_intensity / num_agents
        
        return {
            'service_level': service_level * 100,  # Convert to percentage
            'average_speed_answer': asa,
            'probability_wait': prob_wait * 100,
            'occupancy': occupancy * 100,
            'traffic_intensity': traffic_intensity,
            'required_agents': math.ceil(traffic_intensity + 1)  # Minimum needed
        }
    
    @staticmethod
    def calculate_agents_needed(
        call_rate: float,
        avg_handle_time: float,
        target_service_level: float,
        target_answer_time: float,
        max_agents: int = 100
    ) -> Tuple[int, Dict]:
        """
        Calculate minimum agents needed to meet service level target
        
        Args:
            call_rate: Calls per hour
            avg_handle_time: Average handling time in hours
            target_service_level: Desired service level (0-100)
            target_answer_time: Target answer time in hours
            max_agents: Maximum agents to test
        
        Returns:
            Tuple of (minimum agents needed, metrics at that level)
        """
        for agents in range(1, max_agents + 1):
            metrics = ErlangEngine.calculate_service_level(
                agents, call_rate, avg_handle_time, target_answer_time
            )
            if metrics['service_level'] >= target_service_level:
                return agents, metrics
        
        return max_agents, {}
    
    @staticmethod
    def simulate_abandonment(
        num_agents: int,
        call_rate: float,
        avg_handle_time: float,
        avg_patience_time: float
    ) -> Dict[str, float]:
        """
        Simulate call abandonment based on average patience
        
        Args:
            avg_patience_time: Average time callers wait before abandoning
        
        Returns:
            Abandonment metrics
        """
        metrics = ErlangEngine.calculate_service_level(
            num_agents, call_rate, avg_handle_time, avg_patience_time
        )
        
        # Estimate abandonment rate
        if metrics['average_speed_answer'] < float('inf'):
            # Simple model: abandonment increases with wait time
            abandonment_rate = min(
                metrics['probability_wait'] * (
                    metrics['average_speed_answer'] / avg_patience_time
                ) / 100,
                1.0
            )
        else:
            abandonment_rate = 1.0
        
        return {
            'abandonment_rate': abandonment_rate * 100,
            'estimated_wait_time': metrics.get('average_speed_answer', float('inf')),
            'probability_wait': metrics.get('probability_wait', 100)
        }