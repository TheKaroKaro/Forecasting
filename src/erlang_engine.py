"""
Erlang C Engine for Contact Center Staffing Calculations
Using scipy for mathematical functions
"""

import math
from typing import Tuple, Dict, List
import numpy as np
from scipy.special import factorial

class ErlangEngine:
    """
    Implements Erlang C calculations for workforce management
    All formulas are standard and royalty-free
    """
    
    @staticmethod
    def erlang_c(num_agents: int, call_rate: float, avg_handle_time: float) -> float:
        """
        Calculate Erlang C probability that a call waits
        """
        traffic_intensity = call_rate * avg_handle_time
        if num_agents <= traffic_intensity:
            return 1.0  # System overloaded
        
        # Calculate using scipy factorial
        numerator = (traffic_intensity ** num_agents) / factorial(num_agents)
        
        # Summation term
        summation = 0
        for i in range(num_agents):
            summation += (traffic_intensity ** i) / factorial(i)
        
        denominator = (1 - traffic_intensity / num_agents) * summation + numerator
        
        if denominator == 0:
            return 1.0
        
        return numerator / denominator
    
    # ... rest of the class methods remain the same ...