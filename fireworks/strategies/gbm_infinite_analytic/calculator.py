"""Analytical GBM infinite-horizon ruin probability calculator.

Uses the closed-form solution based on the incomplete Gamma function:
    P(survival) = Γ(a, x) / Γ(a)

For the GBM process:
    dP_t = (P_t * μ - C) * dt + P_t * √(σ²) * dW_t

With constant withdrawal C and infinite time horizon.
"""

from typing import Dict, Any
import numpy as np
from scipy.special import gammaincc, gammaln
from scipy.stats import gamma as gamma_dist


class GBMInfiniteAnalyticCalculator:
    """Analytical calculator for infinite-horizon GBM ruin probability."""

    def __init__(self, market_environment: 'AbstractMarketEnvironment', 
                 consumption_model: 'AbstractConsumptionModel') -> None:
        """
        Initialize the calculator.

        Args:
            market_environment: Market environment defining μ and variance
            consumption_model: Consumption model defining withdrawal C
        """
        self.market_environment = market_environment
        self.consumption_model = consumption_model

    def calculate_ruin_probability_infinite(self, initial_capital: float, 
                                           annual_withdrawal: float) -> float:
        """
        Calculate probability of ruin over infinite time horizon.

        Uses the closed-form analytical solution for GBM with constant
        parameters and constant withdrawal.

        Args:
            initial_capital: Starting portfolio value P_0
            annual_withdrawal: Annual withdrawal amount C

        Returns:
            Probability of ruin (float between 0 and 1)
        """
        # Get market parameters (at t=0, but assumed constant for GBM)
        mu = self.market_environment.get_mean(0)
        variance = self.market_environment.get_variance(0)
        
        # Get consumption (at t=0, currently constant)
        C = self.consumption_model.get_consumption(0, initial_capital)

        # Handle edge cases
        if variance <= 0:
            # Degenerate case
            return 0.0 if annual_withdrawal <= 0 else 1.0
        
        w = annual_withdrawal / initial_capital if initial_capital > 0 else 0

        if w <= 0:
            # No withdrawal means survival is certain
            return 0.0

        # Parameters for incomplete gamma function
        # a = (2*mu / variance) - 1
        # x = 2*w / variance
        # where w = C / P_0 is the withdrawal rate
        
        a = (2.0 * mu / variance) - 1.0
        x = 2.0 * w / variance

        # Check for degenerate parameter a
        if a <= 0:
            # Shape parameter must be positive. This happens when μ < σ²/2
            # In this case, use a softer boundary: if w >= μ, very high ruin probability
            if w >= mu:
                return 0.99
            else:
                # Fall back to simple heuristic
                return min(w / mu, 0.99)

        # Use scipy.stats.gamma.cdf which is more numerically stable
        # than direct incomplete gamma function manipulation
        # Ruin probability = 1 - P(X <= x) where X ~ Gamma(a, scale=1)
        # This is equivalent to 1 - gammaincc(a, x)
        
        # Clamp to valid range
        return float(np.clip(gamma_dist.cdf(x, a), 0.0, 1.0))

    def compute_statistics(self, initial_capital: float, 
                          annual_withdrawal: float) -> Dict[str, Any]:
        """
        Compute analytical statistics for infinite-horizon survival.

        Args:
            initial_capital: Starting portfolio value
            annual_withdrawal: Annual withdrawal amount

        Returns:
            Dictionary with analytical results
        """
        mu = self.market_environment.get_mean(0)
        variance = self.market_environment.get_variance(0)
        sigma = np.sqrt(variance)
        
        w = annual_withdrawal / initial_capital if initial_capital > 0 else 0
        a = (2.0 * mu / variance) - 1.0
        x = 2.0 * w / variance

        ruin_prob = self.calculate_ruin_probability_infinite(initial_capital, annual_withdrawal)
        survival_prob = 1.0 - ruin_prob

        return {
            'ruin_probability': ruin_prob,
            'survival_probability': survival_prob,
            'initial_capital': initial_capital,
            'annual_withdrawal': annual_withdrawal,
            'withdrawal_rate': w,
            'mean_return': mu,
            'variance': variance,
            'sigma': sigma,
            'gamma_a': a,
            'gamma_x': x,
            'horizon': 'infinite',
        }
