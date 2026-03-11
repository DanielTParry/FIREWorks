"""GBM Infinite Analytical Strategy for probability of ruin.

This strategy uses the closed-form analytical solution for the probability
of ruin under Geometric Brownian Motion with constant parameters and constant
withdrawal, assuming infinite time horizon.

The solution employs the incomplete Gamma function and does not require
Monte Carlo simulation.
"""

from typing import Optional, Dict, Any
from fireworks.core.base import AbstractStrategy
from .calculator import GBMInfiniteAnalyticCalculator
from .models import (
    MarketEnvironmentFactory,
    ConsumptionModelFactory,
    AbstractMarketEnvironment,
    AbstractConsumptionModel,
)


class GBMInfiniteAnalyticStrategy(AbstractStrategy):
    """Analytical strategy for infinite-horizon GBM ruin probability."""

    def __init__(self,
                 market_environment: Optional[AbstractMarketEnvironment] = None,
                 consumption_model: Optional[AbstractConsumptionModel] = None) -> None:
        """
        Initialize the GBM Infinite Analytical Strategy.

        Args:
            market_environment: Market environment defining μ(t) and v(t)
                (assumed constant for analytical solution)
                Default: 7% return, 4% variance
            consumption_model: Consumption model defining C(t)
                (assumed constant for analytical solution)
                Default: constant $0 withdrawal
        """
        self.market_environment = market_environment or MarketEnvironmentFactory.constant(0.07, 0.04)
        self.consumption_model = consumption_model or ConsumptionModelFactory.constant(0)
        self.calculator = GBMInfiniteAnalyticCalculator(
            self.market_environment, 
            self.consumption_model
        )

    def calculate_ruin_probability(self, initial_capital: float,
                                  annual_withdrawal: float,
                                  years: float = None) -> float:
        """
        Calculate the probability of ruin over infinite time horizon.

        Note: The 'years' parameter is ignored for this strategy as we assume
        infinite horizon.

        Args:
            initial_capital: Starting portfolio value
            annual_withdrawal: Annual withdrawal amount
            years: Time horizon (ignored, always infinite)

        Returns:
            Probability of ruin (float between 0 and 1)
        """
        return self.calculator.calculate_ruin_probability_infinite(
            initial_capital, 
            annual_withdrawal
        )

    def simulate(self, initial_capital: float,
                annual_withdrawal: float,
                years: float = None,
                num_simulations: int = None) -> Dict[str, Any]:
        """
        Calculate analytical ruin probability and statistics.

        Note: This is not a Monte Carlo simulation. It uses the closed-form
        analytical solution. The 'num_simulations' parameter is ignored.

        Args:
            initial_capital: Starting portfolio value
            annual_withdrawal: Annual withdrawal amount
            years: Time horizon (ignored, always infinite)
            num_simulations: Number of simulations (ignored, not used)

        Returns:
            Dictionary with:
                - ruin_probability: P(ruin) over infinite horizon
                - survival_probability: P(survival) over infinite horizon
                - withdrawal_rate: annual_withdrawal / initial_capital
                - statistics: detailed parameters used in calculation
        """
        ruin_prob = self.calculate_ruin_probability(
            initial_capital, 
            annual_withdrawal, 
            years
        )
        
        stats = self.calculator.compute_statistics(
            initial_capital, 
            annual_withdrawal
        )

        return {
            'ruin_probability': ruin_prob,
            'survival_probability': 1.0 - ruin_prob,
            'withdrawal_rate': annual_withdrawal / initial_capital if initial_capital > 0 else 0,
            'method': 'gbm_infinite_analytical',
            'horizon': 'infinite',
            'statistics': stats,
        }
