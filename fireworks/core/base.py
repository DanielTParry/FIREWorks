"""Base class for withdrawal strategies."""

from abc import ABC, abstractmethod
from typing import Dict, Any


class BaseStrategy(ABC):
    """Abstract base class for all withdrawal strategies."""

    @abstractmethod
    def calculate_ruin_probability(self, initial_capital: float, 
                                  annual_withdrawal: float, market_regime: Any) -> float:
        """
        Calculate the probability of ruin for the given parameters.

        Args:
            initial_capital: Starting portfolio value
            annual_withdrawal: Annual withdrawal amount
            market_regime: Market regime parameters

        Returns:
            Probability of ruin (float between 0 and 1)
        """
        pass

    @abstractmethod
    def simulate(self, initial_capital: float, annual_withdrawal: float, market_regime: Any, 
                years: float, num_simulations: int) -> Dict[str, Any]:
        """
        Run Monte Carlo simulations for the strategy.

        Args:
            initial_capital: Starting portfolio value
            annual_withdrawal: Annual withdrawal amount
            market_regime: Market regime parameters
            years: Number of years to simulate
            num_simulations: Number of simulation paths

        Returns:
            Results dictionary
        """
        pass
