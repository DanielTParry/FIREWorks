"""GBM Finite Analytical Strategy for probability of ruin.

This strategy uses the exact spectral decomposition analytical solution for
the probability of ruin under Geometric Brownian Motion with constant parameters
and constant withdrawal, over a finite time horizon.

The solution employs:
    1. Stationary ground state (incomplete Gamma function)
    2. Discrete bounded states (Monthus & Bouchaud pole residues)
    3. Continuous branch cut (scattering integral)

This method does not require Monte Carlo simulation and provides exact results
for finite time horizons.
"""

from typing import Optional, Dict, Any
from fireworks.core.base import AbstractStrategy
from .calculator import GBMFiniteAnalyticCalculator
from .models import (
    MarketEnvironmentFactory,
    ConsumptionModelFactory,
    AbstractMarketEnvironment,
    AbstractConsumptionModel,
)


class GBMFiniteAnalyticStrategy(AbstractStrategy):
    """Analytical strategy for finite-horizon GBM ruin probability."""

    def __init__(self,
                 market_environment: Optional[AbstractMarketEnvironment] = None,
                 consumption_model: Optional[AbstractConsumptionModel] = None) -> None:
        """
        Initialize the GBM Finite Analytical Strategy.

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
        self.calculator = GBMFiniteAnalyticCalculator(
            self.market_environment,
            self.consumption_model
        )

    def calculate_ruin_probability(self, initial_capital: float,
                                  annual_withdrawal: float,
                                  years: float) -> float:
        """
        Calculate the probability of ruin over finite time horizon.

        Args:
            initial_capital: Starting portfolio value
            annual_withdrawal: Annual withdrawal amount
            years: Time horizon in years (required)

        Returns:
            Probability of ruin (float between 0 and 1)

        Raises:
            ValueError: If years is None or <= 0
        """
        if years is None or years <= 0:
            raise ValueError("years must be a positive number for finite-horizon analysis")

        return self.calculator.calculate_ruin_probability_finite(
            initial_capital,
            annual_withdrawal,
            years
        )

    def simulate(self, initial_capital: float,
                annual_withdrawal: float,
                years: float = None,
                num_simulations: int = None) -> Dict[str, Any]:
        """
        Calculate analytical ruin probability and statistics.

        Note: This is not a Monte Carlo simulation. It uses the exact
        spectral decomposition analytical solution. The 'num_simulations'
        parameter is ignored.

        Args:
            initial_capital: Starting portfolio value
            annual_withdrawal: Annual withdrawal amount
            years: Time horizon in years (required)
            num_simulations: Number of simulations (ignored, not used)

        Returns:
            Dictionary with:
                - ruin_probability: P(ruin) over finite horizon
                - survival_probability: P(survival) over finite horizon
                - withdrawal_rate: annual_withdrawal / initial_capital
                - horizon: time horizon in years
                - spectral_components: breakdown of the three spectral components
                - statistics: detailed parameters used in calculation

        Raises:
            ValueError: If years is None or <= 0
        """
        if years is None or years <= 0:
            raise ValueError("years must be a positive number for finite-horizon analysis")

        ruin_prob = self.calculate_ruin_probability(
            initial_capital,
            annual_withdrawal,
            years
        )

        stats = self.calculator.compute_statistics(
            initial_capital,
            annual_withdrawal,
            years
        )

        return {
            'ruin_probability': ruin_prob,
            'survival_probability': 1.0 - ruin_prob,
            'withdrawal_rate': stats['withdrawal_rate'],
            'horizon': years,
            'spectral_components': stats['spectral_components'],
            'statistics': stats,
        }
