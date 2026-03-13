"""Heston Stiff Perturbative Strategy for ruin probability calculation.

Implements the Strategy interface for computing probability of ruin
using perturbative analytical methods for the Heston stochastic volatility model.
"""

from typing import Dict, Optional

from fireworks.core.base import AbstractStrategy
from .models import AbstractHestonMarketEnvironment
from .calculator import HestonStiffPerturbativeCalculator


class HestonStiffPerturbativeStrategy(AbstractStrategy):
    """
    Strategy computing ruin probability for stochastic volatility (Heston) model.

    Uses perturbative expansion in ε = 1/κ (inverse mean reversion speed)
    to approximate analytical solution for finite retirement horizons.

    Follows same interface as MCStrategy and GBMFiniteAnalyticStrategy.
    """

    def __init__(
        self,
        market_environment: AbstractHestonMarketEnvironment,
        max_perturbative_order: int = 1,
        **kwargs,
    ):
        """
        Initialize Heston stiff perturbative strategy.

        Args:
            market_environment: AbstractHestonMarketEnvironment with all Heston parameters
            max_perturbative_order: Maximum perturbative order (O(1), O(ε), O(ε²), ...)
            **kwargs: Additional arguments (consumption_model, etc.) for interface compatibility
        """
        self.market_environment = market_environment
        self.max_perturbative_order = max_perturbative_order

        # Initialize calculator
        self.calculator = HestonStiffPerturbativeCalculator(
            market_env=market_environment,
        )

    def simulate(
        self,
        initial_capital: float,
        annual_withdrawal: float,
        years: float,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Compute ruin probability using perturbative analytical method.

        Args:
            initial_capital: Starting portfolio value ($)
            annual_withdrawal: Annual withdrawal amount ($)
            years: Investment horizon (years)
            **kwargs: Additional arguments (ignored for deterministic strategy)

        Returns:
            Dictionary with:
            - 'ruin_probability': Probability of ruin ∈ [0, 1]
            - 'survival_probability': 1 - ruin_probability
            - 'order_0': O(1) term
            - 'order_1': O(ε) term (if computed)
            - 'epsilon': Perturbation parameter 1/κ
        """
        result = self.calculator.compute_ruin_probability(
            initial_wealth=initial_capital,
            annual_withdrawal=annual_withdrawal,
            years=years,
            return_details=False,
        )

        # Add survival probability for consistency with other strategies
        result["survival_probability"] = 1.0 - result["ruin_probability"]

        return result

    def get_strategy_name(self) -> str:
        """Return human-readable strategy name."""
        return (
            f"HestonStiffPerturbative(κ={self.market_environment.get_kappa():.2f}, "
            f"θ={self.market_environment.get_long_var():.4f}, "
            f"σ_v={self.market_environment.get_vol_of_vol():.4f}, "
            f"ρ={self.market_environment.get_correlation():.2f})"
        )
