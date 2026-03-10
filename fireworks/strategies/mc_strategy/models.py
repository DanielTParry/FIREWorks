"""Model components for MC Strategy: MarketEnvironment and Consumption models.

These models define the parameter functions in the SDE:
    dP_t = (P_t * μ(t) - C(t)) * dt + P_t * √(v(t)) * dW_t

Where:
    P_t = portfolio value at time t
    μ(t) = mean return at time t (from MarketEnvironment)
    v(t) = variance at time t (from MarketEnvironment)
    C(t) = consumption at time t (from ConsumptionModel)
"""

from abc import ABC, abstractmethod
from typing import Optional


class MarketEnvironment(ABC):
    """Base class for market environments defining mean return and variance."""

    @abstractmethod
    def get_mean(self, t: float) -> float:
        """
        Get the mean return at time t.

        Args:
            t: Time (years)

        Returns:
            Mean return at time t
        """
        pass

    @abstractmethod
    def get_variance(self, t: float) -> float:
        """
        Get the variance at time t.

        Args:
            t: Time (years)

        Returns:
            Variance at time t
        """
        pass


class ConstantMarketEnvironment(MarketEnvironment):
    """Market environment with constant mean and variance."""

    def __init__(self, mu: float, variance: float) -> None:
        """
        Initialize constant market environment.

        Args:
            mu: Constant mean return (e.g., 0.07 for 7%)
            variance: Constant variance (e.g., 0.04 for 4%)
        """
        self.mu = mu
        self.variance = variance

    def get_mean(self, t: float) -> float:
        """Get the mean return (constant)."""
        return self.mu

    def get_variance(self, t: float) -> float:
        """Get the variance (constant)."""
        return self.variance


class ConsumptionModel(ABC):
    """Base class for consumption models C(t)."""

    @abstractmethod
    def get_consumption(self, t: float, portfolio_value: Optional[float], 
                       mu_t: Optional[float] = None, v_t: Optional[float] = None) -> float:
        """
        Get consumption at time t.

        Args:
            t: Time (years)
            portfolio_value: Current portfolio value (used by some models)
            mu_t: Mean return at time t (optional, for state-dependent models)
            v_t: Variance at time t (optional, for state-dependent models)

        Returns:
            Consumption at time t
        """
        pass


class ConstantConsumptionModel(ConsumptionModel):
    """Fixed annual consumption."""

    def __init__(self, annual_consumption: float) -> None:
        """
        Initialize constant consumption model.

        Args:
            annual_consumption: Fixed annual consumption amount
        """
        self.annual_consumption = annual_consumption

    def get_consumption(self, t: float, portfolio_value: Optional[float] = None, 
                       mu_t: Optional[float] = None, v_t: Optional[float] = None) -> float:
        """Get the consumption (constant)."""
        return self.annual_consumption


# Factories for creating models

class MarketEnvironmentFactory:
    """Factory for creating market environments."""

    @staticmethod
    def constant(mu: float, variance: float) -> ConstantMarketEnvironment:
        """Create a constant market environment."""
        return ConstantMarketEnvironment(mu, variance)


class ConsumptionModelFactory:
    """Factory for creating consumption models."""

    @staticmethod
    def constant(annual_consumption: float) -> ConstantConsumptionModel:
        """Create a constant consumption model."""
        return ConstantConsumptionModel(annual_consumption)
