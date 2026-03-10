"""Model components for MC Strategy: Mean, Variance, and Consumption models.

These models define the parameter functions in the SDE:
    dP_t = (P_t * μ(t) - C(t)) * dt + P_t * √(v(t)) * dW_t

Where:
    μ(t) = mean return at time t
    v(t) = variance at time t
    C(t) = consumption at time t
"""

from abc import ABC, abstractmethod


class MeanModel(ABC):
    """Base class for mean return models μ(t)."""

    @abstractmethod
    def get_return(self, t):
        """
        Get the mean return at time t.

        Args:
            t: Time (years)

        Returns:
            Mean return at time t
        """
        pass


class ConstantMeanModel(MeanModel):
    """Mean return is constant over time."""

    def __init__(self, mu):
        """
        Initialize constant mean model.

        Args:
            mu: Constant mean return (e.g., 0.07 for 7%)
        """
        self.mu = mu

    def get_return(self, t):
        """Get the mean return (constant)."""
        return self.mu


class VarianceModel(ABC):
    """Base class for variance models v(t)."""

    @abstractmethod
    def get_variance(self, t):
        """
        Get the variance at time t.

        Args:
            t: Time (years)

        Returns:
            Variance at time t
        """
        pass


class ConstantVarianceModel(VarianceModel):
    """Variance is constant over time."""

    def __init__(self, variance):
        """
        Initialize constant variance model.

        Args:
            variance: Constant variance (e.g., 0.04 for 4%)
        """
        self.variance = variance

    def get_variance(self, t):
        """Get the variance (constant)."""
        return self.variance


class ConsumptionModel(ABC):
    """Base class for consumption models C(t)."""

    @abstractmethod
    def get_consumption(self, t, portfolio_value, mu_t=None, v_t=None):
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

    def __init__(self, annual_consumption):
        """
        Initialize constant consumption model.

        Args:
            annual_consumption: Fixed annual consumption amount
        """
        self.annual_consumption = annual_consumption

    def get_consumption(self, t, portfolio_value=None, mu_t=None, v_t=None):
        """Get the consumption (constant)."""
        return self.annual_consumption


class ConstantRateConsumptionModel(ConsumptionModel):
    """Consumption as a constant percentage of portfolio value."""

    def __init__(self, withdrawal_rate):
        """
        Initialize constant rate consumption model.

        Args:
            withdrawal_rate: Withdrawal rate as fraction (e.g., 0.04 for 4%)
        """
        self.withdrawal_rate = withdrawal_rate

    def get_consumption(self, t, portfolio_value, mu_t=None, v_t=None):
        """Get the consumption (percentage of portfolio)."""
        if portfolio_value is None:
            raise ValueError("portfolio_value required for ConstantRateConsumptionModel")
        return self.withdrawal_rate * portfolio_value


class StateAdjustedConsumptionModel(ConsumptionModel):
    """Consumption adjusted based on market conditions (return and volatility).
    
    Adjusts withdrawal rate based on current expected return and volatility.
    Higher returns or lower volatility increases consumption; lower returns 
    or higher volatility decreases consumption.
    """

    def __init__(self, base_withdrawal_rate, baseline_mu, baseline_v):
        """
        Initialize state-adjusted consumption model.

        Args:
            base_withdrawal_rate: Base withdrawal rate (e.g., 0.04)
            baseline_mu: Baseline mean return for normalization
            baseline_v: Baseline variance for normalization
        """
        self.base_withdrawal_rate = base_withdrawal_rate
        self.baseline_mu = baseline_mu
        self.baseline_v = baseline_v

    def get_consumption(self, t, portfolio_value, mu_t=None, v_t=None):
        """Get consumption adjusted by current market conditions."""
        if portfolio_value is None:
            raise ValueError("portfolio_value required for StateAdjustedConsumptionModel")
        
        if mu_t is None or v_t is None:
            # Fall back to constant rate if market data not provided
            return self.base_withdrawal_rate * portfolio_value
        
        # Adjust withdrawal rate based on return and volatility
        # Higher mu increases consumption, higher v decreases it
        mu_adjustment = mu_t / self.baseline_mu if self.baseline_mu != 0 else 1.0
        v_adjustment = self.baseline_v / v_t if v_t != 0 else 1.0
        
        adjusted_rate = self.base_withdrawal_rate * mu_adjustment * v_adjustment
        return adjusted_rate * portfolio_value


# Factory functions for creating models

class MeanModelFactory:
    """Factory for creating mean models."""

    @staticmethod
    def constant(mu):
        """Create a constant mean model."""
        return ConstantMeanModel(mu)


class VarianceModelFactory:
    """Factory for creating variance models."""

    @staticmethod
    def constant(variance):
        """Create a constant variance model."""
        return ConstantVarianceModel(variance)


class ConsumptionModelFactory:
    """Factory for creating consumption models."""

    @staticmethod
    def constant(annual_consumption):
        """Create a constant consumption model."""
        return ConstantConsumptionModel(annual_consumption)

    @staticmethod
    def constant_rate(withdrawal_rate):
        """Create a constant rate consumption model."""
        return ConstantRateConsumptionModel(withdrawal_rate)

    @staticmethod
    def state_adjusted(base_withdrawal_rate, baseline_mu, baseline_v):
        """Create a state-adjusted consumption model that responds to market conditions."""
        return StateAdjustedConsumptionModel(base_withdrawal_rate, baseline_mu, baseline_v)
