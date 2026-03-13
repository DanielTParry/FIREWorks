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


class AbstractMarketEnvironment(ABC):
    """Abstract base class for market environments defining mean return and variance."""

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


class ConstantMarketEnvironment(AbstractMarketEnvironment):
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


class HestonMarketEnvironment(AbstractMarketEnvironment):
    """Market environment with Heston stochastic volatility dynamics.
    
    Maintains internal variance state that evolves according to Heston mean reversion.
    Compatible with MCStrategy for stochastic volatility simulation.
    
    Dynamics:
        dμ/μ = μ dt + √v dW_S
        dv = κ(θ - v) dt + ξ√v dW_v
    where dW_S · dW_v = ρ dt
    """

    def __init__(
        self,
        mu: float,
        kappa: float,
        theta: float,
        xi: float,
        initial_variance: float,
        rho: float = 0.0,
    ) -> None:
        """
        Initialize Heston market environment.

        Args:
            mu: Constant drift (mean return)
            kappa: Mean reversion speed
            theta: Long-term variance (mean reversion target)
            xi: Volatility of volatility (vol-of-vol)
            initial_variance: Initial variance v(0)
            rho: Correlation between stock and variance innovations

        Raises:
            ValueError: If parameters violate Heston constraints
        """
        if mu <= -1:
            raise ValueError(f"mu must be > -1, got {mu}")
        if kappa <= 0:
            raise ValueError(f"kappa must be > 0, got {kappa}")
        if theta <= 0:
            raise ValueError(f"theta must be > 0, got {theta}")
        if xi <= 0:
            raise ValueError(f"xi must be > 0, got {xi}")
        if initial_variance <= 0:
            raise ValueError(f"initial_variance must be > 0, got {initial_variance}")
        if not (-1 <= rho <= 1):
            raise ValueError(f"rho must be in [-1, 1], got {rho}")
        
        # Feller condition: 2*κ*θ ≥ ξ² ensures variance process doesn't hit zero
        feller_lhs = 2 * kappa * theta
        feller_rhs = xi * xi
        if feller_lhs < feller_rhs:
            raise ValueError(
                f"Feller condition violated: 2*κ*θ = {feller_lhs} < ξ² = {feller_rhs}. "
                f"Increase kappa or theta, or decrease xi."
            )

        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho
        
        # Current state (evolves during simulation)
        self.v_current = initial_variance
        self.v_initial = initial_variance

    def get_mean(self, t: float) -> float:
        """Get the constant mean return."""
        return self.mu

    def get_variance(self, t: float) -> float:
        """Get the current variance state."""
        return self.v_current

    def update_variance(self, v_new: float) -> None:
        """
        Update current variance state during MC simulation.
        
        Called by MC simulator after each step to maintain state.
        
        Args:
            v_new: New variance value
        """
        self.v_current = max(v_new, 0.0)  # Feller boundary: v ≥ 0

    def reset(self) -> None:
        """Reset variance to initial state (for new MC simulation)."""
        self.v_current = self.v_initial


class AbstractConsumptionModel(ABC):
    """Abstract base class for consumption models C(t)."""

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


class ConstantConsumptionModel(AbstractConsumptionModel):
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

    @staticmethod
    def heston(
        mu: float,
        kappa: float,
        theta: float,
        xi: float,
        initial_variance: float,
        rho: float = 0.0,
    ) -> HestonMarketEnvironment:
        """
        Create a Heston stochastic volatility market environment.
        
        Args:
            mu: Constant drift
            kappa: Mean reversion speed
            theta: Long-term variance
            xi: Volatility of volatility
            initial_variance: Initial variance
            rho: Correlation (default 0, no leverage)
            
        Returns:
            HestonMarketEnvironment
        """
        return HestonMarketEnvironment(
            mu=mu,
            kappa=kappa,
            theta=theta,
            xi=xi,
            initial_variance=initial_variance,
            rho=rho,
        )


class ConsumptionModelFactory:
    """Factory for creating consumption models."""

    @staticmethod
    def constant(annual_consumption: float) -> ConstantConsumptionModel:
        """Create a constant consumption model."""
        return ConstantConsumptionModel(annual_consumption)
