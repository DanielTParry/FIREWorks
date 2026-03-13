"""Heston market environment models.

Defines market parameters for the Heston stochastic volatility model:
dS/S = μ dt + √v dW_S
dv = κ(θ - v) dt + σ_v √v dW_v
where dW_S · dW_v = ρ dt
"""

from abc import ABC, abstractmethod
from typing import Optional


class AbstractHestonMarketEnvironment(ABC):
    """Abstract base for Heston market environments with stochastic volatility."""

    @abstractmethod
    def get_mu(self, t: Optional[float] = None) -> float:
        """Stock drift (annual). Can be time-dependent."""
        pass

    @abstractmethod
    def get_initial_variance(self, t: Optional[float] = None) -> float:
        """Initial variance v(0) = σ₀². Can be time-dependent."""
        pass

    @abstractmethod
    def get_kappa(self) -> float:
        """Mean reversion speed κ > 0. Controls timescale κ⁻¹."""
        pass

    @abstractmethod
    def get_long_var(self) -> float:
        """Long-term variance θ > 0. Mean reversion target."""
        pass

    @abstractmethod
    def get_vol_of_vol(self) -> float:
        """Volatility of volatility σ_v > 0. Controls variance diffusion."""
        pass

    @abstractmethod
    def get_correlation(self) -> float:
        """Correlation ρ ∈ [-1, 1]. Leverage: dW_S · dW_v = ρ dt."""
        pass


class ConstantHestonMarketEnvironment(AbstractHestonMarketEnvironment):
    """Heston market with constant parameters."""

    def __init__(
        self,
        mu: float,
        initial_variance: float,
        kappa: float,
        long_var: float,
        vol_of_vol: float,
        correlation: float = 0.0,
    ):
        """
        Initialize constant Heston market environment.

        Args:
            mu: Stock drift (annual return)
            initial_variance: Initial variance v(0)
            kappa: Mean reversion speed (annual)
            long_var: Long-term variance θ
            vol_of_vol: Volatility of volatility σ_v
            correlation: Stock-vol correlation ρ (default 0, no leverage)

        Raises:
            ValueError: If parameters violate Heston constraints
        """
        if mu <= -1:  # Allow negative drift, but not collapse
            raise ValueError(f"mu must be > -1, got {mu}")
        if initial_variance <= 0:
            raise ValueError(f"initial_variance must be > 0, got {initial_variance}")
        if kappa <= 0:
            raise ValueError(f"kappa must be > 0, got {kappa}")
        if long_var <= 0:
            raise ValueError(f"long_var must be > 0, got {long_var}")
        if vol_of_vol <= 0:
            raise ValueError(f"vol_of_vol must be > 0, got {vol_of_vol}")
        if not (-1 <= correlation <= 1):
            raise ValueError(f"correlation must be in [-1, 1], got {correlation}")

        self.mu = mu
        self.initial_variance = initial_variance
        self.kappa = kappa
        self.long_var = long_var
        self.vol_of_vol = vol_of_vol
        self.correlation = correlation

    def get_mu(self, t: Optional[float] = None) -> float:
        return self.mu

    def get_initial_variance(self, t: Optional[float] = None) -> float:
        return self.initial_variance

    def get_kappa(self) -> float:
        return self.kappa

    def get_long_var(self) -> float:
        return self.long_var

    def get_vol_of_vol(self) -> float:
        return self.vol_of_vol

    def get_correlation(self) -> float:
        return self.correlation


class HestonMarketEnvironmentFactory:
    """Factory for creating standard Heston market environments."""

    @staticmethod
    def constant(
        mu: float,
        initial_variance: float,
        kappa: float,
        long_var: float,
        vol_of_vol: float,
        correlation: float = 0.0,
    ) -> ConstantHestonMarketEnvironment:
        """
        Create constant Heston market environment.

        Args:
            mu: Annual stock drift
            initial_variance: Initial variance
            kappa: Mean reversion speed
            long_var: Long-term variance
            vol_of_vol: Volatility of volatility
            correlation: Stock-vol correlation

        Returns:
            ConstantHestonMarketEnvironment
        """
        return ConstantHestonMarketEnvironment(
            mu=mu,
            initial_variance=initial_variance,
            kappa=kappa,
            long_var=long_var,
            vol_of_vol=vol_of_vol,
            correlation=correlation,
        )
