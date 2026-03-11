"""GBM Infinite Analytical Strategy: Closed-form ruin probability under GBM.

This module provides an analytical (non-simulation) strategy for calculating
the probability of ruin under a Geometric Brownian Motion with constant
parameters and constant withdrawal, over an infinite time horizon.

The solution is based on the incomplete Gamma function:
    P(ruin) = 1 - Γ(a, x) / Γ(a)

Where:
    a = (2μ / σ²) - 1
    x = 2w / σ²
    μ = mean return
    σ² = variance
    w = withdrawal rate
"""

from .strategy import GBMInfiniteAnalyticStrategy
from .calculator import GBMInfiniteAnalyticCalculator
from .models import (
    AbstractMarketEnvironment,
    ConstantMarketEnvironment,
    MarketEnvironmentFactory,
    AbstractConsumptionModel,
    ConstantConsumptionModel,
    ConsumptionModelFactory,
)

__all__ = [
    "GBMInfiniteAnalyticStrategy",
    "GBMInfiniteAnalyticCalculator",
    "AbstractMarketEnvironment",
    "ConstantMarketEnvironment",
    "MarketEnvironmentFactory",
    "AbstractConsumptionModel",
    "ConstantConsumptionModel",
    "ConsumptionModelFactory",
]
