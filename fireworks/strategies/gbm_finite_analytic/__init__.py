"""GBM Finite Analytic Strategy module.

Provides exact analytical solution for probability of ruin over finite horizons
using spectral decomposition method.
"""

from .strategy import GBMFiniteAnalyticStrategy
from .calculator import GBMFiniteAnalyticCalculator
from .models import (
    AbstractMarketEnvironment,
    ConstantMarketEnvironment,
    MarketEnvironmentFactory,
    AbstractConsumptionModel,
    ConstantConsumptionModel,
    ConsumptionModelFactory,
)

__all__ = [
    "GBMFiniteAnalyticStrategy",
    "GBMFiniteAnalyticCalculator",
    "AbstractMarketEnvironment",
    "ConstantMarketEnvironment",
    "MarketEnvironmentFactory",
    "AbstractConsumptionModel",
    "ConstantConsumptionModel",
    "ConsumptionModelFactory",
]
