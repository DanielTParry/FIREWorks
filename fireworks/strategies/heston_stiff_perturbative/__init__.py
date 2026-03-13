"""Heston Stiff Perturbative Strategy module.

Provides analytical approximate solution for probability of ruin
using perturbative expansion in 1/κ (inverse mean reversion speed)
for the Heston model with fast mean-reverting volatility.

References:
- Heston (1993) - closed-form solution for European options
- Stochastic volatility with fast mean reversion (κ >> typical drift/vol scales)
- Perturbative methods for small parameter ε = 1/κ
"""

from .strategy import HestonStiffPerturbativeStrategy
from .calculator import HestonStiffPerturbativeCalculator
from .models import (
    AbstractHestonMarketEnvironment,
    ConstantHestonMarketEnvironment,
    HestonMarketEnvironmentFactory,
)

__all__ = [
    "HestonStiffPerturbativeStrategy",
    "HestonStiffPerturbativeCalculator",
    "AbstractHestonMarketEnvironment",
    "ConstantHestonMarketEnvironment",
    "HestonMarketEnvironmentFactory",
]
