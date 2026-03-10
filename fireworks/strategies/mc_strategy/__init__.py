"""MC Strategy: Monte Carlo simulation of withdrawal strategies.

This module provides a Monte Carlo-based strategy for analyzing probability of ruin
under stochastic portfolio dynamics modeled by:

    dP_t = (P_t * μ(t) - C(t)) * dt + P_t * √(v(t)) * dW_t

Where:
    P_t = portfolio value at time t
    μ(t) = mean return at time t (from MarketEnvironment)
    v(t) = variance at time t (from MarketEnvironment)
    C(t) = consumption at time t (from ConsumptionModel)
    dW_t = Wiener process increment
"""

from .strategy import MCStrategy
from .models import (
    MarketEnvironment,
    ConstantMarketEnvironment,
    RegimeSwitchingEnvironment,
    MarketEnvironmentFactory,
    ConsumptionModel,
    ConstantConsumptionModel,
    ConstantRateConsumptionModel,
    StateAdjustedConsumptionModel,
    ConsumptionModelFactory,
)
from .calculator import MCSimulator

__all__ = [
    "MCStrategy",
    "MCSimulator",
    "MarketEnvironment",
    "ConstantMarketEnvironment",
    "RegimeSwitchingEnvironment",
    "MarketEnvironmentFactory",
    "ConsumptionModel",
    "ConstantConsumptionModel",
    "ConstantRateConsumptionModel",
    "StateAdjustedConsumptionModel",
    "ConsumptionModelFactory",
]
