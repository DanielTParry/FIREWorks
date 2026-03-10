"""MC Strategy: Monte Carlo simulation of withdrawal strategies.

This module provides a Monte Carlo-based strategy for analyzing probability of ruin
under stochastic portfolio dynamics modeled by:

    dP_t = (P_t * μ(t) - C(t)) * dt + P_t * √(v(t)) * dW_t

Where:
    P_t = portfolio value at time t
    μ(t) = mean return at time t
    v(t) = variance at time t
    C(t) = consumption at time t
    dW_t = Wiener process increment

Supports both independent parameter models (for flexibility) and joint parameter models 
(to capture cointegration between returns and volatility).
"""

from .strategy import MCStrategy
from .models import (
    MeanModel,
    ConstantMeanModel,
    MeanModelFactory,
    VarianceModel,
    ConstantVarianceModel,
    VarianceModelFactory,
    ConsumptionModel,
    ConstantConsumptionModel,
    ConstantRateConsumptionModel,
    StateAdjustedConsumptionModel,
    ConsumptionModelFactory,
    JointParameterModel,
    ConstantJointModel,
    RegimeSwitchingModel,
    CorrelatedModel,
    JointParameterModelFactory,
)
from .calculator import MCSimulator

__all__ = [
    "MCStrategy",
    "MCSimulator",
    "MeanModel",
    "ConstantMeanModel",
    "MeanModelFactory",
    "VarianceModel",
    "ConstantVarianceModel",
    "VarianceModelFactory",
    "ConsumptionModel",
    "ConstantConsumptionModel",
    "ConstantRateConsumptionModel",
    "StateAdjustedConsumptionModel",
    "ConsumptionModelFactory",
    "JointParameterModel",
    "ConstantJointModel",
    "RegimeSwitchingModel",
    "CorrelatedModel",
    "JointParameterModelFactory",
]
