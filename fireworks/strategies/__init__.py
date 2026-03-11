"""Withdrawal strategies module.

This module contains various FIRE withdrawal strategies for analyzing
probability of ruin under different market regimes.
"""

from .mc_strategy import MCStrategy

__all__ = ["MCStrategy"]
