"""MC Strategy: Monte Carlo simulation of portfolio ruin probability."""

from typing import Dict, Any, Optional
from fireworks.core.base import AbstractStrategy
from .models import (
    MarketEnvironmentFactory,
    ConsumptionModelFactory,
    AbstractMarketEnvironment,
    AbstractConsumptionModel,
)
from .calculator import MCSimulator


class MCStrategy(AbstractStrategy):
    """
    Monte Carlo Strategy for analyzing withdrawal strategies.

    Uses Monte Carlo simulation to estimate probability of ruin under
    a stochastic differential equation model of portfolio dynamics.
    """

    def __init__(
        self,
        market_environment: Optional[AbstractMarketEnvironment] = None,
        consumption_model: Optional[AbstractConsumptionModel] = None,
        num_simulations: int = 10000,
        num_steps: Optional[int] = None,
    ) -> None:
        """
        Initialize MC Strategy.

        Args:
            market_environment: AbstractMarketEnvironment instance (default: constant 7% return, 4% variance)
            consumption_model: AbstractConsumptionModel instance (default: constant $0)
            num_simulations: Number of Monte Carlo paths
            num_steps: Number of time steps per year (default: 1 step per year)
        """
        # Set defaults if not provided
        self.market_environment = market_environment or MarketEnvironmentFactory.constant(0.07, 0.04)
        
        if consumption_model is None:
            self.consumption_model = ConsumptionModelFactory.constant(0)
        else:
            self.consumption_model = consumption_model

        self.num_simulations = num_simulations
        self.num_steps = num_steps

        self.simulator = MCSimulator(
            self.market_environment,
            self.consumption_model,
        )

    def calculate_ruin_probability(
        self,
        initial_capital: float,
        annual_withdrawal: float = 0,
        years: float = 30,
        num_simulations: Optional[int] = None,
    ) -> float:
        """
        Calculate the probability of ruin.

        Args:
            initial_capital: Starting portfolio value
            annual_withdrawal: Annual withdrawal amount (overrides consumption model if > 0)
            years: Time horizon in years
            num_simulations: Override default number of simulations

        Returns:
            Probability of ruin (float between 0 and 1)
        """
        # Use provided consumption or create one from annual_withdrawal
        if annual_withdrawal > 0:
            consumption_model = ConsumptionModelFactory.constant(annual_withdrawal)
        else:
            consumption_model = self.consumption_model

        simulator = MCSimulator(
            self.market_environment,
            consumption_model,
        )

        num_sims = num_simulations or self.num_simulations
        results = simulator.simulate(
            initial_capital,
            years,
            num_sims,
            num_steps=self.num_steps,
        )

        return simulator.compute_ruin_probability(results)

    def simulate(
        self,
        initial_capital: float,
        annual_withdrawal: float = 0,
        years: float = 30,
        num_simulations: Optional[int] = None,
        num_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo simulations for the strategy.

        Args:
            initial_capital: Starting portfolio value
            annual_withdrawal: Annual withdrawal amount (overrides consumption model if > 0)
            years: Time horizon in years
            num_simulations: Override default number of simulations
            num_steps: Override default number of time steps

        Returns:
            Dictionary with simulation results and statistics
        """
        # Use provided consumption or create one from annual_withdrawal
        if annual_withdrawal > 0:
            consumption_model = ConsumptionModelFactory.constant(annual_withdrawal)
        else:
            consumption_model = self.consumption_model

        simulator = MCSimulator(
            self.market_environment,
            consumption_model,
        )

        num_sims = num_simulations or self.num_simulations
        num_steps_to_use = num_steps or self.num_steps

        results = simulator.simulate(
            initial_capital,
            years,
            num_sims,
            num_steps=num_steps_to_use,
        )

        # Compute ruin probability and statistics
        ruin_probability = simulator.compute_ruin_probability(results)
        statistics = simulator.compute_statistics(results)

        # Return a clean result dictionary
        return {
            'ruin_probability': ruin_probability,
            'final_values': results['final_values'],
            'paths': results['paths'],
            'ruin_steps': results['ruin_steps'],
            'statistics': statistics,
            'num_simulations': num_sims,
            'num_steps': num_steps_to_use,
            'years': years,
        }
