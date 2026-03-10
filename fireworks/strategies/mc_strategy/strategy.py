"""MC Strategy: Monte Carlo simulation of portfolio ruin probability."""

from fireworks.core.base import BaseStrategy
from .models import (
    MeanModelFactory,
    VarianceModelFactory,
    ConsumptionModelFactory,
)
from .calculator import MCSimulator


class MCStrategy(BaseStrategy):
    """
    Monte Carlo Strategy for analyzing withdrawal strategies.

    Uses Monte Carlo simulation to estimate probability of ruin under
    a stochastic differential equation model of portfolio dynamics.
    
    Supports both independent models (maximum flexibility) and joint 
    parameter models (to capture cointegration between returns and volatility).
    """

    def __init__(
        self,
        mean_model=None,
        variance_model=None,
        consumption_model=None,
        joint_parameter_model=None,
        num_simulations=10000,
        num_steps=None,
    ):
        """
        Initialize MC Strategy.

        Args:
            mean_model: MeanModel instance (ignored if joint_parameter_model provided)
            variance_model: VarianceModel instance (ignored if joint_parameter_model provided)
            consumption_model: ConsumptionModel instance (default: constant $0)
            joint_parameter_model: JointParameterModel for cointegrated return/variance
            num_simulations: Number of Monte Carlo paths
            num_steps: Number of time steps per year (default: 1 step per year)
        """
        # Validate inputs
        if joint_parameter_model is None:
            if mean_model is None or variance_model is None:
                # Set defaults if not provided
                from .models import MeanModelFactory, VarianceModelFactory
                self.mean_model = mean_model or MeanModelFactory.constant(0.07)
                self.variance_model = variance_model or VarianceModelFactory.constant(0.04)
            else:
                self.mean_model = mean_model
                self.variance_model = variance_model
            self.joint_parameter_model = None
        else:
            self.mean_model = None
            self.variance_model = None
            self.joint_parameter_model = joint_parameter_model

        if consumption_model is None:
            from .models import ConsumptionModelFactory
            self.consumption_model = ConsumptionModelFactory.constant(0)
        else:
            self.consumption_model = consumption_model

        self.num_simulations = num_simulations
        self.num_steps = num_steps

        self.simulator = MCSimulator(
            mean_model=self.mean_model,
            variance_model=self.variance_model,
            consumption_model=self.consumption_model,
            joint_parameter_model=self.joint_parameter_model,
        )

    def calculate_ruin_probability(
        self,
        initial_capital,
        annual_withdrawal=0,
        years=30,
        num_simulations=None,
    ):
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
            from .models import ConsumptionModelFactory
            consumption_model = ConsumptionModelFactory.constant(annual_withdrawal)
        else:
            consumption_model = self.consumption_model

        simulator = MCSimulator(
            mean_model=self.mean_model,
            variance_model=self.variance_model,
            consumption_model=consumption_model,
            joint_parameter_model=self.joint_parameter_model,
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
        initial_capital,
        annual_withdrawal=0,
        years=30,
        num_simulations=None,
        num_steps=None,
    ):
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
            from .models import ConsumptionModelFactory
            consumption_model = ConsumptionModelFactory.constant(annual_withdrawal)
        else:
            consumption_model = self.consumption_model

        simulator = MCSimulator(
            mean_model=self.mean_model,
            variance_model=self.variance_model,
            consumption_model=consumption_model,
            joint_parameter_model=self.joint_parameter_model,
        )

        num_sims = num_simulations or self.num_simulations
        num_steps_to_use = num_steps or self.num_steps

        results = simulator.simulate(
            initial_capital,
            years,
            num_sims,
            num_steps=num_steps_to_use,
        )

        # Add statistics to results
        results['statistics'] = simulator.compute_statistics(results)

        return results
