"""Monte Carlo simulation engine for the MC Strategy.

Simulates the SDE:
    dP_t = (P_t * μ(t) - C(t)) * dt + P_t * √(v(t)) * dW_t

Using Euler-Maruyama discretization:
    P_{t+Δt} = P_t + (P_t * μ(t) - C(t)) * Δt + P_t * √(v(t)) * √(Δt) * Z_t
"""

from typing import Dict, Any
import numpy as np


class MCSimulator:
    """Monte Carlo simulator for portfolio dynamics."""

    def __init__(self, market_environment: 'AbstractMarketEnvironment', 
                 consumption_model: 'AbstractConsumptionModel') -> None:
        """
        Initialize the simulator.

        Args:
            market_environment: AbstractMarketEnvironment instance defining μ(t) and v(t)
            consumption_model: AbstractConsumptionModel instance defining C(t)
        """
        self.market_environment = market_environment
        self.consumption_model = consumption_model

    def simulate(self, initial_capital: float, years: float, num_simulations: int, 
                 num_steps: int = None) -> Dict[str, Any]:
        """
        Run Monte Carlo simulations of portfolio dynamics.

        Args:
            initial_capital: Starting portfolio value
            years: Time horizon in years
            num_simulations: Number of simulation paths
            num_steps: Number of time steps (default: yearly steps)

        Returns:
            Dictionary with simulation results:
                - 'paths': array of shape (num_simulations, num_steps+1)
                - 'final_values': array of final portfolio values
                - 'ruin_steps': array of time steps when ruin occurred (-1 if no ruin)
                - 'num_steps': number of time steps
                - 'dt': time step size
        """
        if num_steps is None:
            num_steps = int(years)  # Default to yearly steps

        dt = years / num_steps
        time_grid = np.linspace(0, years, num_steps + 1)

        # Storage for all paths
        paths = np.zeros((num_simulations, num_steps + 1))
        paths[:, 0] = initial_capital

        ruin_steps = np.full(num_simulations, -1, dtype=int)  # -1 means no ruin

        # Generate random normal increments
        dW = np.random.randn(num_simulations, num_steps) * np.sqrt(dt)

        # Simulate each path
        for step in range(num_steps):
            t = time_grid[step]
            P_t = paths[:, step]

            # Get market parameters at this time
            mu_t = self.market_environment.get_mean(t)
            v_t = self.market_environment.get_variance(t)
            C_t = self.consumption_model.get_consumption(t, P_t, mu_t=mu_t, v_t=v_t)

            # Euler-Maruyama step
            # dP = (P * mu - C) * dt + P * sqrt(v) * dW
            drift = (P_t * mu_t - C_t) * dt
            diffusion = P_t * np.sqrt(v_t) * dW[:, step]

            P_next = P_t + drift + diffusion

            # Record ruin (first time portfolio goes negative)
            newly_ruined = (P_next < 0) & (ruin_steps == -1)
            ruin_steps[newly_ruined] = step + 1

            # Keep negative values for path recording (tracks true process)
            paths[:, step + 1] = P_next

        return {
            'paths': paths,
            'final_values': paths[:, -1],
            'ruin_steps': ruin_steps,
            'num_steps': num_steps,
            'dt': dt,
            'time_grid': time_grid,
        }

    def compute_ruin_probability(self, simulation_results: Dict[str, Any]) -> float:
        """
        Compute probability of ruin from simulation results.

        Args:
            simulation_results: Dictionary returned from simulate()

        Returns:
            Probability of ruin (float between 0 and 1)
        """
        ruin_steps = simulation_results['ruin_steps']
        num_paths = len(ruin_steps)
        num_ruined = np.sum(ruin_steps >= 0)
        return num_ruined / num_paths if num_paths > 0 else 0.0

    def compute_statistics(self, simulation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute various statistics from simulation results.

        Args:
            simulation_results: Dictionary returned from simulate()

        Returns:
            Dictionary with statistics
        """
        final_values = simulation_results['final_values']
        ruin_steps = simulation_results['ruin_steps']

        # Only consider non-ruined paths for positive statistics
        non_ruined = final_values[ruin_steps < 0]

        return {
            'ruin_probability': self.compute_ruin_probability(simulation_results),
            'mean_final_value': np.mean(final_values),
            'median_final_value': np.median(final_values),
            'std_final_value': np.std(final_values),
            'min_final_value': np.min(final_values),
            'max_final_value': np.max(final_values),
            'mean_remaining_capital': np.mean(non_ruined) if len(non_ruined) > 0 else 0,
            'percentile_10': np.percentile(final_values, 10),
            'percentile_90': np.percentile(final_values, 90),
            'num_ruined': np.sum(ruin_steps >= 0),
            'num_non_ruined': len(non_ruined),
        }
