"""Monte Carlo simulation engine for the MC Strategy.

Simulates the SDE:
    dP_t = (P_t * μ(t) - C(t)) * dt + P_t * √(v(t)) * dW_t

Using Euler-Maruyama discretization:
    P_{t+Δt} = P_t + (P_t * μ(t) - C(t)) * Δt + P_t * √(v(t)) * √(Δt) * Z_t
"""

import numpy as np


class MCSimulator:
    """Monte Carlo simulator for portfolio dynamics."""

    def __init__(self, mean_model=None, variance_model=None, consumption_model=None, 
                 joint_parameter_model=None):
        """
        Initialize the simulator.

        Args:
            mean_model: MeanModel instance defining μ(t) (alternative to joint_parameter_model)
            variance_model: VarianceModel instance defining v(t) (alternative to joint_parameter_model)
            consumption_model: ConsumptionModel instance defining C(t)
            joint_parameter_model: JointParameterModel instance defining (μ(t), v(t)) together
                                   If provided, mean_model and variance_model are ignored.
        """
        self.mean_model = mean_model
        self.variance_model = variance_model
        self.consumption_model = consumption_model
        self.joint_parameter_model = joint_parameter_model
        
        # Validate that we have either joint model or independent models
        if joint_parameter_model is None and (mean_model is None or variance_model is None):
            raise ValueError(
                "Must provide either joint_parameter_model or both mean_model and variance_model"
            )

    def simulate(self, initial_capital, years, num_simulations, num_steps=None):
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

            # Get model values at this time
            if self.joint_parameter_model is not None:
                # Use joint model for cointegrated parameters
                mu_t, v_t = self.joint_parameter_model.get_parameters(t)
            else:
                # Use independent models
                mu_t = self.mean_model.get_return(t)
                v_t = self.variance_model.get_variance(t)
            
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

    def compute_ruin_probability(self, simulation_results):
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

    def compute_statistics(self, simulation_results):
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
