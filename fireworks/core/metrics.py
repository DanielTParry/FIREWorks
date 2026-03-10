"""Metrics and calculations: probability of ruin, withdrawal variance, etc."""


def calculate_ruin_probability(simulation_results):
    """
    Calculate probability of ruin from simulation results.

    A portfolio is considered ruined when it reaches zero or negative balance.

    Args:
        simulation_results: Array of final portfolio values

    Returns:
        Probability of ruin (float between 0 and 1)
    """
    num_ruin_paths = sum(1 for val in simulation_results if val <= 0)
    return num_ruin_paths / len(simulation_results) if simulation_results else 0.0


def calculate_withdrawal_variance(withdrawal_paths):
    """
    Calculate variance in annual withdrawals.

    Args:
        withdrawal_paths: Array or matrix of withdrawal amounts over time

    Returns:
        Withdrawal variance statistics
    """
    pass


def calculate_remaining_capital(simulation_results):
    """
    Calculate statistics on leftover capital at end of period.

    Args:
        simulation_results: Array of final portfolio values

    Returns:
        Statistics dict (mean, median, percentiles, etc.)
    """
    pass
