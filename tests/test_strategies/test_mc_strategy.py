"""Tests for MC Strategy."""

import unittest
import numpy as np
from fireworks.strategies.mc_strategy import (
    MCStrategy,
    MCSimulator,
    ConstantMarketEnvironment,
    ConstantConsumptionModel,
    MarketEnvironmentFactory,
    ConsumptionModelFactory,
    AbstractMarketEnvironment,
    AbstractConsumptionModel,
)


class TestConstantMarketEnvironment(unittest.TestCase):
    """Tests for ConstantMarketEnvironment."""

    def setUp(self):
        """Set up test fixtures."""
        self.mu = 0.07
        self.variance = 0.04
        self.env = ConstantMarketEnvironment(self.mu, self.variance)

    def test_constant_market_environment_mean(self):
        """Mean should be constant across time."""
        self.assertEqual(self.env.get_mean(0), self.mu)
        self.assertEqual(self.env.get_mean(10), self.mu)
        self.assertEqual(self.env.get_mean(100), self.mu)

    def test_constant_market_environment_variance(self):
        """Variance should be constant across time."""
        self.assertEqual(self.env.get_variance(0), self.variance)
        self.assertEqual(self.env.get_variance(10), self.variance)
        self.assertEqual(self.env.get_variance(100), self.variance)


class TestMarketEnvironmentFactory(unittest.TestCase):
    """Tests for MarketEnvironmentFactory."""

    def test_factory_constant(self):
        """Factory should create ConstantMarketEnvironment."""
        env = MarketEnvironmentFactory.constant(0.07, 0.04)
        self.assertIsInstance(env, ConstantMarketEnvironment)
        self.assertEqual(env.get_mean(0), 0.07)
        self.assertEqual(env.get_variance(0), 0.04)


class TestConstantConsumptionModel(unittest.TestCase):
    """Tests for ConstantConsumptionModel."""

    def setUp(self):
        """Set up test fixtures."""
        self.consumption = 50000
        self.model = ConstantConsumptionModel(self.consumption)

    def test_constant_consumption(self):
        """Consumption should be constant."""
        self.assertEqual(self.model.get_consumption(0, 1000000), self.consumption)
        self.assertEqual(self.model.get_consumption(10, 1000000), self.consumption)
        self.assertEqual(self.model.get_consumption(0, None), self.consumption)

    def test_consumption_ignores_market_params(self):
        """Constant consumption should ignore market parameters."""
        c = self.model.get_consumption(5, 500000, mu_t=0.1, v_t=0.05)
        self.assertEqual(c, self.consumption)


class TestConsumptionModelFactory(unittest.TestCase):
    """Tests for ConsumptionModelFactory."""

    def test_factory_constant(self):
        """Factory should create ConstantConsumptionModel."""
        model = ConsumptionModelFactory.constant(50000)
        self.assertIsInstance(model, ConstantConsumptionModel)
        self.assertEqual(model.get_consumption(0), 50000)


class TestMCSimulator(unittest.TestCase):
    """Tests for MCSimulator."""

    def setUp(self):
        """Set up test fixtures."""
        self.market_env = MarketEnvironmentFactory.constant(0.07, 0.04)
        self.consumption = ConsumptionModelFactory.constant(10000)
        self.simulator = MCSimulator(self.market_env, self.consumption)

    def test_simulate_shape(self):
        """Simulated paths should have correct shape."""
        initial = 1000000
        years = 10
        num_sims = 100
        num_steps = 10

        results = self.simulator.simulate(initial, years, num_sims, num_steps)

        # Shape should be (num_sims, num_steps + 1)
        self.assertEqual(results['paths'].shape, (num_sims, num_steps + 1))
        self.assertEqual(len(results['final_values']), num_sims)
        self.assertEqual(len(results['ruin_steps']), num_sims)

    def test_simulate_initial_values(self):
        """First column of paths should be initial capital."""
        initial = 1000000
        results = self.simulator.simulate(initial, 10, 50, 10)

        np.testing.assert_array_equal(results['paths'][:, 0], initial)

    def test_simulate_default_steps(self):
        """Default num_steps should be equal to years."""
        results = self.simulator.simulate(1000000, 10, 50, num_steps=None)

        # Default should be int(years) = 10
        self.assertEqual(results['num_steps'], 10)
        self.assertEqual(results['paths'].shape[1], 11)

    def test_ruin_detection(self):
        """Ruin steps should be -1 for non-ruined, >= 0 for ruined."""
        results = self.simulator.simulate(1000000, 30, 100, 30)

        # All ruin_steps should be either -1 or positive
        self.assertTrue(np.all(results['ruin_steps'] >= -1))
        self.assertTrue(np.all(results['ruin_steps'] < results['num_steps'] + 1))

    def test_compute_ruin_probability_bounds(self):
        """Ruin probability should be between 0 and 1."""
        results = self.simulator.simulate(1000000, 10, 100, 10)
        prob = self.simulator.compute_ruin_probability(results)

        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_compute_ruin_probability_all_ruined(self):
        """If all paths ruined, probability should be 1.0."""
        results = self.simulator.simulate(1000, 30, 100, 30)
        prob = self.simulator.compute_ruin_probability(results)

        # With very small capital and long horizon, likely high ruin
        # (not guaranteed but very likely)
        self.assertGreater(prob, 0.5)

    def test_compute_statistics_keys(self):
        """Statistics dict should have all required keys."""
        results = self.simulator.simulate(1000000, 10, 100, 10)
        stats = self.simulator.compute_statistics(results)

        required_keys = [
            'ruin_probability',
            'mean_final_value',
            'median_final_value',
            'std_final_value',
            'min_final_value',
            'max_final_value',
            'mean_remaining_capital',
            'percentile_10',
            'percentile_90',
            'num_ruined',
            'num_non_ruined',
        ]

        for key in required_keys:
            self.assertIn(key, stats)

    def test_compute_statistics_consistency(self):
        """Statistics should be self-consistent."""
        results = self.simulator.simulate(1000000, 10, 100, 10)
        stats = self.simulator.compute_statistics(results)

        # num_ruined + num_non_ruined should equal total sims
        self.assertEqual(stats['num_ruined'] + stats['num_non_ruined'], 100)

        # ruin_probability should match num_ruined / total
        expected_prob = stats['num_ruined'] / 100
        self.assertAlmostEqual(stats['ruin_probability'], expected_prob)


class TestMCStrategy(unittest.TestCase):
    """Tests for MCStrategy."""

    def setUp(self):
        """Set up test fixtures."""
        self.initial_capital = 1000000
        self.annual_withdrawal = 40000
        self.years = 30
        self.num_simulations = 100

    def test_strategy_default_initialization(self):
        """Strategy should initialize with defaults."""
        strategy = MCStrategy()
        self.assertIsNotNone(strategy.market_environment)
        self.assertIsNotNone(strategy.consumption_model)
        self.assertEqual(strategy.num_simulations, 10000)

    def test_strategy_custom_initialization(self):
        """Strategy should initialize with custom models."""
        env = MarketEnvironmentFactory.constant(0.08, 0.05)
        consumption = ConsumptionModelFactory.constant(50000)
        strategy = MCStrategy(
            market_environment=env,
            consumption_model=consumption,
            num_simulations=5000,
        )

        self.assertEqual(strategy.market_environment, env)
        self.assertEqual(strategy.consumption_model, consumption)
        self.assertEqual(strategy.num_simulations, 5000)

    def test_calculate_ruin_probability_bounds(self):
        """Ruin probability should be between 0 and 1."""
        strategy = MCStrategy(num_simulations=self.num_simulations)
        prob = strategy.calculate_ruin_probability(
            self.initial_capital,
            self.annual_withdrawal,
            self.years,
        )

        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    def test_simulate_returns_dict(self):
        """Simulate should return a dictionary."""
        strategy = MCStrategy(num_simulations=self.num_simulations)
        results = strategy.simulate(
            self.initial_capital,
            self.annual_withdrawal,
            self.years,
        )

        self.assertIsInstance(results, dict)
        self.assertIn('paths', results)
        self.assertIn('final_values', results)
        self.assertIn('ruin_steps', results)
        self.assertIn('statistics', results)

    def test_simulate_statistics_included(self):
        """Simulate should include statistics."""
        strategy = MCStrategy(num_simulations=self.num_simulations)
        results = strategy.simulate(
            self.initial_capital,
            self.annual_withdrawal,
            self.years,
        )

        stats = results['statistics']
        self.assertIn('ruin_probability', stats)
        self.assertIn('mean_final_value', stats)

    def test_large_withdrawal_high_ruin(self):
        """Very large withdrawal should have high ruin probability."""
        strategy = MCStrategy(num_simulations=100)
        prob = strategy.calculate_ruin_probability(
            initial_capital=1000000,
            annual_withdrawal=200000,  # 20% withdrawal rate - very high
            years=30,
        )

        # Should have significant ruin probability
        self.assertGreater(prob, 0.3)

    def test_small_withdrawal_low_ruin(self):
        """Small withdrawal should have low ruin probability."""
        strategy = MCStrategy(num_simulations=100)
        prob = strategy.calculate_ruin_probability(
            initial_capital=1000000,
            annual_withdrawal=10000,  # 1% withdrawal rate - very low
            years=30,
        )

        # Should have low ruin probability
        self.assertLess(prob, 0.3)

    def test_zero_withdrawal(self):
        """Zero withdrawal should have zero ruin (growth-only)."""
        strategy = MCStrategy(num_simulations=100)
        prob = strategy.calculate_ruin_probability(
            initial_capital=1000000,
            annual_withdrawal=0,
            years=30,
        )

        # With positive expected return and no withdrawal, should not ruin
        self.assertEqual(prob, 0.0)

    def test_simulate_respects_num_simulations_override(self):
        """Simulate should respect num_simulations override."""
        strategy = MCStrategy(num_simulations=10)
        results = strategy.simulate(
            self.initial_capital,
            self.annual_withdrawal,
            self.years,
            num_simulations=50,
        )

        # Should use override, not default
        self.assertEqual(len(results['final_values']), 50)


class TestMCStrategyEdgeCases(unittest.TestCase):
    """Tests for edge cases in MC Strategy."""

    def test_single_simulation(self):
        """Should handle single simulation."""
        strategy = MCStrategy(num_simulations=1)
        results = strategy.simulate(1000000, 40000, 30)

        self.assertEqual(len(results['final_values']), 1)

    def test_short_time_horizon(self):
        """Should handle short time horizons."""
        strategy = MCStrategy(num_simulations=50)
        results = strategy.simulate(1000000, 40000, 1)

        self.assertGreater(len(results['paths'][0]), 1)

    def test_zero_variance(self):
        """Should handle zero variance (deterministic case)."""
        env = MarketEnvironmentFactory.constant(0.07, 0.0)
        strategy = MCStrategy(market_environment=env, num_simulations=10)
        results = strategy.simulate(1000000, 40000, 10)

        # With zero variance, all paths should be identical
        for i in range(1, len(results['final_values'])):
            self.assertAlmostEqual(
                results['final_values'][i],
                results['final_values'][0],
                places=5
            )


if __name__ == '__main__':
    unittest.main()
