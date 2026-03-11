"""Tests for GBM Infinite Analytical Strategy."""

import unittest
import numpy as np
from fireworks.strategies.gbm_infinite_analytic import (
    GBMInfiniteAnalyticStrategy,
    GBMInfiniteAnalyticCalculator,
    ConstantMarketEnvironment,
    ConstantConsumptionModel,
    MarketEnvironmentFactory,
    ConsumptionModelFactory,
)


class TestGBMInfiniteAnalyticCalculator(unittest.TestCase):
    """Tests for GBMInfiniteAnalyticCalculator."""

    def setUp(self):
        """Set up test fixtures."""
        self.mu = 0.07
        self.variance = 0.04
        self.market_env = ConstantMarketEnvironment(self.mu, self.variance)
        self.consumption = ConstantConsumptionModel(10000)
        self.calculator = GBMInfiniteAnalyticCalculator(self.market_env, self.consumption)

    def test_ruin_probability_bounds(self):
        """Ruin probability should be between 0 and 1."""
        for initial_capital in [100000, 500000, 1000000]:
            for withdrawal in [0, 5000, 10000, 20000, 50000]:
                prob = self.calculator.calculate_ruin_probability_infinite(
                    initial_capital, withdrawal
                )
                self.assertGreaterEqual(prob, 0.0)
                self.assertLessEqual(prob, 1.0)

    def test_zero_withdrawal_zero_ruin(self):
        """Zero withdrawal should have zero ruin probability."""
        prob = self.calculator.calculate_ruin_probability_infinite(1000000, 0)
        self.assertAlmostEqual(prob, 0.0, places=5)

    def test_high_withdrawal_high_ruin(self):
        """High withdrawal rate should have high ruin probability."""
        # Withdrawal rate of 50% should have very high ruin probability
        prob = self.calculator.calculate_ruin_probability_infinite(100000, 50000)
        self.assertGreater(prob, 0.9)

    def test_withdrawal_exceeds_return(self):
        """Withdrawal >= return should give ruin probability near 1."""
        # If withdrawal is >= mean return * capital, ruin is nearly certain
        initial_capital = 100000
        annual_return = initial_capital * self.mu  # 7000
        withdrawal_at_return = annual_return  # 7000
        prob = self.calculator.calculate_ruin_probability_infinite(
            initial_capital, withdrawal_at_return
        )
        self.assertGreater(prob, 0.5)  # Should be substantial

    def test_typical_fire_scenario(self):
        """Typical 4% rule should have low-to-moderate ruin probability."""
        # 4% withdrawal rate has been historically safe over finite horizons
        # but infinite horizon ruin is higher
        initial_capital = 1000000
        withdrawal = initial_capital * 0.04  # $40,000 (4% rule)
        prob = self.calculator.calculate_ruin_probability_infinite(
            initial_capital, withdrawal
        )
        # With 7% return and 4% variance, 4% withdrawal should be safer than 50%
        self.assertLess(prob, 0.5)

    def test_statistics_dict_keys(self):
        """Statistics dict should have all required keys."""
        stats = self.calculator.compute_statistics(1000000, 40000)
        required_keys = {
            'ruin_probability',
            'survival_probability',
            'initial_capital',
            'annual_withdrawal',
            'withdrawal_rate',
            'mean_return',
            'variance',
            'sigma',
            'gamma_a',
            'gamma_x',
            'horizon',
        }
        self.assertTrue(required_keys.issubset(set(stats.keys())))

    def test_statistics_consistency(self):
        """Survival and ruin probabilities should sum to 1."""
        stats = self.calculator.compute_statistics(1000000, 40000)
        total = stats['ruin_probability'] + stats['survival_probability']
        self.assertAlmostEqual(total, 1.0, places=10)

    def test_statistics_horizon_is_infinite(self):
        """Horizon should always be 'infinite'."""
        stats = self.calculator.compute_statistics(1000000, 40000)
        self.assertEqual(stats['horizon'], 'infinite')


class TestGBMInfiniteAnalyticStrategy(unittest.TestCase):
    """Tests for GBMInfiniteAnalyticStrategy."""

    def setUp(self):
        """Set up test fixtures."""
        self.strategy = GBMInfiniteAnalyticStrategy()

    def test_default_initialization(self):
        """Strategy should initialize with default models."""
        self.assertIsNotNone(self.strategy.market_environment)
        self.assertIsNotNone(self.strategy.consumption_model)
        self.assertIsNotNone(self.strategy.calculator)

    def test_custom_initialization(self):
        """Strategy should initialize with custom models."""
        market_env = MarketEnvironmentFactory.constant(0.08, 0.05)
        consumption = ConsumptionModelFactory.constant(50000)
        strategy = GBMInfiniteAnalyticStrategy(market_env, consumption)
        self.assertEqual(strategy.market_environment.get_mean(0), 0.08)
        self.assertEqual(strategy.market_environment.get_variance(0), 0.05)

    def test_calculate_ruin_probability_bounds(self):
        """Ruin probability should be between 0 and 1."""
        for initial_capital in [100000, 500000, 1000000]:
            for withdrawal in [0, 5000, 10000, 40000]:
                prob = self.strategy.calculate_ruin_probability(
                    initial_capital, withdrawal, years=30
                )
                self.assertGreaterEqual(prob, 0.0)
                self.assertLessEqual(prob, 1.0)

    def test_years_parameter_ignored(self):
        """Years parameter should be ignored (always infinite horizon)."""
        initial_capital = 1000000
        withdrawal = 40000
        
        prob_30_years = self.strategy.calculate_ruin_probability(
            initial_capital, withdrawal, years=30
        )
        prob_50_years = self.strategy.calculate_ruin_probability(
            initial_capital, withdrawal, years=50
        )
        prob_infinite = self.strategy.calculate_ruin_probability(
            initial_capital, withdrawal, years=None
        )
        
        # All should be the same (infinite horizon)
        self.assertAlmostEqual(prob_30_years, prob_50_years, places=10)
        self.assertAlmostEqual(prob_30_years, prob_infinite, places=10)

    def test_simulate_returns_dict(self):
        """Simulate should return a dictionary with required keys."""
        result = self.strategy.simulate(1000000, 40000)
        required_keys = {
            'ruin_probability',
            'survival_probability',
            'withdrawal_rate',
            'method',
            'horizon',
            'statistics',
        }
        self.assertTrue(required_keys.issubset(set(result.keys())))

    def test_simulate_method_is_analytical(self):
        """Method should be labeled as analytical."""
        result = self.strategy.simulate(1000000, 40000)
        self.assertEqual(result['method'], 'gbm_infinite_analytical')

    def test_simulate_horizon_is_infinite(self):
        """Horizon should always be 'infinite'."""
        result = self.strategy.simulate(1000000, 40000)
        self.assertEqual(result['horizon'], 'infinite')

    def test_simulate_respects_num_simulations_parameter(self):
        """Num_simulations parameter should be accepted but ignored."""
        # Should not raise error even though it's ignored
        result = self.strategy.simulate(1000000, 40000, num_simulations=100000)
        self.assertIsNotNone(result)
        self.assertIn('ruin_probability', result)

    def test_zero_withdrawal(self):
        """Zero withdrawal should have zero ruin probability."""
        result = self.strategy.simulate(1000000, 0)
        self.assertAlmostEqual(result['ruin_probability'], 0.0, places=5)
        self.assertAlmostEqual(result['survival_probability'], 1.0, places=5)

    def test_small_withdrawal_low_ruin(self):
        """Small withdrawal (2%) should have lower ruin than high withdrawals."""
        initial_capital = 1000000
        result = self.strategy.simulate(initial_capital, initial_capital * 0.02)
        # 2% withdrawal should have lower ruin than 20%
        self.assertLess(result['ruin_probability'], 0.2)

    def test_large_withdrawal_high_ruin(self):
        """Large withdrawal (20%) should have high ruin probability."""
        initial_capital = 1000000
        result = self.strategy.simulate(initial_capital, initial_capital * 0.20)
        self.assertGreater(result['ruin_probability'], 0.5)

    def test_survival_probability_complement(self):
        """Survival and ruin probabilities should sum to 1."""
        result = self.strategy.simulate(1000000, 40000)
        total = result['ruin_probability'] + result['survival_probability']
        self.assertAlmostEqual(total, 1.0, places=10)

    def test_withdrawal_rate_calculation(self):
        """Withdrawal rate should be annual_withdrawal / initial_capital."""
        initial_capital = 1000000
        annual_withdrawal = 40000
        result = self.strategy.simulate(initial_capital, annual_withdrawal)
        expected_rate = annual_withdrawal / initial_capital
        self.assertAlmostEqual(result['withdrawal_rate'], expected_rate, places=10)


class TestGBMInfiniteAnalyticEdgeCases(unittest.TestCase):
    """Tests for edge cases in GBM Infinite Analytical Strategy."""

    def setUp(self):
        """Set up test fixtures."""
        self.strategy = GBMInfiniteAnalyticStrategy()

    def test_zero_initial_capital(self):
        """Should handle zero initial capital gracefully."""
        result = self.strategy.calculate_ruin_probability(0, 10000)
        # With zero capital, can't withdraw, so ruin is likely
        self.assertIsInstance(result, (int, float))
        self.assertGreaterEqual(result, 0.0)
        self.assertLessEqual(result, 1.0)

    def test_negative_withdrawal(self):
        """Negative withdrawal (contribution) should have zero ruin."""
        result = self.strategy.calculate_ruin_probability(1000000, -10000)
        self.assertAlmostEqual(result, 0.0, places=5)

    def test_very_small_capital(self):
        """Should handle very small capital amounts."""
        result = self.strategy.simulate(100, 1)
        self.assertIn('ruin_probability', result)
        self.assertGreaterEqual(result['ruin_probability'], 0.0)
        self.assertLessEqual(result['ruin_probability'], 1.0)

    def test_very_large_capital(self):
        """Should handle very large capital amounts."""
        result = self.strategy.simulate(1e10, 1e8)
        self.assertIn('ruin_probability', result)
        self.assertGreaterEqual(result['ruin_probability'], 0.0)
        self.assertLessEqual(result['ruin_probability'], 1.0)

    def test_different_market_environments(self):
        """Should work with different market parameters."""
        test_cases = [
            (0.05, 0.02),  # Conservative
            (0.07, 0.04),  # Moderate
            (0.10, 0.09),  # Aggressive
        ]
        for mu, variance in test_cases:
            strategy = GBMInfiniteAnalyticStrategy(
                MarketEnvironmentFactory.constant(mu, variance),
                ConsumptionModelFactory.constant(40000)
            )
            result = strategy.simulate(1000000, 40000)
            self.assertIn('ruin_probability', result)
            self.assertGreaterEqual(result['ruin_probability'], 0.0)
            self.assertLessEqual(result['ruin_probability'], 1.0)


if __name__ == '__main__':
    unittest.main()
