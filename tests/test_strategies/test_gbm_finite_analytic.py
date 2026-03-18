"""Tests for GBM Finite Analytical Strategy.

Tests that would trigger the full spectral decomposition (nested mpmath
Whittaker-W quadrature in the branch cut) are mocked so the entire suite
runs in < 1 second.  Numerical accuracy of the spectral decomposition
itself is validated in the GBM-Finite-MC-Comparison notebook.
"""

import unittest
from unittest.mock import patch
import numpy as np
from fireworks.strategies.gbm_finite_analytic import (
    GBMFiniteAnalyticStrategy,
    GBMFiniteAnalyticCalculator,
    ConstantMarketEnvironment,
    ConstantConsumptionModel,
    MarketEnvironmentFactory,
    ConsumptionModelFactory,
)

# Plausible spectral result: (S_stat, S_bounded, S_branch, S_tot)
# S_tot = 0.85 → ruin ≈ 15 %
_MOCK_SPECTRAL = (0.70, 0.10, 0.05, 0.85)


def _mock_spectral(self, mu, sigma_sq, w, T):
    """Drop-in replacement for _exact_spectral_decomposition."""
    return _MOCK_SPECTRAL


class TestGBMFiniteAnalyticCalculator(unittest.TestCase):
    """Tests for GBMFiniteAnalyticCalculator."""

    def setUp(self):
        self.mu = 0.07
        self.variance = 0.04
        self.market_env = ConstantMarketEnvironment(self.mu, self.variance)
        self.consumption = ConstantConsumptionModel(10000)
        self.calculator = GBMFiniteAnalyticCalculator(self.market_env, self.consumption)

    # ---- input validation (no computation) ----

    def test_zero_capital_raises(self):
        with self.assertRaises(ValueError):
            self.calculator.calculate_ruin_probability_finite(0, 40000, 30)

    def test_negative_capital_raises(self):
        with self.assertRaises(ValueError):
            self.calculator.calculate_ruin_probability_finite(-1, 40000, 30)

    def test_negative_withdrawal_raises(self):
        with self.assertRaises(ValueError):
            self.calculator.calculate_ruin_probability_finite(1e6, -1, 30)

    def test_zero_years_raises(self):
        with self.assertRaises(ValueError):
            self.calculator.calculate_ruin_probability_finite(1e6, 40000, 0)

    def test_negative_years_raises(self):
        with self.assertRaises(ValueError):
            self.calculator.calculate_ruin_probability_finite(1e6, 40000, -5)

    # ---- early-return paths (no spectral decomposition) ----

    def test_zero_withdrawal_zero_ruin(self):
        prob = self.calculator.calculate_ruin_probability_finite(1e6, 0, 30)
        self.assertEqual(prob, 0.0)

    def test_withdrawal_exceeds_capital(self):
        prob = self.calculator.calculate_ruin_probability_finite(100000, 100000, 30)
        self.assertEqual(prob, 1.0)

    # ---- method existence ----

    def test_spectral_component_methods_exist(self):
        for name in ('_exact_spectral_decomposition', '_compute_ground_state',
                     '_compute_bounded_states', '_compute_branch_cut'):
            self.assertTrue(hasattr(self.calculator, name), f"missing {name}")

    # ---- mocked spectral tests (instant) ----

    @patch.object(GBMFiniteAnalyticCalculator, '_exact_spectral_decomposition', _mock_spectral)
    def test_ruin_probability_bounds(self):
        prob = self.calculator.calculate_ruin_probability_finite(1e6, 40000, 30)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    @patch.object(GBMFiniteAnalyticCalculator, '_exact_spectral_decomposition', _mock_spectral)
    def test_typical_fire_scenario(self):
        prob = self.calculator.calculate_ruin_probability_finite(1e6, 40000, 30)
        self.assertAlmostEqual(prob, 0.15, places=10)

    @patch.object(GBMFiniteAnalyticCalculator, '_exact_spectral_decomposition', _mock_spectral)
    def test_statistics_dict_keys(self):
        stats = self.calculator.compute_statistics(1e6, 40000, 30)
        required = {'ruin_probability', 'survival_probability',
                     'withdrawal_rate', 'horizon', 'spectral_components'}
        self.assertTrue(required.issubset(stats.keys()),
                        f"Missing: {required - stats.keys()}")

    @patch.object(GBMFiniteAnalyticCalculator, '_exact_spectral_decomposition', _mock_spectral)
    def test_statistics_consistency(self):
        stats = self.calculator.compute_statistics(1e6, 40000, 30)
        self.assertAlmostEqual(
            stats['ruin_probability'] + stats['survival_probability'], 1.0, places=10)

    @patch.object(GBMFiniteAnalyticCalculator, '_exact_spectral_decomposition', _mock_spectral)
    def test_statistics_horizon_recorded(self):
        for years in [10, 30, 50]:
            stats = self.calculator.compute_statistics(1e6, 40000, years)
            self.assertEqual(stats['horizon'], years)

    @patch.object(GBMFiniteAnalyticCalculator, '_exact_spectral_decomposition', _mock_spectral)
    def test_spectral_components_in_statistics(self):
        stats = self.calculator.compute_statistics(1e6, 40000, 30)
        spectral = stats['spectral_components']
        for key in ('ground_state', 'bounded_states', 'branch_cut'):
            self.assertIn(key, spectral)

    @patch.object(GBMFiniteAnalyticCalculator, '_exact_spectral_decomposition', _mock_spectral)
    def test_multiple_horizons_accepted(self):
        for years in [0.01, 1, 10, 30, 50]:
            prob = self.calculator.calculate_ruin_probability_finite(1e6, 40000, years)
            self.assertGreaterEqual(prob, 0.0)
            self.assertLessEqual(prob, 1.0)


class TestGBMFiniteAnalyticStrategy(unittest.TestCase):
    """Tests for GBMFiniteAnalyticStrategy."""

    def setUp(self):
        self.strategy = GBMFiniteAnalyticStrategy()

    # ---- initialisation ----

    def test_default_initialization(self):
        self.assertIsNotNone(self.strategy.market_environment)
        self.assertIsNotNone(self.strategy.consumption_model)
        self.assertIsNotNone(self.strategy.calculator)

    def test_custom_initialization(self):
        market_env = MarketEnvironmentFactory.constant(0.08, 0.05)
        consumption = ConsumptionModelFactory.constant(50000)
        strategy = GBMFiniteAnalyticStrategy(market_env, consumption)
        self.assertEqual(strategy.market_environment.get_mean(0), 0.08)
        self.assertEqual(strategy.market_environment.get_variance(0), 0.05)

    # ---- years validation ----

    def test_calculate_requires_positive_years(self):
        for bad_years in [None, 0, -5]:
            with self.assertRaises(ValueError):
                self.strategy.calculate_ruin_probability(1e6, 40000, years=bad_years)

    def test_simulate_requires_positive_years(self):
        for bad_years in [None, 0]:
            with self.assertRaises(ValueError):
                self.strategy.simulate(1e6, 40000, years=bad_years)

    # ---- early-return paths ----

    def test_zero_withdrawal_calculate(self):
        prob = self.strategy.calculate_ruin_probability(1e6, 0, years=30)
        self.assertEqual(prob, 0.0)

    def test_withdrawal_exceeds_capital(self):
        prob = self.strategy.calculate_ruin_probability(100000, 100000, years=30)
        self.assertEqual(prob, 1.0)

    @patch.object(GBMFiniteAnalyticCalculator, '_exact_spectral_decomposition', _mock_spectral)
    def test_zero_withdrawal_simulate(self):
        result = self.strategy.simulate(1e6, 0, years=30)
        self.assertAlmostEqual(result['ruin_probability'], 0.0, places=5)

    # ---- mocked spectral tests ----

    @patch.object(GBMFiniteAnalyticCalculator, '_exact_spectral_decomposition', _mock_spectral)
    def test_calculate_with_valid_years(self):
        prob = self.strategy.calculate_ruin_probability(1e6, 40000, years=30)
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)

    @patch.object(GBMFiniteAnalyticCalculator, '_exact_spectral_decomposition', _mock_spectral)
    def test_simulate_returns_required_keys(self):
        result = self.strategy.simulate(1e6, 40000, years=30)
        required = {'ruin_probability', 'survival_probability',
                     'withdrawal_rate', 'horizon', 'spectral_components', 'statistics'}
        self.assertTrue(required.issubset(result.keys()),
                        f"Missing: {required - result.keys()}")

    @patch.object(GBMFiniteAnalyticCalculator, '_exact_spectral_decomposition', _mock_spectral)
    def test_simulate_horizon_recorded(self):
        result = self.strategy.simulate(1e6, 40000, years=30)
        self.assertEqual(result['horizon'], 30)

    @patch.object(GBMFiniteAnalyticCalculator, '_exact_spectral_decomposition', _mock_spectral)
    def test_simulate_accepts_num_simulations(self):
        result = self.strategy.simulate(1e6, 40000, years=30, num_simulations=100000)
        self.assertIn('ruin_probability', result)

    @patch.object(GBMFiniteAnalyticCalculator, '_exact_spectral_decomposition', _mock_spectral)
    def test_spectral_components_in_simulate(self):
        spectral = self.strategy.simulate(1e6, 40000, years=30)['spectral_components']
        for key in ('ground_state', 'bounded_states', 'branch_cut'):
            self.assertIn(key, spectral)

    # ---- interface checks ----

    def test_methods_exist(self):
        self.assertTrue(callable(self.strategy.calculate_ruin_probability))
        self.assertTrue(callable(self.strategy.simulate))


if __name__ == '__main__':
    unittest.main()
