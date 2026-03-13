"""Unit tests for Heston Stiff Perturbative Strategy."""

import pytest
import numpy as np

from fireworks.strategies.heston_stiff_perturbative import (
    HestonStiffPerturbativeStrategy,
    HestonMarketEnvironmentFactory,
    HestonStiffPerturbativeCalculator,
)


class TestHestonMarketEnvironment:
    """Test Heston market environment models."""

    def test_constant_heston_env_creation(self):
        """Test creating a constant Heston environment."""
        env = HestonMarketEnvironmentFactory.constant(
            mu=0.07,
            initial_variance=0.0324,  # σ=18%
            kappa=1.0,  # Mean reversion speed
            long_var=0.0324,  # Long-term var = initial var
            vol_of_vol=0.3,  # Vol of vol
            correlation=-0.5,  # Leverage: -50% correlation
        )
        assert env.get_mu() == 0.07
        assert env.get_initial_variance() == 0.0324
        assert env.get_kappa() == 1.0
        assert env.get_long_var() == 0.0324
        assert env.get_vol_of_vol() == 0.3
        assert env.get_correlation() == -0.5

    def test_heston_env_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        # Invalid mu
        with pytest.raises(ValueError, match="mu must be"):
            HestonMarketEnvironmentFactory.constant(
                mu=-2.0,
                initial_variance=0.0324,
                kappa=1.0,
                long_var=0.0324,
                vol_of_vol=0.3,
            )

        # Invalid kappa
        with pytest.raises(ValueError, match="kappa must be"):
            HestonMarketEnvironmentFactory.constant(
                mu=0.07,
                initial_variance=0.0324,
                kappa=-1.0,  # Must be positive
                long_var=0.0324,
                vol_of_vol=0.3,
            )

        # Invalid correlation
        with pytest.raises(ValueError, match="correlation must be"):
            HestonMarketEnvironmentFactory.constant(
                mu=0.07,
                initial_variance=0.0324,
                kappa=1.0,
                long_var=0.0324,
                vol_of_vol=0.3,
                correlation=2.0,  # Out of [-1, 1]
            )


class TestHestonCalculator:
    """Test Heston perturbative calculator."""

    def test_calculator_creation(self):
        """Test creating calculator from market environment."""
        env = HestonMarketEnvironmentFactory.constant(
            mu=0.07,
            initial_variance=0.0324,
            kappa=2.0,
            long_var=0.0324,
            vol_of_vol=0.3,
        )
        calc = HestonStiffPerturbativeCalculator(env)
        assert calc.mu == 0.07
        assert calc.v0 == 0.0324
        assert calc.kappa == 2.0
        assert calc.epsilon == 1.0 / 2.0

    def test_ruin_probability_basic(self):
        """Test basic ruin probability computation."""
        env = HestonMarketEnvironmentFactory.constant(
            mu=0.07,
            initial_variance=0.0324,  # σ = 18%
            kappa=2.0,  # Fast mean reversion
            long_var=0.0324,
            vol_of_vol=0.3,
        )
        calc = HestonStiffPerturbativeCalculator(env, max_order=1)

        result = calc.compute_ruin_probability(
            initial_wealth=1_000_000,
            annual_withdrawal=40_000,  # 4% rule
            years=30,
        )

        assert isinstance(result, dict)
        assert "ruin_probability" in result
        assert "order_0" in result
        assert "order_1" in result
        assert "epsilon" in result

        # Ruin probability should be in [0, 1]
        assert 0.0 <= result["ruin_probability"] <= 1.0
        assert 0.0 <= result["order_0"] <= 1.0

    def test_ruin_probability_safe_withdrawal(self):
        """Test ruin probability at safe withdrawal rates."""
        env = HestonMarketEnvironmentFactory.constant(
            mu=0.07,
            initial_variance=0.0324,
            kappa=1.0,
            long_var=0.0324,
            vol_of_vol=0.3,
        )
        calc = HestonStiffPerturbativeCalculator(env)

        # Very small withdrawal rate - should have low ruin probability
        result = calc.compute_ruin_probability(
            initial_wealth=1_000_000,
            annual_withdrawal=10_000,  # 1% rule
            years=30,
        )
        assert result["ruin_probability"] < 0.05  # Expect <5% ruin

    def test_ruin_probability_risky_withdrawal(self):
        """Test ruin probability at risky withdrawal rates."""
        env = HestonMarketEnvironmentFactory.constant(
            mu=0.07,
            initial_variance=0.0324,
            kappa=1.0,
            long_var=0.0324,
            vol_of_vol=0.3,
        )
        calc = HestonStiffPerturbativeCalculator(env)

        # High withdrawal rate - should have elevated ruin probability
        result = calc.compute_ruin_probability(
            initial_wealth=1_000_000,
            annual_withdrawal=100_000,  # 10% rule
            years=30,
        )
        assert result["ruin_probability"] > 0.1  # Expect >10% ruin

    def test_ruin_probability_invalid_inputs(self):
        """Test that invalid inputs raise errors."""
        env = HestonMarketEnvironmentFactory.constant(
            mu=0.07,
            initial_variance=0.0324,
            kappa=1.0,
            long_var=0.0324,
            vol_of_vol=0.3,
        )
        calc = HestonStiffPerturbativeCalculator(env)

        # Invalid initial wealth
        with pytest.raises(ValueError, match="initial_wealth"):
            calc.compute_ruin_probability(
                initial_wealth=-1_000_000,
                annual_withdrawal=40_000,
                years=30,
            )

        # Invalid years
        with pytest.raises(ValueError, match="years"):
            calc.compute_ruin_probability(
                initial_wealth=1_000_000,
                annual_withdrawal=40_000,
                years=-5,
            )


class TestHestonStrategy:
    """Test Heston Stiff Perturbative Strategy."""

    def test_strategy_creation(self):
        """Test creating strategy."""
        env = HestonMarketEnvironmentFactory.constant(
            mu=0.07,
            initial_variance=0.0324,
            kappa=2.0,
            long_var=0.0324,
            vol_of_vol=0.3,
            correlation=-0.5,
        )
        strategy = HestonStiffPerturbativeStrategy(env)
        assert strategy.market_environment == env
        assert strategy.max_perturbative_order == 1

    def test_strategy_simulate(self):
        """Test strategy.simulate() interface."""
        env = HestonMarketEnvironmentFactory.constant(
            mu=0.07,
            initial_variance=0.0324,
            kappa=1.5,
            long_var=0.0324,
            vol_of_vol=0.25,
        )
        strategy = HestonStiffPerturbativeStrategy(env, max_perturbative_order=1)

        result = strategy.simulate(
            initial_capital=1_000_000,
            annual_withdrawal=40_000,
            years=30,
        )

        assert "ruin_probability" in result
        assert "survival_probability" in result
        assert result["ruin_probability"] + result["survival_probability"] == pytest.approx(1.0)

    def test_strategy_name(self):
        """Test strategy name generation."""
        env = HestonMarketEnvironmentFactory.constant(
            mu=0.07,
            initial_variance=0.0324,
            kappa=1.0,
            long_var=0.0324,
            vol_of_vol=0.3,
        )
        strategy = HestonStiffPerturbativeStrategy(env)
        name = strategy.get_strategy_name()
        assert "Heston" in name
        assert "κ=1.00" in name


class TestPerturbativeExpansion:
    """Test perturbative expansion terms."""

    def test_order_0_term_exists(self):
        """Test that O(1) term is computed."""
        env = HestonMarketEnvironmentFactory.constant(
            mu=0.07,
            initial_variance=0.0324,
            kappa=2.0,
            long_var=0.0324,
            vol_of_vol=0.3,
        )
        calc = HestonStiffPerturbativeCalculator(env, max_order=0)
        result = calc.compute_ruin_probability(
            initial_wealth=1_000_000,
            annual_withdrawal=40_000,
            years=30,
        )
        assert "order_0" in result
        assert 0.0 <= result["order_0"] <= 1.0

    def test_order_1_term_exists(self):
        """Test that O(ε) term is computed when requested."""
        env = HestonMarketEnvironmentFactory.constant(
            mu=0.07,
            initial_variance=0.0324,
            kappa=2.0,
            long_var=0.0324,
            vol_of_vol=0.3,
        )
        calc = HestonStiffPerturbativeCalculator(env, max_order=1)
        result = calc.compute_ruin_probability(
            initial_wealth=1_000_000,
            annual_withdrawal=40_000,
            years=30,
        )
        assert "order_1" in result

    def test_epsilon_scales_correctly(self):
        """Test that perturbation parameter ε = 1/κ is computed correctly."""
        kappa_tests = [1.0, 2.0, 5.0, 10.0]
        for kappa in kappa_tests:
            env = HestonMarketEnvironmentFactory.constant(
                mu=0.07,
                initial_variance=0.0324,
                kappa=kappa,
                long_var=0.0324,
                vol_of_vol=0.3,
            )
            calc = HestonStiffPerturbativeCalculator(env)
            assert calc.epsilon == pytest.approx(1.0 / kappa)
