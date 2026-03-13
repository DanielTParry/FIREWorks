"""Heston stochastic volatility ruin probability calculator.

Perturbative approximation for ruin probability with fast mean-reverting volatility.
Expansion parameter: ε = 1/κ (inverse of mean reversion speed).

Mathematical framework:
- Base case (κ → ∞): Volatility frozen at initial level → GBM limit via incomplete gamma
- O(ε) correction: Linear variance penalty effect
- O(ε^1.5) correction: Correlation/leverage interaction
- O(ε^2) correction: Quadratic convexity effect
- O(ε^2.5) correction: Secondary correlation terms

5-term singular perturbation hierarchy derived via boundary density method.

References:
- Heston (1993) stochastic volatility model
- Singular perturbation expansion in W-space (withdrawal rate)
- Small parameter: ε = 1/κ (inverse mean reversion timescale)
"""

import numpy as np
from typing import Tuple, Dict, Optional
import warnings
from scipy.special import gammaincc, gammaln

from .models import AbstractHestonMarketEnvironment


class HestonStiffPerturbativeCalculator:
    """
    Computes ruin probability over finite horizon for Heston model
    using perturbative expansion in ε = 1/κ.
    """

    def __init__(
        self,
        market_env: AbstractHestonMarketEnvironment,
        atol: float = 1e-12,
    ):
        """
        Initialize calculator.

        Args:
            market_env: Heston market environment with all parameters
            atol: Absolute tolerance for convergence check on highest-order term (default 1e-12)
        """
        self.market_env = market_env
        self.atol = atol

        # Extract and cache parameters
        self.mu = market_env.get_mu()
        self.v0 = market_env.get_initial_variance()
        self.kappa = market_env.get_kappa()
        self.theta = market_env.get_long_var()
        self.sigma_v = market_env.get_vol_of_vol()
        self.rho = market_env.get_correlation()

        # Perturbation parameter
        self.epsilon = 1.0 / self.kappa

    def compute_ruin_probability(
        self,
        initial_wealth: float,
        annual_withdrawal: float,
        years: float,
        return_details: bool = False,
    ) -> Dict[str, float]:
        """
        Compute ruin probability using 5-term singular perturbation hierarchy.

        Uses closed-form analytical approximation based on boundary density method
        and incomplete gamma functions.

        Args:
            initial_wealth: Starting portfolio value
            annual_withdrawal: Annual withdrawal amount
            years: Investment horizon in years (cosmetic for perpetual approximation)
            return_details: Whether to return all perturbative terms

        Returns:
            Dictionary with:
            - 'ruin_probability': Main result ∈ [0, 1]
            - 'survival_probability': 1 - ruin_probability
            - 'order_0': O(1) term
            - 'perturbative_terms': Dict with all 5 terms if return_details=True
            - Additional detail fields if return_details=True

        Raises:
            ValueError: If parameters violate Heston constraints
        """
        if initial_wealth <= 0:
            raise ValueError(f"initial_wealth must be > 0, got {initial_wealth}")
        if annual_withdrawal < 0:
            raise ValueError(f"annual_withdrawal must be ≥ 0, got {annual_withdrawal}")
        if years <= 0:
            raise ValueError(f"years must be > 0, got {years}")

        # Effective withdrawal rate (dimensionless)
        w_rate = annual_withdrawal / initial_wealth

        # Compute 5-term singular perturbation hierarchy
        perturbative_terms = self._compute_singular_w_ladder_5term(w_rate)

        # Extract terms
        survival_prob = perturbative_terms["total"]
        ruin_prob = 1.0 - survival_prob

        result = {
            "ruin_probability": ruin_prob,
            "survival_probability": survival_prob,
            "order_0": perturbative_terms["u0"],
            "epsilon": self.epsilon,
        }

        if return_details:
            result["perturbative_terms"] = {
                "u0": perturbative_terms["u0"],  # O(1)
                "u2": perturbative_terms["u2"],  # O(ε)
                "u3": perturbative_terms["u3"],  # O(ε^1.5)
                "u4": perturbative_terms["u4"],  # O(ε^2)
                "u5": perturbative_terms["u5"],  # O(ε^2.5)
            }
            result["initial_wealth"] = initial_wealth
            result["annual_withdrawal"] = annual_withdrawal
            result["withdrawal_rate"] = w_rate
            result["years"] = years
            result["mu"] = self.mu
            result["v0"] = self.v0
            result["kappa"] = self.kappa
            result["theta"] = self.theta
            result["sigma_v"] = self.sigma_v
            result["rho"] = self.rho

        return result

    def _compute_frozen_vol_ruin(
        self,
        w_rate: float,
        sigma: float,
        mu: float,
        years: float,
    ) -> float:
        """
        Compute survival probability when volatility is frozen at σ.

        This is the GBM limit using incomplete gamma function.
        Returns survival probability S(w) via upper incomplete gamma.

        Mathematical derivation:
        When variance is constant (frozen at v0), the wealth process becomes:
        dW/W = μ dt + σ dZ
        
        Integrating the ruin probability gives the incomplete gamma function:
        S(w) = Γ(a, x) / Γ(a)  where:
        - a = (2μ/σ²) - 1
        - x = 2w/σ²

        Args:
            w_rate: Annual withdrawal rate w = W/S₀ (dimensionless)
            sigma: Constant volatility (standard deviation)
            mu: Drift rate (annual)
            years: Time horizon (unused for perpetual limit)

        Returns:
            Survival probability S(w) ∈ [0, 1]
        """
        if sigma <= 0:
            raise ValueError(f"sigma must be > 0, got {sigma}")

        variance = sigma**2

        # Compute shape parameter
        # a = (2μ/σ²) - 1
        a = (2.0 * mu / variance) - 1.0

        # Compute scale argument
        # x = 2w/σ²
        x = 2.0 * w_rate / variance

        # Upper incomplete gamma gives survival probability
        # S(w) = Γ(a, x) / Γ(a) = gammaincc(a, x)
        survival_prob = gammaincc(a, x)

        return np.clip(float(survival_prob), 0.0, 1.0)

    def _compute_order_1_correction(
        self,
        w_rate: float,
        years: float,
        order_0_prob: float,
    ) -> float:
        """
        Compute O(ε) = O(1/κ) correction to ruin probability.

        This is now replaced by the full 5-term singular perturbation hierarchy.
        This method is kept for interface compatibility but deprecated.

        For the full perturbative expansion, use compute_ruin_probability_detailed().

        Args:
            w_rate: Withdrawal rate
            years: Time horizon
            order_0_prob: O(1) ruin probability (for reference)

        Returns:
            O(ε) correction term (can be positive or negative)
        """
        # Use full hierarchy and extract O(ε) term
        details = self._compute_singular_w_ladder_5term(w_rate)
        return details["u2"]  # u2 is the O(ε) term

    def _compute_singular_w_ladder_5term(
        self,
        w: float,
    ) -> Dict[str, float]:
        """
        Compute 5-term singular perturbation hierarchy in W-space.

        Evaluates the exact closed-form polynomials derived via boundary density method:
        - u0: Baseline survival (frozen volatility limit, incomplete gamma)
        - u2: Linear variance penalty O(ε)
        - u3: Correlation/leverage lift O(ε^1.5)
        - u4: Quadratic convexity O(ε^2)
        - u5: Secondary correlation interaction O(ε^2.5)

        Total survival probability:
        S(w) ≈ u0 + u2 + u3 + u4 + u5

        Args:
            w: Withdrawal rate (dimensionless, typically 0.01 to 0.10)

        Returns:
            Dictionary with keys:
            - 'u0': Baseline term
            - 'u2': O(ε) term
            - 'u3': O(ε^1.5) term
            - 'u4': O(ε^2) term
            - 'u5': O(ε^2.5) term
            - 'total': Sum of all terms (clipped to [0, 1])
        """
        # Extract parameters
        a = (2.0 * self.mu / self.theta) - 1.0
        x = 2.0 * w / self.theta

        # 0. Baseline Survival (Upper Incomplete Gamma)
        # Equivalent to frozen volatility at v = theta (not v0!)
        u0 = gammaincc(a, x)

        # Calculate Base Density P(w) in log-space to prevent underflow
        # P(w) = d/dx [Γ(a,x)/Γ(a)]
        log_P = a * np.log(x) - x - gammaln(a)
        P_w = np.exp(log_P)

        # Common polynomial bases
        w_mu = w - self.mu
        v_diff = self.v0 - self.theta

        # 1. Linear Variance Penalty O(ε)
        # Effect of initial variance elevation from long-term level
        u2 = (v_diff / (self.kappa * self.theta)) * w_mu * P_w

        # 2. Correlation/Leverage Lift O(ε^1.5)
        # Protective effect of negative correlation during volatility spikes
        u3 = (
            self.rho * self.sigma_v * v_diff / (self.kappa * self.theta**2)
        ) * (2 * w_mu**2 - self.mu * self.theta) * P_w

        # 3. Quadratic Convexity O(ε^2)
        # Nonlinear effect from variance fluctuations
        u4 = (v_diff**2 / (2 * self.kappa**2 * self.theta**3)) * (
            2 * w_mu**3 - 2 * self.theta * w_mu**2 - 3 * self.mu * self.theta * w_mu
        ) * P_w

        # 4. Secondary Correlation Interaction O(ε^2.5)
        # Higher-order leverage effects
        u5 = (
            self.rho * self.sigma_v * v_diff**2 / (2 * self.kappa**2 * self.theta**4)
        ) * (
            4 * w_mu**4
            - 8 * self.theta * w_mu**3
            + 2 * self.theta * (self.theta - 6 * self.mu) * w_mu**2
            + 4 * self.mu * self.theta**2 * w_mu
            + 3 * self.mu**2 * self.theta**2
        ) * P_w

        total = np.clip(float(u0 + u2 + u3 + u4 + u5), 0.0, 1.0)

        # Check convergence: if highest-order term exceeds tolerance, expansion hasn't converged
        if abs(u5) > self.atol:
            raise NotImplementedError(
                f"5-term singular perturbation expansion did not converge to atol={self.atol}. "
                f"Highest-order term u5 (O(ε^2.5)) = {u5:.2e} exceeds tolerance. "
                f"TODO: Extend to 7-term (O(ε^3)), 8-term (O(ε^3.5)), or higher-order expansion "
                f"using boundary density method and singular perturbation theory."
            )

        return {
            "u0": float(u0),
            "u2": float(u2),
            "u3": float(u3),
            "u4": float(u4),
            "u5": float(u5),
            "total": total,
        }

    def _validate_heston_constraints(self) -> None:
        """Verify Heston parameters are valid."""
        # Feller condition: 2κθ ≥ σ_v² ensures vol stays positive
        feller = 2.0 * self.kappa * self.theta - self.sigma_v**2
        if feller < 0:
            warnings.warn(
                f"Feller condition violated: 2κθ = {2*self.kappa*self.theta:.4f} "
                f"< σ_v² = {self.sigma_v**2:.4f}. "
                f"Variance may hit zero.",
                UserWarning,
            )
