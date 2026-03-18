"""Analytical GBM finite-horizon ruin probability calculator.

Uses exact spectral decomposition method for the probability of ruin
over finite time horizons under GBM with constant parameters and withdrawal.

The solution decomposes into three components:
    1. Stationary ground state (incomplete Gamma function)
    2. Discrete bounded states (Monthus & Comtet pole residues)
    3. Continuous branch cut (scattering integral)

References:
    Monthus, C., & Comtet, A. (1994). On the flux distribution in a one dimensional 
    disordered system. Journal de Physique I, 4(5), 635-653.
    https://hal.science/jpa-00246938v1/file/ajp-jp1v4p635.pdf
"""

from typing import Dict, Any, Tuple, Callable
import numpy as np
from scipy.integrate import quad
from scipy.special import gammaincc, gamma, genlaguerre

# Disable gmpy2 to avoid Windows DLL loading issues
import os
os.environ['MPMATH_NOGMPY'] = '1'

import mpmath

from fireworks.strategies.mc_strategy.models import (
    AbstractMarketEnvironment,
    AbstractConsumptionModel,
)


class GBMFiniteAnalyticCalculator:
    """Analytical calculator for finite-horizon GBM ruin probability."""

    def __init__(self, market_environment: AbstractMarketEnvironment,
                 consumption_model: AbstractConsumptionModel, *,
                 eps: float = 1e-8,
                 mpmath_extra_dps: int = 16,
                 branch_cut_s_max: float = 15.0,
                 quad_limit: int = 100) -> None:
        """
        Initialize the calculator.

        Args:
            market_environment: Market environment defining μ and variance
            consumption_model: Consumption model defining withdrawal C
            eps: Absolute tolerance for all numerical integration.
            mpmath_extra_dps: Extra decimal places for mpmath arithmetic.
            branch_cut_s_max: Upper limit for the scattering momentum integral.
                The branch-cut integrand envelope behaves as
                    I(s) ~ s^{−ν+1} · exp(−τs²/2 + πs/2)
                and decays as a Gaussian beyond its peak at s* = π/(2τ).
                For typical equity parameters (σ² ≈ 0.04, T = 30 → τ ≈ 0.3),
                the integrand is negligible well before s = 15.  For low-
                volatility regimes (τ < 0.05, e.g. bonds) the peak shifts
                beyond 15 and this value must be increased accordingly.
            quad_limit: Maximum subintervals for scipy.integrate.quad.
        """
        self.market_environment = market_environment
        self.consumption_model = consumption_model
        self._eps = eps
        self._mpmath_dps = int(-np.log10(eps)) + mpmath_extra_dps
        self._branch_cut_s_max = branch_cut_s_max
        self._quad_limit = quad_limit

    def calculate_ruin_probability_finite(self, initial_capital: float,
                                          annual_withdrawal: float,
                                          years: float) -> float:
        """
        Calculate probability of ruin over finite time horizon.

        Uses exact spectral decomposition method combining:
        - Stationary ground state
        - Discrete bounded states  
        - Continuous branch cut

        Args:
            initial_capital: Starting portfolio value P_0 (must be > 0)
            annual_withdrawal: Annual withdrawal amount C (must be ≥ 0)
            years: Time horizon T (must be > 0)

        Returns:
            Probability of ruin (float between 0 and 1)

        Raises:
            ValueError: If initial_capital <= 0, years <= 0, or annual_withdrawal < 0
        """
        # Validate inputs
        if initial_capital <= 0:
            raise ValueError(f"initial_capital must be > 0, got {initial_capital}")
        if years <= 0:
            raise ValueError(f"years must be > 0, got {years}")
        if annual_withdrawal < 0:
            raise ValueError(f"annual_withdrawal must be ≥ 0, got {annual_withdrawal}")
        
        if annual_withdrawal <= 0:
            return 0.0
        if annual_withdrawal >= initial_capital:
            return 1.0
        
        mu = self.market_environment.get_mean(0)
        variance = self.market_environment.get_variance(0)
        
        if variance <= 0:
            return self._ruin_probability_deterministic(mu, annual_withdrawal, initial_capital, years)
        if mu < 0:
            raise ValueError(f"Negative drift (mu={mu}) not supported.") 

        w = annual_withdrawal / initial_capital
        _, _, _, s_tot = self._exact_spectral_decomposition(mu, variance, w, years)
        return float(np.clip(1.0 - s_tot, 0.0, 1.0))

    def _ruin_probability_deterministic(self, mu: float, annual_withdrawal: float,
                                        initial_capital: float, years: float) -> float:
        """
        Ruin probability for deterministic (zero-variance) portfolio.
        
        For μ > 0: Survives if SWR ≤ 1 / ((1 - exp(-μ*T)) / μ)
        For μ ≤ 0: Certain ruin with any withdrawal
        
        Args:
            mu: Constant growth rate
            annual_withdrawal: Annual withdrawal amount
            initial_capital: Starting portfolio value
            years: Time horizon
            
        Returns:
            Ruin probability (0 or 1)
        """
        if mu > 0:
            swr = annual_withdrawal / initial_capital
            portfolio_capacity = (1.0 - np.exp(-mu * years)) / mu
            max_sustainable_swr = 1.0 / portfolio_capacity
            return 0.0 if swr <= max_sustainable_swr else 1.0
        return 1.0

    def _exact_spectral_decomposition(self, mu: float, sigma_sq: float,
                                      w: float, T: float) -> Tuple[float, float, float, float]:
        """
        Decompose ruin probability into three spectral components.
        
        Implements Monthus & Bouchaud spectral decomposition.
        
        TODO: Current implementation assumes μ ≥ 0. For μ < 0, the paper (Monthus & Bouchaud,
        Eq. 3.6; https://hal.science/jpa-00246938v1/file/ajp-jp1v4p635.pdf) 
        prescribes an alternate spectral representation using modified Bessel
        functions K_iq. Extend this method to support negative drift:
            φ(p,N) = exp(-x_τ²)/(4π²) * (p/α)^(p/2) * ∫_{-∞}^{+∞} dq * exp(-αq²) *
                    q*sinh(πq) * |Γ(-μ/2 + iq/2)|² * K_iq(2√(p/α))
        
        Requires: scipy.special.kv (modified Bessel K function implementation)

        Args:
            mu: Mean return
            sigma_sq: Variance (σ²)
            w: Withdrawal rate (C)
            T: Time horizon

        Returns:
            Tuple of (S_stat, S_bounded, S_branch, S_tot)
        """
        theta = sigma_sq
        nu = (2.0 * mu / theta) - 1.0
        tau = (theta / 4.0) * T
        z_target = 2.0 * w / theta

        S_stat = self._compute_ground_state(nu, z_target)
        S_bounded = self._compute_bounded_states(nu, z_target, tau)
        S_branch = self._compute_branch_cut(nu, z_target, tau)
        S_tot = S_stat + S_bounded + S_branch

        return S_stat, S_bounded, S_branch, S_tot

    @staticmethod
    def _mpmath_whittaker_w(kappa: float, mu_param: complex, z: float) -> Any:
        """
        Construct the complex Whittaker W function natively.
        
        W_{kappa, mu}(z) = exp(-z/2) * z**(mu + 0.5) * U(mu - kappa + 0.5, 1 + 2*mu, z)
        
        Args:
            kappa: Whittaker kappa parameter
            mu_param: Whittaker mu parameter  
            z: Argument (real or complex)
            
        Returns:
            Complex Whittaker W function value (mpmath.mpc type)
        """
        a = mu_param - kappa + 0.5
        b = 1.0 + 2.0 * mu_param
        z_mp = mpmath.mpc(float(z))
        U_val = mpmath.hyperu(a, b, z_mp)
        return mpmath.exp(-z_mp / 2.0) * (z_mp**(mu_param + 0.5)) * U_val

    @staticmethod
    def _compute_ground_state(nu: float, z_target: float) -> float:
        """
        Compute stationary ground state using incomplete Gamma function.

        Args:
            nu: Shape parameter
            z_target: Target value 2w/θ

        Returns:
            Ground state contribution
        """
        return gammaincc(nu, z_target)

    def _compute_bounded_states(self, nu: float, z_target: float, tau: float) -> float:
        """
        Compute contribution from discrete bounded states.

        Uses Monthus & Comtet pole residue coefficients with
        generalized Laguerre polynomials.

        Args:
            nu: Shape parameter ν = 2μ/σ² − 1
            z_target: Target value 2w/σ²
            tau: Scaled time (σ²/4)·T

        Returns:
            Bounded states contribution to survival probability
        """
        N_states = max(0, int(np.floor((nu + 1.0) / 2.0 - 1e-7)))
        S_bounded = 0.0
        for n in range(1, N_states + 1):
            lam_n = 0.5 * n * (2.0 * nu - n)
            coef = ((-1.0) ** n * (nu + 1.0 - 2.0 * n)) / gamma(nu + 2.0 - n)
            L_n = genlaguerre(n, nu - 2.0 * n)
            res, _ = quad(lambda x: coef * (x**(nu-n-1.0)) * np.exp(-x) * L_n(x), z_target, np.inf, epsabs=self._eps)
            S_bounded += np.exp(-lam_n * tau) * res
        return S_bounded

    def _compute_branch_cut(self, nu: float, z_target: float, tau: float) -> float:
        """
        Compute contribution from continuous branch cut (Monthus & Comtet Eq 5.5).

        Uses exact nested numerical integration with mpmath for the complex spectral weight.
        Outer integral over scattering momentum s, inner integral over density function.
        """
        mpmath.mp.dps = self._mpmath_dps

        def branch_cut_integrand(s: float) -> float:
            s_mp = mpmath.mpf(s)

            # Time decay over the continuous spectrum
            time_decay = mpmath.exp(-(tau / 2.0) * (nu**2 + s_mp**2))

            # Spectral weight using the complex Gamma function
            gamma_val = mpmath.gamma(mpmath.mpc(-nu/2.0, s_mp/2.0))
            weight = s_mp * mpmath.sinh(mpmath.pi * s_mp) * (mpmath.fabs(gamma_val)**2)

            def inner_u_integrand(u: float) -> float:
                kappa = (1.0 + nu) / 2.0
                mu_param = mpmath.mpc(0, s_mp/2.0)

                whittaker_val = GBMFiniteAnalyticCalculator._mpmath_whittaker_w(kappa, mu_param, u)

                return float((u**((nu - 3.0)/2.0)) * mpmath.re(whittaker_val) * mpmath.exp(-u / 2.0))

            inner_res, _ = quad(inner_u_integrand, z_target, np.inf,
                               epsabs=self._eps)

            return float(time_decay * weight) * inner_res

        cut_integral, _ = quad(branch_cut_integrand, 0.0, self._branch_cut_s_max,
                               epsabs=self._eps,
                               limit=self._quad_limit)

        return cut_integral / (4.0 * np.pi**2)

    def compute_statistics(self, initial_capital: float,
                          annual_withdrawal: float,
                          years: float) -> Dict[str, Any]:
        """
        Compute analytical statistics for finite-horizon survival.

        Args:
            initial_capital: Starting portfolio value
            annual_withdrawal: Annual withdrawal amount
            years: Time horizon

        Returns:
            Dictionary with ruin/survival probabilities, parameters,
            and spectral component breakdown.
        """
        mu = self.market_environment.get_mean(0)
        variance = self.market_environment.get_variance(0)
        w = annual_withdrawal / initial_capital if initial_capital > 0 else 0
        stats = self._exact_spectral_decomposition(mu, variance, w, years)
        ruin_prob = float(np.clip(1.0 - stats[3], 0.0, 1.0))

        return {
            'ruin_probability': ruin_prob,
            'survival_probability': 1.0 - ruin_prob,
            'withdrawal_rate': w,
            'horizon': years,
            'spectral_components': {
                'ground_state': stats[0],
                'bounded_states': stats[1],
                'branch_cut': stats[2],
            },
        }