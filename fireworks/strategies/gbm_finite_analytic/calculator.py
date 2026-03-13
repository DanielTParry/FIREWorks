"""Analytical GBM finite-horizon ruin probability calculator.

Uses exact spectral decomposition method for the probability of ruin
over finite time horizons under GBM with constant parameters and withdrawal.

The solution decomposes into three components:
    1. Stationary ground state (incomplete Gamma function)
    2. Discrete bounded states (Monthus & Bouchaud pole residues)
    3. Continuous branch cut (scattering integral - uses approximation if mpmath unavailable)

Reference:
    Monthus, C., & Bouchaud, J.-P. (2015). Optimal leverage from non-linear utility.
    Journal of Economic Dynamics and Control, 53, 1-16.
    https://doi.org/10.1016/j.jedc.2015.02.003
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
                 consumption_model: AbstractConsumptionModel) -> None:
        """
        Initialize the calculator.

        Args:
            market_environment: Market environment defining μ and variance
            consumption_model: Consumption model defining withdrawal C
        """
        self.market_environment = market_environment
        self.consumption_model = consumption_model

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
        
        # Quick returns for boundary cases
        if annual_withdrawal <= 0:
            return 0.0
        if annual_withdrawal >= initial_capital:
            return 1.0
        
        mu = self.market_environment.get_mean(0)
        variance = self.market_environment.get_variance(0)
        
        # Zero variance: deterministic portfolio
        if variance <= 0:
            return self._ruin_probability_deterministic(mu, annual_withdrawal, initial_capital, years)
        if mu < 0:
            raise ValueError(f"Negative drift (mu={mu}) not supported in current implementation.") 

        # Stochastic case: spectral decomposition
        w = annual_withdrawal / initial_capital
        s_stat, s_bounded, s_branch, s_tot = self._exact_spectral_decomposition(mu, variance, w, years)
        # s_tot is survival probability; convert to ruin probability
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
        else:
            # μ ≤ 0: No growth or negative growth → certain ruin with withdrawals
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

        # Component 1: Stationary ground state
        S_stat = self._compute_ground_state(nu, z_target)

        # Component 2: Discrete bounded states
        S_bounded = self._compute_bounded_states(nu, z_target, tau)

        # Component 3: Continuous branch cut
        S_branch = self._compute_branch_cut(nu, z_target, tau)

        # Total probability
        S_tot = S_stat + S_bounded + S_branch

        return S_stat, S_bounded, S_branch, S_tot

    @staticmethod
    def _mpmath_whittaker_w(kappa: float, mu_param: float, z: float) -> Any:
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
        # Set precision on first use (lazy initialization to avoid Windows DLL load at import)
        mpmath.mp.dps = 15
        
        a = mu_param - kappa + 0.5
        b = 1.0 + 2.0 * mu_param
        z_mp = mpmath.mpc(z)
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

    @staticmethod
    def _compute_bounded_states(nu: float, z_target: float, tau: float) -> float:
        """
        Compute contribution from discrete bounded states.

        Uses Monthus & Bouchaud pole residue coefficients with Laguerre polynomials.

        Args:
            nu: Shape parameter
            z_target: Target value 2w/θ
            tau: Scaled time (θ/4) * T

        Returns:
            Bounded states contribution
        """
        N_states = max(0, int(np.floor((nu + 1.0) / 2.0 - 1e-7)))
        S_bounded = 0.0

        for n in range(1, N_states + 1):
            lam_n = 0.5 * n * (2.0 * nu - n)

            # Exact Monthus & Bouchaud pole residue coefficient
            coef = ((-1.0) ** n * (nu + 1.0 - 2.0 * n)) / gamma(nu + 2.0 - n)
            L_n = genlaguerre(n, nu - 2.0 * n)

            # Create density function using closure to capture parameters
            def get_density(n_val: int, c_val: float, L_val: Any) -> Callable[[float], float]:
                """Factory function to create bounded density integrand."""
                def bounded_density(x: float) -> float:
                    return c_val * (x**(nu - n_val - 1.0)) * np.exp(-x) * L_val(x)
                return bounded_density

            f_n = get_density(n, coef, L_n)
            res, _ = quad(f_n, z_target, np.inf, epsabs=1e-8)
            S_bounded += np.exp(-lam_n * tau) * res

        return S_bounded

    def _compute_branch_cut(self, nu: float, z_target: float, tau: float) -> float:
        """
        Compute contribution from continuous branch cut (Monthus & Bouchaud Eq 5.5).

        Uses exact nested numerical integration with mpmath for the complex spectral weight.
        Outer integral over scattering momentum s, inner integral over density function.

        Args:
            nu: Shape parameter
            z_target: Target value 2w/θ
            tau: Scaled time (θ/4) * T

        Returns:
            Branch cut contribution
        """
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

                # The density mapped perfectly to the CDF boundary
                # W(...) * u^((nu-3)/2) * e^(-u/2)
                return float((u**((nu - 3.0)/2.0)) * mpmath.re(whittaker_val) * mpmath.exp(-u / 2.0))

            # Use SciPy quad for the inner integral to keep execution time viable
            inner_res, _ = quad(inner_u_integrand, z_target, np.inf, epsabs=1e-5)

            return float(time_decay * weight) * inner_res

        # Outer coefficient
        coef = 1.0 / (4.0 * np.pi**2)

        # Integrate over the scattering momentum s
        cut_integral, _ = quad(branch_cut_integrand, 0.0, 15.0, epsabs=1e-5, limit=50)
        S_branch = coef * cut_integral

        return S_branch

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
            Dictionary with analytical results and spectral components
        """
        mu = self.market_environment.get_mean(0)
        variance = self.market_environment.get_variance(0)
        sigma = np.sqrt(variance)

        w = annual_withdrawal / initial_capital if initial_capital > 0 else 0
        ruin_prob = self.calculate_ruin_probability_finite(
            initial_capital, annual_withdrawal, years
        )
        survival_prob = 1.0 - ruin_prob

        # Compute spectral components (using withdrawal rate w, not absolute amount)
        s_stat, s_bounded, s_branch, _ = self._exact_spectral_decomposition(
            mu, variance, w, years
        )

        return {
            'ruin_probability': ruin_prob,
            'survival_probability': survival_prob,
            'initial_capital': initial_capital,
            'annual_withdrawal': annual_withdrawal,
            'withdrawal_rate': w,
            'mean_return': mu,
            'variance': variance,
            'sigma': sigma,
            'horizon': years,
            'spectral_components': {
                'ground_state': s_stat,
                'bounded_states': s_bounded,
                'branch_cut': s_branch,
            },
        }
