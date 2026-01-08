"""
Collision Operators

BGK and TRT collision models for LBM.

The collision step models molecular interactions and drives the distribution
toward equilibrium. The relaxation time tau controls the viscosity:

    nu = c_s^2 * (tau - 0.5) * dt

where c_s^2 = 1/3 for D2Q9 and dt = 1 in lattice units.

Stability requires tau > 0.5 (nu > 0).
"""

import numpy as np
from numba import njit, prange
from .lattice import EX, EY, W, CS2, CS4, Q, OPPOSITE
from .equilibrium import compute_equilibrium, compute_equilibrium_fast
from .observables import compute_macroscopic, compute_macroscopic_fast


def tau_from_viscosity(nu, dt=1.0, cs2=1.0/3.0):
    """
    Compute relaxation time from kinematic viscosity.
    
    tau = nu / (c_s^2 * dt) + 0.5
    
    Parameters
    ----------
    nu : float
        Kinematic viscosity
    dt : float
        Time step (default 1.0 in lattice units)
    cs2 : float
        Sound speed squared (default 1/3)
    
    Returns
    -------
    tau : float
        Relaxation time
    """
    return nu / (cs2* dt) + 0.5


def viscosity_from_tau(tau, dt=1.0, cs2=1.0/3.0):
    """
    Compute kinematic viscosity from relaxation time.
    
    nu = c_s^2 * (tau - 0.5) * dt
    
    Parameters
    ----------
    tau : float
        Relaxation time (must be > 0.5)
    dt : float
        Time step (default 1.0 in lattice units)
    cs2 : float
        Sound speed squared (default 1/3)
    
    Returns
    -------
    nu : float
        Kinematic viscosity
    """
    if tau <= 0.5:
        raise ValueError(f"tau must be > 0.5 for stability, got {tau}")
    return cs2 * (tau - 0.5) * dt
    

def bgk_collision(f, f_eq, tau):
    """
    BGK (Bhatnagar-Gross-Krook) collision operator.
    
    f_out = f - (f - f_eq) / tau
    
    The BGK operator is the simplest single-relaxation-time model.
    
    Parameters
    ----------
    f : ndarray
        Distribution functions (Q, ny, nx)
    f_eq : ndarray
        Equilibrium distribution (Q, ny, nx)
    tau : float
        Relaxation time (tau > 0.5 for stability)
    
    Returns
    -------
    f_out : ndarray
        Post-collision distribution
    """
    if tau <= 0.5:
        raise ValueError(f"tau must be > 0.5 for stability, got {tau}")

    omega = 1.0 / tau # Relaxion frequency
    return f - omega * (f - f_eq)


@njit(parallel=True, cache=True)
def bgk_collision_numba(f, f_eq, omega, f_out):
    """
    Numba-accelerated BGK collision.
    
    Parameters
    ----------
    f : ndarray
        Distribution functions, shape (Q, ny, nx)
    f_eq : ndarray
        Equilibrium distribution, shape (Q, ny, nx)
    omega : float
        Relaxation frequency (1/tau)
    f_out : ndarray
        Output post-collision distribution, shape (Q, ny, nx)
    """
    q, ny, nx = f.shape

    for j in prange(ny):
        for i in range(nx):
            for k in range(q):
                f_out[k, j, i] = f[k, j, i] - omega * (f[k, j, i] - f_eq[k, j, i])


def bgk_collision_fast(f, f_eq, tau):
    """
    Numba-accelerated BGK collision.
    
    Parameters
    ----------
    f : ndarray
        Distribution functions, shape (Q, ny, nx)
    f_eq : ndarray
        Equilibrium distribution, shape (Q, ny, nx)
   tau : float
        Relaxation time
    
    Returns
    -------
    f_out : ndarray
        Post-collision distribution
    """
    if tau <= 0.5:
        raise ValueError(f"tau must be > 0.5 for stability, got {tau}")

    omega = 1.0 / tau
    f_out = np.zeros_like(f)
    bgk_collision_numba(f, f_eq, omega, f_out)
    return f_out

    
def bgk_collision_inplace(f, tau):
    """
    BGK collision with in-place equilibrium computation.
    
    Computes equilibrium internally and updates f in place.
    
    Parameters
    ----------
    f : ndarray
        Distribution functions, shape (Q, ny, nx). Modified in place.
    tau : float
        Relaxation time
    
    Returns
    -------
    rho : ndarray
        Density field
    ux : ndarray
        X-velocity field
    uy : ndarray
        Y-velocity field
    """
    if tau <= 0.5:
        raise ValueError(f"tau must be > 0.5 for stability, got {tau}")

    # Compute macroscopic quantities
    rho, ux, uy = compute_macroscopic_fast(f)

    # Compute equilibrium
    f_eq = compute_equilibrium_fast(rho, ux, uy)

    # Apply collision
    omega = 1.0 / tau
    f -= omega * (f - f_eq)

    return rho, ux, uy

    
def trt_collision(f, f_eq, tau_plus, tau_minus=None, magic_param=0.25):
    """
    TRT (Two-Relaxation-Time) collision operator.
    
    Separates the distribution into symmetric and antisymmetric parts:
        f^+ = 0.5 * (f_i + f_i*)     (symmetric)
        f^- = 0.5 * (f_i - f_i*)     (antisymmetric)
    
    Each part relaxes with its own rate:
        f_out = f - (f^+ - f_eq^+)/tau_+ - (f^- - f_eq^-)/tau_-
    
    The "magic parameter" Lambda = (tau_+ - 0.5)(tau_- - 0.5) controls
    stability. Lambda = 1/4 is optimal for many boundary conditions.
    
    Parameters
    ----------
    f : ndarray
        Distribution functions, shape (Q, ny, nx)
    f_eq : ndarray
        Equilibrium distribution, shape (Q, ny, nx)
    tau_plus : float
        Relaxation time for symmetric part (controls viscosity)
    tau_minus : float, optional
        Relaxation time for antisymmetric part.
        If None, computed from magic_param.
    magic_param : float
            Magic parameter Lambda (default 0.25 for optimal stability)
    
    Returns
    -------
    f_out : ndarray
        Post-collision distribution
    """
    if tau_plus <= 0.5:
        raise ValueError(f"tau_plus must be > 0.5, got {tau_plus}")
       
    # Compute tau_minus from magic parameter if not provided
    if tau_minus is None:
        tau_minus = magic_param / (tau_plus - 0.5) + 0.5

    if tau_minus <= 0.5:
        raise ValueError(f"tau_minus must be > 0.5, got {tau_minus}")

    omega_plus = 1.0 / tau_plus
    omega_minus = 1.0 / tau_minus

    q, ny, nx = f.shape
    f_out = np.zeros_like(f)

   # Process pairs of opposite directions
    for i in range(Q):
        i_opp = OPPOSITE[i]

        if i <= i_opp: # Process each pair once
            # Symmetric and antisymmetric parts of f
            f_plus = 0.5 * (f[i] + f[i_opp])
            f_minus = 0.5 * (f[i] - f[i_opp])

            # Symmetric and antisymmetric parts of f_eq
            f_eq_plus = 0.5 * (f_eq[i] + f_eq[i_opp])
            f_eq_minus = 0.5 * (f_eq[i] - f_eq[i_opp])

            # Non-equilibrium parts
            f_neq_plus = f_plus - f_eq_plus
            f_neq_minus = f_minus - f_eq_minus
            
            # Apply TRT collision
            f_out[i] = f[i] - omega_plus * f_neq_plus - omega_minus * f_neq_minus
            if i != i_opp:
                    f_out[i_opp] = f[i_opp] - omega_plus * f_neq_plus + omega_minus * f_neq_minus

    return f_out


@njit(parallel=True, cache=True)
def trt_collision_numba(f, f_eq, omega_plus, omega_minus, opposite, f_out):
    """
    Numba-accelerated TRT collision.
    
    Parameters
    ----------
    f : ndarray
        Distribution functions, shape (Q, ny, nx)
    f_eq : ndarray
        Equilibrium distribution, shape (Q, ny, nx)
    omega_plus : float
        Relaxation frequency for symmetric part
    omega_minus : float
        Relaxation frequency for antisymmetric part
    opposite : ndarray
        Opposite direction indices
    f_out : ndarray
        Output post-collision distribution
    """
    q, ny, nx = f.shape

    for j in prange(ny):
        for i in range(nx):
            for k in range(q):
                k_opp = opposite[k]

                # Symmetric and antisymmetric parts
                f_plus = 0.5 * (f[k, j, i] + f[k_opp, j, i])
                f_minus = 0.5 * (f[k, j, i] - f[k_opp, j, i])

                f_eq_plus = 0.5 * (f_eq[k, j, i] + f_eq[k_opp, j, i])
                f_eq_minus = 0.5 * (f_eq[k, j, i] - f_eq[k_opp, j, i])

                # Non-equilibrium parts
                f_neq_plus = f_plus - f_eq_plus
                f_neq_minus = f_minus - f_eq_minus

                # TRT collision
                f_out[k, j, i] = f[k, j, i] - omega_plus * f_neq_plus - omega_minus * f_neq_minus


def trt_collision_fast(f, f_eq, tau_plus, tau_minus=None, magic_param = 0.25):
    """
    Fast TRT collision using Numba.
    
    Parameters
    ----------
    f : ndarray
        Distribution functions, shape (Q, ny, nx)
    f_eq : ndarray
        Equilibrium distribution, shape (Q, ny, nx)
    tau_plus : float
        Relaxation time for symmetric part
    tau_minus : float, optional
        Relaxation time for antisymmetric part
    magic_param : float
        Magic parameter (default 0.25)
    
    Returns
    -------
    f_out : ndarray
        Post-collision distribution
    """
    if tau_plus <= 0.5:
        raise ValueError(f"tau_plus must be > 0.5, got {tau_plus}")

    if tau_minus is None:
        tau_minus = magic_param / (tau_plus - 0.5) + 0.5

    omega_plus = 1.0 / tau_plus
    omega_minus = 1.0 / tau_minus

    f_out = np.zeros_like(f)
    trt_collision_numba(f, f_eq, omega_plus, omega_minus, OPPOSITE, f_out)
    return f_out


def validate_tau(tau, name = "tau"):
    """
    Validate that relaxation time is in stable range.
    
    Parameters
    ----------
    tau : float
        Relaxation time to validate
    name : str
        Name for error messages
    
    Raises
    ------
    ValueError
        If tau <= 0.5
    
    Returns
    -------
    tau : float
        Validated tau value
    """
    if tau <= 0.5:
        raise ValueError(
            f"{name} must be > 0.5 for stability (got {tau}). "
            f"This corresponds to nu > 0."
        )
    if tau > 2.0:
        import warnings
        warnings.warn(
            f"{name} = {tau} is large, which may cause slow convergence. "
            f"Consider tau in range (0.5, 2.0) for efficiency."
        )
    return tau