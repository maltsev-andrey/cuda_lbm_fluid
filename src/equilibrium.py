"""
Equilibrium Distribution Functions

Maxwell-Boltzmann equilibrium for D2Q9 lattice.

The equilibrium distribution is derived from the Maxwell-Boltzmann distribution
truncated to second order in velocity. For the D2Q9 lattice:

    f_i^eq = w_i * rho * [1 + (e_i · u)/c_s^2 + (e_i · u)^2/(2*c_s^4) - u^2/(2*c_s^2)]

where:
    - w_i are the lattice weights
    - e_i are the lattice velocities
    - c_s^2 = 1/3 is the lattice sound speed squared
    - rho is the density
    - u = (ux, uy) is the macroscopic velocity
"""

import numpy as np
from numba import njit, prange
from .lattice import EX, EY, W, CS2, CS4, Q


def compute_equilibrium(rho, ux, uy):
    """
    Compute equilibrium distribution for all lattice sites.
    
    Uses vectorized NumPy operations for CPU efficiency.
    
    Parameters
    ----------
    rho : ndarray
        Density field, shape (ny, nx)
    ux : ndarray
        X-velocity field, shape (ny, nx)
    uy : ndarray
        Y-velocity field, shape (ny, nx)
    
    Returns
    -------
    f_eq : ndarray
        Equilibrium distribution, shape (Q, ny, nx)
        Uses Structure of Arrays (SoA) layout for GPU efficiency.
    """
    ny, nx = rho.shape
    f_eq = np.zeros((Q, ny, nx), dtype=np.float64)
    
    # Precompute velocity-dependent terms
    u_sq = ux * ux + uy * uy  # |u|^2
    
    for i in range(Q):
        # e_i · u
        eu = EX[i] * ux + EY[i] * uy
        
        # Equilibrium formula:
        # f_eq = w * rho * (1 + eu/cs2 + eu^2/(2*cs4) - u^2/(2*cs2))
        f_eq[i] = W[i] * rho * (
            1.0 
            + eu / CS2 
            + (eu * eu) / (2.0 * CS4) 
            - u_sq / (2.0 * CS2)
        )
    
    return f_eq


@njit(parallel=True, cache=True)
def compute_equilibrium_numba(rho, ux, uy, f_eq, ex, ey, w, cs2, cs4):
    """
    Numba-accelerated equilibrium computation.
    
    Parameters
    ----------
    rho : ndarray
        Density field, shape (ny, nx)
    ux : ndarray
        X-velocity field, shape (ny, nx)
    uy : ndarray
        Y-velocity field, shape (ny, nx)
    f_eq : ndarray
        Output equilibrium distribution, shape (Q, ny, nx)
    ex, ey : ndarray
        Lattice velocity components
    w : ndarray
        Lattice weights
    cs2, cs4 : float
        Sound speed squared and fourth power
    """
    q, ny, nx = f_eq.shape
    
    for j in prange(ny):
        for i in range(nx):
            rho_ij = rho[j, i]
            ux_ij = ux[j, i]
            uy_ij = uy[j, i]
            u_sq = ux_ij * ux_ij + uy_ij * uy_ij
            
            for k in range(q):
                eu = ex[k] * ux_ij + ey[k] * uy_ij
                f_eq[k, j, i] = w[k] * rho_ij * (
                    1.0 
                    + eu / cs2 
                    + (eu * eu) / (2.0 * cs4) 
                    - u_sq / (2.0 * cs2)
                )


def compute_equilibrium_fast(rho, ux, uy):
    """
    Fast equilibrium computation using Numba.
    
    Parameters
    ----------
    rho : ndarray
        Density field, shape (ny, nx)
    ux : ndarray
        X-velocity field, shape (ny, nx)
    uy : ndarray
        Y-velocity field, shape (ny, nx)
    
    Returns
    -------
    f_eq : ndarray
        Equilibrium distribution, shape (Q, ny, nx)
    """
    ny, nx = rho.shape
    f_eq = np.zeros((Q, ny, nx), dtype=np.float64)
    
    # Convert to float64 arrays for Numba
    ex = EX.astype(np.float64)
    ey = EY.astype(np.float64)
    
    compute_equilibrium_numba(rho, ux, uy, f_eq, ex, ey, W, CS2, CS4)
    
    return f_eq


def equilibrium_single_site(rho, ux, uy):
    """
    Compute equilibrium distribution for a single lattice site.
    
    Useful for boundary conditions and testing.
    
    Parameters
    ----------
    rho : float
        Density at the site
    ux : float
        X-velocity at the site
    uy : float
        Y-velocity at the site
    
    Returns
    -------
    f_eq : ndarray
        Equilibrium distribution, shape (Q,)
    """
    f_eq = np.zeros(Q, dtype=np.float64)
    u_sq = ux * ux + uy * uy
    
    for i in range(Q):
        eu = EX[i] * ux + EY[i] * uy
        f_eq[i] = W[i] * rho * (
            1.0 
            + eu / CS2 
            + (eu * eu) / (2.0 * CS4) 
            - u_sq / (2.0 * CS2)
        )
    
    return f_eq
