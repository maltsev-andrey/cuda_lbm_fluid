"""
Macroscopic Observable Extraction

Compute density, velocity, and derived quantities from distributions.

In LBM, macroscopic quantities are moments of the distribution function:
    - Density (0th moment): rho = sum_i(f_i)
    - Momentum (1st moment): rho*u = sum_i(f_i * e_i)
    - Stress tensor (2nd moment): Pi = sum_i(f_i * e_i * e_i)
"""

import numpy as np
from numba import njit, prange
from .lattice import EX, EY, Q


def compute_density(f):
    """
    Compute density field from distribution functions.
    
    rho = sum_i(f_i)
    
    Parameters
    ----------
    f : ndarray
        Distribution functions, shape (Q, ny, nx)
    
    Returns
    -------
    rho : ndarray
        Density field, shape (ny, nx)
    """
    return np.sum(f, axis=0)


def compute_velocity(f, rho=None):
    """
    Compute velocity field from distribution functions.
    
    rho * u = sum_i(f_i * e_i)
    
    Parameters
    ----------
    f : ndarray
        Distribution functions, shape (Q, ny, nx)
    rho : ndarray, optional
        Density field, shape (ny, nx). If None, computed from f.
    
    Returns
    -------
    ux : ndarray
        X-velocity field, shape (ny, nx)
    uy : ndarray
        Y-velocity field, shape (ny, nx)
    """
    if rho is None:
        rho = compute_density(f)
    
    # Compute momentum
    # rho * ux = sum_i(f_i * ex_i)
    # rho * uy = sum_i(f_i * ey_i)
    
    q, ny, nx = f.shape
    
    # Vectorized computation
    rho_ux = np.zeros((ny, nx), dtype=np.float64)
    rho_uy = np.zeros((ny, nx), dtype=np.float64)
    
    for i in range(Q):
        rho_ux += f[i] * EX[i]
        rho_uy += f[i] * EY[i]
    
    # Avoid division by zero
    rho_safe = np.where(rho > 1e-10, rho, 1.0)
    
    ux = rho_ux / rho_safe
    uy = rho_uy / rho_safe
    
    return ux, uy


def compute_macroscopic(f):
    """
    Compute all macroscopic quantities from distribution functions.
    
    Parameters
    ----------
    f : ndarray
        Distribution functions, shape (Q, ny, nx)
    
    Returns
    -------
    rho : ndarray
        Density field, shape (ny, nx)
    ux : ndarray
        X-velocity field, shape (ny, nx)
    uy : ndarray
        Y-velocity field, shape (ny, nx)
    """
    rho = compute_density(f)
    ux, uy = compute_velocity(f, rho)
    return rho, ux, uy


@njit(parallel=True, cache=True)
def compute_macroscopic_numba(f, rho, ux, uy, ex, ey):
    """
    Numba-accelerated macroscopic quantity computation.
    
    Parameters
    ----------
    f : ndarray
        Distribution functions, shape (Q, ny, nx)
    rho : ndarray
        Output density field, shape (ny, nx)
    ux : ndarray
        Output X-velocity field, shape (ny, nx)
    uy : ndarray
        Output Y-velocity field, shape (ny, nx)
    ex, ey : ndarray
        Lattice velocity components
    """
    q, ny, nx = f.shape
    
    for j in prange(ny):
        for i in range(nx):
            rho_local = 0.0
            rho_ux = 0.0
            rho_uy = 0.0
            
            for k in range(q):
                f_k = f[k, j, i]
                rho_local += f_k
                rho_ux += f_k * ex[k]
                rho_uy += f_k * ey[k]
            
            rho[j, i] = rho_local
            
            if rho_local > 1e-10:
                ux[j, i] = rho_ux / rho_local
                uy[j, i] = rho_uy / rho_local
            else:
                ux[j, i] = 0.0
                uy[j, i] = 0.0


def compute_macroscopic_fast(f):
    """
    Fast macroscopic quantity computation using Numba.
    
    Parameters
    ----------
    f : ndarray
        Distribution functions, shape (Q, ny, nx)
    
    Returns
    -------
    rho : ndarray
        Density field, shape (ny, nx)
    ux : ndarray
        X-velocity field, shape (ny, nx)
    uy : ndarray
        Y-velocity field, shape (ny, nx)
    """
    q, ny, nx = f.shape
    rho = np.zeros((ny, nx), dtype=np.float64)
    ux = np.zeros((ny, nx), dtype=np.float64)
    uy = np.zeros((ny, nx), dtype=np.float64)
    
    ex = EX.astype(np.float64)
    ey = EY.astype(np.float64)
    
    compute_macroscopic_numba(f, rho, ux, uy, ex, ey)
    
    return rho, ux, uy


def compute_vorticity(ux, uy, dx=1.0):
    """
    Compute vorticity field using central differences.
    
    omega = du_y/dx - du_x/dy
    
    Parameters
    ----------
    ux : ndarray
        X-velocity field, shape (ny, nx)
    uy : ndarray
        Y-velocity field, shape (ny, nx)
    dx : float
        Grid spacing (default 1.0 in lattice units)
    
    Returns
    -------
    vorticity : ndarray
        Vorticity field, shape (ny, nx)
    """
    # Central differences with periodic boundary handling
    # du_y/dx
    duy_dx = (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1)) / (2.0 * dx)
    
    # du_x/dy
    dux_dy = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) / (2.0 * dx)
    
    return duy_dx - dux_dy


def compute_strain_rate(ux, uy, dx=1.0):
    """
    Compute strain rate tensor magnitude.
    
    |S| = sqrt(2 * S_ij * S_ij)
    
    Parameters
    ----------
    ux : ndarray
        X-velocity field, shape (ny, nx)
    uy : ndarray
        Y-velocity field, shape (ny, nx)
    dx : float
        Grid spacing
    
    Returns
    -------
    strain_rate : ndarray
        Strain rate magnitude, shape (ny, nx)
    """
    # Velocity gradients
    dux_dx = (np.roll(ux, -1, axis=1) - np.roll(ux, 1, axis=1)) / (2.0 * dx)
    dux_dy = (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0)) / (2.0 * dx)
    duy_dx = (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1)) / (2.0 * dx)
    duy_dy = (np.roll(uy, -1, axis=0) - np.roll(uy, 1, axis=0)) / (2.0 * dx)
    
    # Strain rate tensor components: S_ij = 0.5 * (du_i/dx_j + du_j/dx_i)
    s_xx = dux_dx
    s_yy = duy_dy
    s_xy = 0.5 * (dux_dy + duy_dx)
    
    # Magnitude: |S| = sqrt(2 * S_ij * S_ij)
    return np.sqrt(2.0 * (s_xx**2 + s_yy**2 + 2.0 * s_xy**2))


def compute_pressure(rho, cs2=1.0/3.0):
    """
    Compute pressure field from density.
    
    For incompressible LBM: p = rho * c_s^2
    
    Parameters
    ----------
    rho : ndarray
        Density field, shape (ny, nx)
    cs2 : float
        Sound speed squared (default 1/3)
    
    Returns
    -------
    pressure : ndarray
        Pressure field, shape (ny, nx)
    """
    return rho * cs2


def compute_velocity_magnitude(ux, uy):
    """
    Compute velocity magnitude field.
    
    |u| = sqrt(ux^2 + uy^2)
    
    Parameters
    ----------
    ux : ndarray
        X-velocity field, shape (ny, nx)
    uy : ndarray
        Y-velocity field, shape (ny, nx)
    
    Returns
    -------
    velocity_mag : ndarray
        Velocity magnitude, shape (ny, nx)
    """
    return np.sqrt(ux * ux + uy * uy)