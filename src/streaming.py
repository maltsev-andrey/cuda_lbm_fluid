"""
Streaming Step Implementations

Propagation of distribution functions along lattice velocities.

The streaming step moves each distribution f_i from site x to site x + e_i:
    f_i(x + e_i, t + dt) = f_i^out(x, t)

Two schemes are commonly used:
- Push: Write f_i from x to x + e_i (scatter)
- Pull: Read f_i at x from x - e_i (gather)

The pull scheme is preferred for GPUs due to coalesced writes.
"""

import numpy as np
from numba import njit, prange
from .lattice import EX, EY, Q


def stream_periodic(f):
    """
    Streaming step with periodic boundary conditions.
    
    Uses push scheme: f_i(x + e_i) = f_i(x)
    Implemented with np.roll for efficiency.
    
    Parameters
    ----------
    f : ndarray
        Distribution functions, shape (Q, ny, nx)
    
    Returns
    -------
    f_streamed : ndarray
        Post-streaming distribution
    """
    f_out = np.zeros_like(f)

    for i in range(Q):
        # Roll in x-direction by ex[i], y-direction by ey[i]
        # Note: np.roll uses opposite sign convention
        f_out[i] = np.roll(np.roll(f[i], EX[i], axis=1), EY[i], axis=0)

    return f_out


def stream_periodic_pull(f):
    """
    Streaming step using pull scheme with periodic boundaries.
    
    Pull scheme: f_i(x) = f_i(x - e_i)
    Better for GPU implementation due to coalesced writes.
    
    Parameters
    ----------
    f : ndarray
        Distribution functions, shape (Q, ny, nx)
    
    Returns
    -------
    f_streamed : ndarray
        Post-streaming distribution
    """
    f_out = np.zeros_like(f)

    for i in range(Q):
        # Pull from x - e_i (opposite direction of push)
        f_out[i] = np.roll(np.roll(f[i], -EX[i], axis=1), -EY[i], axis=0)

    return f_out


@njit(parallel=True, cache=True)    
def stream_periodic_numba(f, f_out, ex, ey):
    """
    Numba-accelerated streaming with periodic boundaries.
    
    Uses pull scheme for GPU-friendly access pattern.
    
    Parameters
    ----------
    f : ndarray
        Input distribution functions, shape (Q, ny, nx)
    f_out : ndarray
        Output distribution functions, shape (Q, ny, nx)
    ex, ey : ndarray
        Lattice velocity components
    """
    q, ny, nx = f.shape

    for j in prange(ny):
        for i in range(nx):
            for k in range(q):
                # Source coordinates with periodic wrapping
                i_src = (i - int(ex[k]) + nx) % nx
                j_src = (j - int(ey[k]) + ny) % ny

                f_out[k, j, i] = f[k, j_src, i_src]

                
def stream_periodic_fast(f):
    """
    Fast streaming using Numba.
    
    Parameters
    ----------
    f : ndarray
        Distribution functions, shape (Q, ny, nx)
    
    Returns
    -------
    f_streamed : ndarray
        Post-streaming distribution
    """
    f_out = np.zeros_like(f)
    ex = EX.astype(np.float64)
    ey = EY.astype(np.float64)
    stream_periodic_numba(f, f_out, ex, ey)
    return f_out


@njit(cache=True)    
def stream_and_collide_step(f_in, f_out, rho, ux, uy, omega, ex, ey, w, cs2, cs4):
    """
    Fused streaming and collision step (CPU version).
    
    Combines streaming and collision into a single pass:
    1. Pull distributions from neighbors
    2. Compute macroscopic quantities
    3. Compute equilibrium
    4. Apply BGK collision
    
    Parameters
    ----------
    f_in : ndarray
        Input distribution, shape (Q, ny, nx)
    f_out : ndarray
        Output distribution, shape (Q, ny, nx)
    rho : ndarray
        Output density field, shape (ny, nx)
    ux, uy : ndarray
        Output velocity fields, shape (ny, nx)
    omega : float
        Relaxation frequency (1/tau)
    ex, ey : ndarray
        Lattice velocities
    w : ndarray
        Lattice weights
    cs2, cs4 : float
        Sound speed constants
    """
    q, ny, nx = f_in.shape

    for j in range(ny):
        for i in range(nx):
            # Pull and accumulate microscopic quantities
            f_local = np.zeros(9, dtype=np.float64)
            rho_local = 0.0
            rho_ux = 0.0
            rho_uy = 0.0

            for k in range(q):
                # Source coordinates
                i_src = (i - int(ex[k]) + nx) % nx
                j_src = (j - int(ey[k]) + ny) % ny

                f_local[k] = f_in[k, j_src, i_src]
                rho_local +=f_local[k]
                rho_ux += f_local[k] * ex[k]
                rho_uy += f_local[k] *ey[k]

            # Velocity
            if rho_local > 1e-10:
                ux_local = rho_ux / rho_local
                uy_local = rho_uy / rho_local
            else:
                ux_local = 0.0
                uy_local = 0.0
            
            # Store macroscopic quantities
            rho[j, i] = rho_local
            ux[j, i] = ux_local
            uy[j, i] = uy_local
            
            # Compute equilibrium and collide
            u_sq = ux_local * ux_local + uy_local * uy_local
            
            for k in range(q):
                eu = ex[k] * ux_local + ey[k] * uy_local
                f_eq = w[k] * rho_local * (
                    1.0 + eu / cs2 + (eu * eu) / (2.0 * cs4) - u_sq / (2.0 * cs2)
                )
                f_out[k, j, i] = f_local[k] - omega * (f_local[k] - f_eq)


def collide_and_stream(f, tau):
    """
    Combined collision and streaming step.
    
    Order: Collision first, then streaming.
    
    Parameters
    ----------
    f : ndarray
        Distribution functions, shape (Q, ny, nx)
    tau : float
        Relaxation time
    
    Returns
    -------
    f_out : ndarray
        Updated distribution
    rho : ndarray
        Density field
    ux, uy : ndarray
        Velocity fields
    """
    from .equilibrium import compute_equilibrium_fast
    from .observables import compute_macroscopic_fast
    from .collision import bgk_collision_fast

    # Compute microscopic quantities
    rho, ux, uy = compute_macroscopic_fast(f)

    # Compute equilibrium and collide
    f_eq = compute_equilibrium_fast(rho, ux, uy)

    # Collision
    f_coll = bgk_collision_fast(f, f_eq, tau)

    # Streaming
    f_out = stream_periodic_fast(f_coll)

    return f_out, rho, ux, uy
   