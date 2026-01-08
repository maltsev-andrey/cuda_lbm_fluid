"""
Boundary Condition Handlers

Implements various boundary conditions for LBM:
- Bounce-back (no-slip walls)
- Zou-He (velocity/pressure boundaries)
- Periodic (handled in streaming)
- Equilibrium (open boundaries)

Boundary conditions are critical for physical accuracy and numerical stability.
"""

import numpy as np
from numba import njit, prange
from .lattice import EX, EY, W, CS2, Q, OPPOSITE
from .equilibrium import equilibrium_single_site


# Boundary type flags
FLAG_FLUID = 0
FLAG_SOLID = 1
FLAG_VELOCITY_INLET = 2
FLAG_PRESSURE_OUTLET = 3
FLAG_MOVING_WALL = 4


def apply_bounce_back(f, solid_mask):
    """
    Apply bounce-back boundary condition for solid walls (no-slip).
    
    The bounce-back rule reflects distributions back in the opposite direction:
        f_i'(x_wall) = f_i*(x_wall)
    
    where i* is the opposite direction of i.
    
    Parameters
    ----------
    f : ndarray
        Distribution functions, shape (Q, ny, nx)
    solid_mask : ndarray
        Boolean mask for solid nodes, shape (ny, nx)
    
    Returns
    -------
    f : ndarray
        Distribution with bounce-back applied
    """
    f_new = f.copy()
    
    for i in range(Q):
        i_opp = OPPOSITE[i]
        # At solid nodes, swap with opposite direction
        f_new[i, solid_mask] = f[i_opp, solid_mask]
    
    return f_new


@njit(parallel=True, cache=True)
def apply_bounce_back_numba(f, f_out, solid_mask, opposite):
    """
    Numba-accelerated bounce-back.
    
    Parameters
    ----------
    f : ndarray
        Input distribution, shape (Q, ny, nx)
    f_out : ndarray
        Output distribution, shape (Q, ny, nx)
    solid_mask : ndarray
        Boolean solid mask, shape (ny, nx)
    opposite : ndarray
        Opposite direction indices
    """
    q, ny, nx = f.shape
    
    for j in prange(ny):
        for i in range(nx):
            if solid_mask[j, i]:
                # Bounce-back: swap with opposite direction
                for k in range(q):
                    f_out[k, j, i] = f[opposite[k], j, i]
            else:
                # Fluid node: copy unchanged
                for k in range(q):
                    f_out[k, j, i] = f[k, j, i]


def apply_bounce_back_fast(f, solid_mask):
    """
    Fast bounce-back using Numba.
    
    Parameters
    ----------
    f : ndarray
        Distribution functions, shape (Q, ny, nx)
    solid_mask : ndarray
        Boolean mask for solid nodes, shape (ny, nx)
    
    Returns
    -------
    f_out : ndarray
        Distribution with bounce-back applied
    """
    f_out = np.zeros_like(f)
    apply_bounce_back_numba(f, f_out, solid_mask, OPPOSITE)
    return f_out


def apply_moving_wall_bounce_back(f, wall_mask, wall_velocity, rho_wall=1.0):
    """
    Apply bounce-back with moving wall (e.g., lid-driven cavity).
    
    The moving wall bounce-back rule (Ladd 1994):
        f_i(x_b, t+dt) = f_{i*}(x_b, t) + 2 * w_i * rho * (e_i · u_w) / c_s^2
    
    where:
        - i* is the opposite direction of i
        - e_i is the lattice velocity of direction i
        - u_w is the wall velocity
    
    This injects momentum into the fluid from the moving wall.
    
    Parameters
    ----------
    f : ndarray
        Distribution functions, shape (Q, ny, nx)
    wall_mask : ndarray
        Boolean mask for wall nodes, shape (ny, nx)
    wall_velocity : tuple
        Wall velocity (ux, uy)
    rho_wall : float
        Assumed density at wall (default 1.0)
    
    Returns
    -------
    f : ndarray
        Distribution with moving wall BC applied
    """
    f_new = f.copy()
    ux_wall, uy_wall = wall_velocity
    
    for i in range(Q):
        i_opp = OPPOSITE[i]
        # Use e_i (not e_{i*}) for the velocity dot product
        eu = EX[i] * ux_wall + EY[i] * uy_wall
        # Moving wall correction using w_i
        correction = 2.0 * W[i] * rho_wall * eu / CS2
        f_new[i, wall_mask] = f[i_opp, wall_mask] + correction
    
    return f_new


@njit(cache=True)
def apply_moving_wall_numba(f, f_out, wall_mask, ux_wall, uy_wall, rho_wall,
                            ex, ey, w, cs2, opposite):
    """
    Numba-accelerated moving wall bounce-back.
    """
    q, ny, nx = f.shape
    
    for j in range(ny):
        for i in range(nx):
            if wall_mask[j, i]:
                for k in range(q):
                    k_opp = opposite[k]
                    eu = ex[k] * ux_wall + ey[k] * uy_wall
                    correction = 2.0 * w[k] * rho_wall * eu / cs2
                    f_out[k, j, i] = f[k_opp, j, i] - correction
            else:
                for k in range(q):
                    f_out[k, j, i] = f[k, j, i]


def apply_moving_wall_fast(f, wall_mask, wall_velocity, rho_wall=1.0):
    """
    Fast moving wall bounce-back using Numba.
    """
    f_out = np.zeros_like(f)
    ux_wall, uy_wall = wall_velocity
    ex = EX.astype(np.float64)
    ey = EY.astype(np.float64)
    apply_moving_wall_numba(f, f_out, wall_mask, ux_wall, uy_wall, rho_wall,
                           ex, ey, W, CS2, OPPOSITE)
    return f_out


def zou_he_velocity_inlet_left(f, rho, ux_in, uy_in=0.0):
    """
    Zou-He velocity boundary condition for left inlet (x=0).
    
    Specifies velocity at inlet, computes unknown distributions.
    
    Parameters
    ----------
    f : ndarray
        Distribution functions, shape (Q, ny, nx)
    rho : ndarray
        Density field, shape (ny, nx)
    ux_in : float or ndarray
        Inlet x-velocity
    uy_in : float or ndarray
        Inlet y-velocity (default 0)
    
    Returns
    -------
    f : ndarray
        Distribution with inlet BC applied
    """
    f_new = f.copy()
    ny, nx = rho.shape
    
    # At left boundary (x=0), unknown distributions are: 1, 5, 8
    # Known distributions: 0, 2, 3, 4, 6, 7
    
    # Compute density from known distributions
    # rho = (f0 + f2 + f4 + 2*(f3 + f6 + f7)) / (1 - ux)
    rho_in = (f[0, :, 0] + f[2, :, 0] + f[4, :, 0] + 
              2.0 * (f[3, :, 0] + f[6, :, 0] + f[7, :, 0])) / (1.0 - ux_in)
    
    # Compute unknown distributions
    # f1 = f3 + (2/3) * rho * ux
    f_new[1, :, 0] = f[3, :, 0] + (2.0/3.0) * rho_in * ux_in
    
    # f5 = f7 + (1/6) * rho * ux + 0.5 * (f4 - f2) + 0.5 * rho * uy
    f_new[5, :, 0] = (f[7, :, 0] + (1.0/6.0) * rho_in * ux_in 
                     + 0.5 * (f[4, :, 0] - f[2, :, 0]) + 0.5 * rho_in * uy_in)
    
    # f8 = f6 + (1/6) * rho * ux - 0.5 * (f4 - f2) - 0.5 * rho * uy
    f_new[8, :, 0] = (f[6, :, 0] + (1.0/6.0) * rho_in * ux_in 
                     - 0.5 * (f[4, :, 0] - f[2, :, 0]) - 0.5 * rho_in * uy_in)
    
    return f_new


def zou_he_velocity_inlet_right(f, rho, ux_in, uy_in=0.0):
    """
    Zou-He velocity boundary condition for right outlet (x=nx-1).
    
    Parameters
    ----------
    f : ndarray
        Distribution functions, shape (Q, ny, nx)
    rho : ndarray
        Density field, shape (ny, nx)
    ux_in : float or ndarray
        Outlet x-velocity (typically negative or computed)
    uy_in : float or ndarray
        Outlet y-velocity (default 0)
    
    Returns
    -------
    f : ndarray
        Distribution with outlet BC applied
    """
    f_new = f.copy()
    ny, nx = rho.shape
    
    # At right boundary (x=nx-1), unknown distributions are: 3, 6, 7
    # Known distributions: 0, 1, 2, 4, 5, 8
    
    # Compute density
    rho_out = (f[0, :, -1] + f[2, :, -1] + f[4, :, -1] + 
               2.0 * (f[1, :, -1] + f[5, :, -1] + f[8, :, -1])) / (1.0 + ux_in)
    
    # Compute unknown distributions
    f_new[3, :, -1] = f[1, :, -1] - (2.0/3.0) * rho_out * ux_in
    
    f_new[6, :, -1] = (f[8, :, -1] - (1.0/6.0) * rho_out * ux_in 
                      + 0.5 * (f[4, :, -1] - f[2, :, -1]) - 0.5 * rho_out * uy_in)
    
    f_new[7, :, -1] = (f[5, :, -1] - (1.0/6.0) * rho_out * ux_in 
                      - 0.5 * (f[4, :, -1] - f[2, :, -1]) + 0.5 * rho_out * uy_in)
    
    return f_new


def zou_he_pressure_outlet_right(f, rho_out=1.0):
    """
    Zou-He pressure (density) boundary condition for right outlet.
    
    Specifies density at outlet, velocity is extrapolated.
    
    Parameters
    ----------
    f : ndarray
        Distribution functions, shape (Q, ny, nx)
    rho_out : float
        Outlet density (default 1.0)
    
    Returns
    -------
    f : ndarray
        Distribution with pressure outlet BC applied
    """
    f_new = f.copy()
    
    # Compute velocity from known distributions
    # ux = 1 - (f0 + f2 + f4 + 2*(f1 + f5 + f8)) / rho
    ux_out = 1.0 - (f[0, :, -1] + f[2, :, -1] + f[4, :, -1] + 
                   2.0 * (f[1, :, -1] + f[5, :, -1] + f[8, :, -1])) / rho_out
    
    # Assume uy = 0 at outlet
    uy_out = 0.0
    
    # Compute unknown distributions
    f_new[3, :, -1] = f[1, :, -1] - (2.0/3.0) * rho_out * ux_out
    
    f_new[6, :, -1] = (f[8, :, -1] - (1.0/6.0) * rho_out * ux_out 
                      + 0.5 * (f[4, :, -1] - f[2, :, -1]))
    
    f_new[7, :, -1] = (f[5, :, -1] - (1.0/6.0) * rho_out * ux_out 
                      - 0.5 * (f[4, :, -1] - f[2, :, -1]))
    
    return f_new


def zou_he_velocity_inlet_bottom(f, rho, ux_in=0.0, uy_in=0.0):
    """
    Zou-He velocity boundary condition for bottom inlet (y=0).
    
    Parameters
    ----------
    f : ndarray
        Distribution functions, shape (Q, ny, nx)
    rho : ndarray
        Density field, shape (ny, nx)
    ux_in : float or ndarray
        Inlet x-velocity (default 0)
    uy_in : float or ndarray
        Inlet y-velocity
    
    Returns
    -------
    f : ndarray
        Distribution with bottom inlet BC applied
    """
    f_new = f.copy()
    
    # At bottom boundary (y=0), unknown distributions are: 2, 5, 6
    # Known distributions: 0, 1, 3, 4, 7, 8
    
    # Compute density
    rho_in = (f[0, 0, :] + f[1, 0, :] + f[3, 0, :] + 
              2.0 * (f[4, 0, :] + f[7, 0, :] + f[8, 0, :])) / (1.0 - uy_in)
    
    # Compute unknown distributions
    f_new[2, 0, :] = f[4, 0, :] + (2.0/3.0) * rho_in * uy_in
    
    f_new[5, 0, :] = (f[7, 0, :] + (1.0/6.0) * rho_in * uy_in 
                     + 0.5 * (f[3, 0, :] - f[1, 0, :]) + 0.5 * rho_in * ux_in)
    
    f_new[6, 0, :] = (f[8, 0, :] + (1.0/6.0) * rho_in * uy_in 
                     - 0.5 * (f[3, 0, :] - f[1, 0, :]) - 0.5 * rho_in * ux_in)
    
    return f_new


def zou_he_velocity_inlet_top(f, rho, ux_in=0.0, uy_in=0.0):
    """
    Zou-He velocity boundary condition for top boundary (y=ny-1).
    
    Parameters
    ----------
    f : ndarray
        Distribution functions, shape (Q, ny, nx)
    rho : ndarray
        Density field, shape (ny, nx)
    ux_in : float or ndarray
        Boundary x-velocity (default 0)
    uy_in : float or ndarray
        Boundary y-velocity
    
    Returns
    -------
    f : ndarray
        Distribution with top BC applied
    """
    f_new = f.copy()
    
    # At top boundary (y=ny-1), unknown distributions are: 4, 7, 8
    # Known distributions: 0, 1, 2, 3, 5, 6
    
    # Compute density
    rho_in = (f[0, -1, :] + f[1, -1, :] + f[3, -1, :] + 
              2.0 * (f[2, -1, :] + f[5, -1, :] + f[6, -1, :])) / (1.0 + uy_in)
    
    # Compute unknown distributions
    f_new[4, -1, :] = f[2, -1, :] - (2.0/3.0) * rho_in * uy_in
    
    f_new[7, -1, :] = (f[5, -1, :] - (1.0/6.0) * rho_in * uy_in 
                      - 0.5 * (f[3, -1, :] - f[1, -1, :]) + 0.5 * rho_in * ux_in)
    
    f_new[8, -1, :] = (f[6, -1, :] - (1.0/6.0) * rho_in * uy_in 
                      + 0.5 * (f[3, -1, :] - f[1, -1, :]) - 0.5 * rho_in * ux_in)
    
    return f_new


def create_cylinder_mask(nx, ny, cx, cy, radius):
    """
    Create a solid mask for a circular cylinder.
    
    Parameters
    ----------
    nx, ny : int
        Grid dimensions
    cx, cy : float
        Cylinder center coordinates
    radius : float
        Cylinder radius
    
    Returns
    -------
    mask : ndarray
        Boolean mask (True for solid), shape (ny, nx)
    """
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)
    
    distance = np.sqrt((X - cx)**2 + (Y - cy)**2)
    mask = distance <= radius
    
    return mask


def create_channel_walls(nx, ny):
    """
    Create solid masks for horizontal channel walls.
    
    Parameters
    ----------
    nx, ny : int
        Grid dimensions
    
    Returns
    -------
    wall_mask : ndarray
        Boolean mask for walls (top and bottom rows)
    """
    mask = np.zeros((ny, nx), dtype=bool)
    mask[0, :] = True   # Bottom wall
    mask[-1, :] = True  # Top wall
    return mask


def create_cavity_walls(nx, ny):
    """
    Create solid masks for lid-driven cavity (3 walls, open top).
    
    The cavity has:
    - Bottom wall at y=0 (solid, no-slip)
    - Left wall at x=0 (solid, no-slip)
    - Right wall at x=nx-1 (solid, no-slip)
    - Top at y=ny-1 is the moving lid (handled separately)
    
    Parameters
    ----------
    nx, ny : int
        Grid dimensions
    
    Returns
    -------
    wall_mask : ndarray
        Boolean mask for solid walls (bottom, left, right)
    lid_mask : ndarray
        Boolean mask for moving lid (top)
    """
    wall_mask = np.zeros((ny, nx), dtype=bool)
    wall_mask[0, :] = True    # Bottom wall
    wall_mask[:, 0] = True    # Left wall
    wall_mask[:, -1] = True   # Right wall
    # Note: corners are already included in walls
    
    lid_mask = np.zeros((ny, nx), dtype=bool)
    lid_mask[-1, 1:-1] = True  # Top lid (excluding corners which are walls)
    
    return wall_mask, lid_mask


def apply_equilibrium_boundary(f, boundary_mask, rho, ux, uy):
    """
    Apply equilibrium boundary condition (open boundary).
    
    Sets distributions to equilibrium at boundary nodes.
    Useful for open boundaries where flow should pass through.
    
    Parameters
    ----------
    f : ndarray
        Distribution functions, shape (Q, ny, nx)
    boundary_mask : ndarray
        Boolean mask for boundary nodes
    rho : float or ndarray
        Boundary density
    ux, uy : float or ndarray
        Boundary velocity
    
    Returns
    -------
    f : ndarray
        Distribution with equilibrium BC applied
    """
    f_new = f.copy()
    
    # Get boundary indices
    j_idx, i_idx = np.where(boundary_mask)
    
    for idx in range(len(j_idx)):
        j, i = j_idx[idx], i_idx[idx]
        
        if isinstance(rho, np.ndarray):
            rho_bc = rho[j, i]
        else:
            rho_bc = rho
            
        if isinstance(ux, np.ndarray):
            ux_bc = ux[j, i]
        else:
            ux_bc = ux
            
        if isinstance(uy, np.ndarray):
            uy_bc = uy[j, i]
        else:
            uy_bc = uy
        
        f_eq = equilibrium_single_site(rho_bc, ux_bc, uy_bc)
        f_new[:, j, i] = f_eq
    
    return f_new


class BoundaryConditions:
    """
    Manager class for boundary conditions.
    
    Provides a unified interface for applying multiple boundary conditions.
    """
    
    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny
        self.boundaries = []
    
    def add_bounce_back(self, mask):
        """Add bounce-back boundary."""
        self.boundaries.append(('bounce_back', mask))
    
    def add_moving_wall(self, mask, velocity, rho=1.0):
        """Add moving wall boundary."""
        self.boundaries.append(('moving_wall', mask, velocity, rho))
    
    def add_velocity_inlet_left(self, ux, uy=0.0):
        """Add velocity inlet on left boundary."""
        self.boundaries.append(('velocity_inlet_left', ux, uy))
    
    def add_pressure_outlet_right(self, rho=1.0):
        """Add pressure outlet on right boundary."""
        self.boundaries.append(('pressure_outlet_right', rho))
    
    def apply(self, f, rho=None):
        """
        Apply all boundary conditions.
        
        Parameters
        ----------
        f : ndarray
            Distribution functions
        rho : ndarray, optional
            Density field (computed if needed)
        
        Returns
        -------
        f : ndarray
            Distribution with all BCs applied
        """
        for bc in self.boundaries:
            bc_type = bc[0]
            
            if bc_type == 'bounce_back':
                mask = bc[1]
                f = apply_bounce_back(f, mask)
            
            elif bc_type == 'moving_wall':
                mask, velocity, rho_wall = bc[1], bc[2], bc[3]
                f = apply_moving_wall_bounce_back(f, mask, velocity, rho_wall)
            
            elif bc_type == 'velocity_inlet_left':
                ux, uy = bc[1], bc[2]
                if rho is None:
                    from .observables import compute_density
                    rho = compute_density(f)
                f = zou_he_velocity_inlet_left(f, rho, ux, uy)
            
            elif bc_type == 'pressure_outlet_right':
                rho_out = bc[1]
                f = zou_he_pressure_outlet_right(f, rho_out)
        
        return f