"""
Lid-Driven Cavity Simulation

Classic benchmark case for incompressible flow solvers.

The lid-driven cavity is a square domain with:
- Three stationary walls (no-slip)
- One moving wall (lid) at constant velocity

Reference data from Ghia et al. (1982) "High-Re Solutions for 
Incompressible Flow Using the Navier-Stokes Equations and a 
Multigrid Method", Journal of Computational Physics, 48, 387-411.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.lattice import EX, EY, W, CS2, Q
from src.equilibrium import compute_equilibrium_fast
from src.observables import compute_macroscopic_fast, compute_vorticity
from src.collision import bgk_collision_fast, viscosity_from_tau, tau_from_viscosity
from src.streaming import stream_periodic_fast
from src.boundary import (
    apply_bounce_back, 
    apply_moving_wall_bounce_back,
    create_cavity_walls
)


# Ghia et al. (1982) reference data for centerline velocities
GHIA_DATA = {
    100: {
        # Vertical centerline: u_x vs y
        'y': np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719, 
                       0.2813, 0.4531, 0.5000, 0.6172, 0.7344, 0.8516, 
                       0.9531, 0.9609, 0.9688, 0.9766, 1.0000]),
        'ux': np.array([0.00000, -0.03717, -0.04192, -0.04775, -0.06434, -0.10150,
                        -0.15662, -0.21090, -0.20581, -0.13641, 0.00332, 0.23151,
                        0.68717, 0.73722, 0.78871, 0.84123, 1.00000]),
        # Horizontal centerline: u_y vs x
        'x': np.array([0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563,
                       0.2266, 0.2344, 0.5000, 0.8047, 0.8594, 0.9063,
                       0.9453, 0.9531, 0.9609, 0.9688, 1.0000]),
        'uy': np.array([0.00000, 0.09233, 0.10091, 0.10890, 0.12317, 0.16077,
                        0.17507, 0.17527, 0.05454, -0.24533, -0.22445, -0.16914,
                        -0.10313, -0.08864, -0.07391, -0.05906, 0.00000])
    },
    400: {
        'y': np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719,
                       0.2813, 0.4531, 0.5000, 0.6172, 0.7344, 0.8516,
                       0.9531, 0.9609, 0.9688, 0.9766, 1.0000]),
        'ux': np.array([0.00000, -0.08186, -0.09266, -0.10338, -0.14612, -0.24299,
                        -0.32726, -0.17119, -0.11477, 0.02135, 0.16256, 0.29093,
                        0.55892, 0.61756, 0.68439, 0.75837, 1.00000]),
        'x': np.array([0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563,
                       0.2266, 0.2344, 0.5000, 0.8047, 0.8594, 0.9063,
                       0.9453, 0.9531, 0.9609, 0.9688, 1.0000]),
        'uy': np.array([0.00000, 0.18360, 0.19713, 0.20920, 0.22965, 0.28124,
                        0.30203, 0.30174, 0.05186, -0.38598, -0.44993, -0.23827,
                        -0.22847, -0.19254, -0.15663, -0.12146, 0.00000])
    },
    1000: {
        'y': np.array([0.0000, 0.0547, 0.0625, 0.0703, 0.1016, 0.1719,
                       0.2813, 0.4531, 0.5000, 0.6172, 0.7344, 0.8516,
                       0.9531, 0.9609, 0.9688, 0.9766, 1.0000]),
        'ux': np.array([0.00000, -0.18109, -0.20196, -0.22220, -0.29730, -0.38289,
                        -0.27805, -0.10648, -0.06080, 0.05702, 0.18719, 0.33304,
                        0.46604, 0.51117, 0.57492, 0.65928, 1.00000]),
        'x': np.array([0.0000, 0.0625, 0.0703, 0.0781, 0.0938, 0.1563,
                       0.2266, 0.2344, 0.5000, 0.8047, 0.8594, 0.9063,
                       0.9453, 0.9531, 0.9609, 0.9688, 1.0000]),
        'uy': np.array([0.00000, 0.27485, 0.29012, 0.30353, 0.32627, 0.37095,
                        0.33075, 0.32235, 0.02526, -0.31966, -0.42665, -0.51550,
                        -0.39188, -0.33714, -0.27669, -0.21388, 0.00000])
    }
}


class LidDrivenCavitySolver:
    """
    Solver for lid-driven cavity flow.
    
    Parameters
    ----------
    n : int
        Grid size (n x n cavity)
    re : float
        Reynolds number (Re = U_lid * L / nu)
    u_lid : float
        Lid velocity (default 0.05 in lattice units for stability)
    """
    
    def __init__(self, n, re, u_lid=0.05):
        self.n = n
        self.nx = n
        self.ny = n
        self.re = re
        self.u_lid = u_lid
        
        # Compute viscosity and tau from Reynolds number
        # Re = U * L / nu => nu = U * L / Re
        # L = n - 2 (effective cavity size)
        self.L = n - 2
        self.nu = u_lid * self.L / re
        self.tau = tau_from_viscosity(self.nu)
        
        # Check stability
        if self.tau <= 0.5:
            raise ValueError(
                f"Unstable: tau = {self.tau:.4f} <= 0.5. "
                f"Reduce Re or increase grid size."
            )
        if self.tau < 0.52:
            print(f"Warning: tau = {self.tau:.4f} is very close to stability limit")
        
        # Initialize fields
        self.rho = np.ones((n, n), dtype=np.float64)
        self.ux = np.zeros((n, n), dtype=np.float64)
        self.uy = np.zeros((n, n), dtype=np.float64)
        
        # Initialize distribution at equilibrium
        self.f = compute_equilibrium_fast(self.rho, self.ux, self.uy)
        
        # Create boundary masks
        self.wall_mask, self.lid_mask = create_cavity_walls(n, n)
        
        # Statistics
        self.step_count = 0
    
    def step(self):
        """Perform one LBM timestep."""
        # Collision
        self.rho, self.ux, self.uy = compute_macroscopic_fast(self.f)
        f_eq = compute_equilibrium_fast(self.rho, self.ux, self.uy)
        self.f = bgk_collision_fast(self.f, f_eq, self.tau)
        
        # Streaming
        self.f = stream_periodic_fast(self.f)
        
        # Boundary conditions
        # Stationary walls (bounce-back)
        self.f = apply_bounce_back(self.f, self.wall_mask)
        
        # Moving lid
        self.f = apply_moving_wall_bounce_back(
            self.f, self.lid_mask, (self.u_lid, 0.0), rho_wall=1.0
        )
        
        # Update macroscopic fields
        self.rho, self.ux, self.uy = compute_macroscopic_fast(self.f)
        
        self.step_count += 1
    
    def run(self, num_steps, check_interval=1000, tolerance=1e-6, verbose=True):
        """
        Run simulation until steady state or max steps.
        
        Parameters
        ----------
        num_steps : int
            Maximum number of timesteps
        check_interval : int
            Steps between convergence checks
        tolerance : float
            Convergence tolerance
        verbose : bool
            Print progress
        
        Returns
        -------
        converged : bool
            Whether simulation converged
        """
        ux_old = self.ux.copy()
        uy_old = self.uy.copy()
        
        # Don't check convergence in first few intervals
        min_steps_before_convergence = check_interval * 3
        
        for step in range(num_steps):
            self.step()
            
            if (step + 1) % check_interval == 0:
                # Check convergence (L2 norm of velocity change)
                du = np.sqrt((self.ux - ux_old)**2 + (self.uy - uy_old)**2)
                
                max_change = np.max(du)
                
                ux_old = self.ux.copy()
                uy_old = self.uy.copy()
                
                if verbose:
                    print(f"Step {step + 1}: max velocity change = {max_change:.2e}")
                
                if step + 1 >= min_steps_before_convergence and max_change < tolerance:
                    if verbose:
                        print(f"Converged at step {step + 1}")
                    return True
        
        if verbose:
            print(f"Did not converge after {num_steps} steps")
        return False
    
    def get_centerline_profiles(self):
        """
        Get velocity profiles along centerlines for comparison with Ghia.
        
        Returns
        -------
        y_norm : ndarray
            Normalized y-coordinates (0 to 1)
        ux_centerline : ndarray
            u_x along vertical centerline, normalized by u_lid
        x_norm : ndarray
            Normalized x-coordinates (0 to 1)
        uy_centerline : ndarray
            u_y along horizontal centerline, normalized by u_lid
        """
        # Vertical centerline (x = n/2)
        x_center = self.n // 2
        ux_centerline = self.ux[:, x_center] / self.u_lid
        y_norm = np.linspace(0, 1, self.n)
        
        # Horizontal centerline (y = n/2)
        y_center = self.n // 2
        uy_centerline = self.uy[y_center, :] / self.u_lid
        x_norm = np.linspace(0, 1, self.n)
        
        return y_norm, ux_centerline, x_norm, uy_centerline
    
    def compare_with_ghia(self):
        """
        Compare with Ghia et al. reference data.
        
        Returns
        -------
        ux_error : float
            RMS error for u_x profile
        uy_error : float
            RMS error for u_y profile
        """
        if self.re not in GHIA_DATA:
            print(f"No Ghia data for Re = {self.re}")
            return None, None
        
        ghia = GHIA_DATA[self.re]
        y_norm, ux_profile, x_norm, uy_profile = self.get_centerline_profiles()
        
        # Interpolate LBM results to Ghia locations
        ux_interp = np.interp(ghia['y'], y_norm, ux_profile)
        uy_interp = np.interp(ghia['x'], x_norm, uy_profile)
        
        # RMS errors
        ux_error = np.sqrt(np.mean((ux_interp - ghia['ux'])**2))
        uy_error = np.sqrt(np.mean((uy_interp - ghia['uy'])**2))
        
        return ux_error, uy_error
    
    def plot_results(self, save_path=None):
        """
        Plot velocity field and comparison with Ghia data.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Velocity magnitude
        u_mag = np.sqrt(self.ux**2 + self.uy**2)
        im1 = axes[0, 0].imshow(u_mag / self.u_lid, origin='lower', 
                                 cmap='viridis', aspect='equal')
        axes[0, 0].set_title('Velocity Magnitude')
        plt.colorbar(im1, ax=axes[0, 0], label='|u|/U_lid')
        
        # Vorticity
        vorticity = compute_vorticity(self.ux, self.uy)
        vmax = np.percentile(np.abs(vorticity), 95)
        im2 = axes[0, 1].imshow(vorticity, origin='lower', 
                                 cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
        axes[0, 1].set_title('Vorticity')
        plt.colorbar(im2, ax=axes[0, 1], label='ω')
        
        # Centerline profiles
        y_norm, ux_profile, x_norm, uy_profile = self.get_centerline_profiles()
        
        # u_x along vertical centerline
        axes[1, 0].plot(ux_profile, y_norm, 'b-', linewidth=2, label='LBM')
        if self.re in GHIA_DATA:
            ghia = GHIA_DATA[self.re]
            axes[1, 0].plot(ghia['ux'], ghia['y'], 'ro', markersize=6, label='Ghia et al.')
        axes[1, 0].set_xlabel('$u_x / U_{lid}$')
        axes[1, 0].set_ylabel('$y / L$')
        axes[1, 0].set_title('Vertical Centerline')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # u_y along horizontal centerline
        axes[1, 1].plot(x_norm, uy_profile, 'b-', linewidth=2, label='LBM')
        if self.re in GHIA_DATA:
            axes[1, 1].plot(ghia['x'], ghia['uy'], 'ro', markersize=6, label='Ghia et al.')
        axes[1, 1].set_xlabel('$x / L$')
        axes[1, 1].set_ylabel('$u_y / U_{lid}$')
        axes[1, 1].set_title('Horizontal Centerline')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Lid-Driven Cavity, Re = {self.re}, Grid = {self.n}×{self.n}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved figure to {save_path}")
        
        plt.show()
        
        return fig


def run_lid_driven_cavity(n=129, re=100, max_steps=100000, verbose=True):
    """
    Run lid-driven cavity simulation.
    
    Parameters
    ----------
    n : int
        Grid size
    re : float
        Reynolds number
    max_steps : int
        Maximum simulation steps
    verbose : bool
        Print progress
    
    Returns
    -------
    solver : LidDrivenCavitySolver
        Solver object with results
    """
    if verbose:
        print("Lid-Driven Cavity Simulation")
        print("=" * 50)
        print(f"Grid: {n} x {n}")
        print(f"Reynolds number: {re}")
    
    solver = LidDrivenCavitySolver(n, re)
    
    if verbose:
        print(f"Tau: {solver.tau:.4f}")
        print(f"Viscosity: {solver.nu:.6f}")
        print()
    
    start = time.perf_counter()
    converged = solver.run(max_steps, check_interval=5000, tolerance=1e-7, verbose=verbose)
    elapsed = time.perf_counter() - start
    
    if verbose:
        print()
        print(f"Simulation time: {elapsed:.2f}s")
        print(f"Steps: {solver.step_count}")
        
        if re in GHIA_DATA:
            ux_err, uy_err = solver.compare_with_ghia()
            print(f"RMS Error vs Ghia: ux = {ux_err:.4f}, uy = {uy_err:.4f}")
    
    return solver


if __name__ == "__main__":
    # Run simulation at Re = 100
    solver = run_lid_driven_cavity(n=129, re=100, max_steps=100000)
    
    # Plot results
    solver.plot_results(save_path='results/validation/lid_driven_re100.png')
    
    # Also run at Re = 400 if time permits
    print("\n")
    solver_400 = run_lid_driven_cavity(n=129, re=400, max_steps=200000)
    solver_400.plot_results(save_path='results/validation/lid_driven_re400.png')