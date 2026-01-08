"""
Poiseuille Flow Simulation

Pressure-driven channel flow for validation against analytical solution.

The analytical solution for Poiseuille flow is:
    u_x(y) = (G * H^2) / (8 * nu) * [1 - (2y/H - 1)^2]

where:
    - G is the pressure gradient (body force)
    - H is the channel height
    - nu is the kinematic viscosity
    - y is the vertical coordinate (0 to H)

Maximum velocity occurs at the centerline:
    u_max = G * H^2 / (8 * nu)
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.lattice import EX, EY, W, CS2, Q
from src.equilibrium import compute_equilibrium, compute_equilibrium_fast
from src.observables import compute_macroscopic, compute_macroscopic_fast
from src.collision import bgk_collision_fast, viscosity_from_tau, tau_from_viscosity
from src.streaming import stream_periodic_fast
from src.boundary import apply_bounce_back, create_channel_walls


def analytical_poiseuille(y, H, G, nu):
    """
    Analytical solution for Poiseuille flow.
    
    Parameters
    ----------
    y : ndarray
        Vertical coordinates (0 to H)
    H : float
        Channel height
    G : float
        Pressure gradient (body force per unit mass)
    nu : float
        Kinematic viscosity
    
    Returns
    -------
    ux : ndarray
        Analytical x-velocity profile
    """
    # Normalized coordinate
    y_norm = 2.0 * y / H - 1.0  # Maps [0, H] to [-1, 1]
    
    # Parabolic profile
    u_max = G * H * H / (8.0 * nu)
    ux = u_max * (1.0 - y_norm * y_norm)
    
    return ux


def apply_body_force(f, rho, force_x, force_y=0.0):
    """
    Apply body force using Guo's forcing scheme.
    
    Parameters
    ----------
    f : ndarray
        Distribution functions, shape (Q, ny, nx)
    rho : ndarray
        Density field, shape (ny, nx)
    force_x : float
        Body force in x-direction
    force_y : float
        Body force in y-direction
    
    Returns
    -------
    f : ndarray
        Distribution with force applied
    """
    f_new = f.copy()
    
    for i in range(Q):
        # Force term: F_i = w_i * (e_i - u) · F / cs^2
        # Simplified for low velocity: F_i ≈ w_i * e_i · F / cs^2
        force_term = W[i] * (EX[i] * force_x + EY[i] * force_y) / CS2
        f_new[i] += force_term
    
    return f_new


class PoiseuilleSolver:
    """
    Solver for Poiseuille (channel) flow.
    
    Parameters
    ----------
    nx : int
        Channel length (x-direction)
    ny : int
        Channel height (y-direction), including walls
    tau : float
        Relaxation time
    body_force : float
        Driving force (pressure gradient equivalent)
    """
    
    def __init__(self, nx, ny, tau, body_force):
        self.nx = nx
        self.ny = ny
        self.tau = tau
        self.nu = viscosity_from_tau(tau)
        self.body_force = body_force
        
        # Channel height (excluding wall nodes)
        self.H = ny - 2  # Fluid region is from y=1 to y=ny-2
        
        # Initialize fields
        self.rho = np.ones((ny, nx), dtype=np.float64)
        self.ux = np.zeros((ny, nx), dtype=np.float64)
        self.uy = np.zeros((ny, nx), dtype=np.float64)
        
        # Initialize distribution at equilibrium
        self.f = compute_equilibrium_fast(self.rho, self.ux, self.uy)
        
        # Create wall mask (top and bottom rows are solid)
        self.wall_mask = create_channel_walls(nx, ny)
        
        # Statistics
        self.step_count = 0
    
    def get_analytical_profile(self):
        """
        Get analytical velocity profile for comparison.
        
        Note: In LBM with bounce-back, the no-slip wall is located halfway
        between the wall node and the first fluid node. This affects the
        effective channel height.
        
        Returns
        -------
        y : ndarray
            Y-coordinates of fluid nodes
        ux_analytical : ndarray
            Analytical velocity profile
        """
        # Y-coordinates (fluid region only, y=1 to y=ny-2)
        y = np.arange(1, self.ny - 1).astype(np.float64)
        
        # In bounce-back, the wall is at y=0.5 and y=ny-1.5
        # So the fluid extends from y=0.5 to y=ny-1.5
        # Effective channel height: H_eff = ny - 2
        H_eff = float(self.ny - 2)
        
        # Transform y to [0, H_eff] range
        # y=1 corresponds to y_channel=0.5 (half lattice spacing from wall)
        # y=ny-2 corresponds to y_channel=H_eff-0.5
        y_channel = y - 0.5
        
        # Analytical solution with corrected height
        ux_analytical = analytical_poiseuille(y_channel, H_eff, self.body_force, self.nu)
        
        return y, ux_analytical
    
    def get_numerical_profile(self, x_loc=None):
        """
        Get numerical velocity profile at specified x-location.
        
        Parameters
        ----------
        x_loc : int, optional
            X-location for profile. Default is center.
        
        Returns
        -------
        y : ndarray
            Y-coordinates
        ux_numerical : ndarray
            Numerical velocity profile
        """
        if x_loc is None:
            x_loc = self.nx // 2
        
        y = np.arange(1, self.ny - 1)
        ux_numerical = self.ux[1:-1, x_loc]
        
        return y, ux_numerical
    
    def step(self):
        """Perform one LBM timestep."""
        # Collision
        rho, ux, uy = compute_macroscopic_fast(self.f)
        f_eq = compute_equilibrium_fast(rho, ux, uy)
        self.f = bgk_collision_fast(self.f, f_eq, self.tau)
        
        # Apply body force
        self.f = apply_body_force(self.f, rho, self.body_force)
        
        # Streaming
        self.f = stream_periodic_fast(self.f)
        
        # Boundary conditions (bounce-back on walls)
        self.f = apply_bounce_back(self.f, self.wall_mask)
        
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
            Convergence tolerance for velocity change
        verbose : bool
            Print progress information
        
        Returns
        -------
        converged : bool
            Whether simulation converged
        """
        ux_old = self.ux.copy()
        
        for step in range(num_steps):
            self.step()
            
            if (step + 1) % check_interval == 0:
                # Check convergence
                ux_change = np.max(np.abs(self.ux - ux_old))
                ux_old = self.ux.copy()
                
                if verbose:
                    u_max = np.max(self.ux[1:-1, :])
                    print(f"Step {step + 1}: max(ux) = {u_max:.6f}, change = {ux_change:.2e}")
                
                if ux_change < tolerance:
                    if verbose:
                        print(f"Converged at step {step + 1}")
                    return True
        
        if verbose:
            print(f"Did not converge after {num_steps} steps")
        return False
    
    def compute_error(self):
        """
        Compute L2 error between numerical and analytical solution.
        
        Returns
        -------
        l2_error : float
            L2 norm of the error
        l2_relative : float
            Relative L2 error
        """
        y, ux_analytical = self.get_analytical_profile()
        _, ux_numerical = self.get_numerical_profile()
        
        # L2 error
        diff = ux_numerical - ux_analytical
        l2_error = np.sqrt(np.mean(diff**2))
        
        # Relative error
        l2_relative = l2_error / np.sqrt(np.mean(ux_analytical**2))
        
        return l2_error, l2_relative
    
    def plot_comparison(self, save_path=None):
        """
        Plot numerical vs analytical velocity profile.
        
        Parameters
        ----------
        save_path : str, optional
            Path to save figure
        """
        y_analytical, ux_analytical = self.get_analytical_profile()
        y_numerical, ux_numerical = self.get_numerical_profile()
        
        l2_error, l2_relative = self.compute_error()
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Velocity profile comparison
        ax1.plot(ux_analytical, y_analytical, 'b-', linewidth=2, label='Analytical')
        ax1.plot(ux_numerical, y_numerical, 'ro', markersize=4, label='LBM')
        ax1.set_xlabel('Velocity $u_x$')
        ax1.set_ylabel('Channel height $y$')
        ax1.set_title('Poiseuille Flow: Velocity Profile')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Error distribution
        error = ux_numerical - ux_analytical
        ax2.plot(error, y_numerical, 'k-', linewidth=1.5)
        ax2.axvline(x=0, color='gray', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Error $(u_{LBM} - u_{analytical})$')
        ax2.set_ylabel('Channel height $y$')
        ax2.set_title(f'Error Distribution (L2 relative = {l2_relative:.2e})')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved figure to {save_path}")
        
        plt.show()
        
        return fig


def run_poiseuille_simulation(nx=100, ny=52, tau=0.8, body_force=1e-5, 
                              max_steps=50000, verbose=True):
    """
    Run Poiseuille flow simulation and validate.
    
    Parameters
    ----------
    nx : int
        Channel length
    ny : int
        Channel height (including walls)
    tau : float
        Relaxation time
    body_force : float
        Driving force
    max_steps : int
        Maximum simulation steps
    verbose : bool
        Print progress
    
    Returns
    -------
    solver : PoiseuilleSolver
        Solver object with results
    """
    if verbose:
        print("Poiseuille Flow Simulation")
        print("=" * 50)
        print(f"Grid: {nx} x {ny}")
        print(f"Tau: {tau}, Viscosity: {viscosity_from_tau(tau):.6f}")
        print(f"Body force: {body_force:.2e}")
        print()
    
    solver = PoiseuilleSolver(nx, ny, tau, body_force)
    
    start = time.perf_counter()
    converged = solver.run(max_steps, check_interval=2000, tolerance=1e-8, verbose=verbose)
    elapsed = time.perf_counter() - start
    
    if verbose:
        print()
        print(f"Simulation time: {elapsed:.2f}s")
        print(f"Steps: {solver.step_count}")
        
        l2_error, l2_relative = solver.compute_error()
        print(f"L2 Error: {l2_error:.6e}")
        print(f"L2 Relative Error: {l2_relative:.4%}")
    
    return solver


def convergence_study(grid_sizes=None, tau=0.8, body_force=1e-5, max_steps=100000):
    """
    Perform grid convergence study.
    
    Parameters
    ----------
    grid_sizes : list of int
        List of ny values to test
    tau : float
        Relaxation time
    body_force : float
        Driving force
    max_steps : int
        Maximum steps per simulation
    
    Returns
    -------
    results : dict
        Grid sizes and corresponding errors
    """
    if grid_sizes is None:
        grid_sizes = [12, 22, 42, 82]
    
    print("Grid Convergence Study")
    print("=" * 50)
    
    results = {'ny': [], 'h': [], 'l2_error': [], 'l2_relative': []}
    
    for ny in grid_sizes:
        nx = ny * 2  # Aspect ratio 2:1
        print(f"\nGrid: {nx} x {ny}")
        
        solver = PoiseuilleSolver(nx, ny, tau, body_force)
        solver.run(max_steps, check_interval=5000, tolerance=1e-9, verbose=False)
        
        l2_error, l2_relative = solver.compute_error()
        h = 1.0 / (ny - 2)  # Grid spacing
        
        results['ny'].append(ny)
        results['h'].append(h)
        results['l2_error'].append(l2_error)
        results['l2_relative'].append(l2_relative)
        
        print(f"  h = {h:.4f}, L2 error = {l2_error:.6e}, relative = {l2_relative:.4%}")
    
    # Compute convergence order
    if len(grid_sizes) >= 2:
        h = np.array(results['h'])
        err = np.array(results['l2_error'])
        
        # Linear fit in log-log space
        log_h = np.log(h)
        log_err = np.log(err)
        slope, intercept = np.polyfit(log_h, log_err, 1)
        
        print(f"\nConvergence order: {slope:.2f}")
        print("(Expected: ~2 for second-order accuracy)")
    
    return results


if __name__ == "__main__":
    # Run single simulation
    solver = run_poiseuille_simulation(nx=100, ny=52, tau=0.8, body_force=1e-5)
    
    # Plot results
    solver.plot_comparison(save_path='results/validation/poiseuille_validation.png')
    
    # Run convergence study
    print("\n")
    results = convergence_study()