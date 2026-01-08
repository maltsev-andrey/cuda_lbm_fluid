"""
CPU Baseline Implementation

NumPy-based LBM solver for performance comparison.

This provides the reference implementation against which GPU performance
will be measured. Uses optimized NumPy operations where possible.
"""

import numpy as np
import time
from numba import njit, prange

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.lattice import EX, EY, W, CS2, CS4, Q, OPPOSITE
from src.equilibrium import compute_equilibrium, compute_equilibrium_fast
from src.observables import compute_macroscopic, compute_macroscopic_fast
from src.collision import bgk_collision, bgk_collision_fast, viscosity_from_tau
from src.streaming import stream_periodic, stream_periodic_fast


class CPULBMSolver:
    """
    CPU-based LBM solver using NumPy/Numba.
    
    Implements a complete 2D LBM solver with periodic boundaries.
    Suitable for validation and as a performance baseline.
    
    Parameters
    ----------
    nx : int
        Number of lattice points in x-direction
    ny : int
        Number of lattice points in y-direction
    tau : float
        Relaxation time (must be > 0.5)
    use_fast : bool
        Use Numba-accelerated functions (default True)
    
    Attributes
    ----------
    f : ndarray
        Distribution functions, shape (Q, ny, nx)
    rho : ndarray
        Density field, shape (ny, nx)
    ux, uy : ndarray
        Velocity fields, shape (ny, nx)
    """
    
    def __init__(self, nx, ny, tau, use_fast=True):
        self.nx = nx
        self.ny = ny
        self.tau = tau
        self.omega = 1.0 / tau
        self.viscosity = viscosity_from_tau(tau)
        self.use_fast = use_fast
        
        # Validate tau
        if tau <= 0.5:
            raise ValueError(f"tau must be > 0.5, got {tau}")
        
        # Initialize fields
        self.rho = np.ones((ny, nx), dtype=np.float64)
        self.ux = np.zeros((ny, nx), dtype=np.float64)
        self.uy = np.zeros((ny, nx), dtype=np.float64)
        
        # Initialize distribution at equilibrium
        if use_fast:
            self.f = compute_equilibrium_fast(self.rho, self.ux, self.uy)
        else:
            self.f = compute_equilibrium(self.rho, self.ux, self.uy)
        
        # Statistics
        self.step_count = 0
        self.total_time = 0.0
    
    def initialize_uniform(self, rho=1.0, ux=0.0, uy=0.0):
        """
        Initialize with uniform density and velocity.
        
        Parameters
        ----------
        rho : float
            Uniform density
        ux : float
            Uniform x-velocity
        uy : float
            Uniform y-velocity
        """
        self.rho[:] = rho
        self.ux[:] = ux
        self.uy[:] = uy
        
        if self.use_fast:
            self.f = compute_equilibrium_fast(self.rho, self.ux, self.uy)
        else:
            self.f = compute_equilibrium(self.rho, self.ux, self.uy)
        
        self.step_count = 0
        self.total_time = 0.0
    
    def initialize_from_fields(self, rho, ux, uy):
        """
        Initialize from given density and velocity fields.
        
        Parameters
        ----------
        rho : ndarray
            Density field, shape (ny, nx)
        ux : ndarray
            X-velocity field, shape (ny, nx)
        uy : ndarray
            Y-velocity field, shape (ny, nx)
        """
        self.rho = rho.copy()
        self.ux = ux.copy()
        self.uy = uy.copy()
        
        if self.use_fast:
            self.f = compute_equilibrium_fast(self.rho, self.ux, self.uy)
        else:
            self.f = compute_equilibrium(self.rho, self.ux, self.uy)
        
        self.step_count = 0
        self.total_time = 0.0
    
    def step(self):
        """
        Perform one LBM timestep (collision + streaming).
        
        Returns
        -------
        dt : float
            Time taken for this step (seconds)
        """
        start = time.perf_counter()
        
        if self.use_fast:
            # Compute macroscopic quantities
            self.rho, self.ux, self.uy = compute_macroscopic_fast(self.f)
            
            # Compute equilibrium
            f_eq = compute_equilibrium_fast(self.rho, self.ux, self.uy)
            
            # Collision
            self.f = bgk_collision_fast(self.f, f_eq, self.tau)
            
            # Streaming
            self.f = stream_periodic_fast(self.f)
        else:
            # Compute macroscopic quantities
            self.rho, self.ux, self.uy = compute_macroscopic(self.f)
            
            # Compute equilibrium
            f_eq = compute_equilibrium(self.rho, self.ux, self.uy)
            
            # Collision
            self.f = bgk_collision(self.f, f_eq, self.tau)
            
            # Streaming
            self.f = stream_periodic(self.f)
        
        dt = time.perf_counter() - start
        self.step_count += 1
        self.total_time += dt
        
        return dt
    
    def run(self, num_steps, verbose=True, report_interval=100):
        """
        Run simulation for specified number of steps.
        
        Parameters
        ----------
        num_steps : int
            Number of timesteps to run
        verbose : bool
            Print progress information
        report_interval : int
            Steps between progress reports
        
        Returns
        -------
        mlups : float
            Performance in Million Lattice Updates Per Second
        """
        start = time.perf_counter()
        
        for step in range(num_steps):
            self.step()
            
            if verbose and (step + 1) % report_interval == 0:
                elapsed = time.perf_counter() - start
                mlups = (step + 1) * self.nx * self.ny / elapsed / 1e6
                print(f"Step {step + 1}/{num_steps}, MLUPS: {mlups:.2f}")
        
        total = time.perf_counter() - start
        mlups = num_steps * self.nx * self.ny / total / 1e6
        
        if verbose:
            print(f"Completed {num_steps} steps in {total:.2f}s")
            print(f"Performance: {mlups:.2f} MLUPS")
        
        return mlups
    
    def get_velocity_magnitude(self):
        """Return velocity magnitude field."""
        return np.sqrt(self.ux**2 + self.uy**2)
    
    def get_vorticity(self):
        """Return vorticity field."""
        from src.observables import compute_vorticity
        return compute_vorticity(self.ux, self.uy)
    
    def get_total_mass(self):
        """Return total mass (should be conserved)."""
        return np.sum(self.f)
    
    def get_total_momentum(self):
        """Return total momentum (conserved for periodic boundaries)."""
        mom_x = np.sum(self.f * EX[:, None, None])
        mom_y = np.sum(self.f * EY[:, None, None])
        return mom_x, mom_y
    
    def get_kinetic_energy(self):
        """Return total kinetic energy."""
        return 0.5 * np.sum(self.rho * (self.ux**2 + self.uy**2))
    
    def get_enstrophy(self):
        """Return total enstrophy (integral of vorticity squared)."""
        vorticity = self.get_vorticity()
        return 0.5 * np.sum(vorticity**2)


def benchmark_cpu_solver(grid_sizes=None, num_steps=1000, warmup_steps=100):
    """
    Benchmark CPU solver across different grid sizes.
    
    Parameters
    ----------
    grid_sizes : list of tuples
        List of (nx, ny) grid sizes to test
    num_steps : int
        Number of steps for timing (after warmup)
    warmup_steps : int
        Number of warmup steps (for JIT compilation)
    
    Returns
    -------
    results : dict
        Dictionary with grid sizes as keys, MLUPS as values
    """
    if grid_sizes is None:
        grid_sizes = [
            (64, 64),
            (128, 128),
            (256, 256),
            (512, 512),
            (1024, 1024),
        ]
    
    results = {}
    tau = 0.8
    
    print("CPU LBM Solver Benchmark")
    print("=" * 50)
    print(f"Tau: {tau}, Steps: {num_steps}, Warmup: {warmup_steps}")
    print()
    
    for nx, ny in grid_sizes:
        print(f"Grid size: {nx} x {ny}")
        
        # Create solver
        solver = CPULBMSolver(nx, ny, tau, use_fast=True)
        
        # Warmup (JIT compilation)
        for _ in range(warmup_steps):
            solver.step()
        
        # Reset timing
        solver.step_count = 0
        solver.total_time = 0.0
        
        # Benchmark
        start = time.perf_counter()
        for _ in range(num_steps):
            solver.step()
        elapsed = time.perf_counter() - start
        
        mlups = num_steps * nx * ny / elapsed / 1e6
        results[(nx, ny)] = mlups
        
        print(f"  Time: {elapsed:.2f}s, MLUPS: {mlups:.2f}")
        print()
    
    return results


if __name__ == "__main__":
    # Run benchmark
    results = benchmark_cpu_solver()
    
    print("\nSummary")
    print("=" * 50)
    for (nx, ny), mlups in results.items():
        print(f"{nx:4d} x {ny:4d}: {mlups:8.2f} MLUPS")