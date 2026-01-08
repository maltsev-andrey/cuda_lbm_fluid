"""
Poiseuille Flow Validation Tests

Compares numerical results with analytical solution.

Note: Simple bounce-back boundary conditions introduce O(h) errors in
wall-normal velocity gradients, resulting in ~3-5% errors for typical
grid resolutions. This is expected behavior for the bounce-back scheme.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simulations.poiseuille_flow import PoiseuilleSolver, analytical_poiseuille


class TestPoiseuilleFlow:
    """Validate Poiseuille flow against analytical solution."""
    
    def test_velocity_profile(self):
        """Verify parabolic velocity profile develops."""
        # Use moderate grid for reasonable test time
        nx, ny = 60, 32
        tau = 0.8
        body_force = 1e-5
        
        solver = PoiseuilleSolver(nx, ny, tau, body_force)
        solver.run(20000, check_interval=5000, tolerance=1e-8, verbose=False)
        
        y, ux_numerical = solver.get_numerical_profile()
        y, ux_analytical = solver.get_analytical_profile()
        
        # Check that profile is approximately parabolic
        # With bounce-back, expect ~5-10% error
        l2_error, l2_relative = solver.compute_error()
        
        assert l2_relative < 0.10, f"Relative error {l2_relative:.2%} exceeds 10%"
    
    def test_convergence_order(self):
        """Verify solution improves with grid refinement."""
        tau = 0.8
        body_force = 1e-5
        
        errors = []
        grid_sizes = [22, 42]  # Small grids for fast testing
        
        for ny in grid_sizes:
            nx = ny * 2
            solver = PoiseuilleSolver(nx, ny, tau, body_force)
            solver.run(30000, check_interval=10000, tolerance=1e-9, verbose=False)
            
            _, l2_relative = solver.compute_error()
            errors.append(l2_relative)
        
        # Error should decrease with finer grid
        assert errors[1] < errors[0], "Error should decrease with grid refinement"
    
    def test_mass_conservation(self):
        """Verify mass is conserved during simulation."""
        nx, ny = 40, 22
        tau = 0.8
        body_force = 1e-5
        
        solver = PoiseuilleSolver(nx, ny, tau, body_force)
        
        mass_initial = solver.f.sum()
        solver.run(5000, verbose=False)
        mass_final = solver.f.sum()
        
        # Mass should be conserved to machine precision
        relative_change = abs(mass_final - mass_initial) / mass_initial
        assert relative_change < 1e-10, f"Mass changed by {relative_change:.2e}"
    
    def test_steady_state_convergence(self):
        """Verify simulation reaches steady state."""
        nx, ny = 40, 22
        tau = 0.8
        body_force = 1e-5
        
        solver = PoiseuilleSolver(nx, ny, tau, body_force)
        converged = solver.run(50000, check_interval=5000, tolerance=1e-8, verbose=False)
        
        assert converged, "Simulation should converge to steady state"
    
    def test_symmetry(self):
        """Verify velocity profile is symmetric about centerline."""
        nx, ny = 60, 32
        tau = 0.8
        body_force = 1e-5
        
        solver = PoiseuilleSolver(nx, ny, tau, body_force)
        solver.run(20000, verbose=False)
        
        # Get velocity profile
        _, ux = solver.get_numerical_profile()
        
        # Check symmetry
        n = len(ux)
        ux_top = ux[:n//2]
        ux_bottom = ux[n//2:][::-1]
        
        if len(ux_top) != len(ux_bottom):
            ux_bottom = ux[n//2+1:][::-1]
        
        # Profiles should be nearly identical
        max_asymmetry = np.max(np.abs(ux_top - ux_bottom[:len(ux_top)]))
        assert max_asymmetry < 1e-10, f"Profile asymmetry: {max_asymmetry:.2e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])