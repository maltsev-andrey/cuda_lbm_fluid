"""
Lid-Driven Cavity Validation Tests

Compares with Ghia et al. (1982) reference data.

Note: LBM accuracy depends on grid resolution. For 65x65 grid,
expect ~5-10% error compared to Ghia. Finer grids improve accuracy.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestLidDrivenCavity:
    """Validate lid-driven cavity against reference data."""
    
    def test_centerline_velocity_re100(self):
        """Compare centerline velocities at Re=100."""
        from simulations.lid_driven_cavity import LidDrivenCavitySolver, GHIA_DATA
        
        # Use moderate grid for reasonable test time
        n = 65
        solver = LidDrivenCavitySolver(n, re=100)
        solver.run(40000, check_interval=10000, tolerance=1e-7, verbose=False)
        
        # Compare with Ghia
        ux_err, uy_err = solver.compare_with_ghia()
        
        # For 65x65, expect ~10% error
        assert ux_err < 0.12, f"ux error {ux_err:.2%} exceeds 12%"
        assert uy_err < 0.10, f"uy error {uy_err:.2%} exceeds 10%"
    
    def test_circulation_direction(self):
        """Verify primary vortex rotates in correct direction."""
        from simulations.lid_driven_cavity import LidDrivenCavitySolver
        
        n = 33
        solver = LidDrivenCavitySolver(n, re=100)
        solver.run(20000, verbose=False)
        
        # Primary vortex should be clockwise (negative vorticity in center)
        # Lid moves right (+x), so ux > 0 at top, ux < 0 at bottom
        
        # Check lid is moving right
        assert np.mean(solver.ux[-2, n//4:3*n//4]) > 0, "Lid should move right"
        
        # Check return flow at bottom
        assert np.mean(solver.ux[1, n//4:3*n//4]) < 0, "Return flow should be leftward"
    
    def test_mass_conservation(self):
        """Verify mass is conserved during simulation."""
        from simulations.lid_driven_cavity import LidDrivenCavitySolver
        
        n = 33
        solver = LidDrivenCavitySolver(n, re=100)
        
        mass_initial = np.sum(solver.f)
        solver.run(5000, verbose=False)
        mass_final = np.sum(solver.f)
        
        relative_change = abs(mass_final - mass_initial) / mass_initial
        assert relative_change < 1e-10, f"Mass changed by {relative_change:.2e}"
    
    def test_convergence_to_steady_state(self):
        """Verify simulation reaches steady state."""
        from simulations.lid_driven_cavity import LidDrivenCavitySolver
        
        n = 33
        solver = LidDrivenCavitySolver(n, re=100)
        converged = solver.run(30000, check_interval=5000, tolerance=1e-6, verbose=False)
        
        assert converged, "Simulation should converge to steady state"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])