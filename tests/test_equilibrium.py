"""
Tests for equilibrium distribution functions.

Validates mass and momentum conservation, and physical consistency.
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.lattice import EX, EY, W, CS2, Q
from src.equilibrium import (
    compute_equilibrium,
    compute_equilibrium_fast,
    equilibrium_single_site
)
from src.observables import compute_density, compute_velocity


class TestEquilibriumSingleSite:
    """Test equilibrium distribution at a single lattice site."""
    
    def test_mass_conservation_rest(self):
        """Verify sum of f_eq equals rho for rest fluid."""
        rho = 1.0
        ux, uy = 0.0, 0.0
        
        f_eq = equilibrium_single_site(rho, ux, uy)
        
        assert np.isclose(np.sum(f_eq), rho, rtol=1e-14)
    
    def test_mass_conservation_moving(self):
        """Verify sum of f_eq equals rho for moving fluid."""
        rho = 1.5
        ux, uy = 0.1, -0.05
        
        f_eq = equilibrium_single_site(rho, ux, uy)
        
        assert np.isclose(np.sum(f_eq), rho, rtol=1e-14)
    
    def test_momentum_conservation_rest(self):
        """Verify momentum of f_eq equals rho*u for rest fluid."""
        rho = 1.0
        ux, uy = 0.0, 0.0
        
        f_eq = equilibrium_single_site(rho, ux, uy)
        
        # Compute momentum: sum(f_eq * e_i)
        mom_x = np.sum(f_eq * EX)
        mom_y = np.sum(f_eq * EY)
        
        assert np.isclose(mom_x, rho * ux, atol=1e-14)
        assert np.isclose(mom_y, rho * uy, atol=1e-14)
    
    def test_momentum_conservation_moving(self):
        """Verify momentum of f_eq equals rho*u for moving fluid."""
        rho = 1.2
        ux, uy = 0.15, 0.08
        
        f_eq = equilibrium_single_site(rho, ux, uy)
        
        mom_x = np.sum(f_eq * EX)
        mom_y = np.sum(f_eq * EY)
        
        assert np.isclose(mom_x, rho * ux, rtol=1e-12)
        assert np.isclose(mom_y, rho * uy, rtol=1e-12)
    
    def test_positivity_low_velocity(self):
        """Verify all f_eq are positive for low Mach number."""
        rho = 1.0
        # Low Mach number (|u| << c_s)
        ux, uy = 0.05, 0.03
        
        f_eq = equilibrium_single_site(rho, ux, uy)
        
        assert np.all(f_eq > 0), f"Negative equilibrium values: {f_eq}"
    
    def test_symmetry_at_rest(self):
        """Verify equilibrium is symmetric for rest fluid."""
        rho = 1.0
        ux, uy = 0.0, 0.0
        
        f_eq = equilibrium_single_site(rho, ux, uy)
        
        # Rest particle should have weight w_0
        assert np.isclose(f_eq[0], W[0] * rho)
        
        # Cardinal directions should be equal
        assert np.isclose(f_eq[1], f_eq[2])
        assert np.isclose(f_eq[1], f_eq[3])
        assert np.isclose(f_eq[1], f_eq[4])
        
        # Diagonal directions should be equal
        assert np.isclose(f_eq[5], f_eq[6])
        assert np.isclose(f_eq[5], f_eq[7])
        assert np.isclose(f_eq[5], f_eq[8])


class TestEquilibriumField:
    """Test equilibrium distribution for entire field."""
    
    @pytest.fixture
    def uniform_field(self):
        """Create uniform density and velocity field."""
        nx, ny = 32, 32
        rho = np.ones((ny, nx), dtype=np.float64)
        ux = np.zeros((ny, nx), dtype=np.float64)
        uy = np.zeros((ny, nx), dtype=np.float64)
        return rho, ux, uy
    
    @pytest.fixture
    def varying_field(self):
        """Create spatially varying field."""
        nx, ny = 32, 32
        x = np.arange(nx)
        y = np.arange(ny)
        X, Y = np.meshgrid(x, y)
        
        rho = 1.0 + 0.1 * np.sin(2 * np.pi * X / nx)
        ux = 0.1 * np.cos(2 * np.pi * Y / ny)
        uy = 0.05 * np.sin(2 * np.pi * X / nx)
        
        return rho, ux, uy
    
    def test_mass_conservation_uniform(self, uniform_field):
        """Verify mass conservation for uniform field."""
        rho, ux, uy = uniform_field
        
        f_eq = compute_equilibrium(rho, ux, uy)
        rho_computed = compute_density(f_eq)
        
        np.testing.assert_allclose(rho_computed, rho, rtol=1e-14)
    
    def test_mass_conservation_varying(self, varying_field):
        """Verify mass conservation for varying field."""
        rho, ux, uy = varying_field
        
        f_eq = compute_equilibrium(rho, ux, uy)
        rho_computed = compute_density(f_eq)
        
        np.testing.assert_allclose(rho_computed, rho, rtol=1e-14)
    
    def test_momentum_conservation_varying(self, varying_field):
        """Verify momentum conservation for varying field."""
        rho, ux, uy = varying_field
        
        f_eq = compute_equilibrium(rho, ux, uy)
        ux_computed, uy_computed = compute_velocity(f_eq, rho)
        
        # Use atol for near-zero values
        np.testing.assert_allclose(ux_computed, ux, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(uy_computed, uy, rtol=1e-12, atol=1e-15)
    
    def test_fast_equals_standard(self, varying_field):
        """Verify Numba implementation matches standard."""
        rho, ux, uy = varying_field
        
        f_eq_std = compute_equilibrium(rho, ux, uy)
        f_eq_fast = compute_equilibrium_fast(rho, ux, uy)
        
        np.testing.assert_allclose(f_eq_fast, f_eq_std, rtol=1e-14)
    
    def test_output_shape(self, uniform_field):
        """Verify output shape is correct."""
        rho, ux, uy = uniform_field
        ny, nx = rho.shape
        
        f_eq = compute_equilibrium(rho, ux, uy)
        
        assert f_eq.shape == (Q, ny, nx)


class TestEquilibriumStressTensor:
    """Test second-order moment (stress tensor) properties."""
    
    def test_stress_tensor_isotropy_at_rest(self):
        """Verify stress tensor is isotropic for rest fluid."""
        rho = 1.0
        ux, uy = 0.0, 0.0
        
        f_eq = equilibrium_single_site(rho, ux, uy)
        
        # Compute second-order moments
        pi_xx = np.sum(f_eq * EX * EX)
        pi_yy = np.sum(f_eq * EY * EY)
        pi_xy = np.sum(f_eq * EX * EY)
        
        # For rest fluid: Pi_xx = Pi_yy = rho * c_s^2, Pi_xy = 0
        expected_diag = rho * CS2
        
        assert np.isclose(pi_xx, expected_diag, rtol=1e-12)
        assert np.isclose(pi_yy, expected_diag, rtol=1e-12)
        assert np.isclose(pi_xy, 0.0, atol=1e-14)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])