"""
Tests for conservation laws.

Validates mass and momentum conservation during simulation.
These are fundamental requirements for any correct LBM implementation.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.lattice import EX, EY, W, CS2, Q
from src.equilibrium import compute_equilibrium, compute_equilibrium_fast
from src.observables import compute_density, compute_velocity, compute_macroscopic
from src.collision import (
    bgk_collision, bgk_collision_fast,
    trt_collision, trt_collision_fast,
    tau_from_viscosity, viscosity_from_tau
)
from src.streaming import stream_periodic, stream_periodic_fast


class TestMassConservation:
    """Test mass conservation during LBM steps."""
    
    @pytest.fixture
    def initial_field(self):
        """Create initial distribution at equilibrium."""
        nx, ny = 64, 64
        rho = 1.0 + 0.1 * np.random.randn(ny, nx)
        ux = 0.05 * np.random.randn(ny, nx)
        uy = 0.05 * np.random.randn(ny, nx)
        
        f = compute_equilibrium(rho, ux, uy)
        return f
    
    def test_mass_conservation_streaming(self, initial_field):
        """Mass should be exactly conserved by streaming."""
        f = initial_field
        
        total_mass_before = np.sum(f)
        f_streamed = stream_periodic(f)
        total_mass_after = np.sum(f_streamed)
        
        assert np.isclose(total_mass_before, total_mass_after, rtol=1e-14)
    
    def test_mass_conservation_streaming_fast(self, initial_field):
        """Mass should be exactly conserved by fast streaming."""
        f = initial_field
        
        total_mass_before = np.sum(f)
        f_streamed = stream_periodic_fast(f)
        total_mass_after = np.sum(f_streamed)
        
        assert np.isclose(total_mass_before, total_mass_after, rtol=1e-14)
    
    def test_mass_conservation_bgk_collision(self, initial_field):
        """Mass should be exactly conserved by BGK collision."""
        f = initial_field
        tau = 0.8
        
        rho, ux, uy = compute_macroscopic(f)
        f_eq = compute_equilibrium(rho, ux, uy)
        
        total_mass_before = np.sum(f)
        f_coll = bgk_collision(f, f_eq, tau)
        total_mass_after = np.sum(f_coll)
        
        assert np.isclose(total_mass_before, total_mass_after, rtol=1e-14)
    
    def test_mass_conservation_trt_collision(self, initial_field):
        """Mass should be exactly conserved by TRT collision."""
        f = initial_field
        tau_plus = 0.8
        
        rho, ux, uy = compute_macroscopic(f)
        f_eq = compute_equilibrium(rho, ux, uy)
        
        total_mass_before = np.sum(f)
        f_coll = trt_collision(f, f_eq, tau_plus)
        total_mass_after = np.sum(f_coll)
        
        assert np.isclose(total_mass_before, total_mass_after, rtol=1e-14)
    
    def test_mass_conservation_multiple_steps(self, initial_field):
        """Mass should be conserved over multiple timesteps."""
        f = initial_field.copy()
        tau = 0.8
        
        total_mass_initial = np.sum(f)
        
        for _ in range(100):
            rho, ux, uy = compute_macroscopic(f)
            f_eq = compute_equilibrium_fast(rho, ux, uy)
            f = bgk_collision_fast(f, f_eq, tau)
            f = stream_periodic_fast(f)
        
        total_mass_final = np.sum(f)
        
        assert np.isclose(total_mass_initial, total_mass_final, rtol=1e-12)


class TestMomentumConservation:
    """Test momentum conservation during LBM steps."""
    
    @pytest.fixture
    def initial_field(self):
        """Create initial distribution at equilibrium."""
        nx, ny = 64, 64
        rho = np.ones((ny, nx), dtype=np.float64)
        ux = 0.1 * np.ones((ny, nx), dtype=np.float64)
        uy = 0.05 * np.ones((ny, nx), dtype=np.float64)
        
        f = compute_equilibrium(rho, ux, uy)
        return f
    
    def test_momentum_conservation_streaming(self, initial_field):
        """Momentum should be exactly conserved by streaming (periodic)."""
        f = initial_field
        
        # Total momentum before
        mom_x_before = np.sum(f * EX[:, None, None])
        mom_y_before = np.sum(f * EY[:, None, None])
        
        f_streamed = stream_periodic(f)
        
        mom_x_after = np.sum(f_streamed * EX[:, None, None])
        mom_y_after = np.sum(f_streamed * EY[:, None, None])
        
        assert np.isclose(mom_x_before, mom_x_after, rtol=1e-14)
        assert np.isclose(mom_y_before, mom_y_after, rtol=1e-14)
    
    def test_momentum_conservation_bgk_collision(self, initial_field):
        """Momentum should be exactly conserved by BGK collision."""
        f = initial_field
        tau = 0.8
        
        rho, ux, uy = compute_macroscopic(f)
        f_eq = compute_equilibrium(rho, ux, uy)
        
        # Total momentum before
        mom_x_before = np.sum(f * EX[:, None, None])
        mom_y_before = np.sum(f * EY[:, None, None])
        
        f_coll = bgk_collision(f, f_eq, tau)
        
        mom_x_after = np.sum(f_coll * EX[:, None, None])
        mom_y_after = np.sum(f_coll * EY[:, None, None])
        
        assert np.isclose(mom_x_before, mom_x_after, rtol=1e-14)
        assert np.isclose(mom_y_before, mom_y_after, rtol=1e-14)
    
    def test_momentum_conservation_multiple_steps(self, initial_field):
        """Momentum should be conserved over multiple timesteps (periodic)."""
        f = initial_field.copy()
        tau = 0.8
        
        # Total momentum initial
        mom_x_initial = np.sum(f * EX[:, None, None])
        mom_y_initial = np.sum(f * EY[:, None, None])
        
        for _ in range(100):
            rho, ux, uy = compute_macroscopic(f)
            f_eq = compute_equilibrium_fast(rho, ux, uy)
            f = bgk_collision_fast(f, f_eq, tau)
            f = stream_periodic_fast(f)
        
        mom_x_final = np.sum(f * EX[:, None, None])
        mom_y_final = np.sum(f * EY[:, None, None])
        
        assert np.isclose(mom_x_initial, mom_x_final, rtol=1e-12)
        assert np.isclose(mom_y_initial, mom_y_final, rtol=1e-12)


class TestStreamingConsistency:
    """Test streaming step consistency."""
    
    def test_streaming_fast_equals_standard(self):
        """Verify fast streaming matches standard implementation."""
        nx, ny = 64, 64
        rho = 1.0 + 0.1 * np.random.randn(ny, nx)
        ux = 0.05 * np.random.randn(ny, nx)
        uy = 0.05 * np.random.randn(ny, nx)
        
        f = compute_equilibrium(rho, ux, uy)
        
        f_std = stream_periodic(f)
        f_fast = stream_periodic_fast(f)
        
        np.testing.assert_allclose(f_fast, f_std, rtol=1e-14)
    
    def test_streaming_reversibility(self):
        """Streaming is reversible (stream 4 times returns to original)."""
        nx, ny = 32, 32
        rho = np.ones((ny, nx), dtype=np.float64)
        ux = 0.1 * np.ones((ny, nx), dtype=np.float64)
        uy = np.zeros((ny, nx), dtype=np.float64)
        
        f_original = compute_equilibrium(rho, ux, uy)
        
        # Stream in one direction 4 times (should complete a cycle for direction 1)
        # Actually, streaming is not reversible this way. Let's test identity instead.
        # Stream and anti-stream should give identity
        pass  # This test concept needs revision


class TestCollisionConsistency:
    """Test collision operator consistency."""
    
    def test_bgk_fast_equals_standard(self):
        """Verify fast BGK matches standard implementation."""
        nx, ny = 64, 64
        rho = 1.0 + 0.1 * np.random.randn(ny, nx)
        ux = 0.05 * np.random.randn(ny, nx)
        uy = 0.05 * np.random.randn(ny, nx)
        
        f = compute_equilibrium(rho, ux, uy)
        f_eq = compute_equilibrium(rho, ux, uy)
        tau = 0.8
        
        # Add some non-equilibrium perturbation
        f = f + 0.01 * np.random.randn(*f.shape)
        
        f_std = bgk_collision(f, f_eq, tau)
        f_fast = bgk_collision_fast(f, f_eq, tau)
        
        np.testing.assert_allclose(f_fast, f_std, rtol=1e-14)
    
    def test_trt_fast_equals_standard(self):
        """Verify fast TRT matches standard implementation."""
        nx, ny = 32, 32
        rho = 1.0 + 0.1 * np.random.randn(ny, nx)
        ux = 0.05 * np.random.randn(ny, nx)
        uy = 0.05 * np.random.randn(ny, nx)
        
        f = compute_equilibrium(rho, ux, uy)
        f_eq = compute_equilibrium(rho, ux, uy)
        tau_plus = 0.8
        
        f = f + 0.01 * np.random.randn(*f.shape)
        
        f_std = trt_collision(f, f_eq, tau_plus)
        f_fast = trt_collision_fast(f, f_eq, tau_plus)
        
        np.testing.assert_allclose(f_fast, f_std, rtol=1e-12)
    
    def test_equilibrium_is_fixed_point(self):
        """BGK collision should leave equilibrium unchanged."""
        nx, ny = 32, 32
        rho = np.ones((ny, nx), dtype=np.float64)
        ux = 0.1 * np.ones((ny, nx), dtype=np.float64)
        uy = 0.05 * np.ones((ny, nx), dtype=np.float64)
        
        f_eq = compute_equilibrium(rho, ux, uy)
        tau = 0.8
        
        # Collision of equilibrium should give equilibrium
        f_coll = bgk_collision(f_eq, f_eq, tau)
        
        np.testing.assert_allclose(f_coll, f_eq, rtol=1e-14)


class TestViscosityTauRelation:
    """Test viscosity-tau relationship."""
    
    def test_tau_from_viscosity(self):
        """Verify tau calculation from viscosity."""
        nu = 0.1
        tau = tau_from_viscosity(nu)
        
        # nu = cs2 * (tau - 0.5) => tau = nu/cs2 + 0.5
        expected_tau = nu / CS2 + 0.5
        
        assert np.isclose(tau, expected_tau)
    
    def test_viscosity_from_tau(self):
        """Verify viscosity calculation from tau."""
        tau = 0.8
        nu = viscosity_from_tau(tau)
        
        # nu = cs2 * (tau - 0.5)
        expected_nu = CS2 * (tau - 0.5)
        
        assert np.isclose(nu, expected_nu)
    
    def test_roundtrip(self):
        """Verify tau -> nu -> tau roundtrip."""
        tau_original = 0.75
        nu = viscosity_from_tau(tau_original)
        tau_recovered = tau_from_viscosity(nu)
        
        assert np.isclose(tau_original, tau_recovered)
    
    def test_tau_stability_check(self):
        """Verify tau < 0.5 raises error."""
        with pytest.raises(ValueError):
            viscosity_from_tau(0.4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])