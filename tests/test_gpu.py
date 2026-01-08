"""
Tests for GPU LBM Implementation

Validates GPU kernels against CPU reference implementation.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Check if CUDA is available
try:
    from numba import cuda
    CUDA_AVAILABLE = cuda.is_available()
except:
    CUDA_AVAILABLE = False

# Skip all tests if CUDA not available
pytestmark = pytest.mark.skipif(not CUDA_AVAILABLE, reason="CUDA not available")


class TestGPUEquilibrium:
    """Test GPU equilibrium computation."""
    
    def test_equilibrium_matches_cpu(self):
        """GPU equilibrium should match CPU implementation."""
        from src.kernels.gpu_naive import GPULBMSolver
        from src.equilibrium import compute_equilibrium
        
        nx, ny = 64, 64
        tau = 0.8
        
        # Create non-uniform fields
        rho = 1.0 + 0.1 * np.random.randn(ny, nx)
        ux = 0.05 * np.random.randn(ny, nx)
        uy = 0.05 * np.random.randn(ny, nx)
        
        # CPU reference
        f_eq_cpu = compute_equilibrium(rho, ux, uy)
        
        # GPU computation
        from numba import cuda
        from src.kernels.gpu_naive import compute_equilibrium_kernel
        
        d_rho = cuda.to_device(rho)
        d_ux = cuda.to_device(ux)
        d_uy = cuda.to_device(uy)
        d_f_eq = cuda.device_array((9, ny, nx), dtype=np.float64)
        
        block_size = (16, 16)
        grid_size = ((nx + 15) // 16, (ny + 15) // 16)
        
        compute_equilibrium_kernel[grid_size, block_size](
            d_rho, d_ux, d_uy, d_f_eq, nx, ny
        )
        
        f_eq_gpu = d_f_eq.copy_to_host()
        
        np.testing.assert_allclose(f_eq_gpu, f_eq_cpu, rtol=1e-12)


class TestGPUMacroscopic:
    """Test GPU macroscopic quantity computation."""
    
    def test_macroscopic_matches_cpu(self):
        """GPU macroscopic should match CPU implementation."""
        from src.kernels.gpu_naive import compute_macroscopic_kernel
        from src.observables import compute_macroscopic
        from src.equilibrium import compute_equilibrium
        from numba import cuda
        
        nx, ny = 64, 64
        
        # Create test distribution
        rho = 1.0 + 0.1 * np.random.randn(ny, nx)
        ux = 0.05 * np.random.randn(ny, nx)
        uy = 0.05 * np.random.randn(ny, nx)
        f = compute_equilibrium(rho, ux, uy)
        
        # CPU reference
        rho_cpu, ux_cpu, uy_cpu = compute_macroscopic(f)
        
        # GPU computation
        d_f = cuda.to_device(f)
        d_rho = cuda.device_array((ny, nx), dtype=np.float64)
        d_ux = cuda.device_array((ny, nx), dtype=np.float64)
        d_uy = cuda.device_array((ny, nx), dtype=np.float64)
        
        block_size = (16, 16)
        grid_size = ((nx + 15) // 16, (ny + 15) // 16)
        
        compute_macroscopic_kernel[grid_size, block_size](
            d_f, d_rho, d_ux, d_uy, nx, ny
        )
        
        rho_gpu = d_rho.copy_to_host()
        ux_gpu = d_ux.copy_to_host()
        uy_gpu = d_uy.copy_to_host()
        
        np.testing.assert_allclose(rho_gpu, rho_cpu, rtol=1e-12)
        np.testing.assert_allclose(ux_gpu, ux_cpu, rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(uy_gpu, uy_cpu, rtol=1e-12, atol=1e-15)


class TestGPUCollision:
    """Test GPU collision kernel."""
    
    def test_collision_matches_cpu(self):
        """GPU BGK collision should match CPU implementation."""
        from src.kernels.gpu_naive import bgk_collision_kernel
        from src.collision import bgk_collision
        from src.equilibrium import compute_equilibrium
        from numba import cuda
        
        nx, ny = 64, 64
        tau = 0.8
        omega = 1.0 / tau
        
        # Create test distribution
        rho = np.ones((ny, nx))
        ux = 0.05 * np.random.randn(ny, nx)
        uy = 0.05 * np.random.randn(ny, nx)
        f = compute_equilibrium(rho, ux, uy)
        f_eq = compute_equilibrium(rho, ux, uy)
        
        # Add perturbation
        f = f + 0.01 * np.random.randn(*f.shape)
        
        # CPU reference
        f_cpu = bgk_collision(f.copy(), f_eq, tau)
        
        # GPU computation
        d_f = cuda.to_device(f.copy())
        d_f_eq = cuda.to_device(f_eq)
        
        block_size = (16, 16)
        grid_size = ((nx + 15) // 16, (ny + 15) // 16)
        
        bgk_collision_kernel[grid_size, block_size](
            d_f, d_f_eq, omega, nx, ny
        )
        
        f_gpu = d_f.copy_to_host()
        
        np.testing.assert_allclose(f_gpu, f_cpu, rtol=1e-12)


class TestGPUStreaming:
    """Test GPU streaming kernel."""
    
    def test_streaming_matches_cpu(self):
        """GPU streaming should match CPU implementation."""
        from src.kernels.gpu_naive import streaming_kernel
        from src.streaming import stream_periodic
        from src.equilibrium import compute_equilibrium
        from numba import cuda
        
        nx, ny = 64, 64
        
        # Create test distribution
        rho = 1.0 + 0.1 * np.random.randn(ny, nx)
        ux = 0.05 * np.random.randn(ny, nx)
        uy = 0.05 * np.random.randn(ny, nx)
        f = compute_equilibrium(rho, ux, uy)
        
        # CPU reference
        f_cpu = stream_periodic(f)
        
        # GPU computation
        d_f_src = cuda.to_device(f)
        d_f_dst = cuda.device_array_like(f)
        
        block_size = (16, 16)
        grid_size = ((nx + 15) // 16, (ny + 15) // 16)
        
        streaming_kernel[grid_size, block_size](
            d_f_src, d_f_dst, nx, ny
        )
        
        f_gpu = d_f_dst.copy_to_host()
        
        np.testing.assert_allclose(f_gpu, f_cpu, rtol=1e-12)


class TestGPUFusedKernel:
    """Test GPU fused collision-streaming kernel."""
    
    def test_fused_matches_separate(self):
        """Fused kernel should give same results as separate kernels."""
        from src.kernels.gpu_naive import GPULBMSolver
        
        nx, ny = 64, 64
        tau = 0.8
        
        # Create two solvers
        solver_sep = GPULBMSolver(nx, ny, tau)
        solver_fused = GPULBMSolver(nx, ny, tau)
        
        # Run same number of steps
        for _ in range(10):
            solver_sep.step_separate()
            solver_fused.step_fused()
        
        # Compare distributions
        f_sep = solver_sep.get_distribution()
        f_fused = solver_fused.get_distribution()
        
        np.testing.assert_allclose(f_fused, f_sep, rtol=1e-10)


class TestGPUConservation:
    """Test conservation properties on GPU."""
    
    def test_mass_conservation(self):
        """Mass should be conserved on GPU."""
        from src.kernels.gpu_naive import GPULBMSolver
        
        nx, ny = 64, 64
        tau = 0.8
        
        solver = GPULBMSolver(nx, ny, tau)
        
        f_initial = solver.get_distribution()
        mass_initial = np.sum(f_initial)
        
        # Run steps
        for _ in range(100):
            solver.step()
        
        f_final = solver.get_distribution()
        mass_final = np.sum(f_final)
        
        relative_change = abs(mass_final - mass_initial) / mass_initial
        assert relative_change < 1e-10, f"Mass changed by {relative_change:.2e}"


class TestGPUvsCPU:
    """Compare GPU and CPU solver results."""
    
    def test_gpu_matches_cpu_periodic(self):
        """GPU solver should match CPU solver for periodic domain."""
        from src.kernels.gpu_naive import GPULBMSolver
        from src.kernels.cpu_baseline import CPULBMSolver
        from src.observables import compute_macroscopic_fast
        
        nx, ny = 64, 64
        tau = 0.8
        num_steps = 100
        
        # Create solvers
        gpu_solver = GPULBMSolver(nx, ny, tau)
        cpu_solver = CPULBMSolver(nx, ny, tau)
        
        # Run same number of steps
        for _ in range(num_steps):
            gpu_solver.step()
            cpu_solver.step()
        
        # Compare results
        rho_gpu, ux_gpu, uy_gpu = gpu_solver.get_macroscopic()
        
        # Get CPU results - compute from distribution to ensure consistency
        rho_cpu, ux_cpu, uy_cpu = compute_macroscopic_fast(cpu_solver.f)
        
        np.testing.assert_allclose(rho_gpu, rho_cpu, rtol=1e-10)
        np.testing.assert_allclose(ux_gpu, ux_cpu, rtol=1e-10, atol=1e-15)
        np.testing.assert_allclose(uy_gpu, uy_cpu, rtol=1e-10, atol=1e-15)


if __name__ == "__main__":
    if CUDA_AVAILABLE:
        pytest.main([__file__, "-v"])
    else:
        print("CUDA not available, skipping GPU tests")