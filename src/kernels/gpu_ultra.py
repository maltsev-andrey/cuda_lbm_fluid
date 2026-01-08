"""
Ultra-Optimized GPU LBM Kernels

Target: 4000+ MLUPS on Tesla P100

Optimization strategies:
1. Float32 precision option (2x memory bandwidth)
2. Improved memory coalescing with swizzled access
3. Reduced register pressure
4. Instruction-level parallelism
5. Optimized block sizes for P100 (56 SMs, 64 warps/SM)
"""

import numpy as np
from numba import cuda, float32, float64
from numba.cuda import shared
import time
import warnings

# Suppress performance warnings
try:
    from numba.core.errors import NumbaPerformanceWarning
    warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)
except:
    pass

# Check CUDA
CUDA_AVAILABLE = cuda.is_available() if hasattr(cuda, 'is_available') else False


# =============================================================================
# Float32 Ultra-Fast Kernel (2x bandwidth improvement)
# =============================================================================

@cuda.jit('void(float32[:,:,:], float32[:,:,:], float32, int32, int32)', fastmath=True)
def collide_stream_f32(f_src, f_dst, omega, nx, ny):
    """
    Ultra-optimized collision-streaming with float32.
    
    Float32 provides:
    - 2x memory bandwidth (half the bytes)
    - 2x register capacity
    - Faster FP operations on some GPUs
    """
    i, j = cuda.grid(2)
    
    if i >= nx or j >= ny:
        return
    
    # Precompute neighbor indices (helps compiler optimization)
    im1 = i - 1 if i > 0 else nx - 1
    ip1 = i + 1 if i < nx - 1 else 0
    jm1 = j - 1 if j > 0 else ny - 1
    jp1 = j + 1 if j < ny - 1 else 0
    
    # Pull distributions (coalesced access pattern)
    f0 = f_src[0, j, i]
    f1 = f_src[1, j, im1]
    f2 = f_src[2, jm1, i]
    f3 = f_src[3, j, ip1]
    f4 = f_src[4, jp1, i]
    f5 = f_src[5, jm1, im1]
    f6 = f_src[6, jm1, ip1]
    f7 = f_src[7, jp1, ip1]
    f8 = f_src[8, jp1, im1]
    
    # Macroscopic quantities
    rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8
    inv_rho = float32(1.0) / rho
    ux = (f1 - f3 + f5 - f6 - f7 + f8) * inv_rho
    uy = (f2 - f4 + f5 + f6 - f7 - f8) * inv_rho
    
    # Equilibrium computation (optimized)
    ux2 = ux * ux
    uy2 = uy * uy
    u2 = ux2 + uy2
    u2_15 = float32(1.5) * u2
    
    ux3 = float32(3.0) * ux
    uy3 = float32(3.0) * uy
    
    # Weight constants
    w0 = float32(0.44444444)
    w1 = float32(0.11111111)
    w2 = float32(0.02777778)
    
    # Collision (BGK)
    omp = float32(1.0) - omega
    
    # Direction 0
    f_eq = w0 * rho * (float32(1.0) - u2_15)
    f_dst[0, j, i] = omp * f0 + omega * f_eq
    
    # Direction 1: (+1, 0)
    eu = ux3
    f_eq = w1 * rho * (float32(1.0) + eu + float32(0.5) * eu * eu - u2_15)
    f_dst[1, j, i] = omp * f1 + omega * f_eq
    
    # Direction 2: (0, +1)
    eu = uy3
    f_eq = w1 * rho * (float32(1.0) + eu + float32(0.5) * eu * eu - u2_15)
    f_dst[2, j, i] = omp * f2 + omega * f_eq
    
    # Direction 3: (-1, 0)
    eu = -ux3
    f_eq = w1 * rho * (float32(1.0) + eu + float32(0.5) * eu * eu - u2_15)
    f_dst[3, j, i] = omp * f3 + omega * f_eq
    
    # Direction 4: (0, -1)
    eu = -uy3
    f_eq = w1 * rho * (float32(1.0) + eu + float32(0.5) * eu * eu - u2_15)
    f_dst[4, j, i] = omp * f4 + omega * f_eq
    
    # Direction 5: (+1, +1)
    eu = ux3 + uy3
    f_eq = w2 * rho * (float32(1.0) + eu + float32(0.5) * eu * eu - u2_15)
    f_dst[5, j, i] = omp * f5 + omega * f_eq
    
    # Direction 6: (-1, +1)
    eu = -ux3 + uy3
    f_eq = w2 * rho * (float32(1.0) + eu + float32(0.5) * eu * eu - u2_15)
    f_dst[6, j, i] = omp * f6 + omega * f_eq
    
    # Direction 7: (-1, -1)
    eu = -ux3 - uy3
    f_eq = w2 * rho * (float32(1.0) + eu + float32(0.5) * eu * eu - u2_15)
    f_dst[7, j, i] = omp * f7 + omega * f_eq
    
    # Direction 8: (+1, -1)
    eu = ux3 - uy3
    f_eq = w2 * rho * (float32(1.0) + eu + float32(0.5) * eu * eu - u2_15)
    f_dst[8, j, i] = omp * f8 + omega * f_eq


@cuda.jit('void(float64[:,:,:], float64[:,:,:], float64, int32, int32)', fastmath=True)
def collide_stream_f64_optimized(f_src, f_dst, omega, nx, ny):
    """
    Optimized float64 kernel with improved memory access pattern.
    """
    i, j = cuda.grid(2)
    
    if i >= nx or j >= ny:
        return
    
    # Precompute neighbor indices
    im1 = i - 1 if i > 0 else nx - 1
    ip1 = i + 1 if i < nx - 1 else 0
    jm1 = j - 1 if j > 0 else ny - 1
    jp1 = j + 1 if j < ny - 1 else 0
    
    # Pull all distributions first (better memory access pattern)
    f0 = f_src[0, j, i]
    f1 = f_src[1, j, im1]
    f2 = f_src[2, jm1, i]
    f3 = f_src[3, j, ip1]
    f4 = f_src[4, jp1, i]
    f5 = f_src[5, jm1, im1]
    f6 = f_src[6, jm1, ip1]
    f7 = f_src[7, jp1, ip1]
    f8 = f_src[8, jp1, im1]
    
    # Macroscopic quantities
    rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8
    inv_rho = 1.0 / rho
    ux = (f1 - f3 + f5 - f6 - f7 + f8) * inv_rho
    uy = (f2 - f4 + f5 + f6 - f7 - f8) * inv_rho
    
    # Precompute common terms
    u2_15 = 1.5 * (ux * ux + uy * uy)
    ux3 = 3.0 * ux
    uy3 = 3.0 * uy
    omp = 1.0 - omega
    
    # Collision + Write (interleaved to hide latency)
    eu = 0.0
    f_eq = 0.444444444444444 * rho * (1.0 - u2_15)
    f_dst[0, j, i] = omp * f0 + omega * f_eq
    
    eu = ux3
    f_eq = 0.111111111111111 * rho * (1.0 + eu + 0.5 * eu * eu - u2_15)
    f_dst[1, j, i] = omp * f1 + omega * f_eq
    
    eu = uy3
    f_eq = 0.111111111111111 * rho * (1.0 + eu + 0.5 * eu * eu - u2_15)
    f_dst[2, j, i] = omp * f2 + omega * f_eq
    
    eu = -ux3
    f_eq = 0.111111111111111 * rho * (1.0 + eu + 0.5 * eu * eu - u2_15)
    f_dst[3, j, i] = omp * f3 + omega * f_eq
    
    eu = -uy3
    f_eq = 0.111111111111111 * rho * (1.0 + eu + 0.5 * eu * eu - u2_15)
    f_dst[4, j, i] = omp * f4 + omega * f_eq
    
    eu = ux3 + uy3
    f_eq = 0.027777777777778 * rho * (1.0 + eu + 0.5 * eu * eu - u2_15)
    f_dst[5, j, i] = omp * f5 + omega * f_eq
    
    eu = -ux3 + uy3
    f_eq = 0.027777777777778 * rho * (1.0 + eu + 0.5 * eu * eu - u2_15)
    f_dst[6, j, i] = omp * f6 + omega * f_eq
    
    eu = -ux3 - uy3
    f_eq = 0.027777777777778 * rho * (1.0 + eu + 0.5 * eu * eu - u2_15)
    f_dst[7, j, i] = omp * f7 + omega * f_eq
    
    eu = ux3 - uy3
    f_eq = 0.027777777777778 * rho * (1.0 + eu + 0.5 * eu * eu - u2_15)
    f_dst[8, j, i] = omp * f8 + omega * f_eq


# =============================================================================
# AA-Pattern with Float32 (Best Performance)
# =============================================================================

@cuda.jit('void(float32[:,:,:], float32, int32, int32)', fastmath=True)
def aa_even_f32(f, omega, nx, ny):
    """AA-pattern even step with float32."""
    i, j = cuda.grid(2)
    
    if i >= nx or j >= ny:
        return
    
    # Load all distributions
    f0 = f[0, j, i]
    f1 = f[1, j, i]
    f2 = f[2, j, i]
    f3 = f[3, j, i]
    f4 = f[4, j, i]
    f5 = f[5, j, i]
    f6 = f[6, j, i]
    f7 = f[7, j, i]
    f8 = f[8, j, i]
    
    # Macroscopic
    rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8
    inv_rho = float32(1.0) / rho
    ux = (f1 - f3 + f5 - f6 - f7 + f8) * inv_rho
    uy = (f2 - f4 + f5 + f6 - f7 - f8) * inv_rho
    
    # Equilibrium
    u2_15 = float32(1.5) * (ux * ux + uy * uy)
    ux3 = float32(3.0) * ux
    uy3 = float32(3.0) * uy
    omp = float32(1.0) - omega
    
    w0, w1, w2 = float32(0.44444444), float32(0.11111111), float32(0.02777778)
    
    # Collision in place
    eu = float32(0.0)
    f[0, j, i] = omp * f0 + omega * w0 * rho * (float32(1.0) - u2_15)
    
    eu = ux3
    f[1, j, i] = omp * f1 + omega * w1 * rho * (float32(1.0) + eu + float32(0.5)*eu*eu - u2_15)
    
    eu = uy3
    f[2, j, i] = omp * f2 + omega * w1 * rho * (float32(1.0) + eu + float32(0.5)*eu*eu - u2_15)
    
    eu = -ux3
    f[3, j, i] = omp * f3 + omega * w1 * rho * (float32(1.0) + eu + float32(0.5)*eu*eu - u2_15)
    
    eu = -uy3
    f[4, j, i] = omp * f4 + omega * w1 * rho * (float32(1.0) + eu + float32(0.5)*eu*eu - u2_15)
    
    eu = ux3 + uy3
    f[5, j, i] = omp * f5 + omega * w2 * rho * (float32(1.0) + eu + float32(0.5)*eu*eu - u2_15)
    
    eu = -ux3 + uy3
    f[6, j, i] = omp * f6 + omega * w2 * rho * (float32(1.0) + eu + float32(0.5)*eu*eu - u2_15)
    
    eu = -ux3 - uy3
    f[7, j, i] = omp * f7 + omega * w2 * rho * (float32(1.0) + eu + float32(0.5)*eu*eu - u2_15)
    
    eu = ux3 - uy3
    f[8, j, i] = omp * f8 + omega * w2 * rho * (float32(1.0) + eu + float32(0.5)*eu*eu - u2_15)


@cuda.jit('void(float32[:,:,:], float32, int32, int32)', fastmath=True)
def aa_odd_f32(f, omega, nx, ny):
    """AA-pattern odd step with float32 - includes streaming swap."""
    i, j = cuda.grid(2)
    
    if i >= nx or j >= ny:
        return
    
    # Neighbor indices
    im1 = i - 1 if i > 0 else nx - 1
    ip1 = i + 1 if i < nx - 1 else 0
    jm1 = j - 1 if j > 0 else ny - 1
    jp1 = j + 1 if j < ny - 1 else 0
    
    # Load local
    f0 = f[0, j, i]
    f1 = f[1, j, i]
    f2 = f[2, j, i]
    f3 = f[3, j, i]
    f4 = f[4, j, i]
    f5 = f[5, j, i]
    f6 = f[6, j, i]
    f7 = f[7, j, i]
    f8 = f[8, j, i]
    
    # Macroscopic
    rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8
    inv_rho = float32(1.0) / rho
    ux = (f1 - f3 + f5 - f6 - f7 + f8) * inv_rho
    uy = (f2 - f4 + f5 + f6 - f7 - f8) * inv_rho
    
    # Equilibrium
    u2_15 = float32(1.5) * (ux * ux + uy * uy)
    ux3 = float32(3.0) * ux
    uy3 = float32(3.0) * uy
    omp = float32(1.0) - omega
    
    w0, w1, w2 = float32(0.44444444), float32(0.11111111), float32(0.02777778)
    
    # Collision
    eu = float32(0.0)
    f[0, j, i] = omp * f0 + omega * w0 * rho * (float32(1.0) - u2_15)
    
    # Directions 1,3 swap
    eu = ux3
    f1_new = omp * f1 + omega * w1 * rho * (float32(1.0) + eu + float32(0.5)*eu*eu - u2_15)
    eu = -ux3
    f3_new = omp * f3 + omega * w1 * rho * (float32(1.0) + eu + float32(0.5)*eu*eu - u2_15)
    
    f[1, j, im1] = f3_new
    f[3, j, ip1] = f1_new
    
    # Directions 2,4 swap
    eu = uy3
    f2_new = omp * f2 + omega * w1 * rho * (float32(1.0) + eu + float32(0.5)*eu*eu - u2_15)
    eu = -uy3
    f4_new = omp * f4 + omega * w1 * rho * (float32(1.0) + eu + float32(0.5)*eu*eu - u2_15)
    
    f[2, jm1, i] = f4_new
    f[4, jp1, i] = f2_new
    
    # Directions 5,7 swap
    eu = ux3 + uy3
    f5_new = omp * f5 + omega * w2 * rho * (float32(1.0) + eu + float32(0.5)*eu*eu - u2_15)
    eu = -ux3 - uy3
    f7_new = omp * f7 + omega * w2 * rho * (float32(1.0) + eu + float32(0.5)*eu*eu - u2_15)
    
    f[5, jm1, im1] = f7_new
    f[7, jp1, ip1] = f5_new
    
    # Directions 6,8 swap
    eu = -ux3 + uy3
    f6_new = omp * f6 + omega * w2 * rho * (float32(1.0) + eu + float32(0.5)*eu*eu - u2_15)
    eu = ux3 - uy3
    f8_new = omp * f8 + omega * w2 * rho * (float32(1.0) + eu + float32(0.5)*eu*eu - u2_15)
    
    f[6, jm1, ip1] = f8_new
    f[8, jp1, im1] = f6_new


# =============================================================================
# Benchmark Solver Class
# =============================================================================

class UltraFastLBMSolver:
    """
    Ultra-optimized LBM solver for maximum performance.
    
    Features:
    - Float32 or Float64 precision
    - AA-pattern or standard streaming
    - Optimized block sizes
    """
    
    def __init__(self, nx, ny, tau=0.8, dtype='float32', kernel='aa'):
        """
        Parameters
        ----------
        nx, ny : int
            Grid size
        tau : float
            Relaxation time
        dtype : str
            'float32' or 'float64'
        kernel : str
            'aa' for AA-pattern, 'standard' for double-buffer
        """
        self.nx = nx
        self.ny = ny
        self.tau = tau
        self.omega = 1.0 / tau
        self.dtype = dtype
        self.kernel_type = kernel
        
        # Select numpy dtype
        np_dtype = np.float32 if dtype == 'float32' else np.float64
        
        # Initialize on CPU
        rho = np.ones((ny, nx), dtype=np_dtype)
        ux = np.ones((ny, nx), dtype=np_dtype) * 0.05
        uy = np.zeros((ny, nx), dtype=np_dtype)
        
        # Equilibrium initialization
        f = np.zeros((9, ny, nx), dtype=np_dtype)
        for k in range(9):
            ex = [0, 1, 0, -1, 0, 1, -1, -1, 1][k]
            ey = [0, 0, 1, 0, -1, 1, 1, -1, -1][k]
            w = [4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36][k]
            eu = ex * ux + ey * uy
            u_sq = ux**2 + uy**2
            f[k] = w * rho * (1 + 3*eu + 4.5*eu**2 - 1.5*u_sq)
        
        # Transfer to GPU
        self.d_f = cuda.to_device(f)
        if kernel != 'aa':
            self.d_f_temp = cuda.device_array_like(f)
        
        # Optimal block size for P100
        self.block = (32, 8)
        self.grid = (
            (nx + self.block[0] - 1) // self.block[0],
            (ny + self.block[1] - 1) // self.block[1]
        )
        
        self.step_count = 0
        self.is_even = True
    
    def step(self):
        """Single timestep."""
        omega = np.float32(self.omega) if self.dtype == 'float32' else self.omega
        
        if self.kernel_type == 'aa':
            if self.dtype == 'float32':
                if self.is_even:
                    aa_even_f32[self.grid, self.block](self.d_f, omega, self.nx, self.ny)
                else:
                    aa_odd_f32[self.grid, self.block](self.d_f, omega, self.nx, self.ny)
            else:
                # Float64 AA not implemented, use standard
                collide_stream_f64_optimized[self.grid, self.block](
                    self.d_f, self.d_f_temp, omega, self.nx, self.ny
                )
                self.d_f, self.d_f_temp = self.d_f_temp, self.d_f
        else:
            if self.dtype == 'float32':
                collide_stream_f32[self.grid, self.block](
                    self.d_f, self.d_f_temp, omega, self.nx, self.ny
                )
            else:
                collide_stream_f64_optimized[self.grid, self.block](
                    self.d_f, self.d_f_temp, omega, self.nx, self.ny
                )
            self.d_f, self.d_f_temp = self.d_f_temp, self.d_f
        
        self.is_even = not self.is_even
        self.step_count += 1
    
    def run(self, num_steps, warmup=100):
        """Run benchmark and return MLUPS."""
        # Warmup
        for _ in range(warmup):
            self.step()
        cuda.synchronize()
        
        # Timed run
        start = time.perf_counter()
        for _ in range(num_steps):
            self.step()
        cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        mlups = num_steps * self.nx * self.ny / elapsed / 1e6
        return mlups


def benchmark_ultra_optimized():
    """Run comprehensive benchmark."""
    if not CUDA_AVAILABLE:
        print("CUDA not available!")
        return
    
    print("=" * 70)
    print("Ultra-Optimized LBM Benchmark")
    print("=" * 70)
    
    grid_sizes = [(512, 512), (1024, 1024), (2048, 2048), (4096, 4096)]
    
    results = {}
    
    # Test Float64 Standard
    print("\nFloat64 Standard Streaming:")
    print("-" * 50)
    for nx, ny in grid_sizes:
        try:
            solver = UltraFastLBMSolver(nx, ny, dtype='float64', kernel='standard')
            mlups = solver.run(1000)
            results[(nx, ny, 'f64', 'std')] = mlups
            print(f"  {nx:4d} x {ny:4d}: {mlups:8.1f} MLUPS")
        except Exception as e:
            print(f"  {nx:4d} x {ny:4d}: Error - {e}")
    
    # Test Float32 Standard
    print("\nFloat32 Standard Streaming:")
    print("-" * 50)
    for nx, ny in grid_sizes:
        try:
            solver = UltraFastLBMSolver(nx, ny, dtype='float32', kernel='standard')
            mlups = solver.run(1000)
            results[(nx, ny, 'f32', 'std')] = mlups
            print(f"  {nx:4d} x {ny:4d}: {mlups:8.1f} MLUPS")
        except Exception as e:
            print(f"  {nx:4d} x {ny:4d}: Error - {e}")
    
    # Test Float32 AA-Pattern
    print("\nFloat32 AA-Pattern (Best):")
    print("-" * 50)
    for nx, ny in grid_sizes:
        try:
            solver = UltraFastLBMSolver(nx, ny, dtype='float32', kernel='aa')
            mlups = solver.run(1000)
            results[(nx, ny, 'f32', 'aa')] = mlups
            print(f"  {nx:4d} x {ny:4d}: {mlups:8.1f} MLUPS")
        except Exception as e:
            print(f"  {nx:4d} x {ny:4d}: Error - {e}")
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary - Peak Performance:")
    print("=" * 70)
    
    max_mlups = 0
    best_config = None
    for key, mlups in results.items():
        if mlups > max_mlups:
            max_mlups = mlups
            best_config = key
    
    if best_config:
        nx, ny, dtype, kernel = best_config
        print(f"  Best: {nx}x{ny} {dtype} {kernel} = {max_mlups:.1f} MLUPS")
    
    # Memory bandwidth calculation
    if max_mlups > 0:
        # 9 distributions * 2 (read+write) * bytes per element
        bytes_per_cell = 9 * 2 * (4 if 'f32' in str(best_config) else 8)
        bandwidth = max_mlups * 1e6 * bytes_per_cell / 1e9
        print(f"  Effective Bandwidth: {bandwidth:.1f} GB/s")
        print(f"  P100 Theoretical: 732 GB/s")
        print(f"  Efficiency: {bandwidth/732*100:.1f}%")
    
    return results


if __name__ == "__main__":
    benchmark_ultra_optimized()