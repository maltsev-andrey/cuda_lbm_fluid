"""
Naive GPU Implementation

Basic CUDA kernels using Numba - AoS memory layout.

This is the first GPU implementation, prioritizing correctness over performance.
It establishes the baseline for GPU optimization comparisons.
"""

import numpy as np
from numba import cuda
import math

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from src.lattice import EX, EY, W, CS2, CS4, Q, OPPOSITE


# =============================================================================
# Device Constants - Copy lattice parameters to GPU constant memory
# =============================================================================

# I'll pass these as arguments since Numba doesn't support __constant__ directly
# But I can use cuda.const.array_like for small arrays

@cuda.jit(device=True)
def get_weight(i):
    """Get lattice weight for direction i."""
    if i == 0:
        return 4.0 / 9.0
    elif i < 5:
        return 1.0 / 9.0
    else:
        return 1.0 / 36.0


@cuda.jit(device=True)
def get_ex(i):
    """Get x-component of lattice velocity."""
    # EX = [0, 1, 0, -1, 0, 1, -1, -1, 1]
    if i == 0 or i == 2 or i == 4:
        return 0
    elif i == 1 or i == 5 or i == 8:
        return 1
    else:  # i == 3, 6, 7
        return -1


@cuda.jit(device=True)
def get_ey(i):
    """Get y-component of lattice velocity."""
    # EY = [0, 0, 1, 0, -1, 1, 1, -1, -1]
    if i == 0 or i == 1 or i == 3:
        return 0
    elif i == 2 or i == 5 or i == 6:
        return 1
    else:  # i == 4, 7, 8
        return -1


@cuda.jit(device=True)
def get_opposite(i):
    """Get opposite direction index."""
    # OPPOSITE = [0, 3, 4, 1, 2, 7, 8, 5, 6]
    if i == 0:
        return 0
    elif i == 1:
        return 3
    elif i == 2:
        return 4
    elif i == 3:
        return 1
    elif i == 4:
        return 2
    elif i == 5:
        return 7
    elif i == 6:
        return 8
    elif i == 7:
        return 5
    else:  # i == 8
        return 6


# =============================================================================
# Equilibrium Computation Kernel
# =============================================================================

@cuda.jit
def compute_equilibrium_kernel(rho, ux, uy, f_eq, nx, ny):
    """
    Compute equilibrium distribution on GPU.
    
    Parameters
    ----------
    rho : device array
        Density field, shape (ny, nx)
    ux, uy : device array
        Velocity fields, shape (ny, nx)
    f_eq : device array
        Output equilibrium, shape (Q, ny, nx) - SoA layout
    nx, ny : int
        Grid dimensions
    """
    # 2D thread indexing
    i, j = cuda.grid(2)
    
    if i < nx and j < ny:
        rho_local = rho[j, i]
        ux_local = ux[j, i]
        uy_local = uy[j, i]
        
        u_sq = ux_local * ux_local + uy_local * uy_local
        cs2 = 1.0 / 3.0
        cs4 = cs2 * cs2
        
        for k in range(9):
            ex = get_ex(k)
            ey = get_ey(k)
            w = get_weight(k)
            
            eu = ex * ux_local + ey * uy_local
            
            f_eq[k, j, i] = w * rho_local * (
                1.0 + eu / cs2 + (eu * eu) / (2.0 * cs4) - u_sq / (2.0 * cs2)
            )


# =============================================================================
# Macroscopic Quantities Kernel
# =============================================================================

@cuda.jit
def compute_macroscopic_kernel(f, rho, ux, uy, nx, ny):
    """
    Compute macroscopic quantities (density, velocity) on GPU.
    
    Parameters
    ----------
    f : device array
        Distribution functions, shape (Q, ny, nx)
    rho : device array
        Output density, shape (ny, nx)
    ux, uy : device array
        Output velocity, shape (ny, nx)
    nx, ny : int
        Grid dimensions
    """
    i, j = cuda.grid(2)
    
    if i < nx and j < ny:
        rho_local = 0.0
        rho_ux = 0.0
        rho_uy = 0.0
        
        for k in range(9):
            f_k = f[k, j, i]
            rho_local += f_k
            rho_ux += f_k * get_ex(k)
            rho_uy += f_k * get_ey(k)
        
        rho[j, i] = rho_local
        
        if rho_local > 1e-10:
            ux[j, i] = rho_ux / rho_local
            uy[j, i] = rho_uy / rho_local
        else:
            ux[j, i] = 0.0
            uy[j, i] = 0.0


# =============================================================================
# Collision Kernel (BGK)
# =============================================================================

@cuda.jit
def bgk_collision_kernel(f, f_eq, omega, nx, ny):
    """
    BGK collision kernel.
    
    f_out = f - omega * (f - f_eq)
    
    Parameters
    ----------
    f : device array
        Distribution functions (modified in place), shape (Q, ny, nx)
    f_eq : device array
        Equilibrium distribution, shape (Q, ny, nx)
    omega : float
        Relaxation frequency (1/tau)
    nx, ny : int
        Grid dimensions
    """
    i, j = cuda.grid(2)
    
    if i < nx and j < ny:
        for k in range(9):
            f[k, j, i] = f[k, j, i] - omega * (f[k, j, i] - f_eq[k, j, i])


# =============================================================================
# Streaming Kernel
# =============================================================================

@cuda.jit
def streaming_kernel(f_src, f_dst, nx, ny):
    """
    Streaming kernel with periodic boundaries (pull scheme).
    
    Pull scheme: f_dst[k, j, i] = f_src[k, j-ey, i-ex]
    
    Parameters
    ----------
    f_src : device array
        Source distribution, shape (Q, ny, nx)
    f_dst : device array
        Destination distribution, shape (Q, ny, nx)
    nx, ny : int
        Grid dimensions
    """
    i, j = cuda.grid(2)
    
    if i < nx and j < ny:
        for k in range(9):
            # Source coordinates with periodic wrapping
            ex = get_ex(k)
            ey = get_ey(k)
            
            i_src = (i - ex + nx) % nx
            j_src = (j - ey + ny) % ny
            
            f_dst[k, j, i] = f_src[k, j_src, i_src]


# =============================================================================
# Boundary Condition Kernels
# =============================================================================

@cuda.jit
def bounce_back_kernel(f, solid_mask, nx, ny):
    """
    Bounce-back boundary condition kernel.
    
    At solid nodes, reflect distributions to opposite directions.
    
    Parameters
    ----------
    f : device array
        Distribution functions, shape (Q, ny, nx)
    solid_mask : device array
        Boolean mask for solid nodes, shape (ny, nx)
    nx, ny : int
        Grid dimensions
    """
    i, j = cuda.grid(2)
    
    if i < nx and j < ny:
        if solid_mask[j, i]:
            # Store original values
            f0 = f[0, j, i]
            f1 = f[1, j, i]
            f2 = f[2, j, i]
            f3 = f[3, j, i]
            f4 = f[4, j, i]
            f5 = f[5, j, i]
            f6 = f[6, j, i]
            f7 = f[7, j, i]
            f8 = f[8, j, i]
            
            # Swap with opposites
            f[0, j, i] = f0  # Self
            f[1, j, i] = f3  # 1 <-> 3
            f[2, j, i] = f4  # 2 <-> 4
            f[3, j, i] = f1
            f[4, j, i] = f2
            f[5, j, i] = f7  # 5 <-> 7
            f[6, j, i] = f8  # 6 <-> 8
            f[7, j, i] = f5
            f[8, j, i] = f6


@cuda.jit
def moving_wall_kernel(f, wall_mask, ux_wall, uy_wall, rho_wall, nx, ny):
    """
    Moving wall bounce-back kernel.
    
    f_i = f_{i*} + 2 * w_i * rho * (e_i . u_wall) / cs^2
    
    Parameters
    ----------
    f : device array
        Distribution functions, shape (Q, ny, nx)
    wall_mask : device array
        Boolean mask for wall nodes, shape (ny, nx)
    ux_wall, uy_wall : float
        Wall velocity components
    rho_wall : float
        Wall density (assumed)
    nx, ny : int
        Grid dimensions
    """
    i, j = cuda.grid(2)
    
    cs2 = 1.0 / 3.0
    
    if i < nx and j < ny:
        if wall_mask[j, i]:
            # Store original values
            f_temp = cuda.local.array(9, dtype=np.float64)
            for k in range(9):
                f_temp[k] = f[k, j, i]
            
            # Apply moving wall bounce-back
            for k in range(9):
                k_opp = get_opposite(k)
                ex = get_ex(k)
                ey = get_ey(k)
                w = get_weight(k)
                
                eu = ex * ux_wall + ey * uy_wall
                correction = 2.0 * w * rho_wall * eu / cs2
                
                f[k, j, i] = f_temp[k_opp] + correction


# =============================================================================
# Fused Collision-Streaming Kernel (for better performance)
# =============================================================================

@cuda.jit
def collide_stream_kernel(f_src, f_dst, omega, nx, ny):
    """
    Fused collision and streaming kernel.
    
    Combines:
    1. Pull distributions from neighbors
    2. Compute macroscopic quantities
    3. Compute equilibrium
    4. Apply BGK collision
    
    This reduces memory traffic by avoiding intermediate storage.
    
    Parameters
    ----------
    f_src : device array
        Source distribution, shape (Q, ny, nx)
    f_dst : device array
        Destination distribution, shape (Q, ny, nx)
    omega : float
        Relaxation frequency (1/tau)
    nx, ny : int
        Grid dimensions
    """
    i, j = cuda.grid(2)
    
    if i < nx and j < ny:
        # Local storage for pulled distributions
        f_local = cuda.local.array(9, dtype=np.float64)
        
        # Pull and accumulate macroscopic quantities
        rho_local = 0.0
        rho_ux = 0.0
        rho_uy = 0.0
        
        for k in range(9):
            ex = get_ex(k)
            ey = get_ey(k)
            
            # Source coordinates (pull scheme)
            i_src = (i - ex + nx) % nx
            j_src = (j - ey + ny) % ny
            
            f_k = f_src[k, j_src, i_src]
            f_local[k] = f_k
            
            rho_local += f_k
            rho_ux += f_k * ex
            rho_uy += f_k * ey
        
        # Compute velocity
        if rho_local > 1e-10:
            ux_local = rho_ux / rho_local
            uy_local = rho_uy / rho_local
        else:
            ux_local = 0.0
            uy_local = 0.0
        
        # Compute equilibrium and collide
        u_sq = ux_local * ux_local + uy_local * uy_local
        cs2 = 1.0 / 3.0
        cs4 = cs2 * cs2
        
        for k in range(9):
            ex = get_ex(k)
            ey = get_ey(k)
            w = get_weight(k)
            
            eu = ex * ux_local + ey * uy_local
            f_eq = w * rho_local * (
                1.0 + eu / cs2 + (eu * eu) / (2.0 * cs4) - u_sq / (2.0 * cs2)
            )
            
            f_dst[k, j, i] = f_local[k] - omega * (f_local[k] - f_eq)


# =============================================================================
# GPU Solver Class
# =============================================================================

class GPULBMSolver:
    """
    GPU-accelerated LBM solver using Numba CUDA.
    
    This is the naive implementation for establishing GPU baseline.
    
    Parameters
    ----------
    nx, ny : int
        Grid dimensions
    tau : float
        Relaxation time (must be > 0.5)
    block_size : tuple
        CUDA block dimensions (default (16, 16))
    """
    
    def __init__(self, nx, ny, tau, block_size=(16, 16)):
        self.nx = nx
        self.ny = ny
        self.tau = tau
        self.omega = 1.0 / tau
        self.block_size = block_size
        
        # Compute grid dimensions
        self.grid_size = (
            (nx + block_size[0] - 1) // block_size[0],
            (ny + block_size[1] - 1) // block_size[1]
        )
        
        # Allocate host arrays
        self.rho_host = np.ones((ny, nx), dtype=np.float64)
        self.ux_host = np.zeros((ny, nx), dtype=np.float64)
        self.uy_host = np.zeros((ny, nx), dtype=np.float64)
        
        # Initialize equilibrium distribution on host
        from src.equilibrium import compute_equilibrium
        self.f_host = compute_equilibrium(self.rho_host, self.ux_host, self.uy_host)
        
        # Allocate device arrays
        self.d_f = cuda.to_device(self.f_host)
        self.d_f_temp = cuda.device_array_like(self.f_host)
        self.d_rho = cuda.to_device(self.rho_host)
        self.d_ux = cuda.to_device(self.ux_host)
        self.d_uy = cuda.to_device(self.uy_host)
        self.d_f_eq = cuda.device_array_like(self.f_host)
        
        # Boundary masks (allocated on demand)
        self.d_solid_mask = None
        self.d_lid_mask = None
        
        self.step_count = 0
    
    def set_solid_mask(self, solid_mask):
        """Set solid boundary mask."""
        self.d_solid_mask = cuda.to_device(solid_mask.astype(np.bool_))
    
    def set_lid_mask(self, lid_mask, ux_lid, uy_lid=0.0, rho_lid=1.0):
        """Set moving lid boundary mask and velocity."""
        self.d_lid_mask = cuda.to_device(lid_mask.astype(np.bool_))
        self.ux_lid = ux_lid
        self.uy_lid = uy_lid
        self.rho_lid = rho_lid
    
    def step_separate(self):
        """
        Perform one LBM timestep using separate kernels.
        
        Order: Macroscopic -> Equilibrium -> Collision -> Streaming -> BCs
        """
        # Compute macroscopic quantities
        compute_macroscopic_kernel[self.grid_size, self.block_size](
            self.d_f, self.d_rho, self.d_ux, self.d_uy, self.nx, self.ny
        )
        
        # Compute equilibrium
        compute_equilibrium_kernel[self.grid_size, self.block_size](
            self.d_rho, self.d_ux, self.d_uy, self.d_f_eq, self.nx, self.ny
        )
        
        # Collision (in-place)
        bgk_collision_kernel[self.grid_size, self.block_size](
            self.d_f, self.d_f_eq, self.omega, self.nx, self.ny
        )
        
        # Streaming (double buffer)
        streaming_kernel[self.grid_size, self.block_size](
            self.d_f, self.d_f_temp, self.nx, self.ny
        )
        
        # Swap buffers
        self.d_f, self.d_f_temp = self.d_f_temp, self.d_f
        
        # Apply boundary conditions
        if self.d_solid_mask is not None:
            bounce_back_kernel[self.grid_size, self.block_size](
                self.d_f, self.d_solid_mask, self.nx, self.ny
            )
        
        if self.d_lid_mask is not None:
            moving_wall_kernel[self.grid_size, self.block_size](
                self.d_f, self.d_lid_mask, self.ux_lid, self.uy_lid, 
                self.rho_lid, self.nx, self.ny
            )
        
        self.step_count += 1
    
    def step_fused(self):
        """
        Perform one LBM timestep using fused kernel.
        
        More efficient but BCs need separate handling.
        """
        # Fused collision-streaming
        collide_stream_kernel[self.grid_size, self.block_size](
            self.d_f, self.d_f_temp, self.omega, self.nx, self.ny
        )
        
        # Swap buffers
        self.d_f, self.d_f_temp = self.d_f_temp, self.d_f
        
        # Apply boundary conditions
        if self.d_solid_mask is not None:
            bounce_back_kernel[self.grid_size, self.block_size](
                self.d_f, self.d_solid_mask, self.nx, self.ny
            )
        
        if self.d_lid_mask is not None:
            moving_wall_kernel[self.grid_size, self.block_size](
                self.d_f, self.d_lid_mask, self.ux_lid, self.uy_lid,
                self.rho_lid, self.nx, self.ny
            )
        
        self.step_count += 1
    
    def step(self):
        """Default step method - uses fused kernel."""
        self.step_fused()
    
    def get_macroscopic(self):
        """
        Get macroscopic quantities from GPU.
        
        Returns
        -------
        rho, ux, uy : ndarray
            Density and velocity fields on host
        """
        # Compute on GPU
        compute_macroscopic_kernel[self.grid_size, self.block_size](
            self.d_f, self.d_rho, self.d_ux, self.d_uy, self.nx, self.ny
        )
        
        # Copy to host
        self.rho_host = self.d_rho.copy_to_host()
        self.ux_host = self.d_ux.copy_to_host()
        self.uy_host = self.d_uy.copy_to_host()
        
        return self.rho_host, self.ux_host, self.uy_host
    
    def get_distribution(self):
        """Get distribution functions from GPU."""
        return self.d_f.copy_to_host()
    
    def synchronize(self):
        """Synchronize GPU (wait for all kernels to complete)."""
        cuda.synchronize()
    
    def run(self, num_steps, use_fused=True, verbose=True, report_interval=1000):
        """
        Run simulation for specified number of steps.
        
        Parameters
        ----------
        num_steps : int
            Number of timesteps
        use_fused : bool
            Use fused kernel (default True)
        verbose : bool
            Print progress
        report_interval : int
            Steps between progress reports
        
        Returns
        -------
        mlups : float
            Performance in Million Lattice Updates Per Second
        """
        import time
        
        step_func = self.step_fused if use_fused else self.step_separate
        
        # Warmup
        for _ in range(10):
            step_func()
        self.synchronize()
        
        # Timed run
        start = time.perf_counter()
        
        for step in range(num_steps):
            step_func()
            
            if verbose and (step + 1) % report_interval == 0:
                self.synchronize()
                elapsed = time.perf_counter() - start
                mlups = (step + 1) * self.nx * self.ny / elapsed / 1e6
                print(f"Step {step + 1}/{num_steps}, MLUPS: {mlups:.2f}")
        
        self.synchronize()
        total = time.perf_counter() - start
        mlups = num_steps * self.nx * self.ny / total / 1e6
        
        if verbose:
            print(f"Completed {num_steps} steps in {total:.2f}s")
            print(f"Performance: {mlups:.2f} MLUPS")
        
        return mlups


def check_cuda_available():
    """Check if CUDA is available."""
    try:
        cuda.detect()
        return True
    except:
        return False


def benchmark_gpu_solver(grid_sizes=None, num_steps=1000, use_fused=True):
    """
    Benchmark GPU solver across different grid sizes.
    
    Parameters
    ----------
    grid_sizes : list of tuples
        List of (nx, ny) grid sizes
    num_steps : int
        Number of steps per benchmark
    use_fused : bool
        Use fused kernel
    
    Returns
    -------
    results : dict
        Grid sizes and corresponding MLUPS
    """
    if not check_cuda_available():
        print("CUDA not available!")
        return {}
    
    if grid_sizes is None:
        grid_sizes = [
            (128, 128),
            (256, 256),
            (512, 512),
            (1024, 1024),
            (2048, 2048),
        ]
    
    results = {}
    tau = 0.8
    
    print("GPU LBM Solver Benchmark (Naive Implementation)")
    print("=" * 60)
    print(f"Tau: {tau}, Steps: {num_steps}, Fused: {use_fused}")
    print()
    
    for nx, ny in grid_sizes:
        print(f"Grid size: {nx} x {ny}")
        
        try:
            solver = GPULBMSolver(nx, ny, tau)
            mlups = solver.run(num_steps, use_fused=use_fused, verbose=False)
            results[(nx, ny)] = mlups
            print(f"  Performance: {mlups:.2f} MLUPS")
        except Exception as e:
            print(f"  Error: {e}")
            results[(nx, ny)] = 0
        
        print()
    
    return results


if __name__ == "__main__":
    if check_cuda_available():
        print("CUDA is available!")
        print()
        
        # Run benchmark
        results = benchmark_gpu_solver()
        
        print("\nSummary")
        print("=" * 60)
        for (nx, ny), mlups in results.items():
            print(f"{nx:4d} x {ny:4d}: {mlups:8.2f} MLUPS")
    else:
        print("CUDA is not available. Cannot run GPU benchmarks.")