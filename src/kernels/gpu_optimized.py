"""
Optimized GPU Implementation

Advanced CUDA kernels with memory optimizations:
1. Structure of Arrays (SoA) - already using this
2. Shared memory for neighbor access
3. Register optimization
4. Memory coalescing improvements
5. Reduced kernel launch overhead

Target: 2000+ MLUPS on Tesla P100
"""

import numpy as np
from numba import cuda, float64, int32
from numba.cuda import shared, local
import math

# =============================================================================
# Constants - hardcoded for D2Q9 to enable compiler optimizations
# =============================================================================

# D2Q9 lattice velocities (compile-time constants)
D2Q9_EX = (0, 1, 0, -1, 0, 1, -1, -1, 1)
D2Q9_EY = (0, 0, 1, 0, -1, 1, 1, -1, -1)
D2Q9_W = (4.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 1.0/9.0, 
          1.0/36.0, 1.0/36.0, 1.0/36.0, 1.0/36.0)
D2Q9_OPP = (0, 3, 4, 1, 2, 7, 8, 5, 6)

CS2 = 1.0 / 3.0
CS4 = 1.0 / 9.0


# =============================================================================
# Optimized Fused Kernel with Loop Unrolling
# =============================================================================

@cuda.jit(fastmath=True)
def collide_stream_optimized(f_src, f_dst, omega, nx, ny):
    """
    Highly optimized collision-streaming kernel.
    
    Optimizations:
    - Fully unrolled loops for D2Q9
    - fastmath for faster FP operations
    - Register-based computation
    - Coalesced memory access pattern
    
    Parameters
    ----------
    f_src : device array, shape (9, ny, nx)
        Source distribution (SoA layout)
    f_dst : device array, shape (9, ny, nx)
        Destination distribution
    omega : float
        Relaxation frequency (1/tau)
    nx, ny : int
        Grid dimensions
    """
    i, j = cuda.grid(2)
    
    if i < nx and j < ny:
        # Pull distributions from neighbors (unrolled)
        # Direction 0: (0, 0) - self
        f0 = f_src[0, j, i]
        
        # Direction 1: (1, 0) - pull from left
        i_src = i - 1 if i > 0 else nx - 1
        f1 = f_src[1, j, i_src]
        
        # Direction 2: (0, 1) - pull from below
        j_src = j - 1 if j > 0 else ny - 1
        f2 = f_src[2, j_src, i]
        
        # Direction 3: (-1, 0) - pull from right
        i_src = i + 1 if i < nx - 1 else 0
        f3 = f_src[3, j, i_src]
        
        # Direction 4: (0, -1) - pull from above
        j_src = j + 1 if j < ny - 1 else 0
        f4 = f_src[4, j_src, i]
        
        # Direction 5: (1, 1) - pull from bottom-left
        i_src = i - 1 if i > 0 else nx - 1
        j_src = j - 1 if j > 0 else ny - 1
        f5 = f_src[5, j_src, i_src]
        
        # Direction 6: (-1, 1) - pull from bottom-right
        i_src = i + 1 if i < nx - 1 else 0
        j_src = j - 1 if j > 0 else ny - 1
        f6 = f_src[6, j_src, i_src]
        
        # Direction 7: (-1, -1) - pull from top-right
        i_src = i + 1 if i < nx - 1 else 0
        j_src = j + 1 if j < ny - 1 else 0
        f7 = f_src[7, j_src, i_src]
        
        # Direction 8: (1, -1) - pull from top-left
        i_src = i - 1 if i > 0 else nx - 1
        j_src = j + 1 if j < ny - 1 else 0
        f8 = f_src[8, j_src, i_src]
        
        # Compute macroscopic quantities
        rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8
        
        inv_rho = 1.0 / rho
        ux = (f1 - f3 + f5 - f6 - f7 + f8) * inv_rho
        uy = (f2 - f4 + f5 + f6 - f7 - f8) * inv_rho
        
        # Compute equilibrium (optimized)
        ux_sq = ux * ux
        uy_sq = uy * uy
        u_sq = ux_sq + uy_sq
        u_sq_cs2 = u_sq * 1.5  # u^2 / (2*cs^2) = u^2 * 1.5
        
        # Precompute common terms
        ux_3 = ux * 3.0  # ux / cs^2 = ux * 3
        uy_3 = uy * 3.0
        
        # Direction 0: e=(0,0), w=4/9
        eu = 0.0
        f_eq0 = rho * 0.444444444444444 * (1.0 - u_sq_cs2)
        
        # Direction 1: e=(1,0), w=1/9
        eu = ux_3
        f_eq1 = rho * 0.111111111111111 * (1.0 + eu + 0.5 * eu * eu - u_sq_cs2)
        
        # Direction 2: e=(0,1), w=1/9
        eu = uy_3
        f_eq2 = rho * 0.111111111111111 * (1.0 + eu + 0.5 * eu * eu - u_sq_cs2)
        
        # Direction 3: e=(-1,0), w=1/9
        eu = -ux_3
        f_eq3 = rho * 0.111111111111111 * (1.0 + eu + 0.5 * eu * eu - u_sq_cs2)
        
        # Direction 4: e=(0,-1), w=1/9
        eu = -uy_3
        f_eq4 = rho * 0.111111111111111 * (1.0 + eu + 0.5 * eu * eu - u_sq_cs2)
        
        # Direction 5: e=(1,1), w=1/36
        eu = ux_3 + uy_3
        f_eq5 = rho * 0.027777777777778 * (1.0 + eu + 0.5 * eu * eu - u_sq_cs2)
        
        # Direction 6: e=(-1,1), w=1/36
        eu = -ux_3 + uy_3
        f_eq6 = rho * 0.027777777777778 * (1.0 + eu + 0.5 * eu * eu - u_sq_cs2)
        
        # Direction 7: e=(-1,-1), w=1/36
        eu = -ux_3 - uy_3
        f_eq7 = rho * 0.027777777777778 * (1.0 + eu + 0.5 * eu * eu - u_sq_cs2)
        
        # Direction 8: e=(1,-1), w=1/36
        eu = ux_3 - uy_3
        f_eq8 = rho * 0.027777777777778 * (1.0 + eu + 0.5 * eu * eu - u_sq_cs2)
        
        # BGK collision and store
        one_minus_omega = 1.0 - omega
        f_dst[0, j, i] = one_minus_omega * f0 + omega * f_eq0
        f_dst[1, j, i] = one_minus_omega * f1 + omega * f_eq1
        f_dst[2, j, i] = one_minus_omega * f2 + omega * f_eq2
        f_dst[3, j, i] = one_minus_omega * f3 + omega * f_eq3
        f_dst[4, j, i] = one_minus_omega * f4 + omega * f_eq4
        f_dst[5, j, i] = one_minus_omega * f5 + omega * f_eq5
        f_dst[6, j, i] = one_minus_omega * f6 + omega * f_eq6
        f_dst[7, j, i] = one_minus_omega * f7 + omega * f_eq7
        f_dst[8, j, i] = one_minus_omega * f8 + omega * f_eq8


# =============================================================================
# Shared Memory Kernel for Better Cache Utilization
# =============================================================================

# Block size for shared memory kernel (must be compile-time constant)
BLOCK_X = 32
BLOCK_Y = 8
HALO = 1  # One cell halo for neighbor access

@cuda.jit(fastmath=True)
def collide_stream_shared(f_src, f_dst, omega, nx, ny):
    """
    Collision-streaming with shared memory optimization.
    
    Uses shared memory to cache neighbor values, reducing global memory access.
    
    Parameters
    ----------
    f_src : device array, shape (9, ny, nx)
        Source distribution
    f_dst : device array, shape (9, ny, nx)
        Destination distribution
    omega : float
        Relaxation frequency
    nx, ny : int
        Grid dimensions
    """
    # Shared memory for tile with halos
    # Shape: (9, BLOCK_Y + 2*HALO, BLOCK_X + 2*HALO)
    tile = cuda.shared.array((9, BLOCK_Y + 2, BLOCK_X + 2), dtype=float64)
    
    # Thread indices
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    
    # Global indices
    i = cuda.blockIdx.x * BLOCK_X + tx
    j = cuda.blockIdx.y * BLOCK_Y + ty
    
    # Local indices in shared memory (with halo offset)
    li = tx + HALO
    lj = ty + HALO
    
    # Load center data
    if i < nx and j < ny:
        for k in range(9):
            tile[k, lj, li] = f_src[k, j, i]
    
    # Load halo regions
    # Left halo
    if tx == 0 and i < nx and j < ny:
        i_halo = i - 1 if i > 0 else nx - 1
        for k in range(9):
            tile[k, lj, 0] = f_src[k, j, i_halo]
    
    # Right halo
    if tx == BLOCK_X - 1 and i < nx and j < ny:
        i_halo = i + 1 if i < nx - 1 else 0
        for k in range(9):
            tile[k, lj, BLOCK_X + 1] = f_src[k, j, i_halo]
    
    # Bottom halo
    if ty == 0 and i < nx and j < ny:
        j_halo = j - 1 if j > 0 else ny - 1
        for k in range(9):
            tile[k, 0, li] = f_src[k, j_halo, i]
    
    # Top halo
    if ty == BLOCK_Y - 1 and i < nx and j < ny:
        j_halo = j + 1 if j < ny - 1 else 0
        for k in range(9):
            tile[k, BLOCK_Y + 1, li] = f_src[k, j_halo, i]
    
    # Corner halos
    if tx == 0 and ty == 0 and i < nx and j < ny:
        i_h = i - 1 if i > 0 else nx - 1
        j_h = j - 1 if j > 0 else ny - 1
        for k in range(9):
            tile[k, 0, 0] = f_src[k, j_h, i_h]
    
    if tx == BLOCK_X - 1 and ty == 0 and i < nx and j < ny:
        i_h = i + 1 if i < nx - 1 else 0
        j_h = j - 1 if j > 0 else ny - 1
        for k in range(9):
            tile[k, 0, BLOCK_X + 1] = f_src[k, j_h, i_h]
    
    if tx == 0 and ty == BLOCK_Y - 1 and i < nx and j < ny:
        i_h = i - 1 if i > 0 else nx - 1
        j_h = j + 1 if j < ny - 1 else 0
        for k in range(9):
            tile[k, BLOCK_Y + 1, 0] = f_src[k, j_h, i_h]
    
    if tx == BLOCK_X - 1 and ty == BLOCK_Y - 1 and i < nx and j < ny:
        i_h = i + 1 if i < nx - 1 else 0
        j_h = j + 1 if j < ny - 1 else 0
        for k in range(9):
            tile[k, BLOCK_Y + 1, BLOCK_X + 1] = f_src[k, j_h, i_h]
    
    # Synchronize to ensure all data is loaded
    cuda.syncthreads()
    
    # Compute only for valid cells
    if i < nx and j < ny:
        # Pull from shared memory (unrolled)
        f0 = tile[0, lj, li]
        f1 = tile[1, lj, li - 1]
        f2 = tile[2, lj - 1, li]
        f3 = tile[3, lj, li + 1]
        f4 = tile[4, lj + 1, li]
        f5 = tile[5, lj - 1, li - 1]
        f6 = tile[6, lj - 1, li + 1]
        f7 = tile[7, lj + 1, li + 1]
        f8 = tile[8, lj + 1, li - 1]
        
        # Compute macroscopic quantities
        rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8
        inv_rho = 1.0 / rho
        ux = (f1 - f3 + f5 - f6 - f7 + f8) * inv_rho
        uy = (f2 - f4 + f5 + f6 - f7 - f8) * inv_rho
        
        # Compute equilibrium
        u_sq_cs2 = (ux * ux + uy * uy) * 1.5
        ux_3 = ux * 3.0
        uy_3 = uy * 3.0
        
        f_eq0 = rho * 0.444444444444444 * (1.0 - u_sq_cs2)
        
        eu = ux_3
        f_eq1 = rho * 0.111111111111111 * (1.0 + eu + 0.5 * eu * eu - u_sq_cs2)
        
        eu = uy_3
        f_eq2 = rho * 0.111111111111111 * (1.0 + eu + 0.5 * eu * eu - u_sq_cs2)
        
        eu = -ux_3
        f_eq3 = rho * 0.111111111111111 * (1.0 + eu + 0.5 * eu * eu - u_sq_cs2)
        
        eu = -uy_3
        f_eq4 = rho * 0.111111111111111 * (1.0 + eu + 0.5 * eu * eu - u_sq_cs2)
        
        eu = ux_3 + uy_3
        f_eq5 = rho * 0.027777777777778 * (1.0 + eu + 0.5 * eu * eu - u_sq_cs2)
        
        eu = -ux_3 + uy_3
        f_eq6 = rho * 0.027777777777778 * (1.0 + eu + 0.5 * eu * eu - u_sq_cs2)
        
        eu = -ux_3 - uy_3
        f_eq7 = rho * 0.027777777777778 * (1.0 + eu + 0.5 * eu * eu - u_sq_cs2)
        
        eu = ux_3 - uy_3
        f_eq8 = rho * 0.027777777777778 * (1.0 + eu + 0.5 * eu * eu - u_sq_cs2)
        
        # BGK collision and store to global memory
        one_minus_omega = 1.0 - omega
        f_dst[0, j, i] = one_minus_omega * f0 + omega * f_eq0
        f_dst[1, j, i] = one_minus_omega * f1 + omega * f_eq1
        f_dst[2, j, i] = one_minus_omega * f2 + omega * f_eq2
        f_dst[3, j, i] = one_minus_omega * f3 + omega * f_eq3
        f_dst[4, j, i] = one_minus_omega * f4 + omega * f_eq4
        f_dst[5, j, i] = one_minus_omega * f5 + omega * f_eq5
        f_dst[6, j, i] = one_minus_omega * f6 + omega * f_eq6
        f_dst[7, j, i] = one_minus_omega * f7 + omega * f_eq7
        f_dst[8, j, i] = one_minus_omega * f8 + omega * f_eq8


# =============================================================================
# AA Pattern (In-Place) Kernel - Reduces Memory by Half
# =============================================================================

@cuda.jit(fastmath=True)
def aa_even_step(f, omega, nx, ny):
    """
    AA-pattern even step (local collision only).
    
    The AA pattern alternates between:
    - Even step: collide in place
    - Odd step: collide and swap with neighbors
    
    This reduces memory usage by half (no double buffering needed).
    """
    i, j = cuda.grid(2)
    
    if i < nx and j < ny:
        # Load distributions
        f0 = f[0, j, i]
        f1 = f[1, j, i]
        f2 = f[2, j, i]
        f3 = f[3, j, i]
        f4 = f[4, j, i]
        f5 = f[5, j, i]
        f6 = f[6, j, i]
        f7 = f[7, j, i]
        f8 = f[8, j, i]
        
        # Compute macroscopic
        rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8
        inv_rho = 1.0 / rho
        ux = (f1 - f3 + f5 - f6 - f7 + f8) * inv_rho
        uy = (f2 - f4 + f5 + f6 - f7 - f8) * inv_rho
        
        # Equilibrium
        u_sq_cs2 = (ux * ux + uy * uy) * 1.5
        ux_3 = ux * 3.0
        uy_3 = uy * 3.0
        
        f_eq0 = rho * 0.444444444444444 * (1.0 - u_sq_cs2)
        eu = ux_3
        f_eq1 = rho * 0.111111111111111 * (1.0 + eu + 0.5 * eu * eu - u_sq_cs2)
        eu = uy_3
        f_eq2 = rho * 0.111111111111111 * (1.0 + eu + 0.5 * eu * eu - u_sq_cs2)
        eu = -ux_3
        f_eq3 = rho * 0.111111111111111 * (1.0 + eu + 0.5 * eu * eu - u_sq_cs2)
        eu = -uy_3
        f_eq4 = rho * 0.111111111111111 * (1.0 + eu + 0.5 * eu * eu - u_sq_cs2)
        eu = ux_3 + uy_3
        f_eq5 = rho * 0.027777777777778 * (1.0 + eu + 0.5 * eu * eu - u_sq_cs2)
        eu = -ux_3 + uy_3
        f_eq6 = rho * 0.027777777777778 * (1.0 + eu + 0.5 * eu * eu - u_sq_cs2)
        eu = -ux_3 - uy_3
        f_eq7 = rho * 0.027777777777778 * (1.0 + eu + 0.5 * eu * eu - u_sq_cs2)
        eu = ux_3 - uy_3
        f_eq8 = rho * 0.027777777777778 * (1.0 + eu + 0.5 * eu * eu - u_sq_cs2)
        
        # Collide in place
        one_minus_omega = 1.0 - omega
        f[0, j, i] = one_minus_omega * f0 + omega * f_eq0
        f[1, j, i] = one_minus_omega * f1 + omega * f_eq1
        f[2, j, i] = one_minus_omega * f2 + omega * f_eq2
        f[3, j, i] = one_minus_omega * f3 + omega * f_eq3
        f[4, j, i] = one_minus_omega * f4 + omega * f_eq4
        f[5, j, i] = one_minus_omega * f5 + omega * f_eq5
        f[6, j, i] = one_minus_omega * f6 + omega * f_eq6
        f[7, j, i] = one_minus_omega * f7 + omega * f_eq7
        f[8, j, i] = one_minus_omega * f8 + omega * f_eq8


@cuda.jit(fastmath=True)
def aa_odd_step(f, omega, nx, ny):
    """
    AA-pattern odd step (collision + streaming via neighbor swap).
    
    After collision, swap distributions with neighbors in opposite directions.
    """
    i, j = cuda.grid(2)
    
    if i < nx and j < ny:
        # Load from neighbors (pull)
        f0 = f[0, j, i]
        
        i_src = i - 1 if i > 0 else nx - 1
        f1 = f[1, j, i_src]
        
        j_src = j - 1 if j > 0 else ny - 1
        f2 = f[2, j_src, i]
        
        i_src = i + 1 if i < nx - 1 else 0
        f3 = f[3, j, i_src]
        
        j_src = j + 1 if j < ny - 1 else 0
        f4 = f[4, j_src, i]
        
        i_src = i - 1 if i > 0 else nx - 1
        j_src = j - 1 if j > 0 else ny - 1
        f5 = f[5, j_src, i_src]
        
        i_src = i + 1 if i < nx - 1 else 0
        j_src = j - 1 if j > 0 else ny - 1
        f6 = f[6, j_src, i_src]
        
        i_src = i + 1 if i < nx - 1 else 0
        j_src = j + 1 if j < ny - 1 else 0
        f7 = f[7, j_src, i_src]
        
        i_src = i - 1 if i > 0 else nx - 1
        j_src = j + 1 if j < ny - 1 else 0
        f8 = f[8, j_src, i_src]
        
        # Compute macroscopic
        rho = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8
        inv_rho = 1.0 / rho
        ux = (f1 - f3 + f5 - f6 - f7 + f8) * inv_rho
        uy = (f2 - f4 + f5 + f6 - f7 - f8) * inv_rho
        
        # Equilibrium
        u_sq_cs2 = (ux * ux + uy * uy) * 1.5
        ux_3 = ux * 3.0
        uy_3 = uy * 3.0
        
        f_eq0 = rho * 0.444444444444444 * (1.0 - u_sq_cs2)
        eu = ux_3
        f_eq1 = rho * 0.111111111111111 * (1.0 + eu + 0.5 * eu * eu - u_sq_cs2)
        eu = uy_3
        f_eq2 = rho * 0.111111111111111 * (1.0 + eu + 0.5 * eu * eu - u_sq_cs2)
        eu = -ux_3
        f_eq3 = rho * 0.111111111111111 * (1.0 + eu + 0.5 * eu * eu - u_sq_cs2)
        eu = -uy_3
        f_eq4 = rho * 0.111111111111111 * (1.0 + eu + 0.5 * eu * eu - u_sq_cs2)
        eu = ux_3 + uy_3
        f_eq5 = rho * 0.027777777777778 * (1.0 + eu + 0.5 * eu * eu - u_sq_cs2)
        eu = -ux_3 + uy_3
        f_eq6 = rho * 0.027777777777778 * (1.0 + eu + 0.5 * eu * eu - u_sq_cs2)
        eu = -ux_3 - uy_3
        f_eq7 = rho * 0.027777777777778 * (1.0 + eu + 0.5 * eu * eu - u_sq_cs2)
        eu = ux_3 - uy_3
        f_eq8 = rho * 0.027777777777778 * (1.0 + eu + 0.5 * eu * eu - u_sq_cs2)
        
        # Collide and store (swapped directions for AA pattern)
        one_minus_omega = 1.0 - omega
        f[0, j, i] = one_minus_omega * f0 + omega * f_eq0
        # Swap 1<->3, 2<->4, 5<->7, 6<->8
        f[3, j, i] = one_minus_omega * f1 + omega * f_eq1
        f[4, j, i] = one_minus_omega * f2 + omega * f_eq2
        f[1, j, i] = one_minus_omega * f3 + omega * f_eq3
        f[2, j, i] = one_minus_omega * f4 + omega * f_eq4
        f[7, j, i] = one_minus_omega * f5 + omega * f_eq5
        f[8, j, i] = one_minus_omega * f6 + omega * f_eq6
        f[5, j, i] = one_minus_omega * f7 + omega * f_eq7
        f[6, j, i] = one_minus_omega * f8 + omega * f_eq8


# =============================================================================
# Boundary Condition Kernels (Optimized)
# =============================================================================

@cuda.jit(fastmath=True)
def bounce_back_optimized(f, solid_mask, nx, ny):
    """Optimized bounce-back with unrolled swaps."""
    i, j = cuda.grid(2)
    
    if i < nx and j < ny:
        if solid_mask[j, i]:
            # Load all
            f1 = f[1, j, i]
            f2 = f[2, j, i]
            f3 = f[3, j, i]
            f4 = f[4, j, i]
            f5 = f[5, j, i]
            f6 = f[6, j, i]
            f7 = f[7, j, i]
            f8 = f[8, j, i]
            
            # Swap opposites
            f[1, j, i] = f3
            f[3, j, i] = f1
            f[2, j, i] = f4
            f[4, j, i] = f2
            f[5, j, i] = f7
            f[7, j, i] = f5
            f[6, j, i] = f8
            f[8, j, i] = f6


@cuda.jit(fastmath=True)
def moving_wall_optimized(f, wall_mask, ux_wall, uy_wall, rho_wall, nx, ny):
    """Optimized moving wall BC."""
    i, j = cuda.grid(2)
    
    if i < nx and j < ny:
        if wall_mask[j, i]:
            # Load
            f1 = f[1, j, i]
            f2 = f[2, j, i]
            f3 = f[3, j, i]
            f4 = f[4, j, i]
            f5 = f[5, j, i]
            f6 = f[6, j, i]
            f7 = f[7, j, i]
            f8 = f[8, j, i]
            
            # Precompute corrections
            c = 6.0 * rho_wall  # 2 * rho / cs^2 = 2 * rho * 3 = 6 * rho
            cu_x = c * ux_wall
            cu_y = c * uy_wall
            
            # w=1/9: correction = (1/9) * 6 * rho * (e.u) = (2/3) * rho * (e.u)
            # w=1/36: correction = (1/36) * 6 * rho * (e.u) = (1/6) * rho * (e.u)
            
            # Apply bounce-back with corrections
            f[1, j, i] = f3 + 0.111111111111111 * cu_x
            f[3, j, i] = f1 - 0.111111111111111 * cu_x
            f[2, j, i] = f4 + 0.111111111111111 * cu_y
            f[4, j, i] = f2 - 0.111111111111111 * cu_y
            f[5, j, i] = f7 + 0.027777777777778 * (cu_x + cu_y)
            f[7, j, i] = f5 - 0.027777777777778 * (cu_x + cu_y)
            f[6, j, i] = f8 + 0.027777777777778 * (-cu_x + cu_y)
            f[8, j, i] = f6 - 0.027777777777778 * (-cu_x + cu_y)


@cuda.jit(fastmath=True)
def compute_macroscopic_optimized(f, rho, ux, uy, nx, ny):
    """Optimized macroscopic computation."""
    i, j = cuda.grid(2)
    
    if i < nx and j < ny:
        f0 = f[0, j, i]
        f1 = f[1, j, i]
        f2 = f[2, j, i]
        f3 = f[3, j, i]
        f4 = f[4, j, i]
        f5 = f[5, j, i]
        f6 = f[6, j, i]
        f7 = f[7, j, i]
        f8 = f[8, j, i]
        
        rho_val = f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8
        inv_rho = 1.0 / rho_val
        
        rho[j, i] = rho_val
        ux[j, i] = (f1 - f3 + f5 - f6 - f7 + f8) * inv_rho
        uy[j, i] = (f2 - f4 + f5 + f6 - f7 - f8) * inv_rho


# =============================================================================
# Optimized GPU Solver Class
# =============================================================================

class OptimizedGPULBMSolver:
    """
    Optimized GPU LBM solver with multiple kernel options.
    
    Parameters
    ----------
    nx, ny : int
        Grid dimensions
    tau : float
        Relaxation time
    kernel_type : str
        'fused' - Optimized fused kernel (default)
        'shared' - Shared memory kernel
        'aa' - AA-pattern (in-place, half memory)
    """
    
    def __init__(self, nx, ny, tau, kernel_type='fused'):
        self.nx = nx
        self.ny = ny
        self.tau = tau
        self.omega = 1.0 / tau
        self.kernel_type = kernel_type
        
        # Block/grid configuration
        if kernel_type == 'shared':
            self.block_size = (BLOCK_X, BLOCK_Y)
        else:
            self.block_size = (32, 8)  # Optimized for memory coalescing
        
        self.grid_size = (
            (nx + self.block_size[0] - 1) // self.block_size[0],
            (ny + self.block_size[1] - 1) // self.block_size[1]
        )
        
        # Initialize distribution on host
        from src.equilibrium import compute_equilibrium
        rho_init = np.ones((ny, nx), dtype=np.float64)
        ux_init = np.zeros((ny, nx), dtype=np.float64)
        uy_init = np.zeros((ny, nx), dtype=np.float64)
        f_init = compute_equilibrium(rho_init, ux_init, uy_init)
        
        # Allocate device arrays
        self.d_f = cuda.to_device(f_init)
        if kernel_type != 'aa':
            self.d_f_temp = cuda.device_array_like(f_init)
        
        self.d_rho = cuda.device_array((ny, nx), dtype=np.float64)
        self.d_ux = cuda.device_array((ny, nx), dtype=np.float64)
        self.d_uy = cuda.device_array((ny, nx), dtype=np.float64)
        
        # Boundary masks
        self.d_solid_mask = None
        self.d_lid_mask = None
        self.ux_lid = 0.0
        self.uy_lid = 0.0
        self.rho_lid = 1.0
        
        self.step_count = 0
        self.is_even_step = True
    
    def set_solid_mask(self, solid_mask):
        """Set solid boundary mask."""
        self.d_solid_mask = cuda.to_device(solid_mask.astype(np.bool_))
    
    def set_lid_mask(self, lid_mask, ux_lid, uy_lid=0.0, rho_lid=1.0):
        """Set moving lid boundary."""
        self.d_lid_mask = cuda.to_device(lid_mask.astype(np.bool_))
        self.ux_lid = ux_lid
        self.uy_lid = uy_lid
        self.rho_lid = rho_lid
    
    def step(self):
        """Perform one timestep using selected kernel."""
        if self.kernel_type == 'fused':
            self._step_fused()
        elif self.kernel_type == 'shared':
            self._step_shared()
        elif self.kernel_type == 'aa':
            self._step_aa()
        else:
            self._step_fused()
        
        self.step_count += 1
    
    def _step_fused(self):
        """Step using optimized fused kernel."""
        collide_stream_optimized[self.grid_size, self.block_size](
            self.d_f, self.d_f_temp, self.omega, self.nx, self.ny
        )
        self.d_f, self.d_f_temp = self.d_f_temp, self.d_f
        self._apply_boundary_conditions()
    
    def _step_shared(self):
        """Step using shared memory kernel."""
        collide_stream_shared[self.grid_size, self.block_size](
            self.d_f, self.d_f_temp, self.omega, self.nx, self.ny
        )
        self.d_f, self.d_f_temp = self.d_f_temp, self.d_f
        self._apply_boundary_conditions()
    
    def _step_aa(self):
        """Step using AA-pattern (two half-steps)."""
        if self.is_even_step:
            aa_even_step[self.grid_size, self.block_size](
                self.d_f, self.omega, self.nx, self.ny
            )
        else:
            aa_odd_step[self.grid_size, self.block_size](
                self.d_f, self.omega, self.nx, self.ny
            )
        self.is_even_step = not self.is_even_step
        self._apply_boundary_conditions()
    
    def _apply_boundary_conditions(self):
        """Apply boundary conditions."""
        if self.d_solid_mask is not None:
            bounce_back_optimized[self.grid_size, self.block_size](
                self.d_f, self.d_solid_mask, self.nx, self.ny
            )
        
        if self.d_lid_mask is not None:
            moving_wall_optimized[self.grid_size, self.block_size](
                self.d_f, self.d_lid_mask, self.ux_lid, self.uy_lid,
                self.rho_lid, self.nx, self.ny
            )
    
    def get_macroscopic(self):
        """Get density and velocity fields."""
        compute_macroscopic_optimized[self.grid_size, self.block_size](
            self.d_f, self.d_rho, self.d_ux, self.d_uy, self.nx, self.ny
        )
        return (
            self.d_rho.copy_to_host(),
            self.d_ux.copy_to_host(),
            self.d_uy.copy_to_host()
        )
    
    def get_distribution(self):
        """Get distribution functions."""
        return self.d_f.copy_to_host()
    
    def synchronize(self):
        """Synchronize GPU."""
        cuda.synchronize()
    
    def run(self, num_steps, verbose=True, report_interval=1000):
        """
        Run simulation.
        
        Returns
        -------
        mlups : float
            Performance in Million Lattice Updates Per Second
        """
        import time
        
        # Warmup
        for _ in range(min(100, num_steps // 10)):
            self.step()
        self.synchronize()
        
        # Timed run
        start = time.perf_counter()
        
        for step in range(num_steps):
            self.step()
            
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


def benchmark_optimized(grid_sizes=None, num_steps=1000, kernel_types=None):
    """
    Benchmark optimized kernels.
    
    Parameters
    ----------
    grid_sizes : list of tuple
        Grid sizes to test
    num_steps : int
        Steps per benchmark
    kernel_types : list of str
        Kernel types to test
    
    Returns
    -------
    results : dict
        Results for each kernel type and grid size
    """
    if grid_sizes is None:
        grid_sizes = [
            (256, 256),
            (512, 512),
            (1024, 1024),
            (2048, 2048),
            (4096, 4096),
        ]
    
    if kernel_types is None:
        kernel_types = ['fused', 'shared', 'aa']
    
    tau = 0.8
    results = {}
    
    print("Optimized GPU Kernel Benchmark")
    print("=" * 70)
    
    for kernel_type in kernel_types:
        print(f"\nKernel: {kernel_type}")
        print("-" * 40)
        results[kernel_type] = {}
        
        for nx, ny in grid_sizes:
            try:
                solver = OptimizedGPULBMSolver(nx, ny, tau, kernel_type)
                mlups = solver.run(num_steps, verbose=False)
                results[kernel_type][(nx, ny)] = mlups
                print(f"  {nx:4d} x {ny:4d}: {mlups:8.2f} MLUPS")
            except Exception as e:
                print(f"  {nx:4d} x {ny:4d}: Error - {e}")
                results[kernel_type][(nx, ny)] = 0.0
    
    # Summary
    print("\n" + "=" * 70)
    print("Summary: MLUPS by Kernel Type")
    print("=" * 70)
    
    header = f"{'Grid':<12}"
    for kt in kernel_types:
        header += f" {kt:>12}"
    print(header)
    print("-" * 70)
    
    for nx, ny in grid_sizes:
        row = f"{nx:4d}x{ny:<4d}   "
        for kt in kernel_types:
            mlups = results[kt].get((nx, ny), 0)
            row += f" {mlups:>12.1f}"
        print(row)
    
    return results


if __name__ == "__main__":
    # Run optimized benchmark
    results = benchmark_optimized()