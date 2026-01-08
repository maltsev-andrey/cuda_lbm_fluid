"""
SoA GPU Implementation

Structure of Arrays memory layout for better coalescing.
"""

from numba import cuda
import numpy as np


@cuda.jit
def collision_kernel_soa(f, rho, ux, uy, tau, nx, ny):
    """SoA collision kernel with coalesced memory access."""
    x, y = cuda.grid(2)
    if x < nx and y < ny:
        pass  # TODO: Implement
