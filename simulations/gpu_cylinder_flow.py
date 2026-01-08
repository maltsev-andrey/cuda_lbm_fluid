"""
GPU-Accelerated Cylinder Flow Solver

Complete CUDA implementation including:
- Optimized collision-streaming kernel
- Bounce-back boundary conditions (cylinder)
- Zou-He inlet/outlet boundary conditions
- Control volume force calculation

Target: 2000+ MLUPS for cylinder flow on Tesla P100
"""

import numpy as np
from numba import cuda, float64, boolean
from numba.cuda import shared
import time
import sys
import os

import warnings
from numba.core.errors import NumbaPerformanceWarning  # older numba: numba.errors
warnings.filterwarnings("ignore", category=NumbaPerformanceWarning)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.lattice import EX, EY, W, Q, OPPOSITE
from src.equilibrium import compute_equilibrium_fast
from src.observables import compute_macroscopic_fast, compute_vorticity

# Check CUDA availability
try:
    CUDA_AVAILABLE = cuda.is_available()
except:
    CUDA_AVAILABLE = False


# =============================================================================
# CUDA Kernels for Cylinder Flow
# =============================================================================

@cuda.jit(fastmath=True)
def collide_stream_kernel(f_src, f_dst, omega, solid_mask, nx, ny):
    """
    Fused collision-streaming kernel with bounce-back.
    
    Performs:
    1. Pull distributions from neighbors
    2. Compute macroscopic quantities
    3. Compute equilibrium
    4. BGK collision
    5. Apply bounce-back for solid nodes
    """
    i, j = cuda.grid(2)
    
    if i < nx and j < ny:
        # Check if solid
        if solid_mask[j, i]:
            # Bounce-back: swap opposite directions
            f_dst[0, j, i] = f_src[0, j, i]
            f_dst[1, j, i] = f_src[3, j, i]
            f_dst[2, j, i] = f_src[4, j, i]
            f_dst[3, j, i] = f_src[1, j, i]
            f_dst[4, j, i] = f_src[2, j, i]
            f_dst[5, j, i] = f_src[7, j, i]
            f_dst[6, j, i] = f_src[8, j, i]
            f_dst[7, j, i] = f_src[5, j, i]
            f_dst[8, j, i] = f_src[6, j, i]
            return
        
        # Pull distributions from neighbors (periodic in y, will fix x boundaries later)
        # Direction 0: (0, 0)
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
        
        # Compute equilibrium
        u_sq_cs2 = (ux * ux + uy * uy) * 1.5
        ux_3 = ux * 3.0
        uy_3 = uy * 3.0
        
        # f_eq for each direction
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
        
        # BGK collision
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


@cuda.jit(fastmath=True)
def inlet_bc_kernel(f, u_inlet, ny):
    """
    Zou-He velocity inlet boundary condition at x=0.
    
    Known: f[0], f[2], f[3], f[4], f[6], f[7]
    Unknown: f[1], f[5], f[8]
    """
    j = cuda.grid(1)
    
    if j < ny:
        # Compute density from known distributions
        f0 = f[0, j, 0]
        f2 = f[2, j, 0]
        f3 = f[3, j, 0]
        f4 = f[4, j, 0]
        f6 = f[6, j, 0]
        f7 = f[7, j, 0]
        
        # Check for NaN/Inf and reset if needed
        sum_known = f0 + f2 + f3 + f4 + f6 + f7
        if sum_known != sum_known or sum_known > 1e10 or sum_known < -1e10:
            # Reset to equilibrium
            rho_in = 1.0
            f[0, j, 0] = 0.444444444444444 * rho_in
            f[1, j, 0] = 0.111111111111111 * rho_in * (1.0 + 3.0 * u_inlet)
            f[2, j, 0] = 0.111111111111111 * rho_in
            f[3, j, 0] = 0.111111111111111 * rho_in * (1.0 - 3.0 * u_inlet)
            f[4, j, 0] = 0.111111111111111 * rho_in
            f[5, j, 0] = 0.027777777777778 * rho_in * (1.0 + 3.0 * u_inlet)
            f[6, j, 0] = 0.027777777777778 * rho_in * (1.0 - 3.0 * u_inlet)
            f[7, j, 0] = 0.027777777777778 * rho_in * (1.0 - 3.0 * u_inlet)
            f[8, j, 0] = 0.027777777777778 * rho_in * (1.0 + 3.0 * u_inlet)
            return
        
        rho_in = (f0 + f2 + f4 + 2.0 * (f3 + f6 + f7)) / (1.0 - u_inlet)
        
        # Clamp rho to reasonable values
        if rho_in < 0.5:
            rho_in = 0.5
        elif rho_in > 2.0:
            rho_in = 2.0
        
        # Set unknown distributions
        f[1, j, 0] = f3 + (2.0 / 3.0) * rho_in * u_inlet
        f[5, j, 0] = f7 + (1.0 / 6.0) * rho_in * u_inlet - 0.5 * (f2 - f4)
        f[8, j, 0] = f6 + (1.0 / 6.0) * rho_in * u_inlet + 0.5 * (f2 - f4)


@cuda.jit(fastmath=True)
def outlet_bc_extrapolation_kernel(f, nx, ny):
    """
    Simple extrapolation outlet BC at x=nx-1.
    
    Copies distributions from x=nx-2 (more stable than Zou-He for some cases).
    """
    j = cuda.grid(1)
    
    if j < ny:
        i_out = nx - 1
        i_in = nx - 2
        
        # Copy all distributions from interior
        for k in range(9):
            f[k, j, i_out] = f[k, j, i_in]


@cuda.jit(fastmath=True)
def outlet_bc_kernel(f, rho_outlet, nx, ny):
    """
    Zou-He pressure outlet boundary condition at x=nx-1.
    
    Known: f[0], f[1], f[2], f[4], f[5], f[8]
    Unknown: f[3], f[6], f[7]
    """
    j = cuda.grid(1)
    
    if j < ny:
        i = nx - 1
        
        f0 = f[0, j, i]
        f1 = f[1, j, i]
        f2 = f[2, j, i]
        f4 = f[4, j, i]
        f5 = f[5, j, i]
        f8 = f[8, j, i]
        
        # Check for NaN/Inf
        sum_known = f0 + f1 + f2 + f4 + f5 + f8
        if sum_known != sum_known or sum_known > 1e10 or sum_known < -1e10:
            # Reset to equilibrium with small outflow velocity
            u_out = 0.05
            f[0, j, i] = 0.444444444444444 * rho_outlet
            f[1, j, i] = 0.111111111111111 * rho_outlet * (1.0 + 3.0 * u_out)
            f[2, j, i] = 0.111111111111111 * rho_outlet
            f[3, j, i] = 0.111111111111111 * rho_outlet * (1.0 - 3.0 * u_out)
            f[4, j, i] = 0.111111111111111 * rho_outlet
            f[5, j, i] = 0.027777777777778 * rho_outlet * (1.0 + 3.0 * u_out)
            f[6, j, i] = 0.027777777777778 * rho_outlet * (1.0 - 3.0 * u_out)
            f[7, j, i] = 0.027777777777778 * rho_outlet * (1.0 - 3.0 * u_out)
            f[8, j, i] = 0.027777777777778 * rho_outlet * (1.0 + 3.0 * u_out)
            return
        
        # Compute velocity from known distributions
        u_out = -1.0 + (f0 + f2 + f4 + 2.0 * (f1 + f5 + f8)) / rho_outlet
        
        # Clamp velocity to reasonable values
        if u_out < -0.2:
            u_out = -0.2
        elif u_out > 0.3:
            u_out = 0.3
        
        # Set unknown distributions
        f[3, j, i] = f1 - (2.0 / 3.0) * rho_outlet * u_out
        f[6, j, i] = f8 - (1.0 / 6.0) * rho_outlet * u_out + 0.5 * (f2 - f4)
        f[7, j, i] = f5 - (1.0 / 6.0) * rho_outlet * u_out - 0.5 * (f2 - f4)


@cuda.jit(fastmath=True)
def compute_macroscopic_kernel(f, rho, ux, uy, nx, ny):
    """Compute density and velocity fields."""
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


@cuda.jit(fastmath=True)
def compute_flux_kernel(rho, ux, uy, flux_x, flux_y, col_idx, ny):
    """
    Compute momentum flux at a specific column for control volume force.
    
    flux_x = sum(rho * ux^2 + rho/3)  (momentum + pressure)
    flux_y = sum(rho * ux * uy)
    """
    j = cuda.grid(1)
    
    if j < ny:
        r = rho[j, col_idx]
        u = ux[j, col_idx]
        v = uy[j, col_idx]
        
        # Use atomic add for reduction
        cuda.atomic.add(flux_x, 0, r * u * u + r / 3.0)
        cuda.atomic.add(flux_y, 0, r * u * v)


# =============================================================================
# GPU Cylinder Flow Solver
# =============================================================================

class GPUCylinderFlowSolver:
    """
    GPU-accelerated solver for flow around a cylinder.
    
    All computations run on GPU including boundary conditions.
    
    Parameters
    ----------
    nx, ny : int
        Domain size
    cx, cy : int
        Cylinder center
    radius : int
        Cylinder radius
    re : float
        Reynolds number
    u_inlet : float
        Inlet velocity
    """
    
    def __init__(self, nx, ny, cx, cy, radius, re, u_inlet=0.05):
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA is not available!")
        
        self.nx = nx
        self.ny = ny
        self.cx = cx
        self.cy = cy
        self.radius = radius
        self.diameter = 2 * radius
        self.re = re
        self.u_inlet = u_inlet
        
        # Compute tau from Reynolds number
        self.nu = u_inlet * self.diameter / re
        self.tau = 3.0 * self.nu + 0.5
        self.omega = 1.0 / self.tau
        
        # Stability checks
        if self.tau <= 0.5:
            raise ValueError(f"Unstable: tau = {self.tau:.4f} <= 0.5. Reduce u_inlet or increase domain.")
        
        self.mach = u_inlet * np.sqrt(3)
        if self.mach > 0.3:
            print(f"Warning: Ma = {self.mach:.3f} > 0.3")
        
        # Create cylinder mask on CPU
        y, x = np.ogrid[:ny, :nx]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        self.solid_mask = dist <= radius
        
        # Initialize fields on CPU
        self.rho = np.ones((ny, nx), dtype=np.float64)
        self.ux = np.ones((ny, nx), dtype=np.float64) * u_inlet
        self.uy = np.zeros((ny, nx), dtype=np.float64)
        
        # Zero velocity in solid
        self.ux[self.solid_mask] = 0.0
        self.uy[self.solid_mask] = 0.0
        
        # Small perturbation to trigger vortex shedding
        np.random.seed(42)
        self.uy += 0.001 * u_inlet * np.random.randn(ny, nx)
        self.uy[self.solid_mask] = 0.0
        
        # Initialize distribution
        self.f = compute_equilibrium_fast(self.rho, self.ux, self.uy)
        
        # Transfer to GPU
        self.d_f = cuda.to_device(self.f)
        self.d_f_temp = cuda.device_array_like(self.f)
        self.d_rho = cuda.to_device(self.rho)
        self.d_ux = cuda.to_device(self.ux)
        self.d_uy = cuda.to_device(self.uy)
        self.d_solid_mask = cuda.to_device(self.solid_mask)
        
        # Flux arrays for force calculation (single element for atomic add)
        self.d_flux_x = cuda.to_device(np.zeros(1, dtype=np.float64))
        self.d_flux_y = cuda.to_device(np.zeros(1, dtype=np.float64))
        
        # CUDA grid configuration
        self.block_2d = (32, 8)
        self.grid_2d = (
            (nx + self.block_2d[0] - 1) // self.block_2d[0],
            (ny + self.block_2d[1] - 1) // self.block_2d[1]
        )
        
        self.block_1d = 256
        self.grid_1d = (ny + self.block_1d - 1) // self.block_1d
        
        # Statistics
        self.step_count = 0
        self.force_history = []
        self.lift_history = []
        self.time_history = []
    
    def step(self):
        """Perform one LBM timestep on GPU."""
        # 1. Collision + Streaming + Bounce-back (fused kernel)
        collide_stream_kernel[self.grid_2d, self.block_2d](
            self.d_f, self.d_f_temp, self.omega, self.d_solid_mask, self.nx, self.ny
        )
        
        # Swap buffers
        self.d_f, self.d_f_temp = self.d_f_temp, self.d_f
        
        # 2. Inlet boundary condition (x=0)
        inlet_bc_kernel[self.grid_1d, self.block_1d](
            self.d_f, self.u_inlet, self.ny
        )
        
        # 3. Outlet boundary condition (x=nx-1) - use extrapolation for stability
        outlet_bc_extrapolation_kernel[self.grid_1d, self.block_1d](
            self.d_f, self.nx, self.ny
        )
        
        self.step_count += 1
    
    def compute_forces(self):
        """
        Compute drag and lift using control volume method on GPU.
        
        Returns
        -------
        F_x, F_y : float
            Drag and lift forces
        """
        # First compute macroscopic fields
        compute_macroscopic_kernel[self.grid_2d, self.block_2d](
            self.d_f, self.d_rho, self.d_ux, self.d_uy, self.nx, self.ny
        )
        
        # Inlet flux (column 5)
        self.d_flux_x.copy_to_device(np.zeros(1, dtype=np.float64))
        self.d_flux_y.copy_to_device(np.zeros(1, dtype=np.float64))
        compute_flux_kernel[self.grid_1d, self.block_1d](
            self.d_rho, self.d_ux, self.d_uy, self.d_flux_x, self.d_flux_y, 5, self.ny
        )
        cuda.synchronize()
        flux_in_x = self.d_flux_x.copy_to_host()[0]
        flux_in_y = self.d_flux_y.copy_to_host()[0]
        
        # Outlet flux (column nx-20)
        self.d_flux_x.copy_to_device(np.zeros(1, dtype=np.float64))
        self.d_flux_y.copy_to_device(np.zeros(1, dtype=np.float64))
        compute_flux_kernel[self.grid_1d, self.block_1d](
            self.d_rho, self.d_ux, self.d_uy, self.d_flux_x, self.d_flux_y, self.nx - 20, self.ny
        )
        cuda.synchronize()
        flux_out_x = self.d_flux_x.copy_to_host()[0]
        flux_out_y = self.d_flux_y.copy_to_host()[0]
        
        F_x = flux_in_x - flux_out_x
        F_y = flux_in_y - flux_out_y
        
        return F_x, F_y
    
    def compute_drag_coefficient(self, F_x, apply_blockage_correction=True):
        """Compute drag coefficient with optional blockage correction."""
        C_D = F_x / (0.5 * self.u_inlet**2 * self.diameter)
        
        if apply_blockage_correction:
            blockage = self.diameter / self.ny
            correction = (1 + 0.5 * blockage)**2
            C_D = C_D / correction
        
        return C_D
    
    def compute_lift_coefficient(self, F_y, apply_blockage_correction=True):
        """Compute lift coefficient."""
        C_L = F_y / (0.5 * self.u_inlet**2 * self.diameter)
        
        if apply_blockage_correction:
            blockage = self.diameter / self.ny
            correction = (1 + 0.5 * blockage)**2
            C_L = C_L / correction
        
        return C_L
    
    def get_fields(self):
        """Copy fields from GPU to CPU."""
        compute_macroscopic_kernel[self.grid_2d, self.block_2d](
            self.d_f, self.d_rho, self.d_ux, self.d_uy, self.nx, self.ny
        )
        cuda.synchronize()
        
        self.rho = self.d_rho.copy_to_host()
        self.ux = self.d_ux.copy_to_host()
        self.uy = self.d_uy.copy_to_host()
        
        return {
            'rho': self.rho.copy(),
            'ux': self.ux.copy(),
            'uy': self.uy.copy(),
            'vorticity': compute_vorticity(self.ux, self.uy)
        }
    
    def run(self, num_steps, measure_interval=500, verbose=True):
        """
        Run simulation.
        
        Parameters
        ----------
        num_steps : int
            Number of timesteps
        measure_interval : int
            Steps between force measurements
        verbose : bool
            Print progress
        
        Returns
        -------
        results : dict
            C_D, C_L_rms, MLUPS
        """
        # Warmup
        for _ in range(100):
            self.step()
        cuda.synchronize()
        
        start_time = time.perf_counter()
        
        for step in range(num_steps):
            self.step()
            
            if (step + 1) % measure_interval == 0:
                F_x, F_y = self.compute_forces()
                self.force_history.append(F_x)
                self.lift_history.append(F_y)
                self.time_history.append(self.step_count)
                
                if verbose and (step + 1) % (measure_interval * 10) == 0:
                    C_D = self.compute_drag_coefficient(F_x)
                    C_L = self.compute_lift_coefficient(F_y)
                    print(f"Step {step + 1}: C_D = {C_D:.4f}, C_L = {C_L:.4f}")
        
        cuda.synchronize()
        elapsed = time.perf_counter() - start_time
        mlups = num_steps * self.nx * self.ny / elapsed / 1e6
        
        if verbose:
            print(f"\nCompleted {num_steps} steps in {elapsed:.2f}s")
            print(f"Performance: {mlups:.2f} MLUPS")
        
        return self.analyze_results(mlups)
    
    def analyze_results(self, mlups=0):
        """Analyze force history."""
        if len(self.force_history) < 10:
            return {'MLUPS': mlups}
        
        n_half = len(self.force_history) // 2
        forces = np.array(self.force_history[n_half:])
        lifts = np.array(self.lift_history[n_half:])
        
        blockage = self.diameter / self.ny
        correction = (1 + 0.5 * blockage)**2
        
        C_D_values = forces / (0.5 * self.u_inlet**2 * self.diameter) / correction
        C_L_values = lifts / (0.5 * self.u_inlet**2 * self.diameter) / correction
        
        return {
            'C_D': np.mean(C_D_values),
            'C_D_std': np.std(C_D_values),
            'C_L_rms': np.sqrt(np.mean(C_L_values**2)),
            'Re': self.re,
            'tau': self.tau,
            'MLUPS': mlups
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def run_gpu_cylinder_flow(nx=600, ny=150, re=100, num_steps=40000, verbose=True):
    """
    Run GPU-accelerated cylinder flow simulation.
    
    Parameters
    ----------
    nx, ny : int
        Domain size
    re : float
        Reynolds number
    num_steps : int
        Simulation steps
    verbose : bool
        Print progress
    
    Returns
    -------
    solver : GPUCylinderFlowSolver
        Solver with results
    """
    # Cylinder at 20% from inlet, centered vertically
    cx = nx // 5
    cy = ny // 2
    radius = ny // 10
    diameter = 2 * radius
    
    # Calculate stable u_inlet
    # Target tau ~ 0.55 for stability
    target_tau = 0.55
    nu = (target_tau - 0.5) / 3.0
    u_inlet = nu * re / diameter
    
    # Check Mach number
    mach = u_inlet * np.sqrt(3)
    if mach > 0.1:
        u_inlet = 0.1 / np.sqrt(3)
        nu = u_inlet * diameter / re
        target_tau = 3 * nu + 0.5
    
    if verbose:
        print("GPU Cylinder Flow Simulation")
        print("=" * 50)
        print(f"Domain: {nx} x {ny}")
        print(f"Cylinder: center=({cx}, {cy}), r={radius}, D={diameter}")
        print(f"Reynolds: {re}")
        print(f"Blockage: {diameter/ny:.1%}")
        print()
    
    solver = GPUCylinderFlowSolver(nx, ny, cx, cy, radius, re, u_inlet)
    
    if verbose:
        print(f"Tau: {solver.tau:.4f}")
        print(f"Nu: {solver.nu:.6f}")
        print(f"Ma: {solver.mach:.4f}")
        print(f"U_inlet: {solver.u_inlet:.6f}")
        print()
    
    results = solver.run(num_steps, measure_interval=500, verbose=verbose)
    
    if verbose:
        print(f"\nResults:")
        print(f"  C_D = {results.get('C_D', 0):.4f} +/- {results.get('C_D_std', 0):.4f}")
        print(f"  C_L_rms = {results.get('C_L_rms', 0):.4f}")
        print(f"  Performance: {results.get('MLUPS', 0):.1f} MLUPS")
        
        # Validation
        if re == 100:
            expected = "1.3-1.5"
        elif re == 40:
            expected = "1.5-1.8"
        else:
            expected = "N/A"
        print(f"  Expected C_D for Re={re}: {expected}")
    
    return solver


def benchmark_gpu_cylinder(grid_sizes=None, re=40, num_steps=10000):
    """
    Benchmark GPU cylinder solver across grid sizes.
    
    Parameters
    ----------
    grid_sizes : list of tuple
        (nx, ny) grid sizes to test
    re : float
        Reynolds number
    num_steps : int
        Steps per benchmark
    
    Returns
    -------
    results : dict
        MLUPS for each grid size
    """
    if grid_sizes is None:
        grid_sizes = [
            (300, 80),
            (600, 150),
            (1200, 300),
            (2400, 600),
        ]
    
    print("GPU Cylinder Flow Benchmark")
    print("=" * 60)
    print(f"Re: {re}, Steps: {num_steps}")
    print()
    
    results = {}
    
    for nx, ny in grid_sizes:
        print(f"Grid: {nx} x {ny}...", end=" ", flush=True)
        
        try:
            cx, cy, r = nx // 5, ny // 2, ny // 10
            solver = GPUCylinderFlowSolver(nx, ny, cx, cy, r, re=re, u_inlet=0.04)
            
            # Warmup
            for _ in range(100):
                solver.step()
            cuda.synchronize()
            
            # Timed run
            start = time.perf_counter()
            for _ in range(num_steps):
                solver.step()
            cuda.synchronize()
            elapsed = time.perf_counter() - start
            
            mlups = num_steps * nx * ny / elapsed / 1e6
            results[(nx, ny)] = mlups
            print(f"{mlups:.1f} MLUPS")
            
        except Exception as e:
            print(f"Error: {e}")
            results[(nx, ny)] = 0
    
    print()
    print("=" * 60)
    print("Summary:")
    for (nx, ny), mlups in results.items():
        print(f"  {nx:4d} x {ny:4d}: {mlups:8.1f} MLUPS")
    
    return results


if __name__ == "__main__":
    if not CUDA_AVAILABLE:
        print("CUDA not available!")
        exit(1)
    
    # Run Re=100 simulation
    solver = run_gpu_cylinder_flow(nx=600, ny=150, re=100, num_steps=40000)
    
    # Benchmark
    print("\n\n")
    benchmark_gpu_cylinder()