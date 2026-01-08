"""
Flow Around Cylinder Simulation

Classic benchmark case demonstrating:
- Vortex shedding (Karman vortex street) at Re > 47
- Drag coefficient validation against experimental data
- Strouhal number measurement

Reference Data:
- Drag coefficient: C_D ~ 1.0-1.5 for Re = 20-100
- Critical Re for vortex shedding: Re_crit ~ 47
- Strouhal number: St ~ 0.21 for Re = 100-1000

Physical setup:
- Cylinder centered in channel
- Uniform inlet velocity
- Pressure outlet
- Periodic top/bottom (or no-slip walls)
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.lattice import EX, EY, W, CS2, Q, OPPOSITE
from src.equilibrium import compute_equilibrium, compute_equilibrium_fast
from src.observables import (
    compute_macroscopic, compute_macroscopic_fast, 
    compute_vorticity, compute_velocity_magnitude
)
from src.collision import bgk_collision_fast, tau_from_viscosity, viscosity_from_tau
from src.streaming import stream_periodic_fast
from src.boundary import create_cylinder_mask

# Try to import GPU solver
try:
    from numba import cuda
    CUDA_AVAILABLE = cuda.is_available()
    if CUDA_AVAILABLE:
        from src.kernels.gpu_optimized import (
            collide_stream_optimized, 
            bounce_back_optimized,
            compute_macroscopic_optimized
        )
except:
    CUDA_AVAILABLE = False


class CylinderFlowSolver:
    """
    Solver for flow around a circular cylinder.
    
    Parameters
    ----------
    nx, ny : int
        Domain size (nx = length, ny = height)
    cylinder_x, cylinder_y : float
        Cylinder center position
    cylinder_r : float
        Cylinder radius
    re : float
        Reynolds number based on cylinder diameter
    u_inlet : float
        Inlet velocity (lattice units)
    use_gpu : bool
        Use GPU acceleration if available
    """
    
    def __init__(self, nx, ny, cylinder_x, cylinder_y, cylinder_r, 
                 re, u_inlet=0.05, use_gpu=True):
        self.nx = nx
        self.ny = ny
        self.cx = cylinder_x
        self.cy = cylinder_y
        self.radius = cylinder_r
        self.diameter = 2 * cylinder_r
        self.re = re
        self.u_inlet = u_inlet
        self.use_gpu = use_gpu and CUDA_AVAILABLE
        
        # Compute viscosity and tau from Reynolds number
        # Re = U * D / nu => nu = U * D / Re
        self.nu = u_inlet * self.diameter / re
        self.tau = tau_from_viscosity(self.nu)
        
        # Stability check
        if self.tau <= 0.5:
            raise ValueError(f"Unstable: tau = {self.tau:.4f} <= 0.5")
        if self.tau < 0.52:
            print(f"Warning: tau = {self.tau:.4f} close to stability limit")
        
        # Mach number check
        self.mach = u_inlet * np.sqrt(3)
        if self.mach > 0.3:
            print(f"Warning: Ma = {self.mach:.3f} > 0.3, compressibility effects")
        
        # Create cylinder mask
        self.solid_mask = create_cylinder_mask(nx, ny, cylinder_x, cylinder_y, cylinder_r)
        
        # Initialize fields
        self.rho = np.ones((ny, nx), dtype=np.float64)
        self.ux = np.ones((ny, nx), dtype=np.float64) * u_inlet
        self.uy = np.zeros((ny, nx), dtype=np.float64)
        
        # Set zero velocity inside cylinder
        self.ux[self.solid_mask] = 0.0
        self.uy[self.solid_mask] = 0.0
        
        # Add small perturbation to trigger vortex shedding
        # This breaks the symmetry and allows Karman vortex street to develop
        np.random.seed(42)
        perturbation = 0.001 * u_inlet * np.random.randn(ny, nx)
        self.uy += perturbation
        self.uy[self.solid_mask] = 0.0
        
        # Initialize distribution
        self.f = compute_equilibrium_fast(self.rho, self.ux, self.uy)
        
        # GPU setup
        if self.use_gpu:
            self._setup_gpu()
        
        # Statistics
        self.step_count = 0
        self.time_physical = 0.0
        
        # Force history for drag computation
        self.force_history = []
        self.lift_history = []
        self.time_history = []
    
    def _setup_gpu(self):
        """Initialize GPU arrays."""
        self.d_f = cuda.to_device(self.f)
        self.d_f_temp = cuda.device_array_like(self.f)
        self.d_rho = cuda.to_device(self.rho)
        self.d_ux = cuda.to_device(self.ux)
        self.d_uy = cuda.to_device(self.uy)
        self.d_solid_mask = cuda.to_device(self.solid_mask)
        
        # Block/grid configuration
        self.block_size = (32, 8)
        self.grid_size = (
            (self.nx + 31) // 32,
            (self.ny + 7) // 8
        )
    
    def _apply_inlet_bc(self, f):
        """Apply velocity inlet BC (Zou-He) at x=0."""
        u_in = self.u_inlet
        
        # Compute density from known distributions
        rho_in = (f[0, :, 0] + f[2, :, 0] + f[4, :, 0] + 
                  2.0 * (f[3, :, 0] + f[6, :, 0] + f[7, :, 0])) / (1.0 - u_in)
        
        # Set unknown distributions (1, 5, 8)
        f[1, :, 0] = f[3, :, 0] + (2.0/3.0) * rho_in * u_in
        f[5, :, 0] = f[7, :, 0] + (1.0/6.0) * rho_in * u_in - 0.5 * (f[2, :, 0] - f[4, :, 0])
        f[8, :, 0] = f[6, :, 0] + (1.0/6.0) * rho_in * u_in + 0.5 * (f[2, :, 0] - f[4, :, 0])
        
        return f
    
    def _apply_outlet_bc(self, f):
        """Apply pressure outlet BC (Zou-He) at x=nx-1."""
        rho_out = 1.0
        
        # Compute velocity from known distributions
        u_out = -1.0 + (f[0, :, -1] + f[2, :, -1] + f[4, :, -1] + 
                        2.0 * (f[1, :, -1] + f[5, :, -1] + f[8, :, -1])) / rho_out
        
        # Set unknown distributions (3, 6, 7)
        f[3, :, -1] = f[1, :, -1] - (2.0/3.0) * rho_out * u_out
        f[6, :, -1] = f[8, :, -1] - (1.0/6.0) * rho_out * u_out + 0.5 * (f[2, :, -1] - f[4, :, -1])
        f[7, :, -1] = f[5, :, -1] - (1.0/6.0) * rho_out * u_out - 0.5 * (f[2, :, -1] - f[4, :, -1])
        
        return f
    
    def _apply_bounce_back(self, f):
        """Apply bounce-back on cylinder surface."""
        for i in range(Q):
            i_opp = OPPOSITE[i]
            f[i, self.solid_mask] = f[i_opp, self.solid_mask]
        return f
    
    def compute_forces_momentum_exchange(self, debug=False):
        """
        Compute drag and lift using control volume method.
        
        This method is more reliable than momentum exchange for bounded flows.
        F_drag = momentum_flux_in - momentum_flux_out (at steady state)
        
        Returns
        -------
        F_x, F_y : float
            Drag and lift forces
        """
        # Make sure we have current CPU data
        if self.use_gpu:
            self.f = self.d_f.copy_to_host()
            self.rho, self.ux, self.uy = compute_macroscopic_fast(self.f)
        
        # Control volume: inlet at x=5, outlet at x=nx-20
        inlet_x = 5
        outlet_x = self.nx - 20
        
        # x-momentum flux: rho*u*u + p, where p = rho/3 (LBM pressure)
        flux_in_x = np.sum(self.rho[:, inlet_x] * self.ux[:, inlet_x]**2) + np.sum(self.rho[:, inlet_x]) / 3.0
        flux_out_x = np.sum(self.rho[:, outlet_x] * self.ux[:, outlet_x]**2) + np.sum(self.rho[:, outlet_x]) / 3.0
        
        F_x = flux_in_x - flux_out_x
        
        # y-momentum flux for lift
        flux_in_y = np.sum(self.rho[:, inlet_x] * self.ux[:, inlet_x] * self.uy[:, inlet_x])
        flux_out_y = np.sum(self.rho[:, outlet_x] * self.ux[:, outlet_x] * self.uy[:, outlet_x])
        
        F_y = flux_in_y - flux_out_y
        
        if debug:
            print(f"Control volume force: F_x={F_x:.6f}, F_y={F_y:.6f}")
        
        return F_x, F_y
    
    def compute_drag_coefficient(self, F_x, apply_blockage_correction=True):
        """
        C_D = F_x / (0.5 * rho * U^2 * D)
        
        With optional blockage correction for bounded channels.
        """
        C_D = F_x / (0.5 * self.u_inlet**2 * self.diameter)
        
        if apply_blockage_correction:
            blockage = self.diameter / self.ny
            correction = (1 + 0.5 * blockage)**2
            C_D = C_D / correction
        
        return C_D
    
    def compute_lift_coefficient(self, F_y):
        """C_L = F_y / (0.5 * rho * U^2 * D)"""
        return F_y / (0.5 * self.u_inlet**2 * self.diameter)
    
    def step_cpu(self):
        """One timestep on CPU."""
        # Collision
        self.rho, self.ux, self.uy = compute_macroscopic_fast(self.f)
        f_eq = compute_equilibrium_fast(self.rho, self.ux, self.uy)
        self.f = bgk_collision_fast(self.f, f_eq, self.tau)
        
        # Streaming (with periodic in y, open in x)
        # We use periodic streaming but fix boundaries after
        self.f = stream_periodic_fast(self.f)
        
        # Apply boundary conditions in correct order:
        # 1. Inlet BC (sets distributions entering from left)
        self.f = self._apply_inlet_bc(self.f)
        
        # 2. Outlet BC (sets distributions entering from right)  
        self.f = self._apply_outlet_bc(self.f)
        
        # 3. Cylinder bounce-back (must be after streaming)
        self.f = self._apply_bounce_back(self.f)
        
        # Update macroscopic fields
        self.rho, self.ux, self.uy = compute_macroscopic_fast(self.f)
        
        self.step_count += 1
        self.time_physical += 1.0
    
    def step_gpu(self):
        """One timestep on GPU with CPU boundary conditions."""
        # For cylinder flow with inlet/outlet, we need careful BC handling
        # Use CPU for now to ensure stability
        self.step_cpu()
        return
        
        # GPU version (for periodic-only cases)
        # omega = 1.0 / self.tau
        # collide_stream_optimized[self.grid_size, self.block_size](
        #     self.d_f, self.d_f_temp, omega, self.nx, self.ny
        # )
        # self.d_f, self.d_f_temp = self.d_f_temp, self.d_f
        # bounce_back_optimized[self.grid_size, self.block_size](
        #     self.d_f, self.d_solid_mask, self.nx, self.ny
        # )
        # self.f = self.d_f.copy_to_host()
        # self.f = self._apply_inlet_bc(self.f)
        # self.f = self._apply_outlet_bc(self.f)
        # self.d_f = cuda.to_device(self.f)
        # self.step_count += 1
        # self.time_physical += 1.0
    
    def step(self):
        """Perform one timestep."""
        if self.use_gpu:
            self.step_gpu()
        else:
            self.step_cpu()
    
    def run(self, num_steps, measure_interval=100, verbose=True):
        """
        Run simulation.
        
        Returns
        -------
        results : dict
            C_D, C_L_rms, St
        """
        start_time = time.perf_counter()
        
        for step in range(num_steps):
            self.step()
            
            if (step + 1) % measure_interval == 0:
                if self.use_gpu:
                    self.f = self.d_f.copy_to_host()
                    self.rho, self.ux, self.uy = compute_macroscopic_fast(self.f)
                
                F_x, F_y = self.compute_forces_momentum_exchange()
                self.force_history.append(F_x)
                self.lift_history.append(F_y)
                self.time_history.append(self.time_physical)
                
                if verbose and (step + 1) % (measure_interval * 10) == 0:
                    C_D = self.compute_drag_coefficient(F_x)
                    C_L = self.compute_lift_coefficient(F_y)
                    print(f"Step {step + 1}: C_D = {C_D:.4f}, C_L = {C_L:.4f}")
        
        elapsed = time.perf_counter() - start_time
        mlups = num_steps * self.nx * self.ny / elapsed / 1e6
        
        if verbose:
            print(f"\nCompleted {num_steps} steps in {elapsed:.2f}s")
            print(f"Performance: {mlups:.2f} MLUPS")
        
        return self.analyze_results()
    
    def analyze_results(self):
        """Analyze force history for C_D, C_L, St."""
        if len(self.force_history) < 10:
            return {}
        
        n_half = len(self.force_history) // 2
        forces = np.array(self.force_history[n_half:])
        lifts = np.array(self.lift_history[n_half:])
        times = np.array(self.time_history[n_half:])
        
        # Apply blockage correction
        blockage = self.diameter / self.ny
        correction = (1 + 0.5 * blockage)**2
        
        C_D_values = forces / (0.5 * self.u_inlet**2 * self.diameter) / correction
        C_L_values = lifts / (0.5 * self.u_inlet**2 * self.diameter) / correction
        
        C_D_mean = np.mean(C_D_values)
        C_D_std = np.std(C_D_values)
        C_L_rms = np.sqrt(np.mean(C_L_values**2))
        
        # Strouhal number from FFT
        St = None
        if len(lifts) > 20:
            dt = times[1] - times[0] if len(times) > 1 else 1.0
            fft = np.fft.rfft(C_L_values - np.mean(C_L_values))
            freqs = np.fft.rfftfreq(len(C_L_values), d=dt)
            
            fft_mag = np.abs(fft[1:])
            if len(fft_mag) > 0 and np.max(fft_mag) > 0.01:
                peak_idx = np.argmax(fft_mag) + 1
                f_shed = freqs[peak_idx]
                St = f_shed * self.diameter / self.u_inlet
        
        return {
            'C_D': C_D_mean,
            'C_D_std': C_D_std,
            'C_L_rms': C_L_rms,
            'St': St,
            'Re': self.re,
            'tau': self.tau,
            'nu': self.nu
        }
    
    def get_fields(self):
        """Get velocity and vorticity fields."""
        if self.use_gpu:
            self.f = self.d_f.copy_to_host()
            self.rho, self.ux, self.uy = compute_macroscopic_fast(self.f)
        
        return {
            'rho': self.rho.copy(),
            'ux': self.ux.copy(),
            'uy': self.uy.copy(),
            'vorticity': compute_vorticity(self.ux, self.uy),
            'velocity_mag': compute_velocity_magnitude(self.ux, self.uy)
        }
    
    def plot_results(self, save_path=None):
        """Plot velocity, vorticity, and force history."""
        fields = self.get_fields()
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Velocity magnitude
        ax = axes[0, 0]
        vm = fields['velocity_mag'].copy()
        vm[self.solid_mask] = np.nan
        im = ax.imshow(vm, origin='lower', cmap='viridis', aspect='equal')
        ax.set_title(f'Velocity Magnitude (Re = {self.re})')
        plt.colorbar(im, ax=ax, label='|u|')
        circle = plt.Circle((self.cx, self.cy), self.radius, color='white', fill=True)
        ax.add_patch(circle)
        
        # Vorticity
        ax = axes[0, 1]
        vort = fields['vorticity'].copy()
        vort[self.solid_mask] = np.nan
        vmax = np.nanpercentile(np.abs(vort), 95)
        im = ax.imshow(vort, origin='lower', cmap='RdBu_r', vmin=-vmax, vmax=vmax, aspect='equal')
        ax.set_title('Vorticity (Karman vortex street)')
        plt.colorbar(im, ax=ax, label='omega')
        circle = plt.Circle((self.cx, self.cy), self.radius, color='gray', fill=True)
        ax.add_patch(circle)
        
        # Drag coefficient
        ax = axes[1, 0]
        if len(self.force_history) > 0:
            C_D_hist = np.array(self.force_history) / (0.5 * self.u_inlet**2 * self.diameter)
            ax.plot(self.time_history, C_D_hist, 'b-', linewidth=1)
            mean_cd = np.mean(C_D_hist[len(C_D_hist)//2:])
            ax.axhline(y=mean_cd, color='r', linestyle='--', label=f'Mean = {mean_cd:.3f}')
            ax.set_xlabel('Time')
            ax.set_ylabel('C_D')
            ax.set_title('Drag Coefficient')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Lift coefficient
        ax = axes[1, 1]
        if len(self.lift_history) > 0:
            C_L_hist = np.array(self.lift_history) / (0.5 * self.u_inlet**2 * self.diameter)
            ax.plot(self.time_history, C_L_hist, 'g-', linewidth=1)
            ax.set_xlabel('Time')
            ax.set_ylabel('C_L')
            ax.set_title('Lift Coefficient (Vortex Shedding)')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        return fig


# Reference data
DRAG_COEFFICIENT_REFERENCE = {
    20: (2.0, 2.5),
    40: (1.5, 1.8),
    100: (1.3, 1.5),
    200: (1.2, 1.4),
}


def run_cylinder_flow(nx=400, ny=100, re=100, num_steps=50000, use_gpu=True, verbose=True):
    """Run cylinder flow simulation."""
    cylinder_r = ny // 10
    cylinder_x = nx // 5
    cylinder_y = ny // 2
    
    diameter = 2 * cylinder_r
    
    # For stability, we need tau > 0.55
    # tau = 3*nu + 0.5, so nu = (tau - 0.5)/3
    # Re = U*D/nu, so U = Re*nu/D
    
    # Target tau = 0.7 for good stability
    target_tau = 0.7
    nu = (target_tau - 0.5) / 3.0
    u_inlet = nu * re / diameter
    
    # Check Mach number
    mach = u_inlet * np.sqrt(3)
    
    # If Mach > 0.1, reduce velocity and adjust
    if mach > 0.1:
        u_inlet = 0.1 / np.sqrt(3)  # Ma = 0.1
        nu = u_inlet * diameter / re
        target_tau = 3 * nu + 0.5
    
    if verbose:
        print("Cylinder Flow Simulation")
        print("=" * 50)
        print(f"Domain: {nx} x {ny}")
        print(f"Cylinder: center=({cylinder_x}, {cylinder_y}), r={cylinder_r}")
        print(f"Diameter: {diameter}")
        print(f"Reynolds: {re}")
        print(f"GPU: {use_gpu and CUDA_AVAILABLE}")
    
    solver = CylinderFlowSolver(
        nx, ny, cylinder_x, cylinder_y, cylinder_r,
        re=re, u_inlet=u_inlet, use_gpu=use_gpu
    )
    
    if verbose:
        print(f"Tau: {solver.tau:.4f}, Nu: {solver.nu:.6f}, Ma: {solver.mach:.4f}")
        print(f"U_inlet: {solver.u_inlet:.6f}")
        print()
    
    results = solver.run(num_steps, measure_interval=100, verbose=verbose)
    
    if verbose:
        print(f"\nResults:")
        cd = results.get('C_D', 0)
        cd_std = results.get('C_D_std', 0)
        cl_rms = results.get('C_L_rms', 0)
        st = results.get('St')
        
        if not np.isnan(cd):
            print(f"  C_D = {cd:.4f} +/- {cd_std:.4f}")
            print(f"  C_L_rms = {cl_rms:.4f}")
            if st and not np.isnan(st):
                print(f"  St = {st:.4f}")
        else:
            print("  Simulation may have diverged")
        
        if re in DRAG_COEFFICIENT_REFERENCE:
            cd_min, cd_max = DRAG_COEFFICIENT_REFERENCE[re]
            if not np.isnan(cd):
                status = "PASS" if cd_min <= cd <= cd_max else "CHECK"
                print(f"  Reference: C_D = {cd_min}-{cd_max} [{status}]")
    
    return solver


def validation_study(reynolds_numbers=None, use_gpu=True):
    """Run validation across multiple Re."""
    if reynolds_numbers is None:
        reynolds_numbers = [20, 40, 100]
    
    print("=" * 60)
    print("Cylinder Flow Validation Study")
    print("=" * 60)
    
    all_results = {}
    
    for re in reynolds_numbers:
        print(f"\n--- Re = {re} ---")
        
        if re <= 40:
            nx, ny, steps = 300, 80, 30000
        else:
            nx, ny, steps = 400, 100, 50000
        
        solver = run_cylinder_flow(nx, ny, re, steps, use_gpu, verbose=False)
        results = solver.analyze_results()
        all_results[re] = results
        
        cd = results.get('C_D', 0)
        print(f"  C_D = {cd:.4f}")
        
        if re in DRAG_COEFFICIENT_REFERENCE:
            cd_min, cd_max = DRAG_COEFFICIENT_REFERENCE[re]
            status = "PASS" if cd_min <= cd <= cd_max else "CHECK"
            print(f"  Reference: {cd_min}-{cd_max} [{status}]")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"{'Re':>6} {'C_D':>10} {'C_L_rms':>10} {'Status':>10}")
    print("-" * 60)
    
    for re in reynolds_numbers:
        r = all_results[re]
        cd = r.get('C_D', 0)
        cl = r.get('C_L_rms', 0)
        
        if re in DRAG_COEFFICIENT_REFERENCE:
            cd_min, cd_max = DRAG_COEFFICIENT_REFERENCE[re]
            status = "PASS" if cd_min <= cd <= cd_max else "CHECK"
        else:
            status = "N/A"
        
        print(f"{re:>6} {cd:>10.4f} {cl:>10.4f} {status:>10}")
    
    print("=" * 60)
    return all_results


if __name__ == "__main__":
    # Single simulation
    solver = run_cylinder_flow(nx=400, ny=100, re=100, num_steps=50000)
    solver.plot_results(save_path='results/validation/cylinder_re100.png')
    
    # Validation study
    print("\n\n")
    validation_study([20, 40, 100])