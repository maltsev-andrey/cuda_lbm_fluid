"""
Simplified Cylinder Flow - CPU Only for Debugging

This version removes GPU complexity to ensure correct physics.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.lattice import EX, EY, W, CS2, Q, OPPOSITE
from src.equilibrium import compute_equilibrium_fast
from src.observables import compute_macroscopic_fast, compute_vorticity
from src.collision import bgk_collision_fast
from src.streaming import stream_periodic_fast
from src.boundary import create_cylinder_mask


class SimpleCylinderSolver:
    """
    Simple CPU-only cylinder flow solver.
    """
    
    def __init__(self, nx, ny, cx, cy, radius, re, u_inlet):
        self.nx = nx
        self.ny = ny
        self.cx = cx
        self.cy = cy
        self.radius = radius
        self.diameter = 2 * radius
        self.re = re
        self.u_inlet = u_inlet
        
        # Compute tau
        self.nu = u_inlet * self.diameter / re
        self.tau = 3.0 * self.nu + 0.5
        self.omega = 1.0 / self.tau
        
        print(f"Parameters: tau={self.tau:.4f}, nu={self.nu:.6f}, Ma={u_inlet*np.sqrt(3):.4f}")
        
        if self.tau <= 0.5:
            raise ValueError(f"Unstable: tau={self.tau}")
        
        # Create cylinder mask
        self.solid = create_cylinder_mask(nx, ny, cx, cy, radius)
        print(f"Solid nodes: {np.sum(self.solid)}")
        
        # Initialize with uniform flow + perturbation
        self.rho = np.ones((ny, nx))
        self.ux = np.ones((ny, nx)) * u_inlet
        self.uy = np.zeros((ny, nx))
        
        # Zero velocity in solid
        self.ux[self.solid] = 0
        self.uy[self.solid] = 0
        
        # Small perturbation to break symmetry
        np.random.seed(42)
        self.uy += 0.001 * u_inlet * np.random.randn(ny, nx)
        self.uy[self.solid] = 0
        
        # Initialize f
        self.f = compute_equilibrium_fast(self.rho, self.ux, self.uy)
        
        self.step_count = 0
    
    def inlet_bc(self):
        """Zou-He velocity inlet at x=0."""
        u_in = self.u_inlet
        
        # Known: f[0,2,3,4,6,7], Unknown: f[1,5,8]
        rho_in = (self.f[0,:,0] + self.f[2,:,0] + self.f[4,:,0] + 
                  2*(self.f[3,:,0] + self.f[6,:,0] + self.f[7,:,0])) / (1 - u_in)
        
        self.f[1,:,0] = self.f[3,:,0] + (2/3) * rho_in * u_in
        self.f[5,:,0] = self.f[7,:,0] + (1/6) * rho_in * u_in - 0.5*(self.f[2,:,0] - self.f[4,:,0])
        self.f[8,:,0] = self.f[6,:,0] + (1/6) * rho_in * u_in + 0.5*(self.f[2,:,0] - self.f[4,:,0])
    
    def outlet_bc(self):
        """Zero-gradient (copy) outlet at x=nx-1."""
        # Simple extrapolation - copy from second-to-last column
        self.f[:,:,-1] = self.f[:,:,-2]
    
    def bounce_back(self):
        """Bounce-back on cylinder."""
        # Store values first
        f_temp = self.f.copy()
        for k in range(Q):
            k_opp = OPPOSITE[k]
            self.f[k, self.solid] = f_temp[k_opp, self.solid]
    
    def step(self):
        """One LBM timestep."""
        # Collision
        self.rho, self.ux, self.uy = compute_macroscopic_fast(self.f)
        f_eq = compute_equilibrium_fast(self.rho, self.ux, self.uy)
        self.f = self.f - self.omega * (self.f - f_eq)
        
        # Streaming
        self.f = stream_periodic_fast(self.f)
        
        # Boundary conditions
        self.inlet_bc()
        self.outlet_bc()
        self.bounce_back()
        
        self.step_count += 1
    
    def compute_force(self):
        """
        Momentum exchange force on cylinder.
        
        For bounce-back: F = 2 * sum(f_i * e_i) for all links to solid
        The factor of 2 accounts for the momentum reversal.
        """
        F_x, F_y = 0.0, 0.0
        
        for j in range(self.ny):
            for i in range(self.nx):
                if not self.solid[j, i]:  # fluid node
                    for k in range(1, Q):
                        ni = i + EX[k]
                        nj = j + EY[k]
                        if 0 <= ni < self.nx and 0 <= nj < self.ny:
                            if self.solid[nj, ni]:  # neighbor is solid
                                # Momentum transferred = 2 * f_k (bounce-back reverses momentum)
                                F_x += 2.0 * self.f[k, j, i] * EX[k]
                                F_y += 2.0 * self.f[k, j, i] * EY[k]
        
        return F_x, F_y
    
    def drag_coeff(self, F_x):
        """Drag coefficient."""
        return F_x / (0.5 * self.u_inlet**2 * self.diameter)
    
    def run(self, num_steps, report_every=1000):
        """Run simulation."""
        print(f"\nRunning {num_steps} steps...")
        start = time.time()
        
        for step in range(num_steps):
            self.step()
            
            if (step + 1) % report_every == 0:
                F_x, F_y = self.compute_force()
                C_D = self.drag_coeff(F_x)
                C_L = self.drag_coeff(F_y)
                
                # Check for NaN
                if np.any(np.isnan(self.f)):
                    print(f"Step {step+1}: NaN detected! Stopping.")
                    break
                
                print(f"Step {step+1}: C_D={C_D:.4f}, C_L={C_L:.4f}, max_ux={np.max(self.ux):.4f}")
        
        elapsed = time.time() - start
        mlups = num_steps * self.nx * self.ny / elapsed / 1e6
        print(f"\nDone: {elapsed:.1f}s, {mlups:.2f} MLUPS")
        
        return self.drag_coeff(self.compute_force()[0])


def main():
    """Run cylinder flow test."""
    print("=" * 50)
    print("Simple Cylinder Flow Test")
    print("=" * 50)
    
    # Parameters for Re=40
    nx, ny = 300, 80
    cx, cy = 60, 40
    radius = 8
    re = 40
    u_inlet = 0.04  # Low velocity for stability
    
    solver = SimpleCylinderSolver(nx, ny, cx, cy, radius, re, u_inlet)
    
    C_D = solver.run(20000, report_every=2000)
    
    print(f"\nFinal C_D = {C_D:.4f}")
    print(f"Expected for Re=40: 1.5 - 1.8")
    
    if 1.5 <= C_D <= 1.8:
        print("PASS!")
    elif C_D > 0.5:
        print("Close - may need more iterations")
    else:
        print("CHECK - something may be wrong")


if __name__ == "__main__":
    main()