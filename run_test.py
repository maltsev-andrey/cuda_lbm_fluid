# run_test.py - Using simple_cylinder which works
from simulations.simple_cylinder import SimpleCylinderSolver
import numpy as np

# Parameters for Re=40
nx, ny = 300, 80
cx, cy, r = 60, 40, 8
re = 40
u_inlet = 0.04  # Lower velocity for stability

solver = SimpleCylinderSolver(nx, ny, cx, cy, r, re=re, u_inlet=u_inlet)
print(f"Parameters: Re={re}, tau={solver.tau:.4f}, Ma={u_inlet*np.sqrt(3):.4f}")

# Run to steady state
for step in range(30000):
    solver.step()
    
    if (step + 1) % 5000 == 0:
        # Control volume force calculation
        inlet_x = 5
        outlet_x = nx - 20
        
        flux_in = np.sum(solver.rho[:, inlet_x] * solver.ux[:, inlet_x]**2) + np.sum(solver.rho[:, inlet_x]) / 3.0
        flux_out = np.sum(solver.rho[:, outlet_x] * solver.ux[:, outlet_x]**2) + np.sum(solver.rho[:, outlet_x]) / 3.0
        
        F_x = flux_in - flux_out
        
        # Blockage correction
        blockage = solver.diameter / ny
        correction = (1 + 0.5 * blockage)**2
        
        C_D = F_x / (0.5 * u_inlet**2 * solver.diameter) / correction
        
        print(f"Step {step+1}: C_D = {C_D:.4f}")

print(f"\nFinal C_D = {C_D:.4f}")
print(f"Expected for Re=40: 1.5-1.8")
