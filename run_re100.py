# run_re100.py - Re=100 with STABLE parameters
from simulations.simple_cylinder import SimpleCylinderSolver
import numpy as np

# For Re=100 we need lower u_inlet to get stable tau
# tau = 3*nu + 0.5, nu = U*D/Re
# For tau=0.56: nu = 0.02, U = nu*Re/D = 0.02*100/20 = 0.1 (too high Ma)
# Better: use larger domain with bigger cylinder

nx, ny = 600, 150  # Larger domain
cx, cy, r = 120, 75, 15  # Larger cylinder (D=30)
re = 100
diameter = 2 * r

# Calculate u_inlet for stable tau ~0.56
target_tau = 0.56
nu = (target_tau - 0.5) / 3.0  # = 0.02
u_inlet = nu * re / diameter   # = 0.02 * 100 / 30 = 0.0667

# Check Mach
mach = u_inlet * np.sqrt(3)
print(f"Target: tau={target_tau}, nu={nu:.4f}, u={u_inlet:.4f}, Ma={mach:.4f}")

if mach > 0.1:
    print("Ma too high, reducing velocity...")
    u_inlet = 0.05
    nu = u_inlet * diameter / re
    target_tau = 3 * nu + 0.5
    print(f"Adjusted: tau={target_tau:.4f}, nu={nu:.4f}, u={u_inlet:.4f}")

solver = SimpleCylinderSolver(nx, ny, cx, cy, r, re=re, u_inlet=u_inlet)
print(f"\nActual: tau={solver.tau:.4f}, Ma={u_inlet*np.sqrt(3):.4f}")
print(f"Blockage: {diameter/ny:.1%}")

C_D_history = []
C_L_history = []

for step in range(40000):
    solver.step()
    
    if (step + 1) % 500 == 0:
        inlet_x = 5
        outlet_x = nx - 20
        
        flux_in_x = np.sum(solver.rho[:, inlet_x] * solver.ux[:, inlet_x]**2) + np.sum(solver.rho[:, inlet_x]) / 3.0
        flux_out_x = np.sum(solver.rho[:, outlet_x] * solver.ux[:, outlet_x]**2) + np.sum(solver.rho[:, outlet_x]) / 3.0
        F_x = flux_in_x - flux_out_x
        
        flux_in_y = np.sum(solver.rho[:, inlet_x] * solver.ux[:, inlet_x] * solver.uy[:, inlet_x])
        flux_out_y = np.sum(solver.rho[:, outlet_x] * solver.ux[:, outlet_x] * solver.uy[:, outlet_x])
        F_y = flux_in_y - flux_out_y
        
        blockage = diameter / ny
        correction = (1 + 0.5 * blockage)**2
        
        C_D = F_x / (0.5 * u_inlet**2 * diameter) / correction
        C_L = F_y / (0.5 * u_inlet**2 * diameter) / correction
        
        C_D_history.append(C_D)
        C_L_history.append(C_L)
        
        if (step + 1) % 10000 == 0:
            print(f"Step {step+1}: C_D = {C_D:.4f}, C_L = {C_L:.4f}")

C_D_mean = np.mean(C_D_history[-40:])
C_L_std = np.std(C_L_history[-40:])

print(f"\nResults:")
print(f"  C_D (mean) = {C_D_mean:.4f}")
print(f"  C_L (std)  = {C_L_std:.4f}")
print(f"  Expected C_D for Re=100: 1.3-1.5")
print(f"  Vortex shedding: {'Yes' if C_L_std > 0.01 else 'No'}")
