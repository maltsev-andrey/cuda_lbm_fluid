"""
Vortex Shedding Visualization

Creates beautiful visualizations of Karman vortex street:
- Static plots (velocity, vorticity, streamlines)
- Animated GIF of vortex shedding
- Publication-quality figures

Supports both CPU and GPU solvers.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simulations.simple_cylinder import SimpleCylinderSolver
from src.observables import compute_vorticity, compute_velocity_magnitude

# Try GPU solver
try:
    from simulations.gpu_cylinder_flow import GPUCylinderFlowSolver
    GPU_AVAILABLE = True
except:
    GPU_AVAILABLE = False


def create_vorticity_colormap():
    """Create a custom colormap for vorticity (blue-white-red)."""
    colors = [
        (0.0, 0.2, 0.6),   # Dark blue (negative)
        (0.4, 0.6, 1.0),   # Light blue
        (1.0, 1.0, 1.0),   # White (zero)
        (1.0, 0.6, 0.4),   # Light red
        (0.6, 0.1, 0.1),   # Dark red (positive)
    ]
    return LinearSegmentedColormap.from_list('vorticity', colors, N=256)


def plot_flow_field(solver, save_path=None, title_suffix=""):
    """
    Create a comprehensive flow visualization.
    
    Parameters
    ----------
    solver : SimpleCylinderSolver
        Solver with computed flow field
    save_path : str, optional
        Path to save figure
    title_suffix : str
        Additional text for titles
    """
    # Get fields
    ux = solver.ux.copy()
    uy = solver.uy.copy()
    rho = solver.rho.copy()
    
    # Mask cylinder
    ux[solver.solid] = np.nan
    uy[solver.solid] = np.nan
    
    vorticity = compute_vorticity(ux, uy)
    velocity_mag = compute_velocity_magnitude(ux, uy)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Velocity magnitude
    ax = axes[0, 0]
    vm = velocity_mag.copy()
    vm[solver.solid] = np.nan
    im = ax.imshow(vm, origin='lower', cmap='viridis', aspect='equal',
                   extent=[0, solver.nx, 0, solver.ny])
    ax.set_title(f'Velocity Magnitude |u| {title_suffix}', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='|u|', shrink=0.8)
    
    # Draw cylinder
    circle = plt.Circle((solver.cx, solver.cy), solver.radius, 
                        color='white', ec='black', linewidth=2)
    ax.add_patch(circle)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # 2. Vorticity (Karman vortex street)
    ax = axes[0, 1]
    vort = vorticity.copy()
    vort[solver.solid] = np.nan
    vmax = np.nanpercentile(np.abs(vort), 98)
    
    cmap = create_vorticity_colormap()
    im = ax.imshow(vort, origin='lower', cmap=cmap, aspect='equal',
                   vmin=-vmax, vmax=vmax, extent=[0, solver.nx, 0, solver.ny])
    ax.set_title(f'Vorticity ω (Karman Vortex Street) {title_suffix}', 
                fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='ω', shrink=0.8)
    
    circle = plt.Circle((solver.cx, solver.cy), solver.radius, 
                        color='gray', ec='black', linewidth=2)
    ax.add_patch(circle)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # 3. Streamlines
    ax = axes[1, 0]
    
    # Create meshgrid for streamlines
    x = np.arange(solver.nx)
    y = np.arange(solver.ny)
    X, Y = np.meshgrid(x, y)
    
    # Replace NaN with zeros for streamplot
    ux_stream = np.nan_to_num(ux, nan=0.0)
    uy_stream = np.nan_to_num(uy, nan=0.0)
    
    # Speed for coloring
    speed = np.sqrt(ux_stream**2 + uy_stream**2)
    
    ax.streamplot(X, Y, ux_stream, uy_stream, color=speed, cmap='coolwarm',
                  density=2, linewidth=0.8, arrowsize=0.8)
    
    circle = plt.Circle((solver.cx, solver.cy), solver.radius, 
                        color='lightgray', ec='black', linewidth=2)
    ax.add_patch(circle)
    ax.set_xlim(0, solver.nx)
    ax.set_ylim(0, solver.ny)
    ax.set_title(f'Streamlines {title_suffix}', fontsize=12, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    
    # 4. Pressure field
    ax = axes[1, 1]
    pressure = rho / 3.0  # LBM pressure
    pressure[solver.solid] = np.nan
    
    im = ax.imshow(pressure, origin='lower', cmap='RdYlBu_r', aspect='equal',
                   extent=[0, solver.nx, 0, solver.ny])
    ax.set_title(f'Pressure Field {title_suffix}', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='p = ρ/3', shrink=0.8)
    
    circle = plt.Circle((solver.cx, solver.cy), solver.radius, 
                        color='gray', ec='black', linewidth=2)
    ax.add_patch(circle)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


def create_vortex_animation(nx=400, ny=100, re=100, num_frames=200, 
                            steps_per_frame=100, save_path=None):
    """
    Create an animated GIF of vortex shedding.
    
    Parameters
    ----------
    nx, ny : int
        Domain size
    re : float
        Reynolds number
    num_frames : int
        Number of animation frames
    steps_per_frame : int
        Simulation steps between frames
    save_path : str, optional
        Path to save GIF (default: results/animations/vortex_shedding.gif)
    """
    print("Creating Vortex Shedding Animation")
    print("=" * 50)
    print(f"Domain: {nx} x {ny}")
    print(f"Reynolds: {re}")
    print(f"Frames: {num_frames}")
    print(f"Steps/frame: {steps_per_frame}")
    print()
    
    # Initialize solver
    cx, cy, r = nx // 5, ny // 2, ny // 10
    solver = SimpleCylinderSolver(nx, ny, cx, cy, r, re=re, u_inlet=0.04)
    
    print(f"Tau: {solver.tau:.4f}")
    print(f"Running warmup ({steps_per_frame * 50} steps)...")
    
    # Warmup to establish flow
    for _ in range(steps_per_frame * 50):
        solver.step()
    
    print("Generating frames...")
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(12, 4))
    
    # Initial vorticity
    vorticity = compute_vorticity(solver.ux, solver.uy)
    vorticity[solver.solid] = np.nan
    vmax = 0.02  # Fixed scale for animation
    
    cmap = create_vorticity_colormap()
    im = ax.imshow(vorticity, origin='lower', cmap=cmap, aspect='equal',
                   vmin=-vmax, vmax=vmax, extent=[0, nx, 0, ny],
                   animated=True)
    
    circle = plt.Circle((cx, cy), r, color='gray', ec='black', linewidth=2)
    ax.add_patch(circle)
    
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_xlabel('x', fontsize=10)
    ax.set_ylabel('y', fontsize=10)
    
    title = ax.set_title(f'Karman Vortex Street (Re={re}, t=0)', fontsize=12)
    
    plt.colorbar(im, ax=ax, label='Vorticity ω', shrink=0.8)
    
    frames_data = []
    
    # Generate frames
    start_time = time.perf_counter()
    for frame in range(num_frames):
        # Run simulation
        for _ in range(steps_per_frame):
            solver.step()
        
        # Compute vorticity
        vorticity = compute_vorticity(solver.ux, solver.uy)
        vorticity[solver.solid] = np.nan
        frames_data.append(vorticity.copy())
        
        if (frame + 1) % 20 == 0:
            elapsed = time.perf_counter() - start_time
            eta = elapsed / (frame + 1) * (num_frames - frame - 1)
            print(f"  Frame {frame + 1}/{num_frames} (ETA: {eta:.0f}s)")
    
    print("Creating animation...")
    
    def animate(frame_idx):
        im.set_array(frames_data[frame_idx])
        t = (frame_idx + 50) * steps_per_frame  # Account for warmup
        title.set_text(f'Karman Vortex Street (Re={re}, t={t})')
        return [im, title]
    
    anim = animation.FuncAnimation(fig, animate, frames=num_frames,
                                   interval=50, blit=True)
    
    # Save
    if save_path is None:
        save_path = 'results/animations/vortex_shedding.gif'
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    print(f"Saving to {save_path}...")
    anim.save(save_path, writer='pillow', fps=20, dpi=100)
    print(f"Animation saved: {save_path}")
    
    plt.close()
    
    return save_path


def create_publication_figure(save_path=None):
    """
    Create a publication-quality figure showing vortex shedding development.
    """
    print("Creating Publication Figure")
    print("=" * 50)
    
    nx, ny = 500, 120
    cx, cy, r = 100, 60, 12
    re = 100
    
    solver = SimpleCylinderSolver(nx, ny, cx, cy, r, re=re, u_inlet=0.04)
    
    # Time points to capture
    time_points = [0, 5000, 15000, 30000]
    
    fig, axes = plt.subplots(len(time_points), 1, figsize=(14, 10))
    
    cmap = create_vorticity_colormap()
    
    for idx, t_target in enumerate(time_points):
        # Run to target time
        current_step = 0 if idx == 0 else time_points[idx-1]
        steps_to_run = t_target - current_step
        
        print(f"Running to t={t_target}...")
        for _ in range(steps_to_run):
            solver.step()
        
        # Plot vorticity
        ax = axes[idx]
        vorticity = compute_vorticity(solver.ux, solver.uy)
        vorticity[solver.solid] = np.nan
        
        vmax = 0.015
        im = ax.imshow(vorticity, origin='lower', cmap=cmap, aspect='equal',
                       vmin=-vmax, vmax=vmax, extent=[0, nx, 0, ny])
        
        circle = plt.Circle((cx, cy), r, color='gray', ec='black', linewidth=1.5)
        ax.add_patch(circle)
        
        ax.set_ylabel('y', fontsize=10)
        ax.set_title(f't = {t_target}', fontsize=11, loc='left', fontweight='bold')
        
        if idx == len(time_points) - 1:
            ax.set_xlabel('x', fontsize=10)
        else:
            ax.set_xticklabels([])
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Vorticity ω')
    
    fig.suptitle(f'Development of Karman Vortex Street (Re = {re})', 
                fontsize=14, fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 0.9, 0.96])
    
    if save_path is None:
        save_path = 'results/figures/vortex_development.png'
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
    print(f"Saved: {save_path}")
    
    return fig


def create_drag_lift_plot(solver, save_path=None):
    """Create drag and lift coefficient history plot."""
    if len(solver.force_history) < 10:
        print("Not enough force data")
        return None
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    times = np.array(solver.time_history) if hasattr(solver, 'time_history') else np.arange(len(solver.force_history))
    
    # Compute coefficients
    forces = np.array(solver.force_history)
    lifts = np.array(solver.lift_history) if hasattr(solver, 'lift_history') else np.zeros_like(forces)
    
    blockage = solver.diameter / solver.ny
    correction = (1 + 0.5 * blockage)**2
    
    C_D = forces / (0.5 * solver.u_inlet**2 * solver.diameter) / correction
    C_L = lifts / (0.5 * solver.u_inlet**2 * solver.diameter) / correction
    
    # Drag coefficient
    ax = axes[0]
    ax.plot(times, C_D, 'b-', linewidth=1.5, label='$C_D$')
    ax.axhline(y=np.mean(C_D[len(C_D)//2:]), color='r', linestyle='--', 
               label=f'Mean = {np.mean(C_D[len(C_D)//2:]):.3f}')
    ax.set_ylabel('Drag Coefficient $C_D$', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Force Coefficients (Re = {solver.re})', fontsize=14, fontweight='bold')
    
    # Lift coefficient  
    ax = axes[1]
    ax.plot(times, C_L, 'g-', linewidth=1.5, label='$C_L$')
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.set_ylabel('Lift Coefficient $C_L$', fontsize=12)
    ax.set_xlabel('Time (lattice units)', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Add RMS annotation
    C_L_rms = np.sqrt(np.mean(C_L[len(C_L)//2:]**2))
    ax.annotate(f'RMS = {C_L_rms:.3f}', xy=(0.98, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"Saved: {save_path}")
    
    return fig


def create_publication_figure_stable(save_path=None):
    """Create publication figure with stable parameters (Re=40)."""
    print("Creating Publication Figure (Re=40, stable)")
    
    nx, ny = 400, 100
    cx, cy, r = 80, 50, 10
    re = 40
    u_inlet = 0.04
    
    solver = SimpleCylinderSolver(nx, ny, cx, cy, r, re=re, u_inlet=u_inlet)
    print(f"   tau={solver.tau:.4f}")
    
    # Time points
    time_points = [0, 5000, 15000, 25000]
    
    fig, axes = plt.subplots(len(time_points), 1, figsize=(14, 10))
    cmap = create_vorticity_colormap()
    
    for idx, t_target in enumerate(time_points):
        current_step = 0 if idx == 0 else time_points[idx-1]
        steps_to_run = t_target - current_step
        
        print(f"   Running to t={t_target}...")
        for _ in range(steps_to_run):
            solver.step()
        
        ax = axes[idx]
        vorticity = compute_vorticity(solver.ux, solver.uy)
        vorticity[solver.solid] = np.nan
        
        vmax = 0.01
        im = ax.imshow(vorticity, origin='lower', cmap=cmap, aspect='equal',
                       vmin=-vmax, vmax=vmax, extent=[0, nx, 0, ny])
        
        circle = plt.Circle((cx, cy), r, color='gray', ec='black', linewidth=1.5)
        ax.add_patch(circle)
        
        ax.set_ylabel('y', fontsize=10)
        ax.set_title(f't = {t_target}', fontsize=11, loc='left', fontweight='bold')
        
        if idx == len(time_points) - 1:
            ax.set_xlabel('x', fontsize=10)
        else:
            ax.set_xticklabels([])
    
    fig.suptitle(f'Development of Flow Around Cylinder (Re = {re})', 
                fontsize=14, fontweight='bold', y=0.98)
    
    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Vorticity')
    
    if save_path is None:
        save_path = 'results/figures/vortex_development.png'
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"   Saved: {save_path}")
    
    plt.close()
    return fig


def create_vortex_animation_stable(save_path=None):
    """Create animation with stable parameters (Re=40)."""
    print("Creating Animation (Re=40, stable)")
    
    nx, ny = 300, 80
    cx, cy, r = 60, 40, 8
    re = 40
    u_inlet = 0.04
    num_frames = 80
    steps_per_frame = 200
    
    solver = SimpleCylinderSolver(nx, ny, cx, cy, r, re=re, u_inlet=u_inlet)
    print(f"   tau={solver.tau:.4f}")
    
    # Warmup
    print("   Warmup (5000 steps)...")
    for _ in range(5000):
        solver.step()
    
    print("   Generating frames...")
    
    # Setup figure
    fig, ax = plt.subplots(figsize=(10, 4))
    
    vorticity = compute_vorticity(solver.ux, solver.uy)
    vorticity[solver.solid] = np.nan
    vmax = 0.008
    
    cmap = create_vorticity_colormap()
    im = ax.imshow(vorticity, origin='lower', cmap=cmap, aspect='equal',
                   vmin=-vmax, vmax=vmax, extent=[0, nx, 0, ny])
    
    circle = plt.Circle((cx, cy), r, color='gray', ec='black', linewidth=2)
    ax.add_patch(circle)
    
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    title = ax.set_title(f'Cylinder Flow (Re={re}, t=0)', fontsize=12)
    plt.colorbar(im, ax=ax, label='Vorticity', shrink=0.8)
    
    frames_data = []
    
    for frame in range(num_frames):
        for _ in range(steps_per_frame):
            solver.step()
        
        vorticity = compute_vorticity(solver.ux, solver.uy)
        vorticity[solver.solid] = np.nan
        frames_data.append(vorticity.copy())
        
        if (frame + 1) % 20 == 0:
            print(f"     Frame {frame + 1}/{num_frames}")
    
    print("   Creating animation...")
    
    def animate(frame_idx):
        im.set_array(frames_data[frame_idx])
        t = (frame_idx + 25) * steps_per_frame
        title.set_text(f'Cylinder Flow (Re={re}, t={t})')
        return [im, title]
    
    anim = animation.FuncAnimation(fig, animate, frames=num_frames,
                                   interval=50, blit=True)
    
    if save_path is None:
        save_path = 'results/animations/vortex_shedding.gif'
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    anim.save(save_path, writer='pillow', fps=15, dpi=100)
    print(f"   Saved: {save_path}")
    
    plt.close()
    return save_path


def run_visualization_demo():
    """Run complete visualization demonstration using GPU if available."""
    print("=" * 60)
    print("LBM Vortex Shedding Visualization Demo")
    print("=" * 60)
    print(f"GPU Available: {GPU_AVAILABLE}")
    
    # Create output directories
    os.makedirs('results/figures', exist_ok=True)
    os.makedirs('results/animations', exist_ok=True)
    
    if GPU_AVAILABLE:
        run_gpu_visualization()
    else:
        run_cpu_visualization()


def run_gpu_visualization():
    """Run visualization using GPU solver (fast + stable)."""
    from simulations.gpu_cylinder_flow import GPUCylinderFlowSolver
    from numba import cuda
    
    print("\n*** Using GPU Acceleration ***\n")
    
    # 1. Run GPU simulation for Re=100
    print("1. Running Re=100 simulation on GPU...")
    nx, ny = 600, 150
    cx, cy, r = 120, 75, 15
    re = 100
    
    # Calculate stable u_inlet
    target_tau = 0.55
    nu = (target_tau - 0.5) / 3.0
    diameter = 2 * r
    u_inlet = nu * re / diameter
    
    # Cap Mach number
    if u_inlet * np.sqrt(3) > 0.1:
        u_inlet = 0.1 / np.sqrt(3)
    
    solver = GPUCylinderFlowSolver(nx, ny, cx, cy, r, re=re, u_inlet=u_inlet)
    print(f"   tau={solver.tau:.4f}, Ma={solver.mach:.4f}")
    
    # Track forces
    force_history = []
    lift_history = []
    time_history = []
    
    start = time.perf_counter()
    for step in range(40000):
        solver.step()
        
        if (step + 1) % 500 == 0:
            F_x, F_y = solver.compute_forces()
            force_history.append(F_x)
            lift_history.append(F_y)
            time_history.append(step + 1)
        
        if (step + 1) % 10000 == 0:
            elapsed = time.perf_counter() - start
            mlups = (step + 1) * nx * ny / elapsed / 1e6
            print(f"   Step {step + 1}/40000 ({mlups:.0f} MLUPS)")
    
    cuda.synchronize()
    elapsed = time.perf_counter() - start
    print(f"   Completed in {elapsed:.1f}s ({40000 * nx * ny / elapsed / 1e6:.0f} MLUPS)")
    
    # Get fields for plotting
    fields = solver.get_fields()
    
    # Store for plotting
    solver.force_history = force_history
    solver.lift_history = lift_history
    solver.time_history = time_history
    solver.solid = solver.solid_mask
    
    # 2. Create flow field plot
    print("\n2. Creating flow field visualization...")
    plot_flow_field_from_arrays(fields, solver, 'results/figures/flow_field_re100_gpu.png', '(Re=100, GPU)')
    
    # 3. Create force plot
    print("\n3. Creating force coefficient plot...")
    create_drag_lift_plot_gpu(solver, force_history, lift_history, time_history, 
                              'results/figures/force_coefficients_gpu.png')
    
    # 4. Create animation
    print("\n4. Creating vortex animation on GPU...")
    create_gpu_animation(save_path='results/animations/vortex_shedding_gpu.gif')
    
    print("\n" + "=" * 60)
    print("GPU Visualization Complete!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - results/figures/flow_field_re100_gpu.png")
    print("  - results/figures/force_coefficients_gpu.png")
    print("  - results/animations/vortex_shedding_gpu.gif")


def plot_flow_field_from_arrays(fields, solver, save_path, title_suffix=""):
    """Create flow visualization from pre-computed fields."""
    ux = fields['ux'].copy()
    uy = fields['uy'].copy()
    rho = fields['rho'].copy()
    vorticity = fields['vorticity'].copy()
    
    # Mask solid
    solid = solver.solid_mask if hasattr(solver, 'solid_mask') else solver.solid
    ux[solid] = np.nan
    uy[solid] = np.nan
    vorticity[solid] = np.nan
    
    velocity_mag = np.sqrt(ux**2 + uy**2)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # 1. Velocity magnitude
    ax = axes[0, 0]
    im = ax.imshow(velocity_mag, origin='lower', cmap='viridis', aspect='equal',
                   extent=[0, solver.nx, 0, solver.ny])
    ax.set_title(f'Velocity Magnitude |u| {title_suffix}', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='|u|', shrink=0.8)
    circle = plt.Circle((solver.cx, solver.cy), solver.radius, color='white', ec='black', linewidth=2)
    ax.add_patch(circle)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # 2. Vorticity
    ax = axes[0, 1]
    vmax = np.nanpercentile(np.abs(vorticity), 98)
    cmap = create_vorticity_colormap()
    im = ax.imshow(vorticity, origin='lower', cmap=cmap, aspect='equal',
                   vmin=-vmax, vmax=vmax, extent=[0, solver.nx, 0, solver.ny])
    ax.set_title(f'Vorticity (Karman Vortex Street) {title_suffix}', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='omega', shrink=0.8)
    circle = plt.Circle((solver.cx, solver.cy), solver.radius, color='gray', ec='black', linewidth=2)
    ax.add_patch(circle)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    # 3. Streamlines
    ax = axes[1, 0]
    x = np.arange(solver.nx)
    y = np.arange(solver.ny)
    X, Y = np.meshgrid(x, y)
    ux_stream = np.nan_to_num(ux, nan=0.0)
    uy_stream = np.nan_to_num(uy, nan=0.0)
    speed = np.sqrt(ux_stream**2 + uy_stream**2)
    ax.streamplot(X, Y, ux_stream, uy_stream, color=speed, cmap='coolwarm',
                  density=2, linewidth=0.8, arrowsize=0.8)
    circle = plt.Circle((solver.cx, solver.cy), solver.radius, color='lightgray', ec='black', linewidth=2)
    ax.add_patch(circle)
    ax.set_xlim(0, solver.nx)
    ax.set_ylim(0, solver.ny)
    ax.set_title(f'Streamlines {title_suffix}', fontsize=12, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    
    # 4. Pressure
    ax = axes[1, 1]
    pressure = rho / 3.0
    pressure[solid] = np.nan
    im = ax.imshow(pressure, origin='lower', cmap='RdYlBu_r', aspect='equal',
                   extent=[0, solver.nx, 0, solver.ny])
    ax.set_title(f'Pressure Field {title_suffix}', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='p', shrink=0.8)
    circle = plt.Circle((solver.cx, solver.cy), solver.radius, color='gray', ec='black', linewidth=2)
    ax.add_patch(circle)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"   Saved: {save_path}")
    
    plt.close()
    return fig


def create_drag_lift_plot_gpu(solver, forces, lifts, times, save_path):
    """Create force coefficient plot from GPU data."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    forces = np.array(forces)
    lifts = np.array(lifts)
    times = np.array(times)
    
    blockage = solver.diameter / solver.ny
    correction = (1 + 0.5 * blockage)**2
    
    C_D = forces / (0.5 * solver.u_inlet**2 * solver.diameter) / correction
    C_L = lifts / (0.5 * solver.u_inlet**2 * solver.diameter) / correction
    
    # Drag
    ax = axes[0]
    ax.plot(times, C_D, 'b-', linewidth=1.5)
    mean_cd = np.mean(C_D[len(C_D)//2:])
    ax.axhline(y=mean_cd, color='r', linestyle='--', label=f'Mean = {mean_cd:.3f}')
    ax.set_ylabel('$C_D$', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title(f'Force Coefficients (Re={solver.re}, GPU)', fontsize=14, fontweight='bold')
    
    # Lift
    ax = axes[1]
    ax.plot(times, C_L, 'g-', linewidth=1.5)
    ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.5)
    ax.set_ylabel('$C_L$', fontsize=12)
    ax.set_xlabel('Time', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    C_L_rms = np.sqrt(np.mean(C_L[len(C_L)//2:]**2))
    ax.annotate(f'RMS = {C_L_rms:.3f}', xy=(0.98, 0.95), xycoords='axes fraction',
                ha='right', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        print(f"   Saved: {save_path}")
    
    plt.close()


def create_gpu_animation(save_path=None):
    """Create vortex animation using GPU (fast!)."""
    from simulations.gpu_cylinder_flow import GPUCylinderFlowSolver
    from numba import cuda
    
    print("   Initializing GPU solver...")
    
    nx, ny = 600, 150
    cx, cy, r = 120, 75, 15
    re = 100
    num_frames = 100
    steps_per_frame = 200
    
    # Stable parameters
    target_tau = 0.55
    nu = (target_tau - 0.5) / 3.0
    diameter = 2 * r
    u_inlet = nu * re / diameter
    if u_inlet * np.sqrt(3) > 0.1:
        u_inlet = 0.1 / np.sqrt(3)
    
    solver = GPUCylinderFlowSolver(nx, ny, cx, cy, r, re=re, u_inlet=u_inlet)
    
    # Warmup
    print("   Warmup (10000 steps)...")
    for _ in range(10000):
        solver.step()
    cuda.synchronize()
    
    print("   Generating frames...")
    
    # Collect frames
    frames_data = []
    start = time.perf_counter()
    
    for frame in range(num_frames):
        for _ in range(steps_per_frame):
            solver.step()
        
        # Get vorticity
        fields = solver.get_fields()
        vorticity = fields['vorticity'].copy()
        vorticity[solver.solid_mask] = np.nan
        frames_data.append(vorticity)
        
        if (frame + 1) % 25 == 0:
            elapsed = time.perf_counter() - start
            print(f"     Frame {frame + 1}/{num_frames} ({elapsed:.1f}s)")
    
    cuda.synchronize()
    print(f"   Frames generated in {time.perf_counter() - start:.1f}s")
    
    # Create animation
    print("   Creating animation...")
    
    fig, ax = plt.subplots(figsize=(12, 4))
    
    vmax = 0.015
    cmap = create_vorticity_colormap()
    im = ax.imshow(frames_data[0], origin='lower', cmap=cmap, aspect='equal',
                   vmin=-vmax, vmax=vmax, extent=[0, nx, 0, ny])
    
    circle = plt.Circle((cx, cy), r, color='gray', ec='black', linewidth=2)
    ax.add_patch(circle)
    
    ax.set_xlim(0, nx)
    ax.set_ylim(0, ny)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    title = ax.set_title(f'Karman Vortex Street (Re={re}, GPU)', fontsize=12)
    plt.colorbar(im, ax=ax, label='Vorticity', shrink=0.8)
    
    def animate(frame_idx):
        im.set_array(frames_data[frame_idx])
        t = (frame_idx + 50) * steps_per_frame
        title.set_text(f'Karman Vortex Street (Re={re}, t={t})')
        return [im, title]
    
    anim = animation.FuncAnimation(fig, animate, frames=num_frames,
                                   interval=50, blit=True)
    
    if save_path is None:
        save_path = 'results/animations/vortex_shedding_gpu.gif'
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    anim.save(save_path, writer='pillow', fps=20, dpi=100)
    print(f"   Saved: {save_path}")
    
    plt.close()


def run_cpu_visualization():
    """Fallback to CPU visualization with stable Re=40."""
    print("\n*** Using CPU (GPU not available) ***\n")
    
    # Use Re=40 for stability on CPU
    print("1. Running Re=40 simulation on CPU...")
    nx, ny = 300, 80
    cx, cy, r = 60, 40, 8
    re = 40
    u_inlet = 0.04
    
    solver = SimpleCylinderSolver(nx, ny, cx, cy, r, re=re, u_inlet=u_inlet)
    print(f"   tau={solver.tau:.4f}")
    
    solver.force_history = []
    solver.lift_history = []
    solver.time_history = []
    
    for step in range(30000):
        solver.step()
        
        if (step + 1) % 200 == 0:
            inlet_x, outlet_x = 5, nx - 20
            flux_in = np.sum(solver.rho[:, inlet_x] * solver.ux[:, inlet_x]**2) + np.sum(solver.rho[:, inlet_x]) / 3.0
            flux_out = np.sum(solver.rho[:, outlet_x] * solver.ux[:, outlet_x]**2) + np.sum(solver.rho[:, outlet_x]) / 3.0
            F_x = flux_in - flux_out
            
            flux_in_y = np.sum(solver.rho[:, inlet_x] * solver.ux[:, inlet_x] * solver.uy[:, inlet_x])
            flux_out_y = np.sum(solver.rho[:, outlet_x] * solver.ux[:, outlet_x] * solver.uy[:, outlet_x])
            F_y = flux_in_y - flux_out_y
            
            solver.force_history.append(F_x)
            solver.lift_history.append(F_y)
            solver.time_history.append(step + 1)
        
        if (step + 1) % 10000 == 0:
            print(f"   Step {step + 1}/30000")
    
    print("\n2. Creating visualizations...")
    plot_flow_field(solver, 'results/figures/flow_field_re40.png', '(Re=40)')
    create_drag_lift_plot(solver, 'results/figures/force_coefficients.png')
    
    print("\n" + "=" * 60)
    print("CPU Visualization Complete!")
    print("=" * 60)


if __name__ == "__main__":
    run_visualization_demo()