[![Status](https://img.shields.io/badge/Status-In%20Development-yellow)]()
[![GPU](https://img.shields.io/badge/GPU-Tesla%20P100-green)]()
[![CUDA](https://img.shields.io/badge/CUDA-Numba-blue)]()

# Code Architecture

## Overview

This document describes the software architecture of the CUDA LBM fluid dynamics solver. The design follows a layered approach separating core algorithmic components from application-specific implementations, enabling code reuse across different flow configurations while maintaining flexibility for problem-specific customization.

## Design Principles

The architecture adheres to several guiding principles that emerged from the requirements of high-performance GPU computing combined with the need for physical validation and extensibility.

Separation of concerns drives the module structure. The core LBM algorithm is decomposed into independent operations (equilibrium computation, collision, streaming, boundary conditions) that can be tested and optimized individually. Application solvers compose these operations into complete simulation workflows without duplicating fundamental code.

Performance transparency ensures that optimization decisions are explicit rather than hidden. Each GPU kernel implementation resides in a separate file with clear documentation of its optimization strategy and expected performance characteristics. This allows users to select the appropriate implementation for their hardware and precision requirements.

Validation integration treats testing as a first-class concern. The module structure facilitates unit testing of individual components against analytical solutions, while the simulation layer enables integration testing against established benchmark data.

## Module Structure

### Core Library (src/)

The source directory contains the fundamental building blocks of the LBM algorithm. These modules are designed to be stateless where possible, operating on arrays passed as arguments rather than maintaining internal state.

**lattice.py** defines the D2Q9 lattice structure including discrete velocities, weights, speed of sound, and opposite direction mappings. All constants are provided as both Python lists for flexibility and NumPy arrays for computational efficiency. The module exports:
- EX, EY: velocity components for each of the 9 directions
- W: quadrature weights
- CS2: speed of sound squared (1/3 in lattice units)
- Q: number of velocities (9)
- OPPOSITE: mapping from each direction to its opposite

**equilibrium.py** implements the Maxwell-Boltzmann equilibrium distribution truncated to second order in velocity. Two implementations are provided: a pure NumPy version for clarity and a Numba-accelerated version for production use. The equilibrium function accepts density and velocity fields and returns the complete distribution tensor.

**collision.py** contains the BGK single-relaxation-time collision operator. The implementation computes the post-collision distribution as a weighted average of the current distribution and equilibrium. A Two-Relaxation-Time (TRT) operator is also provided for improved stability at high Reynolds numbers, though BGK suffices for the validation cases in this project.

**streaming.py** implements the propagation step where distributions move to neighboring nodes according to their velocity directions. The periodic streaming function handles wrap-around at domain boundaries. Directed streaming variants support open boundaries where periodicity is applied selectively.

**boundary.py** provides boundary condition implementations. The bounce-back function enforces no-slip conditions by swapping distributions with their opposites at solid nodes. The Zou-He implementation handles velocity and pressure boundaries through the non-equilibrium bounce-back formulation. Helper functions create solid masks for common geometries including cylinders and rectangular obstacles.

**observables.py** computes macroscopic quantities from the distribution functions. Density is the zeroth moment (sum of all distributions), momentum is the first moment (weighted sum by velocities), and velocity follows from momentum divided by density. The vorticity function computes the curl of the velocity field using central differences. All functions are provided in both NumPy and Numba-accelerated forms.

### GPU Kernels (src/kernels/)

The kernels subdirectory contains progressively optimized GPU implementations. Each file represents a distinct optimization strategy with documented performance characteristics.

**cpu_baseline.py** provides the reference implementation using Numba's parallel CPU compilation. This establishes the baseline for speedup calculations and serves as a fallback when GPU hardware is unavailable. Performance is approximately 15 MLUPS on modern multi-core processors.

**gpu_naive.py** directly translates the CPU algorithm to CUDA without memory layout optimization. Each thread processes one lattice node, reading from and writing to global memory with the same array-of-structures layout used by the CPU version. Performance reaches approximately 1,300 MLUPS, limited by uncoalesced memory access.

**gpu_optimized.py** introduces structure-of-arrays data layout and fused collision-streaming kernels. Consecutive threads access consecutive memory locations, achieving coalesced memory transactions. Loop unrolling eliminates branching overhead within the D2Q9 velocity loop. The AA-pattern streaming option reduces memory traffic by performing streaming in-place through alternating even and odd steps. Performance reaches 3,698 MLUPS with 72.7% memory bandwidth efficiency.

**gpu_ultra.py** extends optimization to single-precision arithmetic. Since LBM is memory-bandwidth limited, halving the data size nearly doubles throughput. The float32 kernels achieve 7,248 MLUPS while maintaining sufficient precision for the flow conditions examined in this project. The module also provides benchmark utilities for systematic performance measurement.

### Simulation Solvers (simulations/)

The simulations directory contains complete solver implementations for specific flow configurations. Each solver combines core library functions with problem-specific initialization and boundary condition logic.

**poiseuille_flow.py** implements pressure-driven channel flow between parallel plates. The solver applies periodic boundaries in the streamwise direction with bounce-back at the upper and lower walls. A body force drives the flow, and the resulting parabolic velocity profile is compared against the analytical solution.

**lid_driven_cavity.py** implements the classical cavity benchmark with a moving top wall. The solver handles the velocity discontinuity at the upper corners through regularization and provides comparison against the Ghia et al. reference data at multiple Reynolds numbers.

**simple_cylinder.py** implements flow around a circular cylinder using CPU computation. The solver applies Zou-He velocity inlet, extrapolation outlet, and bounce-back on the cylinder surface. Force calculation uses the control volume method integrating momentum flux across inlet and outlet planes. This implementation prioritizes stability and correctness for validation purposes.

**gpu_cylinder_flow.py** provides the GPU-accelerated cylinder solver. All operations including boundary conditions execute on the GPU to minimize data transfer overhead. The solver achieves approximately 350 MLUPS for the cylinder configuration, lower than the periodic benchmark due to the additional boundary condition kernels.

### Visualization (visualization/)

The visualization module generates publication-quality figures and animations from simulation results.

**vortex_visualization.py** is the primary visualization driver. It detects GPU availability and selects the appropriate solver, runs simulations to steady state or through the vortex shedding development period, and generates multi-panel figures showing velocity magnitude, vorticity, streamlines, and pressure fields. Animation support creates GIF files showing the temporal evolution of the flow.

Supporting modules handle specific visualization tasks: field_plots.py for contour and pseudocolor plots, streamlines.py for flow path visualization, animation.py for temporal sequences, and validation_plots.py for comparison against reference data.

### Test Suite (tests/)

The test directory contains pytest-compatible validation tests organized by the component or phenomenon being tested.

**test_equilibrium.py** verifies that the equilibrium distribution satisfies the required moment constraints: the zeroth moment equals density, the first moment equals momentum, and the second moment tensor has the correct form for an ideal gas.

**test_conservation.py** confirms that collision preserves mass and momentum to machine precision, and that the complete algorithm conserves these quantities in periodic domains without forcing.

**test_poiseuille.py** compares the developed velocity profile against the analytical parabolic solution, testing the collision operator and wall boundary conditions in combination.

**test_lid_driven.py** compares centerline velocity profiles against the Ghia benchmark data, validating the moving wall boundary condition implementation.

**test_cylinder.py** verifies drag coefficient predictions against literature values for the cylinder flow configuration.

**test_gpu.py** ensures GPU kernels produce identical results to CPU implementations within floating-point tolerance, and measures performance to detect regressions.

## Data Flow

A typical simulation proceeds through the following stages, with data flowing between modules as indicated.

Initialization begins in the solver constructor. The lattice module provides geometric constants, the boundary module creates solid masks from geometric parameters, and the equilibrium module generates the initial distribution from specified density and velocity fields. For GPU solvers, arrays are transferred to device memory at this stage.

The main time-stepping loop alternates between collision and streaming. The collision module reads the current distribution array and writes the post-collision distribution either in-place or to a separate buffer depending on the streaming algorithm. The streaming module propagates distributions to neighboring nodes. Boundary modules then modify distributions at domain edges and solid surfaces.

Periodically during the simulation, the observables module computes macroscopic fields for output or analysis. Force calculation integrates these fields across control surfaces. The visualization module consumes the resulting arrays to generate figures.

## Extension Points

The architecture supports extension through several mechanisms.

New collision operators can be added to the collision module following the existing function signatures. The MRT operator would accept additional relaxation parameters while maintaining the same input/output interface.

Additional boundary conditions require implementing the appropriate distribution modifications in the boundary module. Complex geometries can be handled through the existing mask-based approach or through more sophisticated interpolated bounce-back schemes.

Three-dimensional implementations would extend the lattice module with D3Q15, D3Q19, or D3Q27 velocity sets while maintaining the same modular structure. Kernel implementations would require adaptation for the additional memory traffic and changed access patterns.

Multi-GPU support would add a communication layer between the streaming and boundary stages, exchanging halo regions between GPUs using CUDA-aware MPI or NCCL primitives.
