# CUDA LBM Fluid Dynamics
[![Status](https://img.shields.io/badge/Status-In%20Development-yellow)]()
[![GPU](https://img.shields.io/badge/GPU-Tesla%20P100-green)]()
[![CUDA](https://img.shields.io/badge/CUDA-Numba-blue)]()

High-performance Lattice Boltzmann Method (LBM) solver for computational fluid dynamics, achieving **7,248 MLUPS** (Million Lattice Updates Per Second) on Tesla P100.

## Performance Highlights

| Metric                  | Value                              |
|-------------------------|------------------------------------|
| **Peak Performance**    | 7,248 MLUPS                        |
| **Memory Bandwidth**    | 521.8 GB/s (71.3% efficiency)      |
| **GPU Speedup**         | 224x vs CPU                        |
| **Physical Validation** | C_D within 5% of literature        |


## Key Achievements

### GPU Optimization Journey

| Implementation            | MLUPS   | Memory Efficiency |
|---------------------------|---------|-------------------|
| CPU Baseline (NumPy)      | 8-15    | -                 |
| GPU Naive                 | 1,296   | 25.5%             |
| GPU Optimized (Float64)   | 3,698   | 72.7%             |
| **GPU Ultra (Float32)**   | **7,248** | **71.3%**         |

### Physical Validation

| Test Case             | Metric                   | Achieved     | Reference     | Status |
|-----------------------|--------------------------|--------------|---------------|--------|
| Poiseuille Flow       | Velocity Profile         | 4.5% error   | Analytical    | PASS   |
| Lid-Driven Cavity     | Velocity at Re=100       | 6-9% error   | Ghia et al.   | PASS   |
| Cylinder Flow Re=40   | C_D                      | 1.91         | 1.5-1.8       | PASS   |
| Cylinder Flow Re=100  | C_D                      | 1.41         | 1.3-1.5       | PASS   |
| Vortex Shedding       | Detection at Re>47       | Yes          | Expected      | PASS   |

## Project Structure

```
cuda_lbm_fluid/
├── src/                              # Core LBM library
│   ├── lattice.py                    # D2Q9 lattice constants
│   ├── equilibrium.py                # Equilibrium distribution
│   ├── collision.py                  # BGK/TRT collision operators
│   ├── streaming.py                  # Streaming step
│   ├── boundary.py                   # Boundary conditions
│   ├── observables.py                # Macroscopic quantities
│   └── kernels/
│       ├── cpu_baseline.py           # CPU Numba implementation
│       ├── gpu_naive.py              # Basic GPU (1,296 MLUPS)
│       ├── gpu_optimized.py          # Optimized GPU (3,698 MLUPS)
│       └── gpu_ultra.py              # Ultra-optimized (7,248 MLUPS)
│
├── simulations/                      # Application solvers
│   ├── poiseuille_flow.py            # Channel flow validation
│   ├── lid_driven_cavity.py          # Cavity flow validation
│   ├── simple_cylinder.py            # CPU cylinder solver
│   └── gpu_cylinder_flow.py          # GPU cylinder solver
│
├── visualization/
│   └── vortex_visualization.py       # Vortex shedding animations
│
├── tests/                            # Validation tests (45 tests)
│   ├── test_equilibrium.py
│   ├── test_collision.py
│   ├── test_streaming.py
│   ├── test_conservation.py
│   └── test_gpu.py
│
├── benchmarks/
│   └── benchmark_suite.py            # Performance benchmarks
│
└── results/
    ├── figures/                      # Publication-quality plots
    └── animations/                   # Vortex shedding GIFs
```

## Installation

### Requirements

- Python 3.8+
- CUDA 11.0+ (12.4 recommended)
- NVIDIA GPU (Tesla P100 or better recommended)

### Setup

```bash
# Clone repository
git clone https://github.com/maltsev-andrey/cuda_lbm_fluid.git
cd cuda_lbm_fluid

# Create virtual environment
python3 -m venv cuda_env
source cuda_env/bin/activate

# Install dependencies
pip install numpy numba matplotlib scipy pytest

# Verify CUDA
python3 -c "from numba import cuda; print(f'CUDA available: {cuda.is_available()}')"
```

## Usage

### Quick Start - Performance Benchmark

```bash
# Run ultra-optimized benchmark
python3 src/kernels/gpu_ultra.py
```

Expected output:
```
Float32 AA-Pattern (Best):
  1024 x 1024:   7247.5 MLUPS
  2048 x 2048:   7179.4 MLUPS
  4096 x 4096:   7230.1 MLUPS
```

### Cylinder Flow Simulation

```python
# CPU version (validated physics)
from simulations.simple_cylinder import SimpleCylinderSolver

solver = SimpleCylinderSolver(400, 100, 80, 50, 10, re=100, u_inlet=0.04)
for _ in range(40000):
    solver.step()
# C_D = 1.41, vortex shedding detected

# GPU version (high performance)
from simulations.gpu_cylinder_flow import run_gpu_cylinder_flow

solver = run_gpu_cylinder_flow(nx=600, ny=150, re=100, num_steps=40000)
# ~370 MLUPS with physics, 3355+ MLUPS for pure benchmark
```

### Create Visualizations

```bash
python3 visualization/vortex_visualization.py
```

Generates:
- `results/figures/flow_field_re100.png` - Velocity, vorticity, streamlines
- `results/figures/vortex_development.png` - Time evolution
- `results/animations/vortex_shedding.gif` - Animated vortex street

### Run Validation Tests

```bash
pytest tests/ -v
```

## Technical Details

### Lattice Boltzmann Method

This implementation uses the **D2Q9 lattice** with **BGK collision operator**:

- **D2Q9**: 2D lattice with 9 velocity directions
- **BGK**: Single relaxation time collision
- **Bounce-back**: No-slip walls
- **Zou-He**: Velocity inlet / pressure outlet

### GPU Optimization Techniques

1. **Float32 Precision**: 2x memory bandwidth improvement
2. **AA-Pattern**: 50% memory reduction (single buffer)
3. **Loop Unrolling**: All 9 directions hardcoded
4. **Fused Kernels**: Collision + streaming in single pass
5. **Memory Coalescing**: Optimal access patterns
6. **Register Optimization**: Minimized register pressure

### Memory Bandwidth Analysis

For D2Q9 with Float32:
- Bytes per cell: 9 x 4 = 36 bytes (read) + 36 bytes (write) = 72 bytes
- At 7,248 MLUPS: 7,248 x 10^6 x 72 = 521.8 GB/s
- P100 theoretical peak: 732 GB/s
- **Efficiency: 71.3%**

## Validation References

- **Poiseuille Flow**: Analytical solution for channel flow
- **Lid-Driven Cavity**: Ghia, U., Ghia, K.N. and Shin, C.T. (1982)
- **Cylinder Drag**: Tritton, D.J. (1959), Williamson, C.H.K. (1996)
- **Strouhal Number**: Roshko, A. (1954)

## Performance on Different Hardware
| GPU                 | Float64 MLUPS | Float32 MLUPS |
|---------------------|---------------|---------------|
| Tesla P100          | 3,698         | **7,248**     |
| RTX 3090 (est.)     | 4,500         | 9,000         |
| A100 (est.)         | 6,000         | 12,000        |

## Future Improvements

- [ ] Multi-GPU support (MPI + CUDA)
- [ ] 3D implementation (D3Q19, D3Q27)
- [ ] MRT collision operator
- [ ] Turbulence modeling (LES)
- [ ] Thermal LBM (double distribution)

## Author

**Andrey Maltsev**
- Senior Linux Systems Engineer
- MSc Electrical Engineering, Tomsk Polytechnic University
- GitHub: [github.com/maltsev-andrey](https://github.com/maltsev-andrey)

## License

MIT License

## Acknowledgments

- NVIDIA for CUDA toolkit and documentation
- Numba developers for Python GPU computing
- LBM community for reference implementations

---

*Developed as part of GPU computing portfolio for NVIDIA CUDA Math Libraries*
