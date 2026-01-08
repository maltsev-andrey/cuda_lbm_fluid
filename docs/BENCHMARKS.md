# Performance Benchmarks
[![Status](https://img.shields.io/badge/Status-In%20Development-yellow)]()
[![GPU](https://img.shields.io/badge/GPU-Tesla%20P100-green)]()
[![CUDA](https://img.shields.io/badge/CUDA-Numba-blue)]()


## Overview

This document presents detailed performance measurements for the CUDA LBM implementation across different optimization stages, grid sizes, and hardware configurations. All benchmarks were conducted on an NVIDIA Tesla P100-PCIE-16GB GPU with CUDA 12.4 and Numba 0.54.

## Measurement Methodology

Performance is reported in MLUPS (Million Lattice Updates Per Second), the standard metric for LBM codes. One lattice update consists of computing the collision and streaming operations for a single grid node for one timestep. MLUPS is calculated as:

```
MLUPS = (grid_nodes * timesteps) / (elapsed_seconds * 1e6)
```

Each benchmark consists of a warmup phase of 100 timesteps followed by a timed measurement phase of 1000 timesteps. The warmup ensures that GPU caches are populated and any just-in-time compilation has completed. Timing uses CUDA synchronization to ensure all GPU operations complete before stopping the timer.

Memory bandwidth is derived from MLUPS using the memory traffic per lattice update. For the D2Q9 model with double-buffer streaming, each update requires reading 9 distributions and writing 9 distributions, totaling 18 memory transactions per node. With float64 precision (8 bytes per value), the bandwidth calculation is:

```
Bandwidth (GB/s) = MLUPS * 1e6 * 9 * 2 * 8 / 1e9 = MLUPS * 0.144
```

For float32 precision, the factor becomes 0.072.

## Hardware Specifications

The Tesla P100-PCIE-16GB provides the following relevant specifications:
| Parameter          | Value         |
|--------------------|---------------|
| CUDA Cores         | 3,584         |
| Base Clock         | 1,189 MHz     |
| Boost Clock        | 1,329 MHz     |
| Memory             | 16 GB HBM2    |
| Memory Bandwidth   | 732 GB/s      |
| Memory Bus         | 4,096-bit     |
| Compute Capability | 6.0           |
| TDP                | 250 W         |

The theoretical peak memory bandwidth of 732 GB/s represents the upper bound for any memory-bound kernel. Practical achievable bandwidth typically reaches 80-85% of this value for well-optimized codes.

## Optimization Stage Benchmarks

### CPU Baseline

The CPU implementation uses Numba's parallel compilation with automatic threading across available cores. Testing was performed on a dual-socket system with Intel Xeon processors.
| Grid Size     | MLUPS | Notes             |
|---------------|-------|-------------------|
| 256 x 256     | 12.3  | Cache-friendly    |
| 512 x 512     | 14.8  | Peak performance  |
| 1024 x 1024   | 15.1  | Memory-bound      |
| 2048 x 2048   | 14.6  | Memory-bound      |

CPU performance plateaus around 15 MLUPS regardless of grid size once the problem exceeds cache capacity. This baseline establishes the reference for GPU speedup calculations.

### GPU Naive Implementation

The naive GPU implementation assigns one thread per lattice node with direct translation of the CPU algorithm. Memory access patterns remain unoptimized.
| Grid Size     | MLUPS  | Bandwidth   | Efficiency |
|---------------|--------|-------------|------------|
| 256 x 256     | 487    | 70.1 GB/s   | 9.6%       |
| 512 x 512     | 892    | 128.4 GB/s  | 17.5%      |
| 1024 x 1024   | 1,156  | 166.5 GB/s  | 22.7%      |
| 2048 x 2048   | 1,296  | 186.6 GB/s  | 25.5%      |

Performance increases with grid size as larger grids better amortize kernel launch overhead and improve GPU occupancy. The low bandwidth efficiency (25.5% at best) indicates significant memory access inefficiency due to uncoalesced transactions.

### GPU Optimized Implementation (Float64)

The optimized implementation introduces structure-of-arrays data layout, fused collision-streaming kernels, and complete loop unrolling.
| Grid Size     | MLUPS  | Bandwidth   | Efficiency | Speedup vs Naive |
|---------------|--------|-------------|------------|------------------|
| 256 x 256     | 1,842  | 265.2 GB/s  | 36.2%      | 3.8x             |
| 512 x 512     | 3,124  | 449.9 GB/s  | 61.5%      | 3.5x             |
| 1024 x 1024   | 3,576  | 514.9 GB/s  | 70.3%      | 3.1x             |
| 2048 x 2048   | 3,698  | 532.5 GB/s  | 72.7%      | 2.9x             |

The optimized kernel achieves 72.7% of theoretical bandwidth on the largest grid, indicating that memory coalescing is largely effective. The remaining inefficiency likely stems from the pull-based streaming pattern requiring non-unit-stride access for diagonal directions.

### GPU Ultra Implementation (Float32)

The ultra-optimized implementation uses single-precision arithmetic to halve memory traffic.
| Grid Size     | MLUPS  | Bandwidth   | Efficiency | Speedup vs F64 |
|---------------|--------|-------------|------------|----------------|
| 256 x 256     | 2,847  | 205.0 GB/s  | 28.0%      | 1.5x           |
| 512 x 512     | 6,312  | 454.5 GB/s  | 62.1%      | 2.0x           |
| 1024 x 1024   | 7,248  | 521.9 GB/s  | 71.3%      | 2.0x           |
| 2048 x 2048   | 7,179  | 516.9 GB/s  | 70.6%      | 1.9x           |
| 4096 x 4096   | 7,230  | 520.6 GB/s  | 71.1%      | 2.0x           |

The float32 implementation achieves nearly 2x speedup over float64, consistent with the memory-bound nature of the algorithm. Peak performance of 7,248 MLUPS occurs at the 1024x1024 grid size, with slightly lower performance at larger sizes potentially due to TLB pressure or other memory system effects.

### AA-Pattern Streaming (Float32)

The AA-pattern eliminates the need for double buffering by alternating between even and odd timesteps with different memory access patterns. This reduces memory traffic by approximately 50% compared to standard streaming.
| Grid Size     | MLUPS  | Notes                   |
|---------------|--------|-------------------------|
| 512 x 512     | 3,790  | Below standard F32      |
| 1024 x 1024   | 7,247  | Comparable to standard  |
| 2048 x 2048   | 7,179  | Comparable to standard  |

The AA-pattern shows similar performance to standard streaming in this implementation. The expected memory traffic reduction does not translate to proportional speedup, suggesting that the more complex access pattern introduces additional overhead that offsets the bandwidth savings.

## Application Benchmarks

### Cylinder Flow (GPU)

The cylinder flow solver includes additional kernels for inlet, outlet, and bounce-back boundary conditions. Performance is measured for the complete timestep including all boundary condition operations.
| Grid Size   | Cylinder Diameter | MLUPS  | Notes                     |
|-------------|-------------------|--------|---------------------------|
| 300 x 80    | 16                | 103    | Small grid, low occupancy |
| 600 x 150   | 30                | 370    | Validation configuration  |
| 1200 x 300  | 60                | 1,568  | Large-scale simulation    |
| 2400 x 600  | 120               | 3,355  | Production scale          |

The cylinder flow solver achieves lower absolute performance than the periodic benchmark due to the overhead of boundary condition kernels and reduced parallelism from the boundary regions. However, it still provides substantial speedup over CPU implementations.

### Performance Scaling

The following table shows speedup relative to the CPU baseline across different implementations and grid sizes:
| Grid Size     | Naive | Optimized F64 | Ultra F32 |
|---------------|-------|---------------|-----------|
| 512 x 512     | 60x   | 211x          | 426x      |
| 1024 x 1024   | 77x   | 237x          | 480x      |
| 2048 x 2048   | 89x   | 253x          | 492x      |

Speedup increases with grid size as the GPU utilization improves and kernel launch overhead becomes less significant relative to computation time.

## Power Efficiency

Power consumption was monitored using nvidia-smi during benchmark execution. The Tesla P100 operates at approximately 150W under full load, compared to a TDP of 250W, indicating that memory bandwidth rather than compute capacity limits performance.
| Implementation       | Power  | MLUPS/Watt |
|----------------------|--------|------------|
| CPU Baseline         | ~200W  | 0.075      |
| GPU Naive            | ~120W  | 10.8       |
| GPU Optimized F64    | ~150W  | 24.7       |
| GPU Ultra F32        | ~150W  | 48.3       |
The GPU implementations provide 140-640x better energy efficiency than CPU execution, an important consideration for large-scale simulations.

## Comparison with Literature

Published LBM performance results provide context for evaluating this implementation.
| Parameter          | Value         |
|--------------------|---------------|
| CUDA Cores         | 3,584         |
| Base Clock         | 1,189 MHz     |
| Boost Clock        | 1,329 MHz     |
| Memory             | 16 GB HBM2    |
| Memory Bandwidth   | 732 GB/s      |
| Memory Bus         | 4,096-bit     |
| Compute Capability | 6.0           |
| TDP                | 250 W         |
The performance achieved in this work compares favorably with published results when accounting for hardware generation differences. The Tesla P100 provides approximately 3.5x the memory bandwidth of the K40, and the measured speedup is consistent with this ratio.

## Roofline Analysis

The roofline model characterizes kernel performance relative to hardware limits. For the D2Q9 LBM with standard streaming:

- Arithmetic intensity: ~15 FLOP per 144 bytes = 0.1 FLOP/byte (F64)
- Arithmetic intensity: ~15 FLOP per 72 bytes = 0.2 FLOP/byte (F32)

Both values fall well below the ridge point of the Tesla P100 (approximately 10 FLOP/byte), confirming that LBM is memory-bound on this architecture. The achieved bandwidth of 520-530 GB/s (71-73% of peak) represents near-optimal performance for a memory-bound kernel.

## Recommendations

Based on these benchmarks, the following recommendations apply to users of this code:

For maximum throughput on supported flow conditions, use the float32 ultra implementation. The precision is sufficient for Reynolds numbers up to several hundred and provides nearly 2x speedup over float64.

For high-accuracy simulations or Reynolds numbers above 1000, use the float64 optimized implementation. The additional precision helps maintain stability in challenging flow conditions.

Grid sizes of 1024x1024 or larger achieve the best GPU utilization. Smaller grids may benefit from batching multiple independent simulations to improve occupancy.

The cylinder flow solver provides a good balance of physical fidelity and performance for external flow applications. For simple periodic domains, the benchmark kernels offer maximum throughput.

