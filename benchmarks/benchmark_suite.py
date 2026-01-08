"""
Comprehensive Benchmark Suite

Performance testing for all LBM implementations.
Compares CPU baseline, GPU naive, and optimized GPU versions.
"""

import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Check CUDA availability
try:
    from numba import cuda
    CUDA_AVAILABLE = cuda.is_available()
except:
    CUDA_AVAILABLE = False


def benchmark_cpu(nx, ny, tau, num_steps, warmup_steps=50):
    """
    Benchmark CPU LBM solver.
    
    Returns
    -------
    mlups : float
        Million Lattice Updates Per Second
    """
    from src.equilibrium import compute_equilibrium_fast
    from src.observables import compute_macroscopic_fast
    from src.collision import bgk_collision_fast
    from src.streaming import stream_periodic_fast
    
    # Initialize
    rho = np.ones((ny, nx), dtype=np.float64)
    ux = np.zeros((ny, nx), dtype=np.float64)
    uy = np.zeros((ny, nx), dtype=np.float64)
    f = compute_equilibrium_fast(rho, ux, uy)
    
    # Warmup
    for _ in range(warmup_steps):
        rho, ux, uy = compute_macroscopic_fast(f)
        f_eq = compute_equilibrium_fast(rho, ux, uy)
        f = bgk_collision_fast(f, f_eq, tau)
        f = stream_periodic_fast(f)
    
    # Timed run
    start = time.perf_counter()
    for _ in range(num_steps):
        rho, ux, uy = compute_macroscopic_fast(f)
        f_eq = compute_equilibrium_fast(rho, ux, uy)
        f = bgk_collision_fast(f, f_eq, tau)
        f = stream_periodic_fast(f)
    elapsed = time.perf_counter() - start
    
    mlups = num_steps * nx * ny / elapsed / 1e6
    return mlups


def benchmark_gpu_naive(nx, ny, tau, num_steps, warmup_steps=50, use_fused=True):
    """
    Benchmark naive GPU LBM solver.
    """
    if not CUDA_AVAILABLE:
        return 0.0
    
    from src.kernels.gpu_naive import GPULBMSolver
    
    solver = GPULBMSolver(nx, ny, tau)
    step_func = solver.step_fused if use_fused else solver.step_separate
    
    # Warmup
    for _ in range(warmup_steps):
        step_func()
    cuda.synchronize()
    
    # Timed run
    start = time.perf_counter()
    for _ in range(num_steps):
        step_func()
    cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    mlups = num_steps * nx * ny / elapsed / 1e6
    return mlups


def benchmark_gpu_optimized(nx, ny, tau, num_steps, warmup_steps=100, kernel_type='fused'):
    """
    Benchmark optimized GPU LBM solver.
    """
    if not CUDA_AVAILABLE:
        return 0.0
    
    from src.kernels.gpu_optimized import OptimizedGPULBMSolver
    
    solver = OptimizedGPULBMSolver(nx, ny, tau, kernel_type)
    
    # Warmup
    for _ in range(warmup_steps):
        solver.step()
    cuda.synchronize()
    
    # Timed run
    start = time.perf_counter()
    for _ in range(num_steps):
        solver.step()
    cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    mlups = num_steps * nx * ny / elapsed / 1e6
    return mlups


def run_full_benchmark(grid_sizes=None, tau=0.8, num_steps=1000):
    """
    Run complete benchmark suite comparing all implementations.
    """
    if grid_sizes is None:
        grid_sizes = [
            (64, 64),
            (128, 128),
            (256, 256),
            (512, 512),
            (1024, 1024),
            (2048, 2048),
        ]
    
    print("=" * 80)
    print("LBM Performance Benchmark Suite - All Implementations")
    print("=" * 80)
    print(f"Tau: {tau}")
    print(f"Steps: {num_steps}")
    print(f"CUDA Available: {CUDA_AVAILABLE}")
    print()
    
    results = {}
    
    # CPU Benchmark (limited to smaller grids)
    cpu_grids = [(nx, ny) for nx, ny in grid_sizes if nx <= 512]
    print("Benchmarking CPU (Numba parallel)...")
    print("-" * 40)
    results['cpu'] = {}
    for nx, ny in cpu_grids:
        try:
            mlups = benchmark_cpu(nx, ny, tau, num_steps)
            results['cpu'][(nx, ny)] = mlups
            print(f"  {nx:4d} x {ny:4d}: {mlups:8.2f} MLUPS")
        except Exception as e:
            print(f"  {nx:4d} x {ny:4d}: Error - {e}")
            results['cpu'][(nx, ny)] = 0.0
    print()
    
    if CUDA_AVAILABLE:
        # GPU Naive Fused
        print("Benchmarking GPU Naive (fused kernel)...")
        print("-" * 40)
        results['gpu_naive'] = {}
        for nx, ny in grid_sizes:
            try:
                mlups = benchmark_gpu_naive(nx, ny, tau, num_steps, use_fused=True)
                results['gpu_naive'][(nx, ny)] = mlups
                print(f"  {nx:4d} x {ny:4d}: {mlups:8.2f} MLUPS")
            except Exception as e:
                print(f"  {nx:4d} x {ny:4d}: Error - {e}")
                results['gpu_naive'][(nx, ny)] = 0.0
        print()
        
        # GPU Optimized Fused
        print("Benchmarking GPU Optimized (fused + unrolled)...")
        print("-" * 40)
        results['gpu_opt_fused'] = {}
        for nx, ny in grid_sizes:
            try:
                mlups = benchmark_gpu_optimized(nx, ny, tau, num_steps, kernel_type='fused')
                results['gpu_opt_fused'][(nx, ny)] = mlups
                print(f"  {nx:4d} x {ny:4d}: {mlups:8.2f} MLUPS")
            except Exception as e:
                print(f"  {nx:4d} x {ny:4d}: Error - {e}")
                results['gpu_opt_fused'][(nx, ny)] = 0.0
        print()
        
        # GPU Optimized Shared Memory
        print("Benchmarking GPU Optimized (shared memory)...")
        print("-" * 40)
        results['gpu_opt_shared'] = {}
        for nx, ny in grid_sizes:
            try:
                mlups = benchmark_gpu_optimized(nx, ny, tau, num_steps, kernel_type='shared')
                results['gpu_opt_shared'][(nx, ny)] = mlups
                print(f"  {nx:4d} x {ny:4d}: {mlups:8.2f} MLUPS")
            except Exception as e:
                print(f"  {nx:4d} x {ny:4d}: Error - {e}")
                results['gpu_opt_shared'][(nx, ny)] = 0.0
        print()
        
        # GPU Optimized AA-Pattern
        print("Benchmarking GPU Optimized (AA-pattern, half memory)...")
        print("-" * 40)
        results['gpu_opt_aa'] = {}
        for nx, ny in grid_sizes:
            try:
                mlups = benchmark_gpu_optimized(nx, ny, tau, num_steps, kernel_type='aa')
                results['gpu_opt_aa'][(nx, ny)] = mlups
                print(f"  {nx:4d} x {ny:4d}: {mlups:8.2f} MLUPS")
            except Exception as e:
                print(f"  {nx:4d} x {ny:4d}: Error - {e}")
                results['gpu_opt_aa'][(nx, ny)] = 0.0
        print()
    
    # Summary Table
    print("=" * 80)
    print("SUMMARY: Performance Comparison (MLUPS)")
    print("=" * 80)
    
    headers = ['Grid', 'CPU', 'GPU Naive', 'GPU Opt', 'GPU Shared', 'GPU AA', 'Best Speedup']
    header_line = f"{'Grid':<12} {'CPU':>8} {'Naive':>10} {'Opt':>10} {'Shared':>10} {'AA':>10} {'Speedup':>10}"
    print(header_line)
    print("-" * 80)
    
    for nx, ny in grid_sizes:
        cpu = results.get('cpu', {}).get((nx, ny), 0)
        naive = results.get('gpu_naive', {}).get((nx, ny), 0)
        opt = results.get('gpu_opt_fused', {}).get((nx, ny), 0)
        shared = results.get('gpu_opt_shared', {}).get((nx, ny), 0)
        aa = results.get('gpu_opt_aa', {}).get((nx, ny), 0)
        
        best_gpu = max(naive, opt, shared, aa)
        if cpu > 0 and best_gpu > 0:
            speedup = f"{best_gpu / cpu:.0f}x"
        else:
            speedup = "N/A"
        
        print(f"{nx:4d}x{ny:<4d}    {cpu:>8.1f} {naive:>10.1f} {opt:>10.1f} {shared:>10.1f} {aa:>10.1f} {speedup:>10}")
    
    print("=" * 80)
    
    # Find peak performance
    all_mlups = []
    for impl, data in results.items():
        for (nx, ny), mlups in data.items():
            if mlups > 0:
                all_mlups.append((mlups, impl, nx, ny))
    
    if all_mlups:
        all_mlups.sort(reverse=True)
        peak_mlups, peak_impl, peak_nx, peak_ny = all_mlups[0]
        print(f"\nPeak Performance: {peak_mlups:.1f} MLUPS")
        print(f"  Implementation: {peak_impl}")
        print(f"  Grid Size: {peak_nx}x{peak_ny}")
        
        # Memory bandwidth analysis
        bytes_per_site = 9 * 8 * 2  # 9 directions * 8 bytes * 2 (read+write) = 144
        bandwidth_gb = peak_mlups * bytes_per_site / 1000
        theoretical_peak = 732  # GB/s for P100
        efficiency = bandwidth_gb / theoretical_peak * 100
        
        print(f"\nMemory Bandwidth Analysis:")
        print(f"  Effective Bandwidth: {bandwidth_gb:.1f} GB/s")
        print(f"  Theoretical Peak: {theoretical_peak} GB/s")
        print(f"  Memory Efficiency: {efficiency:.1f}%")
    
    print("=" * 80)
    
    return results


def compute_memory_bandwidth(mlups, bytes_per_site=144):
    """Compute effective memory bandwidth from MLUPS."""
    return mlups * bytes_per_site / 1000


def print_memory_analysis(results, theoretical_peak_gb_s=732):
    """Print memory bandwidth analysis."""
    print()
    print("Memory Bandwidth Analysis (Best GPU Results)")
    print("=" * 60)
    print(f"Theoretical Peak: {theoretical_peak_gb_s} GB/s")
    print()
    print(f"{'Grid':<12} {'MLUPS':>10} {'Bandwidth':>12} {'Efficiency':>12}")
    print("-" * 60)
    
    # Get best GPU result for each grid size
    grid_sizes = set()
    for impl, data in results.items():
        if 'gpu' in impl:
            for gs in data.keys():
                grid_sizes.add(gs)
    
    for (nx, ny) in sorted(grid_sizes):
        best_mlups = 0
        for impl, data in results.items():
            if 'gpu' in impl:
                mlups = data.get((nx, ny), 0)
                best_mlups = max(best_mlups, mlups)
        
        if best_mlups > 0:
            bandwidth = compute_memory_bandwidth(best_mlups)
            efficiency = bandwidth / theoretical_peak_gb_s * 100
            print(f"{nx:4d}x{ny:<4d}   {best_mlups:>10.1f} {bandwidth:>10.1f} GB/s {efficiency:>10.1f}%")
    
    print("=" * 60)


if __name__ == "__main__":
    # Run full benchmark
    results = run_full_benchmark()
    
    # Memory analysis
    if CUDA_AVAILABLE:
        print_memory_analysis(results)