from simulations.gpu_cylinder_flow import run_gpu_cylinder_flow
solver = run_gpu_cylinder_flow(nx=600, ny=150, re=100, num_steps=40000)
