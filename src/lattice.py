"""
D2Q9 Lattice Constants and Utilities

Defines the D2Q9 lattice model for 2D fluid simulations.
"""
import numpy as np

# D2Q9 lattice velocities
#     6   2   5
#       \ | /
#     3 - 0 - 1
#       / | \
#     7   4   8

# Lattice velocity components
EX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.int32)
EY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.int32)

# Lattice weights
W = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=np.float64)

# Opposite direction indices (for bounce-back)
OPPOSITE = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)

# Lattice sound speed squared
CS2 = 1.0 / 3.0
CS4 = CS2 * CS2

# Number of lattice velocities
Q = 9