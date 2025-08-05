# `Kernel.py` — Developer Documentation

This module defines optimized mathematical kernels used for transforming molecular geometries and computing probabilistic measures. The functions are performance-accelerated using Numba’s `@jit` and operate primarily on NumPy arrays and OpenMM `Quantity` objects.

These routines are central to aptamer sampling, entropy estimation, and structure perturbation workflows.

## `catchZero(numeric)`

**Name of step**: Numerical stability for zero
**Input object type**: In-memory float or array
**Input object example**: `0.0`, `np.array([0.1, 0.0, 0.2])`
**Output object type**: In-memory float or array
**Output object example**: `1e-50`, `np.array([0.1, 1e-50, 0.2])`
**Steps performed**: Modify by adding `1e-50`
**Package/code**: `Python builtin`, `catchZero(numeric)`

## `rotateKernel(positions, element, axis, angle)`

**Name of step**: Rotate atoms in a local region
**Input object types**:

- `positions` → in-memory NumPy array (`Quantity` with units of angstroms)
- `element` → 3-element tuple like `(residue, start, end)`
- `axis` → in-memory 3D vector
- `angle` → in-memory float
  **Input object example**:
- `positions = Quantity(np.random.rand(20, 3), unit.angstrom)`
- `element = ("A", 5, 10)`
- `axis = [0, 0, 1]`, `angle = np.pi/3`
  **Output object type**: `Quantity` (positions in angstroms)
  **Output object example**: `Quantity(np.array([...]), unit.angstrom)`
  **Steps performed**:
- Shift selected atoms
- Apply rotation matrix
- Return updated coordinates
  **Package/code**: `numba`, `rotateKernel(...)`

## `translateKernel(positions, element, shift)`

**Name of step**: Translate a region of the molecule
**Input object types**:

- `positions` → in-memory NumPy array
- `element` → tuple `(start, mid, end)`
- `shift` → 3D vector
  **Input object example**:
- `positions = np.array([[0,0,0], [1,0,0], ..., [10,0,0]])`
- `element = (5, 6, 10)`, `shift = [1.0, 1.0, 0.0]`
  **Output object type**: in-memory NumPy array
  **Output object example**: `np.array([...])`
  **Steps performed**: Modify: displace a region by shift vector
  **Package/code**: `numba`, `translateKernel(...)`

## `centerOfMass(positions)`

**Name of step**: Compute center of geometry
**Input object type**: in-memory NumPy array
**Input object example**: `positions = np.array([[0,0,0], [1,0,0], [0,1,0]])`
**Output object type**: in-memory 3D vector
**Output object example**: `np.array([0.33, 0.33, 0.0])`
**Steps performed**: Create: average over all position vectors
**Package/code**: `numba`, `centerOfMass(...)`

## `radius(center, positions)`

**Name of step**: Compute max radial distance from center
**Input object types**:

- `center` → 3D NumPy array
- `positions` → 2D NumPy array
  **Input object example**:
- `center = np.array([0,0,0])`, `positions = np.array([[1,0,0],[0,2,0],[0,0,3]])`
  **Output object type**: float
  **Output object example**: `3.0`
  **Steps performed**: Create: compute max norm of displacement vectors
  **Package/code**: `numba`, `radius(...)`

## `kullbackLeiblerDivergenceKernel(sample, reference_sample)`

**Name of step**: Compute KL divergence
**Input object types**:

- `sample` → NumPy array of probabilities
- `reference_sample` → NumPy array of reference probabilities
  **Input object example**:
- `sample = np.array([0.2, 0.3, 0.5])`, `reference_sample = np.array([0.1, 0.4, 0.5])`
  **Output object type**: float (scalar)
  **Output object example**: `0.045`
  **Steps performed**: Create: compute `∑ p log(p/q)`
  **Package/code**: `numba`, `kullbackLeiblerDivergenceKernel(...)`

## `EntropyKernel(sample)`

**Name of step**: Compute Shannon entropy
**Input object type**: in-memory NumPy array of probabilities
**Input object example**: `sample = np.array([0.2, 0.3, 0.5])`
**Output object type**: float
**Output object example**: `1.485`
**Steps performed**: Create: compute `-∑ p log(p)`
**Package/code**: `numba`, `EntropyKernel(...)`

## `ZPS(sample, beta=0.001)`

**Name of step**: Compute partition function, probability distribution, entropy
**Input object types**:

- `sample` → NumPy array of energies
- `beta` → float
  **Input object example**: `sample = np.array([0.2, 0.1, 0.4])`, `beta = 0.001`
  **Output object types**:
- `Z` → float
- `P` → NumPy array (normalized Boltzmann distribution)
- `S` → float
  **Output object examples**:
- `Z = 2.999`, `P = np.array([0.333, 0.333, 0.333])`, `S = 1.098`
  **Steps performed**:
- Create: compute softmax-like distribution from energies
- Normalize
- Compute entropy
  **Package/code**: `numba`, `ZPS(...)`

## `S(sample, beta=0.001)`

**Name of step**: Return only entropy from `ZPS(...)`
**Input object types**:

- `sample` → NumPy array
- `beta` → float
  **Input object example**: `sample = np.array([0.1, 0.5])`, `beta = 0.001`
  **Output object type**: float
  **Output object example**: `S = 0.69`
  **Steps performed**: Extract `S` from result of `ZPS()`
  **Package/code**: `numba`, `S(...)`
