# 1.pytorch_practice_basic_code.py

This file contains a beginner-friendly walkthrough of **PyTorch fundamentals**, covering tensor creation, operations, reshaping, GPU usage, and reproducibility — all crucial concepts for anyone starting out with deep learning using PyTorch.

---

## Topics Covered

Each section corresponds to a core PyTorch concept, demonstrated with code and inline comments for better understanding.

---

###  1. Tensor Basics
- Creating scalars, vectors, matrices, and tensors using:
  - `torch.tensor()`
  - `torch.zeros()`, `torch.ones()`, `torch.arange()`, `torch.rand()`
- Checking:
  - `tensor.shape`
  - `tensor.dtype`
  - `tensor.device`

---

###  2. Tensor Operations
- Element-wise operations:
  - `+`, `-`, `*`, `/`
  - `torch.add()`, `torch.sub()`, `torch.mul()`, `torch.div()`
- Matrix Multiplication:
  - `torch.matmul()`
  - `@` operator
- Matrix shape compatibility & fixing using `.T` (transpose)

---

###  3. Tensor Manipulation
- `reshape()` – Reshape tensor into a new shape
- `view()` – Reshape tensor sharing the same memory
- `stack()` – Stack tensors vertically/horizontally
- `squeeze()` – Remove dimensions of size 1
- `unsqueeze()` – Add dimensions of size 1
- `permute()` – Rearrange tensor dimensions (e.g., HWC ➝ CHW)

---

###  4. NumPy ↔ PyTorch Interoperability
- NumPy → PyTorch: `torch.from_numpy(ndarray)`
- PyTorch → NumPy: `tensor.numpy()`
- Notes on shared memory and data types (`float64` vs `float32`)

---

###  5. Randomness & Reproducibility
- Generating random tensors: `torch.rand()`
- Set manual seed: `torch.manual_seed(seed)`
- Difference between reproducible and non-reproducible tensors

---

###  6. Running on GPU (and Faster Computing)
- Check for GPU:
  - `torch.cuda.is_available()`
  - `torch.backends.mps.is_available()` (Apple Silicon)
- Device-agnostic code setup:
  ```python
  device = "cuda" if torch.cuda.is_available() else "cpu"
