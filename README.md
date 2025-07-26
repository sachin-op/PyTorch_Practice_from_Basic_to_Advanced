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

# 2. pytorch_practice_workflow_code.py

This file walks through a complete **PyTorch machine learning workflow** using a simple linear regression task. It demonstrates how to prepare data, build a model, train it, evaluate performance, and save/load trained models — all with clean, reproducible, and device-agnostic code.

---

## Topics Covered

Each section covers an essential part of the typical ML workflow in PyTorch, suitable for beginners and intermediate users.

---

### 1. Synthetic Data Generation

- Create data using a known linear function:  
  \[
  y = 0.7x + 0.3
  \]
- Use `torch.arange()` to create `X` and generate `y`
- Split data into training and testing sets

---

### 2. Model Building (Two Approaches)

- **Manual model (`LinearRegressionModel`)**: uses `nn.Parameter` for weights/bias
- **Standard model (`LinearRegressionModelV2`)**: uses `nn.Linear`

Each model defines a custom `forward()` method.

---

### 3. Training Loop

- Uses:
  - `model.train()`
  - `loss_fn = nn.L1Loss()` (MAE)
  - `optimizer = torch.optim.SGD()`
- Typical training steps:
  1. Forward pass
  2. Compute loss
  3. Zero gradients
  4. Backward pass
  5. Update weights

- Plots training and test loss over epochs

---

### 4. Testing Loop

- Set model to `.eval()`
- Disable gradient tracking with `torch.inference_mode()`
- Run predictions on test data
- Evaluate performance using MAE loss

---

### 5. Making Predictions

- Once trained, perform predictions on unseen data
- Use `.eval()` and `with torch.inference_mode():`
- Visualize predicted vs actual values with Matplotlib

---

### 6. Saving & Loading Models

- Save the model with:
  ```python
  torch.save(model.state_dict(), "models/model_1.pth")



