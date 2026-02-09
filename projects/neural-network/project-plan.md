# Neural Network from Scratch - Implementation Plan

## Project Overview

**Goal**: Build a rigorous, portfolio-quality Neural Network implementation from scratch (NumPy only) that demonstrates deep understanding of ML fundamentals while showcasing software engineering skills.

**Strategic Positioning**:
- **RAG Project**: Demonstrates systems engineering (Docker, evaluation frameworks, production deployment)
- **NN Project**: Demonstrates theory fundamentals (backprop math, gradient derivations, optimization algorithms)
- **Together**: Positions candidate as "complete package" who understands both building systems AND the underlying math

**Timeline**: Week 6 Day 2 - Week 7 Day 2 (12 hours total over 5 days)

**Scope**: Hybrid quality approach
- ✅ Include: Xavier/He init, numerical gradient checking, Adam optimizer, modular PyTorch-like design
- ❌ Skip: Advanced visualizations (loss landscapes, gradient histograms) - time-intensive, lower ROI

---

## Implementation Plan

### Day 1 (Week 6 Day 2, Dec 2) - 2.5 hours ✅ COMPLETE

**Goal**: Build modular forward pass with proper initialization

**Tasks**:
1. **Project Setup** (15 min) ✅
   - Create `projects/neural-network/` directory structure
   - Set up `requirements.txt` (numpy, matplotlib, scikit-learn for MNIST)
   - Create skeleton files: `nn/module.py`, `nn/layers.py`, `nn/activations.py`, `nn/losses.py`, `nn/initializers.py`

2. **Module Base Class** (30 min) ✅
   - Implement PyTorch-like `Module` base class with flexible signatures
   - Used `*args` for duck typing (layers take 1 arg, losses take 2 args)
   - Public `parameters` and `gradients` dicts (no getter methods)
   - Type unions: `np.ndarray | float` for flexibility
   - **Design insight**: Loss functions as Modules (PyTorch pattern for consistency)

3. **Initializers** (20 min) ✅
   - Implement Xavier/Glorot initialization: `Uniform(-√(6/(fan_in+fan_out)), √(6/(fan_in+fan_out)))`
   - Implement He initialization: `Normal(0, √(2/fan_in))`
   - **Interview point**: "Xavier for sigmoid/tanh, He for ReLU" (vanishing gradient problem)
   - **Gap identified**: Knew pairing but not underlying math (70% on knowledge check)

4. **Linear Layer** (30 min) ✅
   - Implement `Linear(in_features, out_features, init='xavier')` class
   - Forward: `y = x @ W + b`
   - Store `x` for backward pass
   - Initialize weights with chosen strategy
   - **Bug fixed**: Added bias gradient in backward pass

5. **Activation Functions** (25 min) ✅
   - ReLU: `forward = max(0, x)`, `backward = (x > 0) * grad`
   - Sigmoid: `forward = 1/(1+exp(-x))`, `backward = sigmoid(x) * (1-sigmoid(x)) * grad`
   - Softmax: Implemented numerically stable version (subtract max before exp)

6. **Loss Functions** (30 min) ✅
   - CrossEntropyLoss (forward returns float, backward with batch normalization)
   - MSE (for testing)
   - Both include epsilon (1e-8) for numerical stability
   - **Bug fixed**: Missing batch_size normalization in backward pass

**Deliverable**: Modular forward pass working on toy XOR data ✅
- Test: 2-layer network (2→4→2) with ReLU, forward pass produces output
- Test successful: `PYTHONPATH=. python tests/test_forward.py` produces 4x2 output

**Files Created**: ✅
```
projects/neural-network/
├── nn/
│   ├── __init__.py       # Package exports
│   ├── module.py         # Base Module class with flexible signatures
│   ├── layers.py         # Linear layer with forward + backward
│   ├── activations.py    # ReLU, Sigmoid, Softmax (forward + backward)
│   ├── losses.py         # CrossEntropyLoss, MSE with epsilon
│   └── initializers.py   # Xavier, He (plain functions)
└── tests/
    └── test_forward.py   # Quick validation tests

references/
└── Week6-Day2-NN-Day1-Reference.md  # Quick reference sheet
```

**Knowledge Check Results**: 92.5% (A-)
- Action: Reinforce He/Xavier math during Day 2 numerical gradient checking

---

## Project Resumed: Feb 4, 2026

**Context**:
- Paused Dec 2, 2025 (Day 1 complete, 60% done)
- Resuming for learning closure, not portfolio need
- Discovered Linear.backward() was already implemented on Day 1 ✅
- Fixed Softmax backward implementation (Feb 3)

**Current Status**:
- ✅ Linear.backward() - Already done (layers.py:27-31)
- ✅ ReLU.backward() - Already done (activations.py:14-16)
- ✅ Sigmoid.backward() - Already done (activations.py:27-28)
- ✅ Softmax.forward() - Fixed to save self.output (activations.py:39)
- ✅ Softmax.backward() - Implemented Jacobian formula (activations.py:42-48)
- ✅ Vector calculus deep dive - Documented in key-learnings.md

**Key Learnings** (Feb 3):
- Softmax Jacobian: `J[i,j] = s_i * (δ_ij - s_j)` → backward: `s * (grad - grad_sum)`
- Bias gradient: Sum over batch (axis=0) because parameter is shared
- Activation gradient: Sum over features (axis=1) for per-sample Jacobian
- See `key-learnings.md` for detailed derivations and study resources

---

### Day 2 (Resumed Feb 4, 2026) - 3 hours

**Goal**: Numerical gradient verification + vector calculus mastery

**Tasks**:
1. **Linear Layer Backward** ✅ ALREADY DONE (Dec 2)
   - Gradients implemented correctly:
     - `∂L/∂W = x.T @ grad` (batch dimension handled in matmul)
     - `∂L/∂b = sum(grad, axis=0)` (sum over batch)
     - `∂L/∂x = grad @ W.T` (chain rule to previous layer)

2. **Activation Backward** ✅ ALREADY DONE (Dec 2) + FIXED (Feb 3)
   - ReLU: `grad * (x > 0)` ✅
   - Sigmoid: `grad * sigmoid(x) * (1 - sigmoid(x))` ✅
   - Softmax: Fixed Jacobian implementation ✅
     ```python
     s = self.output
     grad_sum = np.sum(grad * s, axis=1, keepdims=True)
     return s * (grad - grad_sum)
     ```

3. **Loss Backward** ✅ ALREADY DONE (Dec 2)
   - CrossEntropyLoss: `grad = -y_true / (y_pred + ε) / batch_size` ✅
   - MSELoss: `grad = 2 * (y_pred - y_true) / batch_size` ✅

4. **Vector Calculus Study** ✅ COMPLETE (Feb 3)
   - Studied Softmax Jacobian derivation
   - Understood axis=0 vs axis=1 (parameter vs activation gradients)
   - Created comprehensive study guide in `key-learnings.md`
   - Recommended resources: Parr & Howard paper, CS231n notes, 3Blue1Brown

5. **Numerical Gradient Checking** ✅ COMPLETE (Feb 5)
   - Implemented `numerical_gradient()` in `nn/utils.py`
   - Created comprehensive tests in `tests/test_gradient_check.py`
   - Coverage: Linear (W, b, x), ReLU, Sigmoid, Softmax, full network
   - All tests pass with relative error < 1e-7

6. **Combined SoftmaxCrossEntropyLoss** ✅ COMPLETE (Feb 6)
   - Implemented in `nn/losses.py` (operates on logits, not softmax outputs)
   - Uses log-sum-exp trick for numerical stability
   - Backward: `grad = (softmax - y_true) / batch_size` (simpler + more stable)
   - Added gradient check tests: `test_cross_entropy_loss_backward()`, `test_softmax_cross_entropy_loss_backward()`
   - Code review: 9/10 - Excellent implementation

7. **End-to-End Test** (15 min) ⭐ **NEXT**
   - Small synthetic dataset (XOR or 2D spiral)
   - Run 10 gradient descent steps
   - Verify loss decreases (don't need convergence yet)

**Deliverable**: Verified backpropagation implementation
- Numerical gradient check passes with relative error < 1e-7
- Can run basic gradient descent (even if not converging yet)

**Files Updated**:
```
nn/
├── layers.py         # Add Linear.backward()
├── activations.py    # Add backward() for each
├── losses.py         # Add backward()
└── utils.py          # NEW: numerical_gradient() function

tests/
├── test_backward.py   # Backprop correctness
└── test_gradients.py  # Numerical gradient checking
```

---

### Day 3 (Resumed Feb 7-9, 2026) - 4 hours ✅ **COMPLETE**

**Goal**: Implement training loop with SGD and Adam optimizers

**Tasks**:
1. **Optimizer Base Class** (20 min) ✅ COMPLETE (Feb 7)
   - PyTorch-style design: stores list of layers
   - Accesses `layer.parameters` and `layer.gradients` dicts
   ```python
   class Optimizer:
       def __init__(self, layers: list[Module]): ...
       def zero_grad(self): ...
       def step(self): raise NotImplementedError
   ```

2. **SGD Optimizer** (25 min) ✅ COMPLETE (Feb 7)
   - Implemented vanilla SGD: `param -= lr * grad`
   - Iterates through layers, then through parameters in each layer
   - Works correctly with activation layers (empty parameters dict)

3. **Adam Optimizer** (45 min) ✅ COMPLETE (Feb 9)
   - Implemented Adam with bias correction (first + second moment)
   - State tracking per parameter: `m` (momentum), `v` (RMSprop), `t` (timestep)
   - Bias correction: `m_hat = m / (1 - beta1^t)`, `v_hat = v / (1 - beta2^t)`
   - Update rule: `param -= lr * m_hat / (sqrt(v_hat) + eps)`
   - Hyperparameters: beta1=0.9, beta2=0.999, eps=1e-8
   - **Interview point**: "Adam combines momentum (first moment) and RMSProp (second moment) with bias correction for early steps"

4. **Basic Training Loop** (30 min) ✅ COMPLETE (Feb 7)
   - Implemented in `train.ipynb`
   - Core functionality: forward → loss → backward → optimizer.step() → zero_grad()
   - Trained on XOR dataset (4 samples, 2 classes)
   - Architecture: Linear(2,4,he) → ReLU → Linear(4,2,xavier) → SoftmaxCrossEntropyLoss

   **SGD Results** (LR=0.1):
   - Loss: 0.666 → 0.00093 (5000 epochs)
   - 100% accuracy on XOR ✅

   **Adam Results** (LR=0.01):
   - Loss: 0.660 → 0.000032 (35× lower than SGD!)
   - 100% accuracy on XOR ✅
   - **Key finding**: Adam 5× better at epoch 500 (0.0044 vs 0.0256) despite 10× smaller LR

   **Advanced features** (moved to Optional Enhancements):
   - Mini-batch processing (not needed for toy datasets)
   - Shuffle data each epoch (not needed for full-batch)
   - Early stopping (nice-to-have)

**Deliverable**: Working training loop + optimizer comparison ✅
- ✅ Loss decreases to near-zero (SGD: 0.00093, Adam: 0.000032)
- ✅ 2-layer network trains successfully on XOR (100% accuracy)
- ✅ Compare SGD vs. Adam: Adam converges 35× better!

**Files Created**:
```
nn/
└── optimizers.py     # Optimizer base class + SGD + Adam ✅

train.ipynb           # Training loop + XOR test + SGD/Adam comparison ✅
```

**Key Learnings** (Feb 7, Feb 9):
- **Optimizer design**: PyTorch-style (store layers) is cleaner than storing separate param/grad lists
- **Initialization matters**: He for layers before ReLU, Xavier for output layer before softmax
- **Inference pattern**: Network outputs logits, apply softmax separately (or just argmax for predictions)
- **Gradient behavior**: Current implementation overwrites gradients (not accumulates), which works fine for full-batch training

---

### Day 4 (Week 7 Day 1, Dec 8) - 2.5 hours

**Goal**: Train on MNIST, achieve >95% accuracy, generate visualizations

**Tasks**:
1. **MNIST Data Loading** (30 min)
   - Use sklearn.datasets.load_digits() (8x8 images, 10 classes) OR
   - Download MNIST via keras/torchvision (28x28 images)
   - Normalize: `X = X / 255.0`
   - Flatten: `X = X.reshape(N, -1)`
   - One-hot encode labels: `y_onehot = np.eye(10)[y]`
   - Split: 80% train, 10% val, 10% test

2. **Network Architecture** (20 min)
   - Build 2-3 layer MLP: `784 → 128 → 64 → 10` (for 28x28 MNIST)
   - Use ReLU activations between layers
   - Xavier initialization for ReLU layers

3. **Full Training Run** (60 min - most is waiting)
   - Train for 20-50 epochs (early stop if validation plateaus)
   - Batch size: 128
   - Learning rate: 0.001 (Adam) or 0.01 (SGD)
   - Track train loss, train accuracy, val accuracy per epoch
   - Save best model checkpoint (lowest val loss)
   - **Target**: >95% test accuracy (should achieve 97-98% easily)

4. **Basic Visualizations** (40 min)
   - Loss curve: Train loss vs. epoch
   - Accuracy curve: Train acc vs. val acc vs. epoch
   - Confusion matrix on test set
   - Misclassified examples: Show 10 examples where model failed
   - (Optional) First layer weight visualization (10x784 → 10x28x28 images)

**Deliverable**: Trained MNIST model with evaluation results
- Test accuracy >95% (ideally 97-98%)
- Loss/accuracy curves saved as PNG
- Confusion matrix
- Misclassification analysis

**Files Created**:
```
train_mnist.py        # MNIST training script
evaluate.py           # Evaluation + visualization script

outputs/
├── checkpoints/
│   └── best_model.npz   # Saved weights
└── figures/
    ├── loss_curve.png
    ├── accuracy_curve.png
    ├── confusion_matrix.png
    └── misclassified.png
```

---

### Day 5 (Week 7 Day 2, Dec 9) - 2 hours

**Goal**: Documentation, README, and interview preparation

**Tasks**:
1. **Code Documentation** (30 min)
   - Add docstrings to all classes and key methods
   - Type hints: `def forward(self, x: np.ndarray) -> np.ndarray:`
   - Inline comments for tricky math (e.g., softmax numerical stability)

2. **README.md** (60 min) ⭐ **CRITICAL FOR INTERVIEWS**

   **Structure**:
   ```markdown
   # Neural Network from Scratch

   > A modular deep learning framework built with NumPy to master backpropagation fundamentals

   ## Features
   - Modular PyTorch-like design (Module base class, composable layers)
   - Proper weight initialization (Xavier, He)
   - Verified with numerical gradient checking (relative error < 1e-7)
   - SGD and Adam optimizers with bias correction
   - Trained on MNIST: 97.5% test accuracy

   ## Architecture

   ### Forward Pass
   [Show equations for linear layer, ReLU, softmax, cross-entropy]

   ### Backward Pass
   [Show gradient derivations: ∂L/∂W, ∂L/∂b, ∂L/∂x]

   ### Numerical Gradient Verification
   [Show comparison table: Analytical vs. Numerical, Relative Error]

   ## Training Results

   | Model | Test Accuracy | Epochs | Optimizer |
   |-------|---------------|--------|-----------|
   | 784→128→64→10 | 97.5% | 30 | Adam |
   | 784→128→64→10 | 96.2% | 40 | SGD |

   [Include loss curve image]

   ## Key Insights

   1. **Initialization matters**: He init converged faster than Xavier for ReLU networks
   2. **Adam vs. SGD**: Adam reached 95% accuracy in 15 epochs vs. 30 for SGD
   3. **Gradient checking**: Critical for catching implementation bugs - caught dimension mismatch in linear layer backward pass

   ## Usage

   ```python
   from nn import Linear, ReLU, CrossEntropyLoss
   from nn.optimizers import Adam

   # Build model
   model = Sequential([
       Linear(784, 128, init='he'),
       ReLU(),
       Linear(128, 64, init='he'),
       ReLU(),
       Linear(64, 10, init='xavier')
   ])

   # Train
   optimizer = Adam(model.parameters(), lr=0.001)
   train(model, X_train, y_train, optimizer, epochs=30)
   ```

   ## Interview Talking Points

   > "While my RAG project relies on modern libraries, I built this NN framework from scratch to ensure I mastered the first principles. I manually derived the gradients for Cross-Entropy and Softmax, implemented He Initialization to stabilize training, and verified my calculus with numerical gradient checking. It taught me exactly why we need optimizers like Adam over vanilla SGD - the adaptive learning rates and bias correction make a huge difference in early training stages."

   ## Future Improvements

   - Batch normalization (stabilize training for deeper networks)
   - Dropout (regularization)
   - Convolutional layers (for image data)
   - GPU acceleration via CuPy (maintain same API)

   ## License
   MIT
   ```

3. **Quick Reference Sheet** (20 min)
   - Create `references/Week6-NN-Reference.md`
   - Backprop equations, optimizer formulas, key insights
   - Interview Q&A prep (common gradient questions)

4. **Final Polish** (10 min)
   - Add `requirements.txt`
   - Update `.gitignore` (don't commit models, datasets)
   - Quick spell-check on README

**Deliverable**: Complete, interview-ready project
- Professional README with math, results, talking points
- All code documented
- Ready to push to GitHub as portfolio piece

**Files Created**:
```
README.md                              # Comprehensive documentation
requirements.txt                       # numpy==1.24.0, matplotlib, scikit-learn
.gitignore                             # *.npz, *.pkl, data/, __pycache__/
references/Week6-NN-Reference.md       # Quick reference for interviews
```

---

## Final Project Structure

```
projects/neural-network/
├── README.md                    # ⭐ Comprehensive, interview-ready
├── requirements.txt
├── .gitignore
├── nn/
│   ├── __init__.py
│   ├── module.py                # Base Module class
│   ├── layers.py                # Linear layer with verified backprop
│   ├── activations.py           # ReLU, Sigmoid, Softmax (forward + backward)
│   ├── losses.py                # CrossEntropyLoss, MSE
│   ├── initializers.py          # Xavier, He
│   ├── optimizers.py            # SGD, Adam with bias correction
│   └── utils.py                 # numerical_gradient, helper functions
├── train.py                     # Generic training loop
├── train_mnist.py               # MNIST-specific script
├── evaluate.py                  # Evaluation + visualization
├── tests/
│   ├── test_forward.py
│   ├── test_backward.py
│   └── test_gradients.py        # ⭐ Numerical gradient checking
├── outputs/
│   ├── checkpoints/
│   │   └── best_model.npz
│   └── figures/
│       ├── loss_curve.png
│       ├── accuracy_curve.png
│       └── confusion_matrix.png
└── notebooks/
    └── experiments.ipynb        # Optional: quick experiments
```

---

## Portfolio Value Assessment

**What This Project Demonstrates**:

1. **Deep Understanding**: Not just using PyTorch, but understanding what it does under the hood
2. **Mathematical Rigor**: Numerical gradient checking proves you derived gradients correctly
3. **Software Engineering**: Modular design, OOP principles applied to ML
4. **Optimization Knowledge**: Implementing Adam shows understanding beyond basic SGD
5. **Completeness**: From initialization to training to evaluation

**Interview Talking Points**:

| Question | Your Answer (Backed by Implementation) |
|----------|----------------------------------------|
| "Explain backpropagation" | "I implemented it from scratch. Here's my Linear layer backward pass: ∂L/∂W = ∂L/∂y @ x.T. I verified it with numerical gradient checking - matched to 1e-7 precision." |
| "Why Xavier initialization?" | "Prevents vanishing/exploding gradients. I compared Xavier vs. random init in my project - Xavier converged 2× faster. Used He init for ReLU since variance differs." |
| "What's the difference between SGD and Adam?" | "I implemented both. Adam combines momentum (first moment) and RMSProp (second moment) with bias correction. In my MNIST experiments, Adam reached 95% in 15 epochs vs. 30 for SGD." |
| "How do you debug ML code?" | "Numerical gradient checking is critical. I caught a dimension mismatch bug in my linear layer backward - analytical grad had wrong shape, but numerical grad check revealed it immediately." |

**Combined with RAG Project**:
- RAG demonstrates: Systems engineering, evaluation frameworks, production deployment
- NN demonstrates: ML fundamentals, mathematical rigor, algorithm implementation
- Together: "I understand both how to build production ML systems AND the underlying theory"

---

## Time Budget Summary

| Day | Date | Task | Hours | Cumulative |
|-----|------|------|-------|------------|
| Day 1 | Dec 2 | Forward pass + initialization | 2.5h | 2.5h |
| Day 2 | Dec 3 | Backprop + gradient checking | 3h | 5.5h |
| Day 3 | Dec 4 | Training loop + optimizers | 2h | 7.5h |
| Day 4 | Dec 8 | MNIST training + eval | 2.5h | 10h |
| Day 5 | Dec 9 | Documentation + README | 2h | 12h |

**Total**: 12 hours (within Week 6-7 schedule)

---

## Success Criteria

**Technical** (Core - Learning Objectives):
- ✅ Modular PyTorch-like design
- ✅ Xavier/He initialization implemented
- ✅ Numerical gradient checking passes (rel. error < 1e-7)
- ✅ SGD optimizer implemented
- ✅ Adam optimizer with bias correction
- ✅ Training loop working (XOR: 100% accuracy, loss → 0.000032)
- ✅ SGD vs Adam comparison (Adam 35× better convergence)

**Technical** (Optional - Portfolio Polish):
- ⬜ MNIST training >95% test accuracy (deferred)
- ⬜ Loss/accuracy curves visualization (deferred)
- ⬜ Professional README with math formulas (deferred)

**Portfolio** (Core - Achieved):
- ✅ Interview talking points documented (`key-learnings.md`)
- ✅ Code is readable and modular
- ✅ Demonstrates theory + engineering skills (backprop, optimizers, initialization)

**Interview Readiness** (Core - Achieved):
- ✅ Can explain backprop on whiteboard
- ✅ Can discuss initialization strategies (He vs Xavier)
- ✅ Can compare SGD vs. Adam with data (35× convergence improvement)
- ✅ Can explain gradient checking importance (verified to 1e-7)

**Status**: ✅ **CORE COMPLETE** (90% done, learning closure achieved)
- Feb 4-5: Backprop + gradient checking
- Feb 7: SGD + training loop
- Feb 9: Adam optimizer
- Optional: MNIST + documentation (not needed for interview prep)

---

## Risk Mitigation

**Potential Issues**:
1. **Gradient checking fails**: Most common = dimension mismatch. Debug with small toy examples (2x2 matrix).
2. **Training doesn't converge**: Check learning rate (too high), initialization (wrong type), loss computation (numerical stability).
3. **Low MNIST accuracy**: Ensure normalization (X/255), correct one-hot encoding, sufficient epochs (20-30 minimum).
4. **Time overruns**: Skip advanced visualizations if needed - focus on core rigor (gradient checking) and documentation.

**Backup Plan**:
If pressed for time on Day 5, README priority order:
1. Math formulas (forward/backward equations) ← Must have
2. Numerical gradient verification table ← Must have
3. Interview talking points ← Must have
4. Training results and curves ← Nice to have
5. Usage examples ← Nice to have

---

## Optional Enhancements

These are nice-to-have features that improve code quality but aren't critical for learning closure:

### 1. Advanced Training Features (30-45 min) - Production-ready training loop

**Current**: Basic full-batch training on XOR (works fine for toy datasets)

**Enhancements**:
- **Mini-batch processing**: Split data into batches, iterate through them
- **Data shuffling**: Shuffle dataset each epoch (reduces overfitting)
- **Early stopping**: Monitor validation loss, stop if no improvement
- **Learning rate scheduling**: Decay LR over time

**Benefits**:
- ✅ Scales to large datasets (MNIST, ImageNet)
- ✅ Better generalization (shuffling prevents order bias)
- ✅ Prevents overtraining (early stopping)

**When to implement**: If training on MNIST or larger datasets

---

### 2. Sequential Container (30 min) - Makes forward/backward cleaner

**Problem**: Nested backward calls are awkward:
```python
grad = linear1.backward(relu.backward(linear2.backward(softmax.backward(loss.backward()))))
```

**Solution**: PyTorch-style Sequential container
```python
model = Sequential([Linear(3, 2), ReLU(), Linear(2, 3), Softmax()])
y = model(x)
grad = model.backward(loss.backward())
```

**Implementation**:
- Create `nn/sequential.py` with `Sequential` class
- Forward: iterate through layers
- Backward: iterate through reversed(layers)
- 20-30 lines of code

**Benefits**:
- ✅ Cleaner API (matches PyTorch)
- ✅ Easier to add/remove layers
- ✅ Better for debugging (can iterate and inspect)

**When to implement**: After Day 3 (training loop works) or Day 5 (polish phase)

---

### 3. Other Potential Enhancements (Low priority)

- **Layer naming**: Access layers as `model.layer1.W` instead of `model.layers[0].parameters["W"]`
- **Computation graph**: Automatic differentiation (like PyTorch autograd) - requires ~200+ lines
- **Checkpoint saving/loading**: Serialize model to disk
- **Learning rate scheduling**: Decay LR over training
- **Batch normalization**: Stabilize training for deeper networks
