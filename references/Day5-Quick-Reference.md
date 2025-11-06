# ML Quick Reference - Day 5 Theory Refresh

**Date**: 2025-11-01 (Weekend, limited time)
**Topics**: Boosting (AdaBoost, Gradient Boost), Parametric vs Non-parametric Models

---

## 1. Boosting Overview

### **What is Boosting?**

**Ensemble method** that combines **weak learners sequentially** to create a strong learner.

**Key characteristics**:
- Sequential (not parallel like Random Forest)
- Each learner focuses on errors of previous learners
- Reduces **bias** (vs Random Forest which reduces **variance**)

**Bagging vs Boosting**:
| | Bagging (Random Forest) | Boosting (AdaBoost, Gradient Boost) |
|---|---|---|
| **Training** | Parallel (independent) | Sequential (dependent) |
| **Focus** | Each tree sees random sample | Each learner focuses on errors |
| **Reduces** | Variance | Bias |
| **Weak/Strong** | Strong learners (deep trees) | Weak learners (stumps/shallow trees) |
| **Speed** | Fast (parallel) | Slower (sequential) |

---

## 2. AdaBoost (Adaptive Boosting)

### **Algorithm Components**

**Weak learners**: Stumps (tree with 1 node, 2 leaves)
- Each stump makes predictions
- Stumps have different "amount of say" in final decision
- Each stump influences how next stump is built

### **Algorithm Steps**

**1. Initialize sample weights**:
```python
# All samples start with equal weight
sample_weights = np.ones(n) / n
```

**2. Build first stump**:
- Try all features and thresholds
- Pick one with **lowest Gini index**
- Calculate total error: `total_error = sum of sample_weights for misclassified samples`

**3. Calculate "amount of say"**:
```python
amount_of_say = 0.5 * log((1 - total_error) / total_error)
```
- Lower error → higher amount of say
- Higher error → lower amount of say
- Range: (-∞, +∞), but typically (-3, 3)

**4. Adjust sample weights**:
```python
# For incorrectly classified samples:
new_weight = old_weight * exp(amount_of_say)  # Increase

# For correctly classified samples:
new_weight = old_weight * exp(-amount_of_say)  # Decrease

# Normalize so weights sum to 1
sample_weights = sample_weights / sum(sample_weights)
```

**5. Build next stump** (two options):

**Option 1**: Use weighted Gini index
- Calculate Gini with sample weights

**Option 2**: Resample dataset (more common)
- Sample original dataset with replacement using sample_weights as probabilities
- Create new dataset same size as original
- Build stump on sampled data
- **But**: Run stump on **original dataset** to calculate error and adjust weights

**6. Repeat** until:
- Reach predefined number of stumps (e.g., 100)
- Or reach acceptable error rate
- Or validation performance degrades

**7. Final prediction**:
```python
# For each sample, sum votes from all stumps weighted by amount_of_say
final_prediction = sign(sum(stump_prediction * amount_of_say for all stumps))
```

---

### **Remarks**

**Same feature can appear multiple times**:
- After reweighting, best feature might be same as before but different threshold
- Or different feature becomes more important
- Not limited to one stump per feature

---

## 3. Gradient Boost (For Regression)

### **Key Differences from AdaBoost**

| | AdaBoost | Gradient Boost |
|---|---|---|
| **Weak learners** | Stumps (1 split) | Small trees (8-32 leaves) |
| **Focus mechanism** | Reweight samples | Predict residuals |
| **What's predicted** | Original target y | Residual errors |
| **Weighting** | Amount of say (varies per stump) | Learning rate (same for all trees) |
| **Task** | Classification (typically) | Regression (can do classification too) |

---

### **Algorithm Steps**

**1. Initialize with single leaf**:
```python
# First prediction: average of all y values
initial_prediction = mean(y)
```

**2. For each tree (limited size, typically 8-32 leaves)**:

**a. Calculate pseudo-residuals**:
```python
residuals = y_true - current_predictions
# current_predictions = sum of all previous tree predictions
```

**b. Build small tree to predict residuals**:
- Limit tree depth (e.g., 2-5 levels)
- If multiple samples end up in same leaf, that leaf predicts **average residual**

**c. Add tree to ensemble**:
```python
# Scale by learning rate
predictions += learning_rate * new_tree_predictions
```

**3. Final prediction**:
```python
prediction = initial_leaf + learning_rate * sum(all tree predictions)
# Note: Initial leaf is NOT scaled by learning rate
```

---

### **Hyperparameters**

**Learning rate** (η, typically 0.01-0.3):
- Smaller → more trees needed, less overfitting
- Larger → fewer trees, faster training, risk overfitting
- Common: 0.1

**Number of trees**:
- More trees → better fit, risk overfitting
- Use early stopping on validation set

**Tree size** (max leaves):
- Typical: 8-32 leaves
- Smaller → more bias, less variance
- Larger → more variance, risk overfitting

---

### **Why "Gradient" Boost?**

**Not covered in StatQuest Part 1**, but the name comes from:
- Residuals are the **negative gradient** of loss function
- More general framework: can optimize any differentiable loss
- AdaBoost is a special case of Gradient Boost

---

## 4. Parametric vs Non-parametric Models

### **Core Difference**

**Parametric Models**:
- **Fixed number of parameters** (independent of training data size)
- Makes **assumptions** about data distribution/relationship
- Parameters learned during training, **training data discarded** after
- **Fast prediction**, **small model size**

**Non-parametric Models**:
- Model **complexity grows** with training data size
- **Fewer assumptions** about data
- Often **stores training data** or structure grows with data
- More **flexible**, but **slower/larger**

---

### **Examples**

| Model | Type | "Parameters" | Size Depends on Data? |
|-------|------|--------------|----------------------|
| **Linear Regression** | Parametric | w, b | No - fixed size |
| **Logistic Regression** | Parametric | w, b | No - fixed size |
| **Neural Network** | Parametric | All weights/biases | No - fixed architecture |
| **Naive Bayes** | Parametric | Class probabilities | No - fixed per class |
| **K-NN** | Non-parametric | Entire X_train, y_train | Yes - stores all data |
| **Decision Tree** | Non-parametric | Tree structure | Yes - tree grows with data |
| **Random Forest** | Non-parametric | Ensemble of trees | Yes - trees grow |
| **K-Means** | Borderline | k centroids | Debatable - fixed k, but stores centroids |
| **AdaBoost** | Non-parametric | Collection of stumps | Yes - ensemble grows |
| **SVM (with kernel)** | Non-parametric | Support vectors | Yes - stores support vectors |

---

### **Key Tradeoffs**

**Parametric** ✅:
- ✅ **Fast predictions**: Just matrix multiply `X @ w + b`
- ✅ **Small model size**: Store only parameters (KB-MB)
- ✅ **Works well** when assumptions hold
- ✅ **Interpretable**: Can inspect weights
- ❌ **Less flexible**: Strong assumptions (linearity, etc.)
- ❌ **May underfit**: Can't capture complex patterns

**Non-parametric** ✅:
- ✅ **Very flexible**: Adapts to data shape
- ✅ **Fewer assumptions**: No need to know relationship form
- ✅ **Can model complex patterns**: Non-linear, multi-modal, etc.
- ❌ **Slower predictions**: K-NN is O(n), trees require traversal
- ❌ **Larger model size**: Stores data or large structures (MB-GB)
- ❌ **Risk of overfitting**: Especially with small data

---

### **When to Use Which**

**Use Parametric when**:
- You understand the problem structure (e.g., linear relationship)
- Need fast predictions (production with latency constraints)
- Small model size required (embedded systems, mobile)
- Want interpretability (coefficients have meaning)
- Example: Logistic regression for simple binary classification

**Use Non-parametric when**:
- Relationship is complex or unknown
- Have enough data to avoid overfitting
- Accuracy more important than speed/size
- Data has complex patterns (clusters, non-linear boundaries)
- Example: Random Forest for tabular data with interactions

---

## 5. Interview Quick Answers

### **"What is boosting and how does it differ from bagging?"**

> "Boosting is an ensemble method that builds weak learners **sequentially**, where each learner focuses on correcting errors made by previous learners. It reduces **bias**.
>
> Bagging (like Random Forest) builds strong learners **in parallel** independently, each on a random sample of data. It reduces **variance**.
>
> Key difference: Boosting is sequential and adaptive (each model depends on previous), while bagging is parallel and independent. Boosting typically uses weak learners (stumps, shallow trees), while bagging uses strong learners (deep trees)."

---

### **"Explain how AdaBoost works"**

> "AdaBoost builds an ensemble of stumps (trees with one split) sequentially:
>
> 1. All samples start with equal weights
> 2. Build a stump using weighted data (pick feature/threshold with lowest Gini)
> 3. Calculate 'amount of say' based on stump's error: `0.5 * log((1-error)/error)`
> 4. Adjust sample weights: increase for misclassified samples (`weight * e^amount_of_say`), decrease for correct ones (`weight * e^-amount_of_say`)
> 5. Normalize weights and repeat
>
> Final prediction is weighted vote of all stumps using their 'amount of say'. This forces subsequent stumps to focus on previously misclassified samples."

---

### **"What's the difference between AdaBoost and Gradient Boost?"**

> "Both are sequential boosting methods, but they focus on mistakes differently:
>
> **AdaBoost** reweights samples - increases weight for misclassified samples so next learner focuses on them. Uses stumps (1 split) and each has an 'amount of say' based on its error.
>
> **Gradient Boost** builds trees to predict residuals - each new tree predicts the difference between true values and current predictions. Uses small trees (8-32 leaves) and scales all by the same learning rate.
>
> In practice, Gradient Boost is more flexible (can optimize any differentiable loss) and generally more powerful."

---

### **"What's the difference between parametric and non-parametric models?"**

> "Parametric models have a **fixed number of parameters** that doesn't change with training data size. For example, linear regression always has weights and bias, whether trained on 100 or 1 million samples. After training, you keep just the parameters and discard the data. They're fast and small but make strong assumptions.
>
> Non-parametric models' **complexity grows with data**. K-NN stores the entire training dataset, decision trees grow deeper with more data. They're more flexible and make fewer assumptions, but predictions can be slower and they require more memory.
>
> I'd choose parametric when the problem structure is understood and I need fast predictions - like logistic regression for binary classification. I'd choose non-parametric when relationships are complex or unknown - like Random Forest for tabular data with feature interactions."

---

### **"Is deep learning parametric or non-parametric?"**

> "Neural networks are **parametric**. Even though 'deep learning' sounds complex, the model has a fixed architecture - say 784 input neurons → 128 hidden → 10 output. Once trained, you just store the weights and biases. The model size doesn't change whether you train on 1,000 or 1 million images. The number of parameters is determined by the architecture, not the training data size.
>
> This is why you can download a pre-trained ResNet model as a fixed-size file - it's just parameters."

---

---

## 7. Additional Notes

### **Boosting in Practice**

**Popular libraries**:
```python
# AdaBoost
from sklearn.ensemble import AdaBoostClassifier

# Gradient Boosting
from sklearn.ensemble import GradientBoostingRegressor
```

**When to use boosting**:
- Tabular data (not images/text - use neural nets)
- Need high accuracy, can afford sequential training
- Have enough regularization to prevent overfitting

**When NOT to use boosting**:
- Need very fast training (use Random Forest)
- Very high-dimensional sparse data (use linear models)
- Images/text (use CNNs/Transformers)

---

**Last Updated**: 2025-11-01 (Day 5)
**Status**: Theory-only day (weekend, limited time ~45 min)
**Topics Covered**: Boosting mastery, parametric/non-parametric understanding
**Next**: Continue with algorithm implementations or start projects
