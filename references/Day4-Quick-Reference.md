# ML Quick Reference - Day 4 Theory & Implementation

**Date**: 2025-10-31
**Topics**: Regularization, Regression Metrics, K-Means Clustering

---

## 1. Regularization

### **Problem**: Low bias on training data but high variance (overfitting)
**Solution**: Trade slight increase in bias for lower variance and better generalization

---

### **L2 Regularization (Ridge)**

**Formula**:
```python
Loss = MSE + λ * Σ(w_j²)
```

**Effect**:
- Shrinks weights **asymptotically toward 0** (never exactly 0)
- Also called "weight decay"
- Makes predictions less sensitive to training data

**When to use**:
- Most features are useful
- Want to keep all features but prevent large weights
- Default choice when unsure

**Hyperparameter**:
- λ (lambda): Regularization strength, tune via cross-validation
- Larger λ → stronger regularization → smaller weights

---

### **L1 Regularization (Lasso)**

**Formula**:
```python
Loss = MSE + λ * Σ|w_j|
```

**Effect**:
- Shrinks weights to **exact 0** (sparse solution)
- Performs automatic **feature selection**
- Makes predictions less sensitive to training data

**When to use**:
- Many useless/redundant features
- Want interpretable model (fewer features)
- Need feature selection

**Key difference from L2**:
- L2: All weights small but non-zero
- L1: Some weights exactly 0 (feature removed)

---

### **Elastic Net**

**Formula**:
```python
Loss = MSE + λ₁ * Σ|w_j| + λ₂ * Σ(w_j²)
```

**Effect**:
- Combines L1 + L2
- Gets L1's feature selection + L2's grouping effect

**When to use**:
- Too many variables, don't know which are useful beforehand
- **Correlated features**: Groups correlated features together
  - L1 (Lasso) picks one arbitrarily → unstable
  - L2 (Ridge) keeps all with similar weights → no selection
  - Elastic Net: Keeps/drops correlated features as a group → stable selection

**Why better for correlated variables**:
- Example: `height_cm` and `height_inches` (highly correlated)
  - Lasso: Keeps `height_cm`, drops `height_inches` (arbitrary)
  - Ridge: Keeps both with w₁=0.3, w₂=0.3
  - Elastic Net: Keeps both with w₁=0.25, w₂=0.25 (stable grouping)

---

### **Common to All**

**Similarities**:
- Make predictions less sensitive to input
- Find λ via cross-validation to minimize variance
- Work for both continuous and discrete variables
- Help when data size is small (solvable even if #points < #parameters)

**How to choose λ**:
- Cross-validation: Try multiple λ values, pick one with best validation performance
- Grid search or random search common approaches

---

## 2. Regression Metrics

### **MSE (Mean Squared Error)**

**Formula**:
```python
MSE = (1/n) * Σ(y_true - y_pred)²
```

**Properties**:
- Smooth, easy to optimize (differentiable everywhere)
- Penalizes large errors more (quadratic)
- **Sensitive to outliers** (squared term amplifies large errors)

**When to use**:
- Large errors are particularly bad
- During model training (gradient descent friendly)

---

### **RMSE (Root Mean Squared Error)**

**Formula**:
```python
RMSE = √MSE = √[(1/n) * Σ(y_true - y_pred)²]
```

**Properties**:
- Similar to MSE but **same units as target variable**
- More interpretable than MSE
- Still sensitive to outliers

**When to use**:
- Want interpretable error in original units
- Large errors should be penalized more
- Common in competitions (Kaggle)

**Relationship**: RMSE ≥ MAE (equality only when all errors equal)

---

### **MAE (Mean Absolute Error)**

**Formula**:
```python
MAE = (1/n) * Σ|y_true - y_pred|
```

**Properties**:
- Penalizes all errors equally (linear)
- **Robust to outliers** (no squared term)
- Not differentiable at 0 (use subgradients in practice)

**When to use**:
- All errors equally important
- Data has outliers
- Want robust metric

---

### **R² (Coefficient of Determination)**

**Formula**:
```python
R² = 1 - (SS_residual / SS_total)

where:
  SS_residual = Σ(y_true - y_pred)²    # Model error
  SS_total = Σ(y_true - y_mean)²       # Baseline error
```

**Range**: (-∞, 1], typically [0, 1]

**Interpretation**:
- **R² = 1**: Perfect predictions (explains 100% of variance)
- **R² = 0.8**: Model explains 80% of variance
- **R² = 0**: Model no better than predicting mean
- **R² < 0**: Model **worse** than predicting mean (very bad!)
  - Only possible on test/validation set, not training set

**Properties**:
- **Scale-independent**: Can compare models across different units
- Doesn't tell you absolute error magnitude
- Measures linear fit quality

**When to use**:
- Want to know % of variance explained
- Comparing models on same dataset
- Assessing model fit quality

---

### **R² Limitations**

**1. Always increases with more features** (MAIN LIMITATION):
- Even useless features increase R² slightly
- Encourages overfitting
- **Solution**: Use **Adjusted R²**

**Adjusted R²**:
```python
Adjusted R² = 1 - [(1 - R²) * (n - 1) / (n - p - 1)]

where:
  n = number of data points
  p = number of features
```

- Penalizes adding features
- Only increases if new feature improves fit enough to justify complexity
- Can decrease when adding useless features
- Use for model comparison with different #features

**2. Scale-independent (pro and con)**:
- Pro: Comparable across different units
- Con: Doesn't tell if errors are practically acceptable
- RMSE = $100 tells you error magnitude, R² = 0.8 doesn't

**3. Can be negative on test sets**:
- Happens when model is worse than mean baseline
- Clear sign of severe overfitting

---

### **Metric Selection Guide**

| Scenario | Recommended Metric |
|----------|-------------------|
| All errors equally important | **MAE** |
| Large errors particularly bad | **RMSE** |
| Data has outliers | **MAE** (robust) |
| Want % variance explained | **R²** |
| Comparing models with different #features | **Adjusted R²** |
| Need interpretable error in original units | **RMSE** or **MAE** |
| Model training/optimization | **MSE** (smooth gradient) |

---

## 3. K-Means Clustering

### **Algorithm**

**Goal**: Partition n data points into K clusters by minimizing within-cluster variance

**Steps**:
```python
1. Initialize: Randomly select K centroids from data points
2. Repeat until convergence:
   a. Assignment step: Assign each point to nearest centroid
   b. Update step: Recompute centroids as mean of assigned points
   c. Check convergence: If centroids don't change (or change < ε), stop
3. If max iterations reached, stop
```

---

### **Implementation Details**

**Distance calculation** (Euclidean):
```python
distance = np.sqrt(np.sum((point - centroid) ** 2))
# Or squared distance (faster, same result):
distance² = np.sum((point - centroid) ** 2)
```

**Assignment step**:
```python
# For each point, find index of closest centroid
labels = np.argmin(distances, axis=1)  # distances: (n, k) → labels: (n,)
```

**Update step**:
```python
# For each cluster, compute mean of assigned points
for k in range(K):
    centroids[k] = X[labels == k].mean(axis=0)
```

**Convergence check**:
```python
# Check if centroids moved less than tolerance
if np.allclose(old_centroids, new_centroids, atol=ε):
    break
```

**Vectorization** (broadcasting):
```python
# Compute all pairwise distances at once
# X: (n, m), centroids: (k, m)
distances = np.sum(
    (X[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2,
    axis=2
)  # Result: (n, k)
```

---

### **Edge Cases**

**Empty cluster**:
```python
cluster_points = X[labels == k]
if len(cluster_points) == 0:
    # Reinitialize this centroid randomly
    centroids[k] = X[np.random.choice(n)]
```

**n < k**:
```python
if len(X) < k:
    raise ValueError("Number of samples must be >= k")
```

---

### **Time Complexity**

**Per iteration**: O(n × k × d)
- n = number of points
- k = number of clusters
- d = number of dimensions

**Total**: O(n × k × d × i)
- i = number of iterations until convergence
- Typically i is small (10-100)

**Space**: O(n × k) for distance matrix (or O(n × d) if computing distances on-the-fly)

---

### **Limitations & Solutions**

#### **1. Sensitive to Initialization**

**Problem**: K-Means finds local minimum, not global
- Bad initialization → poor clustering
- Results unstable across runs

**Solutions**:
- **K-Means++**: Smart initialization
  - Pick first centroid randomly
  - Pick subsequent centroids with probability ∝ distance² from existing
  - Spreads out initial centroids
  - Much better results

- **Multiple runs**: Run algorithm multiple times with different random seeds
  - Keep result with lowest inertia (within-cluster variance)
  - sklearn default: `n_init=10`

- **Fixed random seed**: For reproducibility
  ```python
  np.random.seed(42)
  ```

#### **2. Must Choose K in Advance**

**Solutions**:
- **Elbow method**: Plot inertia vs k, look for "elbow"
- **Silhouette score**: Measures cluster quality
- **Domain knowledge**: Often k is known from problem context

#### **3. Assumes Spherical Clusters**

**Problem**: Doesn't work well for:
- Non-convex shapes
- Elongated clusters
- Different cluster sizes/densities

**Solutions**:
- Use different algorithms: DBSCAN, Gaussian Mixture Models
- Transform features first

#### **4. Sensitive to Outliers**

**Problem**: Outliers affect centroid calculation (mean is sensitive)

**Solutions**:
- K-Medoids: Use medoids (actual data points) instead of centroids
- Remove outliers before clustering

---

### **Evaluation Metrics**

**Inertia (Within-cluster sum of squares)**:
```python
inertia = Σ(for all clusters) Σ(points in cluster) ||point - centroid||²
```
- Lower is better
- Used to compare same algorithm with different k or initialization

**Silhouette Score**:
- Range: [-1, 1]
- Measures how similar point is to its cluster vs other clusters
- Higher is better

---

## 4. Interview Quick Answers

### **"Explain regularization and when to use L1 vs L2"**

> "Regularization adds a penalty term to the loss function to prevent overfitting by constraining model complexity.
>
> L2 (Ridge) adds λΣw² to the loss, which shrinks weights toward zero but never exactly zero. Use L2 when most features are useful and you want to keep all of them.
>
> L1 (Lasso) adds λΣ|w| to the loss, which can shrink weights to exactly zero, performing automatic feature selection. Use L1 when you have many irrelevant features.
>
> Elastic Net combines both, handling correlated features better by grouping them together rather than arbitrarily picking one like Lasso does."

---

### **"When would you use MAE vs RMSE?"**

> "Both measure prediction error, but RMSE squares errors before averaging, so it penalizes large errors more heavily. Use RMSE when large errors are particularly costly, like in medical predictions or pricing.
>
> MAE treats all errors equally and is more robust to outliers since it doesn't square the errors. Use MAE when outliers are present or when all errors should be weighted equally.
>
> Both are in the same units as the target variable, making them interpretable. RMSE is always ≥ MAE."

---

### **"What are R²'s limitations?"**

> "The main limitation is that R² always increases when adding features, even useless ones, because it measures training fit without penalizing complexity. This encourages overfitting. We use adjusted R² for model comparison, which penalizes additional features.
>
> R² is also scale-independent - while this lets you compare models across different units, it doesn't tell you if errors are practically acceptable. An R² of 0.9 could still have unacceptable prediction errors depending on the domain.
>
> Finally, R² can be negative on test data, meaning the model is worse than just predicting the mean - a clear sign of severe overfitting."

---

### **"Explain the K-Means algorithm"**

> "K-Means partitions data into K clusters by minimizing within-cluster variance. The algorithm works as follows:
>
> 1. Randomly initialize K centroids from the data
> 2. Assignment step: Assign each point to its nearest centroid using Euclidean distance
> 3. Update step: Recompute each centroid as the mean of its assigned points
> 4. Repeat steps 2-3 until centroids converge or max iterations reached
>
> Time complexity is O(nkdi) where n is points, k is clusters, d is dimensions, and i is iterations.
>
> Main limitation: K-Means is sensitive to initialization and finds local minima. Solutions include K-Means++ initialization and running multiple times with different seeds, keeping the result with lowest inertia."

---

### **"How would you choose the number of clusters K?"**

> "Several approaches:
>
> 1. **Elbow method**: Plot inertia (within-cluster sum of squares) vs K. Look for an 'elbow' where adding more clusters gives diminishing returns.
>
> 2. **Silhouette score**: Measures how well points fit their clusters. Higher scores indicate better clustering. Can compare across different K values.
>
> 3. **Domain knowledge**: Often K is known from the problem context - like customer segments or document categories.
>
> 4. **Cross-validation**: If clustering is for a downstream task, evaluate the downstream task performance for different K values.
>
> In practice, I'd combine these approaches - use domain knowledge to narrow the range, then use elbow method and silhouette score to refine."

---

---

## 6. Key Python Gotchas Learned

### **Array Assignment**:
```python
# Wrong: Both point to same array!
centroids = new_centroids

# Right: Independent copy
centroids = new_centroids.copy()
```

Similar to Java for objects/arrays:
```java
int[] arr1 = {1, 2, 3};
int[] arr2 = arr1;  // arr2 points to same array!
int[] arr3 = arr1.clone();  // arr3 is a copy
```

---

**Last Updated**: 2025-10-31 (Day 4)
**Status**: Theory refresh (regularization, regression metrics) + K-Means implementation complete
