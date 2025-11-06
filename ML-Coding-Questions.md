# ML Coding Interview Questions

This document contains coding questions commonly asked in ML Engineer interviews. These test your ability to implement ML algorithms, manipulate data, and write production-quality code.

---

## Question Categories

1. **Implement ML Algorithms from Scratch** (most common)
2. **Data Preprocessing & Feature Engineering**
3. **Model Training & Evaluation**
4. **Neural Network Implementation**
5. **ML-specific Data Structures**

---

## Category 1: Implement ML Algorithms from Scratch

These questions test your understanding of ML fundamentals. You should be able to implement without using sklearn or high-level libraries.

### Question 1.1: Linear Regression (Easy)

**Problem**: Implement linear regression from scratch using gradient descent.

**Requirements**:
- Fit function: Takes X (features) and y (labels)
- Predict function: Returns predictions for new data
- Use gradient descent for optimization
- Calculate and return MSE (mean squared error)

**Starter Code**:
```python
import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Train the model using gradient descent.
        X: numpy array of shape (n_samples, n_features)
        y: numpy array of shape (n_samples,)
        """
        # TODO: Implement
        pass

    def predict(self, X):
        """
        Make predictions on new data.
        X: numpy array of shape (n_samples, n_features)
        Returns: predictions of shape (n_samples,)
        """
        # TODO: Implement
        pass

    def mse(self, y_true, y_pred):
        """Calculate mean squared error."""
        # TODO: Implement
        pass

# Test
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3  # y = 1*x1 + 2*x2 + 3

model = LinearRegression()
model.fit(X, y)
predictions = model.predict(X)
print(f"MSE: {model.mse(y, predictions)}")
```

**Key Points to Discuss**:
- Gradient descent formula: w = w - lr * dL/dw
- Vectorization for efficiency
- Feature normalization (if asked)
- Learning rate selection

---

### Question 1.2: Logistic Regression (Easy-Medium)

**Problem**: Implement binary logistic regression from scratch.

**Requirements**:
- Use sigmoid function for predictions
- Binary cross-entropy loss
- Gradient descent optimization
- Return predicted probabilities and classes

**Starter Code**:
```python
import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def sigmoid(self, z):
        """Sigmoid activation function."""
        # TODO: Implement
        pass

    def fit(self, X, y):
        """
        Train the model.
        y: binary labels (0 or 1)
        """
        # TODO: Implement
        # Use binary cross-entropy loss
        # Update weights using gradient descent
        pass

    def predict_proba(self, X):
        """Return probability predictions."""
        # TODO: Implement
        pass

    def predict(self, X, threshold=0.5):
        """Return class predictions (0 or 1)."""
        # TODO: Implement
        pass

# Test on a simple dataset
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, random_state=42)

model = LogisticRegression()
model.fit(X, y)
accuracy = np.mean(model.predict(X) == y)
print(f"Accuracy: {accuracy}")
```

**Key Concepts**:
- Sigmoid function: σ(z) = 1 / (1 + e^(-z))
- Binary cross-entropy loss: -[y*log(ŷ) + (1-y)*log(1-ŷ)]
- Gradient: dL/dw = (1/n) * X^T * (ŷ - y)
- Numerical stability (avoid log(0))

---

### Question 1.3: K-Means Clustering (Medium)

**Problem**: Implement K-means clustering from scratch.

**Requirements**:
- Initialize centroids (random or k-means++)
- Assign points to nearest centroid
- Update centroids
- Iterate until convergence
- Return cluster assignments and centroids

**Starter Code**:
```python
import numpy as np

class KMeans:
    def __init__(self, n_clusters=3, max_iterations=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iterations = max_iterations
        self.random_state = random_state
        self.centroids = None

    def fit(self, X):
        """
        Fit K-means clustering.
        X: numpy array of shape (n_samples, n_features)
        """
        # TODO: Implement
        # 1. Initialize centroids
        # 2. Repeat until convergence:
        #    a. Assign each point to nearest centroid
        #    b. Update centroids as mean of assigned points
        pass

    def predict(self, X):
        """
        Predict cluster for each sample.
        Returns: array of cluster assignments
        """
        # TODO: Implement
        pass

    def _euclidean_distance(self, x1, x2):
        """Calculate Euclidean distance."""
        # TODO: Implement
        pass

# Test
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=100, centers=3, random_state=42)

kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
labels = kmeans.predict(X)
print(f"Cluster assignments: {labels}")
```

**Key Concepts**:
- Distance metrics (Euclidean)
- Convergence criteria (centroids don't change)
- K-means++ initialization (better than random)
- Handling empty clusters

---

### Question 1.4: Decision Tree (Hard)

**Problem**: Implement a binary decision tree classifier from scratch.

**Requirements**:
- Calculate information gain or Gini impurity
- Find best split (feature and threshold)
- Recursively build tree
- Support max_depth parameter
- Predict using the trained tree

**Starter Code**:
```python
import numpy as np

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # Feature index to split on
        self.threshold = threshold  # Threshold value for split
        self.left = left           # Left child node
        self.right = right         # Right child node
        self.value = value         # Class label (for leaf nodes)

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        """Build the decision tree."""
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X, y, depth):
        """Recursively build the tree."""
        # TODO: Implement
        # 1. Check stopping criteria (max_depth, min_samples, pure node)
        # 2. Find best split
        # 3. Create left and right subtrees
        # 4. Return node
        pass

    def _best_split(self, X, y):
        """Find the best feature and threshold to split on."""
        # TODO: Implement
        # Try all features and thresholds
        # Return feature_idx and threshold with best information gain
        pass

    def _information_gain(self, y, left_y, right_y):
        """Calculate information gain of a split."""
        # TODO: Implement using entropy
        pass

    def _entropy(self, y):
        """Calculate entropy."""
        # TODO: Implement
        pass

    def predict(self, X):
        """Predict class labels."""
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        """Traverse tree to make prediction for a single sample."""
        # TODO: Implement
        pass

# Test
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=100, n_features=4, random_state=42)

tree = DecisionTree(max_depth=3)
tree.fit(X, y)
predictions = tree.predict(X)
accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy}")
```

**Key Concepts**:
- Entropy: H(S) = -Σ p_i * log2(p_i)
- Information Gain: IG = H(parent) - [weighted average of H(children)]
- Gini impurity (alternative to entropy)
- Recursive tree building
- Stopping criteria

---

### Question 1.5: K-Nearest Neighbors (Easy-Medium)

**Problem**: Implement KNN classifier from scratch.

**Requirements**:
- Store training data
- For prediction, find k nearest neighbors
- Use majority vote for classification
- Support different distance metrics

**Starter Code**:
```python
import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        """Store training data."""
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        """Predict class labels for X."""
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        """Predict for a single sample."""
        # TODO: Implement
        # 1. Calculate distances to all training samples
        # 2. Find k nearest neighbors
        # 3. Return majority class
        pass

    def _euclidean_distance(self, x1, x2):
        """Calculate Euclidean distance."""
        return np.sqrt(np.sum((x1 - x2) ** 2))

# Test
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=100, n_features=4, random_state=42)

knn = KNN(k=5)
knn.fit(X, y)
predictions = knn.predict(X[:10])
print(f"Predictions: {predictions}")
```

**Key Concepts**:
- Distance metrics (Euclidean, Manhattan, etc.)
- Choosing k (odd number for binary classification)
- Efficiency concerns (KD-tree, Ball tree for large datasets)
- Weighted voting (closer neighbors have more weight)

---

## Category 2: Data Preprocessing & Feature Engineering

### Question 2.1: Implement StandardScaler (Easy)

**Problem**: Implement feature standardization (z-score normalization).

```python
class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, X):
        """Calculate mean and standard deviation."""
        # TODO: Implement
        pass

    def transform(self, X):
        """Standardize features: (X - mean) / std"""
        # TODO: Implement
        pass

    def fit_transform(self, X):
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)
```

---

### Question 2.2: Handle Missing Values (Easy)

**Problem**: Implement a simple imputer for missing values.

```python
def impute_missing(X, strategy='mean'):
    """
    Impute missing values (NaN) in numpy array.
    strategy: 'mean', 'median', or 'most_frequent'
    """
    # TODO: Implement
    pass

# Test
X = np.array([[1, 2, np.nan],
              [3, np.nan, 5],
              [6, 7, 8]])
X_imputed = impute_missing(X, strategy='mean')
```

---

### Question 2.3: One-Hot Encoding (Easy)

**Problem**: Implement one-hot encoding for categorical features.

```python
def one_hot_encode(y, num_classes=None):
    """
    Convert class labels to one-hot encoded matrix.
    y: array of shape (n_samples,) with integer class labels
    Returns: array of shape (n_samples, num_classes)
    """
    # TODO: Implement
    pass

# Test
y = np.array([0, 1, 2, 1, 0])
encoded = one_hot_encode(y, num_classes=3)
# Expected: [[1,0,0], [0,1,0], [0,0,1], [0,1,0], [1,0,0]]
```

---

### Question 2.4: Train-Test Split (Easy)

**Problem**: Implement train-test split with shuffle.

```python
def train_test_split(X, y, test_size=0.2, random_state=None):
    """
    Split data into train and test sets.
    Returns: X_train, X_test, y_train, y_test
    """
    # TODO: Implement
    # 1. Shuffle data
    # 2. Split based on test_size
    pass
```

---

### Question 2.5: Create Polynomial Features (Medium)

**Problem**: Generate polynomial features for linear regression.

```python
def polynomial_features(X, degree=2):
    """
    Generate polynomial features up to given degree.
    E.g., for degree=2 and X=[x1, x2]:
    Returns: [1, x1, x2, x1^2, x1*x2, x2^2]
    """
    # TODO: Implement
    pass

# Test
X = np.array([[1, 2], [3, 4]])
X_poly = polynomial_features(X, degree=2)
```

---

## Category 3: Model Training & Evaluation

### Question 3.1: K-Fold Cross Validation (Medium)

**Problem**: Implement k-fold cross validation from scratch.

```python
def k_fold_cross_validation(X, y, model, k=5):
    """
    Perform k-fold cross validation.
    Returns: list of scores for each fold
    """
    # TODO: Implement
    # 1. Split data into k folds
    # 2. For each fold:
    #    - Use fold as validation, rest as training
    #    - Train model, evaluate on validation
    # 3. Return scores
    pass

# Example usage
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=100, random_state=42)
model = LogisticRegression()
scores = k_fold_cross_validation(X, y, model, k=5)
print(f"Cross-validation scores: {scores}")
print(f"Mean accuracy: {np.mean(scores)}")
```

---

### Question 3.2: Confusion Matrix and Metrics (Easy-Medium)

**Problem**: Implement confusion matrix and calculate precision, recall, F1.

```python
def confusion_matrix(y_true, y_pred):
    """
    Calculate confusion matrix for binary classification.
    Returns: 2x2 numpy array
    """
    # TODO: Implement
    pass

def precision_recall_f1(y_true, y_pred):
    """
    Calculate precision, recall, and F1 score.
    Returns: tuple (precision, recall, f1)
    """
    # TODO: Implement from confusion matrix
    pass

# Test
y_true = np.array([0, 1, 1, 0, 1, 1, 0, 0, 1, 0])
y_pred = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 0])
cm = confusion_matrix(y_true, y_pred)
p, r, f1 = precision_recall_f1(y_true, y_pred)
print(f"Confusion Matrix:\n{cm}")
print(f"Precision: {p:.2f}, Recall: {r:.2f}, F1: {f1:.2f}")
```

**Key Concepts**:
- TP, TN, FP, FN
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
- F1 = 2 * (Precision * Recall) / (Precision + Recall)

---

### Question 3.3: ROC Curve and AUC (Medium)

**Problem**: Calculate ROC curve and AUC score.

```python
def roc_curve(y_true, y_scores):
    """
    Calculate ROC curve.
    y_scores: predicted probabilities
    Returns: (fpr, tpr, thresholds)
    """
    # TODO: Implement
    # 1. Sort by scores
    # 2. For each unique threshold:
    #    - Calculate TPR and FPR
    pass

def auc(fpr, tpr):
    """Calculate area under curve using trapezoidal rule."""
    # TODO: Implement
    pass
```

---

## Category 4: Neural Network Implementation

### Question 4.1: Single Layer Neural Network (Medium-Hard)

**Problem**: Implement a single-layer neural network with backpropagation.

```python
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.lr = learning_rate
        # Initialize weights
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return z * (1 - z)

    def forward(self, X):
        """
        Forward propagation.
        Returns: output, hidden_activation
        """
        # TODO: Implement
        # Layer 1: z1 = X @ W1 + b1, a1 = sigmoid(z1)
        # Layer 2: z2 = a1 @ W2 + b2, a2 = sigmoid(z2)
        pass

    def backward(self, X, y, output, hidden):
        """
        Backward propagation.
        Updates weights and biases.
        """
        # TODO: Implement
        # Calculate gradients using chain rule
        # Update W1, b1, W2, b2
        pass

    def fit(self, X, y, epochs=1000):
        """Train the network."""
        for epoch in range(epochs):
            # Forward pass
            output, hidden = self.forward(X)
            # Backward pass
            self.backward(X, y, output, hidden)

    def predict(self, X):
        """Make predictions."""
        output, _ = self.forward(X)
        return (output > 0.5).astype(int)

# Test
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])  # XOR problem

nn = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)
nn.fit(X, y, epochs=10000)
predictions = nn.predict(X)
print(f"Predictions:\n{predictions}")
```

**Key Concepts**:
- Forward propagation: z = Wx + b, a = σ(z)
- Backpropagation: Calculate gradients using chain rule
- Weight initialization (Xavier, He)
- Activation functions (sigmoid, ReLU, tanh)

---

### Question 4.2: Implement Softmax and Cross-Entropy (Medium)

**Problem**: Implement softmax activation and cross-entropy loss.

```python
def softmax(z):
    """
    Softmax activation for multi-class classification.
    z: array of shape (n_samples, n_classes)
    Returns: probabilities of shape (n_samples, n_classes)
    """
    # TODO: Implement
    # Hint: Use numerical stability trick (subtract max)
    pass

def cross_entropy_loss(y_true, y_pred):
    """
    Cross-entropy loss for multi-class classification.
    y_true: one-hot encoded labels of shape (n_samples, n_classes)
    y_pred: predicted probabilities of shape (n_samples, n_classes)
    """
    # TODO: Implement
    pass

# Test
z = np.array([[2.0, 1.0, 0.1],
              [1.0, 3.0, 0.2]])
probs = softmax(z)
print(f"Softmax probabilities:\n{probs}")
print(f"Sum of probabilities: {probs.sum(axis=1)}")  # Should be [1, 1]
```

**Key Concepts**:
- Softmax: exp(z_i) / Σ exp(z_j)
- Numerical stability: subtract max before exp
- Cross-entropy: -Σ y * log(ŷ)

---

## Category 5: ML-Specific Problems

### Question 5.1: Implement Mini-Batch Gradient Descent (Medium)

**Problem**: Modify gradient descent to use mini-batches.

```python
def mini_batch_gradient_descent(X, y, learning_rate=0.01,
                                 batch_size=32, epochs=100):
    """
    Implement mini-batch gradient descent for linear regression.
    """
    # TODO: Implement
    # 1. Initialize weights
    # 2. For each epoch:
    #    a. Shuffle data
    #    b. For each mini-batch:
    #       - Compute gradient on batch
    #       - Update weights
    pass
```

---

### Question 5.2: Implement L2 Regularization (Medium)

**Problem**: Add L2 regularization to linear regression.

```python
class RidgeRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000, lambda_reg=1.0):
        self.lr = learning_rate
        self.n_iterations = n_iterations
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        """
        Train with L2 regularization.
        Loss = MSE + lambda * ||w||^2
        """
        # TODO: Implement
        # Add regularization term to gradient
        pass
```

---

### Question 5.3: Principal Component Analysis (PCA) (Hard)

**Problem**: Implement PCA for dimensionality reduction.

```python
class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        """
        Fit PCA.
        Steps:
        1. Center the data (subtract mean)
        2. Compute covariance matrix
        3. Compute eigenvectors and eigenvalues
        4. Select top n_components eigenvectors
        """
        # TODO: Implement
        pass

    def transform(self, X):
        """Project data onto principal components."""
        # TODO: Implement
        pass

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

# Test
X = np.random.randn(100, 5)
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
print(f"Original shape: {X.shape}, Reduced shape: {X_reduced.shape}")
```

---

### Question 5.4: Implement Gradient Checking (Medium)

**Problem**: Verify backpropagation implementation using numerical gradients.

```python
def numerical_gradient(f, x, epsilon=1e-5):
    """
    Calculate numerical gradient of function f at x.
    f: function that takes x and returns scalar loss
    x: parameters (can be multi-dimensional)
    """
    # TODO: Implement
    # Use finite difference: (f(x + eps) - f(x - eps)) / (2 * eps)
    pass

def gradient_check(analytical_grad, numerical_grad, epsilon=1e-7):
    """
    Check if analytical gradient is correct.
    Returns True if they match within epsilon.
    """
    # TODO: Implement
    pass
```

---

### Question 5.5: Implement Early Stopping (Medium)

**Problem**: Add early stopping to prevent overfitting.

```python
class ModelWithEarlyStopping:
    def __init__(self, patience=10):
        self.patience = patience

    def fit(self, X_train, y_train, X_val, y_val, epochs=1000):
        """
        Train with early stopping.
        Stop if validation loss doesn't improve for 'patience' epochs.
        """
        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            # TODO: Implement
            # 1. Train for one epoch
            # 2. Evaluate on validation set
            # 3. Check if validation loss improved
            # 4. If not, increment patience counter
            # 5. If patience exceeded, stop training
            pass
```

---

## Bonus: Real-World Coding Problems

### Bonus 1: Imbalanced Dataset Handling

**Problem**: Implement SMOTE (Synthetic Minority Over-sampling Technique).

```python
def smote(X, y, k=5, ratio=1.0):
    """
    Generate synthetic samples for minority class.
    k: number of nearest neighbors to consider
    ratio: desired ratio of minority to majority samples
    """
    # TODO: Implement
    # 1. Identify minority class
    # 2. For each minority sample:
    #    a. Find k nearest neighbors (same class)
    #    b. Generate synthetic samples along line to neighbors
    pass
```

---

### Bonus 2: Implement Learning Rate Scheduler

**Problem**: Implement learning rate decay.

```python
def learning_rate_schedule(initial_lr, epoch, schedule='exponential'):
    """
    Calculate learning rate for given epoch.
    Schedules:
    - 'exponential': lr = initial_lr * decay_rate^epoch
    - 'step': lr = initial_lr * drop_rate^floor(epoch / step_size)
    - 'cosine': cosine annealing
    """
    # TODO: Implement
    pass
```

---

## Interview Tips for Coding Questions

### Preparation
1. **Practice implementing 5-10 algorithms from scratch**
2. **Know NumPy well** - vectorization is key
3. **Understand time/space complexity** of your implementations
4. **Test your code** with simple examples

### During Interview
1. **Clarify requirements** - Ask about input/output formats, edge cases
2. **Start with pseudocode** - Outline approach before coding
3. **Think out loud** - Explain your reasoning
4. **Start simple, then optimize** - Get a working solution first
5. **Test as you go** - Use simple examples to verify
6. **Discuss trade-offs** - Time vs. space, accuracy vs. speed

### Common Pitfalls
- Not handling edge cases (empty input, single sample, etc.)
- Inefficient loops (use vectorization)
- Not normalizing/standardizing features
- Incorrect matrix dimensions
- Numerical instability (log(0), exp overflow)

---

## Practice Strategy

**Week 1-2**: Implement 5 basic algorithms
- Linear regression
- Logistic regression
- K-means
- KNN
- Decision tree (optional)

**Week 3**: Data preprocessing
- Scaling, encoding, imputation
- Train-test split, cross-validation

**Week 4**: Evaluation metrics
- Confusion matrix, precision, recall, F1
- ROC, AUC

**Week 5+**: Neural networks and advanced topics
- Backpropagation
- Regularization
- Optimization

**Practice on**:
- Your own implementations (refer to this document)
- LeetCode (if ML problems available)
- Kaggle notebooks (implement algorithms yourself)

---

## Resources

- NumPy documentation: https://numpy.org/doc/
- "Hands-On Machine Learning" by Aurélien Géron (book)
- "Machine Learning from Scratch" tutorials
- Your own implementations (best way to learn!)

**Next Step**: Start implementing! Begin with Linear Regression and work your way through the list.
