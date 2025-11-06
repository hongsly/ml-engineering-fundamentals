# ML Theory Interview Questions

This document contains theory and concept questions commonly asked in ML Engineer interviews. These test your understanding of ML fundamentals, algorithms, and best practices.

---

## Categories

1. **Fundamentals & General Concepts**
2. **Classical Machine Learning**
3. **Deep Learning & Neural Networks**
4. **Model Training & Optimization**
5. **Model Evaluation & Validation**
6. **Feature Engineering & Data**
7. **Production ML & MLOps**
8. **Modern AI & LLMs**

---

## 1. Fundamentals & General Concepts

### Q1.1: Explain the bias-variance tradeoff.

**Answer**:
- **Bias**: Error from incorrect assumptions in the model. High bias → underfitting (model too simple)
- **Variance**: Error from sensitivity to small fluctuations in training data. High variance → overfitting (model too complex)
- **Tradeoff**: As model complexity increases, bias ↓ but variance ↑
- **Goal**: Find sweet spot that minimizes total error = bias² + variance + irreducible error

**Example**:
- High bias: Linear regression for non-linear relationship
- High variance: Deep neural network on small dataset
- Balanced: Regularized model with appropriate complexity

**Follow-up**: How do you diagnose and fix high bias vs. high variance?
- High bias: Increase model complexity, add features, reduce regularization
- High variance: Get more data, reduce features, increase regularization, use ensemble methods

---

### Q1.2: What is overfitting and how do you prevent it?

**Answer**:
Overfitting occurs when model learns training data too well, including noise, and performs poorly on new data.

**Signs of overfitting**:
- High training accuracy, low validation/test accuracy
- Model performs well on training data but poorly in production
- Large gap between train and validation loss

**Prevention methods**:
1. **More data**: Dilutes noise
2. **Regularization**: L1, L2, dropout, early stopping
3. **Simpler model**: Reduce parameters/complexity
4. **Cross-validation**: Better estimate of generalization
5. **Data augmentation**: Artificially increase training data
6. **Ensemble methods**: Average multiple models
7. **Feature selection**: Remove irrelevant features

---

### Q1.3: Supervised vs. Unsupervised vs. Semi-supervised vs. Reinforcement Learning

**Supervised Learning**:
- Has labeled data (X, y pairs)
- Goal: Learn mapping X → y
- Examples: Classification, regression
- Algorithms: Linear/logistic regression, decision trees, neural networks

**Unsupervised Learning**:
- No labels, only features (X)
- Goal: Find patterns/structure in data
- Examples: Clustering, dimensionality reduction, anomaly detection
- Algorithms: K-means, PCA, autoencoders

**Semi-supervised Learning**:
- Small amount of labeled data + large amount of unlabeled data
- Use unlabeled data to improve model
- Common in real-world (labeling is expensive)
- Techniques: Self-training, co-training, pseudo-labeling

**Reinforcement Learning**:
- Agent learns by interacting with environment
- Receives rewards/penalties for actions
- Goal: Maximize cumulative reward
- Examples: Game playing, robotics, recommendation systems
- Algorithms: Q-learning, policy gradients, actor-critic

---

### Q1.4: Explain precision, recall, and F1-score. When would you prioritize one over the other?

**Definitions**:
- **Precision**: Of predicted positives, how many are actually positive? TP / (TP + FP)
  - "How precise/accurate are positive predictions?"
- **Recall** (Sensitivity): Of actual positives, how many did we predict? TP / (TP + FN)
  - "How well do we recall/capture all positives?"
- **F1-Score**: Harmonic mean of precision and recall: 2 × (P × R) / (P + R)

**When to prioritize**:

| Scenario | Metric | Reason |
|----------|--------|--------|
| Spam detection | Precision | Don't want to mark legitimate emails as spam (false positives costly) |
| Cancer diagnosis | Recall | Don't want to miss cancer cases (false negatives deadly) |
| Fraud detection | Balance (F1) | Both false positives (blocking legit transactions) and false negatives (missing fraud) are costly |
| Information retrieval | Precision@K | Users only see top K results, all should be relevant |

**F-beta score**: Generalization that allows weighting
- F2: Weights recall higher (cancer detection)
- F0.5: Weights precision higher (spam detection)

---

### Q1.5: What is cross-validation and why is it important?

**Answer**:
Cross-validation is a technique to assess model generalization by training and evaluating on different subsets of data.

**K-Fold Cross-Validation**:
1. Split data into K folds
2. For each fold:
   - Train on K-1 folds
   - Validate on remaining fold
3. Average performance across K folds

**Why important**:
- Better estimate of generalization (uses all data for both training and validation)
- Reduces variance in performance estimate
- Helps detect overfitting
- More robust than single train-test split

**Variants**:
- **Stratified K-Fold**: Maintains class distribution in each fold (for imbalanced data)
- **Leave-One-Out (LOO)**: K = n (expensive but low bias)
- **Time Series Split**: Respects temporal order (can't use future to predict past)

**When to use**:
- Small datasets (need to use all data)
- Hyperparameter tuning
- Model selection
- Estimating model performance

---

## 2. Classical Machine Learning

### Q2.1: Compare decision trees, random forests, and gradient boosting.

| Aspect | Decision Tree | Random Forest | Gradient Boosting |
|--------|---------------|---------------|-------------------|
| **Type** | Single model | Ensemble (bagging) | Ensemble (boosting) |
| **Training** | Greedy splits | Multiple trees in parallel | Sequential trees |
| **Prediction** | One tree | Average/vote | Weighted sum |
| **Overfitting** | High tendency | Low (averaging) | Can overfit if not careful |
| **Interpretability** | High | Low | Low |
| **Speed** | Fast | Moderate | Slow (sequential) |
| **Handling outliers** | Poor | Good | Sensitive |

**When to use**:
- **Decision Tree**: Need interpretability, simple baseline
- **Random Forest**: General-purpose, robust, handles high variance
- **Gradient Boosting**: Need highest accuracy, willing to tune extensively

---

### Q2.2: What is regularization? Explain L1 vs. L2.

**Regularization**: Technique to prevent overfitting by penalizing model complexity.

**L2 Regularization (Ridge)**:
- Penalty: λ × Σ w²
- Effect: Shrinks weights toward 0 (but not exactly 0)
- All features retained with small weights
- Smooth weight distribution
- **Use when**: All features are relevant, want to reduce impact of collinearity

**L1 Regularization (Lasso)**:
- Penalty: λ × Σ |w|
- Effect: Drives some weights to exactly 0
- Performs feature selection
- Sparse weights
- **Use when**: Many irrelevant features, want automatic feature selection

**Elastic Net**:
- Combines L1 and L2: λ1 × Σ |w| + λ2 × Σ w²
- Gets benefits of both
- **Use when**: High-dimensional data with correlated features

**Other regularization techniques**:
- Dropout (neural networks)
- Early stopping
- Data augmentation
- Batch normalization

---

### Q2.3: How does k-means clustering work? What are its limitations?

**Algorithm**:
1. Initialize K centroids (randomly or k-means++)
2. Assign each point to nearest centroid
3. Update centroids as mean of assigned points
4. Repeat until convergence (centroids don't change)

**Limitations**:
1. **Must specify K**: Need to know number of clusters beforehand
2. **Assumes spherical clusters**: Doesn't work well for elongated/irregular shapes
3. **Sensitive to initialization**: Can converge to local optima (solution: k-means++)
4. **Sensitive to outliers**: Outliers pull centroids
5. **Assumes equal-sized clusters**: May split large clusters
6. **Only Euclidean distance**: Doesn't work well in high dimensions

**Improvements**:
- K-means++: Better initialization
- Elbow method / silhouette score: Choose K
- DBSCAN: Handles arbitrary shapes, doesn't require K
- Hierarchical clustering: Provides dendrogram

---

### Q2.4: Explain how SVM (Support Vector Machine) works.

**Core Idea**: Find hyperplane that maximally separates classes.

**Key Concepts**:
- **Hyperplane**: Decision boundary (line in 2D, plane in 3D, etc.)
- **Support Vectors**: Points closest to hyperplane (these define it)
- **Margin**: Distance between hyperplane and nearest points from each class
- **Goal**: Maximize margin

**For non-linearly separable data**:
- **Soft margin**: Allow some misclassifications with penalty (C parameter)
- **Kernel trick**: Map to higher dimension where data is separable
  - Linear kernel: No mapping
  - Polynomial kernel: Creates polynomial features
  - RBF (Gaussian) kernel: Infinite-dimensional mapping

**Pros**:
- Effective in high dimensions
- Memory efficient (only uses support vectors)
- Versatile (different kernels)

**Cons**:
- Slow for large datasets
- Sensitive to feature scaling
- Choosing right kernel is non-trivial
- Doesn't output probabilities directly

---

### Q2.5: What is the difference between bagging and boosting?

**Bagging (Bootstrap Aggregating)**:
- Train multiple models **in parallel** on different bootstrap samples
- Combine via averaging (regression) or voting (classification)
- Reduces **variance** (helps with overfitting)
- Models are **independent**
- Example: Random Forest

**Boosting**:
- Train models **sequentially**, each correcting previous model's errors
- Combine via weighted sum
- Reduces **bias** (improves underfitting)
- Models are **dependent** (sequential)
- Examples: AdaBoost, Gradient Boosting, XGBoost

| Aspect | Bagging | Boosting |
|--------|---------|----------|
| Training | Parallel | Sequential |
| Focus | Reduce variance | Reduce bias |
| Weak learners | Can be strong | Must be weak |
| Weights | Equal | Weighted |
| Overfitting | Less prone | More prone |
| Examples | Random Forest | XGBoost, AdaBoost |

---

## 3. Deep Learning & Neural Networks

### Q3.1: Explain forward and backward propagation.

**Forward Propagation**:
- Pass input through network layer by layer
- At each layer: z = Wx + b, a = activation(z)
- Output final prediction
- Calculate loss

**Backward Propagation**:
- Calculate gradient of loss w.r.t. weights
- Use chain rule to propagate gradients backward
- Update weights: w = w - learning_rate × gradient

**Example** (simple 2-layer network):
```
Forward:
  z1 = W1 × x + b1
  a1 = σ(z1)
  z2 = W2 × a1 + b2
  y_pred = σ(z2)
  Loss = (y - y_pred)²

Backward:
  dL/dW2 = dL/dy_pred × dy_pred/dz2 × dz2/dW2
  dL/dW1 = dL/dy_pred × dy_pred/dz2 × dz2/da1 × da1/dz1 × dz1/dW1
```

---

### Q3.2: What is the vanishing/exploding gradient problem?

**Vanishing Gradient**:
- Gradients become extremely small as they propagate backward
- Layers far from output receive tiny updates → learn slowly
- **Cause**: Multiplying many small numbers (< 1)
- Common with sigmoid/tanh activation (gradients 0-0.25)
- Problem in deep networks and RNNs

**Solutions**:
- Use ReLU activation (gradient = 1 for positive values)
- Batch normalization
- Residual connections (ResNet)
- Careful weight initialization (Xavier, He)
- LSTM/GRU for RNNs
- Gradient clipping

**Exploding Gradient**:
- Gradients become extremely large
- **Cause**: Multiplying many large numbers (> 1)
- Leads to unstable training, NaN values

**Solutions**:
- Gradient clipping (cap gradient magnitude)
- Lower learning rate
- Batch normalization
- Weight regularization

---

### Q3.3: Compare different activation functions.

| Activation | Formula | Range | Pros | Cons | Use Case |
|------------|---------|-------|------|------|----------|
| **Sigmoid** | 1/(1+e^(-x)) | (0, 1) | Smooth, interpretable as probability | Vanishing gradient, not zero-centered | Output layer (binary classification) |
| **Tanh** | (e^x - e^(-x))/(e^x + e^(-x)) | (-1, 1) | Zero-centered, smooth | Vanishing gradient | RNNs (less common now) |
| **ReLU** | max(0, x) | [0, ∞) | Fast, no vanishing gradient for x>0 | Dead neurons (x<0), not zero-centered | Default for hidden layers |
| **Leaky ReLU** | max(0.01x, x) | (-∞, ∞) | Fixes dead ReLU | Small negative slope still arbitrary | Alternative to ReLU |
| **ELU** | x if x>0, α(e^x-1) if x≤0 | (-α, ∞) | Smooth, negative values | Slower computation | When smooth gradients matter |
| **Softmax** | e^x_i / Σe^x_j | (0, 1), sum=1 | Outputs probability distribution | N/A | Output layer (multi-class) |

**General rules**:
- **Hidden layers**: Start with ReLU, try Leaky ReLU if issues
- **Output layer (classification)**: Sigmoid (binary), Softmax (multi-class)
- **Output layer (regression)**: Linear (no activation)
- **RNNs**: Tanh (historically), but LSTM/GRU have built-in activations

---

### Q3.4: What is dropout and how does it work?

**Dropout**: Regularization technique that randomly "drops" (sets to 0) neurons during training.

**How it works**:
1. During training: Each neuron kept with probability p (e.g., 0.5)
2. Dropped neurons don't contribute to forward or backward pass
3. At test time: Use all neurons, scale outputs by p

**Why it works**:
- Prevents co-adaptation (neurons becoming too dependent on each other)
- Forces network to learn redundant representations
- Equivalent to training ensemble of networks
- Acts as strong regularizer

**Best practices**:
- Use p=0.5 for hidden layers, p=0.8 for input layer
- Not needed if you have lots of data
- Less effective for CNNs (use batch norm instead)
- Turn off during inference

---

### Q3.5: Explain batch normalization.

**Batch Normalization**: Normalizes inputs to each layer using mini-batch statistics.

**What it does**:
For each mini-batch at layer l:
1. Calculate mean and variance of activations
2. Normalize: x_norm = (x - mean) / sqrt(variance + ε)
3. Scale and shift: y = γ × x_norm + β (learnable parameters)

**Benefits**:
- Faster training (can use higher learning rates)
- Reduces internal covariate shift
- Acts as regularization (slight noise from batch statistics)
- Reduces sensitivity to weight initialization

**Where to apply**:
- After linear transformation, before activation: Dense → BatchNorm → Activation
- Or after activation (less common now)

**Drawbacks**:
- Batch size dependent (performance degrades with very small batches)
- Different behavior in train vs. test (uses moving averages at test time)
- Not ideal for RNNs (use Layer Norm instead)

---

## 4. Model Training & Optimization

### Q4.1: Compare SGD, Momentum, Adam, and other optimizers.

**Stochastic Gradient Descent (SGD)**:
- Update: w = w - lr × gradient
- Pros: Simple, works well if tuned
- Cons: Slow convergence, can get stuck in local minima

**SGD with Momentum**:
- Adds "velocity": v = β × v + gradient, w = w - lr × v
- Helps escape local minima and accelerate in consistent directions
- β typically 0.9

**RMSprop**:
- Adapts learning rate per parameter based on recent gradient magnitudes
- Good for RNNs and non-stationary problems

**Adam (Adaptive Moment Estimation)**:
- Combines momentum + RMSprop
- Maintains both first moment (momentum) and second moment (variance)
- Most popular optimizer (good default choice)
- Hyperparameters: lr=0.001, β1=0.9, β2=0.999

**AdamW**:
- Adam with decoupled weight decay
- Better regularization than L2 in Adam
- Preferred for transformers/LLMs

**When to use**:
- **Adam**: Default choice, works well out of the box
- **SGD + Momentum**: Can achieve better generalization if tuned well
- **AdamW**: Transformers and modern architectures

---

### Q4.2: What is learning rate and how do you choose it?

**Learning Rate**: Step size for weight updates in gradient descent.

**Too high**: Training unstable, loss oscillates or diverges
**Too low**: Training slow, may get stuck in local minima

**Choosing learning rate**:
1. **Learning rate finder**: Gradually increase lr and plot loss (choose lr before loss increases)
2. **Start with default**: Adam: 0.001, SGD: 0.01-0.1
3. **Learning rate schedule**: Adjust during training

**Learning rate schedules**:
- **Step decay**: Reduce by factor every N epochs
- **Exponential decay**: lr = lr0 × e^(-kt)
- **Cosine annealing**: Smooth decrease following cosine curve
- **Reduce on plateau**: Decrease when validation loss stops improving
- **Warmup + decay**: Start small, increase, then decrease (common for transformers)

**Adaptive optimizers** (Adam, RMSprop): Less sensitive to learning rate choice.

---

### Q4.3: What is transfer learning and when should you use it?

**Transfer Learning**: Using a model trained on one task as starting point for another task.

**How it works**:
1. Take pre-trained model (e.g., ResNet on ImageNet)
2. Remove final layer(s)
3. Add new layers for your task
4. Fine-tune:
   - **Option A**: Freeze early layers, train only new layers (small dataset)
   - **Option B**: Fine-tune all layers with small learning rate (larger dataset)

**When to use**:
- **Small dataset**: Pre-trained features prevent overfitting
- **Limited compute**: Don't need to train from scratch
- **Similar domain**: Source and target tasks are related

**Common scenarios**:
- **Computer vision**: Pre-trained on ImageNet (ResNet, VGG, EfficientNet)
- **NLP**: Pre-trained on large text corpus (BERT, GPT, RoBERTa)
- **Speech**: Pre-trained on large audio corpus (Wav2Vec)

**Benefits**:
- Faster training
- Better performance with less data
- Leverages knowledge from large datasets

---

### Q4.4: Explain data augmentation and give examples.

**Data Augmentation**: Artificially creating new training samples by applying transformations.

**Benefits**:
- Increases effective training data size
- Improves generalization
- Reduces overfitting
- Makes model robust to variations

**Computer Vision**:
- Geometric: Rotation, flip, crop, zoom, shear
- Color: Brightness, contrast, saturation, hue adjustment
- Noise: Gaussian noise, blur
- Advanced: Cutout, mixup, cutmix, random erasing

**NLP**:
- Synonym replacement
- Random insertion/deletion of words
- Sentence shuffling
- Back translation (translate to another language and back)
- Paraphrasing with LLMs

**Audio**:
- Time stretching
- Pitch shifting
- Adding background noise
- Time masking, frequency masking (SpecAugment)

**Best practices**:
- Apply augmentation during training, not testing
- Use domain-appropriate augmentations (don't flip medical images if orientation matters)
- Too much augmentation can hurt performance

---

### Q4.5: What is early stopping?

**Early Stopping**: Stop training when validation performance stops improving.

**How it works**:
1. Monitor validation loss (or other metric) each epoch
2. If no improvement for N epochs (patience), stop training
3. Restore best model weights

**Why it works**:
- Prevents overfitting (model starts memorizing training data)
- Saves computation time
- Acts as regularization

**Parameters**:
- **Patience**: How many epochs to wait (e.g., 5-10)
- **Min delta**: Minimum improvement to count as improvement
- **Metric**: Validation loss (most common) or accuracy

**Implementation considerations**:
- Save model checkpoint at each improvement
- Balance: Too early → underfitting, too late → overfitting
- Use with learning rate schedule (don't stop during decay)

---

## 5. Model Evaluation & Validation

### Q5.1: What is AUC-ROC and when is it useful?

**ROC Curve (Receiver Operating Characteristic)**:
- Plot of True Positive Rate (TPR) vs. False Positive Rate (FPR) at different thresholds
- TPR = Recall = TP/(TP+FN)
- FPR = FP/(FP+TN)

**AUC (Area Under Curve)**:
- Single number summarizing ROC curve
- Range: 0 to 1
- Interpretation: Probability that model ranks random positive higher than random negative
- 0.5 = random classifier, 1.0 = perfect classifier

**When to use**:
- **Imbalanced datasets**: Not affected by class imbalance (unlike accuracy)
- **Comparing models**: Single metric for comparison
- **Threshold-independent**: Evaluates model across all thresholds

**When NOT to use**:
- **Severe imbalance + care about precision**: Use Precision-Recall AUC instead
- **Specific threshold required**: Look at precision/recall at that threshold
- **Multi-class**: Need to use one-vs-rest or other approach

---

### Q5.2: How do you handle imbalanced datasets?

**Problem**: One class has many more samples than others (e.g., fraud detection: 99.9% legitimate, 0.1% fraud).

**Solutions**:

**1. Resampling**:
- **Oversample minority**: Duplicate or generate synthetic samples (SMOTE)
- **Undersample majority**: Remove samples from majority class
- **Combine both**: Hybrid approach

**2. Algorithmic approaches**:
- **Class weights**: Penalize misclassification of minority class more
- **Focal loss**: Focus on hard examples (used in object detection)
- **Ensemble methods**: Combine multiple models trained on balanced subsets

**3. Evaluation metrics**:
- Don't use accuracy (can be high by predicting majority class)
- Use: Precision, Recall, F1, AUC-ROC, Precision-Recall AUC

**4. Collect more data**:
- For minority class specifically

**5. Anomaly detection**:
- Frame as outlier detection if extreme imbalance

**6. Two-stage approach**:
- Stage 1: Simple model to filter out obvious majority
- Stage 2: Complex model on balanced subset

---

### Q5.3: What is the difference between parametric and non-parametric models?

**Parametric Models**:
- Fixed number of parameters (independent of training data size)
- Make assumptions about data distribution
- Examples: Linear regression, logistic regression, neural networks
- **Pros**: Fast inference, work well with less data if assumptions hold
- **Cons**: Limited flexibility, assumptions may not hold

**Non-parametric Models**:
- Number of parameters grows with training data
- Few assumptions about data distribution
- Examples: KNN, decision trees, kernel SVM
- **Pros**: Flexible, can model complex relationships
- **Cons**: Slow inference, need more data, risk of overfitting

| Aspect | Parametric | Non-parametric |
|--------|------------|----------------|
| Parameters | Fixed | Grows with data |
| Assumptions | Strong | Few |
| Data required | Less | More |
| Interpretability | Often higher | Often lower |
| Speed (inference) | Fast | Can be slow |

---

### Q5.4: Explain holdout, cross-validation, and bootstrap.

**Holdout Validation**:
- Split data once: Train (70-80%), Validation (10-15%), Test (10-15%)
- Train on train set, tune on validation set, final eval on test set
- **Pros**: Simple, fast
- **Cons**: High variance (depends on split), wastes data

**Cross-Validation**:
- K-fold: Split into K parts, train on K-1, validate on 1, repeat K times
- **Pros**: Uses all data, lower variance estimate
- **Cons**: K times more computation
- Use for: Model selection, hyperparameter tuning

**Bootstrap**:
- Repeatedly sample with replacement from dataset
- Train on sample, test on out-of-bag samples
- **Pros**: Good estimate of uncertainty
- **Cons**: Computationally expensive
- Use for: Confidence intervals, ensemble methods (bagging)

**Best practice**:
- Always have separate test set (never used during development)
- Use cross-validation for model selection/tuning
- Report performance on held-out test set

---

### Q5.5: What metrics would you use for a regression problem?

**Common Regression Metrics**:

**MSE (Mean Squared Error)**:
- Average of squared errors: (1/n) Σ (y - ŷ)²
- Pros: Smooth, differentiable (good for optimization)
- Cons: Sensitive to outliers (squares large errors)

**RMSE (Root Mean Squared Error)**:
- sqrt(MSE)
- Same units as target variable (more interpretable)

**MAE (Mean Absolute Error)**:
- Average of absolute errors: (1/n) Σ |y - ŷ|
- Pros: Less sensitive to outliers than MSE
- Cons: Not differentiable at 0

**R² (R-squared, Coefficient of Determination)**:
- Proportion of variance explained: 1 - (SS_res / SS_tot)
- Range: (-∞, 1], 1 = perfect, 0 = baseline (mean)
- Pros: Interpretable, normalized
- Cons: Can be misleading with non-linear relationships

**MAPE (Mean Absolute Percentage Error)**:
- (1/n) Σ |y - ŷ| / |y| × 100%
- Pros: Scale-independent, easy to explain
- Cons: Undefined if y=0, sensitive to small denominators

**Choose based on**:
- **Outliers important**: Use MAE
- **Penalize large errors more**: Use MSE/RMSE
- **Scale-independent comparison**: Use R² or MAPE
- **Business metric**: Align with actual cost function

---

## 6. Feature Engineering & Data

### Q6.1: How do you handle missing data?

**Strategies**:

**1. Deletion**:
- **Listwise deletion**: Remove entire row if any value missing
- **Column deletion**: Remove feature if too many missing values
- Use when: Data is MCAR (Missing Completely At Random) and dataset is large

**2. Imputation**:
- **Mean/median/mode**: Simple, fast
- **Forward/backward fill**: For time series
- **Model-based**: Predict missing values using other features (KNN, regression)
- **Multiple imputation**: Generate multiple imputed datasets

**3. Indicator variable**:
- Create binary flag indicating missingness
- Missingness itself may be informative

**4. Use algorithms that handle missing data**:
- XGBoost, LightGBM can handle NaN natively
- Decision trees can treat missing as separate category

**Best practice**:
- Understand why data is missing (MCAR, MAR, MNAR)
- Impute only after train/test split (use train statistics for test)
- Compare performance with different strategies
- Consider missingness as a feature

---

### Q6.2: How do you handle categorical features?

**Encoding Techniques**:

**1. One-Hot Encoding**:
- Create binary column for each category
- Use when: Few categories (< 10-20), no ordinal relationship
- Cons: High dimensionality with many categories

**2. Label Encoding**:
- Assign integer to each category (0, 1, 2, ...)
- Use when: Ordinal relationship exists (low, medium, high)
- Don't use: For nominal categories (implies ordering)

**3. Target Encoding**:
- Replace category with mean target value
- Use when: High cardinality features
- Pros: Reduces dimensionality
- Cons: Risk of leakage (use cross-validation)

**4. Frequency Encoding**:
- Replace with frequency of category
- Captures importance without dimensionality explosion

**5. Embedding**:
- Learn dense vector representation
- Use in neural networks for high cardinality

**6. Binary Encoding**:
- Convert to binary, create column per bit
- Reduces dimensionality vs. one-hot

**Choose based on**:
- Cardinality (number of unique values)
- Presence of ordinal relationship
- Model type (tree-based vs. linear)

---

### Q6.3: What is feature scaling and why is it important?

**Feature Scaling**: Transform features to similar ranges.

**Why important**:
- **Gradient descent**: Converges faster with similar scales
- **Distance-based algorithms**: KNN, SVM, K-means give more weight to large-scale features
- **Regularization**: L1/L2 penalty affects features differently based on scale
- **Neural networks**: Speeds up training, prevents vanishing/exploding gradients

**Not needed for**:
- Tree-based models (decision trees, random forest, XGBoost)
- Algorithms invariant to monotonic transformations

**Methods**:

**1. Standardization (Z-score normalization)**:
- x' = (x - mean) / std
- Results in mean=0, std=1
- Preserves outliers
- Use when: Features are normally distributed

**2. Min-Max Scaling**:
- x' = (x - min) / (max - min)
- Results in range [0, 1]
- Sensitive to outliers
- Use when: Need bounded range

**3. Robust Scaling**:
- x' = (x - median) / IQR
- Less sensitive to outliers
- Use when: Data has outliers

**Best practice**:
- Fit scaler on training data only, apply to train/val/test
- Scale after splitting to avoid data leakage

---

### Q6.4: How do you detect and handle outliers?

**Detection Methods**:

**1. Statistical**:
- **Z-score**: Points with |z| > 3 are outliers
- **IQR method**: Outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
- **Grubbs test**: Statistical test for outliers

**2. Visualization**:
- Box plots
- Scatter plots
- Histograms

**3. Model-based**:
- **Isolation Forest**: Isolates anomalies
- **LOF (Local Outlier Factor)**: Density-based
- **One-class SVM**: Learns boundary of normal data
- **Autoencoders**: High reconstruction error for outliers

**Handling Strategies**:

**1. Remove**:
- If data error or extreme anomaly
- If small percentage of data

**2. Cap/Winsorize**:
- Set to threshold (e.g., 95th percentile)
- Preserves data but reduces impact

**3. Transform**:
- Log transformation reduces skewness
- Makes outliers less extreme

**4. Separate model**:
- Treat outliers as separate class
- Build specialized model

**5. Use robust algorithms**:
- Tree-based methods less sensitive
- Robust loss functions (Huber loss)

**6. Keep them**:
- If they represent real, important cases
- Use robust scaling and algorithms

---

### Q6.5: What is feature engineering and give examples?

**Feature Engineering**: Creating new features from existing data to improve model performance.

**Techniques**:

**1. Numeric transformations**:
- Polynomial features: x², x³, x₁×x₂
- Log, sqrt, exponential
- Binning: Convert continuous to categorical

**2. Datetime features**:
- Extract: year, month, day, day_of_week, hour
- Cyclical encoding: sin/cos for hour (23 and 0 are close)
- Time since event: days since last purchase

**3. Categorical combinations**:
- Interaction features: "city_dayofweek"
- Aggregations: count, mean by category

**4. Text features**:
- Length, word count
- TF-IDF scores
- Presence of keywords
- Sentiment score

**5. Domain-specific**:
- E-commerce: recency, frequency, monetary value (RFM)
- Finance: moving averages, volatility
- Healthcare: BMI from height/weight

**6. Aggregations**:
- Rolling statistics: moving average, std
- Group statistics: mean purchase by user

**7. Target encoding**:
- Mean target value per category (with cross-validation)

**Best practices**:
- Understand domain (most important)
- Iterate: Create, test, evaluate
- Avoid data leakage
- Consider feature interactions
- Use feature importance to prune

---

## 7. Production ML & MLOps

### Q7.1: What is data leakage and how do you prevent it?

**Data Leakage**: Information from test/future data "leaks" into training, causing overly optimistic performance estimates.

**Types**:

**1. Train-Test Contamination**:
- Test data influenced by training data
- Example: Scaling before splitting (test statistics used in training)
- **Prevention**: Always split first, then preprocess

**2. Temporal Leakage**:
- Using future information to predict past
- Example: Using avg_purchase (calculated on all data) to predict purchase
- **Prevention**: Use only past data, respect temporal order

**3. Feature Leakage**:
- Features that won't be available at prediction time
- Example: Using "is_fraud" label to predict fraud
- **Prevention**: Think about inference time - what will be available?

**4. Target Leakage**:
- Feature is direct or indirect proxy for target
- Example: Using "late_payment_fee" to predict loan default
- **Prevention**: Remove features too closely related to target

**Common causes**:
- Preprocessing on entire dataset
- Using future data in time series
- Data collection process (labels influence features)
- Duplicates in train and test

**Detection**:
- Performance too good to be true
- Single feature with perfect predictive power
- Poor production performance despite good validation

---

### Q7.2: How do you monitor ML models in production?

**What to Monitor**:

**1. Model Performance**:
- Accuracy, precision, recall, AUC
- Prediction latency
- Throughput (requests/second)

**2. Data Quality**:
- Missing values
- Feature distributions (data drift)
- Outliers, anomalies
- Schema changes

**3. Data Drift**:
- **Covariate shift**: P(X) changes but P(Y|X) stays same
- **Concept drift**: P(Y|X) changes
- Detection: Compare distributions (KS test, Jensen-Shannon divergence)

**4. Model Drift**:
- Performance degrades over time
- Compare recent vs. historical performance

**5. System Health**:
- Server uptime, error rates
- Memory, CPU usage
- API response times

**Strategies**:

**Logging**:
- Log predictions and confidence scores
- Log features (for debugging)
- Sample logging if high volume

**Alerting**:
- Set thresholds for metrics
- Alert when drift detected
- Alert on errors or latency spikes

**Retraining**:
- Scheduled: Retrain weekly/monthly
- Triggered: Retrain when drift detected
- Online learning: Continuous updates

**A/B Testing**:
- Compare new model vs. current in production
- Gradually roll out if better

---

### Q7.3: What is A/B testing and how do you use it for ML?

**A/B Testing**: Experiment comparing two versions (A = control, B = treatment) to determine which performs better.

**For ML models**:
- A: Current model in production
- B: New model/algorithm
- Randomly assign users to A or B
- Measure business metrics (not just model metrics)
- Determine if B is statistically significantly better

**Steps**:
1. **Define hypothesis**: "New model will increase CTR"
2. **Choose metric**: CTR, revenue, engagement, etc.
3. **Determine sample size**: Power analysis (typically need 1000s of samples)
4. **Randomize**: Assign users/requests randomly
5. **Run experiment**: Collect data for sufficient time
6. **Analyze**: Statistical test (t-test, chi-square)
7. **Decide**: Roll out B if significantly better

**Important considerations**:
- **Duration**: Run long enough to account for day-of-week effects
- **Statistical significance**: Typically p < 0.05
- **Practical significance**: Is improvement large enough to matter?
- **Multiple testing**: Correct for multiple comparisons
- **Novelty effect**: Users may engage more initially

**Variants**:
- **Multi-armed bandit**: Dynamically allocate traffic to better variant
- **Interleaving**: Show both results, see which user clicks

---

### Q7.4: How do you handle model versioning and rollback?

**Model Versioning**: Track different versions of models, data, and code.

**What to version**:
- Model weights/artifacts
- Training code
- Training data (or data version/hash)
- Hyperparameters
- Dependencies (requirements.txt)
- Evaluation metrics
- Preprocessing steps

**Tools**:
- **MLflow**: Track experiments, models, parameters
- **DVC (Data Version Control)**: Version data and models
- **Weights & Biases**: Experiment tracking
- **Git**: Version code
- **Docker**: Version environment

**Deployment Strategy**:

**1. Blue-Green Deployment**:
- Run old (blue) and new (green) in parallel
- Switch traffic from blue to green
- Keep blue for rollback

**2. Canary Deployment**:
- Route small % of traffic to new model
- Gradually increase if metrics good
- Rollback if issues

**3. Shadow Mode**:
- New model makes predictions but doesn't serve
- Compare with current model
- Deploy if performance acceptable

**Rollback Strategy**:
- Keep previous model version ready
- Monitor metrics closely after deployment
- Automatic rollback if metrics degrade
- Have rollback plan documented

---

### Q7.5: What are some challenges of deploying ML models?

**Technical Challenges**:

**1. Latency**:
- Inference must be fast (often < 100ms)
- Solutions: Model compression, caching, async processing

**2. Scalability**:
- Handle high request volume
- Solutions: Load balancing, horizontal scaling, batching

**3. Model Size**:
- Large models (LLMs) are expensive to serve
- Solutions: Quantization, distillation, pruning

**4. Feature Store**:
- Need to fetch features quickly at inference
- Solutions: Cache features, precompute, feature store (Feast, Tecton)

**5. Real-time vs. Batch**:
- Real-time: Serve predictions on demand
- Batch: Precompute predictions
- Choice depends on use case

**Data & Concept Drift**:
- Model becomes stale as data distribution changes
- Solutions: Monitoring, retraining pipeline

**Reproducibility**:
- Need to reproduce exact model
- Solutions: Version everything (model, data, code, env)

**Testing**:
- Hard to test ML systems (non-deterministic)
- Solutions: Unit tests, integration tests, shadow mode

**Operational**:
- Monitoring, alerting, on-call
- Model explainability for stakeholders
- Compliance, fairness, privacy concerns

---

## 8. Modern AI & LLMs

### Q8.1: Explain the Transformer architecture.

**Core Components**:

**1. Self-Attention Mechanism**:
- Allows each token to attend to all other tokens
- Captures dependencies regardless of distance
- Attention(Q, K, V) = softmax(QK^T / √d_k) × V

**2. Multi-Head Attention**:
- Multiple attention heads in parallel
- Each head learns different relationships
- Concatenate and project outputs

**3. Positional Encoding**:
- Add position information (Transformers have no inherent order)
- Sine/cosine functions or learned embeddings

**4. Feed-Forward Networks**:
- Applied to each position independently
- Two linear layers with activation (usually GELU)

**5. Layer Normalization & Residual Connections**:
- Stabilize training
- Help gradients flow

**Architecture**:
```
Input → Embedding + Positional Encoding
  ↓
[Encoder Blocks] × N
  - Multi-Head Self-Attention
  - Add & Norm
  - Feed-Forward
  - Add & Norm
  ↓
[Decoder Blocks] × N
  - Masked Multi-Head Self-Attention
  - Add & Norm
  - Multi-Head Cross-Attention (on encoder output)
  - Add & Norm
  - Feed-Forward
  - Add & Norm
  ↓
Output Linear & Softmax
```

**Advantages**:
- Parallelizable (unlike RNNs)
- Captures long-range dependencies
- State-of-the-art for NLP and beyond

**Variants**:
- **Encoder-only**: BERT (good for classification, NLU)
- **Decoder-only**: GPT (good for generation)
- **Encoder-decoder**: T5, BART (good for seq2seq)

---

### Q8.2: What is attention and why is it important?

**Attention**: Mechanism that allows model to focus on relevant parts of input when making predictions.

**Intuition**: When reading this sentence, you pay more attention to certain words based on context.

**How it works**:
1. **Query (Q)**: What we're looking for
2. **Key (K)**: What each input offers
3. **Value (V)**: Actual content of each input
4. Calculate similarity between Q and all K
5. Use similarity as weights for V

**Formula**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) × V
```

**Types**:

**1. Self-Attention**:
- Attention within same sequence
- Each word attends to all words in sentence

**2. Cross-Attention**:
- Attention between two sequences
- Decoder attends to encoder output

**3. Masked Attention**:
- Prevent attending to future tokens (for autoregressive models like GPT)

**Why important**:
- **Long-range dependencies**: Can connect distant words
- **Interpretability**: Can visualize what model attends to
- **Parallelization**: Unlike RNNs, can process all tokens simultaneously
- **Flexibility**: Works for variable-length sequences

---

### Q8.3: Explain fine-tuning vs. prompting vs. RAG for LLMs.

**Fine-tuning**:
- Continue training LLM on your specific data
- Updates model weights
- **Pros**: Best performance, adapts model to domain
- **Cons**: Expensive, need labeled data, risk of catastrophic forgetting
- **Use when**: Have sufficient labeled data, need specialized model

**Prompting (In-Context Learning)**:
- Provide examples and instructions in prompt
- No weight updates
- **Types**: Zero-shot, few-shot, chain-of-thought
- **Pros**: Fast, no training, flexible
- **Cons**: Limited by context window, inconsistent, expensive tokens
- **Use when**: Simple tasks, no training data, need flexibility

**RAG (Retrieval-Augmented Generation)**:
- Retrieve relevant documents, add to prompt
- Model generates answer using retrieved context
- **Pros**: Leverages external knowledge, cites sources, updatable
- **Cons**: Requires retrieval system, adds latency
- **Use when**: Need up-to-date info, domain-specific knowledge, want citations

**Comparison**:

| Aspect | Fine-tuning | Prompting | RAG |
|--------|-------------|-----------|-----|
| Cost | High | Low | Medium |
| Data needed | Lots | None/few | Documents |
| Latency | Low | Low | Medium |
| Updatable | Hard | Easy | Easy |
| Accuracy | Highest | Lower | High |
| Use case | Specialized tasks | General tasks | QA, knowledge |

**Combining approaches**:
- Fine-tune + RAG: Best of both
- Prompt engineering + RAG: Common for production

---

### Q8.4: What are embeddings and how are they used?

**Embeddings**: Dense vector representations of discrete objects (words, sentences, images).

**Properties**:
- **Dense**: Most values non-zero (vs. one-hot sparse)
- **Low-dimensional**: 128-1024 dims (vs. vocabulary size)
- **Semantic**: Similar items have similar embeddings (cosine similarity)

**Types**:

**1. Word Embeddings**:
- Word2Vec, GloVe, FastText
- Each word → fixed vector
- Captures semantic similarity: king - man + woman ≈ queen

**2. Sentence/Document Embeddings**:
- BERT, Sentence-BERT, Universal Sentence Encoder
- Entire sentence → single vector

**3. Image Embeddings**:
- ResNet, CLIP
- Image → vector

**4. Multimodal Embeddings**:
- CLIP: Images and text in same space
- Can find similar images with text query

**Use Cases**:

**1. Similarity Search**:
- Find similar documents, products, images
- Use cosine similarity or Euclidean distance

**2. Clustering**:
- Group similar items (K-means on embeddings)

**3. Classification**:
- Use embeddings as features

**4. Recommendation**:
- Recommend items with similar embeddings

**5. Retrieval (RAG)**:
- Embed query and documents
- Retrieve most similar documents

**6. Feature Engineering**:
- Replace categorical features with embeddings

**Vector Databases**: Store and search embeddings efficiently (Pinecone, Weaviate, Qdrant, FAISS)

---

### Q8.5: What is prompt engineering?

**Prompt Engineering**: Designing prompts to get desired behavior from LLMs.

**Techniques**:

**1. Zero-Shot**:
- Task description only
- "Translate to French: Hello"

**2. Few-Shot**:
- Provide examples in prompt
- Example: "Positive: I love this! Negative: This is terrible. Positive: Amazing product."

**3. Chain-of-Thought (CoT)**:
- Ask model to explain reasoning step-by-step
- "Let's think step by step..."

**4. Role-Playing**:
- Assign role to model
- "You are an expert Python developer..."

**5. Template-Based**:
- Use consistent structure
- "Context: {context}\nQuestion: {question}\nAnswer:"

**6. Constraints**:
- Specify format, length, style
- "Answer in 3 bullet points"

**7. Iterative Refinement**:
- Start broad, refine based on output
- Add examples of failures

**Best Practices**:
- Be specific and clear
- Provide examples (few-shot)
- Specify output format
- Use delimiters (###, """) to structure
- Iterate and test variations
- Use system prompts for persistent behavior

**Challenges**:
- Prompt sensitivity (small changes → big differences)
- Context window limits
- Cost (long prompts = more tokens)
- No guarantees of consistency

**Tools**:
- LangChain: Prompt templates and chains
- Guidance: Structured prompting
- OpenAI Playground: Test prompts

---

### Q8.6: How does RAG (Retrieval-Augmented Generation) work?

**RAG**: Combine retrieval (search) with generation (LLM).

**How it works**:

1. **Index Phase** (offline):
   - Split documents into chunks (e.g., 500 tokens)
   - Generate embeddings for each chunk
   - Store in vector database

2. **Query Phase** (online):
   - User asks question
   - Embed question
   - Retrieve top K similar chunks (vector search)
   - Add retrieved chunks to prompt
   - LLM generates answer using retrieved context

**Architecture**:
```
User Query
  ↓
Query Embedding (Encoder)
  ↓
Vector Search (Similarity)
  ↓
Retrieved Documents (Top K)
  ↓
Prompt Template (Query + Context)
  ↓
LLM Generation
  ↓
Answer (with citations)
```

**Components**:

**1. Document Processing**:
- Chunking strategy (size, overlap)
- Metadata extraction

**2. Embedding Model**:
- Sentence-BERT, OpenAI embeddings, etc.
- Same model for docs and query

**3. Vector Store**:
- Pinecone, Weaviate, FAISS, Chroma
- Fast approximate nearest neighbor search

**4. Retrieval Strategy**:
- Dense retrieval (embeddings)
- Hybrid: Dense + sparse (BM25)
- Reranking: Retrieve more, then rerank

**5. Generation**:
- LLM (GPT, Claude, Llama)
- Prompt template to combine query + context

**Advantages**:
- Up-to-date information (update docs, not model)
- Domain-specific knowledge
- Citations/sources
- Lower hallucination

**Challenges**:
- Quality depends on retrieval
- Context window limits (can only fit so many docs)
- Latency (retrieval + generation)
- Chunk size optimization

**Improvements**:
- Hybrid search (dense + sparse)
- Query expansion/rewriting
- Reranking retrieved docs
- Hierarchical retrieval (summaries → full docs)

---

## Bonus: Tricky Conceptual Questions

### B1: Why does XGBoost generally outperform Random Forest?

**Answer**:
- **Boosting vs. Bagging**: XGBoost (boosting) sequentially corrects errors; Random Forest (bagging) averages independent trees
- **Bias-Variance**: XGBoost reduces bias; RF reduces variance
- **Weak learners**: XGBoost uses shallow trees (reduces variance), RF uses deep trees
- **Regularization**: XGBoost has built-in L1/L2, RF doesn't
- **Optimization**: XGBoost uses second-order gradients (Newton method), more precise

But Random Forest can be better when:
- Data is very noisy (RF more robust)
- Need interpretability
- Less hyperparameter tuning required

---

### B2: Can you have high bias and high variance simultaneously?

**Answer**:
Generally no in typical scenarios, but possible in some cases:

**Typical**: Bias-variance tradeoff (one ↑, other ↓)

**Possible exception**:
- Very poor model choice that's both too simple (high bias) AND unstable (high variance)
- Example: Poorly initialized/trained neural network
- But this is pathological, not common

**Better framing**: Think in terms of total error
- Simple model: High bias, low variance
- Complex model: Low bias, high variance
- Right complexity: Balanced

---

### B3: When would linear regression outperform a neural network?

**Linear regression better when**:
1. **Small dataset**: NN needs lots of data, would overfit
2. **Interpretability required**: Linear model is explainable
3. **Truly linear relationship**: No benefit from NN complexity
4. **Low noise, good features**: If features already engineered well
5. **Limited compute**: LR is much faster to train
6. **Need confidence intervals**: LR has statistical guarantees

**Example**: Predicting house price with 100 samples and 10 features, where relationship is approximately linear

---

### B4: Why do we use mini-batch instead of full batch gradient descent?

**Advantages of mini-batch**:
1. **Faster updates**: Update weights more frequently
2. **Regularization effect**: Noise in gradients helps escape local minima
3. **Memory efficient**: Don't need entire dataset in memory
4. **Parallelization**: Modern hardware optimized for batch operations
5. **Better generalization**: Noise prevents overfitting

**Batch sizes**:
- Small (16-32): More noise, better generalization, slower
- Large (256-1024): Less noise, faster per epoch, may overfit
- Sweet spot: 32-128 for most applications

---

### B5: What's the difference between generative and discriminative models?

**Discriminative Models**:
- Learn P(Y|X) (conditional probability)
- Learn decision boundary directly
- Examples: Logistic Regression, SVM, Neural Networks
- **Goal**: Classify/predict Y given X

**Generative Models**:
- Learn P(X|Y) and P(Y), can compute P(Y|X) via Bayes
- Model data distribution
- Examples: Naive Bayes, GANs, VAEs, Diffusion Models
- **Goal**: Generate new samples, understand data distribution

| Aspect | Discriminative | Generative |
|--------|----------------|------------|
| What it learns | P(Y\|X) | P(X\|Y), P(Y) |
| Use | Classification | Generation + Classification |
| Data needed | Less | More |
| Performance (classification) | Usually better | Usually worse |
| Can generate samples | No | Yes |

---

## How to Prepare

1. **Understand, don't memorize**: Focus on concepts and intuition
2. **Practice explaining**: Teach concepts to someone else
3. **Draw diagrams**: Visualize architectures, processes
4. **Relate to projects**: Connect theory to your practical work
5. **Anticipate follow-ups**: Every answer can lead to deeper questions
6. **Stay current**: Read recent papers, blog posts (especially for LLMs)

## During Interview

- **Clarify question**: Make sure you understand what's being asked
- **Structure answer**: Start with definition, then details, then examples
- **Show reasoning**: Explain your thought process
- **Admit gaps**: If you don't know, say so, then reason through it
- **Connect to experience**: Reference your projects when relevant
- **Ask questions**: Shows curiosity and engagement

---

## Day 3 Self-Assessment: Theory Gaps Identified (2025-10-30)

### **Assessment Process**:
Reviewed all theory questions in this document to identify:
- What I know well (high-level understanding)
- Where knowledge is incomplete or rusty
- What needs priority refresh for interviews

---

### **Overall Pattern Observed** 🟡:

**Strengths** ✅:
- Have **high-level understanding** of most topics
- Can recognize and discuss concepts
- Strong on fundamentals (bias-variance, overfitting, basic ML algorithms)
- Solid on production ML (from industry experience)

**Gaps** 🟡:
- **Implementation details**: Know main idea but not all specifics
- **Variants & advanced versions**: Know basic version, not alternatives
- **Best practices**: Know the technique, not the typical hyperparameters
- **Pros/cons**: Can name 1-2 points, not comprehensive list
- **When to use**: Know main use case, not all scenarios

---

### **Specific Gaps Identified**:

#### **Optimization & Training** (HIGH PRIORITY) ← **Refreshed on Day 3** ✅:
- ~~Adam optimizer details~~ ✅ **Now know**: Combines momentum + RMSprop, bias correction
- ~~Momentum mechanism~~ ✅ **Now know**: Exponentially weighted averages, β=0.9
- ~~RMSprop~~ ✅ **Now know**: Adaptive per-parameter learning rates
- 🟡 Learning rate schedules (step decay, cosine annealing) - still need review
- 🟡 Batch normalization placement/details - vague

#### **Evaluation Metrics** (HIGH PRIORITY) ← **Partially Refreshed** ✅:
- ~~AUC-ROC details~~ ✅ **Now know**: TPR vs FPR, when to use vs Precision-Recall
- 🟡 Regression metrics: Only know MSE, need MAE, RMSE, R²
- 🟡 When to use which metric - partial knowledge

#### **Modern NLP/Transformers** (HIGH PRIORITY) ← **Refreshed on Day 3** ✅:
- ~~Attention mechanism~~ ✅ **Now know**: Q-K-V, similarity → softmax → weighted sum
- ~~Transformers basics~~ ✅ **Now know**: Self-attention, BERT vs GPT, parallel processing
- ✅ **Have production experience**: Used BERT and LLMs in production
- 🟡 Transformer variants (BERT, GPT, T5) - high level only

#### **Classical ML** (MEDIUM PRIORITY):
- 🟡 Boosting (AdaBoost, Gradient Boosting, XGBoost) - not familiar with details
- 🟡 Elastic Net - didn't know it's L1 + L2 combined
- 🟡 Kernel trick (SVMs) - forgot details
- 🟡 Parametric vs non-parametric models - not clear on definition

#### **Deep Learning Details** (MEDIUM PRIORITY):
- 🟡 Dropout rates: Know dropout, but not that p=0.5 is typical
- 🟡 Activation functions: Know ReLU, sigmoid, tanh, but not ELU, Leaky ReLU variants
- 🟡 Batch normalization: Know concept, not placement details
- 🟡 Weight initialization strategies - vague

#### **Advanced Topics** (LOW PRIORITY):
- 🔴 Reinforcement Learning: High-level idea only, don't know algorithms (Q-learning, policy gradients)
- 🟡 GANs, VAEs: Know they exist, not architecture details
- 🟡 Model compression: Quantization, pruning, distillation - aware but not detailed

---

### **Day 3 Progress** ✅:

**Watched Videos & Created Reference**:
1. ✅ Optimizers (DeepLearning.AI): Momentum, RMSprop, Adam
2. ✅ AUC-ROC (StatQuest): TPR vs FPR, when to use
3. ✅ Attention (StatQuest): Core mechanism, transformers basics
4. ✅ Created `references/Day3-Quick-Reference.md` with consolidated notes

**Questions Clarified**:
- ✅ Bias correction in Adam (why needed, how it works)
- ✅ Query-Key-Value calculation in attention
- ✅ Parallel processing: BERT (full parallel) vs GPT (sequential generation, parallel attention)

**Knowledge Upgraded**:
- 🔴 → ✅ Adam optimizer: Now interview-ready
- 🔴 → ✅ AUC-ROC: Now interview-ready
- 🟡 → ✅ Attention/Transformers: Now interview-ready

---

### **Remaining Priority Tasks** (Week 1-2):

#### **High Priority** (Essential for interviews):
1. 🟡 **Regularization details**: L1 vs L2 specifics, when to use (30 min)
2. 🟡 **Regression metrics**: MAE, RMSE, R² (20 min)
3. 🟡 **Learning rate schedules**: Quick overview (15 min)
4. 🟡 **Batch normalization**: Where to place, how it works (20 min)

#### **Medium Priority** (Nice to have):
5. 🟡 **Boosting**: AdaBoost, XGBoost basics (30 min)
6. 🟡 **Kernel trick**: SVMs with non-linear boundaries (15 min)
7. 🟡 **Activation functions**: ELU, Leaky ReLU (10 min)
8. 🟡 **Parametric vs non-parametric**: Definition and examples (10 min)

#### **Can Skip** (Unless role-specific):
- ❌ Reinforcement learning algorithms (not common for ML Engineer roles)
- ❌ GAN architecture details (unless generative AI role)
- ❌ Advanced model compression (unless edge/mobile ML role)

**Total time for remaining high-priority items**: ~2-3 hours (spread over Week 1-2)

---

### **Strategy Going Forward**:

**Week 1** (Days 4-7):
- Focus on algorithm implementations (K-NN, K-Means)
- Start portfolio projects (image classification)
- Quick theory lookups as needed (15-20 min per topic)

**Week 2** (Days 8-14):
- Continue projects
- Dedicated theory session: Remaining high-priority topics (2 hours)
- Add medium-priority topics opportunistically

**Interview Prep** (Week 3+):
- Practice explaining refreshed topics out loud
- Connect theory to production ML projects
- Use `Day3-Quick-Reference.md` for review

---

### **Key Insight**:

My rust is **superficial** (confirmed by Day 1-3 results):
- ✅ Day 1-2: Linear/Logistic Regression implementation - 4x speed improvement in one day
- ✅ Day 3: Theory refresh - Quickly regained detailed understanding with targeted videos

**This means**:
- I don't need to "learn" ML - I need to **refresh** specific details
- Targeted 15-30 min sessions are highly effective
- Can fill remaining gaps efficiently over 1-2 weeks
- Priority should remain on hands-on practice (algorithms + projects)

---

**Last Updated**: 2025-10-30 (after Day 3 self-assessment)
**Next**: Complete K-NN implementation (Day 3 Session 2)

---

**Good luck with your interview prep!** These questions cover most ML Engineer interview topics. Review regularly and practice explaining out loud.
