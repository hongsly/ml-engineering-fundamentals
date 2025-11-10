# Day 13 Quick Reference: Regression Diagnostics & Covariance

**Week 2, Day 6** | Focus: Linear regression assumption testing, covariance vs correlation

---

## **1. Five Linear Regression Assumptions**

### **Overview**:
1. **Linearity**: Mean of Y has linear relationship with X
2. **Independence**: Errors of observations are independent
3. **Homoscedasticity**: Errors have constant variance
4. **Normality**: Residuals follow normal distribution
5. **Non-collinearity**: Predictors are not linearly related to each other

---

## **2. Linearity**

### **Assumption**:
The mean of Y has a linear relationship with X.

**Note**: Can still use linear regression for non-linear relationships by adding polynomial terms:
- y = c₁·x + c₂·x² (still linear in **parameters**)

### **If Violated**:
Poor model fit, biased predictions

### **Testing**:
**Residual plot against predicted values**:
- ✅ Good: Random scatter, no pattern
- ❌ Bad: Parabolic/curved shape suggests non-linearity

---

## **3. Independence**

### **Assumption**:
The error of one observation is independent from all other observations.

### **If Violated**:
- **Underestimated standard errors** → p-values too small
- Conclude significance when shouldn't (Type I error for coefficients)
- **Confidence intervals and hypothesis tests become unreliable**

### **Testing**:

**Method 1: Durbin-Watson (DW) Test**
```
DW statistic range: 0 to 4
- 2 = perfectly independent
- 1.5-2.5 = usually OK to conclude independence
- <1.5 = positive autocorrelation
- >2.5 = negative autocorrelation
```

**Example**: DW = 0.9 → positive autocorrelation → likely time series data

**Method 2: Residual plot against time/observation order**
- ✅ Good: Random scatter, no pattern
- ❌ Bad: Trends, cycles, clustering

---

## **4. Homoscedasticity**

### **Assumption**:
The errors of different observations have about the same scatter (constant variance).

### **If Violated**:
- **Underestimated standard errors** → unreliable confidence intervals
- Type I error (false positives for coefficients)

### **Testing**:

**Method 1: Residual plot against predicted values**
- ✅ Good: Random scatter, constant width
- ❌ Bad: Cone/fan shape (variance increases with predicted value)

**Method 2: Breusch-Pagan (BP) Test**
```
Null hypothesis: Homoscedasticity (constant variance)
- p < 0.05 → Reject null → Heteroscedasticity present
- p ≥ 0.05 → Fail to reject → Homoscedasticity holds

⚠️ Can be over-sensitive for large samples
```

**Example**: BP test p-value = 0.03 → Reject null → Heteroscedasticity violation

---

## **5. Normality**

### **Assumption**:
The residuals should follow normal distribution.

### **If Violated**:
- Invalid p-values and confidence intervals **for small samples** (n<30)
- Usually OK for large samples (n≥30) due to Central Limit Theorem

### **Testing**:

**Method 1: Q-Q Plot**
- Plot residuals against theoretical normal distribution quantiles
- ✅ Good: Points follow diagonal line
- ❌ Bad: Systematic deviations from line

**Method 2: Shapiro-Wilk Test**
```
Null hypothesis: Residuals are normally distributed
- p < 0.05 → Reject null → Residuals not normal
- p ≥ 0.05 → Fail to reject → Normality holds

⚠️ Can be over-sensitive for large samples
```

**Example**: Shapiro-Wilk p-value = 0.25 → Fail to reject → Normality holds ✅

---

## **6. Non-collinearity (Multicollinearity)**

### **Assumption**:
The independent variables (predictors) should not have linear relationships among themselves.

### **If Violated**:
- **Unstable coefficients**: Small data changes → large coefficient changes
- **Inflated standard errors** → wider confidence intervals → harder to detect significance (Type II error, false negatives)

### **Testing**:

**VIF (Variance Inflation Factor)**
```
Formula: VIF_i = 1 / (1 - R²_i)

Where R²_i = R² from regressing X_i on all other predictors

Interpretation:
- VIF = 1: Perfect non-collinearity (no correlation with other predictors)
- VIF < 5: Good
- VIF < 10: OK
- VIF ≥ 10: Problematic multicollinearity

Example: VIF₁=3.2 (✅), VIF₂=8.5 (⚠️), VIF₃=15.7 (❌)
```

### **Solutions**:
1. **Remove predictor variable** with high VIF
2. **PCA** (Principal Component Analysis) to create uncorrelated features
3. **Regularization** (Ridge/Lasso) to stabilize coefficients

---

## **7. Covariance**

### **Formula**:
```
Cov(X,Y) = Σ[(x_i - x̄)(y_i - ȳ)] / (n-1)

(Sample covariance uses n-1 for unbiased estimator)
```

Note: Cov(X, X) is Var(X)

### **Interpretation**:
- **Positive covariance**: X and Y have positive linear relationship
- **Negative covariance**: X and Y have negative linear relationship
- **Zero covariance**: No linear relationship

### **Properties**:
- **Scale-dependent**: Units = (X units) × (Y units)
- **Unbounded**: Range from -∞ to +∞
- **Hard to interpret** by itself without knowing variable scales

### **Use Cases**:
- Stepping stone for other calculations (correlation, PCA)
- Portfolio variance calculation (actual magnitude matters)

---

## **8. Correlation**

### **Formula**:
```
ρ(X,Y) = Cov(X,Y) / (σ_X × σ_Y)

(Correlation = normalized covariance)
```

Intuitively: how much variance is explained by covariance

### **Interpretation**:
- **Scale-independent**: Unitless
- **Bounded**: Range from -1 to 1
  - ρ = 1: Perfect positive linear relationship (all points on line with positive slope)
  - ρ = -1: Perfect negative linear relationship (all points on line with negative slope)
  - ρ = 0: No linear relationship
  - |ρ| closer to 1 → stronger linear relationship (points are closer to the line)

### **Properties**:
- **Invariant to scale changes**: Scaling X or Y doesn't change correlation
- **Only captures linear relationships**: Can be 0 even if strong non-linear relationship exists

### **Correlation & P-value**:
- **Low p-value**: High confidence in the measured correlation
- **High p-value**: Low confidence (sample too small, relationship unclear)

**Critical insight**:
- More samples → smaller p-value (more confidence)
- **Small p-value + weak correlation = confident about a weak relationship**
  - Example: ρ = 0.3, p = 0.001 → Statistically significant but poor predictor (only 9% variance explained, r²=0.09)
  - **Statistical significance ≠ practical significance** ⚠️

---

## **9. Covariance vs Correlation**

| Aspect | Covariance | Correlation |
|--------|-----------|-------------|
| **Units** | Unit-dependent (X units × Y units) | Unitless (normalized) |
| **Bounds** | Unbounded (-∞ to +∞) | Bounded [-1, 1] |
| **Interpretation** | Difficult (depends on scale) | Easy (strength of linear relationship) |
| **Scale invariance** | Changes with scale | Invariant to scale |
| **Formula** | Cov(X,Y) = E[(X-μₓ)(Y-μᵧ)] | ρ = Cov(X,Y) / (σₓ × σᵧ) |

### **When to use**:
- **Covariance**: When actual magnitude matters (e.g., portfolio variance)
- **Correlation**: When comparing relationships across different scales

### **Example**:
```
Dataset A: X=[1,2,3], Y=[2,4,6]
Dataset B: X=[100,200,300], Y=[2,4,6]

Covariance: B has much larger covariance (X has 100× larger scale)
Correlation: Both have same correlation (scale-invariant)
```

---

## **10. Interview Q&A**

### **Q: How do you check linear regression assumptions?**
**A**: "I check 5 assumptions systematically:

1. **Linearity**: Residual plot vs predicted values - look for random scatter
2. **Independence**: Durbin-Watson test (looking for values 1.5-2.5) or residual plot vs time order
3. **Homoscedasticity**: Breusch-Pagan test (p>0.05 for constant variance) or residual plot for cone shape
4. **Normality**: Shapiro-Wilk test (p>0.05 for normal residuals) or Q-Q plot
5. **Non-collinearity**: VIF test (VIF<10 preferred, <5 ideal)

For quick visual checks, I plot residuals vs fitted values and Q-Q plot. For formal testing, I use BP, SW, and VIF tests."

---

### **Q: You run Durbin-Watson test and get DW=0.8. What's the problem?**
**A**: "DW=0.8 indicates strong positive autocorrelation in residuals, violating the independence assumption. This commonly happens with time series data where adjacent observations are correlated. The issue is that standard errors will be underestimated, leading to overly narrow confidence intervals and p-values that are too small - we might conclude coefficients are significant when they aren't. Solutions include using time series models (ARIMA), adding time-based features, or using robust standard errors."

---

### **Q: What's the difference between covariance and correlation?**
**A**: "Covariance measures the direction and magnitude of linear relationship between two variables, but it's scale-dependent with unbounded range. Correlation is the normalized version - dividing covariance by the product of standard deviations - making it scale-independent and bounded between -1 and 1.

For example, if I scale X by 100×, covariance increases 100×, but correlation stays the same. In practice, I use covariance when actual magnitude matters (like portfolio variance calculations) and correlation when comparing relationships across different scales or presenting to non-technical stakeholders since it's easier to interpret."

---

### **Q: VIF for predictor X₁ is 15. What should you do?**
**A**: "VIF=15 indicates severe multicollinearity - X₁ is highly correlated with other predictors. This causes unstable coefficients and inflated standard errors. I have three options:

1. **Remove X₁** if it's redundant or less important
2. **PCA** to create uncorrelated features (loses interpretability)
3. **Ridge regularization** to stabilize coefficients while keeping all features

I'd first check which other predictors X₁ is correlated with, then decide based on domain knowledge whether to remove, combine, or regularize."

---

### **Q: A model has correlation=0.3 with p<0.001. Is this a good predictor?**
**A**: "No. While p<0.001 means we're highly confident the correlation exists, r=0.3 means only 9% of variance is explained (r²=0.09). This is a case of statistical significance without practical significance. The low p-value comes from having a large sample size, which increases confidence but doesn't change the weak relationship. For a good predictor, I'd want r>0.7 (50%+ variance explained) or evaluate using domain-specific thresholds for acceptable prediction quality."

---

## **11. Common Pitfalls**

❌ **"Residuals have some large values, so normality is violated"**
✅ Few outliers are OK. Check Shapiro-Wilk p-value or Q-Q plot. Large samples (n>30) are robust to non-normality.

❌ **"VIF > 10 means I must immediately drop that variable"**
✅ Consider domain context first. Ridge regularization can handle multicollinearity while keeping all features. VIF>10 is a warning, not absolute rule.

❌ **"High correlation means strong causation"**
✅ Correlation measures linear association, not causation. Can have spurious correlations (ice cream sales vs drowning).

❌ **"Covariance is more useful than correlation"**
✅ Correlation is usually preferred for interpretation. Covariance is mainly a stepping stone calculation.

❌ **"p-value tells me how strong the relationship is"**
✅ p-value indicates confidence, not strength. r (correlation coefficient) or R² (variance explained) measures strength.

❌ **"Durbin-Watson = 2.8 is close to 2, so independence holds"**
✅ 2.8 > 2.5 indicates negative autocorrelation. While weaker than DW=0.8, it still violates independence.

---

## **12. Key Formulas**

### **Regression Diagnostics**:
```
Durbin-Watson statistic: Range [0, 4], target ~2

VIF_i = 1 / (1 - R²_i)

Breusch-Pagan, Shapiro-Wilk: p < 0.05 → violation
```

### **Covariance & Correlation**:
```
Cov(X,Y) = Σ[(x_i - x̄)(y_i - ȳ)] / (n-1)

ρ(X,Y) = Cov(X,Y) / (σ_X × σ_Y)

Coefficient of determination: R² = ρ²
```

---

## **13. Study Tips**

- **Visual checks first**: Residual plots and Q-Q plots catch most violations quickly
- **Formal tests second**: Use BP, SW, VIF to confirm statistical evidence
- **Large sample robustness**: Many violations (normality, heteroscedasticity) matter less with n>30
- **Context matters**: Statistical significance (p-value) ≠ practical significance (effect size)
- **Multicollinearity**: Only affects inference (standard errors), not predictions
- **Independence**: Most critical assumption - time series data almost always violates this

---

## **Day 13 Study Outcome**

**Topics Mastered**:
- ✅ 4 regression diagnostic tests (DW, BP, SW, VIF)
- ✅ 5 linear regression assumptions
- ✅ Covariance vs correlation (formulas, interpretation, use cases)
- ✅ Statistical vs practical significance

**Knowledge Check**: 99.5% (A+)

**Next**: Day 14 - MLE derivations + Hypothesis testing core
