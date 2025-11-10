# Week 2: Statistics & Probability Topic Coverage Check

**Purpose**: Comprehensive assessment of statistics/probability knowledge to identify gaps and plan study time

**Date Started**: 2025-11-09 (Day 6, Week 2)

**Current Baseline**: 65% (from Day 6 Gap Analysis Q1-Q63)

**Target**: 85-90% interview readiness

---

## Instructions

For each topic, mark your current understanding:
- ✅ **Know**: Can explain confidently in an interview, can derive/apply
- 🟡 **Unsure**: Recognize the concept but need review, partial understanding
- ❌ **Dunno**: Don't know or very weak understanding

---

## Section 1: Fundamentals (5 topics)

| # | Topic | Status | Notes |
|---|-------|--------|-------|
| 1.1 | Probability vs Statistics (difference, when to use each) | [unsure] | |
| 1.2 | PDF vs CDF vs PMF (definitions, relationships) | [unsure] | |
| 1.3 | Common distributions (Normal, Exponential, Binomial, Poisson) | [unsure] | Forgot Exponential and Poisson |
| 1.4 | Expected value and variance (definition, properties) | [unsure] | know definition, may forgot properties |
| 1.5 | Law of Large Numbers vs Central Limit Theorem | [unsure] | forgot central limit theorem |

**Section 1 Summary**: ___/5 Know, ___/5 Unsure, ___/5 Dunno

---

## Section 2: Descriptive Statistics (6 topics)

| # | Topic | Status | Notes |
|---|-------|--------|-------|
| 2.1 | Mean, median, mode (when to use each, robustness) | [unsure] | forgot mode's properies |
| 2.2 | **Covariance vs Correlation** (units, bounds, interpretation) | [know] | **Gap Q3: 0%** → Day 13: 100% ✅ |
| 2.3 | **Skewness** (definition, left vs right, impact on mean/median) | [know] | **Gap Q4: 60%** |
| 2.4 | Kurtosis (heavy-tailed vs light-tailed distributions) | [unsure] | |
| 2.5 | Percentiles and quantiles (Q1, Q3, IQR) | [unsure] | forgot what is IQR |
| 2.6 | Standard deviation vs standard error | [know] | |

**Section 2 Summary**: ___/6 Know, ___/6 Unsure, ___/6 Dunno

---

## Section 3: Hypothesis Testing (10 topics)

| # | Topic | Status | Notes |
|---|-------|--------|-------|
| 3.1 | **T-test vs Z-test** (when to use n<30 vs n≥30, assumptions) | [unsure] | **Day 6 focus** |
| 3.2 | P-value (definition, interpretation, common mistakes) | [unsure] | rusty |
| 3.3 | Confidence intervals (95% CI calculation, interpretation) | [unsure] | rusty |
| 3.4 | Type I error (α) vs Type II error (β) | [know] | |
| 3.5 | Statistical power (1-β, how to increase) | [unsure] | rusty |
| 3.6 | One-tailed vs two-tailed tests (when to use each) | [unsure] | rusty |
| 3.7 | **Normal distribution calculations** (P(X>2) for N(0,1), Z-scores) | [unsure] | **Gap Q27: 0%** |
| 3.8 | Chi-square test (goodness of fit, independence) | [dunno] | |
| 3.9 | **A/B Testing Design & Pitfalls** (metric choice, segmentation, novelty effect, full loop) | [unsure] | **MUST-KNOW for MLE** |
| 3.10 | **Multiple Comparisons Problem** (Bonferroni, False Discovery Rate, family-wise error) | [unsure] | Nice-to-have for senior/research roles |

**Section 3 Summary**: ___/10 Know, ___/10 Unsure, ___/10 Dunno

---

## Section 4: Regression Assumptions & Diagnostics (7 topics)

| # | Topic | Status | Notes |
|---|-------|--------|-------|
| 4.1 | **5 Linear Regression Assumptions** (linearity, independence, homoscedasticity, normality, no multicollinearity) | [know] | **Day 6 focus** → Day 13: Mastered ✅ |
| 4.2 | Residual plots (how to interpret, what patterns indicate) | [know] | Day 13: Mastered (parabolic, cone, random scatter) ✅ |
| 4.3 | Durbin-Watson test (autocorrelation in residuals) | [know] | Day 13: 100% (DW 0-4, 2=independent, 1.5-2.5 OK) ✅ |
| 4.4 | Breusch-Pagan test (heteroscedasticity detection) | [know] | Day 13: 100% (p<0.05 → violation) ✅ |
| 4.5 | Shapiro-Wilk test (normality of residuals) | [know] | Day 13: 100% (p≥0.05 → normal) ✅ |
| 4.6 | VIF (Variance Inflation Factor for multicollinearity) | [know] | Day 13: 95% (VIF<5 good, <10 OK, ≥10 problem) ✅ |
| 4.7 | **Linear regression with input noise** (attenuation bias, how noise affects coefficients) | [unsure] | **Gap Q60: 50%** |

**Section 4 Summary**: ___/7 Know, ___/7 Unsure, ___/7 Dunno

---

## Section 5: Probability Theory (8 topics)

| # | Topic | Status | Notes |
|---|-------|--------|-------|
| 5.1 | Bayes' theorem (P(A\|B) formula, application) | [know] | Gap Q26: 95% (strong) |
| 5.2 | Conditional probability vs independence (definitions, examples) | [know] | |
| 5.3 | **Gambler's ruin problem** (probability formula, random walk) | [unsure] | **Gap Q31: 20%** |
| 5.4 | **Fair coin from unfair coin** (von Neumann trick) | [know] | **Gap area: 30%** |
| 5.5 | Combinatorics (permutations, combinations, when to use) | [know] | |
| 5.6 | Joint, marginal, and conditional distributions | [unsure] | rusty on marginal |
| 5.7 | **Bayesian vs Frequentist Statistics** (P(θ\|D) vs P(D\|θ), philosophical differences) | [know] | **MUST-KNOW for Applied Scientist** |
| 5.8 | **Markov Chains** (basic properties, steady-state distribution, transition matrix) | [unsure] | Relevant for rec systems, MCMC sampling |

**Section 5 Summary**: ___/8 Know, ___/8 Unsure, ___/8 Dunno

---

## Section 6: ML-Specific Statistics (7 topics)

| # | Topic | Status | Notes |
|---|-------|--------|-------|
| 6.1 | **MLE for common distributions** (exponential λ̂=1/mean, Gaussian μ̂, σ̂²) | [dunno] | **Gap Q29: 0%** |
| 6.2 | **Sigmoid for regression** (why wrong for unbounded targets, bounded [0,1] output) | [know] | **Gap Q19, Q43: 0%** |
| 6.3 | **Logistic regression on separable data** (weights diverge without regularization) | [unsure] | **Gap Q43: 0%** |
| 6.4 | **Dependence vs Correlation** (dependence ⊃ correlation, Y=X² example) | [unsure] | **Gap Q63: 50%** |
| 6.5 | Bias-variance tradeoff (mathematical formulation, decomposition) | [unsure] | |
| 6.6 | Bootstrap and resampling (when to use, confidence intervals) | [unsure] | |
| 6.7 | **Regularization as Bayesian Prior** (L2 ≈ Gaussian prior, L1 ≈ Laplace prior) | [dunno] | Conceptual connection, nice-to-have |

**Section 6 Summary**: ___/7 Know, ___/7 Unsure, ___/7 Dunno

---

## 📊 Overall Summary Scorecard

| Section | Know | Unsure | Dunno | Priority |
|---------|------|--------|-------|----------|
| 1. Fundamentals | ___/5 | ___/5 | ___/5 | [ ] High [ ] Med [ ] Low |
| 2. Descriptive Statistics | ___/6 | ___/6 | ___/6 | [ ] High [ ] Med [ ] Low |
| 3. Hypothesis Testing | ___/10 | ___/10 | ___/10 | [ ] High [ ] Med [ ] Low |
| 4. Regression Assumptions | ___/7 | ___/7 | ___/7 | [ ] High [ ] Med [ ] Low |
| 5. Probability Theory | ___/8 | ___/8 | ___/8 | [ ] High [ ] Med [ ] Low |
| 6. ML-Specific Statistics | ___/7 | ___/7 | ___/7 | [ ] High [ ] Med [ ] Low |
| **TOTAL** | **___/43** | **___/43** | **___/43** | |

**Overall Readiness**: ___% (Know / 43 × 100)

---

## 🎯 Adaptive Time Allocation

**Based on Dunno count (out of 43 topics)**:

- **0-9 Dunno topics** (79%+ Know): ✅ **Stay with Day 6-7 plan (2-3 hours)**
  - Light gaps, original timeline sufficient
  - Focus: Quick review + practice exercises

- **10-15 Dunno topics** (65-77% Know): ⚠️ **Extend to Day 6-8 (4-5 hours)**
  - Moderate gaps, need additional study day
  - Focus: Targeted study on Dunno topics + practice

- **16+ Dunno topics** (<63% Know): 🔴 **Extend to Day 6-9 (6-8 hours)**
  - Major gaps, need comprehensive coverage
  - Focus: Deep study + multiple practice sessions

---

## 📝 Priority Gaps (From Day 6 Gap Analysis)

**Critical (0-30%)** - MUST address:
- [ ] Covariance vs correlation (Q3: 0%) → Section 2.2
- [ ] Normal distribution calculations (Q27: 0%) → Section 3.7
- [ ] MLE for exponential distribution (Q29: 0%) → Section 6.1
- [ ] Sigmoid for regression (Q19, Q43: 0%) → Section 6.2
- [ ] Logistic regression on separable data (Q43: 0%) → Section 6.3
- [ ] Gambler's ruin probability (Q31: 20%) → Section 5.3

**Moderate (40-70%)** - Should address:
- [ ] Skewness (Q4: 60%) → Section 2.3
- [ ] Linear regression with input noise (Q60: 50%) → Section 4.7
- [ ] Dependence vs correlation (Q63: 50%) → Section 6.4

**High-Impact Additions** (5 new topics for senior/applied scientist roles):
- [ ] **A/B Testing Design & Pitfalls** (3.9) - MUST-KNOW for any MLE role
- [ ] **Bayesian vs Frequentist Statistics** (5.7) - MUST-KNOW for applied scientist
- [ ] **Multiple Comparisons Problem** (3.10) - Nice-to-have for senior/research roles
- [ ] **Markov Chains** (5.8) - Relevant for rec systems, MCMC
- [ ] **Regularization as Bayesian Prior** (6.7) - Nice conceptual connection

---

## 📚 Study Resources (To be filled after assessment)

**For Section ___** (weakest section):
- Resource 1:
- Resource 2:

**For Critical Gaps**:
- Covariance vs correlation:
- MLE derivations:
- Logistic regression edge cases:

---

## 🔄 Post-Study Assessment

**After completing study sessions, update this section:**

**Date Completed**: ___________

**Final Readiness**: ___% (up from 65%)

**Sections Mastered**:
- Section ___: ___% → ___%
- Section ___: ___% → ___%

**Remaining Weak Topics** (<80%):
1.
2.
3.

**Knowledge Check Score**: ___% (Day ___ check)

**Next Steps**:
- [ ] Practice exercises (statsmodels regression diagnostics)
- [ ] Create Statistics Quick Reference cheat sheet
- [ ] Review weak topics before Week 3

---

## Notes

- Cross-reference with `gap_analysis/Day6-Gap-Analysis.md` for detailed Q&A
- This complements Week 2 LLM Systems check (83% achieved)
- Target overall readiness after stats: 78% → 82-85%

---

## 📝 Day 13 Progress Update (2025-11-09)

**Topics Mastered** (Dunno/Unsure → Know):
- ✅ 2.2: Covariance vs Correlation (100%)
- ✅ 4.1: 5 Linear Regression Assumptions (100%)
- ✅ 4.2: Residual plots interpretation (100%)
- ✅ 4.3: Durbin-Watson test (100%)
- ✅ 4.4: Breusch-Pagan test (100%)
- ✅ 4.5: Shapiro-Wilk test (100%)
- ✅ 4.6: VIF test (95%)

**Progress**:
- Know: 8 → 13 (+5 topics)
- Dunno: 8 → 3 (-5 topics: all Section 4 diagnostics moved to Know)
- Unsure: 25 → 25 (no change, but Section 4.1-4.2 upgraded)

**Updated Readiness**: ~30% (13 Know / 43 topics) - up from 18.6%

**Knowledge Check**: 99.5% (A+) - Perfect diagnostics understanding

**Section 4 Status**: 6/7 Know (86% mastery) - Strongest section! ✅
- Only 4.7 (input noise/attenuation bias) remains Unsure

**Next Session (Day 14)**: MLE derivations, Chi-square, T-test vs Z-test, Normal distribution calculations
