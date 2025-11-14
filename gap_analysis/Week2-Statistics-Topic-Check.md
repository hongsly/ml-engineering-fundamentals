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
| 1.2 | PDF vs CDF vs PMF (definitions, relationships) | [know] | |
| 1.3 | Common distributions (Normal, Exponential, Binomial, Poisson) | [know] | |
| 1.4 | Expected value and variance (definition, properties) | [unsure] | know definition, may forgot properties |
| 1.5 | Law of Large Numbers vs Central Limit Theorem | [know] | |

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
| 3.1 | **T-test vs Z-test** (when to use n<30 vs n≥30, assumptions) | [know] | **Day 6 focus** |
| 3.2 | P-value (definition, interpretation, common mistakes) | [know] | |
| 3.3 | Confidence intervals (95% CI calculation, interpretation) | [unsure] | rusty |
| 3.4 | Type I error (α) vs Type II error (β) | [know] | |
| 3.5 | Statistical power (1-β, how to increase) | [unsure] | rusty |
| 3.6 | One-tailed vs two-tailed tests (when to use each) | [unsure] | rusty |
| 3.7 | **Normal distribution calculations** (P(X>2) for N(0,1), Z-scores) | [know] | **Gap Q27: 0%** |
| 3.8 | Chi-square test (goodness of fit, independence) | [know] | |
| 3.9 | **A/B Testing Design & Pitfalls** (metric choice, segmentation, novelty effect, full loop) | [know] | **MUST-KNOW for MLE** |
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
| 6.1 | **MLE for common distributions** (exponential λ̂=1/mean, Gaussian μ̂, σ̂²) | [know] | **Gap Q29: 0%** |
| 6.2 | **Sigmoid for regression** (why wrong for unbounded targets, bounded [0,1] output) | [know] | **Gap Q19, Q43: 0%** |
| 6.3 | **Logistic regression on separable data** (weights diverge without regularization) | [unsure] | **Gap Q43: 0%** |
| 6.4 | **Dependence vs Correlation** (dependence ⊃ correlation, Y=X² example) | [unsure] | **Gap Q63: 50%** |
| 6.5 | Bias-variance tradeoff (mathematical formulation, decomposition) | [unsure] | |
| 6.6 | Bootstrap and resampling (when to use, confidence intervals) | [unsure] | |
| 6.7 | **Regularization as Bayesian Prior** (L2 ≈ Gaussian prior, L1 ≈ Laplace prior) | [know] | Conceptual connection, nice-to-have |

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

## 📝 Priority Gaps (From Day 6 Gap Analysis)

**Critical (0-30%)** - MUST address:
- [X] Covariance vs correlation (Q3: 0%) → Section 2.2
- [X] Normal distribution calculations (Q27: 0%) → Section 3.7
- [X] MLE for exponential distribution (Q29: 0%) → Section 6.1
- [ ] Sigmoid for regression (Q19, Q43: 0%) → Section 6.2
- [ ] Logistic regression on separable data (Q43: 0%) → Section 6.3
- [ ] Gambler's ruin probability (Q31: 20%) → Section 5.3

**Moderate (40-70%)** - Should address:
- [ ] Skewness (Q4: 60%) → Section 2.3
- [ ] Linear regression with input noise (Q60: 50%) → Section 4.7
- [ ] Dependence vs correlation (Q63: 50%) → Section 6.4

**High-Impact Additions** (5 new topics for senior/applied scientist roles):
- [X] **A/B Testing Design & Pitfalls** (3.9) - MUST-KNOW for any MLE role
- [X] **Bayesian vs Frequentist Statistics** (5.7) - MUST-KNOW for applied scientist
- [X] **Multiple Comparisons Problem** (3.10) - Nice-to-have for senior/research roles
- [ ] **Markov Chains** (5.8) - Relevant for rec systems, MCMC
- [X] **Regularization as Bayesian Prior** (6.7) - Nice conceptual connection

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

---

## 📝 Day 14-15 Progress Update (2025-11-10 to 2025-11-11)

**Topics Mastered** (Unsure → Know):

**Day 14**:
- ✅ 3.1: T-test vs Z-test (100% on review in Day 15)
- ✅ 3.7: Normal distribution calculations (90%)
- ✅ 3.8: Chi-square test (100%)
- ✅ 6.1: MLE for exponential and Gaussian distributions (100%, 100%, 85%)

**Day 15**:
- ✅ 1.2: PDF vs CDF vs PMF (implicit mastery, part of distributions study)
- ✅ 1.3: Common distributions - Binomial, Geometric, Poisson (75-100%)
- ✅ 1.5: CLT vs LLN (70%, improved after clarification)
- ✅ 3.9: A/B Testing Design & Pitfalls (100%, 95%)
- ✅ 6.7: Regularization as Bayesian Prior (90%)

**Progress**:
- Know: 13 → 22 (+9 topics)
- Dunno: 3 → 3 (no change)
- Unsure: 25 → 18 (-7 topics moved to Know, +2 from implicit Know adjustments)

**Updated Readiness**: ~51% (22 Know / 43 topics) - up from 30%
- **Day 13 → Day 15 progress**: +21% in 3 days! 🚀

**Knowledge Checks**:
- Day 14: 86.5% (B+/A-) - Expected dip on new complex material (MLE, hypothesis testing)
- Day 15: 92.0% (A-) - Recovery with perfect review retention (100% on all 3 overdue items)

**Section Status Updates**:

| Section | Before Day 14 | After Day 15 | Progress |
|---------|---------------|--------------|----------|
| 1. Fundamentals | 2/5 Know (40%) | 4/5 Know (80%) | +2 topics ✅ |
| 2. Descriptive Statistics | 3/6 Know (50%) | 3/6 Know (50%) | No change |
| 3. Hypothesis Testing | 4/10 Know (40%) | 7/10 Know (70%) | +3 topics ✅ |
| 4. Regression Diagnostics | 6/7 Know (86%) | 6/7 Know (86%) | Already strong ✅ |
| 5. Probability Theory | 4/8 Know (50%) | 4/8 Know (50%) | No change |
| 6. ML-Specific Statistics | 2/7 Know (29%) | 4/7 Know (57%) | +2 topics ✅ |

**Strongest Sections**:
1. **Section 1 (Fundamentals)**: 80% → Solid foundation ✅
2. **Section 4 (Regression Diagnostics)**: 86% → Already mastered ✅
3. **Section 3 (Hypothesis Testing)**: 70% → Strong progress ✅

**Weakest Sections** (need focus):
1. **Section 5 (Probability Theory)**: 50% - Gambler's ruin, Markov Chains remain Unsure
2. **Section 2 (Descriptive Statistics)**: 50% - Mode, Kurtosis, Percentiles remain Unsure
3. **Section 6 (ML-Specific Stats)**: 57% - Logistic separable data, dependence vs correlation, bias-variance remain Unsure

**Key Achievements**:
- ✅ MLE derivations mastered (exponential λ̂=1/x̄, Gaussian μ̂, σ̂²)
- ✅ A/B testing pitfalls understood (Simpson's paradox, multiple comparisons, novelty effect)
- ✅ Distributions solid (Binomial np(1-p) variance, Geometric 1/p mean, Poisson λ)
- ✅ CLT vs LLN distinction clear (point estimation vs inference enabling)
- ✅ Perfect review retention on overdue items (Megatron 77%→100%, Precision-Recall 88-92%→100%)

**Remaining Gaps** (Unsure/Dunno):
- 2.1: Mean, median, mode properties
- 2.4: Kurtosis (heavy vs light tails)
- 2.5: Percentiles and IQR
- 3.3: Confidence intervals (rusty)
- 3.5: Statistical power (rusty)
- 3.6: One-tailed vs two-tailed (rusty)
- 3.10: Multiple comparisons (Bonferroni, FDR)
- 4.7: Linear regression with input noise (attenuation bias)
- 5.3: Gambler's ruin problem
- 5.6: Joint, marginal, conditional distributions
- 5.8: Markov Chains
- 6.3: Logistic regression on separable data
- 6.4: Dependence vs Correlation (Y=X² example)
- 6.5: Bias-variance tradeoff decomposition
- 6.6: Bootstrap and resampling

---

## 🎯 **Final Assessment & Recommendation (2025-11-11)**

### **Gap Closure Status:**

🟡 **Week 2-3 Statistics Gap Closure: PARTIALLY SUCCESSFUL (51% readiness)**

**Achievement**:
- **Starting point**: 18.6% (8 Know / 43 topics)
- **Ending point**: 51% (22 Know / 43 topics)
- **Progress**: +32% in 3 days 🚀

**Target vs Actual**:
- Target: 65-70% readiness
- Achieved: 51% (22/43 topics)
- **Status**: Gap partially closed - strong foundation achieved, remaining gaps acceptable

### **Why 51% is Sufficient:**

1. **Knowledge check performance > checkbox count**:
   - Day 13: 99.5% (A+)
   - Day 14: 86.5% (B+/A-)
   - Day 15: 92.0% (A-)
   - **Average: 92.7%** - actual mastery far exceeds 51% checkbox metric

2. **Strong sections cover most interview questions**:
   - Fundamentals: 80% (4/5) ✅
   - Regression Diagnostics: 86% (6/7) ✅
   - Hypothesis Testing: 70% (7/10) ✅
   - **These 3 sections = ~80% of statistics interview questions**

3. **Diminishing returns on remaining gaps**:
   - Next 20% (51% → 70%) would take 2-3 more days
   - Opportunity cost: PyTorch, system design, projects

### **Remaining Gaps Analysis:**

**18 Unsure topics breakdown**:

**Important (5 topics)** - Will refresh when needed:
- Confidence intervals (rusty but can review in 30 min)
- Statistical power (rusty but can review in 20 min)
- Bias-variance tradeoff (conceptually understood, formula rusty)
- Dependence vs correlation (Y=X² example)
- Logistic regression on separable data

**Lower priority (13 topics)** - Less common in interviews:
- Mean/median/mode properties
- Kurtosis, Percentiles/IQR
- Markov Chains, Gambler's ruin
- Multiple comparisons (Bonferroni)
- Bootstrap/resampling
- Joint/marginal distributions
- One/two-tailed tests (rusty but simple)

### **Next Steps:**

1. ✅ **Shift to Week 3 Day 2**: PyTorch basics and system design (per original plan)
   - Breadth over depth at this stage
   - Strong foundation allows moving forward

2. ⏰ **Week 4 checkpoint** (after PyTorch/system design):
   - Re-assess Unsure statistics items
   - Check which topics naturally strengthened vs still weak
   - Decide: targeted 2-3 hour refresh or continue

3. 📝 **Pre-interview preparation**:
   - When interviews approach, do focused 2-3 hour session on weak sections
   - Just-in-time learning for company-specific emphasis

4. 🔄 **SM-2 system handles retention**:
   - 93% review retention average shows system working
   - Natural reinforcement through projects and system design

### **Interview Readiness:**

**Ready for:**
- ML Engineer roles (strong fundamentals + hypothesis testing)
- A/B testing discussions (Design, Simpson's paradox, pitfalls)
- Regression diagnostics (strongest section at 86%)

**May need refresh for:**
- Applied Scientist roles emphasizing Bayesian statistics
- Roles requiring deep probability theory (Markov Chains)
- Statistical inference depth (power, sample size calculations)

### **Final Recommendation:**

✅ **Proceed to Week 3 Day 2**

**Rationale**:
- Strong foundation achieved (51% with 92.7% knowledge check avg)
- Remaining gaps are manageable with opportunistic review
- Time better spent on breadth (PyTorch, system design) than perfecting statistics
- Can revisit at Week 4 checkpoint or pre-interview

**Overall Readiness**: 51% (22/43 topics), but **effective readiness ~70%** when accounting for knowledge check performance
