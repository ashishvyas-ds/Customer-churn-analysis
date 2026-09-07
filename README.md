# Customer Churn Analysis — Retail Bank

## Business Problem

This dataset is from Kaggle and has no real business behind it. The scenario below is a reasonable hypothetical constructed for this project.

The bank is experiencing a ~20% annual customer churn rate. Retention efforts today are reactive — the bank typically only becomes aware a customer is dissatisfied after they've already complained or left. There is no system in place to flag at-risk customers *before* that point.

**Objective:** Identify the key drivers of customer churn and build a model that ranks customers by churn risk, so the retention team can proactively prioritize outreach to the customers most likely to leave — before they complain or exit.

**Success looks like:** a model and set of insights that let the bank act *earlier* than it does today, with clear reasoning the retention team can trust, not just a black-box score.

**Scope note:** This project focuses on identifying risk drivers and building a risk-ranking model. It does not design specific retention offers or measure real-world campaign ROI, since no real financial or campaign data exists for this dataset. Cost/benefit figures used later in this project are illustrative assumptions, clearly labeled as such.

**Modeling constraint:** Because `Complain` and `Satisfaction Score` only exist after a complaint event (see Data Leakage section below), they cannot be used for early prediction. The model must rely on signals available *before* a customer shows dissatisfaction.

---

## Dataset Overview

- 10,000 customer records from a retail bank operating in France, Germany, and Spain
- 18 columns covering demographics, account details, engagement, and churn outcome (`Exited`)
- Source: Kaggle

**Assumptions from the data dictionary (going in):**

| Column | Assumed relationship to churn |
|---|---|
| CreditScore | Higher score → less likely to churn |
| Geography | Location may affect churn (exploratory) |
| Gender | May affect churn (exploratory) |
| Age | Older customers → less likely to churn |
| Tenure | Longer tenure → more loyal, less likely to churn |
| Balance | Higher balance → less likely to churn |
| NumOfProducts | More products → assumed more engaged/loyal |
| HasCrCard | Having a credit card → less likely to churn |
| IsActiveMember | Active members → less likely to churn |
| EstimatedSalary | Lower salary → more likely to churn |
| Complain | Post-event field (has customer complained) |
| Satisfaction Score | Post-event field — score for complaint *resolution* specifically |

`RowNumber`, `CustomerId`, and `Surname` are identifiers with no behavioral meaning and were dropped before analysis.

---

## Step 2: Data Cleaning

**What:** Checked for missing values, duplicates, wrong data types, and obvious errors.

**Why:** Given the dataset size (10,000 rows), cleaning was done directly in pandas. In a production setting with much larger data, this cleaning logic would typically run as SQL directly against the source database (or a distributed engine like Spark), with only a reduced dataset pulled into Python for modeling.

**How/where:** Python (pandas)

**Findings:**
- No missing values in any column
- No fully duplicated rows, no duplicate `CustomerId`s
- All data types correct; all categorical values clean (no typos/inconsistent casing)
- All numeric ranges sane: Age 18–92, CreditScore 350–850, Tenure 0–10, Balance $0–$250,898
- **Dataset required no corrective cleaning** — noted here as a finding in itself

---

## Step 3: Exploratory Data Analysis (EDA)

**What:** Examined how churn rate varies across demographic, financial, and engagement variables, and tested the dataset's own stated assumptions against the actual data.

**Why:** Before modeling, it's necessary to know which variables genuinely relate to churn, whether relationships are linear or not, and to surface any data leakage before it silently inflates model performance.

**How/where:** Python (pandas for aggregation, matplotlib/seaborn for visualization)

**Overall churn rate:** ~20.4%

### Assumption vs. Reality

| Variable | Assumed Effect | What the Data Actually Shows |
|---|---|---|
| Age | Older = more loyal | **Reversed** — churned customers are notably older (median ~45 vs ~37 for retained) |
| Tenure | Longer = more loyal | **No effect** — churn rate flat (~17–23%) across all tenure values |
| Balance | Higher = more loyal | **Reversed** — churners have a higher average balance (~$91K vs ~$73K); zero-balance customers are concentrated almost entirely among retained customers |
| CreditScore | Higher = more loyal | **No effect** — nearly identical distributions (mean ~645 vs ~652) |
| NumOfProducts | More = more loyal (implied) | **Non-linear** — churn drops from 27% (1 product) to ~7% (2 products, safest), then spikes to ~82% (3 products) and ~100% (4 products) |
| Card Type | Not stated as predictive | **No effect** — churn rate nearly identical across all card types (19.3–21.8%) |
| Satisfaction Score | Post-event field | **No effect** — flat across all scores (19.6–21.8%), consistent with it being a narrow post-complaint metric, not a general health indicator |
| IsActiveMember | Active = more loyal | **Confirmed** — negative correlation with churn (-0.16); active members churn less |
| Geography | Exploratory | **Real effect** — Germany churns notably higher (~32.4%) than Spain (~16.7%) and France (~16.2%) |
| Gender | Exploratory | **Real effect** — Female customers churn more (~25.1%) than Male customers (~16.5%) |

### Geography and Gender

- **Geography:** Germany has notably higher churn (~32.4%) than Spain (~16.7%) or France (~16.2%) — roughly double. Worth investigating further in a real setting (e.g., product fit, local competition, service quality), though outside the scope of this dataset.
- **Gender:** Female customers churn at a higher rate (~25.1%) than Male customers (~16.5%). A meaningful gap, though the underlying cause isn't available in this dataset.

Both variables were flagged as "exploratory" in the original data dictionary rather than assumed predictors, and both turned out to show a real, non-trivial difference in churn rate — unlike several of the variables the dictionary *did* assume would matter (Tenure, CreditScore, Card Type).

### Key Finding: Non-Linear Effect of Number of Products

Churn rate by `NumOfProducts`: 1 product → 27%, 2 products → ~7% (lowest), 3 products → ~82%, 4 products → ~100%.

This is a U-shaped, non-linear pattern, not a straight-line relationship. Two-product customers appear to be the most loyal segment, while customers holding 3 or more products represent a sharply higher-risk group.

**Segment sizes (checked via `value_counts()`):**

| NumOfProducts | Customer Count | % of Total |
|---|---|---|
| 1 | 5,084 | 50.8% |
| 2 | 4,590 | 45.9% |
| 3 | 266 | 2.7% |
| 4 | 60 | 0.6% |

The 1- and 2-product comparison (96.7% of customers combined) is based on large samples and is solid. The 3- and 4-product churn rates (~82% and ~100%) are directionally real and worth flagging, but rest on much smaller groups (266 and 60 customers respectively) — the 100% figure in particular is fragile, since a handful of retained customers in that group of 60 would shift it noticeably. This is still one of the strongest and most actionable patterns in the dataset, but the confidence in the extreme end of it should be stated proportionally to sample size.

### Data Leakage: `Complain` and `Satisfaction Score`

A crosstab of `Complain` vs. `Exited` shows:

|  | Exited = 0 | Exited = 1 |
|---|---|---|
| Complain = 0 | 7,952 | 4 |
| Complain = 1 | 10 | 2,034 |

- Non-complainers: 99.95% stayed
- Complainers: 99.5% churned

`Complain` and `Exited` are effectively the same event recorded in two columns (correlation = 1.00). This is confirmed further by the correlation heatmap, and is consistent with `Satisfaction Score` being described in the data dictionary as a score for *complaint resolution* — meaning both fields only exist **after** a churn-related event has already begun.

**Modeling implication:** `Complain` and `Satisfaction Score` must be excluded from any realistic, early-warning churn model. Including them would let a model "cheat" by referencing information that is only available after the outcome has already occurred. Two model variants are planned: one including these fields (to illustrate the leakage effect), and one realistic, deployable version that excludes them.

### Correlation Summary

From the correlation heatmap against `Exited`:

- `Complain`: 1.00 (leakage, excluded from modeling)
- `Age`: 0.29 (strongest genuine linear predictor)
- `IsActiveMember`: -0.16 (active members churn less)
- `Balance`: 0.12 (mild positive)
- `NumOfProducts`: -0.05 (appears weak here, but this understates its real importance — the relationship is non-linear/U-shaped, which linear correlation does not capture well)
- `CreditScore`, `Tenure`, `EstimatedSalary`, `HasCrCard`, `Satisfaction Score`, `Point Earned`: all near 0.00, no meaningful linear relationship with churn

**Note on correlation limitations:** Correlation coefficients only capture linear relationships. `NumOfProducts` is a clear example of a variable that looks unimportant in the heatmap but is actually one of the strongest churn drivers once its non-linear shape is examined directly.

---

## Step 4: Statistical Validation

**What:** Formally tested whether the patterns found in EDA are statistically significant, using chi-square tests for categorical variables (Geography, Gender, NumOfProducts, HasCrCard, IsActiveMember, Card Type, Satisfaction Score, Complain) and Welch's t-tests for numeric variables (Age, Tenure, Balance, CreditScore, EstimatedSalary), each against `Exited`.

**Why:** Visual patterns from EDA can be misleading — some may be random noise, others (like the small `NumOfProducts` 3–4 segments) needed confirmation that the pattern holds up formally despite smaller sample sizes. This step also distinguishes **statistical significance** from **practical significance** — with 10,000 rows, even trivially small differences can register as "significant," so effect size still matters alongside the p-value.

**How/where:** Python (`scipy.stats` — `chi2_contingency`, `ttest_ind`)

### Chi-Square Results (Categorical Variables)

| Variable | Chi2 | p-value | Significant? |
|---|---|---|---|
| Geography | 300.63 | 5.2×10⁻⁶⁶ | Yes |
| Gender | 112.40 | 2.9×10⁻²⁶ | Yes |
| NumOfProducts | 1501.50 | ~0.0 | Yes — strongest relationship in the dataset |
| HasCrCard | 0.45 | 0.503 | No |
| IsActiveMember | 243.69 | 6.2×10⁻⁵⁵ | Yes |
| Card Type | 5.05 | 0.168 | No |
| Satisfaction Score | 3.80 | 0.434 | No |
| Complain | 9907.91 | ~0.0 | Yes — extreme; confirms leakage numerically |

### T-Test Results (Numeric Variables)

| Variable | t-stat | p-value | Significant? |
|---|---|---|---|
| Age | 30.42 | 4.4×10⁻¹⁷⁹ | Yes |
| Tenure | -1.35 | 0.177 | No |
| Balance | 12.48 | 5.8×10⁻³⁵ | Yes |
| CreditScore | -2.60 | 0.0093 | Statistically yes, practically negligible (see note) |
| EstimatedSalary | 1.24 | 0.214 | No |

**Note on CreditScore:** Although p=0.0093 clears the standard significance threshold, the actual effect size is small — mean CreditScore is ~652 (retained) vs ~645 (churned), a ~7-point gap on an 850-point scale. With a sample of 10,000, even minor differences can produce a low p-value. This is a case where **statistical significance does not imply practical significance** — the gap is too small to be a useful signal for identifying at-risk customers in practice.

### Final Driver Verdict

| Variable | Statistically Significant? | Practically Meaningful? |
|---|---|---|
| Age | Yes | Yes — strong driver |
| Balance | Yes | Yes — meaningful, counter-intuitive direction |
| Geography | Yes | Yes — Germany ~2x higher churn |
| Gender | Yes | Yes — meaningful gap |
| NumOfProducts | Yes (strongest overall) | Yes — most actionable finding |
| IsActiveMember | Yes | Yes — real protective effect |
| CreditScore | Yes (p=0.009) | **No** — effect size negligible (~7 points) |
| Tenure | No | No |
| EstimatedSalary | No | No |
| HasCrCard | No | No |
| Card Type | No | No |
| Satisfaction Score | No | No |
| Complain | Yes (extreme) | Leakage — excluded from modeling |

**Conclusion:** Age, Balance, Geography, Gender, NumOfProducts, and IsActiveMember are confirmed as genuine, statistically-backed churn drivers and will carry the most weight in modeling (Step 7). Tenure, EstimatedSalary, HasCrCard, Card Type, and Satisfaction Score are confirmed as non-drivers and can be deprioritized or dropped. CreditScore is a borderline case, statistically detectable but practically weak. `Complain` is definitively confirmed as leakage and will be excluded from the realistic model.

---

## Next Steps

- **Step 5:** SQL-based analysis (churn by geography, cohort/tenure analysis)
- **Step 6:** Customer segmentation (clustering)
- **Step 7:** Predictive modeling (with and without leakage fields)
- **Step 8–9:** Evaluation and explainability (SHAP)
- **Step 10:** Business translation and risk tiering
