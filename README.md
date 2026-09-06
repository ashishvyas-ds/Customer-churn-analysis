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

### Key Finding: Non-Linear Effect of Number of Products

Churn rate by `NumOfProducts`: 1 product → 27%, 2 products → ~7% (lowest), 3 products → ~82%, 4 products → ~100%.

This is a U-shaped, non-linear pattern, not a straight-line relationship. Two-product customers appear to be the most loyal segment, while customers holding 3 or more products represent a sharply higher-risk group — possibly linked to over-selling, dissatisfaction-driven multi-product acquisition, or a small, unusual sample size in the higher-count buckets (to be confirmed via segment sizes). This is one of the strongest and most actionable findings in the dataset.

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

## Next Steps

- **Step 4:** Statistical validation (chi-square, t-tests) to confirm which patterns above are statistically significant
- **Step 5:** SQL-based analysis (churn by geography, cohort/tenure analysis)
- **Step 6:** Customer segmentation (clustering)
- **Step 7:** Predictive modeling (with and without leakage fields)
- **Step 8–9:** Evaluation and explainability (SHAP)
- **Step 10:** Business translation and risk tiering
