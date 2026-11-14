# Bank Marketing Campaign — Exploratory Data Analysis

**Project 01 of 22 · Machine Learning Portfolio · Python / pandas / Matplotlib**

An end-to-end exploratory data analysis (EDA) of a bank's direct-marketing campaign records. The goal is to understand the dataset, examine data quality, compare customer segments, and identify **associations** with term-deposit subscription outcomes before attempting any predictive modeling.

> **Scope:** This is an exploratory, descriptive analysis—not a predictive machine-learning model, causal study, or validated marketing optimization system. The findings suggest questions to investigate, not guaranteed interventions.

## Table of contents

1. [Project overview](#project-overview)
2. [Dataset](#dataset)
3. [Tools and techniques](#tools-and-techniques)
4. [Repository structure](#repository-structure)
5. [Getting started](#getting-started)
6. [Analysis workflow](#analysis-workflow)
7. [Key findings](#key-findings)
8. [Data-quality considerations](#data-quality-considerations)
9. [Interpretation and business recommendations](#interpretation-and-business-recommendations)
10. [Limitations and responsible use](#limitations-and-responsible-use)
11. [Skills developed](#skills-developed)
12. [Potential next steps](#potential-next-steps)
13. [Data attribution](#data-attribution)

## Project overview

A bank runs direct-marketing campaigns to encourage customers to subscribe to term deposits. The dataset captures customer characteristics, prior campaign information, and the outcome of the current marketing contact.

This project answers questions such as:

- How frequently did customers subscribe to a term deposit?
- How do subscription rates vary across occupations and ages?
- Are higher account balances associated with higher subscription rates?
- How is the number of contacts within a campaign associated with conversion?
- How does call duration differ between successful and unsuccessful contacts?
- What problems arise from `unknown` values, sentinel values, extreme observations, and imbalanced groups?
- Which available fields could lead to **data leakage** if used in a model intended to predict *before* a call?

**Main deliverable:** [`Bank_Marketing_EDA_Portfolio_Clean.ipynb`](Bank_Marketing_EDA_Portfolio_Clean.ipynb), a self-contained, structured notebook with tables, visualizations, and written interpretations.

## Dataset

**Source:** [UCI Machine Learning Repository — Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)  
**File analyzed:** `bank-full.csv` (the full, older 17-column version)  
**Rows:** 45,211  
**Columns:** 17 (16 input variables plus the outcome `y`)  
**Target:** `y`, whether the customer subscribed to a term deposit (`yes` / `no`)  
**File format:** Semicolon-delimited CSV (`sep=";"`)

The variables include:

| Category | Fields | Description |
|---|---|---|
| Customer information | `age`, `job`, `marital`, `education` | Demographic and occupational characteristics |
| Financial information | `default`, `balance`, `housing`, `loan` | Credit, account, and loan information |
| Current contact | `contact`, `day`, `month`, `duration`, `campaign` | Contact method, timing, duration, and number of campaign contacts |
| Previous campaign | `pdays`, `previous`, `poutcome` | Prior contact history and outcome |
| Outcome | `y` | Term-deposit subscription status |

Some fields have important context-specific meanings. For example, `pdays = -1` indicates no previous contact, and `unknown` is an explicitly encoded category in several columns rather than a missing value automatically recognized by pandas.

**Important:** The notebook targets **`bank-full.csv`**, not the differently structured `bank-additional-full.csv` variant.

## Tools and techniques

- **Python 3** — analysis and notebook execution
- **pandas** — importing data, filtering, cross-tabulation, grouped aggregation, quantiles, and binning
- **Matplotlib** — histograms, box plots, and comparative bar charts
- **Jupyter Notebook / Kaggle Notebooks** — reproducible analysis workflow

Statistical and analytical concepts explored:

- Descriptive statistics (mean, median, quartiles, standard deviation)
- Univariate and bivariate analysis
- Missing or `unknown` category investigation
- Frequency counts and within-group conversion rates
- Distribution shape, skewness, and outlier detection using the **1.5 × IQR rule**
- Fixed-width categorization (`pd.cut`) and equal-frequency binning (`pd.qcut`)
- Class imbalance, small-sample instability, selection effects, and data leakage
- Association versus causation

## Repository structure

A suggested layout for publishing the project:

```text
bank-marketing-eda/
├── README.md
├── Bank_Marketing_EDA_Portfolio_Clean.ipynb
├── data/
│   └── bank-full.csv          # Download separately from UCI
└── .gitignore                # Optional
```

The notebook does **not** require a separate Python script or pre-generated chart images. Plots are generated when its cells run. You can omit `data/` from your public repository and document how to obtain the data instead.

## Getting started

### Option A — Run in Kaggle

1. Open [Kaggle Notebooks](https://www.kaggle.com/code).
2. Create a notebook or import `Bank_Marketing_EDA_Portfolio_Clean.ipynb`.
3. Attach a dataset containing `bank-full.csv`.
4. Run all cells from top to bottom.

The notebook first checks a set of known Kaggle file paths, then searches Kaggle's `/kaggle/input` directory for `bank-full.csv`.

### Option B — Run locally

1. Clone or download this repository.
2. Ensure Python 3 is installed.
3. Install the required packages:

   ```bash
   python -m pip install pandas matplotlib notebook ipython
   ```

4. Download the **Bank Marketing** dataset from [UCI](https://archive.ics.uci.edu/dataset/222/bank+marketing), extract it, and place `bank-full.csv` at `data/bank-full.csv`.
5. From the project directory, launch Jupyter:

   ```bash
   jupyter notebook
   ```

6. Open `Bank_Marketing_EDA_Portfolio_Clean.ipynb` and choose **Run All**.

The notebook also supports `bank-full.csv` in the repository root. It expects the file to use semicolons as separators.

**Expected initial check:** A successful load reports **45,211 rows × 17 columns**. If the notebook reports a single column, confirm that the file was read using `sep=";"`.

### Reproducibility note

Run cells in order from a fresh kernel. The age, campaign, and balance groups are constructed during notebook execution. The stored notebook may not contain executed outputs; running the notebook produces the tables and figures.

## Analysis workflow

The notebook is organized into **10 sections**:

| Section | Analysis | Purpose |
|---|---|---|
| 1. Setup and overview | Load CSV, inspect rows, types, and descriptive summaries | Establish the structure and scale of the data |
| 2. Data quality | Count `unknown` categories; cross-tabulate `previous` and `poutcome` | Distinguish encoded unknowns from ordinary nulls |
| 3. Distributions and outliers | Balance histograms, IQR calculations, box plot | Identify skewness and unusual account balances |
| 4. Subscription outcomes | Target frequencies and proportions | Establish the baseline conversion rate and class balance |
| 5. Occupation | Customer counts and conversion rates by job | Compare volume with within-occupation success rates |
| 6. Age | Box plots and age-group conversion rates | Examine differences and identify small-sample risks |
| 7. Campaign contacts | Grouped contact frequency and conversion | Investigate a potential relationship with repeated outreach |
| 8. Call duration | Median duration and comparative box plots | Understand engagement patterns and prediction-time leakage |
| 9. Account balance | Median comparisons and quintile conversion rates | Examine financial segmentation without dropping negative balances |
| 10. Findings | Synthesized evidence, caveats, and recommendations | Communicate actionable hypotheses responsibly |

## Key findings

The following are **descriptive results from the full dataset**. They should not be interpreted as causal effects or proof of which marketing intervention works best.

### 1. Overall subscription outcome

| Outcome | Customers | Share |
|---|---:|---:|
| Subscribed (`yes`) | 5,289 | 11.7% |
| Did not subscribe (`no`) | 39,922 | 88.3% |
| **Total** | **45,211** | **100%** |

The positive class is relatively uncommon. If this project is later extended to classification, accuracy alone will be an insufficient performance measure; precision, recall, and other metrics will matter.

### 2. Occupation is associated with subscription outcomes

| Occupation | Successful subscriptions | Subscription rate |
|---|---:|---:|
| Student | 269 | 28.68% |
| Retired | 516 | 22.79% |
| Management | 1,301 | 13.76% |
| Blue-collar | 708 | 7.27% |

Students have the highest observed subscription rate, while management generates the most successful subscriptions among these groups. This illustrates why **rate** and **volume** should be evaluated separately.

These differences may also reflect other characteristics, such as age or financial circumstances.

### 3. Age-group conversion rates differ substantially

| Age group used in notebook | Customers | Subscription rate |
|---|---:|---:|
| Young (≤18) | 12 | 58.33% |
| Young adults (19–30) | 7,018 | 16.22% |
| Adults (31–45) | 23,733 | 9.88% |
| Middle-aged (46–60) | 13,260 | 9.78% |
| Seniors (61–75) | 927 | 41.53% |
| Elderly (76+) | 261 | 44.83% |

The youngest group's rate is based on only **12 records**, so it must be interpreted cautiously. The older groups show elevated observed subscription rates, but the analysis does not establish that age itself causes subscriptions.

The median age is similar across outcomes: **38** for subscribers and **39** for non-subscribers. This illustrates why medians alone can conceal differences in subgroup behavior.

### 4. More campaign contacts are associated with lower conversion

| Number of contacts during campaign | Customers | Subscription rate |
|---|---:|---:|
| Low (1–2) | 30,049 | 13.19% |
| Moderate (3–5) | 10,807 | 9.94% |
| High (6–10) | 3,159 | 6.52% |
| Very high (11+) | 1,196 | 3.93% |

The observed subscription rate falls as the number of campaign contacts rises. This **does not** justify a conclusion that more calls *cause* lower conversion or that the bank should stop after two calls: more difficult-to-convert customers might receive more attempts.

### 5. Subscribers have longer recorded call durations

| Outcome | Median last-call duration |
|---|---:|
| Not subscribed | 164 seconds |
| Subscribed | 426 seconds |

The median last-call duration is about **2.6 times longer** for subscribers. However, final call duration is only known during or after the call.

**Data-leakage implication:** A model intended to predict subscription **before** calling a customer must not use the final duration of that upcoming call. It can still be studied in historical EDA.

### 6. Higher account-balance quintiles show higher subscription rates

Five quantile-based groups contain roughly equal numbers of customers, unlike arbitrary balance bands that may leave most records in a single category.

| Account-balance quintile | Customers | Subscription rate |
|---|---:|---:|
| Very low | 9,053 | 6.96% |
| Low | 9,060 | 10.00% |
| Medium | 9,017 | 11.71% |
| High | 9,039 | 13.46% |
| Very high | 9,042 | 16.37% |

The highest quintile has approximately **2.35 times** the subscription rate of the lowest. The median balance is **733** among subscribers versus **417** among non-subscribers.

These figures describe association only. They do not establish what causes subscriptions or which customers are most profitable.

### 7. Balance distribution is right-skewed

| Measure | Value |
|---|---:|
| First quartile (Q1) | 72 |
| Median | 448 |
| Third quartile (Q3) | 1,428 |
| IQR | 1,356 |
| Lower IQR fence | -1,962 |
| Upper IQR fence | 3,462 |
| Lower-fence observations | 17 |
| Upper-fence observations | 4,712 |
| **Total flagged by 1.5 × IQR** | **4,729 (10.46%)** |

Most flagged observations are on the upper side, consistent with the observed long right tail. The IQR rule flags **potential** outliers; it does not establish that these balances are mistakes. The notebook retains the original data.

## Data-quality considerations

### Encoded unknowns

Although pandas reports no conventional null values in the full dataset, several fields encode `unknown` as a literal string. The notebook examines `job`, `education`, `contact`, and `poutcome` rather than blindly replacing or deleting these records.

### Special values

In `pdays`, `-1` indicates no previous campaign contact. The notebook checks the relationship between `previous = 0` and `poutcome = unknown` to understand what those values mean in context.

### Extreme values

The account-balance range includes negative values as well as unusually large positive balances. Both must be considered when selecting bins and interpreting summaries. Filtering to nonnegative values would silently exclude part of the dataset.

### Small segments

A high conversion percentage based on a handful of customers can be unstable. The analysis reports group sizes alongside rates so readers can recognize this issue.

## Interpretation and business recommendations

The findings suggest **areas for investigation**, not ready-made targeting rules:

1. **Compare conversion rate and conversion volume.** Students and retired customers have high historical rates, while management contributes more absolute subscriptions.
2. **Investigate customer segmentation.** Age and account balance show differences in conversion, but patterns should be tested after accounting for potential confounding factors.
3. **Review campaign contact strategies.** The decline in observed conversion with higher contact counts may motivate controlled tests of follow-up policies; the observational trend alone cannot determine an optimal stopping rule.
4. **Validate operational usefulness.** A feature useful for retrospective analysis may not be available when an operational prediction must be made.
5. **Include economics before optimizing business decisions.** Subscription conversion alone does not quantify deposits placed, customer lifetime value, campaign costs, or profit.

## Limitations and responsible use

- **Observational data:** The analysis finds associations, not causal relationships.
- **No predictive model:** There is no train/test split, model fitting, or model-performance claim in this notebook.
- **Data leakage risk:** `duration` should not be included in a pre-call prediction feature set; other features must also be checked for availability at the chosen prediction point.
- **Group-size differences:** Subgroup conversion rates should be read alongside sample sizes and, in further work, uncertainty intervals.
- **Historical context:** The dataset reflects a specific bank's marketing records; findings may not generalize to other institutions, countries, or time periods.
- **Limited financial data:** The dataset does not support conclusions about segment profitability or ROI.
- **No automated decision-making:** Findings should not be used by themselves to make consequential customer decisions.

## Skills developed

This project provided practical experience with:

- Reading and validating real CSV datasets
- Understanding data types and variable semantics
- Interpreting missing and sentinel values
- Selecting appropriate descriptive statistics
- Building and interpreting histograms, box plots, and bar charts
- Calculating IQR fences and assessing potential outliers
- Avoiding misleading conclusions from imbalanced groups
- Using `groupby`, `crosstab`, `cut`, and `qcut` for segmentation
- Distinguishing correlation from causation
- Recognizing prediction-time data leakage
- Writing evidence-based business insights with limitations

## Potential next steps

Possible extensions after this EDA project include:

1. Assess uncertainty in subgroup conversion rates with confidence intervals.
2. Explore combinations of variables rather than relying on single-feature comparisons.
3. Investigate class imbalance and appropriate classification evaluation methods in a **separate ML modeling project**.
4. Define a precise prediction time and build a leakage-aware feature set.
5. Design a controlled experiment to test whether changes in outreach strategy improve incremental conversions and profitability.

## Data attribution

The underlying data comes from the **UCI Machine Learning Repository — Bank Marketing** dataset.

- Dataset page: https://archive.ics.uci.edu/dataset/222/bank+marketing
- Reference: Moro, S., Cortez, P., and Rita, P. (2014). *A Data-Driven Approach to Predict the Success of Bank Telemarketing.* Decision Support Systems, 62, 22–31.

Please consult the UCI dataset page for the current licensing and citation requirements before redistributing the original data. This repository's analysis and figures are independent project work based on that dataset.

---

**Project status:** EDA completed; notebook can be executed and published as a portfolio project after verifying its outputs in the intended environment.
