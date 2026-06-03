# Automated Data Quality Auditor

A data quality pipeline that combines **rule-based validation** with **ML-based NLP** to automatically detect, categorize, and fix data quality issues. Uses TF-IDF vectorization for semantic duplicate detection and fuzzy string matching for naming inconsistency analysis — all running locally with no paid API calls.

## Problem

Real-world datasets have layers of quality issues: missing values, type mismatches, outliers, format inconsistencies, duplicate rows, and — hardest to catch — *semantic* duplicates where the same concept is represented differently (e.g., "Credit Card" vs "CreditCard", or dates in different formats that look similar). Traditional rule-based checks catch structural issues but miss semantic ones. This pipeline layers ML-based NLP on top to catch both.

## Dataset

**Source:** [Messy E-Commerce Dataset (Kaggle)](https://www.kaggle.com/datasets/saidnizam/messy-e-commerce-dataset) — two CSV files with real data quality issues.

**Files:**
- `customers.csv` — 1,010 rows, 9 columns (customer_id, name, age, gender, state, signup_date, email, phone_number, subscribe)
- `transactions.csv` — 8,199 rows, 10 columns (transaction_id, customer_id, transaction_date, product_id, quantity, unit_price, payment_method, discount_applied, transaction_status, review_text)

**Known issues in the raw data:**
- Missing values across 4 columns (discount_applied, review_text, age, state)
- Outliers in unit_price and phone_number
- 189 exact duplicate rows
- Mixed case formatting in review_text
- Naming inconsistencies across categorical columns (payment methods, states)
- Semantically similar date strings detectable via TF-IDF

## Pipeline Architecture

```
Load Data ──> Profile ──> Rule Checks ──> ML Checks ──> Score ──> Auto-Fix ──> Report
                │              │               │           │          │
                ▼              ▼               ▼           ▼          ▼
           Column stats   MISSING         SEMANTIC     Quality   Before/After
           Null rates     TYPE_MISMATCH   DUPLICATE    Score     Comparison
           Distributions  OUTLIER         NAMING       7.8/10
                          FORMAT
                          DUPLICATE
                          EMAIL
                          NEGATIVE
```

### Step 1: Load & Merge

- Loads both CSV files
- Detects that they share `customer_id` as a join key
- Performs a **left join** of transactions onto customers, enriching each transaction with customer demographics
- Result: 8,388 rows, 18 columns

### Step 2: Data Profiling

For every column, computes:
- Data type (`dtype`)
- Null count and percentage
- Unique value count
- 5 sample values
- For numeric columns: mean, std, min, max

This profile is saved in the audit log for reference.

### Step 3: Rule-Based Quality Checks

Seven deterministic checks, each returning typed, severity-rated issues:

| Check | What it detects | Severity logic |
|-------|-----------------|----------------|
| **Missing values** | Columns with any nulls | HIGH if >30%, MEDIUM if >10%, LOW otherwise |
| **Type mismatches** | Object columns where >80% of sampled values are numeric | MEDIUM — likely wrong dtype |
| **Outliers (IQR)** | Values outside Q1 - 1.5×IQR to Q3 + 1.5×IQR | HIGH if >5% outliers, LOW otherwise |
| **Format inconsistencies** | Mixed case patterns, leading/trailing whitespace | LOW — cosmetic but important for joins |
| **Duplicate rows** | Exact row-level duplicates via `df.duplicated()` | MEDIUM |
| **Email validation** | Regex check against RFC-compliant email pattern | MEDIUM |
| **Negative prices** | Negative values in price/cost/quantity/amount columns | HIGH — likely data entry error |

**Why 1.5x IQR?** This is the standard Tukey fence, appropriate for transactional data where outliers are likely data entry errors rather than natural variation. For sensor data (see Project 3), we use 3x IQR because natural spikes are expected.

### Step 4: ML-Assisted Quality Checks

Two NLP-based checks that catch issues rule-based logic can't:

#### TF-IDF Semantic Duplicate Detection

```
Text values ──> TF-IDF Vectorizer ──> Cosine Similarity Matrix ──> Flag pairs > 0.7
```

- Selects text columns with >5 unique values and >10 non-null entries
- Builds a TF-IDF matrix (max 1,000 features, English stop words removed)
- Computes pairwise cosine similarity
- Flags pairs with similarity between 0.7 and 1.0 (identical pairs excluded)
- **Why 0.7?** Below that, common product description boilerplate causes too many false positives. Above 0.9, you're basically catching exact matches which the duplicate check already handles.

#### Fuzzy Naming Inconsistency Detection

```
Unique values ──> rapidfuzz token_sort_ratio ──> Group values scoring 75-100 ──> Flag groups
```

- Targets categorical columns with 2-200 unique values
- For each unique value, finds the top 5 fuzzy matches
- Groups values with 75-100% similarity (100% excluded — those are exact matches)
- Reports the groups with suggested standardized values (first occurrence)

**Example findings:**
- `payment_method`: "Credit Card" vs "CreditCard" → suggest standardize
- `state`: Regional name variations detected via fuzzy matching

### Step 5: Quality Scoring

Computes an automated quality score (1-10) based on:

```
Score = 10.0
  - min(3.0, HIGH_issues × 0.5)      # Penalize critical problems
  - min(2.0, MEDIUM_issues × 0.2)    # Penalize moderate problems
  - min(2.0, (100 - completeness%) / 20)  # Penalize missing data
  = max(1.0, result)
```

For this dataset: **7.8/10** — moderate cleaning needed.

### Step 6: Auto-Apply Fixes

Each issue has a `suggested_fix` field. The pipeline auto-applies these:

| Fix Type | What it does | Applied to |
|----------|-------------|------------|
| `impute_median` | Fills nulls with column median | discount_applied, age |
| `impute_mode` | Fills nulls with most frequent value | review_text, state |
| `clip_to_iqr_bounds` | Clips values to [Q1-1.5×IQR, Q3+1.5×IQR] | unit_price, phone_number |
| `drop_duplicates` | Removes exact duplicate rows | full dataset |
| `standardize_to_lower` | Lowercases all values | review_text |
| `strip_whitespace` | Removes leading/trailing spaces | text columns |
| `convert_to_numeric` | Casts string column to numeric | detected mismatches |

## Results

| Metric | Before | After |
|--------|--------|-------|
| Total null values | 6,429 | 0 |
| Null reduction | — | **100%** |
| Rows | 8,388 | 8,388 |
| Issues detected | — | 12 (2 HIGH, 5 MEDIUM, 5 LOW) |
| Quality score | — | 7.8/10 |
| Fixes applied | — | 7 |

**Issues breakdown:**
- 4 MISSING issues (discount_applied, review_text, age, state)
- 2 OUTLIER issues (unit_price, phone_number)
- 1 FORMAT issue (mixed case in review_text)
- 1 DUPLICATE issue (189 exact duplicates)
- 1 SEMANTIC_DUPLICATE issue (192 similar date pairs via TF-IDF)
- 3 NAMING issues (payment_method, review_text, state inconsistencies)

## Project Structure

```
project2_ai_quality_auditor/
├── auditor.py               # Main pipeline — run this
├── rules_engine.py          # Standalone rule engine (reusable module)
├── requirements.txt         # Python dependencies
├── data/                    # Raw Kaggle CSVs (not committed)
├── output/
│   ├── cleaned_data.csv              # Full cleaned dataset
│   ├── sample_cleaned_data.csv       # First 100 rows for quick inspection
│   ├── audit_report.md              # Human-readable audit report
│   └── audit_log.json               # Full issue log with details
└── docs/
    └── index.html           # GitHub Pages dashboard
```

### File Details

**`auditor.py`** (main pipeline): Orchestrates all 6 steps. Each check function returns a list of issue dicts with consistent schema: `{type, column, severity, detail, suggested_fix}`. This makes issues composable and machine-readable.

**`rules_engine.py`** (reusable module): A standalone `RulesEngine` class with pluggable rule functions. Can be imported and used independently:
```python
from rules_engine import create_default_engine
engine = create_default_engine()
issues = engine.run(df)
print(engine.summary())  # {'FORMAT': 2, 'OUTLIER': 1, 'DUPLICATE': 1}
```

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the dataset from Kaggle
kaggle datasets download -d saidnizam/messy-e-commerce-dataset --unzip -p data/

# 3. Run the pipeline
python auditor.py
```

## Tech Stack

| Library | Purpose |
|---------|---------|
| `pandas` | Data loading, profiling, manipulation, type detection |
| `scikit-learn` | `TfidfVectorizer` for text vectorization, `cosine_similarity` for pairwise comparison |
| `rapidfuzz` | `fuzz.token_sort_ratio` for fuzzy string matching, `process.extract` for top-k matching |
| `numpy` | Numeric operations, NaN handling, statistical computations |

## Design Philosophy

**Why classical ML instead of LLMs?**

These are classical ML/NLP techniques — not large language models. Chosen deliberately because:
1. **Deterministic**: Same input always produces same output. Critical for data auditing.
2. **Auditable**: You can explain exactly why a pair was flagged (cosine similarity = 0.83, fuzzy score = 78).
3. **Fast**: Processes 8,000+ rows in seconds, not minutes.
4. **Free**: No API costs, no rate limits, no data leaving your machine.
5. **Extensible**: An LLM layer could be added on top for ambiguous cases where TF-IDF/fuzzy matching is insufficient.
