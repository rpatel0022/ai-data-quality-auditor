# Data Quality Audit Report

**Datasets:** customers.csv, transactions.csv
**Rows:** 8,388
**Issues found:** 12

## Quality Assessment
Data Quality Score: 7.8/10

- Dataset completeness: 95.7% (6,429 null values across 150,984 cells)
- Issues found: 12 total (2 HIGH, 5 MEDIUM)
- Issue breakdown: {"MISSING": 4, "OUTLIER": 2, "FORMAT": 1, "DUPLICATE": 1, "AI_SEMANTIC_DUPLICATE": 1, "AI_NAMING": 3}
- Most critical: Fix missing values and outliers first
- Recommendation: Moderate cleaning needed

## Issues by Type

### AI_NAMING (3 issues)
- **payment_method** [MEDIUM]: 1 naming inconsistency groups in 'payment_method'
  - Fix: standardize_similar_names
- **review_text** [MEDIUM]: 6 naming inconsistency groups in 'review_text'
  - Fix: standardize_similar_names
- **state** [MEDIUM]: 4 naming inconsistency groups in 'state'
  - Fix: standardize_similar_names

### AI_SEMANTIC_DUPLICATE (1 issues)
- **transaction_date** [MEDIUM]: 192 semantically similar value pairs detected via TF-IDF
  - Fix: review_and_merge_similar_values

### DUPLICATE (1 issues)
- **ALL_ROWS** [MEDIUM]: 189 exact duplicate rows (2.3%)
  - Fix: drop_duplicates

### FORMAT (1 issues)
- **review_text** [LOW]: Mixed case formats: UPPER=0, lower=4, Title=0
  - Fix: standardize_to_lower

### MISSING (4 issues)
- **discount_applied** [LOW]: 440 missing values (5.2%)
  - Fix: impute_median
- **review_text** [HIGH]: 5031 missing values (60.0%)
  - Fix: impute_mode
- **age** [LOW]: 484 missing values (5.8%)
  - Fix: impute_median
- **state** [LOW]: 474 missing values (5.7%)
  - Fix: impute_mode

### OUTLIER (2 issues)
- **unit_price** [LOW]: 110 outliers (1.3%) outside [-155000.00, 725000.00]
  - Fix: clip_to_iqr_bounds
- **phone_number** [HIGH]: 1843 outliers (22.0%) outside [-131046569618.00, 220775142614.00]
  - Fix: clip_to_iqr_bounds

## Fixes Applied
- Filled discount_applied missing with median (25.00)
- Filled review_text missing with mode (Honest seller, item as described)
- Filled age missing with median (41.00)
- Filled state missing with mode (Banten)
- Clipped 110 outliers in unit_price
- Clipped 1843 outliers in phone_number
- Standardized review_text to lowercase

## Before vs After
- **Null values:** 6,429 -> 0 (100.0% reduction)
- **Rows:** 8,388 -> 8,388