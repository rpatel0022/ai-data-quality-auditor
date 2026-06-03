# Automated Data Quality Auditor

Detects, categorizes, and fixes data quality issues using rule-based validation + ML-based NLP.

## What it does

1. **Merges and profiles** customer + transaction datasets (types, nulls, stats)
2. **Rule-based checks**: missing values, type mismatches, outliers (IQR), format inconsistencies, duplicate rows, email validation, negative prices
3. **ML-assisted checks** (no paid APIs):
   - TF-IDF + cosine similarity for semantic duplicate detection
   - Fuzzy matching (rapidfuzz) for naming inconsistencies in categorical columns
4. **Auto-applies fixes**: imputation, type conversion, outlier clipping, case standardization, deduplication
5. **Generates audit report** with before/after comparison and quality score

## Approach

The pipeline combines two layers:
- **Rule layer**: deterministic checks that catch structural issues (nulls, types, outliers, formats)
- **ML layer**: TF-IDF vectorization + cosine similarity catches semantic duplicates (e.g. "120V" vs "120 Volts"), fuzzy string matching catches naming inconsistencies across categories

These are classical ML/NLP techniques -- not LLMs. Chosen because they're deterministic, fast, free, and auditable (you can explain every detection). An LLM layer could be added on top for ambiguous cases.

## Results

| Metric | Value |
|--------|-------|
| Datasets merged | 2 (customers + transactions) |
| Rows | 8,388 |
| Issues found | 12 |
| Quality score | 7.8/10 |
| Null reduction | 6,429 -> 0 (100%) |
| Issue types | MISSING, OUTLIER, FORMAT, DUPLICATE, SEMANTIC_DUPLICATE, NAMING |

## Key design decisions

- **1.5x IQR for outlier detection**: standard Tukey fence because transactional data outliers are likely data entry errors, not natural variation (for sensor data you'd use 3x)
- **TF-IDF cosine similarity threshold = 0.7**: below that, common product description boilerplate causes too many false positives
- **Median imputation for numeric, mode for categorical**: preserves distribution shape better than mean for skewed data

## Run it

```bash
pip install -r requirements.txt

# Download data from Kaggle:
# kaggle datasets download -d saidnizam/messy-e-commerce-dataset --unzip -p data/

python auditor.py
```

## Output

- `output/cleaned_data.csv` -- cleaned dataset
- `output/sample_cleaned_data.csv` -- first 100 rows for quick inspection
- `output/audit_report.md` -- human-readable audit report
- `output/audit_log.json` -- detailed issue log with all findings

## Tech

`pandas`, `scikit-learn` (TF-IDF), `rapidfuzz`, `numpy`
