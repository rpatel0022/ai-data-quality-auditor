# AI-Powered Data Quality Auditor

Automatically detects, categorizes, and fixes data quality issues using ML/NLP techniques.

**Interview story:** "I built a tool that uses AI to automatically detect data quality issues -- like inconsistent naming, outliers, and semantic duplicates -- and auto-applies fixes. This is exactly how AI can accelerate datasheet auditing."

## What it does

1. **Profiles the dataset** (types, nulls, unique counts, statistics)
2. **Rule-based checks**: missing values, type mismatches, outliers (IQR), format inconsistencies, duplicate rows, email validation, negative price detection
3. **ML-assisted checks** (no paid APIs):
   - TF-IDF + cosine similarity for semantic duplicate detection
   - Fuzzy matching (rapidfuzz) for naming inconsistencies across categorical columns
4. **Auto-applies fixes**: imputation, type conversion, outlier clipping, case standardization, deduplication
5. **Generates audit report** with before/after comparison

## Results

| Metric | Value |
|--------|-------|
| Datasets merged | 2 (customers + transactions) |
| Rows | 8,388 |
| Issues found | 12 |
| Quality score | 7.8/10 |
| Null reduction | 6,429 -> 0 (100%) |
| Issue types | MISSING, OUTLIER, FORMAT, DUPLICATE, AI_SEMANTIC_DUPLICATE, AI_NAMING |

## Run it

```bash
# Download data from Kaggle first:
# kaggle datasets download -d saidnizam/messy-e-commerce-dataset --unzip -p data/

python auditor.py
```

## Output

- `output/cleaned_data.csv` -- cleaned dataset
- `output/audit_report.md` -- human-readable audit report
- `output/audit_log.json` -- detailed issue log with all findings

## Tech

`pandas`, `scikit-learn` (TF-IDF), `rapidfuzz`, `numpy`
