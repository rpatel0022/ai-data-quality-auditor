"""
Automated Data Quality Auditor
================================
Combines rule-based validation with ML-based NLP to detect and fix data quality issues.

Approach:
- Rule layer: deterministic checks (nulls, types, outliers, formats, duplicates)
- ML layer: TF-IDF cosine similarity for semantic duplicate detection,
  fuzzy string matching for inconsistent naming in categorical fields
- These are classical ML/NLP techniques — not LLMs. Chosen because they're
  deterministic, fast, free, and auditable (you can explain every detection).
  An LLM layer could be added on top for ambiguous cases.

Pipeline:
1. Profile the dataset (column types, nulls, unique counts)
2. Rule-based checks: missing values, type mismatches, outliers, format issues
3. ML-assisted checks: TF-IDF semantic duplicates, fuzzy naming inconsistencies
4. Generate audit log with categorized issues and suggested fixes
5. Auto-apply fixes and output clean dataset + before/after summary
"""

import os
import re
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from rapidfuzz import fuzz, process

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Step 1: Data Profiling
# ---------------------------------------------------------------------------
def profile_dataset(df):
    """Generate a detailed profile of the dataset."""
    profile = {}
    for col in df.columns:
        profile[col] = {
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isna().sum()),
            "null_pct": round(df[col].isna().mean() * 100, 2),
            "unique_count": int(df[col].nunique()),
            "sample_values": [str(v) for v in df[col].dropna().head(5).tolist()],
        }
        if pd.api.types.is_numeric_dtype(df[col]):
            profile[col].update({
                "mean": round(float(df[col].mean()), 2) if not df[col].isna().all() else None,
                "std": round(float(df[col].std()), 2) if not df[col].isna().all() else None,
                "min": float(df[col].min()) if not df[col].isna().all() else None,
                "max": float(df[col].max()) if not df[col].isna().all() else None,
            })
    return profile


# ---------------------------------------------------------------------------
# Step 2: Rule-Based Checks
# ---------------------------------------------------------------------------
def check_missing_values(df):
    """Flag columns with significant missing values."""
    issues = []
    for col in df.columns:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            pct = n_missing / len(df) * 100
            severity = "HIGH" if pct > 30 else "MEDIUM" if pct > 10 else "LOW"
            issues.append({
                "type": "MISSING",
                "column": col,
                "severity": severity,
                "detail": f"{n_missing} missing values ({pct:.1f}%)",
                "suggested_fix": "impute_median" if pd.api.types.is_numeric_dtype(df[col]) else "impute_mode",
            })
    return issues


def check_type_mismatches(df):
    """Detect columns that look numeric but are stored as strings."""
    issues = []
    for col in df.select_dtypes(include=["object"]).columns:
        sample = df[col].dropna().head(100)
        numeric_count = sum(1 for v in sample if re.match(r"^-?\d+\.?\d*$", str(v).strip()))
        if len(sample) > 0 and numeric_count > len(sample) * 0.8:
            issues.append({
                "type": "TYPE_MISMATCH",
                "column": col,
                "severity": "MEDIUM",
                "detail": f"Column appears numeric but stored as text ({numeric_count}/{len(sample)} samples are numbers)",
                "suggested_fix": "convert_to_numeric",
            })
    return issues


def check_outliers(df):
    """Detect statistical outliers using IQR method.

    Using 1.5x IQR (standard Tukey fence) because this is transactional data
    where outliers are likely real data entry errors, not natural variation.
    For sensor data you'd use 3x IQR to be more conservative.
    """
    issues = []
    for col in df.select_dtypes(include=[np.number]).columns:
        values = df[col].dropna()
        if len(values) < 10:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        outliers = values[(values < lower) | (values > upper)]

        if len(outliers) > 0:
            pct = len(outliers) / len(values) * 100
            issues.append({
                "type": "OUTLIER",
                "column": col,
                "severity": "HIGH" if pct > 5 else "LOW",
                "detail": f"{len(outliers)} outliers ({pct:.1f}%) outside [{lower:.2f}, {upper:.2f}]",
                "suggested_fix": "clip_to_iqr_bounds",
                "bounds": {"lower": float(lower), "upper": float(upper)},
            })
    return issues


def check_format_inconsistencies(df):
    """Detect inconsistent formats using regex patterns."""
    issues = []

    for col in df.select_dtypes(include=["object"]).columns:
        sample = df[col].dropna().astype(str)
        if len(sample) == 0:
            continue

        total_checked = min(200, len(sample))
        sample_head = sample.head(total_checked)

        # Check for mixed case patterns
        has_upper = sum(1 for v in sample_head if v == v.upper() and v != v.lower())
        has_lower = sum(1 for v in sample_head if v == v.lower() and v != v.upper())
        has_title = sum(1 for v in sample_head if v == v.title())

        cases = {"UPPER": has_upper, "lower": has_lower, "Title": has_title}
        dominant = max(cases, key=lambda k: cases[k])

        if cases[dominant] < total_checked * 0.8 and total_checked > 10:
            issues.append({
                "type": "FORMAT",
                "column": col,
                "severity": "LOW",
                "detail": f"Mixed case formats: UPPER={has_upper}, lower={has_lower}, Title={has_title}",
                "suggested_fix": f"standardize_to_{dominant.lower()}",
            })

        # Check for leading/trailing whitespace
        has_whitespace = sum(1 for v in sample_head if v != v.strip())
        if has_whitespace > 0:
            issues.append({
                "type": "FORMAT",
                "column": col,
                "severity": "LOW",
                "detail": f"{has_whitespace} values with leading/trailing whitespace",
                "suggested_fix": "strip_whitespace",
            })

    return issues


def check_duplicate_rows(df):
    """Check for exact duplicate rows."""
    n_dupes = df.duplicated().sum()
    issues = []
    if n_dupes > 0:
        issues.append({
            "type": "DUPLICATE",
            "column": "ALL_ROWS",
            "severity": "MEDIUM",
            "detail": f"{n_dupes} exact duplicate rows ({n_dupes/len(df)*100:.1f}%)",
            "suggested_fix": "drop_duplicates",
        })
    return issues


def check_email_format(df):
    """Validate email columns."""
    issues = []
    email_cols = [c for c in df.columns if "email" in c.lower()]
    pattern = re.compile(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    for col in email_cols:
        invalid = df[col].dropna().apply(lambda x: not bool(pattern.match(str(x))))
        n_invalid = invalid.sum()
        if n_invalid > 0:
            issues.append({
                "type": "FORMAT",
                "column": col,
                "severity": "MEDIUM",
                "detail": f"{n_invalid} invalid email formats",
                "suggested_fix": "flag_invalid_emails",
            })
    return issues


def check_negative_prices(df):
    """Check for negative prices or quantities."""
    issues = []
    for col in df.columns:
        if any(kw in col.lower() for kw in ["price", "cost", "quantity", "amount"]):
            if pd.api.types.is_numeric_dtype(df[col]):
                n_neg = (df[col] < 0).sum()
                if n_neg > 0:
                    issues.append({
                        "type": "OUTLIER",
                        "column": col,
                        "severity": "HIGH",
                        "detail": f"{n_neg} negative values in {col}",
                        "suggested_fix": "flag_negative_values",
                    })
    return issues


# ---------------------------------------------------------------------------
# Step 3: ML-Assisted Checks (FREE — no API)
# ---------------------------------------------------------------------------
def ml_detect_semantic_duplicates(df):
    """Use TF-IDF + cosine similarity to find semantically similar text entries.

    TF-IDF captures word importance relative to the corpus, so "120V" and "120 Volts"
    would score high similarity even though exact string match would miss it.
    Cosine similarity threshold of 0.7 was chosen empirically — below that we get
    too many false positives from common product description boilerplate.
    """
    issues = []
    text_cols = [c for c in df.select_dtypes(include=["object"]).columns
                 if df[c].nunique() > 5 and df[c].notna().sum() > 10]

    for col in text_cols[:3]:  # Limit to 3 columns
        values = df[col].dropna().astype(str).unique()
        if len(values) < 3 or len(values) > 5000:
            continue

        # TF-IDF vectorization
        try:
            vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")
            tfidf_matrix = vectorizer.fit_transform(values)
            sim_matrix = cosine_similarity(tfidf_matrix)

            # Find pairs with high similarity (but not identical)
            similar_pairs = []
            for i in range(len(values)):
                for j in range(i + 1, min(i + 50, len(values))):
                    if 0.7 < sim_matrix[i, j] < 1.0:
                        similar_pairs.append((values[i], values[j], round(float(sim_matrix[i, j]), 3)))

            if similar_pairs:
                issues.append({
                    "type": "AI_SEMANTIC_DUPLICATE",
                    "column": col,
                    "severity": "MEDIUM",
                    "detail": f"{len(similar_pairs)} semantically similar value pairs detected via TF-IDF",
                    "sample_pairs": similar_pairs[:5],
                    "suggested_fix": "review_and_merge_similar_values",
                })
        except Exception as e:
            print(f"  TF-IDF check failed for {col}: {e}")

    return issues


def ml_detect_naming_inconsistencies(df):
    """Use fuzzy matching to detect inconsistent naming in categorical columns."""
    issues = []
    cat_cols = [c for c in df.select_dtypes(include=["object"]).columns
                if 2 < df[c].nunique() < 200]

    for col in cat_cols[:5]:
        unique_vals = df[col].dropna().unique().tolist()
        if len(unique_vals) < 3:
            continue

        # Find fuzzy matches among unique values
        inconsistencies = []
        seen = set()
        for val in unique_vals:
            if val in seen:
                continue
            matches = process.extract(str(val), [str(v) for v in unique_vals],
                                      scorer=fuzz.token_sort_ratio, limit=5)
            group = []
            for match_val, score, _ in matches:
                if 75 < score < 100 and match_val != str(val):
                    group.append(match_val)
                    seen.add(match_val)

            if group:
                inconsistencies.append({
                    "original": str(val),
                    "similar_to": group,
                    "suggested_standard": str(val),  # Keep the first one as standard
                })
                seen.add(val)

        if inconsistencies:
            issues.append({
                "type": "AI_NAMING",
                "column": col,
                "severity": "MEDIUM",
                "detail": f"{len(inconsistencies)} naming inconsistency groups in '{col}'",
                "groups": inconsistencies[:10],
                "suggested_fix": "standardize_similar_names",
            })

    return issues


def generate_ai_summary(df, all_issues):
    """Generate a data quality summary using statistical analysis (no API needed)."""
    issue_counts = {}
    for issue in all_issues:
        t = issue["type"]
        issue_counts[t] = issue_counts.get(t, 0) + 1

    total_nulls = df.isna().sum().sum()
    total_cells = df.shape[0] * df.shape[1]
    completeness = (1 - total_nulls / total_cells) * 100

    high_severity = sum(1 for i in all_issues if i.get("severity") == "HIGH")
    medium_severity = sum(1 for i in all_issues if i.get("severity") == "MEDIUM")

    # Score: start at 10, deduct for issues
    score = 10.0
    score -= min(3, high_severity * 0.5)
    score -= min(2, medium_severity * 0.2)
    score -= min(2, (100 - completeness) / 20)
    score = max(1, round(score, 1))

    summary_lines = [
        f"Data Quality Score: {score}/10",
        f"",
        f"- Dataset completeness: {completeness:.1f}% ({int(total_nulls):,} null values across {total_cells:,} cells)",
        f"- Issues found: {len(all_issues)} total ({high_severity} HIGH, {medium_severity} MEDIUM)",
        f"- Issue breakdown: {json.dumps(issue_counts)}",
        f"- Most critical: Fix {'missing values and outliers' if high_severity > 0 else 'format inconsistencies'} first",
        f"- Recommendation: {'Significant cleaning needed' if score < 6 else 'Moderate cleaning needed' if score < 8 else 'Minor cleanup only'}",
    ]

    return "\n".join(summary_lines)


# ---------------------------------------------------------------------------
# Step 4: Auto-Apply Fixes
# ---------------------------------------------------------------------------
def apply_fixes(df, issues):
    """Apply suggested fixes from the audit."""
    fixes_applied = []

    for issue in issues:
        col = issue.get("column")
        if col and col not in df.columns:
            continue

        fix = issue.get("suggested_fix", "")

        if fix == "strip_whitespace" and col:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace("nan", np.nan)
            fixes_applied.append(f"Stripped whitespace from {col}")

        elif fix == "convert_to_numeric" and col:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            fixes_applied.append(f"Converted {col} to numeric")

        elif fix == "impute_median" and col:
            if pd.api.types.is_numeric_dtype(df[col]):
                median_val = df[col].median()
                if pd.notna(median_val):
                    df[col] = df[col].fillna(median_val)
                    fixes_applied.append(f"Filled {col} missing with median ({median_val:.2f})")

        elif fix == "impute_mode" and col:
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val.iloc[0])
                fixes_applied.append(f"Filled {col} missing with mode ({mode_val.iloc[0]})")

        elif fix == "clip_to_iqr_bounds" and "bounds" in issue and col:
            bounds = issue["bounds"]
            before_outliers = int(((df[col] < bounds["lower"]) | (df[col] > bounds["upper"])).sum())
            df[col] = df[col].clip(lower=bounds["lower"], upper=bounds["upper"])
            fixes_applied.append(f"Clipped {before_outliers} outliers in {col}")

        elif fix == "drop_duplicates":
            before = len(df)
            df = df.drop_duplicates()
            fixes_applied.append(f"Removed {before - len(df)} exact duplicate rows")

        elif fix.startswith("standardize_to_") and col:
            target_case = fix.replace("standardize_to_", "")
            if target_case == "title":
                df[col] = df[col].astype(str).str.title()
                df[col] = df[col].replace("Nan", np.nan)
                fixes_applied.append(f"Standardized {col} to title case")
            elif target_case == "lower":
                df[col] = df[col].astype(str).str.lower()
                df[col] = df[col].replace("nan", np.nan)
                fixes_applied.append(f"Standardized {col} to lowercase")

    return df, fixes_applied


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------
def run_pipeline():
    print("=" * 60)
    print("AI-POWERED DATA QUALITY AUDITOR")
    print("(Using ML/NLP — no paid APIs)")
    print("=" * 60)

    # Load data — merge both CSVs if available
    print("\n[1/6] Loading dataset...")
    csv_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])
    if not csv_files:
        raise FileNotFoundError(f"No CSV files in {DATA_DIR}")

    dfs = {}
    for f in csv_files:
        dfs[f] = pd.read_csv(os.path.join(DATA_DIR, f), low_memory=False)
        print(f"  Loaded: {f} ({len(dfs[f])} rows, {len(dfs[f].columns)} cols)")

    # Use the larger/more interesting file as primary, or merge if they share a key
    if len(dfs) == 2 and "customers.csv" in dfs and "transactions.csv" in dfs:
        # Merge on customer_id for a richer dataset
        df = dfs["transactions.csv"].merge(dfs["customers.csv"], on="customer_id", how="left")
        print(f"  Merged transactions + customers: {len(df)} rows, {len(df.columns)} cols")
    else:
        df = list(dfs.values())[0]

    df_original = df.copy()

    # Profile
    print("\n[2/6] Profiling dataset...")
    profile = profile_dataset(df)
    for col, info in list(profile.items())[:5]:
        print(f"  {col}: {info['dtype']}, {info['null_pct']}% null, {info['unique_count']} unique")
    print(f"  ... {len(profile)} columns total")

    # Rule-based checks
    print("\n[3/6] Running rule-based quality checks...")
    all_issues = []

    missing_issues = check_missing_values(df)
    print(f"  Missing value issues: {len(missing_issues)}")
    all_issues.extend(missing_issues)

    type_issues = check_type_mismatches(df)
    print(f"  Type mismatch issues: {len(type_issues)}")
    all_issues.extend(type_issues)

    outlier_issues = check_outliers(df)
    print(f"  Outlier issues: {len(outlier_issues)}")
    all_issues.extend(outlier_issues)

    format_issues = check_format_inconsistencies(df)
    print(f"  Format issues: {len(format_issues)}")
    all_issues.extend(format_issues)

    dup_issues = check_duplicate_rows(df)
    print(f"  Duplicate issues: {len(dup_issues)}")
    all_issues.extend(dup_issues)

    email_issues = check_email_format(df)
    print(f"  Email format issues: {len(email_issues)}")
    all_issues.extend(email_issues)

    neg_issues = check_negative_prices(df)
    print(f"  Negative value issues: {len(neg_issues)}")
    all_issues.extend(neg_issues)

    # ML-assisted checks (FREE)
    print("\n[4/6] Running ML-assisted quality checks...")
    semantic_issues = ml_detect_semantic_duplicates(df)
    print(f"  Semantic duplicate issues: {len(semantic_issues)}")
    all_issues.extend(semantic_issues)

    naming_issues = ml_detect_naming_inconsistencies(df)
    print(f"  Naming inconsistency issues: {len(naming_issues)}")
    all_issues.extend(naming_issues)

    # Summary
    print("\n[5/6] Generating quality summary...")
    ai_summary = generate_ai_summary(df, all_issues)
    print(f"\n{ai_summary}\n")

    # Apply fixes
    print("\n[6/6] Applying fixes...")
    df_clean, fixes_applied = apply_fixes(df, all_issues)
    for fix in fixes_applied:
        print(f"  {fix}")

    # Before/after
    nulls_before = int(df_original.isna().sum().sum())
    nulls_after = int(df_clean.isna().sum().sum())
    null_reduction = round((1 - nulls_after / max(nulls_before, 1)) * 100, 1) if nulls_before > 0 else 0
    print(f"\n  Nulls: {nulls_before:,} -> {nulls_after:,} ({null_reduction}% reduction)")
    print(f"  Rows: {len(df_original):,} -> {len(df_clean):,}")

    # Save outputs
    df_clean.to_csv(os.path.join(OUTPUT_DIR, "cleaned_data.csv"), index=False)
    print(f"\n  Saved: output/cleaned_data.csv")

    # Save audit log
    audit_log = {
        "datasets": csv_files,
        "rows": len(df_original),
        "issues_found": len(all_issues),
        "issues": all_issues,
        "fixes_applied": fixes_applied,
        "before_after": {
            "rows": {"before": len(df_original), "after": len(df_clean)},
            "nulls": {"before": nulls_before, "after": nulls_after},
            "null_reduction_pct": null_reduction,
        },
        "summary": ai_summary,
        "profile": profile,
    }
    with open(os.path.join(OUTPUT_DIR, "audit_log.json"), "w") as f:
        json.dump(audit_log, f, indent=2, default=str)
    print(f"  Saved: output/audit_log.json")

    # Save readable report
    _save_readable_report(audit_log)

    return df_clean, audit_log


def _save_readable_report(audit_log):
    """Save a markdown-formatted audit report."""
    lines = []
    lines.append("# Data Quality Audit Report")
    lines.append("")
    lines.append(f"**Datasets:** {', '.join(audit_log['datasets'])}")
    lines.append(f"**Rows:** {audit_log['rows']:,}")
    lines.append(f"**Issues found:** {audit_log['issues_found']}")
    lines.append("")

    lines.append("## Quality Assessment")
    lines.append(audit_log["summary"])
    lines.append("")

    # Issues by type
    lines.append("## Issues by Type")
    by_type = {}
    for issue in audit_log["issues"]:
        t = issue["type"]
        by_type.setdefault(t, []).append(issue)

    for issue_type, issues in sorted(by_type.items()):
        lines.append(f"\n### {issue_type} ({len(issues)} issues)")
        for issue in issues[:5]:
            lines.append(f"- **{issue['column']}** [{issue['severity']}]: {issue['detail']}")
            lines.append(f"  - Fix: {issue['suggested_fix']}")
        if len(issues) > 5:
            lines.append(f"- ... and {len(issues) - 5} more")

    # Fixes applied
    lines.append("\n## Fixes Applied")
    for fix in audit_log["fixes_applied"]:
        lines.append(f"- {fix}")

    # Before/after
    ba = audit_log["before_after"]
    lines.append("\n## Before vs After")
    lines.append(f"- **Null values:** {ba['nulls']['before']:,} -> {ba['nulls']['after']:,} ({ba['null_reduction_pct']}% reduction)")
    lines.append(f"- **Rows:** {ba['rows']['before']:,} -> {ba['rows']['after']:,}")

    report_path = os.path.join(OUTPUT_DIR, "audit_report.md")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved: output/audit_report.md")


if __name__ == "__main__":
    run_pipeline()
