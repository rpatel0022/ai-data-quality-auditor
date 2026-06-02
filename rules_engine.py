"""
Rules Engine
============
Standalone rule-based data quality checks that can run without AI.
Useful as a fallback or for quick checks.
"""

import re
import pandas as pd
import numpy as np


class RulesEngine:
    """Configurable rule-based data quality checker."""

    def __init__(self):
        self.rules = []
        self.results = []

    def add_rule(self, name, check_fn, severity="MEDIUM"):
        self.rules.append({"name": name, "check": check_fn, "severity": severity})

    def run(self, df):
        self.results = []
        for rule in self.rules:
            try:
                issues = rule["check"](df)
                for issue in issues:
                    issue["rule"] = rule["name"]
                    issue["severity"] = issue.get("severity", rule["severity"])
                self.results.extend(issues)
            except Exception as e:
                self.results.append({
                    "rule": rule["name"],
                    "type": "ERROR",
                    "detail": f"Rule failed: {e}",
                    "severity": "HIGH",
                })
        return self.results

    def summary(self):
        by_type = {}
        for r in self.results:
            t = r.get("type", "UNKNOWN")
            by_type[t] = by_type.get(t, 0) + 1
        return by_type


# Pre-built rules
def email_format_rule(df):
    """Check email columns for valid format."""
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
                "detail": f"{n_invalid} invalid email formats",
            })
    return issues


def negative_price_rule(df):
    """Check for negative values in price columns."""
    issues = []
    price_cols = [c for c in df.columns if "price" in c.lower() or "cost" in c.lower()]
    for col in price_cols:
        if pd.api.types.is_numeric_dtype(df[col]):
            n_neg = (df[col] < 0).sum()
            if n_neg > 0:
                issues.append({
                    "type": "OUTLIER",
                    "column": col,
                    "detail": f"{n_neg} negative values in price column",
                    "severity": "HIGH",
                })
    return issues


def duplicate_rows_rule(df):
    """Check for exact duplicate rows."""
    n_dupes = df.duplicated().sum()
    issues = []
    if n_dupes > 0:
        issues.append({
            "type": "DUPLICATE",
            "column": "ALL",
            "detail": f"{n_dupes} exact duplicate rows ({n_dupes/len(df)*100:.1f}%)",
        })
    return issues


def create_default_engine():
    """Create a rules engine with all default rules."""
    engine = RulesEngine()
    engine.add_rule("email_format", email_format_rule, "MEDIUM")
    engine.add_rule("negative_price", negative_price_rule, "HIGH")
    engine.add_rule("duplicate_rows", duplicate_rows_rule, "MEDIUM")
    return engine
