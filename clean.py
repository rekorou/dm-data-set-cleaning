#!/usr/bin/env python3
"""
IllustraMeta -- cleaning stage.

Implements ONLY the leak-safe operations from Section C.1 of the proposal:
stages 1 and 2, plus verification. Everything that learns a parameter from
the data -- imputation, scaling, encoding -- belongs in the sklearn Pipeline
in model.py, NOT here. Applying those to all 600 rows before the train/test
split would leak test information and inflate every reported metric.

  python clean.py --in data/raw/illustrameta_safebooru.csv \
                  --out data/processed/illustrameta_clean.csv

Prints a report you can quote directly in the write-up.
"""

import argparse
import sys

import pandas as pd

POST_URL = "https://safebooru.org/index.php?page=post&s=view&id="

# Excluded per Section 6. Reasons are printed so the log is self-documenting.
DROP_REASONS = {
    "_leak_tags_removed": "audit counter -- equals the class label by construction",
    "change_hour":        "time-derived; 4.7% class ID overlap makes it a recency detector",
    "change_day_of_week": "time-derived; same reason",
    "change_month":       "time-derived; same reason",
}

NON_MODELLING = ["post_id", "owner_hashed", "post_url"]


def rule(title):
    print(f"\n{title}\n{'-' * len(title)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", default="data/raw/illustrameta_safebooru.csv")
    ap.add_argument("--out", dest="dst", default="data/processed/illustrameta_clean.csv")
    args = ap.parse_args()

    df = pd.read_csv(args.src)
    n0, c0 = df.shape

    rule("INPUT")
    print(f"  {n0} records, {c0} attributes")
    print(f"  class balance: {dict(df.is_ai_generated.value_counts())}")

    # --- integrity checks (report, don't silently fix) ---------------------
    rule("INTEGRITY CHECKS")

    dupes = df.post_id.duplicated().sum()
    print(f"  duplicate post_id:            {dupes}")
    if dupes:
        df = df.drop_duplicates(subset="post_id")
        print(f"    -> removed, {len(df)} remain")

    bad_dims = ((df.image_width <= 0) | (df.image_height <= 0)).sum()
    print(f"  non-positive dimensions:      {bad_dims}")

    bad_ar = (~df.aspect_ratio.between(0.01, 100)).sum()
    print(f"  implausible aspect ratios:    {bad_ar}")

    bad_tags = (df.tag_count < 0).sum()
    print(f"  negative tag counts:          {bad_tags}")

    if bad_dims or bad_ar or bad_tags:
        print("  !! investigate before proceeding -- do not auto-drop")

    # --- stage 2: missingness indicator BEFORE any imputation -------------
    rule("MISSINGNESS INDICATOR")
    df["score_missing"] = df.score.isna().astype(int)
    ct = pd.crosstab(df.is_ai_generated, df.score_missing)
    print(ct.to_string())
    pct = df.score_missing.mean() * 100
    print(f"\n  score missing: {df.score.isna().sum()}/{len(df)} ({pct:.1f}%)")
    print("  derived before imputation -- imputing first would destroy this signal")

    # --- traceability ------------------------------------------------------
    df["post_url"] = POST_URL + df.post_id.astype(str)
    print(f"\n  post_url added for verification (excluded from modelling)")

    # --- stage 1: attribute exclusion --------------------------------------
    rule("ATTRIBUTE EXCLUSION")

    zero_var, near_zero = [], []
    for c in df.columns:
        if c in NON_MODELLING or c == "is_ai_generated":
            continue
        nun = df[c].nunique(dropna=True)
        if nun <= 1:
            zero_var.append(c)
        elif nun <= 3 and df[c].value_counts(dropna=True).iloc[0] / len(df) > 0.99:
            # Varies, but in so few records it cannot support a split.
            near_zero.append(c)

    for c in zero_var:
        print(f"  drop {c:<22} zero variance (single value across all records)")
    for c in near_zero:
        dom = df[c].value_counts().iloc[0]
        print(f"  drop {c:<22} near-zero variance ({dom}/{len(df)} records share one value)")

    for c, why in DROP_REASONS.items():
        if c in df.columns and c not in zero_var and c not in near_zero:
            print(f"  drop {c:<22} {why}")

    to_drop = set(zero_var) | set(near_zero) | (set(DROP_REASONS) & set(df.columns))
    df = df.drop(columns=list(to_drop))

    # --- leak scan ---------------------------------------------------------
    rule("LEAK SCAN")
    y = df.is_ai_generated
    leaks = []
    for c in df.columns:
        if c in NON_MODELLING + ["is_ai_generated"]:
            continue
        a = df.loc[y == 1, c].dropna()
        b = df.loc[y == 0, c].dropna()
        if a.empty or b.empty:
            continue

        if df[c].nunique(dropna=True) <= 20:
            # Discrete or categorical: disjoint value sets = leak.
            if not (set(a) & set(b)):
                leaks.append(c)
                print(f"  LEAK: {c} -- classes use disjoint value sets")
        else:
            # Continuous: set-disjointness is meaningless (random floats rarely
            # collide). A real leak means a single threshold separates them.
            if pd.api.types.is_numeric_dtype(df[c]) and (a.max() < b.min() or b.max() < a.min()):
                leaks.append(c)
                print(f"  LEAK: {c} -- a single threshold separates the classes")
    if not leaks:
        print("  clean -- no single attribute separates the classes")

    # --- output ------------------------------------------------------------
    predictors = [c for c in df.columns
                  if c not in NON_MODELLING + ["is_ai_generated"]]

    rule("OUTPUT")
    print(f"  {len(df)} records, {df.shape[1]} attributes")
    print(f"  {len(predictors)} predictors: {', '.join(sorted(predictors))}")
    print(f"  non-modelling (kept for traceability): {', '.join(NON_MODELLING)}")

    try:
        df.to_csv(args.dst, index=False)
        print(f"\n  written -> {args.dst}")
    except FileNotFoundError:
        print(f"\n  !! directory for {args.dst} does not exist -- mkdir it first")
        sys.exit(1)

    print("\nNEXT: imputation, encoding, and scaling go in the sklearn Pipeline")
    print("in model.py, fitted per training fold. Do NOT apply them here.")


if __name__ == "__main__":
    main()
