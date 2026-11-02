#!/usr/bin/env python3
"""Strict, train-only preprocessing + leakage-controlled basket-completion evaluation.

For the original JuliaKarpienia/master retail thesis repository.  Uses the
EXISTING `leakage_free_basket_completion.py` and `paired_basket_significance.py`
located in this same directory, reusing their model, masking, and bootstrap code.

Run LOCALLY inside the thesis project (do not upload raw/customer-level data):

    python strict_preprocessing_basket_eval.py \
      --raw-dir data/raw \
      --out evaluation_private/strict_preprocessing \
      --seeds 42 43 44

Algorithm:
  1) Merge the eight cp1250 raw POS exports (no use of pre-winsorized CSV).
  2) Apply only fixed, row-wise parsing and the thesis's deterministic filters.
  3) Split whole customer+invoice keys chronologically (roughly 80/20).
  4) Fit the 99th-percentile Quantity and SalesValue caps on TRAIN only.
  5) Apply TRAIN caps to both partitions (NO estimation from TEST).
  6) Fit B2B Item-CF, B2B/B2C Apriori, and Popularity from TRAIN only.
  7) Mask one product in each multi-item test invoice for seeds 42/43/44.
  8) Report metrics, availability, paired customer-cluster bootstrap CIs.

All reports contain aggregate statistics ONLY: NO customer names, invoice IDs,
product names, or individual transactions are written to disk.

The script should yield a split near 2025-02-01 on the original private data;
compare its counts to your prior split audit before interpreting changes.
"""
from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ANON = "SPRZEDAZ FISKALNA"
RENAME = {
    "#": "RowID", "Ilość magazynowa": "Quantity", "Wartość nabycia": "PurchaseValue",
    "Wartość sprzedaży": "SalesValue", "Nazwa kontrahenta": "CustomerName",
    "Skrot nazwy kontrahenta": "CustomerShortName", "Indeks": "ProductID",
    "Nazwa": "ProductName", "Data nabycia": "PurchaseDate",
    "Nr dok. obrotowego": "DocNumber", "Data sprzedaży": "SaleDate",
    "Nr faktury": "InvoiceID", "Operator": "Operator",
    "Forma płatności": "PaymentMethod", "Zysk": "Profit",
    "Narzut": "Markup", "Marża": "Margin",
}
CRITICAL = ["SaleDate", "InvoiceID", "ProductID", "CustomerName", "ProductName", "Quantity", "SalesValue"]


def import_companion(path: Path, module_name: str):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path.name}. Put it beside this script (same directory)."
        )
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def parse_polish_number(series: pd.Series) -> pd.Series:
    # Numerical input passes through. Supports Polish '1 750,00', '1.750,00',
    # and normal decimal-dot values. Dot is removed as a thousands separator
    # only when a comma decimal separator is present.
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    s = series.astype("string").str.replace("\u00a0", "", regex=False)
    s = s.str.replace(" ", "", regex=False).str.strip()
    has_comma = s.str.contains(",", regex=False, na=False)
    s = s.where(~has_comma, s.str.replace(".", "", regex=False))
    s = s.str.replace(",", ".", regex=False)
    return pd.to_numeric(s, errors="coerce")


def parse_sale_dates(series: pd.Series) -> pd.Series:
    # 'mixed' works on pandas >=2 and supports both ISO and Polish-style dates.
    # Fallback handles older versions.
    try:
        return pd.to_datetime(series, errors="coerce", format="mixed", dayfirst=True)
    except (TypeError, ValueError):
        return pd.to_datetime(series, errors="coerce", dayfirst=True)


def read_unfitted_raw(raw_dir: Path, pattern: str) -> tuple[pd.DataFrame, dict]:
    files = sorted(raw_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No raw files found in {raw_dir} matching {pattern!r}")
    frames = []
    for path in files:
        frame = pd.read_csv(path, encoding="cp1250", low_memory=False)
        frame = frame.rename(columns=RENAME)
        missing = set(CRITICAL) - set(frame.columns)
        if missing:
            raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    original_n = len(df)

    # Fixed parsing/quality rules from original Notebook 1. None are fitted.
    for col in ("CustomerName", "InvoiceID", "ProductID", "ProductName"):
        df[col] = df[col].astype("string").str.strip().replace("", pd.NA)
    df["SaleDate"] = parse_sale_dates(df["SaleDate"])
    for col in ("Quantity", "SalesValue"):
        df[col] = parse_polish_number(df[col])

    df = df.dropna(subset=CRITICAL).copy()
    n_after_missing = len(df)
    df = df[(df["Quantity"] > 0) & (df["SalesValue"] > 0)].copy()
    n_after_positive = len(df)
    if df.empty:
        raise ValueError("All rows were removed during deterministic preprocessing")
    df["CustomerName"] = df["CustomerName"].astype(str)
    df["InvoiceID"] = df["InvoiceID"].astype(str)
    df["ProductName"] = df["ProductName"].astype(str)

    # Invoice identity aligns with the earlier basket evaluation implementation.
    df["_basket_id"] = pd.factorize(
        pd.MultiIndex.from_frame(df[["CustomerName", "InvoiceID"]]), sort=False
    )[0].astype(np.int64)
    df["_segment"] = np.where(df["CustomerName"] == ANON, "B2C", "B2B")
    audit = {
        "raw_files_found": len(files),
        "raw_rows_total": original_n,
        "rows_after_dropping_missing_required_fields": n_after_missing,
        "rows_after_removing_nonpositive_quantity_or_sales": n_after_positive,
        "rows_removed_deterministic_preprocessing": original_n - n_after_positive,
        "unique_customer_invoice_keys": int(df["_basket_id"].nunique()),
    }
    return df, audit


def train_only_caps(train: pd.DataFrame, test: pd.DataFrame):
    quantity_cap = float(train["Quantity"].quantile(0.99))
    sales_cap = float(train["SalesValue"].quantile(0.99))
    # The same fitted caps are applied to both partitions, never fit on test.
    train = train.copy()
    test = test.copy()
    for data in (train, test):
        data["Quantity"] = data["Quantity"].clip(upper=quantity_cap)
        data["SalesValue"] = data["SalesValue"].clip(upper=sales_cap)
    return train, test, quantity_cap, sales_cap


def evaluate_seed(m, paired, test, segment, model_state, seed, k, bootstrap):
    item_user, product_index, product_names, b2b_rules, b2c_rules, popularity = model_state
    cases, eligible, all_invoices = m.choose_masked_test_cases(
        test, segment, max_test_baskets=0,
        seed=seed + (0 if segment == "B2B" else 10000),
    )
    if not cases:
        raise ValueError(f"No eligible test baskets for {segment}")
    n = len(cases)
    # Same case order as the companion module: ascending internal basket ID.
    eligible_ids = (
        test.loc[test._segment == segment]
        .groupby("_basket_id")["ProductName"].nunique()
    )
    ids = np.sort(eligible_ids[eligible_ids >= 2].index.to_numpy())
    if len(ids) != n:
        raise AssertionError("Masking/case order mismatch")
    owner_lookup = test.drop_duplicates("_basket_id").set_index("_basket_id")["CustomerName"]
    groups = owner_lookup.loc[ids].to_numpy(dtype=str) if segment == "B2B" else ids.astype(str)

    model_names = ["Popularity", "Association Rules", "Item-CF", "Hybrid"] if segment == "B2B" else ["Popularity", "Association Rules", "Hybrid"]
    hits = {label: np.zeros(n, dtype=np.int8) for label in model_names}
    ndcg = {label: np.zeros(n, dtype=np.float64) for label in model_names}
    availability = {label: np.zeros(n, dtype=np.int8) for label in model_names}
    flags = {key: 0 for key in ("cf_available", "ar_available", "both_available", "only_cf", "only_ar", "popularity_fallback")}
    rules = b2b_rules if segment == "B2B" else b2c_rules

    for i, (observed, hidden) in enumerate(cases):
        pop = m.recommend_popularity(observed, popularity, k)
        ar = m.recommend_ar(observed, rules, k)
        cf = m.recommend_item_cf(observed, item_user, product_index, product_names, k) if segment == "B2B" else []
        hybrid = m.hybrid_interleave(cf, ar, pop, k) if segment == "B2B" else (ar or pop)
        candidates = {"Popularity": pop, "Association Rules": ar, "Item-CF": cf, "Hybrid": hybrid}
        flags["cf_available"] += int(bool(cf))
        flags["ar_available"] += int(bool(ar))
        flags["both_available"] += int(bool(cf) and bool(ar))
        flags["only_cf"] += int(bool(cf) and not ar)
        flags["only_ar"] += int(bool(ar) and not cf)
        flags["popularity_fallback"] += int(not cf and not ar)
        for name in model_names:
            rec = candidates[name]
            availability[name][i] = int(bool(rec))
            result = m.score(rec, hidden, k)
            hits[name][i] = int(result[0])
            ndcg[name][i] = float(result[3])

    metrics = []
    for name in model_names:
        h = hits[name]
        metrics.append({
            "seed": seed, "segment": segment, "model": name,
            "n_test_baskets": n, "n_with_recs": int(availability[name].sum()),
            "availability": float(availability[name].mean()), "n_hits": int(h.sum()),
            f"Precision@{k}": float(h.mean() / k),
            f"Recall@{k}": float(h.mean()),
            f"NDCG@{k}": float(ndcg[name].mean()),
        })

    av = {"seed": seed, "segment": segment, "n_test_baskets": n}
    for key, count in flags.items():
        av[key + "_count"] = count
        av[key + "_pct"] = round(100 * count / n, 3)

    comparisons = [("Hybrid", "Item-CF"), ("Hybrid", "Popularity"), ("Hybrid", "Association Rules")]
    comparisons = [(x, y) for x, y in comparisons if y in model_names]
    paired_rows = []
    for j, (a, b) in enumerate(comparisons):
        aa, bb = hits[a], hits[b]
        only_a = int(((aa == 1) & (bb == 0)).sum())
        only_b = int(((aa == 0) & (bb == 1)).sum())
        both = int(((aa == 1) & (bb == 1)).sum())
        neither = int(((aa == 0) & (bb == 0)).sum())
        diff = aa.astype(float) - bb.astype(float)
        low, high = paired.ci_cluster_paired(diff, groups, bootstrap, seed + 1000 + j)
        discordant = only_a + only_b
        p = float(binomtest(min(only_a, only_b), discordant, 0.5).pvalue) if discordant else 1.0
        paired_rows.append({
            "seed": seed, "segment": segment, "comparison": f"{a} vs {b}",
            "n_test_baskets": n, "n_resampling_units": len(np.unique(groups)),
            "hybrid_hits": int(aa.sum()), "baseline_hits": int(bb.sum()),
            "hybrid_hit_rate_pct": float(aa.mean() * 100),
            "baseline_hit_rate_pct": float(bb.mean() * 100),
            "hit_rate_difference_pp": float(diff.mean() * 100),
            "cluster_bootstrap_CI95_lower_pp": float(low * 100),
            "cluster_bootstrap_CI95_upper_pp": float(high * 100),
            "both_hit": both, "only_hybrid_hit": only_a,
            "only_baseline_hit": only_b, "neither_hit": neither,
            "mcnemar_exact_p_exploratory": p,
        })
    return metrics, av, paired_rows, eligible, all_invoices


def run(args):
    companion_dir = Path(__file__).resolve().parent
    m = import_companion(companion_dir / "leakage_free_basket_completion.py", "basket_eval_strict_companion")
    paired = import_companion(companion_dir / "paired_basket_significance.py", "basket_paired_strict_companion")
    print("Reading original POS exports locally (only aggregate reports will be saved)...")
    df, preprocessing_audit = read_unfitted_raw(args.raw_dir, args.pattern)
    train, test, cutoff = m.chronological_invoice_split(df, args.test_fraction)
    # Full-period caps below are for AUDIT COMPARISON ONLY; never used for fitting.
    full_quantity_p99_diagnostic = float(df["Quantity"].quantile(0.99))
    full_salesvalue_p99_diagnostic = float(df["SalesValue"].quantile(0.99))
    train, test, q_cap, s_cap = train_only_caps(train, test)

    # Defensive checks: training never sees test invoices, and test has later dates.
    assert set(train._basket_id.unique()).isdisjoint(test._basket_id.unique())
    assert train.SaleDate.max() < test.SaleDate.min()
    assert len(train) + len(test) == len(df)

    print(f"Chronological cutoff: {cutoff.date()}")
    print(f"Fitted training-only caps: Quantity={q_cap:.6f}, SalesValue={s_cap:.6f}")
    print("Fitting B2B Item-CF and Apriori models on TRAIN only...")
    b2b = train.loc[train._segment == "B2B"]
    b2c = train.loc[train._segment == "B2C"]
    item_user, product_index, product_names, n_cf_customers = m.fit_item_cf(b2b)
    b2b_rules, b2b_stats = m.fit_rules(b2b, 0.005, 1.2)
    b2c_rules, b2c_stats = m.fit_rules(b2c, 0.001, 1.2)
    popularity = m.fit_popularity(train)
    state = (item_user, product_index, product_names, b2b_rules, b2c_rules, popularity)
    metrics_rows, availability_rows, paired_rows, report_counts = [], [], [], {}
    for seed in args.seeds:
        for segment in ("B2B", "B2C"):
            print(f"Evaluating {segment} masked baskets, seed={seed}...")
            x, y, z, eligible, total = evaluate_seed(m, paired, test, segment, state, seed, args.k, args.bootstrap)
            metrics_rows.extend(x)
            availability_rows.append(y)
            paired_rows.extend(z)
            report_counts[segment] = (eligible, total)

    args.out.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metrics_rows)
    av = pd.DataFrame(availability_rows)
    paired = pd.DataFrame(paired_rows)
    metrics.to_csv(args.out / "metrics_by_seed.csv", index=False, float_format="%.6f")
    av.to_csv(args.out / "availability_by_seed.csv", index=False, float_format="%.6f")
    paired.to_csv(args.out / "paired_comparison_by_seed.csv", index=False, float_format="%.8g")
    summary = (
        metrics.groupby(["segment", "model"], sort=False)
        .agg(n_test_baskets=("n_test_baskets", "first"),
             hit_rate_mean=(f"Recall@{args.k}", "mean"),
             hit_rate_min=(f"Recall@{args.k}", "min"),
             hit_rate_max=(f"Recall@{args.k}", "max"),
             ndcg_mean=(f"NDCG@{args.k}", "mean"),
             availability_mean=("availability", "mean"))
        .reset_index()
    )
    summary.to_csv(args.out / "metrics_summary.csv", index=False, float_format="%.6f")

    audit = [
        "DESIGN: Original raw POS CSVs; deterministic parsing/filters; chronological invoice split; fit 99th percentile caps on TRAIN ONLY; mask 1 product per test basket.",
        "No test invoices used for Item-CF/AR/Popularity training or data-dependent caps.",
        "Only aggregate reports written, no customer or invoice records.",
        f"seeds: {','.join(map(str,args.seeds))}", f"K: {args.k}",
        f"train_fraction_approx: {1 - args.test_fraction:.3f}",
        f"cutoff_day: {cutoff.date()}",
        f"train_last_date: {train.SaleDate.max().isoformat()}",
        f"test_first_date: {test.SaleDate.min().isoformat()}",
        f"train_rows: {len(train)}", f"test_rows: {len(test)}",
        f"train_invoices: {train._basket_id.nunique()}",
        f"test_invoices: {test._basket_id.nunique()}",
        "test_invoices_not_in_training: True",
        f"train_fitted_quantity_cap_p99: {q_cap:.8f}",
        f"train_fitted_salesvalue_cap_p99: {s_cap:.8f}",
        f"full_period_quantity_cap_p99_FOR_AUDIT_ONLY_NEVER_USED: {full_quantity_p99_diagnostic:.8f}",
        f"full_period_salesvalue_cap_p99_FOR_AUDIT_ONLY_NEVER_USED: {full_salesvalue_p99_diagnostic:.8f}",
    ]
    for key, val in preprocessing_audit.items():
        audit.append(f"{key}: {val}")
    for segment, (eligible, total) in report_counts.items():
        audit.append(f"{segment}_test_invoices_total: {total}")
        audit.append(f"{segment}_test_invoices_eligible_2plus_items: {eligible}")
    (args.out / "preprocessing_audit.txt").write_text("\n".join(audit) + "\n", encoding="utf-8")
    training_info = [
        "B2B Item-CF: quantity-weighted customer-product matrix, customers with >=2 TRAIN invoices",
        f"B2B Item-CF trained customers: {n_cf_customers}",
        f"B2B Item-CF products: {len(product_names)}",
        "B2B Apriori: min_support=0.005, min_lift=1.2",
        f"B2B Apriori (train invoices, frequent products, rules): {b2b_stats}",
        "B2C Apriori: min_support=0.001, min_lift=1.2",
        f"B2C Apriori (train invoices, frequent products, rules): {b2c_stats}",
        "Popularity: number of unique TRAIN invoices per product",
        "Hybrid: B2B interleave CF and AR, B2C AR; popularity fallback only if neither source returns candidates",
        "All-case mean scores: empty recommendation outputs scored 0",
        "McNemar p-values are exploratory; for B2B use customer-clustered bootstrap CIs as primary inference",
        "Seeds are sensitivity analyses on the SAME temporal split, not independent samples",
    ]
    (args.out / "training_info.txt").write_text("\n".join(training_info) + "\n", encoding="utf-8")
    print("\nSummary across seeds:\n", summary.to_string(index=False))
    print("\nSaved aggregate results to:", args.out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--pattern", type=str, default="[12]-202*.csv",
                        help="Match the eight original half-year files, e.g. 1-2022.csv")
    parser.add_argument("--out", type=Path, default=Path("evaluation_private/strict_preprocessing"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--bootstrap", type=int, default=2000)
    args = parser.parse_args()
    if args.k < 1 or args.bootstrap < 200 or not args.seeds or len(set(args.seeds)) != len(args.seeds):
        parser.error("k must be positive; bootstrap >=200; seeds must be distinct")
    run(args)


if __name__ == "__main__":
    main()
