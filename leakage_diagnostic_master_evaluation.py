"""Leakage-free B2B evaluation and controlled leakage diagnosis for JuliaKarpienia/master.

Run locally, with the confidential cleaned transaction CSV. Never upload the
raw/cleaned data or identifiable evaluation logs to a public GitHub repository.

Usage:
  pip install pandas numpy scipy scikit-learn mlxtend
  python leakage_free_master_evaluation.py \
      --data data/preprocessed/cleaned_retail_data_2022_2025.csv \
      --out evaluation_private

Methodology:
  1) Select the same 100 B2B customers as the original notebook (seed=42).
  2) Hold out their latest invoices from ALL subsequent model fitting.
  3) Rebuild the B2B user-item matrix (>=2 training invoices per customer),
     item-item cosine similarities on demand, B2B Apriori rules, and popularity.
  4) Evaluate the same four recommendation methods on prior purchases.
  5) Report standard metrics averaged over ALL evaluated customers, plus a
     separate legacy-style (non-empty-recommendation-only) summary and coverage.

Cautions:
  * This reproduces the overall notebook-2 protocol, not a strictly historical
    all-customers-before-test-date protocol. Other customers' later transactions
    may be present in training; consider a global chronological split separately.
  * Uses binary invoice-item baskets for rules and quantity-based customer-item
    data for CF, like the thesis notebooks.
  * For mathematically correct recall, duplicate products in one invoice are
    deduplicated in the ground truth. The original notebook did not do this.
  * No customer names, test invoices or product-level exports are saved.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.preprocessing import normalize
from mlxtend.frequent_patterns import apriori, association_rules

ANON = "SPRZEDAZ FISKALNA"
COLUMNS = ["CustomerName", "InvoiceID", "SaleDate", "ProductName", "Quantity"]


def make_basket_matrix(rows: pd.DataFrame) -> pd.DataFrame:
    """Sparse Boolean invoice-product matrix compatible with mlxtend."""
    inv = pd.Categorical(rows["InvoiceID"].astype(str))
    prod = pd.Categorical(rows["ProductName"].astype(str))
    coo = coo_matrix(
        (np.ones(len(rows), dtype=np.uint8), (inv.codes, prod.codes)),
        shape=(len(inv.categories), len(prod.categories)),
        dtype=np.uint8,
    )
    csr = coo.tocsr()
    csr.data[:] = 1  # repeated invoice-item lines represent one basket interaction
    return pd.DataFrame.sparse.from_spmatrix(
        csr.astype(bool), index=inv.categories, columns=prod.categories
    )


def mine_b2b_rules(train_b2b: pd.DataFrame) -> dict[str, list[tuple[str, float]]]:
    basket = make_basket_matrix(train_b2b)
    print(f"B2B training baskets: {basket.shape[0]:,} x {basket.shape[1]:,}")
    frequent = apriori(
        basket, min_support=0.005, use_colnames=True, low_memory=True
    )
    if frequent.empty:
        return {}
    rules = association_rules(frequent, metric="lift", min_threshold=1.2)
    rules = rules.sort_values(["lift", "confidence"], ascending=[False, False])
    # Mirrors original notebook-2's flattening of rules to a source -> target.
    lookup: dict[str, list[tuple[str, float]]] = {}
    for _, rule in rules.iterrows():
        if not rule["antecedents"] or not rule["consequents"]:
            continue
        source = next(iter(rule["antecedents"]))
        target = next(iter(rule["consequents"]))
        lookup.setdefault(source, []).append((target, float(rule["lift"])))
    print(f"Rules: {len(rules):,}")
    return lookup


def build_item_cf(train_b2b: pd.DataFrame) -> tuple[csr_matrix, dict[str, int], list[str]]:
    customer_invoices = train_b2b.groupby("CustomerName")["InvoiceID"].nunique()
    eligible = customer_invoices[customer_invoices >= 2].index
    cf_rows = train_b2b[train_b2b["CustomerName"].isin(eligible)]
    ui = cf_rows.pivot_table(
        index="CustomerName", columns="ProductName", values="Quantity",
        aggfunc="sum", fill_value=0,
    )
    product_names = list(ui.columns)
    # Rows = products, columns = B2B customers. L2 normalize for cosine similarity.
    item_user = csr_matrix(ui.to_numpy(dtype=np.float32).T)
    normalized = normalize(item_user, norm="l2", axis=1, copy=False)
    print(f"Training CF matrix: {len(ui):,} customers x {len(product_names):,} products")
    return normalized, {p: i for i, p in enumerate(product_names)}, product_names


def recommend_cf(items: list[str], normed: csr_matrix,
                 product_index: dict[str, int], product_names: list[str], k: int) -> list[str]:
    scores = np.zeros(len(product_names), dtype=np.float32)
    # Only compute cosine similarities for query items; avoids an enormous
    # dense 12,000 x 12,000 similarity matrix on most laptops.
    for item in items:
        idx = product_index.get(item)
        if idx is None:
            continue
        row = (normed[idx] @ normed.T).toarray().ravel()
        np.maximum(scores, row, out=scores)
    excluded = set(items)
    ranked = sorted(
        (i for i, score in enumerate(scores) if score > 0 and product_names[i] not in excluded),
        key=lambda i: (-float(scores[i]), product_names[i]),
    )
    return [product_names[i] for i in ranked[:k]]


def recommend_ar(items: list[str], rules_lookup: dict[str, list[tuple[str, float]]], k: int) -> list[str]:
    scores: dict[str, float] = {}
    excluded = set(items)
    for item in items:
        for target, lift in rules_lookup.get(item, []):
            if target not in excluded and target not in scores:
                scores[target] = lift
    return sorted(scores, key=lambda p: -scores[p])[:k]


def hybrid_interleave(cf: list[str], ar: list[str], pop: list[str], k: int) -> list[str]:
    out, seen = [], set()
    cf_iter, ar_iter = iter(cf), iter(ar)
    cf_done = ar_done = False
    while len(out) < k:
        if not cf_done:
            try:
                p = next(cf_iter)
                if p not in seen:
                    out.append(p); seen.add(p)
            except StopIteration:
                cf_done = True
        if len(out) >= k:
            break
        if not ar_done:
            try:
                p = next(ar_iter)
                if p not in seen:
                    out.append(p); seen.add(p)
            except StopIteration:
                ar_done = True
        if cf_done and ar_done:
            break
    # Original notebook applied fallback ONLY when BOTH source lists were empty.
    return out[:k] if out else pop[:k]


def scores(recs: list[str], relevant: set[str], k: int) -> tuple[float, float, float]:
    ranked = list(dict.fromkeys(recs[:k]))
    hits = sum(p in relevant for p in ranked)
    precision = hits / k
    recall = hits / len(relevant) if relevant else 0.
    dcg = sum(1 / np.log2(i + 2) for i, p in enumerate(ranked) if p in relevant)
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return precision, recall, dcg / idcg if idcg else 0.


def evaluate(data: pd.DataFrame, sample_n=100, seed=42, k=5, training_scope="train_only"):
    if training_scope not in {"train_only", "full_dataset_diagnostic"}:
        raise ValueError("Unsupported training_scope")
    missing = set(COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    data = data.copy()
    data["SaleDate"] = pd.to_datetime(data["SaleDate"], errors="coerce")
    data["Quantity"] = pd.to_numeric(data["Quantity"], errors="coerce")
    data = data.dropna(subset=COLUMNS).copy()
    data["CustomerName"] = data["CustomerName"].astype(str)
    data["InvoiceID"] = data["InvoiceID"].astype(str)
    data["ProductName"] = data["ProductName"].astype(str)
    if (data["Quantity"] <= 0).any():
        raise ValueError("Expected cleaned positive-quantity transactions.")

    # Notebook-2 groups baskets only by invoice number. Detect invoice-number
    # reuse across customers; this would require a safer composite basket ID.
    sharing = data.groupby("InvoiceID")["CustomerName"].nunique()
    if (sharing > 1).any():
        raise ValueError(
            f"Found {(sharing > 1).sum():,} invoice numbers shared by multiple customers. "
            "Check invoice identity before rerunning the split."
        )

    known = data[data.CustomerName != ANON]
    counts = known.groupby("CustomerName")["InvoiceID"].nunique()
    eligible = counts[counts >= 2].index.tolist()
    if len(eligible) < sample_n:
        print(f"Only {len(eligible)} eligible B2B customers; evaluating all.")
        chosen = eligible
    else:
        # Same numpy selection used in the thesis's Notebook 2.
        np.random.seed(seed)
        chosen = list(np.random.choice(eligible, sample_n, replace=False))

    test_pairs = []
    for c in chosen:
        cd = known[known.CustomerName == c].sort_values("SaleDate", kind="stable")
        last_inv = cd.InvoiceID.unique()[-1]
        test_pairs.append((c, last_inv))
    test_idx = pd.MultiIndex.from_tuples(test_pairs, names=["CustomerName", "InvoiceID"])
    all_idx = pd.MultiIndex.from_frame(data[["CustomerName", "InvoiceID"]])
    is_test = all_idx.isin(test_idx)
    train, test = data.loc[~is_test].copy(), data.loc[is_test].copy()

    assert not pd.MultiIndex.from_frame(train[["CustomerName", "InvoiceID"]]).isin(test_idx).any()
    assert len(set(test_pairs)) == len(chosen)
    print(f"Held out {len(chosen)} B2B invoices; train rows {len(train):,}, test rows {len(test):,}")

    # IMPORTANT: recommendation INPUT always uses customer history in train.
    # The optional leaky fit is a diagnostic counterfactual only, never a valid score.
    train_b2b = train[train.CustomerName != ANON]
    fit_data = train if training_scope == "train_only" else data
    fit_b2b = fit_data[fit_data.CustomerName != ANON]
    print(f"Fit mode: {training_scope}; fitting on {len(fit_data):,} transaction rows")
    normed, product_index, product_names = build_item_cf(fit_b2b)
    rules_lookup = mine_b2b_rules(fit_b2b)
    popularity = fit_data.groupby("ProductName")["InvoiceID"].nunique().sort_values(ascending=False).index.tolist()

    per_case = []
    for c, inv in test_pairs:
        historical = train_b2b.loc[train_b2b.CustomerName == c, "ProductName"].drop_duplicates().tolist()
        target = set(test.loc[(test.CustomerName == c) & (test.InvoiceID == inv), "ProductName"])
        if not historical or not target:
            continue
        pop = [p for p in popularity if p not in historical][:k]
        cf = recommend_cf(historical, normed, product_index, product_names, k)
        ar = recommend_ar(historical, rules_lookup, k)
        hybrid = hybrid_interleave(cf, ar, pop, k)
        for model, recs in (("Popularity", pop), ("Item-CF", cf), ("Assoc. Rules", ar), ("Hybrid", hybrid)):
            p, r, n = scores(recs, target, k)
            per_case.append({"case": len(per_case) // 4, "model": model,
                             "precision": p, "recall": r, "ndcg": n,
                             "hits": len(set(recs[:k]) & target),
                             "returned_candidates": bool(recs)})
    cases = pd.DataFrame(per_case)
    if cases.empty:
        raise ValueError("No eligible evaluation cases.")

    def summarize(group):
        all_stats = group[["precision", "recall", "ndcg"]].mean()
        nonempty = group[group.returned_candidates]
        return pd.Series({
            "n_test_cases": len(group), "n_with_recs": len(nonempty),
            "availability": len(nonempty) / len(group),
            "total_hits": int(group.hits.sum()),
            "n_hit_cases": int((group.hits > 0).sum()),
            "Precision@5_all": all_stats["precision"],
            "Recall@5_all": all_stats["recall"],
            "NDCG@5_all": all_stats["ndcg"],
            "Precision@5_nonempty_only": nonempty.precision.mean() if len(nonempty) else 0.,
            "Recall@5_nonempty_only": nonempty.recall.mean() if len(nonempty) else 0.,
            "NDCG@5_nonempty_only": nonempty.ndcg.mean() if len(nonempty) else 0.,
        })

    summary = cases.groupby("model").apply(summarize, include_groups=False)
    return summary, {"test_customers": len(chosen), "train_rows": len(train),
                     "test_rows": len(test), "train_invoices": train.InvoiceID.nunique(),
                     "test_invoices": test.InvoiceID.nunique(), "evaluated_cases": cases.case.nunique()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Local confidential cleaned CSV")
    parser.add_argument("--out", default="evaluation_private", help="Local output directory")
    parser.add_argument("--sample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diagnose-leakage", action="store_true", help=(
        "ALSO train on all invoices solely for a controlled leakage comparison. "
        "The full-data diagnostic is invalid for reporting predictive performance."))
    args = parser.parse_args()
    print("Loading local data (transaction/customer data never saved by this script)")
    data = pd.read_csv(args.data)
    summary, audit = evaluate(data, sample_n=args.sample, seed=args.seed, training_scope="train_only")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    summary.round(6).to_csv(out / "leakage_free_metrics.csv")
    (out / "split_audit.txt").write_text("\n".join(f"{k}: {v}" for k, v in audit.items()))
    print("\nLEAKAGE-FREE:\n", summary.round(5).to_string(), "\n", audit)
    if args.diagnose_leakage:
        print("\nRUNNING INVALID FULL-DATA FIT FOR CONTROLLED DIAGNOSTIC ONLY...")
        leaky, audit_leaky = evaluate(data, sample_n=args.sample, seed=args.seed,
                                       training_scope="full_dataset_diagnostic")
        assert audit_leaky == audit, "Cohort / split changed between experiments!"
        leaky.round(6).to_csv(out / "LEAKY_FULL_DATA_DIAGNOSTIC_NOT_FOR_PUBLICATION.csv")
        metrics = ["Precision@5_all", "Recall@5_all", "NDCG@5_all"]
        comp = pd.DataFrame(index=summary.index)
        for col in metrics:
            comp[f"{col}_train_only"] = summary[col]
            comp[f"{col}_invalid_full_data_fit"] = leaky[col]
            comp[f"{col}_inflation_from_leakage"] = leaky[col] - summary[col]
        comp["train_only_total_hits"] = summary["total_hits"]
        comp["invalid_full_data_total_hits"] = leaky["total_hits"]
        comp.round(6).to_csv(out / "controlled_leakage_comparison.csv")
        print("\nCONTROLLED COMPARISON (leaky fit is INVALID for publication):\n", comp.round(5).to_string())
    print(f"\nSaved aggregate results to: {out.absolute()}")
    print("Do not commit any confidential CSVs, customer names or pickle artifacts.")


if __name__ == "__main__":
    main()
