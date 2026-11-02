"""Leakage-free basket-completion evaluation for JuliaKarpienia/master.

Run privately, in the root of your local repository:

    python leakage_free_basket_completion.py \
      --data data/preprocessed/cleaned_retail_data_2022_2025.csv \
      --out evaluation_private/basket_completion \
      --max-test-baskets-per-group 500

IMPORTANT:
- Use confidential transactions ONLY locally; do not commit data or output to GitHub.
- The source is the cleaned dataset from your thesis; raw exports are not required.
- This is a NEW basket-completion experiment, not a reproduction of your
  original 100-customer future-invoice leave-one-out experiment.
- INPUT NOTE: The cleaned CSV was previously winsorized on the full period.
  This experiment excludes all test invoices from model fitting and feature
  selection, but a fully strict train-only preprocessing pipeline should also
  refit any learned preprocessing thresholds (e.g., winsorization) on TRAIN.
- Complete INVOICES are split by date BEFORE fitting ANY model. On held-out
  invoices, ONE product is masked and used solely as the ground truth.
- B2B Item-CF uses quantity-weighted customer-product interactions (>=2
  training invoices), matching your Notebook 1. At recommendation time it uses
  the visible basket, not the selected customer's direct purchase history.
- Separate B2B/B2C Apriori rules are trained on the training invoices only.
- Popularity is the number of unique TRAINING invoices containing a product.
- Hybrid interleaves Item-CF and B2B rules, falling back to popularity only if
  neither produced candidates; B2C uses rules with popularity fallback.
- Empty recommendations are scored as zero and their rate is reported.
- Percentages are per evaluated eligible test basket in each customer group.

Outputs: metrics.csv, availability.csv, split_audit.txt, training_info.txt
Each is aggregate; none contains customer names, invoice numbers or products.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.preprocessing import normalize

ANON = "SPRZEDAZ FISKALNA"
REQUIRED = {"CustomerName", "InvoiceID", "SaleDate", "ProductName", "Quantity"}


def load_clean_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"InvoiceID": "string", "CustomerName": "string", "ProductName": "string"})
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    df["SaleDate"] = pd.to_datetime(df["SaleDate"], errors="coerce")
    df["Quantity"] = pd.to_numeric(df["Quantity"], errors="coerce")
    df = df.dropna(subset=list(REQUIRED)).copy()
    df["CustomerName"] = df["CustomerName"].astype(str).str.strip()
    df["InvoiceID"] = df["InvoiceID"].astype(str).str.strip()
    df["ProductName"] = df["ProductName"].astype(str).str.strip()
    if df.empty or (df["Quantity"] <= 0).any():
        raise ValueError("Expected nonempty cleaned data with strictly positive Quantity.")

    # An invoice is identified by BOTH owner and invoice number, preventing
    # accidental merging if the same invoice number appears for two customers.
    df["_basket_id"] = pd.factorize(
        pd.MultiIndex.from_frame(df[["CustomerName", "InvoiceID"]]), sort=False
    )[0].astype(np.int64)
    df["_segment"] = np.where(df["CustomerName"] == ANON, "B2C", "B2B")
    return df


def chronological_invoice_split(df: pd.DataFrame, test_fraction: float):
    if not 0 < test_fraction < 1:
        raise ValueError("--test-fraction must be strictly between 0 and 1")
    meta = df.groupby("_basket_id", sort=False).agg(
        date=("SaleDate", "max"), segment=("_segment", "first")
    )
    meta = meta.sort_values(["date"], kind="stable")
    split_at = max(1, int(len(meta) * (1 - test_fraction)))
    if split_at >= len(meta):
        raise ValueError("Not enough invoices for the requested test fraction")
    # Keep invoices from the same calendar day together; test is strictly later.
    cutoff_day = meta.iloc[split_at]["date"].normalize()
    train_ids = set(meta.index[meta.date < cutoff_day].tolist())
    test_ids = set(meta.index[meta.date >= cutoff_day].tolist())
    train = df[df._basket_id.isin(train_ids)].copy()
    test = df[df._basket_id.isin(test_ids)].copy()
    if train.empty or test.empty:
        raise ValueError("Chronological split resulted in an empty partition")
    assert train_ids.isdisjoint(test_ids)
    assert train.SaleDate.max() < test.SaleDate.min(), "Unexpected temporal overlap"
    return train, test, cutoff_day


def fit_item_cf(train_b2b: pd.DataFrame):
    invoice_counts = train_b2b.groupby("CustomerName")["_basket_id"].nunique()
    eligible = invoice_counts[invoice_counts >= 2].index
    data = train_b2b[train_b2b.CustomerName.isin(eligible)]
    if data.empty:
        return csr_matrix((0, 0)), {}, [], 0
    # Like the thesis's pivot_table(index=CustomerName, columns=ProductName,
    # values=Quantity, aggfunc=sum), but without constructing a dense table.
    cp = (data.groupby(["CustomerName", "ProductName"], sort=True, observed=True)["Quantity"]
              .sum().reset_index())
    c_codes, customers = pd.factorize(cp.CustomerName, sort=True)
    p_codes, products = pd.factorize(cp.ProductName, sort=True)
    matrix = coo_matrix(
        (cp.Quantity.to_numpy(dtype=np.float32), (p_codes, c_codes)),
        shape=(len(products), len(customers)), dtype=np.float32
    ).tocsr()
    matrix = normalize(matrix, norm="l2", axis=1, copy=False)
    product_names = products.astype(str).tolist()
    return matrix, {name: i for i, name in enumerate(product_names)}, product_names, len(customers)


def recommend_item_cf(items, item_user, index, product_names, k):
    if not len(product_names):
        return []
    scores = np.zeros(len(product_names), dtype=np.float32)
    for item in items:
        pos = index.get(item)
        if pos is not None:
            sim = (item_user[pos] @ item_user.T).toarray().ravel()
            np.maximum(scores, sim, out=scores)
    excluded = set(items)
    available = np.flatnonzero(scores > 0)
    ranked = sorted(
        (int(i) for i in available if product_names[i] not in excluded),
        key=lambda i: (-float(scores[i]), product_names[i])
    )
    return [product_names[i] for i in ranked[:k]]


def make_sparse_basket_matrix(data: pd.DataFrame, minimum_support: float):
    """Build sparse boolean invoice x product matrix using training data only.

    An item that does not satisfy singleton support can never occur in a frequent
    itemset; prefiltering such items is lossless for Apriori at this support.
    """
    if data.empty:
        return pd.DataFrame(), 0
    inv_prod = data[["_basket_id", "ProductName"]].drop_duplicates()
    n_baskets = data._basket_id.nunique()
    counts = inv_prod.ProductName.value_counts()
    kept = counts[counts >= math.ceil(minimum_support * n_baskets - 1e-12)].index
    inv_prod = inv_prod[inv_prod.ProductName.isin(kept)]
    if inv_prod.empty:
        return pd.DataFrame(), n_baskets
    inv_codes, invoice_ids = pd.factorize(inv_prod._basket_id, sort=True)
    product_codes, products = pd.factorize(inv_prod.ProductName, sort=True)
    sparse_matrix = coo_matrix(
        (np.ones(len(inv_prod), dtype=np.uint8), (inv_codes, product_codes)),
        shape=(len(invoice_ids), len(products)), dtype=np.uint8
    ).tocsr()
    sparse_matrix.data[:] = 1
    # Preserve ALL invoices as denominator, even invoices with no surviving SKUs.
    all_invoice_ids = np.sort(data._basket_id.unique())
    row_lookup = pd.Index(all_invoice_ids).get_indexer(invoice_ids)
    mapped_coo = sparse_matrix.tocoo()
    complete = coo_matrix(
        (mapped_coo.data, (row_lookup[mapped_coo.row], mapped_coo.col)),
        shape=(len(all_invoice_ids), len(products)), dtype=np.uint8
    ).tocsr()
    complete.data[:] = 1
    matrix = pd.DataFrame.sparse.from_spmatrix(
        complete.astype(bool), index=all_invoice_ids, columns=products.astype(str)
    )
    return matrix, n_baskets


def fit_rules(data: pd.DataFrame, minimum_support: float, minimum_lift: float):
    basket, n_baskets = make_sparse_basket_matrix(data, minimum_support)
    if basket.shape[1] == 0:
        return [], (n_baskets, 0, 0)
    frequent = apriori(
        basket, min_support=minimum_support, use_colnames=True, low_memory=True
    )
    if frequent.empty:
        return [], (n_baskets, basket.shape[1], 0)
    rules_df = association_rules(frequent, metric="lift", min_threshold=minimum_lift)
    rules = []
    for r in rules_df.itertuples(index=False):
        ant = frozenset(r.antecedents)
        cons = frozenset(r.consequents)
        if ant and cons:
            rules.append((ant, cons, float(r.lift)))
    rules.sort(key=lambda r: -r[2])
    return rules, (n_baskets, basket.shape[1], len(rules))


def recommend_ar(items, rules, k):
    basket = set(items)
    candidates = {}
    for antecedents, consequents, lift in rules:
        if antecedents.issubset(basket):
            for product in consequents - basket:
                candidates[product] = max(candidates.get(product, float("-inf")), lift)
    return sorted(candidates, key=lambda p: (-candidates[p], p))[:k]


def fit_popularity(data):
    counts = data[["_basket_id", "ProductName"]].drop_duplicates().ProductName.value_counts()
    return sorted(counts.index.astype(str).tolist(), key=lambda p: (-int(counts[p]), p))


def recommend_popularity(items, popular, k):
    excluded = set(items)
    return [p for p in popular if p not in excluded][:k]


def hybrid_interleave(cf, ar, pop, k):
    out, seen = [], set()
    from_cf, from_ar = iter(cf), iter(ar)
    cf_done = ar_done = False
    while len(out) < k:
        if not cf_done:
            try:
                p = next(from_cf)
                if p not in seen:
                    out.append(p)
                    seen.add(p)
            except StopIteration:
                cf_done = True
        if len(out) >= k:
            break
        if not ar_done:
            try:
                p = next(from_ar)
                if p not in seen:
                    out.append(p)
                    seen.add(p)
            except StopIteration:
                ar_done = True
        if cf_done and ar_done:
            break
    return out[:k] if out else pop[:k]


def score(recs, hidden, k):
    top = list(dict.fromkeys(recs[:k]))
    hit = int(hidden in top)
    precision = hit / k
    recall = float(hit)  # exactly ONE relevant hidden product per basket
    ndcg = 1 / math.log2(top.index(hidden) + 2) if hit else 0.0
    return hit, precision, recall, ndcg


def choose_masked_test_cases(test, segment, max_test_baskets, seed):
    segment_df = test.loc[test._segment == segment]
    products = segment_df.groupby("_basket_id")["ProductName"].agg(lambda x: sorted(set(x)))
    eligible = products[products.map(len) >= 2]
    rng = np.random.default_rng(seed)
    basket_ids = eligible.index.to_numpy()
    if max_test_baskets and len(basket_ids) > max_test_baskets:
        basket_ids = rng.choice(basket_ids, size=max_test_baskets, replace=False)
    cases = []
    for bid in sorted(basket_ids):
        ps = eligible.loc[bid]
        mask_idx = int(rng.integers(0, len(ps)))
        hidden = ps[mask_idx]
        observed = [p for p in ps if p != hidden]
        cases.append((observed, hidden))
    return cases, int(len(eligible)), int(segment_df._basket_id.nunique())


def run(data_path, output_dir, fraction, max_test, seed, k):
    df = load_clean_data(data_path)
    train, test, cutoff = chronological_invoice_split(df, fraction)
    train_b2b = train.loc[train._segment == "B2B"]
    train_b2c = train.loc[train._segment == "B2C"]

    print(f"Training ends before {cutoff.date()}; test invoices are on/after this date.")
    print(f"Train: {train._basket_id.nunique():,} invoices; Test: {test._basket_id.nunique():,} invoices.")
    print("Fitting B2B Item-CF from TRAINING transactions only...")
    item_user, product_index, product_names, n_cf_customers = fit_item_cf(train_b2b)
    print("Mining B2B Apriori rules from TRAINING invoices only...")
    rules_b2b, stats_b2b = fit_rules(train_b2b, 0.005, 1.2)
    print("Mining B2C Apriori rules from TRAINING invoices only...")
    rules_b2c, stats_b2c = fit_rules(train_b2c, 0.001, 1.2)
    popularity = fit_popularity(train)

    rows = []
    availability_rows = []
    sampling_stats = {}
    for segment in ("B2B", "B2C"):
        cases, eligible, all_test_invoices = choose_masked_test_cases(
            test, segment, max_test, seed + (0 if segment == "B2B" else 10000)
        )
        sampling_stats[segment] = (len(cases), eligible, all_test_invoices)
        if not cases:
            print(f"No eligible test baskets for {segment} (need >=2 distinct items)")
            continue
        print(f"Evaluating {segment}: {len(cases):,} masked test baskets...")
        rules = rules_b2b if segment == "B2B" else rules_b2c
        flags = {"cf_available": 0, "ar_available": 0, "both_available": 0,
                 "only_cf": 0, "only_ar": 0, "popularity_fallback": 0}
        scores = {m: [] for m in (["Popularity", "Association Rules", "Item-CF", "Hybrid"] if segment == "B2B"
                                else ["Popularity", "Association Rules", "Hybrid"])}

        for observed, hidden in cases:
            pop = recommend_popularity(observed, popularity, k)
            ar = recommend_ar(observed, rules, k)
            cf = (recommend_item_cf(observed, item_user, product_index, product_names, k)
                  if segment == "B2B" else [])
            hybrid = hybrid_interleave(cf, ar, pop, k) if segment == "B2B" else (ar or pop)

            flags["cf_available"] += int(bool(cf))
            flags["ar_available"] += int(bool(ar))
            flags["both_available"] += int(bool(cf) and bool(ar))
            flags["only_cf"] += int(bool(cf) and not ar)
            flags["only_ar"] += int(bool(ar) and not cf)
            flags["popularity_fallback"] += int(not cf and not ar)

            for name, recs in (("Popularity", pop), ("Association Rules", ar),
                               ("Item-CF", cf), ("Hybrid", hybrid)):
                if name in scores:
                    scores[name].append((int(bool(recs)),) + score(recs, hidden, k))

        n = len(cases)
        for model_name, values in scores.items():
            a = np.asarray(values, dtype=float)  # has_recs, hits, precision, recall, ndcg
            rows.append({
                "segment": segment, "model": model_name, "n_test_baskets": n,
                "n_with_recs": int(a[:, 0].sum()), "availability": float(a[:, 0].mean()),
                "n_hits": int(a[:, 1].sum()), f"Precision@{k}": float(a[:, 2].mean()),
                f"Recall@{k}": float(a[:, 3].mean()), f"NDCG@{k}": float(a[:, 4].mean()),
            })
        record = {"segment": segment, "n_test_baskets": n}
        record.update({name + "_count": int(num) for name, num in flags.items()})
        record.update({name + "_pct": round(100 * num / n, 3) for name, num in flags.items()})
        availability_rows.append(record)

    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_dir / "metrics.csv", index=False, float_format="%.6f")
    pd.DataFrame(availability_rows).to_csv(output_dir / "availability.csv", index=False)
    audit_lines = [
        "DESIGN: chronological invoice-level split; one randomly masked product per eligible TEST invoice",
        "The same evaluated baskets are used for all models WITHIN each customer segment.",
        f"seed: {seed}", f"K: {k}", f"train_fraction_approx: {1 - fraction:.3f}",
        f"test_start_date: {cutoff.date()}",
        f"train_max_date: {train.SaleDate.max().isoformat()}",
        f"test_min_date: {test.SaleDate.min().isoformat()}",
        f"train_rows: {len(train)}", f"test_rows: {len(test)}",
        f"train_invoices: {train._basket_id.nunique()}",
        f"test_invoices: {test._basket_id.nunique()}",
        f"test_invoices_not_in_training: {set(train._basket_id.unique()).isdisjoint(set(test._basket_id.unique()))}",
    ]
    for segment, (evaluated, eligible, all_invoices) in sampling_stats.items():
        audit_lines += [f"{segment}_test_invoices_total: {all_invoices}",
                        f"{segment}_test_invoices_eligible_2plus_items: {eligible}",
                        f"{segment}_test_baskets_evaluated: {evaluated}"]
    (output_dir / "split_audit.txt").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
    training_info = [
        "Item-CF: B2B customers with >=2 TRAIN invoices; quantity-weighted customer-product interactions",
        f"B2B Item-CF trained customers: {n_cf_customers}",
        f"B2B Item-CF products: {len(product_names)}",
        "B2B AR: Apriori min_support=0.005, min_lift=1.2",
        f"B2B AR train invoices, frequent products, generated rules: {stats_b2b}",
        "B2C AR: Apriori min_support=0.001, min_lift=1.2",
        f"B2C AR train invoices, frequent products, generated rules: {stats_b2c}",
        "Popularity: global TRAIN invoice frequency", "Hybrid: B2B round-robin CF/AR; B2C AR; fallback popularity if neither available",
        "Rule matching: whole antecedent set must be present in observed basket",
        "Basket completion: mask one uniformly random distinct product per eligible test basket",
        "Relevant set size is one, so Recall@K = HitRate@K",
        "No customer names, invoice IDs, or individual products are exported.",
        "Preprocessing caveat: source cleaned CSV may contain full-period winsorization caps; refit caps on train only for strict end-to-end leakage isolation.",
    ]
    (output_dir / "training_info.txt").write_text("\n".join(training_info) + "\n", encoding="utf-8")
    print("\nBASKET COMPLETION METRICS\n" + pd.DataFrame(rows).to_string(index=False))
    print(f"\nSaved aggregate reports in {output_dir}")
    return pd.DataFrame(rows), pd.DataFrame(availability_rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", type=Path, required=True, help="Private CLEANED POS CSV (never upload it to public GitHub)")
    ap.add_argument("--out", type=Path, default=Path("evaluation_private/basket_completion"))
    ap.add_argument("--test-fraction", type=float, default=0.20,
                    help="Approximate chronological fraction of invoices reserved for testing (default: 0.20)")
    ap.add_argument("--max-test-baskets-per-group", type=int, default=500,
                    help="Maximum sampled eligible B2B and B2C test baskets each; 0 evaluates all (default: 500)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k", type=int, default=5)
    args = ap.parse_args()
    if args.max_test_baskets_per_group < 0 or args.k < 1:
        ap.error("--max-test-baskets-per-group must be >=0 and --k must be >=1")
    run(args.data, args.out, args.test_fraction, args.max_test_baskets_per_group, args.seed, args.k)


if __name__ == "__main__":
    main()
