#!/usr/bin/env python3
"""Paired significance diagnostics for the private basket-completion experiment.

Place beside leakage_free_basket_completion.py. Fits models on chronological
TRAIN invoices, replicates that script's masking for a fixed seed, and compares
matched hit/miss outcomes across models on the same TEST baskets.

Outputs AGGREGATE paired_comparison.csv and audit.txt only. No customer names,
invoice identifiers, or products are written to disk.

B2B bootstrap resamples *customers* (all their eligible baskets together),
accounting for within-customer dependence. B2C bootstrap resamples invoices.
McNemar exact p-values are exploratory (they assume independent basket pairs;
B2B repeat customers violate that assumption). Use cluster-bootstrap CI as the
primary uncertainty summary; seeds 43/44 are sensitivity analyses, not
independent samples.
"""
from __future__ import annotations
import argparse
import importlib.util
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import binomtest


def load_original(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Required companion script not found: {path}")
    spec = importlib.util.spec_from_file_location("basket_eval_local", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def ci_cluster_paired(diff: np.ndarray, owner_ids: np.ndarray, replicates: int, seed: int):
    """Cluster-bootstrap of mean(hit_A - hit_B) using resampled owners."""
    frame = pd.DataFrame({'group': owner_ids, 'delta': diff})
    clusters = frame.groupby('group', sort=False).delta.agg(['sum', 'size'])
    cluster_sums = clusters['sum'].to_numpy(dtype=np.float64)
    cluster_sizes = clusters['size'].to_numpy(dtype=np.float64)
    n = len(clusters)
    rng = np.random.default_rng(seed)
    sampled_deltas = np.empty(replicates)
    for start in range(0, replicates, 100):
        batch = min(100, replicates-start)
        idx = rng.integers(0, n, size=(batch, n))
        sampled_deltas[start:start+batch] = cluster_sums[idx].sum(axis=1)/cluster_sizes[idx].sum(axis=1)
    return np.quantile(sampled_deltas, [0.025, 0.975])


def evaluate_one_segment(m, test, segment, rules, item_user, index, products, popularity, seed, k, reps):
    # The original function sorts eligible _basket_id values before masking.
    cases, eligible, n_test = m.choose_masked_test_cases(
        test, segment, max_test_baskets=0,
        seed=seed + (0 if segment == 'B2B' else 10000)
    )
    eligible_ids = (
        test.loc[test._segment == segment]
        .groupby('_basket_id')['ProductName'].nunique()
    )
    ids = np.sort(eligible_ids[eligible_ids >= 2].index.to_numpy())
    if len(ids) != len(cases):
        raise AssertionError("Case alignment failure; check companion script's masking function")
    owners = (
        test.drop_duplicates('_basket_id').set_index('_basket_id')['CustomerName']
    )
    # For anonymous B2C purchases, cluster per invoice, NOT the shared B2C name.
    group_ids = (owners.loc[ids].to_numpy(dtype=str) if segment == 'B2B'
                 else ids.astype(str))
    names = ['Popularity', 'Association Rules', 'Item-CF', 'Hybrid'] if segment == 'B2B' else ['Popularity', 'Association Rules', 'Hybrid']
    hits = {name: np.empty(len(cases), dtype=np.int8) for name in names}
    for i,(visible, hidden) in enumerate(cases):
        pop = m.recommend_popularity(visible, popularity, k)
        ar = m.recommend_ar(visible, rules, k)
        cf = m.recommend_item_cf(visible, item_user, index, products, k) if segment == 'B2B' else []
        hyb = m.hybrid_interleave(cf, ar, pop, k) if segment == 'B2B' else (ar or pop)
        candidates = {'Popularity':pop, 'Association Rules':ar,'Item-CF':cf,'Hybrid':hyb}
        for name in names:
            hits[name][i] = int(hidden in candidates[name][:k])

    comparisons = [('Hybrid','Item-CF'), ('Hybrid','Popularity'), ('Hybrid','Association Rules')]
    comparisons = [(a,b) for a,b in comparisons if b in names]
    rows = []
    for comp_i,(a,b) in enumerate(comparisons):
        aa,bb=hits[a],hits[b]
        n10 = int(((aa==1)&(bb==0)).sum())
        n01 = int(((aa==0)&(bb==1)).sum())
        n11 = int(((aa==1)&(bb==1)).sum())
        n00 = int(((aa==0)&(bb==0)).sum())
        discordants=n10+n01
        p = float(binomtest(min(n10,n01), discordants, 0.5, alternative='two-sided').pvalue) if discordants else 1.0
        delta=aa.astype(float)-bb.astype(float)
        lo,hi=ci_cluster_paired(delta,group_ids,reps,seed+1000+comp_i)
        rows.append({
            'segment':segment,'seed':seed,'comparison':f'{a} vs {b}',
            'n_test_baskets':len(cases),'n_resampling_units':len(np.unique(group_ids)),
            'hybrid_hits' if a=='Hybrid' else 'model_A_hits':int(aa.sum()),
            'baseline_hits':int(bb.sum()),
            'hybrid_hit_rate_pct' if a=='Hybrid' else 'model_A_hit_rate_pct':round(float(aa.mean()*100),3),
            'baseline_hit_rate_pct':round(float(bb.mean()*100),3),
            'hit_rate_difference_pp':round(float(delta.mean()*100),3),
            'cluster_bootstrap_CI95_lower_pp':round(float(lo*100),3),
            'cluster_bootstrap_CI95_upper_pp':round(float(hi*100),3),
            'both_hit':n11,'only_hybrid_hit':n10,'only_baseline_hit':n01,'neither_hit':n00,
            'mcnemar_exact_p_exploratory':p,
        })
    return rows, {name:int(hits[name].sum()) for name in names},len(cases)


def run(args):
    m = load_original(Path(__file__).resolve().parent/'leakage_free_basket_completion.py')
    df=m.load_clean_data(args.data)
    train,test,cutoff=m.chronological_invoice_split(df,args.test_fraction)
    print('Fitting B2B Item-CF on TRAIN...')
    item_user, index, products, n_customers=m.fit_item_cf(train[train._segment=='B2B'])
    print('Fitting B2B Association Rules on TRAIN...')
    rules_b2b,_=m.fit_rules(train[train._segment=='B2B'],0.005,1.2)
    print('Fitting B2C Association Rules on TRAIN...')
    rules_b2c,_=m.fit_rules(train[train._segment=='B2C'],0.001,1.2)
    popularity=m.fit_popularity(train)
    output_rows=[]
    audit=[f'test_cutoff_date: {cutoff.date()}',f'seed: {args.seed}',f'K: {args.k}',
           f'bootstrap_resamples: {args.bootstrap}',
           'B2B bootstrap: customer-cluster resampling (all eligible baskets per customer)',
           'B2C bootstrap: invoice-level resampling',
           'McNemar exact p-values: exploratory only; repeated baskets per B2B customer are dependent.',
           'Same test invoices and mask as leakage_free_basket_completion.py',
           'Output contains no raw transactions, customer identities, invoice IDs or product names.',
           'Caveat: original cleaned CSV may have full-data preprocessing thresholds.']
    for segment, rules in [('B2B',rules_b2b),('B2C',rules_b2c)]:
        print(f'Paired evaluation {segment}...')
        rows, totals, n=evaluate_one_segment(m,test,segment,rules,item_user,index,products,popularity,
                                              args.seed,args.k,args.bootstrap)
        output_rows.extend(rows)
        audit.append(f'{segment}: n={n}, per-model hits={totals}')
    args.out.mkdir(parents=True,exist_ok=True)
    table=pd.DataFrame(output_rows)
    table.to_csv(args.out/'paired_comparison.csv',index=False,float_format='%.6f')
    (args.out/'audit.txt').write_text('\n'.join(audit)+'\n',encoding='utf8')
    print('\n'+table.to_string(index=False))
    print('\nSaved: '+str(args.out/'paired_comparison.csv'))

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--out',type=Path,default=Path('evaluation_private/paired_seed42'))
    ap.add_argument('--test-fraction',type=float,default=.20)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--k',type=int,default=5)
    ap.add_argument('--bootstrap',type=int,default=2000)
    args=ap.parse_args()
    if args.bootstrap < 200 or args.k < 1:
        ap.error('--bootstrap must be >= 200 and --k >=1')
    run(args)
