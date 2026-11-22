"""Step 8: offline GO enrichment by hypergeometric test, using GPL17077's own GO_ID column.
Background = all probes that survived preprocessing. No external servers needed."""
import pandas as pd, numpy as np, io
from scipy.stats import hypergeom

lines = open("data/GPL17077.txt", encoding="latin-1").read().splitlines()
s = next(i for i,l in enumerate(lines) if l.startswith("!platform_table_begin"))+1
e = next((i for i,l in enumerate(lines) if l.startswith("!platform_table_end")), len(lines))
ann = pd.read_csv(io.StringIO("\n".join(lines[s:e])), sep="\t", low_memory=False).set_index("ID")

bg_probes = pd.read_pickle("results/X_train.pkl").index          # analysed background
sig_probes = pd.read_csv("results/taskB_selected_probes.csv")["probe"]

def go_map(probes):
    d = {}
    for p, g in ann.reindex(probes)["GO_ID"].dropna().items():
        sym = ann.at[p, "GENE_SYMBOL"] if p in ann.index else None
        for t in str(g).split("|"):
            t = t.strip()
            if t:
                d.setdefault(t, set()).add(sym if pd.notna(sym) else p)
    return d

bg, sg = go_map(bg_probes), go_map(sig_probes)
N = len(set().union(*bg.values())); K = len(set().union(*sg.values()))
print(f"background genes: {N}   signature genes: {K}   GO terms tested: {len(sg)}")

rows = []
for t, genes in sg.items():
    k = len(genes); n = len(bg.get(t, set()))
    if k < 3 or n < 5: continue
    p = hypergeom.sf(k-1, N, n, K)
    rows.append(dict(GO=t, sig_genes=k, bg_genes=n,
                     expected=round(K*n/N, 2), fold=round((k/K)/(n/N), 2), p=p,
                     genes=",".join(sorted(list(genes))[:10])))
res = pd.DataFrame(rows).sort_values("p")
res["q_BH"] = np.minimum.accumulate(
    (res["p"]*len(res)/np.arange(1, len(res)+1)).to_numpy()[::-1])[::-1].clip(0,1)
res.to_csv("results/go_enrichment.csv", index=False)

print("\n" + "="*100)
print("TOP 20 ENRICHED GO TERMS (hypergeometric, BH-corrected)")
print("="*100)
print(res.head(20)[["GO","sig_genes","bg_genes","expected","fold","p","q_BH"]].to_string(index=False))
print(f"\nterms at q<0.05: {(res['q_BH']<0.05).sum()}")
print("\n--- genes driving the top 8 terms ---")
for _, r in res.head(8).iterrows():
    print(f"{r['GO']}  (fold {r['fold']}, q {r['q_BH']:.2e}): {r['genes']}")
