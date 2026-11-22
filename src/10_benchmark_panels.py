"""
Step 10: head-to-head benchmark of PUBLISHED gastric precancerous gene panels
on the ordinal staging task, under identical leave-one-cohort-out validation.
Includes size-matched random-panel null distributions.
"""
import gzip, io, re, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                             mean_absolute_error)
import mord

LES = ["LGIN", "HGIN", "Cancer"]
rng = np.random.default_rng(0)

# ---------------- published panels ----------------
PANELS = {
 "Ji 2024 (immune hub)":        ["PTPRC","PLEK","CD53","CD48","LCP1","HCLS1","IKZF1","BTK","VAV1"],
 "Jin 2022 (HGIN key)":         ["MYC","SOX2","CDX2","TBX3","KRT7","CDKN2A","MUC5AC"],
 "Yi 2025 (GPL/H.pylori hub)":  ["IL6","FCGR3A","CCL3","CXCR4","CXCL9","CCL4","CCR1"],
 "Rezaei 2021 (WGCNA GC)":      ["ITGAX","CCL14","ADHFE1","HOXB13"],
 "Luu 2024 (GC hub)":           ["SPP1","INHBA","MMP7","THBS2","FAP"],
 "FrontGenet 2025 (WGCNA)":     ["FCRL3","EFEMP1","ANKRD29","STOX2"],
 "Jiang 2021 (CXCR4)":          ["CXCR4"],
 "Chen 2020 (ICAM1)":           ["ICAM1"],
}

def parse(path):
    titles, acc, tbl, inb = [], [], [], False
    with gzip.open(path, "rt", encoding="latin-1") as fh:
        for l in fh:
            if l.startswith("!Sample_title"): titles=[x.strip().strip('"') for x in l.rstrip().split("\t")[1:]]
            elif l.startswith("!Sample_geo_accession"): acc=[x.strip().strip('"') for x in l.rstrip().split("\t")[1:]]
            elif l.startswith("!series_matrix_table_begin"): inb=True
            elif l.startswith("!series_matrix_table_end"): inb=False
            elif inb: tbl.append(l)
    X = pd.read_csv(io.StringIO("".join(tbl)), sep="\t", index_col=0)
    X.columns=[c.strip().strip('"') for c in X.columns]
    return X, pd.DataFrame({"gsm":acc,"title":titles}).set_index("gsm")

def lab(t):
    s=t.lower().replace(" ","")
    if "lowgradeintraepithelialneoplasia" in s or "lgin" in s: return "LGIN"
    if "highgradeintraepithelialneoplasia" in s or "hgin" in s: return "HGIN"
    if any(k in s for k in ["intestinalgastriccancer","gastriccancer","carcinoma","egc"]): return "Cancer"
    if "gastritis" in s: return "Gastritis"
    return None

X1 = pd.read_pickle("results/X_train.pkl"); X2 = pd.read_pickle("results/X_valid.pkl")
m1 = pd.read_csv("results/meta_train.csv", index_col=0); m2 = pd.read_csv("results/meta_valid.csv", index_col=0)
X3r, m3 = parse("data/GSE87666_series_matrix.txt.gz")
m3["stage"]=m3["title"].map(lab)
probes = X1.index
X3 = X3r.reindex(probes).astype(float); X3 = X3.sub(X3.median(axis=0), axis=1)

p2s = pd.read_csv("results/probe2symbol.csv", index_col=0)["GENE_SYMBOL"]
p2s = p2s.reindex(probes).dropna()

def gene_level(X):
    """collapse probes to genes by mean"""
    sub = X.loc[p2s.index]
    return sub.groupby(p2s.values).mean()

G = {}
for nm, X, m in [("GSE130823",X1,m1), ("GSE55696",X2,m2), ("GSE87666",X3,m3)]:
    keep = m["stage"].isin(LES).to_numpy()
    g = gene_level(X).T[keep]
    G[nm] = (g, m.loc[keep,"stage"].map({s:i for i,s in enumerate(LES)}).to_numpy())

genes_all = sorted(set.intersection(*[set(G[n][0].columns) for n in G]))
# drop genes with any NaN in any cohort
ok = [g for g in genes_all if all(G[n][0][g].notna().all() for n in G)]
print(f"gene-level matrix: {len(ok)} genes shared and complete across all three cohorts")
for n in G: G[n] = (G[n][0][ok], G[n][1])

names = list(G)
def loco(gene_list, select_k=None):
    """evaluate a fixed gene list (or in-fold selection if select_k given)"""
    out=[]
    for held in names:
        tr=[n for n in names if n!=held]
        Xtr = pd.concat([G[n][0] for n in tr]); ytr = np.concatenate([G[n][1] for n in tr])
        Xte = G[held][0]; yte = G[held][1]
        if select_k is not None:
            sel = SelectKBest(f_classif, k=min(select_k, Xtr.shape[1])).fit(Xtr.to_numpy(float), ytr)
            cols = Xtr.columns[sel.get_support()]
        else:
            cols = [g for g in gene_list if g in Xtr.columns]
            if len(cols) == 0: return None, 0
        A = Xtr[cols].to_numpy(float); B = Xte[cols].to_numpy(float)
        sc = StandardScaler().fit(A)
        mdl = mord.LogisticAT(alpha=1.0).fit(sc.transform(A), ytr)
        p = np.clip(np.round(mdl.predict(sc.transform(B))).astype(int), 0, 2)
        out.append((yte, p))
    yt = np.concatenate([o[0] for o in out]); yp = np.concatenate([o[1] for o in out])
    return dict(acc=accuracy_score(yt,yp), macroF1=f1_score(yt,yp,average="macro"),
                QWK=cohen_kappa_score(yt,yp,weights="quadratic"),
                stageMAE=mean_absolute_error(yt,yp),
                severe=int(np.sum(np.abs(yt-yp)>=2))), len(yt)

rows=[]
for nm, gl in PANELS.items():
    found=[g for g in gl if g in ok]
    r,_ = loco(found)
    if r: rows.append(dict(panel=nm, n_genes_claimed=len(gl), n_genes_found=len(found), **r))

# our signature, in-fold selection, size-matched and full
for k in [9, 50, 300]:
    r,_ = loco(None, select_k=k)
    rows.append(dict(panel=f"OURS (in-fold top-{k})", n_genes_claimed=k, n_genes_found=k, **r))

res = pd.DataFrame(rows).sort_values("QWK", ascending=False)
print("\n" + "="*104)
print("BENCHMARK: PUBLISHED PANELS vs OURS — ordinal staging, leave-one-cohort-out, all 127 samples")
print("="*104)
print(res.round(3).to_string(index=False))

# ---------- random-panel null distributions ----------
print("\n" + "="*104)
print("RANDOM-PANEL NULL CONTROLS (100 random gene panels per size)")
print("="*104)
null = {}
for size in sorted(set(res["n_genes_found"]) | {9}):
    if size < 1: continue
    qs=[]
    for i in range(100):
        gl = list(rng.choice(ok, size=size, replace=False))
        r,_ = loco(gl)
        if r: qs.append(r["QWK"])
    null[size] = np.array(qs)
    print(f"size {size:3d}: random QWK mean {qs and np.mean(qs):.3f}, "
          f"95th pct {np.percentile(qs,95):.3f}, max {np.max(qs):.3f}")

print("\n--- each panel vs its size-matched random null ---")
for _, r in res.iterrows():
    s = int(r["n_genes_found"])
    if s in null and len(null[s]):
        pct = 100*(null[s] < r["QWK"]).mean()
        emp_p = (1 + (null[s] >= r["QWK"]).sum()) / (1 + len(null[s]))
        print(f"{r['panel']:28s} QWK {r['QWK']:.3f} -> beats {pct:5.1f}% of random panels "
              f"(empirical p = {emp_p:.3f})")

res.to_csv("results/panel_benchmark.csv", index=False)
pd.DataFrame({f"size_{k}": pd.Series(v) for k,v in null.items()}).to_csv(
    "results/panel_benchmark_null.csv", index=False)
print("\nsaved results/panel_benchmark.csv + panel_benchmark_null.csv")
