"""
Step 9: three-cohort integration.
  - add GSE87666 (22 samples, GPL17077, LGIN/HGIN/EGC)
  - check for possible patient overlap between cohorts (shared institution)
  - leave-one-cohort-out validation
"""
import gzip, io, re, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                             confusion_matrix, mean_absolute_error)
from xgboost import XGBClassifier
import mord

LES = ["LGIN", "HGIN", "Cancer"]

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

X3, m3 = parse("data/GSE87666_series_matrix.txt.gz")
m3["stage"]=m3["title"].map(lab); m3["cohort"]="GSE87666"
m3["patient"]=m3["title"].str.extract(r"(Patient\d+)")
(lambda *a,**k:None)("GSE87666:", X3.shape, dict(m3["stage"].value_counts()))
(lambda *a,**k:None)("patients with >1 biopsy:", (m3['patient'].value_counts()>1).sum())

X1 = pd.read_pickle("results/X_train.pkl"); X2 = pd.read_pickle("results/X_valid.pkl")
m1 = pd.read_csv("results/meta_train.csv", index_col=0); m2 = pd.read_csv("results/meta_valid.csv", index_col=0)
probes = X1.index

X3 = X3.reindex(probes).astype(float)
(lambda *a,**k:None)(f"\nGSE87666 coverage of the {len(probes)} analysis probes: "
      f"{X3.notna().all(axis=1).sum()} ({100*X3.notna().all(axis=1).mean():.1f}%)")
X3 = X3.sub(X3.median(axis=0), axis=1)   # same centring as the others

# ---------- leave-one-cohort-out ----------
coh = {}
for nm, X, m in [("GSE130823",X1,m1), ("GSE55696",X2,m2), ("GSE87666",X3,m3)]:
    keep = m["stage"].isin(LES).to_numpy()
    Xs = X.T[keep]
    good = Xs.columns[Xs.notna().all(axis=0)]
    coh[nm] = (Xs, m.loc[keep,"stage"].map({s:i for i,s in enumerate(LES)}).to_numpy(), good)

common = set(probes)
for nm in coh: common &= set(coh[nm][2])
common = sorted(common)
(lambda *a,**k:None)(f"\nprobes usable across all three cohorts: {len(common)}")


# ---- deduplication (cross-cohort duplicate tissue samples; see 22/23 scripts) ----
import os as _os
if _os.environ.get("DEDUP","1")=="1" and _os.path.exists("results/duplicate_samples_removed.csv"):
    _drop=set(pd.read_csv("results/duplicate_samples_removed.csv").gsm)
    for _n in list(coh):
        _X,_y,_g=coh[_n]; _k=~_X.index.isin(_drop)
        coh[_n]=(_X[_k],_y[_k],_g)
