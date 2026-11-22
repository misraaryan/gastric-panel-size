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
print("GSE87666:", X3.shape, dict(m3["stage"].value_counts()))
print("patients with >1 biopsy:", (m3['patient'].value_counts()>1).sum())

X1 = pd.read_pickle("results/X_train.pkl"); X2 = pd.read_pickle("results/X_valid.pkl")
m1 = pd.read_csv("results/meta_train.csv", index_col=0); m2 = pd.read_csv("results/meta_valid.csv", index_col=0)
probes = X1.index

X3 = X3.reindex(probes).astype(float)
print(f"\nGSE87666 coverage of the {len(probes)} analysis probes: "
      f"{X3.notna().all(axis=1).sum()} ({100*X3.notna().all(axis=1).mean():.1f}%)")
X3 = X3.sub(X3.median(axis=0), axis=1)   # same centring as the others

# ---------- cross-cohort duplicate check ----------
print("\n" + "="*74); print("PATIENT-OVERLAP CHECK (same institution across all three cohorts)")
print("="*74)
def corrmax(A, B, na, nb):
    A = A.loc[probes].astype(float); B = B.loc[probes].astype(float)
    ok = A.notna().all(axis=1) & B.notna().all(axis=1)
    Az = ((A[ok] - A[ok].mean())/A[ok].std()).to_numpy()
    Bz = ((B[ok] - B[ok].mean())/B[ok].std()).to_numpy()
    C = np.corrcoef(Az.T, Bz.T)[:Az.shape[1], Az.shape[1]:]
    print(f"{na} vs {nb}: max r = {C.max():.3f}, 99.9th pct = {np.percentile(C,99.9):.3f}, "
          f"pairs with r>0.95: {(C>0.95).sum()}")
    return C
corrmax(X1, X2, "GSE130823", "GSE55696")
corrmax(X1, X3, "GSE130823", "GSE87666")
corrmax(X2, X3, "GSE55696 ", "GSE87666")
print("(r>0.95 would indicate the same biological sample appearing twice)")

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
print(f"\nprobes usable across all three cohorts: {len(common)}")

def build(name):
    return {
      "Multinomial LogReg": ("lin", LogisticRegression(max_iter=5000, class_weight="balanced")),
      "Random Forest": ("raw", RandomForestClassifier(n_estimators=800, min_samples_leaf=2,
                        class_weight="balanced_subsample", random_state=0, n_jobs=-1)),
      "XGBoost": ("raw", XGBClassifier(n_estimators=300, max_depth=2, learning_rate=0.06,
                  subsample=0.9, colsample_bytree=0.5, reg_lambda=3.0,
                  tree_method="hist", random_state=0, n_jobs=-1)),
      "ORDINAL LogisticAT": ("lin", mord.LogisticAT(alpha=1.0)),
    }[name]

print("\n" + "="*74); print("LEAVE-ONE-COHORT-OUT VALIDATION (train on two, test on the held-out one)")
print("="*74)
rows=[]; P=[]
names = list(coh)
for held in names:
    tr_names=[n for n in names if n!=held]
    Xtr = pd.concat([coh[n][0][common] for n in tr_names]).to_numpy(float)
    ytr = np.concatenate([coh[n][1] for n in tr_names])
    Xte = coh[held][0][common].to_numpy(float); yte = coh[held][1]
    sel = SelectKBest(f_classif, k=min(300, len(common))).fit(Xtr, ytr)
    A,B = sel.transform(Xtr), sel.transform(Xte)
    sc = StandardScaler().fit(A); Az,Bz = sc.transform(A), sc.transform(B)
    for mn in ["Multinomial LogReg","Random Forest","XGBoost","ORDINAL LogisticAT"]:
        kind, mdl = build(mn)
        a,b = (Az,Bz) if kind=="lin" else (A,B)
        mdl.fit(a,ytr); p = np.clip(np.round(mdl.predict(b)).astype(int),0,2)
        P.append(pd.DataFrame(dict(held=held,model=mn,y=yte,p=p)))
        rows.append(dict(held_out=held, n_test=len(yte), model=mn,
            acc=accuracy_score(yte,p), macroF1=f1_score(yte,p,average="macro"),
            QWK=cohen_kappa_score(yte,p,weights="quadratic"),
            stageMAE=mean_absolute_error(yte,p),
            severe_err=int(sum(abs(yte-p)>=2))))
r = pd.DataFrame(rows)
print(r.round(3).to_string(index=False))
print("\n--- mean across the three held-out cohorts ---")
print(r.groupby("model")[["acc","macroF1","QWK","stageMAE","severe_err"]].mean().round(3).to_string())
pd.concat(P).to_csv("results/loco_predictions.csv",index=False)
print(f"\ntotal lesion samples across 3 cohorts: {sum(len(coh[n][1]) for n in names)}")
