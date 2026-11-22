"""
Step 5: corrected design.
  Task A: lesion vs paired adjacent gastritis (within-patient, binary)
  Task B: ORDINAL staging among lesions only  LGIN -> HGIN -> Cancer
          trained on GSE130823 (n=47), externally validated on GSE55696 (n=58)
Compares plain multi-class vs ordinal models.
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                             confusion_matrix, mean_absolute_error, roc_auc_score)
from xgboost import XGBClassifier
import mord

STAGES = ["Gastritis", "LGIN", "HGIN", "Cancer"]
LES = ["LGIN", "HGIN", "Cancer"]

Xtr_all = pd.read_pickle("results/X_train.pkl").T
Xva_all = pd.read_pickle("results/X_valid.pkl").T
mtr = pd.read_csv("results/meta_train.csv", index_col=0).loc[Xtr_all.index]
mva = pd.read_csv("results/meta_valid.csv", index_col=0).loc[Xva_all.index]
probes = Xtr_all.columns  # identical in both after step 2

# ---------- Task B: lesions only ----------
tr_m = mtr["stage"].isin(LES).to_numpy()
va_m = mva["stage"].isin(LES).to_numpy()
Xtr = Xtr_all[tr_m].to_numpy(float); ytr = mtr.loc[tr_m, "stage"].map({s:i for i,s in enumerate(LES)}).to_numpy()
Xva = Xva_all[va_m].to_numpy(float); yva = mva.loc[va_m, "stage"].map({s:i for i,s in enumerate(LES)}).to_numpy()
print("="*74)
print("TASK B — ORDINAL STAGING OF LESIONS (LGIN -> HGIN -> Cancer)")
print("="*74)
print(f"train GSE130823: n={len(ytr)}  {dict(zip(LES, np.bincount(ytr)))}")
print(f"valid GSE55696 : n={len(yva)}  {dict(zip(LES, np.bincount(yva)))}")

K = 300
sel = SelectKBest(f_classif, k=K).fit(Xtr, ytr)
A, B = sel.transform(Xtr), sel.transform(Xva)
sc = StandardScaler().fit(A); As, Bs = sc.transform(A), sc.transform(B)
sig_probes = probes[sel.get_support()]
pd.Series(sig_probes).to_csv("results/taskB_selected_probes.csv", index=False, header=["probe"])

defs = {
  "Multinomial LogReg": ("lin", LogisticRegression(max_iter=5000, class_weight="balanced")),
  "Random Forest":      ("raw", RandomForestClassifier(n_estimators=800, min_samples_leaf=2,
                                class_weight="balanced_subsample", random_state=0, n_jobs=-1)),
  "XGBoost":            ("raw", XGBClassifier(n_estimators=300, max_depth=2, learning_rate=0.06,
                                subsample=0.9, colsample_bytree=0.5, reg_lambda=3.0,
                                tree_method="hist", random_state=0, n_jobs=-1)),
  "ORDINAL LogisticAT": ("lin", mord.LogisticAT(alpha=1.0)),
  "ORDINAL LogisticIT": ("lin", mord.LogisticIT(alpha=1.0)),
}

def metrics(t, p):
    return dict(acc=accuracy_score(t,p), macroF1=f1_score(t,p,average="macro"),
                QWK=cohen_kappa_score(t,p,weights="quadratic"), stageMAE=mean_absolute_error(t,p))

rows = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
for name, (kind, mdl) in defs.items():
    Atr, Bte = (As, Bs) if kind == "lin" else (A, B)
    # internal CV (each lesion = one patient here, so sample split == patient split)
    oof = np.zeros_like(ytr)
    for i, j in cv.split(Atr, ytr):
        import copy; m = copy.deepcopy(mdl); m.fit(Atr[i], ytr[i]); oof[j] = np.round(m.predict(Atr[j])).astype(int)
    oof = np.clip(oof, 0, 2)
    m2 = mdl.fit(Atr, ytr)
    pv = np.clip(np.round(m2.predict(Bte)).astype(int), 0, 2)
    r_in, r_ex = metrics(ytr, oof), metrics(yva, pv)
    rows.append({"model": name, **{f"cv_{k}": v for k, v in r_in.items()},
                 **{f"ext_{k}": v for k, v in r_ex.items()}})
    if name.startswith("ORDINAL Logistic AT") or name == "ORDINAL LogisticAT":
        best_cm = confusion_matrix(yva, pv)

res = pd.DataFrame(rows).set_index("model")
print("\n--- internal CV on GSE130823 (n=47) ---")
print(res[[c for c in res.columns if c.startswith("cv_")]].round(3).to_string())
print("\n--- EXTERNAL VALIDATION on GSE55696 (n=58, different platform, different patients) ---")
print(res[[c for c in res.columns if c.startswith("ext_")]].round(3).to_string())
print(f"\n(majority-class external accuracy = {np.bincount(yva).max()/len(yva):.3f})")

best = res["ext_QWK"].idxmax()
print(f"\nbest external QWK: {best} = {res.loc[best,'ext_QWK']:.3f}")
kind, mdl = defs[best]; Atr, Bte = (As, Bs) if kind=="lin" else (A, B)
pv = np.clip(np.round(mdl.fit(Atr,ytr).predict(Bte)).astype(int), 0, 2)
cm = confusion_matrix(yva, pv)
print(f"\nexternal confusion matrix ({best}):")
print(pd.DataFrame(cm, index=[f"true_{s}" for s in LES],
                   columns=[f"pred_{s}" for s in LES]).to_string())
adj = sum(cm[i,j] for i in range(3) for j in range(3) if abs(i-j)==1)
far = sum(cm[i,j] for i in range(3) for j in range(3) if abs(i-j)>=2)
print(f"\nerror structure: adjacent-stage errors={adj}, two-stage errors={far} "
      f"({100*far/max(1,adj+far):.0f}% of errors are severe)")
res.to_csv("results/taskB_results.csv")

# ---------- Task A: lesion vs paired adjacent gastritis ----------
print("\n" + "="*74)
print("TASK A — LESION vs PAIRED ADJACENT GASTRITIS (within-patient, GSE130823)")
print("="*74)
from sklearn.model_selection import StratifiedGroupKFold
ya = (mtr["stage"] != "Gastritis").astype(int).to_numpy()
ga = mtr["patient"].to_numpy(); Xa = Xtr_all.to_numpy(float)
cvg = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)
pr = np.zeros(len(ya)); pd_ = np.zeros(len(ya), dtype=int)
for i, j in cvg.split(Xa, ya, groups=ga):
    s = SelectKBest(f_classif, k=300).fit(Xa[i], ya[i])
    m = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.07, subsample=0.9,
        colsample_bytree=0.5, reg_lambda=2.0, tree_method="hist", random_state=0, n_jobs=-1)
    m.fit(s.transform(Xa[i]), ya[i])
    pr[j] = m.predict_proba(s.transform(Xa[j]))[:,1]; pd_[j] = m.predict(s.transform(Xa[j]))
print(f"AUC={roc_auc_score(ya,pr):.3f}  acc={accuracy_score(ya,pd_):.3f}  F1={f1_score(ya,pd_):.3f}")
print(pd.DataFrame(confusion_matrix(ya,pd_), index=["true_Gastritis","true_Lesion"],
                   columns=["pred_Gastritis","pred_Lesion"]).to_string())
