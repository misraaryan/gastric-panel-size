"""
Step 3: THE headline experiment.
Same data, same models, same folds count — only the splitting rule changes:
  (A) sample-level StratifiedKFold      <- what the literature does (leaky here)
  (B) patient-level StratifiedGroupKFold <- correct for a paired cohort
The gap between A and B is the leakage estimate.
Feature selection happens INSIDE every fold.
"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                             mean_absolute_error)
from xgboost import XGBClassifier

STAGES = ["Gastritis", "LGIN", "HGIN", "Cancer"]
X = pd.read_pickle("results/X_train.pkl").T          # samples x probes
meta = pd.read_csv("results/meta_train.csv", index_col=0).loc[X.index]
y = meta["stage"].map({s: i for i, s in enumerate(STAGES)}).to_numpy()
groups = meta["patient"].to_numpy()
print(f"X {X.shape}, classes {np.bincount(y)}, patients {len(set(groups))}")

K = 500          # features selected inside each fold
N_SPLITS, N_REPEATS = 5, 6

def models():
    return {
        "LogReg-L2": Pipeline([("sc", StandardScaler()),
            ("m", LogisticRegression(max_iter=4000, C=1.0, class_weight="balanced"))]),
        "RandomForest": Pipeline([("m", RandomForestClassifier(
            n_estimators=600, min_samples_leaf=2, class_weight="balanced_subsample",
            random_state=0, n_jobs=-1))]),
        "XGBoost": Pipeline([("m", XGBClassifier(
            n_estimators=350, max_depth=3, learning_rate=0.07,
            subsample=0.85, colsample_bytree=0.4, reg_lambda=2.0,
            objective="multi:softprob", num_class=4, tree_method="hist",
            random_state=0, n_jobs=-1))]),
    }

def run(split_mode, seed):
    if split_mode == "sample":
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        it = cv.split(X, y)
    else:
        cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
        it = cv.split(X, y, groups=groups)
    out = {k: {"true": [], "pred": []} for k in models()}
    for tr_i, te_i in it:
        Xtr, Xte = X.iloc[tr_i].to_numpy(float), X.iloc[te_i].to_numpy(float)
        ytr, yte = y[tr_i], y[te_i]
        sel = SelectKBest(f_classif, k=min(K, Xtr.shape[1])).fit(Xtr, ytr)
        Xtr_s, Xte_s = sel.transform(Xtr), sel.transform(Xte)
        for name, mdl in models().items():
            mdl.fit(Xtr_s, ytr)
            out[name]["true"].extend(yte); out[name]["pred"].extend(mdl.predict(Xte_s))
    return out

rows = []
for mode in ["sample", "patient"]:
    agg = {k: {"true": [], "pred": []} for k in models()}
    for rep in range(N_REPEATS):
        r = run(mode, seed=rep)
        for k in agg:
            agg[k]["true"].extend(r[k]["true"]); agg[k]["pred"].extend(r[k]["pred"])
    for k, d in agg.items():
        t, p = np.array(d["true"]), np.array(d["pred"])
        rows.append({
            "split": mode, "model": k,
            "accuracy": accuracy_score(t, p),
            "macro_F1": f1_score(t, p, average="macro"),
            "QWK": cohen_kappa_score(t, p, weights="quadratic"),
            "stage_MAE": mean_absolute_error(t, p),
        })

res = pd.DataFrame(rows)
piv = res.pivot(index="model", columns="split")
print("\n" + "="*72)
print("LEAKAGE EXPERIMENT  (5-fold x 6 repeats, feature selection inside folds)")
print("="*72)
for m in ["accuracy", "macro_F1", "QWK", "stage_MAE"]:
    sub = res.pivot(index="model", columns="split", values=m)
    sub = sub[["sample", "patient"]]
    sub["delta"] = sub["sample"] - sub["patient"]
    print(f"\n--- {m} ---")
    print(sub.round(3).to_string())

res.to_csv("results/leakage_experiment.csv", index=False)
print("\nSaved results/leakage_experiment.csv")
