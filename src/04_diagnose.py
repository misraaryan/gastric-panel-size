"""Step 4: diagnostics. Is there real signal, and where does it fail?"""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, accuracy_score, f1_score,
                             cohen_kappa_score, roc_auc_score)
from xgboost import XGBClassifier

STAGES = ["Gastritis", "LGIN", "HGIN", "Cancer"]
X = pd.read_pickle("results/X_train.pkl").T
meta = pd.read_csv("results/meta_train.csv", index_col=0).loc[X.index]
y = meta["stage"].map({s: i for i, s in enumerate(STAGES)}).to_numpy()
groups = meta["patient"].to_numpy()
Xn = X.to_numpy(float)

print("="*70); print("BASELINES"); print("="*70)
maj = np.bincount(y).argmax()
print(f"majority class = {STAGES[maj]} ({np.bincount(y)[maj]}/{len(y)})")
print(f"majority-class accuracy : {np.mean(y == maj):.3f}")
print(f"majority-class macro F1 : {f1_score(y, np.full_like(y, maj), average='macro'):.3f}")
print(f"majority-class QWK      : {cohen_kappa_score(y, np.full_like(y, maj), weights='quadratic'):.3f}")

def cv_pred(Xd, yd, grp, k=500, model="xgb", n_splits=5, seed=0):
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pred = np.zeros_like(yd); prob = np.zeros((len(yd), len(np.unique(yd))))
    for tr, te in cv.split(Xd, yd, groups=grp):
        sel = SelectKBest(f_classif, k=min(k, Xd.shape[1])).fit(Xd[tr], yd[tr])
        a, b = sel.transform(Xd[tr]), sel.transform(Xd[te])
        if model == "xgb":
            m = XGBClassifier(n_estimators=300, max_depth=3, learning_rate=0.07,
                subsample=0.85, colsample_bytree=0.4, reg_lambda=2.0,
                tree_method="hist", random_state=0, n_jobs=-1)
        else:
            m = Pipeline([("sc", StandardScaler()),
                 ("m", LogisticRegression(max_iter=4000, class_weight="balanced"))])
        m.fit(a, yd[tr]); pred[te] = m.predict(b); prob[te] = m.predict_proba(b)
    return pred, prob

print("\n" + "="*70); print("4-CLASS CONFUSION MATRIX (patient-grouped CV, XGBoost)"); print("="*70)
pred, prob = cv_pred(Xn, y, groups)
cm = confusion_matrix(y, pred)
print(pd.DataFrame(cm, index=[f"true_{s}" for s in STAGES],
                   columns=[f"pred_{s}" for s in STAGES]).to_string())
print(f"\naccuracy {accuracy_score(y,pred):.3f} | macroF1 {f1_score(y,pred,average='macro'):.3f} "
      f"| QWK {cohen_kappa_score(y,pred,weights='quadratic'):.3f}")
print("\nper-stage recall:")
for i, s in enumerate(STAGES):
    n = (y == i).sum()
    print(f"  {s:10s} {cm[i,i]}/{n} = {cm[i,i]/n:.2f}")

print("\n" + "="*70); print("BINARY SUB-TASKS (patient-grouped CV) — where is the signal?")
print("="*70)
tasks = {
    "Gastritis vs any lesion": (y > 0).astype(int),
    "Cancer vs non-cancer":    (y == 3).astype(int),
    "HGIN+Cancer vs Gastritis+LGIN": (y >= 2).astype(int),
}
for name, yb in tasks.items():
    pb, prb = cv_pred(Xn, yb, groups)
    auc = roc_auc_score(yb, prb[:, 1])
    print(f"{name:32s} n+={yb.sum():3d}  AUC={auc:.3f}  acc={accuracy_score(yb,pb):.3f}  "
          f"F1={f1_score(yb,pb):.3f}")

print("\n" + "="*70); print("LESION-ONLY 3-CLASS (drop the paired gastritis controls)")
print("="*70)
les = y > 0
Xl, yl, gl = Xn[les], y[les] - 1, groups[les]
print(f"n={len(yl)} (LGIN {np.sum(yl==0)}, HGIN {np.sum(yl==1)}, Cancer {np.sum(yl==2)}), "
      f"patients={len(set(gl))}")
pl, _ = cv_pred(Xl, yl, gl, n_splits=4)
cml = confusion_matrix(yl, pl)
print(pd.DataFrame(cml, index=["true_LGIN","true_HGIN","true_Cancer"],
                   columns=["pred_LGIN","pred_HGIN","pred_Cancer"]).to_string())
print(f"accuracy {accuracy_score(yl,pl):.3f} | macroF1 {f1_score(yl,pl,average='macro'):.3f} "
      f"| QWK {cohen_kappa_score(yl,pl,weights='quadratic'):.3f}")
majl = np.bincount(yl).argmax()
print(f"(majority baseline accuracy {np.mean(yl==majl):.3f})")

np.save("results/cv_pred_4class.npy", pred)
pd.DataFrame(cm, index=STAGES, columns=STAGES).to_csv("results/confusion_4class.csv")
