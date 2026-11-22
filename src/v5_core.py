"""v5 model comparison, bootstrap CIs, whole-pipeline permutation (deduplicated, leave-one-platform-out)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,"src"); from v5data import *
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import cohen_kappa_score as kap, accuracy_score
from xgboost import XGBClassifier
import mord
def mk(m):
    return {"Ordinal":mord.LogisticAT(alpha=1.0),
            "Multinomial":LogisticRegression(max_iter=5000,class_weight="balanced"),
            "Random forest":RandomForestClassifier(n_estimators=800,min_samples_leaf=2,class_weight="balanced_subsample",random_state=0,n_jobs=-1),
            "XGBoost":XGBClassifier(n_estimators=300,max_depth=2,learning_rate=0.06,subsample=0.9,colsample_bytree=0.5,reg_lambda=3.0,tree_method="hist",random_state=0,n_jobs=-1)}[m]
def run(D, models, k=300):
    out={m:([],[],[]) for m in models}
    for held in names:
        tr=[n for n in names if n!=held]
        A=np.vstack([D[n][0] for n in tr]); ytr=np.concatenate([D[n][1] for n in tr]); B,yte=D[held]
        sel=SelectKBest(f_classif,k=k).fit(A,ytr); A2,B2=sel.transform(A),sel.transform(B)
        sc=StandardScaler().fit(A2); Az,Bz=sc.transform(A2),sc.transform(B2)
        for m in models:
            lin=m in("Ordinal","Multinomial")
            p=np.clip(np.round(mk(m).fit(Az if lin else A2,ytr).predict(Bz if lin else B2)).astype(int),0,2)
            out[m][0].extend(yte); out[m][1].extend(p); out[m][2].extend([held]*len(yte))
    return {m:pd.DataFrame(dict(y=v[0],p=v[1],grp=v[2])) for m,v in out.items()}
def met(d):
    y,p=d.y.values,d.p.values
    return dict(acc=accuracy_score(y,p),QWK=kap(y,p,weights="quadratic"),MAE=np.abs(y-p).mean(),severe=int((np.abs(y-p)>=2).sum()),
                **{f"QWK_{g}":kap(d[d.grp==g].y,d[d.grp==g].p,weights="quadratic") for g in names})
if __name__=="__main__":
    M=["Ordinal","Multinomial","Random forest","XGBoost"]
    R=run(PNP,M); tab=pd.DataFrame({m:met(R[m]) for m in M}).T; print(tab.round(3).to_string(),flush=True)
    pd.concat([R[m].assign(model=m) for m in M]).to_csv("results/v5_predictions.csv",index=False)
    # bootstrap (resample within group, same indices across models)
    rng=np.random.default_rng(0); n=len(R["Ordinal"]); B=[]
    idx={g:np.where(R["Ordinal"].grp.values==g)[0] for g in names}
    for b in range(2000):
        ii=np.concatenate([rng.choice(idx[g],len(idx[g])) for g in names])
        B.append({m:kap(R[m].y.values[ii],R[m].p.values[ii],weights="quadratic") for m in M})
    B=pd.DataFrame(B)
    for m in M: tab.loc[m,"lo"],tab.loc[m,"hi"]=np.percentile(B[m],[2.5,97.5])
    for m in M[1:]:
        d=B["Ordinal"]-B[m]; tab.loc[m,"diff_vs_ord"]=tab.loc["Ordinal","QWK"]-tab.loc[m,"QWK"]
        tab.loc[m,"diff_lo"],tab.loc[m,"diff_hi"]=np.percentile(d,[2.5,97.5]); tab.loc[m,"P_diff_le0"]=(d<=0).mean()
    tab.to_csv("results/v5_models.csv"); print(tab.round(3).to_string(),flush=True)
    # whole-pipeline permutation, ordinal
    NP=1000; perm=[]
    for i in range(NP):
        D={g:(PNP[g][0],rng.permutation(PNP[g][1])) for g in names}
        d=run(D,["Ordinal"])["Ordinal"]; perm.append(kap(d.y,d.p,weights="quadratic"))
    perm=np.array(perm); obs=tab.loc["Ordinal","QWK"]
    pd.DataFrame({"perm":perm}).to_csv("results/v5_permutation.csv",index=False)
    print(f"PERM mean {perm.mean():.3f} 95th {np.percentile(perm,95):.3f} max {perm.max():.3f} p={(1+(perm>=obs).sum())/(1+NP):.4f}",flush=True)
