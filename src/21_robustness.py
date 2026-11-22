"""Robustness checks: (a) whole-pipeline label permutation, (b) duplicate-threshold justification,
(c) stability selection, (d) ComBat sensitivity."""
import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"src"); from common import *
from sklearn.linear_model import LogisticRegression
rng=np.random.default_rng(42)
names=list(coh)
def loco(Xd, yd, k=300, models=("ord","mn")):
    out={m:[] for m in models}; P={m:[] for m in models}
    for held in names:
        tr=[n for n in names if n!=held]
        Xtr=np.vstack([Xd[n] for n in tr]); ytr=np.concatenate([yd[n] for n in tr])
        Xte=Xd[held]; yte=yd[held]
        sel=SelectKBest(f_classif,k=k).fit(Xtr,ytr); A,B=sel.transform(Xtr),sel.transform(Xte)
        sc=StandardScaler().fit(A); A,B=sc.transform(A),sc.transform(B)
        for m in models:
            mdl=mord.LogisticAT(alpha=1.0) if m=="ord" else LogisticRegression(max_iter=5000,class_weight="balanced")
            p=np.clip(np.round(mdl.fit(A,ytr).predict(B)).astype(int),0,2)
            out[m].append(cohen_kappa_score(yte,p,weights="quadratic")); P[m].append((yte,p))
    return {m:np.mean(v) for m,v in out.items()}, P
Xd={n:coh[n][0][common].to_numpy(float) for n in names}; yd={n:coh[n][1] for n in names}
obs,_=loco(Xd,yd); print("observed mean QWK:",{k:round(v,3) for k,v in obs.items()},flush=True)

# (a) permutation
NP=int(sys.argv[1]) if len(sys.argv)>1 else 1000
perm=[]
for i in range(NP):
    yp={n:rng.permutation(yd[n]) for n in names}
    perm.append(loco(Xd,yp,models=("ord",))[0]["ord"])
    if (i+1)%100==0: print(i+1,flush=True)
perm=np.array(perm); p=(1+(perm>=obs["ord"]).sum())/(1+NP)
pd.DataFrame({"perm_QWK":perm}).to_csv("results/permutation_qwk.csv",index=False)
print(f"PERMUTATION: mean {perm.mean():.3f} sd {perm.std():.3f} 95th {np.percentile(perm,95):.3f} max {perm.max():.3f} p={p:.4f}",flush=True)
