"""ComBat sensitivity: batch = source series, no biological covariates; re-run model comparison and a reduced size sweep."""
import sys, numpy as np, pandas as pd
sys.path.insert(0,"src"); import v5data as V
from combat.pycombat import pycombat
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score as kap
import mord, v5_core as C
def harmonise(D, src):
    X=np.vstack([D[g][0] for g in V.names]); b=np.concatenate([src[g] for g in V.names])
    Xc=pycombat(pd.DataFrame(X.T),list(b)).values.T
    out={};i=0
    for g in V.names: n=len(D[g][1]); out[g]=(Xc[i:i+n],D[g][1]); i+=n
    return out
src={g:V.P[g][2] for g in V.names}
PC=harmonise(V.PNP,src); GC=harmonise(V.NPY,src)
R=C.run(PC,["Ordinal","Multinomial"])
for m in R: print("ComBat",m,{k:round(v,3) if isinstance(v,float) else v for k,v in C.met(R[m]).items()},flush=True)
rng=np.random.default_rng(11); NREP=200
def cv(idx):
    yt,yp=[],[]
    for held in V.names:
        tr=[n for n in V.names if n!=held]
        A=np.vstack([GC[n][0][:,idx[held]] for n in tr]); ytr=np.concatenate([GC[n][1] for n in tr]); B=GC[held][0][:,idx[held]]
        sc=StandardScaler().fit(A); p=mord.LogisticAT(alpha=1.0).fit(sc.transform(A),ytr).predict(sc.transform(B))
        yt+=list(GC[held][1]); yp+=list(np.clip(np.round(p).astype(int),0,2))
    return kap(yt,yp,weights="quadratic")
F={}
for held in V.names:
    tr=[n for n in V.names if n!=held]; F[held],_=f_classif(np.vstack([GC[n][0] for n in tr]),np.concatenate([GC[n][1] for n in tr]))
rows=[]
for k in [5,10,50,200,1000]:
    q=cv({h:np.argsort(-F[h])[:k] for h in V.names})
    nd=np.array([cv({h:rng.choice(len(V.genes),k,replace=False) for h in V.names}) for _ in range(NREP)])
    rows.append(dict(size=k,ours=q,rand_mean=nd.mean(),p=(1+(nd>=q).sum())/(1+NREP))); print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv("results/v5_combat_sweep.csv",index=False)
