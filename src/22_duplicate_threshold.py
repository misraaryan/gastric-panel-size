import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"src"); from common import *
def Z(A):
    A=A.loc[probes].astype(float); return A
def corr(A,B):
    ok=A.notna().all(axis=1)&B.notna().all(axis=1)
    a=((A[ok]-A[ok].mean())/A[ok].std()).to_numpy(); b=((B[ok]-B[ok].mean())/B[ok].std()).to_numpy()
    return np.corrcoef(a.T,b.T)[:a.shape[1],a.shape[1]:]
D={"GSE130823":(Z(X1),m1["patient"].reindex(X1.columns).values),
   "GSE55696":(Z(X2),np.array([None]*X2.shape[1])),
   "GSE87666":(Z(X3),m3["patient"].reindex(X3.columns).values)}
rows=[]; diff=[]; same=[]
for n,(A,pt) in D.items():
    C=corr(A,A); iu=np.triu_indices_from(C,1)
    for i,j in zip(*iu):
        if pt[i] is None or (isinstance(pt[i],float)): kind="unknown"
        else: kind="same" if pt[i]==pt[j] else "different"
        rows.append(dict(cohort=n,kind=kind,r=C[i,j]))
d=pd.DataFrame(rows)
print(d.groupby(["cohort","kind"]).r.describe(percentiles=[.5,.95,.99]).round(3).to_string())
dd=d[d.kind=="different"].r
print("pooled different-patient within-cohort: 95th %.3f 99th %.3f max %.3f n=%d"%(dd.quantile(.95),dd.quantile(.99),dd.max(),len(dd)))
ss=d[d.kind=="same"].r; print("same-patient: median %.3f min %.3f max %.3f n=%d"%(ss.median(),ss.min(),ss.max(),len(ss)))
for a,b,na,nb in [(X1,X2,"130823","55696"),(X1,X3,"130823","87666"),(X2,X3,"55696","87666")]:
    C=corr(Z(a),Z(b)); print(f"cross {na}-{nb}: max {C.max():.3f}, pct of different-patient within-cohort pairs below it: {(dd<C.max()).mean()*100:.1f}%")
d.to_csv("results/within_cohort_correlations.csv",index=False)
