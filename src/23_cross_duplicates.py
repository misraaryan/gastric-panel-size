import sys, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"src"); from common import *
def corr(A,B):
    A=A.loc[probes].astype(float);B=B.loc[probes].astype(float)
    ok=A.notna().all(axis=1)&B.notna().all(axis=1)
    a=((A[ok]-A[ok].mean())/A[ok].std()).to_numpy(); b=((B[ok]-B[ok].mean())/B[ok].std()).to_numpy()
    return pd.DataFrame(np.corrcoef(a.T,b.T)[:a.shape[1],a.shape[1]:],index=A.columns,columns=B.columns)
st={**m1["stage"].to_dict(),**m2["stage"].to_dict(),**m3["stage"].to_dict()}
pt={**m1["patient"].to_dict(),**m3["patient"].to_dict()}
tt={**m1["title"].to_dict(),**m2["title"].to_dict(),**m3["title"].to_dict()}
allrows=[]
for a,b,na,nb in [(X1,X2,"GSE130823","GSE55696"),(X1,X3,"GSE130823","GSE87666"),(X2,X3,"GSE55696","GSE87666")]:
    C=corr(a,b); v=C.values.ravel()
    print(f"\n{na} vs {nb}: pairs {v.size}; >0.6: {(v>0.6).sum()}, >0.7: {(v>0.7).sum()}, >0.8: {(v>0.8).sum()}; 99.9th {np.percentile(v,99.9):.3f}")
    # mutual best matches
    rb=C.idxmax(axis=1); cb=C.idxmax(axis=0)
    for s in C.index:
        t=rb[s]
        if cb[t]==s and C.loc[s,t]>0.6:
            # gap to second best
            row=C.loc[s].sort_values(ascending=False); col=C[t].sort_values(ascending=False)
            allrows.append(dict(pair=f"{na}-{nb}",a=s,a_title=tt[s],a_stage=st[s],a_pt=pt.get(s),b=t,b_title=tt[t],b_stage=st[t],
                r=round(C.loc[s,t],3),second_row=round(row.iloc[1],3),second_col=round(col.iloc[1],3)))
r=pd.DataFrame(allrows).sort_values("r",ascending=False)
pd.set_option("display.width",250); print(r.to_string(index=False))
r.to_csv("results/cross_cohort_mutual_matches.csv",index=False)
