"""v5 published panels and panel-size sweep vs size-matched random nulls (gene level, pooled QWK)."""
import sys, io, numpy as np, pandas as pd
sys.path.insert(0,"src"); from v5data import *
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score as kap
import mord
NREP=1000; rng=np.random.default_rng(7)
def cv(idx_by_fold):
    yt,yp=[],[]
    for held in names:
        tr=[n for n in names if n!=held]; idx=idx_by_fold[held]
        A=np.vstack([NPY[n][0][:,idx] for n in tr]); ytr=np.concatenate([NPY[n][1] for n in tr])
        B=NPY[held][0][:,idx]; sc=StandardScaler().fit(A)
        p=mord.LogisticAT(alpha=1.0).fit(sc.transform(A),ytr).predict(sc.transform(B))
        yt+=list(NPY[held][1]); yp+=list(np.clip(np.round(p).astype(int),0,2))
    return kap(yt,yp,weights="quadratic")
# training-fold F statistics (for in-fold selection and in-fold stage pool)
F={};PV={}
for held in names:
    tr=[n for n in names if n!=held]
    f,pv=f_classif(np.vstack([NPY[n][0] for n in tr]),np.concatenate([NPY[n][1] for n in tr])); F[held]=f; PV[held]=pv
lines=open("data/GPL17077.txt",encoding="latin-1").read().splitlines()
s=next(i for i,l in enumerate(lines) if l.startswith("!platform_table_begin"))+1
e=next((i for i,l in enumerate(lines) if l.startswith("!platform_table_end")),len(lines))
ann=pd.read_csv(io.StringIO("\n".join(lines[s:e])),sep="\t",low_memory=False)
KEY=["immune","inflammat","cytokine","chemokine","interferon","lymphocyte","leukocyte","antigen","T cell","B cell","apoptot","apoptosis","cell cycle","cell prolifer","cell migration","angiogen","wnt","notch","p53","mapk","nf-kappab","growth factor","cell adhesion","extracellular matrix","oncogen","tumor","carcinog"]
poolB=sorted(set(ann[ann["GO_ID"].astype(str).str.contains("|".join(KEY),case=False,na=False)]["GENE_SYMBOL"].dropna())&set(genes))
iA=np.arange(len(genes)); iB=np.array([gi[g] for g in poolB]); iC={h:np.where(PV[h]<0.05)[0] for h in names}
print(f"pool A {len(iA)} | pool B {len(iB)} | pool C in-fold {[len(iC[h]) for h in names]}",flush=True)
def null(pool,k): return np.array([cv({h:rng.choice(pool[h] if isinstance(pool,dict) else pool,size=k,replace=False) for h in names}) for _ in range(NREP)])
emp=lambda nd,o:(1+(nd>=o).sum())/(1+NREP)
PANELS={"Yi 2025":["IL6","FCGR3A","CCL3","CXCR4","CXCL9","CCL4","CCR1"],
 "Ji 2024":["PTPRC","PLEK","CD53","CD48","LCP1","HCLS1","IKZF1","BTK","VAV1"],
 "Jin 2022":["MYC","SOX2","CDX2","TBX3","KRT7","CDKN2A","MUC5AC"],
 "Rezaei 2021":["ITGAX","CCL14","ADHFE1","HOXB13"],
 "Luu 2024":["SPP1","INHBA","MMP7","THBS2","FAP"],
 "Li 2025":["FCRL3","EFEMP1","ANKRD29","STOX2"]}
rows=[];cache={}
for nm,gl in PANELS.items():
    idx=np.array([gi[g] for g in gl if g in gi]); k=len(idx); q=cv({h:idx for h in names})
    if k not in cache: cache[k]=(null(iA,k),null(iB,k),null(iC,k))
    a,b,c=cache[k]
    rows.append(dict(panel=nm,k_claimed=len(gl),k=k,QWK=q,p_A=emp(a,q),mean_A=a.mean(),p_B=emp(b,q),mean_B=b.mean(),p_C=emp(c,q),mean_C=c.mean()))
    print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv("results/v5_panels.csv",index=False)
srows=[]
for k in [5,10,25,50,100,200,300,500,1000]:
    q=cv({h:np.argsort(-F[h])[:k] for h in names}); a=null(iA,k); c=null(iC,k)
    srows.append(dict(size=k,ours=q,randA_mean=a.mean(),randA_sd=a.std(),p_A=emp(a,q),randC_mean=c.mean(),randC_sd=c.std(),p_C=emp(c,q)))
    print(srows[-1],flush=True)
pd.DataFrame(srows).to_csv("results/v5_sweep.csv",index=False); print("DONE",flush=True)
