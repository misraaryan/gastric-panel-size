"""v5 immune-score baseline: ESTIMATE immune score and MCP-counter populations, leave-one-platform-out ordinal staging,
nested comparison with published panels, Jonckheere-Terpstra trend test."""
import sys, numpy as np, pandas as pd
from scipy.stats import norm
sys.path.insert(0,"src"); from v5data import *
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score as kap
import mord
gs=pd.read_csv("data/sig/estimate_gene_sets.csv"); mcp=pd.read_csv("data/sig/mcp_genes.txt",sep="\t")
def estimate(X):   # X: samples x genes
    R=X.rank(axis=1)*10000/X.shape[1]; out={}
    for col in ["immune_signature","stromal_signature"]:
        S=[g for g in gs[col] if g in X.columns]; es=[]
        for _,row in R.iterrows():
            o=row.sort_values(ascending=False); hit=o.index.isin(S); v=o.values**0.25
            es.append(((np.cumsum(hit*v)/v[hit].sum())-(np.cumsum(~hit)/(~hit).sum())).sum())
        out[col.split("_")[0]]=np.array(es); print(col,len(S),"of",len(gs[col]),flush=True)
    return pd.DataFrame(out,index=X.index)
S={}
for g in names:
    X=G[g][0]; e=estimate(X)
    for pop,d in mcp.groupby("Cell population"):
        gg=[x for x in d["HUGO symbols"] if x in X.columns]
        if len(gg): e["MCP_"+pop]=X[gg].mean(axis=1)
    e["y"]=G[g][1]; e["grp"]=g; S[g]=e
allS=pd.concat(S.values()); allS.to_csv("results/v5_immune_scores.csv")
print("MCP populations:",[c for c in allS.columns if c.startswith("MCP_")],flush=True)
def cv_feats(cols_fn):
    yt,yp,gg=[],[],[]
    for held in names:
        tr=[n for n in names if n!=held]
        A=np.vstack([cols_fn(n) for n in tr]); ytr=np.concatenate([S[n].y for n in tr]); B=cols_fn(held)
        sc=StandardScaler().fit(A); p=mord.LogisticAT(alpha=1.0).fit(sc.transform(A),ytr).predict(sc.transform(B))
        yt+=list(S[held].y); yp+=list(np.clip(np.round(p).astype(int),0,2)); gg+=[held]*len(B)
    yt,yp=np.array(yt),np.array(yp)
    return dict(QWK=kap(yt,yp,weights="quadratic"),acc=(yt==yp).mean(),MAE=np.abs(yt-yp).mean(),severe=int((np.abs(yt-yp)>=2).sum())),(yt,yp,np.array(gg))
mcpcols=[c for c in allS.columns if c.startswith("MCP_")]
PANELS={"Yi 2025":["IL6","FCGR3A","CCL3","CXCR4","CXCL9","CCL4","CCR1"],
 "Ji 2024":["PTPRC","PLEK","CD53","CD48","LCP1","HCLS1","IKZF1","BTK","VAV1"],
 "Jin 2022":["MYC","SOX2","CDX2","TBX3","KRT7","CDKN2A","MUC5AC"],
 "Rezaei 2022":["ITGAX","CCL14","ADHFE1","HOXB13"],
 "Luu 2024":["SPP1","INHBA","MMP7","THBS2","FAP"],
 "Li 2025":["FCRL3","EFEMP1","ANKRD29","STOX2"]}
rows=[]; preds={}
base={"ESTIMATE immune":lambda n:S[n][["immune"]].values,
      "ESTIMATE immune+stromal":lambda n:S[n][["immune","stromal"]].values,
      "MCP-counter (10 populations)":lambda n:S[n][mcpcols].values}
for nm,fn in base.items():
    r,pr=cv_feats(fn); rows.append(dict(model=nm,**r)); preds[nm]=pr
for nm,gl in PANELS.items():
    gl=[x for x in gl if x in genes]
    r,pr=cv_feats(lambda n:G[n][0][gl].values); rows.append(dict(model=nm,**r)); preds[nm]=pr
    r2,pr2=cv_feats(lambda n:np.hstack([S[n][["immune"]].values,G[n][0][gl].values])); rows.append(dict(model=nm+" + immune",**r2)); preds[nm+" + immune"]=pr2
T=pd.DataFrame(rows); print(T.round(3).to_string(index=False),flush=True)
# bootstrap: immune vs each panel, and panel+immune vs immune
rng=np.random.default_rng(1); yt,_,gg=preds["ESTIMATE immune"]; idx={g:np.where(gg==g)[0] for g in names}
comp=[]
for nm in PANELS:
    for a,b in [(nm,"ESTIMATE immune"),(nm+" + immune","ESTIMATE immune")]:
        ya,pa,_=preds[a]; yb,pb,_=preds[b]; d=[]
        for _ in range(2000):
            ii=np.concatenate([rng.choice(idx[g],len(idx[g])) for g in names])
            d.append(kap(ya[ii],pa[ii],weights="quadratic")-kap(yb[ii],pb[ii],weights="quadratic"))
        comp.append(dict(a=a,b=b,diff=kap(ya,pa,weights="quadratic")-kap(yb,pb,weights="quadratic"),lo=np.percentile(d,2.5),hi=np.percentile(d,97.5)))
C=pd.DataFrame(comp); print(C.round(3).to_string(index=False),flush=True)
T.to_csv("results/v5_immune_models.csv",index=False); C.to_csv("results/v5_immune_nested.csv",index=False)
# Jonckheere-Terpstra trend test (normal approximation with tie correction)
def jt(x,y):
    x=np.asarray(x);y=np.asarray(y);J=0.0
    gs_=[x[y==k] for k in sorted(set(y))]
    for i in range(len(gs_)):
        for j in range(i+1,len(gs_)):
            a,b=gs_[i][:,None],gs_[j][None,:]; J+=(b>a).sum()+0.5*(b==a).sum()
    n=np.array([len(g) for g in gs_]);N=n.sum()
    mu=(N**2-(n**2).sum())/4; var=(N**2*(2*N+3)-(n**2*(2*n+3)).sum())/72
    z=(J-mu)/np.sqrt(var); return z,1-norm.cdf(z)
for g in names+["both"]:
    d=allS if g=="both" else allS[allS.grp==g]
    for col in ["immune","stromal"]+mcpcols:
        z,p=jt(d[col],d.y); print(f"JT {g:9s} {col:40s} z={z:6.2f} p={p:.2g}  means by stage: {d.groupby('y')[col].mean().round(2).tolist()}")
