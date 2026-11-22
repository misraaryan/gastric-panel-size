"""Independent check of the panel-size effect in GSE78523 (Spain; Affymetrix Almac Xcel):
ordinal task healthy mucosa (0) -> intestinal metaplasia without progression (1) -> metaplasia that progressed to cancer (2)."""
import gzip, io, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score as kap, roc_auc_score
from sklearn.linear_model import LogisticRegression
import mord
L=gzip.open("data/ext/GSE78523_series_matrix.txt.gz","rt").read().splitlines()
s=L.index("!series_matrix_table_begin")+1; e=L.index("!series_matrix_table_end")
X=pd.read_csv(io.StringIO("\n".join(L[s:e])),sep="\t",index_col=0)
m=pd.read_csv("data/ext/GSE78523_meta.csv",index_col=0).loc[X.columns]
lab={"Healthy Control":0,"CIM Control":1,"IIM Control":1,"CIM-GC":2,"IIM-GC":2}; y=m.status.map(lab).values
ann=pd.read_csv("data/ext/GPL18990_symbols.csv")
ann=ann[~ann["Gene Symbol"].str.contains("///")].set_index("ID")["Gene Symbol"]
ids=X.index.intersection(ann.index); print(f"probes {X.shape[0]}, annotated via GPL18990 {len(ids)}",flush=True)
G=X.loc[ids].groupby(ann[ids].values).mean().T; G=G.loc[:,G.notna().all()]; Xg=G.values; genes=G.columns; print("genes",len(genes),"samples",Xg.shape[0],np.bincount(y),flush=True)
rng=np.random.default_rng(21)
def cv(idx_fn, seed):
    sk=StratifiedKFold(5,shuffle=True,random_state=seed); yp=np.zeros_like(y)
    for tr,te in sk.split(Xg,y):
        idx=idx_fn(tr); A=Xg[tr][:,idx]; B=Xg[te][:,idx]; sc=StandardScaler().fit(A)
        yp[te]=np.clip(np.round(mord.LogisticAT(alpha=1.0).fit(sc.transform(A),y[tr]).predict(sc.transform(B))).astype(int),0,2)
    return kap(y,yp,weights="quadratic")
def sel(k):
    def f(tr):
        F,_=f_classif(Xg[tr],y[tr]); return np.argsort(-np.nan_to_num(F))[:k]
    return f
NREP=500; rows=[]
for k in [5,10,25,50,100,200,500,1000]:
    ours=np.mean([cv(sel(k),s_) for s_ in range(10)])
    nd=np.array([cv(lambda tr,k=k:rng.choice(len(genes),k,replace=False),int(rng.integers(1e9))) for _ in range(NREP)])
    rows.append(dict(size=k,ours=ours,rand_mean=nd.mean(),rand_sd=nd.std(),p=(1+(nd>=ours).sum())/(1+NREP))); print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv("results/v5_external_GSE78523.csv",index=False); print("DONE")
