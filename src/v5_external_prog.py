"""GSE78523: can expression separate metaplasia that progressed to cancer from metaplasia that did not? (n=30, AUC)"""
import numpy as np, pandas as pd, sys, warnings; warnings.filterwarnings("ignore")
exec(open("src/v5_external.py").read().split("rng=np.random")[0])
from sklearn.linear_model import LogisticRegression
k_=y>0; Xi=Xg[k_]; yi=(y[k_]==2).astype(int)
rng=np.random.default_rng(5)
def auc(idx_fn,seed):
    sk=StratifiedKFold(5,shuffle=True,random_state=seed); s=np.zeros(len(yi))
    for tr,te in sk.split(Xi,yi):
        idx=idx_fn(tr); sc=StandardScaler().fit(Xi[tr][:,idx])
        s[te]=LogisticRegression(C=0.1,max_iter=5000).fit(sc.transform(Xi[tr][:,idx]),yi[tr]).decision_function(sc.transform(Xi[te][:,idx]))
    return roc_auc_score(yi,s)
for k in [5,10,50,200,1000]:
    def sel(tr,k=k):
        F,_=f_classif(Xi[tr],yi[tr]); return np.argsort(-np.nan_to_num(F))[:k]
    o=np.mean([auc(sel,s_) for s_ in range(10)]); nd=np.array([auc(lambda tr,k=k:rng.choice(len(genes),k,replace=False),int(rng.integers(1e9))) for _ in range(200)])
    print(dict(size=k,ours_auc=round(o,3),rand_mean=round(nd.mean(),3),p=round((1+(nd>=o).sum())/201,3)),flush=True)
