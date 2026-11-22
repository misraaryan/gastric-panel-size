import sys, numpy as np, pandas as pd
sys.path.insert(0,"src"); from v5data import *
from sklearn.feature_selection import f_classif
rng=np.random.default_rng(3); NB=100; K=50
freq={}
for held in names:
    tr=[n for n in names if n!=held]; X=np.vstack([NPY[n][0] for n in tr]); y=np.concatenate([NPY[n][1] for n in tr])
    c=np.zeros(len(genes))
    for b in range(NB):
        ii=np.concatenate([rng.choice(np.where(y==k)[0],(y==k).sum()) for k in range(3)])
        f,_=f_classif(X[ii],y[ii]); c[np.argsort(-f)[:K]]+=1
    freq[held]=c/NB
F=pd.DataFrame(freq,index=genes); F["min"]=F.min(axis=1)
st=F[F["min"]>=0.8].sort_values("min",ascending=False)
print(f"genes selected in >=80% of bootstraps in both folds (top-{K}): {len(st)}")
print(st.round(2).to_string())
for t in [0.5,0.6,0.7,0.8,0.9]: print(t,(F["min"]>=t).sum())
F.to_csv("results/v5_stability.csv")
# per-fold top-10 overlap (single fit)
tops={}
for held in names:
    tr=[n for n in names if n!=held]; f,_=f_classif(np.vstack([NPY[n][0] for n in tr]),np.concatenate([NPY[n][1] for n in tr]))
    tops[held]=set(np.array(genes)[np.argsort(-f)[:10]]); 
a,b=tops.values(); print("top10 per fold:",tops,"Jaccard",len(a&b)/len(a|b))
