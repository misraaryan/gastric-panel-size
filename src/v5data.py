"""v5 data: deduplicated lesions, two independent platform groups.
Group A = GSE130823 + GSE87666 (GPL17077, same lab, shared samples removed)
Group B = GSE55696 (GPL6480)."""
import sys, os, numpy as np, pandas as pd, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
os.environ["DEDUP"]="1"
from common import coh, common, probes, X1, X2, X3, m1, m2, m3, LES
GROUPS={"GPL17077":["GSE130823","GSE87666"],"GPL6480":["GSE55696"]}
names=list(GROUPS)
def _cat(lst, cols):
    X=pd.concat([coh[c][0][cols] for c in lst]); y=np.concatenate([coh[c][1] for c in lst])
    src=np.concatenate([[c]*len(coh[c][1]) for c in lst]); return X,y,src
# probe level
P={g:_cat(GROUPS[g],common) for g in names}
# gene level (mean of probes per symbol), complete in both groups
p2s=pd.read_csv("results/probe2symbol.csv",index_col=0)["GENE_SYMBOL"].reindex(common).dropna()
Gt={}
for g in names:
    X,y,src=P[g]; Gt[g]=(X[p2s.index].T.groupby(p2s.values).mean().T,y,src)
genes=sorted(set(Gt[names[0]][0].columns)&set(Gt[names[1]][0].columns))
genes=[x for x in genes if all(Gt[g][0][x].notna().all() for g in names)]
G={g:(Gt[g][0][genes],Gt[g][1],Gt[g][2]) for g in names}
gi={x:i for i,x in enumerate(genes)}
NPY={g:(G[g][0].to_numpy(float),G[g][1]) for g in names}
PNP={g:(P[g][0].to_numpy(float),P[g][1]) for g in names}
if __name__=="__main__":
    for g in names: print(g, P[g][0].shape, np.bincount(P[g][1]), dict(zip(*np.unique(P[g][2],return_counts=True))))
    print("probes",len(common),"genes",len(genes))
