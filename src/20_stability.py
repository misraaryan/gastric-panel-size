"""Step 16: nulls at 1,000 permutations.
Supersedes 12_matched_null.py (100 reps) and 13_sweep.py (30 reps).
Identical pipeline; only the permutation count changes. Uses position-indexed
numpy for speed."""
import gzip, io, time, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import cohen_kappa_score, accuracy_score
import mord

LES=["LGIN","HGIN","Cancer"]; NREP=1000; rng=np.random.default_rng(7)

def parse(p):
    t,a,tb,ib=[],[],[],False
    with gzip.open(p,"rt",encoding="latin-1") as fh:
        for l in fh:
            if l.startswith("!Sample_title"): t=[x.strip().strip('"') for x in l.rstrip().split("\t")[1:]]
            elif l.startswith("!Sample_geo_accession"): a=[x.strip().strip('"') for x in l.rstrip().split("\t")[1:]]
            elif l.startswith("!series_matrix_table_begin"): ib=True
            elif l.startswith("!series_matrix_table_end"): ib=False
            elif ib: tb.append(l)
    X=pd.read_csv(io.StringIO("".join(tb)),sep="\t",index_col=0)
    X.columns=[c.strip().strip('"') for c in X.columns]
    return X,pd.DataFrame({"gsm":a,"title":t}).set_index("gsm")

def lab(t):
    s=t.lower().replace(" ","")
    if "lowgradeintraepithelialneoplasia" in s or "lgin" in s: return "LGIN"
    if "highgradeintraepithelialneoplasia" in s or "hgin" in s: return "HGIN"
    if any(k in s for k in ["intestinalgastriccancer","gastriccancer","carcinoma","egc"]): return "Cancer"
    if "gastritis" in s: return "Gastritis"

X1=pd.read_pickle("results/X_train.pkl"); X2=pd.read_pickle("results/X_valid.pkl")
m1=pd.read_csv("results/meta_train.csv",index_col=0); m2=pd.read_csv("results/meta_valid.csv",index_col=0)
X3r,m3=parse("data/GSE87666_series_matrix.txt.gz"); m3["stage"]=m3["title"].map(lab)
probes=X1.index; X3=X3r.reindex(probes).astype(float); X3=X3.sub(X3.median(axis=0),axis=1)
p2s=pd.read_csv("results/probe2symbol.csv",index_col=0)["GENE_SYMBOL"].reindex(probes).dropna()

G={}
for nm,X,m in [("GSE130823",X1,m1),("GSE55696",X2,m2),("GSE87666",X3,m3)]:
    k=m["stage"].isin(LES).to_numpy()
    g=X.loc[p2s.index].groupby(p2s.values).mean().T[k]
    G[nm]=(g,np.array([LES.index(s) for s in m.loc[k,"stage"]]))
genes=sorted(set.intersection(*[set(G[n][0].columns) for n in G]))
genes=[g for g in genes if all(G[n][0][g].notna().all() for n in G)]
for n in G: G[n]=(G[n][0][genes],G[n][1])
names=list(G); gi={g:i for i,g in enumerate(genes)}
NPY={n:(G[n][0].to_numpy(float),G[n][1]) for n in names}
print(f"{len(genes)} genes, {sum(len(G[n][1]) for n in names)} samples",flush=True)


from collections import Counter
for K in [10,50]:
    sets={}
    for held in names:
        tr=[n for n in names if n!=held]
        Xt=np.vstack([NPY[n][0] for n in tr]); yt_=np.concatenate([NPY[n][1] for n in tr])
        F,_=f_classif(Xt,yt_); top=np.argsort(-F)[:K]; sets[held]=[genes[i] for i in top]
    c=Counter(g for v in sets.values() for g in v)
    all3=[g for g,n in c.items() if n==3]; two=[g for g,n in c.items() if n==2]
    J=[len(set(sets[a])&set(sets[b]))/len(set(sets[a])|set(sets[b])) for i,a in enumerate(names) for b in names[i+1:]]
    print(f"K={K}: in all 3 folds {len(all3)}: {all3}; in 2 folds {len(two)}; mean Jaccard {np.mean(J):.2f}")
    for h in names: print("  ",h,sets[h])
