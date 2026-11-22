"""
Step 13: full panel-size sweep.
For each panel size: our in-fold-selected signature vs random panels drawn from
 (A) all genes and (B) stage-associated genes. Leave-one-cohort-out throughout.
"""
import gzip, io, numpy as np, pandas as pd, warnings, time
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, f_classif as fc
from sklearn.metrics import cohen_kappa_score, accuracy_score
import mord

LES=["LGIN","HGIN","Cancer"]; rng=np.random.default_rng(7)
def parse(p):
    t,a,tb,ib=[],[],[],False
    with gzip.open(p,"rt",encoding="latin-1") as fh:
        for l in fh:
            if l.startswith("!Sample_title"): t=[x.strip().strip('"') for x in l.rstrip().split("\t")[1:]]
            elif l.startswith("!Sample_geo_accession"): a=[x.strip().strip('"') for x in l.rstrip().split("\t")[1:]]
            elif l.startswith("!series_matrix_table_begin"): ib=True
            elif l.startswith("!series_matrix_table_end"): ib=False
            elif ib: tb.append(l)
    X=pd.read_csv(io.StringIO("".join(tb)),sep="\t",index_col=0); X.columns=[c.strip().strip('"') for c in X.columns]
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
names=list(G)
NP=len(genes); print(f"{NP} genes, {sum(len(G[n][1]) for n in names)} samples", flush=True)

Xc=pd.concat([G[n][0] for n in names]); yc=np.concatenate([G[n][1] for n in names])
_,pv=fc(Xc.to_numpy(float),yc)
poolS=[g for g,p in zip(genes,pv) if p<0.05]
print(f"stage-associated pool: {len(poolS)}", flush=True)

def loco(gl=None, k=None):
    yt,yp=[],[]
    for held in names:
        tr=[n for n in names if n!=held]
        Xtr=pd.concat([G[n][0] for n in tr]); ytr=np.concatenate([G[n][1] for n in tr])
        Xte=G[held][0]; yte=G[held][1]
        if k is not None:
            sel=SelectKBest(f_classif,k=min(k,Xtr.shape[1])).fit(Xtr.to_numpy(float),ytr)
            A=sel.transform(Xtr.to_numpy(float)); B=sel.transform(Xte.to_numpy(float))
        else:
            A=Xtr[gl].to_numpy(float); B=Xte[gl].to_numpy(float)
        sc=StandardScaler().fit(A)
        pr=mord.LogisticAT(alpha=1.0).fit(sc.transform(A),ytr).predict(sc.transform(B))
        yt+=list(yte); yp+=list(np.clip(np.round(pr).astype(int),0,2))
    yt,yp=np.array(yt),np.array(yp)
    return cohen_kappa_score(yt,yp,weights="quadratic"), accuracy_score(yt,yp)

SIZES=[5,10,25,50,100,200,300,500,1000]
NREP=30
rows=[]
t0=time.time()
for k in SIZES:
    q_ours,a_ours = loco(k=k)
    qa=[loco(gl=list(rng.choice(genes,size=k,replace=False)))[0] for _ in range(NREP)]
    qs=[loco(gl=list(rng.choice(poolS,size=min(k,len(poolS)),replace=False)))[0] for _ in range(NREP)]
    qa,qs=np.array(qa),np.array(qs)
    rows.append(dict(size=k, ours_QWK=q_ours, ours_acc=a_ours,
        randAll_mean=qa.mean(), randAll_p95=np.percentile(qa,95), randAll_sd=qa.std(),
        randStage_mean=qs.mean(), randStage_p95=np.percentile(qs,95), randStage_sd=qs.std(),
        p_vs_all=(1+(qa>=q_ours).sum())/(1+NREP),
        p_vs_stage=(1+(qs>=q_ours).sum())/(1+NREP),
        gap_all=q_ours-qa.mean(), gap_stage=q_ours-qs.mean()))
    print(f"size {k:5d}  ours {q_ours:.3f} | randAll {qa.mean():.3f} (p={rows[-1]['p_vs_all']:.3f}) "
          f"| randStage {qs.mean():.3f} (p={rows[-1]['p_vs_stage']:.3f})  [{time.time()-t0:.0f}s]", flush=True)

res=pd.DataFrame(rows); res.to_csv("results/size_sweep.csv",index=False)
print("\n"+"="*100)
print("PANEL-SIZE SWEEP — leave-one-cohort-out, 127 samples, 30 random panels per size")
print("="*100)
print(res[["size","ours_QWK","ours_acc","randAll_mean","p_vs_all","randStage_mean","p_vs_stage",
           "gap_all","gap_stage"]].round(3).to_string(index=False))
best=res.loc[res["ours_QWK"].idxmax()]
print(f"\nbest size by QWK: {int(best['size'])} (QWK {best['ours_QWK']:.3f}, acc {best['ours_acc']:.3f})")
sig=res[(res["p_vs_all"]<0.05)&(res["p_vs_stage"]<0.05)]
if len(sig): print(f"sizes clearing BOTH nulls at p<0.05: {sig['size'].tolist()}")
else: print("no size clears both nulls at p<0.05")
print(f"\ngap vs all-gene null shrinks: {res['gap_all'].iloc[0]:.3f} (size 5) -> {res['gap_all'].iloc[-1]:.3f} (size {int(res['size'].iloc[-1])})")
