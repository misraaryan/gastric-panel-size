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


# ---- in-fold stage-associated pool: selected on the two TRAINING cohorts only ----
poolC_fold={}
for held in names:
    tr=[n for n in names if n!=held]
    Xt=np.vstack([NPY[n][0] for n in tr]); yt_=np.concatenate([NPY[n][1] for n in tr])
    _,pv=f_classif(Xt,yt_); poolC_fold[held]=np.where(pv<0.05)[0]
    print(f"held {held}: in-fold pool C = {len(poolC_fold[held])} genes",flush=True)

def loco_folds(idx_by_fold):
    yt,yp=[],[]
    for held in names:
        idx=idx_by_fold[held]; tr=[n for n in names if n!=held]
        A=np.vstack([NPY[n][0][:,idx] for n in tr]); ytr=np.concatenate([NPY[n][1] for n in tr])
        B=NPY[held][0][:,idx]; yte=NPY[held][1]
        sc=StandardScaler().fit(A)
        pr=mord.LogisticAT(alpha=1.0).fit(sc.transform(A),ytr).predict(sc.transform(B))
        yt+=list(yte); yp+=list(np.clip(np.round(pr).astype(int),0,2))
    return cohen_kappa_score(np.array(yt),np.array(yp),weights="quadratic")

def null_infold(k,nrep=NREP):
    return np.array([loco_folds({h:rng.choice(poolC_fold[h],size=k,replace=False) for h in names}) for _ in range(nrep)])
def emp_p(nullq,obs,nrep=NREP): return (1+(nullq>=obs).sum())/(1+nrep)

PANELS={"Yi 2025":["IL6","FCGR3A","CCL3","CXCR4","CXCL9","CCL4","CCR1"],
 "Ji 2024":["PTPRC","PLEK","CD53","CD48","LCP1","HCLS1","IKZF1","BTK","VAV1"],
 "Jin 2022":["MYC","SOX2","CDX2","TBX3","KRT7","CDKN2A","MUC5AC"],
 "Rezaei 2021":["ITGAX","CCL14","ADHFE1","HOXB13"],
 "Luu 2024":["SPP1","INHBA","MMP7","THBS2","FAP"],
 "FrontGenet 2025":["FCRL3","EFEMP1","ANKRD29","STOX2"]}
old=pd.read_csv("results/matched_null_1000.csv").set_index("panel")
rows=[];cache={}
for nm,gl in PANELS.items():
    idx=np.array([gi[g] for g in gl if g in gi]); k=len(idx)
    q=loco_folds({h:idx for h in names})
    if k not in cache: cache[k]=null_infold(k)
    rows.append(dict(panel=nm,k=k,QWK=round(q,3),p_C_old=old.loc[nm,"p_C"],nullmean_C_old=old.loc[nm,"nullmean_C"],
        p_C_infold=emp_p(cache[k],q),nullmean_C_infold=round(cache[k].mean(),3)))
    print(rows[-1],flush=True)
pd.DataFrame(rows).to_csv("results/panels_poolC_infold.csv",index=False)
sw_old=pd.read_csv("results/size_sweep_1000.csv").set_index("size")
srows=[]
for k in [5,10,25,50,100,200,300,500,1000]:
    q=sw_old.loc[k,"ours_QWK"]; qc=null_infold(k)
    srows.append(dict(size=k,ours_QWK=q,randStage_mean_old=sw_old.loc[k,"randStage_mean"],p_stage_old=sw_old.loc[k,"p_vs_stage"],
        randStage_mean_infold=round(qc.mean(),4),randStage_sd_infold=round(qc.std(),4),p_stage_infold=emp_p(qc,q)))
    print(srows[-1],flush=True)
pd.DataFrame(srows).to_csv("results/sweep_poolC_infold.csv",index=False)
print("DONE",flush=True)
