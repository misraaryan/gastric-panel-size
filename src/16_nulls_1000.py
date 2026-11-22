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

# ---- pool B: immune / cancer GO-annotated ----
lines=open("data/GPL17077.txt",encoding="latin-1").read().splitlines()
s=next(i for i,l in enumerate(lines) if l.startswith("!platform_table_begin"))+1
e=next((i for i,l in enumerate(lines) if l.startswith("!platform_table_end")),len(lines))
ann=pd.read_csv(io.StringIO("\n".join(lines[s:e])),sep="\t",low_memory=False)
KEY=["immune","inflammat","cytokine","chemokine","interferon","lymphocyte","leukocyte",
     "antigen","T cell","B cell","apoptot","apoptosis","cell cycle","cell prolifer",
     "cell migration","angiogen","wnt","notch","p53","mapk","nf-kappab","growth factor",
     "cell adhesion","extracellular matrix","oncogen","tumor","carcinog"]
rel_probes=ann[ann["GO_ID"].astype(str).str.contains("|".join(KEY),case=False,na=False)]["ID"]
poolB=sorted(set(ann[ann["ID"].isin(rel_probes)]["GENE_SYMBOL"].dropna()) & set(genes))

# ---- pool C: stage-associated ----
Xc=pd.concat([G[n][0] for n in names]); yc=np.concatenate([G[n][1] for n in names])
_,pv=f_classif(Xc.to_numpy(float),yc)
poolC=[g for g,p in zip(genes,pv) if p<0.05]
print(f"pool A {len(genes)} | pool B {len(poolB)} | pool C {len(poolC)}",flush=True)
iA=np.arange(len(genes)); iB=np.array([gi[g] for g in poolB]); iC=np.array([gi[g] for g in poolC])

def loco_idx(idx):
    yt,yp=[],[]
    for held in names:
        tr=[n for n in names if n!=held]
        A=np.vstack([NPY[n][0][:,idx] for n in tr]); ytr=np.concatenate([NPY[n][1] for n in tr])
        B=NPY[held][0][:,idx]; yte=NPY[held][1]
        sc=StandardScaler().fit(A)
        pr=mord.LogisticAT(alpha=1.0).fit(sc.transform(A),ytr).predict(sc.transform(B))
        yt+=list(yte); yp+=list(np.clip(np.round(pr).astype(int),0,2))
    yt,yp=np.array(yt),np.array(yp)
    return cohen_kappa_score(yt,yp,weights="quadratic"), accuracy_score(yt,yp)

def loco_select(k):
    yt,yp=[],[]
    for held in names:
        tr=[n for n in names if n!=held]
        Xtr=np.vstack([NPY[n][0] for n in tr]); ytr=np.concatenate([NPY[n][1] for n in tr])
        Xte=NPY[held][0]; yte=NPY[held][1]
        sel=SelectKBest(f_classif,k=min(k,Xtr.shape[1])).fit(Xtr,ytr)
        A=sel.transform(Xtr); B=sel.transform(Xte)
        sc=StandardScaler().fit(A)
        pr=mord.LogisticAT(alpha=1.0).fit(sc.transform(A),ytr).predict(sc.transform(B))
        yt+=list(yte); yp+=list(np.clip(np.round(pr).astype(int),0,2))
    yt,yp=np.array(yt),np.array(yp)
    return cohen_kappa_score(yt,yp,weights="quadratic"), accuracy_score(yt,yp)

def nulldist(pool_idx,k,nrep=NREP):
    return np.array([loco_idx(rng.choice(pool_idx,size=min(k,len(pool_idx)),replace=False))[0]
                     for _ in range(nrep)])

def emp_p(nullq,obs,nrep=NREP):
    return (1+(nullq>=obs).sum())/(1+nrep)

# ================= PART 1: published panels vs three nulls =================
PANELS={
 "Yi 2025":        ["IL6","FCGR3A","CCL3","CXCR4","CXCL9","CCL4","CCR1"],
 "Ji 2024":        ["PTPRC","PLEK","CD53","CD48","LCP1","HCLS1","IKZF1","BTK","VAV1"],
 "Jin 2022":       ["MYC","SOX2","CDX2","TBX3","KRT7","CDKN2A","MUC5AC"],
 "Rezaei 2022":    ["ITGAX","CCL14","ADHFE1","HOXB13"],
 "Luu 2024":       ["SPP1","INHBA","MMP7","THBS2","FAP"],
 "FrontGenet 2025":["FCRL3","EFEMP1","ANKRD29","STOX2"],
}
t0=time.time(); rows=[]; cache={}
for nm,gl in PANELS.items():
    found=[g for g in gl if g in gi]
    q,_=loco_idx(np.array([gi[g] for g in found])); k=len(found)
    for key,pool in [("A",iA),("B",iB),("C",iC)]:
        if (key,k) not in cache: cache[(key,k)]=nulldist(pool,k)
    rows.append(dict(panel=nm, n_claimed=len(gl), n_tested=k, QWK=round(q,3),
        p_A=emp_p(cache[("A",k)],q), nullmean_A=round(cache[("A",k)].mean(),3),
        p_B=emp_p(cache[("B",k)],q), nullmean_B=round(cache[("B",k)].mean(),3),
        p_C=emp_p(cache[("C",k)],q), nullmean_C=round(cache[("C",k)].mean(),3)))
    print(f"  {nm:18s} k={k} QWK {q:.3f}  pA={rows[-1]['p_A']:.4f} pB={rows[-1]['p_B']:.4f} "
          f"pC={rows[-1]['p_C']:.4f}  [{time.time()-t0:.0f}s]",flush=True)
pan=pd.DataFrame(rows).sort_values("QWK",ascending=False)
pan.to_csv("results/matched_null_1000.csv",index=False)
print("\n"+"="*96); print(f"PANELS vs THREE NULLS ({NREP} random panels each)"); print("="*96)
print(pan.to_string(index=False),flush=True)

# ================= PART 2: full size sweep =================
SIZES=[5,10,25,50,100,200,300,500,1000]
t0=time.time(); srows=[]
for k in SIZES:
    q,a=loco_select(k)
    qa=nulldist(iA,k); qc=nulldist(iC,k)
    srows.append(dict(size=k, ours_QWK=round(q,4), ours_acc=round(a,4),
        randAll_mean=round(qa.mean(),4), randAll_p95=round(np.percentile(qa,95),4), randAll_sd=round(qa.std(),4),
        randStage_mean=round(qc.mean(),4), randStage_p95=round(np.percentile(qc,95),4), randStage_sd=round(qc.std(),4),
        p_vs_all=emp_p(qa,q), p_vs_stage=emp_p(qc,q),
        gap_all=round(q-qa.mean(),4), gap_stage=round(q-qc.mean(),4)))
    print(f"size {k:5d} ours {q:.3f} | randA {qa.mean():.3f} p={srows[-1]['p_vs_all']:.4f} "
          f"| randC {qc.mean():.3f} p={srows[-1]['p_vs_stage']:.4f} [{time.time()-t0:.0f}s]",flush=True)
sw=pd.DataFrame(srows); sw.to_csv("results/size_sweep_1000.csv",index=False)
print("\n"+"="*110); print(f"PANEL-SIZE SWEEP ({NREP} random panels per size per pool)"); print("="*110)
print(sw[["size","ours_QWK","ours_acc","randAll_mean","p_vs_all","randStage_mean","p_vs_stage"]].to_string(index=False))
sig=sw[(sw["p_vs_all"]<0.05)&(sw["p_vs_stage"]<0.05)]
print(f"\nsizes clearing BOTH nulls at p<0.05: {sig['size'].tolist() if len(sig) else 'none'}")
best=sw.loc[sw['ours_QWK'].idxmax()]; print(f"best size: {int(best['size'])} (QWK {best['ours_QWK']:.3f})")
print("DONE",flush=True)
