"""
Step 12: HARDER null. Instead of drawing random panels from all genes, draw them
from biologically relevant pools:
  Null A (previous): all genes
  Null B: genes annotated to immune / inflammation / cancer-related GO terms
  Null C: genes nominally stage-associated in training data (ANOVA p<0.05)
"""
import gzip, io, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import f_classif
from sklearn.metrics import cohen_kappa_score
from scipy import stats
import mord

LES = ["LGIN","HGIN","Cancer"]
rng = np.random.default_rng(1)

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
    return X, pd.DataFrame({"gsm":a,"title":t}).set_index("gsm")
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

# ---------- build the biologically relevant pool from GO annotations ----------
lines=open("data/GPL17077.txt",encoding="latin-1").read().splitlines()
s=next(i for i,l in enumerate(lines) if l.startswith("!platform_table_begin"))+1
e=next((i for i,l in enumerate(lines) if l.startswith("!platform_table_end")),len(lines))
ann=pd.read_csv(io.StringIO("\n".join(lines[s:e])),sep="\t",low_memory=False)
KEY=["immune","inflammat","cytokine","chemokine","interferon","lymphocyte","leukocyte",
     "antigen","T cell","B cell","apoptot","apoptosis","cell cycle","cell prolifer",
     "cell migration","angiogen","wnt","notch","p53","mapk","nf-kappab","growth factor",
     "cell adhesion","extracellular matrix","oncogen","tumor","carcinog"]
pat="|".join(KEY)
rel_probes=ann[ann["GO_ID"].astype(str).str.contains(pat,case=False,na=False)]["ID"]
rel_genes=sorted(set(ann[ann["ID"].isin(rel_probes)]["GENE_SYMBOL"].dropna()) & set(genes))
print(f"total analysable genes            : {len(genes)}")
print(f"POOL B (immune/cancer GO-annotated): {len(rel_genes)}")

# ---------- pool C: nominally stage-associated genes ----------
Xc=pd.concat([G[n][0] for n in names]); yc=np.concatenate([G[n][1] for n in names])
F,p=f_classif(Xc.to_numpy(float),yc)
poolC=[g for g,pv in zip(genes,p) if pv<0.05]
print(f"POOL C (stage-associated p<0.05)   : {len(poolC)}")

def loco_qwk(gl):
    yt,yp=[],[]
    for held in names:
        tr=[n for n in names if n!=held]
        A=pd.concat([G[n][0] for n in tr])[gl].to_numpy(float)
        ytr=np.concatenate([G[n][1] for n in tr])
        B=G[held][0][gl].to_numpy(float); yte=G[held][1]
        sc=StandardScaler().fit(A)
        pr=mord.LogisticAT(alpha=1.0).fit(sc.transform(A),ytr).predict(sc.transform(B))
        yt+=list(yte); yp+=list(np.clip(np.round(pr).astype(int),0,2))
    return cohen_kappa_score(yt,yp,weights="quadratic")

PANELS={
 "Yi 2025":        ["IL6","FCGR3A","CCL3","CXCR4","CXCL9","CCL4","CCR1"],
 "Ji 2024":        ["PTPRC","PLEK","CD53","CD48","LCP1","HCLS1","IKZF1","BTK","VAV1"],
 "Jin 2022":       ["MYC","SOX2","CDX2","TBX3","KRT7","CDKN2A","MUC5AC"],
 "Rezaei 2021":    ["ITGAX","CCL14","ADHFE1","HOXB13"],
 "Luu 2024":       ["SPP1","INHBA","MMP7","THBS2","FAP"],
 "FrontGenet 2025":["FCRL3","EFEMP1","ANKRD29","STOX2"],
}
obs={k:(loco_qwk([g for g in v if g in genes]), len([g for g in v if g in genes]))
     for k,v in PANELS.items()}

NREP=100
pools={"A all genes":genes, "B immune/cancer GO":rel_genes, "C stage-associated":poolC}
sizes=sorted(set(n for _,n in obs.values()))
null={}
for pname,pool in pools.items():
    for sz in sizes:
        qs=[loco_qwk(list(rng.choice(pool,size=sz,replace=False))) for _ in range(NREP)]
        null[(pname,sz)]=np.array(qs)
    print(f"pool {pname}: done")

print("\n"+"="*94)
print("PANELS AGAINST THREE NULLS (empirical p; 100 random panels each)")
print("="*94)
rows=[]
for k,(q,sz) in obs.items():
    r={"panel":k,"n":sz,"QWK":round(q,3)}
    for pname in pools:
        d=null[(pname,sz)]
        r[f"p_{pname.split()[0]}"]=round((1+(d>=q).sum())/(1+len(d)),3)
        r[f"nullmean_{pname.split()[0]}"]=round(d.mean(),3)
    rows.append(r)
res=pd.DataFrame(rows).sort_values("QWK",ascending=False)
print(res.to_string(index=False))

print("\n--- how much each pool raises the bar (mean random QWK) ---")
for sz in sizes:
    a,b,c=[null[(p,sz)].mean() for p in pools]
    print(f"size {sz:2d}:  all genes {a:.3f}  ->  immune/cancer {b:.3f}  ->  stage-assoc {c:.3f}")
res.to_csv("results/matched_null_benchmark.csv",index=False)
