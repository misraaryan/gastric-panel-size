"""Step 15: regenerate Fig. 4 (confusion matrix) using the SAME probe-level
pipeline as 09_three_cohort.py, so the figure's reported QWK and severe-error
count match Table II of the paper exactly. Run after 11_figures.py."""
import gzip, io, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import mord

SURF="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"
SEQ=["#cde2fb","#9ec5f4","#6da7ec","#3987e5","#256abf","#184f95","#0d366b"]
plt.rcParams.update({
    "figure.facecolor":SURF,"axes.facecolor":SURF,"savefig.facecolor":SURF,
    "font.family":"DejaVu Sans","font.size":9,
    "axes.edgecolor":"#d8d7d2","axes.linewidth":0.8,"axes.labelcolor":INK,
    "axes.titlesize":10,"axes.titleweight":"bold",
    "xtick.color":INK2,"ytick.color":INK2,"xtick.labelsize":8,"ytick.labelsize":8,
    "axes.spines.top":False,"axes.spines.right":False})
LES=["LGIN","HGIN","Cancer"]

def parse(path):
    titles,acc,tbl,inb=[],[],[],False
    with gzip.open(path,"rt",encoding="latin-1") as fh:
        for l in fh:
            if l.startswith("!Sample_title"): titles=[x.strip().strip('"') for x in l.rstrip().split("\t")[1:]]
            elif l.startswith("!Sample_geo_accession"): acc=[x.strip().strip('"') for x in l.rstrip().split("\t")[1:]]
            elif l.startswith("!series_matrix_table_begin"): inb=True
            elif l.startswith("!series_matrix_table_end"): inb=False
            elif inb: tbl.append(l)
    X=pd.read_csv(io.StringIO("".join(tbl)),sep="\t",index_col=0)
    X.columns=[c.strip().strip('"') for c in X.columns]
    return X, pd.DataFrame({"gsm":acc,"title":titles}).set_index("gsm")

def lab(t):
    s=t.lower().replace(" ","")
    if "lowgradeintraepithelialneoplasia" in s or "lgin" in s: return "LGIN"
    if "highgradeintraepithelialneoplasia" in s or "hgin" in s: return "HGIN"
    if any(k in s for k in ["intestinalgastriccancer","gastriccancer","carcinoma","egc"]): return "Cancer"
    if "gastritis" in s: return "Gastritis"
    return None

# ---- identical loading to 09_three_cohort.py ----
X3,m3 = parse("data/GSE87666_series_matrix.txt.gz")
m3["stage"]=m3["title"].map(lab)
X1=pd.read_pickle("results/X_train.pkl"); X2=pd.read_pickle("results/X_valid.pkl")
m1=pd.read_csv("results/meta_train.csv",index_col=0); m2=pd.read_csv("results/meta_valid.csv",index_col=0)
probes=X1.index
X3=X3.reindex(probes).astype(float); X3=X3.sub(X3.median(axis=0),axis=1)

coh={}
for nm,X,m in [("GSE130823",X1,m1),("GSE55696",X2,m2),("GSE87666",X3,m3)]:
    keep=m["stage"].isin(LES).to_numpy()
    Xs=X.T[keep]; good=Xs.columns[Xs.notna().all(axis=0)]
    coh[nm]=(Xs, m.loc[keep,"stage"].map({s:i for i,s in enumerate(LES)}).to_numpy(), good)
common=set(probes)
for nm in coh: common &= set(coh[nm][2])
common=sorted(common)

names=list(coh); yt_all=[]; yp_all=[]; fold=[]
for held in names:
    tr=[n for n in names if n!=held]
    Xtr=pd.concat([coh[n][0][common] for n in tr]).to_numpy(float)
    ytr=np.concatenate([coh[n][1] for n in tr])
    Xte=coh[held][0][common].to_numpy(float); yte=coh[held][1]
    sel=SelectKBest(f_classif,k=min(300,len(common))).fit(Xtr,ytr)
    A,B=sel.transform(Xtr),sel.transform(Xte)
    sc=StandardScaler().fit(A)
    p=np.clip(np.round(mord.LogisticAT(alpha=1.0)
        .fit(sc.transform(A),ytr).predict(sc.transform(B))).astype(int),0,2)
    fold.append(cohen_kappa_score(yte,p,weights="quadratic"))
    yt_all+=list(yte); yp_all+=list(p)

yt_all=np.array(yt_all); yp_all=np.array(yp_all)
cm=confusion_matrix(yt_all,yp_all); cmn=cm/cm.sum(axis=1,keepdims=True)
qwk=float(np.mean(fold)); sev=int(np.sum(np.abs(yt_all-yp_all)>=2))

cmap=LinearSegmentedColormap.from_list("seqblue",SEQ)
fig,ax=plt.subplots(figsize=(3.9,3.5),dpi=300)
im=ax.imshow(cmn,cmap=cmap,vmin=0,vmax=1)
for i in range(3):
    for j in range(3):
        ax.text(j,i,f"{cm[i,j]}\n{cmn[i,j]*100:.0f}%",ha="center",va="center",
                fontsize=8,color="#ffffff" if cmn[i,j]>0.45 else INK)
ax.set_xticks(range(3)); ax.set_xticklabels(LES)
ax.set_yticks(range(3)); ax.set_yticklabels(LES)
ax.set_xlabel("Predicted stage"); ax.set_ylabel("True stage")
ax.set_title(f"Ordinal model, leave-one-cohort-out\nmean QWK {qwk:.3f} · {sev}/127 severe errors",
             loc="left",fontsize=9)
for sp in ax.spines.values(): sp.set_visible(False)
fig.colorbar(im,ax=ax,fraction=0.045,label="Row fraction")
fig.tight_layout(); fig.savefig("figures/fig4_confusion.png",bbox_inches="tight"); plt.close(fig)
print(f"fig4 regenerated: per-fold QWK {[round(f,3) for f in fold]}")
print(f"  mean QWK {qwk:.3f}  severe {sev}/{len(yt_all)}  (Table II: 0.589 / 6)")
print(cm)
