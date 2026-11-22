"""Step 17: clinical baseline. Stage from age and sex ALONE, under the same
leave-one-cohort-out protocol, to establish what the transcriptome has to beat."""
import gzip, io, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, f1_score, cohen_kappa_score,
                             mean_absolute_error)
import mord

LES=["LGIN","HGIN","Cancer"]

def parse_meta(path):
    """pull sample ids, titles and every characteristics row"""
    acc=titles=None; chars=[]
    with gzip.open(path,"rt",encoding="latin-1") as fh:
        for l in fh:
            if l.startswith("!Sample_geo_accession"):
                acc=[x.strip().strip('"') for x in l.rstrip().split("\t")[1:]]
            elif l.startswith("!Sample_title"):
                titles=[x.strip().strip('"') for x in l.rstrip().split("\t")[1:]]
            elif l.startswith("!Sample_characteristics_ch1"):
                chars.append([x.strip().strip('"') for x in l.rstrip().split("\t")[1:]])
            elif l.startswith("!series_matrix_table_begin"): break
    m=pd.DataFrame({"gsm":acc,"title":titles}).set_index("gsm")
    for row in chars:
        if not row: continue
        keys=[c.split(":")[0].strip().lower() for c in row if ":" in c]
        if not keys: continue
        key=max(set(keys),key=keys.count)
        vals=[c.split(":",1)[1].strip() if ":" in c else None for c in row]
        if key in ("age","gender","sex"): m[key]=vals
    return m

def lab(t):
    s=str(t).lower().replace(" ","")
    if "lowgradeintraepithelialneoplasia" in s or "lgin" in s: return "LGIN"
    if "highgradeintraepithelialneoplasia" in s or "hgin" in s: return "HGIN"
    if any(k in s for k in ["intestinalgastriccancer","gastriccancer","carcinoma","egc"]): return "Cancer"
    if "gastritis" in s: return "Gastritis"

FILES={"GSE130823":"data/GSE130823_series_matrix.txt.gz",
       "GSE55696":"data/GSE55696_series_matrix.txt.gz",
       "GSE87666":"data/GSE87666_series_matrix.txt.gz"}
coh={}
for nm,f in FILES.items():
    m=parse_meta(f)
    m["stage"]=m["title"].map(lab)
    if "stage" not in m or m["stage"].isna().all():
        raise SystemExit(f"{nm}: no stages parsed")
    # GSE87666 carries stage in a characteristics row, title may be terse
    if m["stage"].isna().any():
        for c in m.columns:
            if c.startswith("patholog"): m["stage"]=m["stage"].fillna(m[c].map(lab))
    sexcol="gender" if "gender" in m else ("sex" if "sex" in m else None)
    keep=m["stage"].isin(LES) & m["age"].notna() & m[sexcol].notna()
    mm=m[keep].copy()
    mm["age_num"]=pd.to_numeric(mm["age"],errors="coerce")
    mm["male"]=mm[sexcol].str.lower().str.startswith("m").astype(int)
    mm=mm[mm["age_num"].notna()]
    coh[nm]=mm
    print(f"{nm}: {len(mm)} lesion samples with age+sex "
          f"(age {mm['age_num'].mean():.1f}±{mm['age_num'].std():.1f}, {mm['male'].sum()} male)")

tot=sum(len(v) for v in coh.values())
print(f"\ntotal with complete age+sex: {tot}")

names=list(coh)
MODELS={
  "Majority class":        ("dummy", None),
  "Age + sex, multinomial":("lin", LogisticRegression(max_iter=5000,class_weight="balanced")),
  "Age + sex, ordinal":    ("lin", mord.LogisticAT(alpha=1.0)),
  "Age only, ordinal":     ("age", mord.LogisticAT(alpha=1.0)),
}
rows=[]
for mn,(kind,proto) in MODELS.items():
    yt,yp=[],[]
    for held in names:
        tr=[n for n in names if n!=held]
        def feats(d):
            if kind=="age": return d[["age_num"]].to_numpy(float)
            return d[["age_num","male"]].to_numpy(float)
        Atr=feats(pd.concat([coh[n] for n in tr]))
        ytr=np.concatenate([[LES.index(s) for s in coh[n]["stage"]] for n in tr])
        Ate=feats(coh[held]); yte=np.array([LES.index(s) for s in coh[held]["stage"]])
        if kind=="dummy":
            md=DummyClassifier(strategy="most_frequent").fit(Atr,ytr); pr=md.predict(Ate)
        else:
            sc=StandardScaler().fit(Atr)
            import copy; md=copy.deepcopy(proto).fit(sc.transform(Atr),ytr)
            pr=np.clip(np.round(md.predict(sc.transform(Ate))).astype(int),0,2)
        yt+=list(yte); yp+=list(pr)
    yt,yp=np.array(yt),np.array(yp)
    rows.append(dict(model=mn, n=len(yt),
        acc=round(accuracy_score(yt,yp),3),
        macroF1=round(f1_score(yt,yp,average="macro"),3),
        QWK=round(cohen_kappa_score(yt,yp,weights="quadratic"),3),
        stageMAE=round(mean_absolute_error(yt,yp),3),
        severe=int(np.sum(np.abs(yt-yp)>=2))))
    print(f"  {mn:24s} acc {rows[-1]['acc']:.3f}  QWK {rows[-1]['QWK']:.3f}  "
          f"MAE {rows[-1]['stageMAE']:.3f}  severe {rows[-1]['severe']}")

df=pd.DataFrame(rows); df.to_csv("results/clinical_baseline.csv",index=False)
print("\n"+"="*86); print("CLINICAL BASELINE — leave-one-cohort-out, age and sex only"); print("="*86)
print(df.to_string(index=False))
print("\nFor comparison, transcriptomic models on the same task:")
print("  Ordinal, 300 probes      acc 0.645  QWK 0.589  MAE 0.401  severe 6")
print("  Yi 2025, 6 genes         acc 0.630  QWK 0.649  MAE 0.386  severe 2")
