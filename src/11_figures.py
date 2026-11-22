"""Step 11: publication figures. Palette from the validated reference instance."""
import gzip, io, numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import confusion_matrix, cohen_kappa_score
import mord

# ---- validated palette ----
SURF = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#8a8982"
CAT  = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # categorical, fixed order
STG  = ["#86b6ef", "#2a78d6", "#104281"]                      # ordinal ramp (validated)
SEQ  = ["#cde2fb","#9ec5f4","#6da7ec","#3987e5","#256abf","#184f95","#0d366b"]

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": "#d8d7d2", "axes.linewidth": 0.8,
    "axes.labelcolor": INK, "axes.titlesize": 10, "axes.titleweight": "bold",
    "xtick.color": INK2, "ytick.color": INK2,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "grid.color": "#ebeae5", "grid.linewidth": 0.7,
    "legend.frameon": False, "legend.fontsize": 8,
    "axes.spines.top": False, "axes.spines.right": False,
})
LES = ["LGIN", "HGIN", "Cancer"]

def parse(path):
    titles, acc, tbl, inb = [], [], [], False
    with gzip.open(path, "rt", encoding="latin-1") as fh:
        for l in fh:
            if l.startswith("!Sample_title"): titles=[x.strip().strip('"') for x in l.rstrip().split("\t")[1:]]
            elif l.startswith("!Sample_geo_accession"): acc=[x.strip().strip('"') for x in l.rstrip().split("\t")[1:]]
            elif l.startswith("!series_matrix_table_begin"): inb=True
            elif l.startswith("!series_matrix_table_end"): inb=False
            elif inb: tbl.append(l)
    X = pd.read_csv(io.StringIO("".join(tbl)), sep="\t", index_col=0)
    X.columns=[c.strip().strip('"') for c in X.columns]
    return X, pd.DataFrame({"gsm":acc,"title":titles}).set_index("gsm")
def lab(t):
    s=t.lower().replace(" ","")
    if "lowgradeintraepithelialneoplasia" in s or "lgin" in s: return "LGIN"
    if "highgradeintraepithelialneoplasia" in s or "hgin" in s: return "HGIN"
    if any(k in s for k in ["intestinalgastriccancer","gastriccancer","carcinoma","egc"]): return "Cancer"
    if "gastritis" in s: return "Gastritis"

X1 = pd.read_pickle("results/X_train.pkl"); X2 = pd.read_pickle("results/X_valid.pkl")
m1 = pd.read_csv("results/meta_train.csv", index_col=0); m2 = pd.read_csv("results/meta_valid.csv", index_col=0)
X3r, m3 = parse("data/GSE87666_series_matrix.txt.gz"); m3["stage"]=m3["title"].map(lab)
probes = X1.index
X3 = X3r.reindex(probes).astype(float); X3 = X3.sub(X3.median(axis=0), axis=1)
p2s = pd.read_csv("results/probe2symbol.csv", index_col=0)["GENE_SYMBOL"].reindex(probes).dropna()

COH = {"GSE130823": (X1, m1), "GSE55696": (X2, m2), "GSE87666": (X3, m3)}
G = {}
for nm,(X,m) in COH.items():
    keep = m["stage"].isin(LES).to_numpy()
    g = X.loc[p2s.index].groupby(p2s.values).mean().T[keep]
    G[nm] = (g, m.loc[keep,"stage"].to_numpy())
genes = sorted(set.intersection(*[set(G[n][0].columns) for n in G]))
genes = [g for g in genes if all(G[n][0][g].notna().all() for n in G)]
for n in G: G[n] = (G[n][0][genes], G[n][1])
names = list(G)

# ============== FIG 1: PCA ==============
Xall = pd.concat([G[n][0] for n in names]); 
coh_lab = np.concatenate([[n]*len(G[n][1]) for n in names])
stg_lab = np.concatenate([G[n][1] for n in names])
Z = PCA(n_components=2, random_state=0).fit(StandardScaler().fit_transform(Xall))
P = Z.transform(StandardScaler().fit_transform(Xall))
ev = Z.explained_variance_ratio_*100

fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.3), dpi=300)
for i,n in enumerate(names):
    s = coh_lab==n
    ax[0].scatter(P[s,0], P[s,1], s=26, c=CAT[i], edgecolor=SURF, linewidth=0.8,
                  label=f"{n} (n={s.sum()})", alpha=0.95)
ax[0].set_title("A  Coloured by cohort", loc="left")
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5,-0.17), ncol=3, fontsize=7, handletextpad=0.2, columnspacing=0.8)
for i,st in enumerate(LES):
    s = stg_lab==st
    ax[1].scatter(P[s,0], P[s,1], s=26, c=STG[i], edgecolor=SURF, linewidth=0.8,
                  label=f"{st} (n={s.sum()})", alpha=0.95)
ax[1].set_title("B  Coloured by stage", loc="left")
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5,-0.17), ncol=3, fontsize=7, handletextpad=0.2, columnspacing=0.8)
for a in ax:
    a.set_xlabel(f"PC1 ({ev[0]:.1f}%)"); a.set_ylabel(f"PC2 ({ev[1]:.1f}%)")
    a.grid(True, alpha=0.6); a.set_axisbelow(True)
fig.tight_layout(); fig.savefig("figures/fig1_pca.png", bbox_inches="tight"); plt.close(fig)
print("fig1 done")

# ============== FIG 2: random-null benchmark (KEY FIGURE) ==============
bm = pd.read_csv("results/panel_benchmark.csv")
nullz = pd.read_csv("results/panel_benchmark_null.csv")
sizes = sorted([int(c.split("_")[1]) for c in nullz.columns])
fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=300)
pos = np.arange(len(sizes))
data = [nullz[f"size_{s}"].dropna().to_numpy() for s in sizes]
vp = ax.violinplot(data, positions=pos, widths=0.72, showextrema=False)
for b in vp["bodies"]:
    b.set_facecolor("#d8d7d2"); b.set_edgecolor("#bfbeb8"); b.set_alpha(0.9); b.set_linewidth(0.8)
for i,d in enumerate(data):
    ax.plot([pos[i]-0.3,pos[i]+0.3],[np.percentile(d,95)]*2, color=MUTED, lw=1.6, zorder=3)
ax.plot([],[], color=MUTED, lw=1.6, label="95th pct of random panels")
ax.plot([],[], marker="s", ls="none", ms=7, color="#d8d7d2", label="random panels (n=100 each)")

ours = bm[bm["panel"].str.startswith("OURS")]
pub  = bm[~bm["panel"].str.startswith("OURS")]
def xof(s): return pos[sizes.index(int(s))] if int(s) in sizes else None
for _,r in pub.iterrows():
    x = xof(r["n_genes_found"])
    if x is None: continue
    ax.scatter(x, r["QWK"], s=58, marker="o", c=CAT[1], edgecolor=SURF, lw=1.0, zorder=5)
    ax.annotate(r["panel"].split(" (")[0], (x, r["QWK"]), textcoords="offset points",
                xytext=(9,-1), fontsize=7, color=INK2, va="center")
for _,r in ours.iterrows():
    x = xof(r["n_genes_found"])
    if x is None: continue
    ax.scatter(x, r["QWK"], s=78, marker="D", c=CAT[0], edgecolor=SURF, lw=1.0, zorder=6)
    ax.annotate("ours", (x, r["QWK"]), textcoords="offset points", xytext=(-26,-2),
                fontsize=7, color=CAT[0], fontweight="bold", va="center")
ax.scatter([],[], s=58, marker="o", c=CAT[1], label="published panel")
ax.scatter([],[], s=78, marker="D", c=CAT[0], label="this study")
ax.set_xticks(pos); ax.set_xticklabels([str(s) for s in sizes])
ax.set_xlabel("Panel size (number of genes)")
ax.set_ylabel("Quadratic weighted kappa (leave-one-cohort-out, n=127)")
ax.set_title("Gene panels versus size-matched random panels", loc="left")
ax.grid(True, axis="y", alpha=0.6); ax.set_axisbelow(True)
ax.legend(loc="lower right", ncol=2)
fig.tight_layout(); fig.savefig("figures/fig2_random_null.png", bbox_inches="tight"); plt.close(fig)
print("fig2 done")

# ============== FIG 3: model comparison ==============
loco = pd.read_csv("results/leave_one_cohort_out.csv")
agg = loco.groupby("model").agg(QWK=("QWK","mean"), acc=("acc","mean"),
                                sev=("severe_err","sum")).reset_index()
order = ["ORDINAL LogisticAT","Multinomial LogReg","Random Forest","XGBoost"]
agg = agg.set_index("model").reindex(order).reset_index()
fig, ax = plt.subplots(1, 2, figsize=(7.2, 3.2), dpi=300)
y = np.arange(len(agg))
for i,(mn,q) in enumerate(zip(agg["model"], agg["QWK"])):
    ax[0].barh(y[i], q, height=0.62, color=CAT[0] if i==0 else "#b9c9dd",
               edgecolor=SURF, linewidth=2)
    ax[0].text(q+0.012, y[i], f"{q:.3f}", va="center", fontsize=8, color=INK)
NAMES={"ORDINAL LogisticAT":"Ordinal AT","Multinomial LogReg":"Multinomial","Random Forest":"Random forest","XGBoost":"XGBoost"}
ax[0].set_yticks(y); ax[0].set_yticklabels([NAMES[m] for m in agg["model"]])
ax[0].invert_yaxis(); ax[0].set_xlim(0, max(agg["QWK"])*1.22)
ax[0].set_xlabel("Mean QWK across held-out cohorts")
ax[0].set_title("A  Staging agreement", loc="left")
ax[0].grid(True, axis="x", alpha=0.6); ax[0].set_axisbelow(True)
for i,(mn,s) in enumerate(zip(agg["model"], agg["sev"])):
    ax[1].barh(y[i], s, height=0.62, color=CAT[1] if s==agg["sev"].min() else "#f3c3ae",
               edgecolor=SURF, linewidth=2)
    ax[1].text(s+0.18, y[i], f"{int(s)}", va="center", fontsize=8, color=INK)
ax[1].set_yticks(y); ax[1].set_yticklabels([]); ax[1].invert_yaxis()
ax[1].set_xlim(0, max(agg["sev"])*1.25)
ax[1].set_xlabel("Two-stage (severe) errors out of 127")
ax[1].set_title("B  Clinically severe errors", loc="left")
ax[1].grid(True, axis="x", alpha=0.6); ax[1].set_axisbelow(True)
fig.tight_layout(); fig.savefig("figures/fig3_models.png", bbox_inches="tight"); plt.close(fig)
print("fig3 done")

# ============== FIG 4: pooled LOCO confusion matrix (ordinal) ==============
yt_all, yp_all = [], []
for held in names:
    tr=[n for n in names if n!=held]
    Xtr = pd.concat([G[n][0] for n in tr]); ytr = np.concatenate(
        [[LES.index(s) for s in G[n][1]] for n in tr])
    Xte = G[held][0]; yte = np.array([LES.index(s) for s in G[held][1]])
    sel = SelectKBest(f_classif, k=50).fit(Xtr.to_numpy(float), ytr)
    A = sel.transform(Xtr.to_numpy(float)); B = sel.transform(Xte.to_numpy(float))
    sc = StandardScaler().fit(A)
    p = mord.LogisticAT(alpha=1.0).fit(sc.transform(A), ytr).predict(sc.transform(B))
    yt_all += list(yte); yp_all += list(np.clip(np.round(p).astype(int),0,2))
yt_all, yp_all = np.array(yt_all), np.array(yp_all)
cm = confusion_matrix(yt_all, yp_all)
cmn = cm/cm.sum(axis=1, keepdims=True)
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list("seqblue", SEQ)
fig, ax = plt.subplots(figsize=(3.9, 3.5), dpi=300)
im = ax.imshow(cmn, cmap=cmap, vmin=0, vmax=1)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{cm[i,j]}\n{cmn[i,j]*100:.0f}%", ha="center", va="center",
                fontsize=8, color="#ffffff" if cmn[i,j]>0.45 else INK)
ax.set_xticks(range(3)); ax.set_xticklabels(LES); ax.set_yticks(range(3)); ax.set_yticklabels(LES)
ax.set_xlabel("Predicted stage"); ax.set_ylabel("True stage")
qwk = cohen_kappa_score(yt_all, yp_all, weights="quadratic")
sev = int(np.sum(np.abs(yt_all-yp_all)>=2))
ax.set_title(f"Ordinal model, pooled LOCO\nQWK {qwk:.3f} · {sev}/127 severe errors", loc="left", fontsize=9)
for sp in ax.spines.values(): sp.set_visible(False)
fig.colorbar(im, ax=ax, fraction=0.045, label="Row fraction")
fig.tight_layout(); fig.savefig("figures/fig4_confusion.png", bbox_inches="tight"); plt.close(fig)
print(f"fig4 done (pooled QWK {qwk:.3f}, severe {sev})")

# ============== FIG 5: gene stage trends ==============
sig = pd.read_csv("results/taskB_signature_full.csv")
top = sig[sig["dir_replicates"] & sig["monotonic"]].sort_values("abs_coef", ascending=False)
pick = [g for g in top["gene"].dropna().unique() if g in genes][:6]
fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.2), dpi=300, sharex=True)
for k,(g,a) in enumerate(zip(pick, axs.ravel())):
    for i,n in enumerate(names):
        vals = [G[n][0][g].to_numpy()[G[n][1]==st].mean() for st in LES]
        a.plot(range(3), vals, marker="o", ms=5, lw=2, color=CAT[i], label=n if k==0 else None)
    a.set_title(g, loc="left", fontsize=9)
    a.set_xticks(range(3)); a.set_xticklabels(LES, fontsize=7)
    a.grid(True, alpha=0.6); a.set_axisbelow(True)
    if k%3==0: a.set_ylabel("Mean expression\n(log ratio)", fontsize=8)
axs.ravel()[0].legend(loc="upper left", fontsize=7)
fig.suptitle("Stage trends of top staging genes, replicated across all three cohorts",
             fontsize=10, fontweight="bold", x=0.01, ha="left")
fig.tight_layout(rect=[0,0,1,0.95]); fig.savefig("figures/fig5_gene_trends.png", bbox_inches="tight"); plt.close(fig)
print("fig5 done:", pick)

# ============== FIG 6: GO enrichment ==============
go = pd.read_csv("results/go_enrichment.csv").head(12).iloc[::-1]
lbl = [t.split("(",1)[1].rstrip(")") if "(" in t else t for t in go["GO"]]
lbl = [l.replace("antigen processing and presentation of peptide or polysaccharide antigen via MHC class II","antigen presentation via MHC class II*") for l in lbl]
fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=300)
yy = np.arange(len(go))
norm = (np.log10(go["fold"]) - np.log10(go["fold"]).min())
norm = norm/norm.max() if norm.max()>0 else norm
cols = [cmap(0.25+0.65*v) for v in norm]
ax.barh(yy, go["fold"], height=0.66, color=cols, edgecolor=SURF, linewidth=2)
for i,(f,q,n) in enumerate(zip(go["fold"], go["q_BH"], go["sig_genes"])):
    ax.text(f+0.6, yy[i], f"{f:.1f}×   n={n}   q={q:.1e}", va="center", fontsize=7, color=INK2)
ax.set_yticks(yy); ax.set_yticklabels(lbl, fontsize=8)
ax.set_xlim(0, go["fold"].max()*1.42)
ax.set_xlabel("Fold enrichment over the analysed background")
ax.set_title("GO enrichment of the staging signature", loc="left")
ax.grid(True, axis="x", alpha=0.6); ax.set_axisbelow(True)
fig.tight_layout(); fig.savefig("figures/fig6_go.png", bbox_inches="tight"); plt.close(fig)
print("fig6 done")
