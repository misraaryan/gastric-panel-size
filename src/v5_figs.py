import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
SURF="#ffffff"; INK="#0b0b0b"; INK2="#52514e"; MUTED="#8a8982"; CAT=["#2a78d6","#eb6834","#1baf7a"]
plt.rcParams.update({"figure.facecolor":SURF,"axes.facecolor":SURF,"savefig.facecolor":SURF,"font.family":"DejaVu Sans","font.size":9,
 "axes.edgecolor":"#bdbcb6","axes.linewidth":0.8,"axes.labelcolor":INK,"axes.titlesize":9.5,"axes.titleweight":"bold","xtick.color":INK2,
 "ytick.color":INK2,"xtick.labelsize":8,"ytick.labelsize":8,"legend.frameon":False,"legend.fontsize":7.5,"axes.spines.top":False,"axes.spines.right":False})
# ---- Fig 1: duplicate evidence ----
w=pd.read_csv("results/within_cohort_correlations.csv"); dup=pd.read_csv("results/duplicate_pairs.csv")
fig,(ax,rg)=plt.subplots(2,1,figsize=(3.5,2.6),dpi=300,sharex=True,gridspec_kw={"height_ratios":[3,1.1],"hspace":0.08})
bins=np.linspace(-0.5,1.0,61)
dif=w[w.kind=="different"].r; same=w[w.kind=="same"].r
ax.hist(dif,bins=bins,color=MUTED,alpha=0.6,density=True)
ax.axvline(dif.max(),color=INK2,lw=0.8,ls=":")
ax.text(dif.max()+0.02,ax.get_ylim()[1]*0.9,f"max {dif.max():.2f}",fontsize=6.5,color=INK2,va="top")
ax.set_ylabel("Density"); ax.tick_params(axis="x",length=0)
ax.text(0.30,1.75,"same cohort,\ndifferent patients\n(n=4,552)",fontsize=6.5,color=INK2,va="top")
for y,vals,col,lab in [(1,same,CAT[0],"same patient (n=50)"),(0,dup.r,CAT[1],"duplicates (n=19)")]:
    rg.vlines(vals,y-0.32,y+0.32,color=col,lw=0.9)
rg.set_yticks([1,0]); rg.set_yticklabels(["same patient\n(n=50)","cross-cohort\nduplicates (n=19)"],fontsize=6.5)
rg.tick_params(axis="y",length=0); rg.set_ylim(-0.6,1.6); rg.spines["left"].set_visible(False)
rg.axvline(dif.max(),color=INK2,lw=0.8,ls=":")
rg.set_xlim(-0.5,1.0); rg.set_xlabel("Pearson correlation between two samples")
fig.savefig("figures/v5_fig1_duplicates.png",bbox_inches="tight"); plt.close()
# ---- Fig 2: size sweep, (a) precancer cohorts, (b) GSE78523 ----
r=pd.read_csv("results/v5_sweep.csv"); x=r["size"].to_numpy(); imm=pd.read_csv("results/v5_immune_models.csv").set_index("model").loc["ESTIMATE immune","QWK"]
e=pd.read_csv("results/v5_external_GSE78523.csv")
fig,axs=plt.subplots(1,2,figsize=(7.2,2.6),dpi=300,sharey=True)
ax=axs[0]
ax.fill_between(x,r.randA_mean-r.randA_sd,r.randA_mean+r.randA_sd,color=MUTED,alpha=0.2,lw=0)
ax.fill_between(x,r.randC_mean-r.randC_sd,r.randC_mean+r.randC_sd,color=CAT[2],alpha=0.15,lw=0)
ax.plot(x,r.randA_mean,lw=1.6,color=MUTED,marker="o",ms=3.5,label="random genes, all")
ax.plot(x,r.randC_mean,lw=1.6,color=CAT[2],marker="^",ms=4,label="random genes, stage-associated")
ax.plot(x,r.ours,lw=1.8,color=CAT[0],marker="D",ms=4,label="selected genes (in-fold)",zorder=5)
ax.axhline(imm,color=CAT[1],ls="--",lw=1.3,label=f"one immune score")
ax.set_title("(a) LGIN, HGIN, early cancer (n=109)",fontsize=8.5)
ax2=axs[1]; xe=e["size"].to_numpy()
ax2.fill_between(xe,e.rand_mean-e.rand_sd,e.rand_mean+e.rand_sd,color=MUTED,alpha=0.2,lw=0)
ax2.plot(xe,e.rand_mean,lw=1.6,color=MUTED,marker="o",ms=3.5)
ax2.plot(xe,e.ours,lw=1.8,color=CAT[0],marker="D",ms=4,zorder=5)
ax2.set_title("(b) Independent cohort GSE78523 (n=45)",fontsize=8.5)
for a_,xx in zip(axs,[x,xe]):
    a_.set_xscale("log"); a_.set_xticks(xx); a_.set_xticklabels([str(v) for v in xx],fontsize=7); a_.minorticks_off(); a_.set_xlabel("Panel size (genes)")
axs[0].set_ylabel("QWK (held out)"); axs[0].set_ylim(0,0.8)
fig.legend(*axs[0].get_legend_handles_labels(),loc="lower center",ncol=4,fontsize=7,bbox_to_anchor=(0.5,-0.04))
fig.tight_layout(rect=(0,0.07,1,1)); fig.savefig("figures/v5_fig2_sweep.png",bbox_inches="tight"); plt.close()
# ---- Fig 3: confusion matrices ----
d=pd.read_csv("results/v5_predictions.csv"); L=["LGIN","HGIN","Cancer"]
fig,axs=plt.subplots(1,2,figsize=(3.5,1.9),dpi=300)
for ax,m in zip(axs,["Ordinal","Multinomial"]):
    s=d[d.model==m]; C=pd.crosstab(s.y,s.p).reindex(index=[0,1,2],columns=[0,1,2],fill_value=0).values
    ax.imshow(C,cmap="Blues",vmin=0,vmax=C.max()*1.15)
    for i in range(3):
        for j in range(3):
            ax.text(j,i,C[i,j],ha="center",va="center",fontsize=8,color=(CAT[1] if abs(i-j)==2 and C[i,j] else (INK if C[i,j]<C.max()*0.6 else "white")),fontweight=("bold" if abs(i-j)==2 else None))
    sev=int((abs(s.y-s.p)>=2).sum())
    ax.set_title(f"{m} ({sev} severe)",fontsize=8); ax.set_xticks(range(3)); ax.set_yticks(range(3))
    ax.set_xticklabels(L,fontsize=6.5); ax.set_yticklabels(L if m=="Ordinal" else [],fontsize=6.5)
    ax.set_xlabel("Predicted",fontsize=7)
    for sp in ax.spines.values(): sp.set_visible(False)
axs[0].set_ylabel("True",fontsize=7)
fig.tight_layout(); fig.savefig("figures/v5_fig3_confusion.png",bbox_inches="tight"); plt.close()
print("ok")
