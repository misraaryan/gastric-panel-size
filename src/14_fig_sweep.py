import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
SURF="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; MUTED="#8a8982"
CAT=["#2a78d6","#eb6834","#1baf7a"]
plt.rcParams.update({"figure.facecolor":SURF,"axes.facecolor":SURF,"savefig.facecolor":SURF,
 "font.family":"DejaVu Sans","font.size":9,"axes.edgecolor":"#d8d7d2","axes.linewidth":0.8,
 "axes.labelcolor":INK,"axes.titlesize":10,"axes.titleweight":"bold","xtick.color":INK2,
 "ytick.color":INK2,"xtick.labelsize":8,"ytick.labelsize":8,"grid.color":"#ebeae5",
 "grid.linewidth":0.7,"legend.frameon":False,"legend.fontsize":8,
 "axes.spines.top":False,"axes.spines.right":False})

r=pd.read_csv("results/size_sweep_final.csv"); x=r["size"].to_numpy()
fig,ax=plt.subplots(figsize=(7.2,4.4),dpi=300)

ax.fill_between(x, r.randAll_mean-r.randAll_sd, r.randAll_mean+r.randAll_sd,
                color=MUTED, alpha=0.18, lw=0)
ax.fill_between(x, r.randStage_mean-r.randStage_sd, r.randStage_mean+r.randStage_sd,
                color=CAT[2], alpha=0.14, lw=0)
ax.plot(x, r.randAll_mean, lw=2, color=MUTED, marker="o", ms=5,
        label="random, all genes")
ax.plot(x, r.randStage_mean, lw=2, color=CAT[2], marker="^", ms=6,
        label="random, stage-associated genes (in-fold)")
ax.plot(x, r.ours_QWK, lw=2.4, color=CAT[0], marker="D", ms=7,
        label="selected signature (in-fold)", zorder=5)

# significance ticks
for _,row in r.iterrows():
    if row.p_vs_all<0.05 and row.p_vs_stage<0.05:
        ax.annotate("✓", (row["size"], row.ours_QWK+0.045), ha="center",
                    fontsize=11, color=CAT[0], fontweight="bold")
ax.annotate("✓ = beats BOTH nulls (p<0.05)", (5.4, 0.72), fontsize=7.5,
            color=CAT[0], fontweight="bold")

# crossover
ax.annotate("selection falls below\nthe random mean",
            (1000, r.ours_QWK.iloc[-1]), textcoords="offset points", xytext=(-78,-30),
            fontsize=7.5, color=CAT[1], ha="center",
            arrowprops=dict(arrowstyle="->", color=CAT[1], lw=1.1))

ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels([str(v) for v in x])
ax.set_xlabel("Panel size (number of genes, log scale)")
ax.set_ylabel("Quadratic weighted kappa (leave-one-cohort-out, n=127)")
ax.set_title("Apparent performance rises with panel size — for random genes too", loc="left")
ax.grid(True, alpha=0.6); ax.set_axisbelow(True); ax.set_ylim(0.1,0.80)
ax.legend(loc="upper center", bbox_to_anchor=(0.5,-0.13), ncol=3, fontsize=7.5, columnspacing=1.0)
fig.tight_layout(); fig.savefig("figures/fig2_size_sweep.png",bbox_inches="tight"); plt.close(fig)
print("saved figures/fig2_size_sweep.png")
