"""Step 7: the staging signature — ordinal coefficients, monotonic trends, SHAP."""
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore")
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from scipy import stats
import mord

LES = ["LGIN", "HGIN", "Cancer"]
Xtr = pd.read_pickle("results/X_train.pkl").T
Xva = pd.read_pickle("results/X_valid.pkl").T
mtr = pd.read_csv("results/meta_train.csv", index_col=0).loc[Xtr.index]
mva = pd.read_csv("results/meta_valid.csv", index_col=0).loc[Xva.index]
p2s = pd.read_csv("results/probe2symbol.csv", index_col=0)["GENE_SYMBOL"]

tm = mtr["stage"].isin(LES); vm = mva["stage"].isin(LES)
A = Xtr[tm]; ytr = mtr.loc[tm,"stage"].map({s:i for i,s in enumerate(LES)}).to_numpy()
V = Xva[vm]; yva = mva.loc[vm,"stage"].map({s:i for i,s in enumerate(LES)}).to_numpy()

sel = SelectKBest(f_classif, k=300).fit(A.to_numpy(float), ytr)
probes = A.columns[sel.get_support()]
As = StandardScaler().fit(A[probes].to_numpy(float))
Az = As.transform(A[probes].to_numpy(float)); Vz = As.transform(V[probes].to_numpy(float))

mdl = mord.LogisticAT(alpha=1.0).fit(Az, ytr)
coef = pd.Series(mdl.coef_, index=probes, name="ordinal_coef")

# stage means (training) + monotonic trend test across the 3 lesion stages
rows = []
for p in probes:
    v = A[p].to_numpy(float)
    means = [v[ytr==i].mean() for i in range(3)]
    rho, pv = stats.spearmanr(v, ytr)
    # external replication of direction
    ve = V[p].to_numpy(float)
    rho_e, _ = stats.spearmanr(ve, yva)
    rows.append(dict(probe=p, gene=p2s.get(p, np.nan),
                     coef=coef[p],
                     LGIN=means[0], HGIN=means[1], Cancer=means[2],
                     rho_train=rho, p_train=pv, rho_valid=rho_e,
                     monotonic=(means[0]<means[1]<means[2]) or (means[0]>means[1]>means[2]),
                     dir_replicates=(np.sign(rho)==np.sign(rho_e))))
sig = pd.DataFrame(rows)
sig["abs_coef"] = sig["coef"].abs()
sig = sig.sort_values("abs_coef", ascending=False)
sig.to_csv("results/taskB_signature_full.csv", index=False)

print("="*86)
print("TOP 25 STAGING GENES BY ORDINAL COEFFICIENT")
print("  coef>0 pushes toward a LATER stage; stage means are log-ratio units")
print("="*86)
cols = ["gene","coef","LGIN","HGIN","Cancer","rho_train","rho_valid","monotonic","dir_replicates"]
print(sig[cols].head(25).round(3).to_string(index=False))

print("\n" + "="*86)
print("REPRODUCIBILITY OF THE SIGNATURE")
print("="*86)
print(f"probes with monotonic stage trend (train) : {sig['monotonic'].sum()}/300 "
      f"({100*sig['monotonic'].mean():.0f}%)")
print(f"direction replicates in GSE55696          : {sig['dir_replicates'].sum()}/300 "
      f"({100*sig['dir_replicates'].mean():.0f}%)")
both = (sig['monotonic'] & sig['dir_replicates']).sum()
print(f"monotonic AND replicating                 : {both}/300 ({100*both/300:.0f}%)")
# binomial test vs chance for direction replication
bt = stats.binomtest(int(sig['dir_replicates'].sum()), 300, 0.5, alternative="greater")
print(f"binomial test vs 50% chance               : p = {bt.pvalue:.3e}")

print("\n" + "="*86)
print("GENES UP WITH PROGRESSION (top 15, replicating)")
print("="*86)
up = sig[(sig['rho_train']>0) & sig['dir_replicates']].sort_values("abs_coef", ascending=False)
print(up[["gene","coef","LGIN","HGIN","Cancer","rho_valid"]].head(15).round(3).to_string(index=False))
print("\nGENES DOWN WITH PROGRESSION (top 15, replicating)")
dn = sig[(sig['rho_train']<0) & sig['dir_replicates']].sort_values("abs_coef", ascending=False)
print(dn[["gene","coef","LGIN","HGIN","Cancer","rho_valid"]].head(15).round(3).to_string(index=False))

# gene list for enrichment
glist = sig["gene"].dropna().astype(str).unique().tolist()
open("results/taskB_gene_list.txt","w").write("\n".join(glist))
print(f"\nwrote results/taskB_gene_list.txt ({len(glist)} unique genes)")
