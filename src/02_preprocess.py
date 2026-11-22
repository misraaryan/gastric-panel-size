"""
Step 2: harmonise the two cohorts.
  - restrict to A_* gene probes shared by GPL17077 and GPL6480
  - per-sample median centring (GSE55696 is not centred; GSE130823 is)
  - identify and remove sex-associated probes (confounder: male fraction rises with stage)
  - drop near-constant probes
Outputs harmonised matrices + the sex-probe audit.
"""
import pandas as pd, numpy as np
from scipy import stats

STAGES = ["Gastritis", "LGIN", "HGIN", "Cancer"]

tr = pd.read_pickle("results/GSE130823_expr.pkl")
va = pd.read_pickle("results/GSE55696_expr.pkl")
mtr = pd.read_csv("results/GSE130823_meta.csv", index_col=0)
mva = pd.read_csv("results/GSE55696_meta.csv", index_col=0)

# clinical metadata (age/sex) from the SOFT file, for the training cohort
clin = pd.read_csv("results/GSE130823_clinical_meta.csv", index_col=0)
mtr = mtr.join(clin[["gender", "age"]], how="left")

# 1) shared real gene probes
shared = sorted((set(tr.index) & set(va.index)))
shared = [p for p in shared if str(p).startswith("A_")]
tr, va = tr.loc[shared], va.loc[shared]
print(f"shared A_* gene probes: {len(shared)}")

# 2) per-sample median centring (makes the two cohorts comparable)
tr = tr.sub(tr.median(axis=0), axis=1)
va = va.sub(va.median(axis=0), axis=1)
print("per-sample median centring applied to both cohorts")

# 3) sex-associated probe detection, TRAINING DATA ONLY
sex = mtr["gender"].str.lower()
m_idx = mtr.index[sex == "male"]; f_idx = mtr.index[sex == "female"]
print(f"\nsex labels available: {len(m_idx)} male, {len(f_idx)} female")
t, p = stats.ttest_ind(tr[m_idx].to_numpy(float), tr[f_idx].to_numpy(float),
                       axis=1, equal_var=False)
# Benjamini-Hochberg
order = np.argsort(p); ranked = p[order]
q = ranked * len(p) / (np.arange(len(p)) + 1)
q = np.minimum.accumulate(q[::-1])[::-1]
qv = np.empty_like(q); qv[order] = np.clip(q, 0, 1)

sex_probes = pd.DataFrame({"probe": shared, "t": t, "p": p, "q": qv})
# strict: strongly significant AND large effect — this is the XIST / Y-gene signature
md = tr[m_idx].mean(axis=1).to_numpy() - tr[f_idx].mean(axis=1).to_numpy()
sex_probes["mean_diff"] = md
flag = (sex_probes["q"] < 0.01) & (sex_probes["mean_diff"].abs() > 1.0)
sex_probes["sex_linked"] = flag
n_sex = int(flag.sum())
print(f"sex-associated probes flagged (q<0.01 & |Δ|>1.0): {n_sex}")
print("\ntop 12 by |mean difference| (expect XIST and Y-chromosome genes):")
print(sex_probes.loc[flag].reindex(
      sex_probes.loc[flag, 'mean_diff'].abs().sort_values(ascending=False).index
      )[["probe", "t", "q", "mean_diff"]].head(12).to_string(index=False))
sex_probes.to_csv("results/sex_associated_probes.csv", index=False)

keep = ~flag.to_numpy()
tr_ns, va_ns = tr.loc[keep], va.loc[keep]

# 4) drop near-constant probes (training variance)
v = tr_ns.var(axis=1)
nz = v > 1e-6
tr_ns, va_ns = tr_ns.loc[nz.to_numpy()], va_ns.loc[nz.to_numpy()]
print(f"\nafter removing sex probes and near-constant probes: {tr_ns.shape[0]} probes")

for nm, df in [("train_with_sex", tr), ("valid_with_sex", va),
               ("train", tr_ns), ("valid", va_ns)]:
    df.to_pickle(f"results/X_{nm}.pkl")
mtr.to_csv("results/meta_train.csv"); mva.to_csv("results/meta_valid.csv")

print("\n--- final shapes ---")
print(f"train (GSE130823): {tr_ns.shape[1]} samples x {tr_ns.shape[0]} probes, "
      f"{mtr['patient'].nunique()} patients")
print(f"valid (GSE55696) : {va_ns.shape[1]} samples x {va_ns.shape[0]} probes")
print("\ntrain stages:", mtr['stage'].value_counts().reindex(STAGES).to_dict())
print("valid stages:", mva['stage'].value_counts().reindex(STAGES).to_dict())
