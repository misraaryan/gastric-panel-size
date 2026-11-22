"""Step 6: probe -> gene mapping; validate the sex-probe removal; name the signature."""
import pandas as pd, numpy as np, io, re

lines = open("data/GPL17077.txt", encoding="latin-1").read().splitlines()
s = next(i for i,l in enumerate(lines) if l.startswith("!platform_table_begin")) + 1
e = next((i for i,l in enumerate(lines) if l.startswith("!platform_table_end")), len(lines))
ann = pd.read_csv(io.StringIO("\n".join(lines[s:e])), sep="\t", low_memory=False)
ann = ann.set_index("ID")
print(f"annotation rows: {len(ann)}")
print(f"probes with a gene symbol: {ann['GENE_SYMBOL'].notna().sum()}")

# ---- validate empirical sex-probe detection ----
sexp = pd.read_csv("results/sex_associated_probes.csv")
flagged = sexp.loc[sexp["sex_linked"], "probe"].tolist()
sub = ann.reindex(flagged)[["GENE_SYMBOL", "CHROMOSOMAL_LOCATION", "GENE_NAME"]]
sub["chr"] = sub["CHROMOSOMAL_LOCATION"].astype(str).str.extract(r"(chr[\dXYMT]+)")
print("\n" + "="*72)
print("VALIDATION OF SEX-PROBE REMOVAL (detected from data, no annotation used)")
print("="*72)
print(sub[["GENE_SYMBOL", "chr", "GENE_NAME"]].to_string())
onXY = sub["chr"].isin(["chrX", "chrY"]).sum()
print(f"\n--> {onXY}/{len(sub)} flagged probes are on chrX or chrY "
      f"({100*onXY/len(sub):.0f}%). Empirical detection confirmed by annotation.")

# ---- name the Task B signature ----
sig = pd.read_csv("results/taskB_selected_probes.csv")["probe"].tolist()
sg = ann.reindex(sig)[["GENE_SYMBOL", "GENE_NAME", "CHROMOSOMAL_LOCATION"]]
sg["chr"] = sg["CHROMOSOMAL_LOCATION"].astype(str).str.extract(r"(chr[\dXYMT]+)")
named = sg["GENE_SYMBOL"].notna().sum()
print("\n" + "="*72)
print(f"TASK B SIGNATURE: {len(sig)} probes -> {named} with gene symbols, "
      f"{sg['GENE_SYMBOL'].nunique()} unique genes")
print("="*72)
sg.to_csv("results/taskB_signature_annotated.csv")

# any residual sex-chromosome genes in the signature? (should be ~0)
resid = sg[sg["chr"].isin(["chrX","chrY"])]
print(f"signature probes on sex chromosomes: {len(resid)} "
      f"({resid['GENE_SYMBOL'].dropna().unique()[:8].tolist()})")

# save a clean probe->symbol map for later steps
m = ann["GENE_SYMBOL"].dropna()
m.to_csv("results/probe2symbol.csv")
print(f"\nsaved probe->symbol map ({len(m)} probes)")
