"""
Step 1: Parse GEO series matrix files, derive Correa-cascade stage labels
and (critically) patient IDs, then write tidy matrices + metadata.

Expected inputs in ./data/ (gzipped or plain):
  GSE130823_series_matrix.txt(.gz)   training cohort  (Agilent GPL17077)
  GSE55696_series_matrix.txt(.gz)    validation cohort(Agilent GPL6480)
Optional:
  GSE191275 supplementary counts     cross-technology test (RNA-seq)
"""
import gzip, io, os, re, sys
import pandas as pd

DATA = os.path.join(os.path.dirname(__file__), "..", "data")

# Ordered cascade. Index == ordinal stage value.
STAGES = ["Gastritis", "LGIN", "HGIN", "Cancer"]

# Map the raw strings GEO uses onto our four ordered stages.
LABEL_PATTERNS = [
    # order matters: specific -> general. Matched against lowercased, space-stripped titles.
    (r"lowgradeintraepithelialneoplasia|\blgin\b|lgin", "LGIN"),
    (r"highgradeintraepithelialneoplasia|\bhgin\b|hgin", "HGIN"),
    (r"intestinalgastriccancer|gastriccancer|adenocarcinoma|carcinoma|\begc\b|egc", "Cancer"),
    (r"gastritis", "Gastritis"),          # fallback
]

def _open(path):
    return gzip.open(path, "rt", encoding="latin-1") if path.endswith(".gz") \
        else open(path, "r", encoding="latin-1")

def label_from_title(title):
    t = title.lower().replace(" ", "")
    for pat, lab in LABEL_PATTERNS:
        if re.search(pat, t, flags=re.I):
            return lab
    return None

def patient_from_title(title, gse):
    """
    GSE130823 titles look like 'B0294-1: Gastritis' / 'B0294-2: ...LowGrade...'
    -> patient 'B0294'. The two samples are PAIRED biopsies from one person,
    so they must never be split across train/test.
    """
    if gse == "GSE130823":
        m = re.match(r"\s*([A-Za-z]+\d+)\s*-\s*\d+", title)
        if m:
            return m.group(1)
    return None  # GSE55696 titles carry no patient linkage

def parse_series_matrix(path, gse):
    """Return (expr DataFrame: probes x samples, meta DataFrame)."""
    titles, geo_acc = [], []
    table_lines = []
    in_table = False
    with _open(path) as fh:
        for line in fh:
            if line.startswith("!Sample_title"):
                titles = [x.strip().strip('"') for x in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!Sample_geo_accession"):
                geo_acc = [x.strip().strip('"') for x in line.rstrip("\n").split("\t")[1:]]
            elif line.startswith("!series_matrix_table_begin"):
                in_table = True
            elif line.startswith("!series_matrix_table_end"):
                in_table = False
            elif in_table:
                table_lines.append(line)

    if not table_lines:
        raise RuntimeError(f"{path}: no expression table found")

    expr = pd.read_csv(io.StringIO("".join(table_lines)), sep="\t",
                       index_col=0, na_values=["null", "NA", ""])
    expr.columns = [c.strip().strip('"') for c in expr.columns]
    expr.index.name = "probe"

    meta = pd.DataFrame({"gsm": geo_acc, "title": titles})
    meta["stage"] = meta["title"].map(label_from_title)
    meta["patient"] = meta["title"].map(lambda t: patient_from_title(t, gse))
    meta["cohort"] = gse
    meta["stage_ord"] = meta["stage"].map({s: i for i, s in enumerate(STAGES)})
    meta = meta.set_index("gsm")

    # align
    common = [c for c in expr.columns if c in meta.index]
    expr = expr[common]
    meta = meta.loc[common]
    return expr, meta

def main():
    found = {}
    for gse in ["GSE130823", "GSE55696"]:
        hit = None
        for fn in sorted(os.listdir(DATA)) if os.path.isdir(DATA) else []:
            if gse in fn and "series_matrix" in fn:
                hit = os.path.join(DATA, fn); break
        if hit is None:
            print(f"[MISSING] {gse}_series_matrix.txt(.gz) not in {DATA}")
            continue
        expr, meta = parse_series_matrix(hit, gse)
        found[gse] = (expr, meta)

        print(f"\n=== {gse} ===")
        print(f"file            : {os.path.basename(hit)}")
        print(f"probes x samples: {expr.shape[0]} x {expr.shape[1]}")
        print(f"missing values  : {int(expr.isna().sum().sum())}")
        print("stage counts    :")
        print(meta['stage'].value_counts().reindex(STAGES).fillna(0).astype(int).to_string())
        if meta['patient'].notna().any():
            n_pat = meta['patient'].nunique()
            print(f"patients        : {n_pat}  (samples per patient: "
                  f"{meta['patient'].value_counts().unique().tolist()})")
            print("  -> PAIRED design: split by patient, never by sample.")
        unlabelled = meta['stage'].isna().sum()
        if unlabelled:
            print(f"[WARN] {unlabelled} samples could not be labelled:")
            print(meta[meta['stage'].isna()]['title'].head(10).to_string())

        expr.to_pickle(f"results/{gse}_expr.pkl")
        meta.to_csv(f"results/{gse}_meta.csv")

    if not found:
        print("\nNothing parsed. Place the series matrix files in ./data/ and re-run.")
        sys.exit(1)
    print("\nWrote tidy matrices to results/.")

if __name__ == "__main__":
    main()
