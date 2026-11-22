"""Extract per-sample clinical metadata from a GEO SOFT sample-metadata file."""
import re, sys, pandas as pd

STAGES = ["Gastritis", "LGIN", "HGIN", "Cancer"]
def stage_of(title):
    t = title.lower().replace(" ", "")
    if "lowgradeintraepithelialneoplasia" in t: return "LGIN"
    if "highgradeintraepithelialneoplasia" in t: return "HGIN"
    if "intestinalgastriccancer" in t or "cancer" in t or "carcinoma" in t: return "Cancer"
    if "gastritis" in t: return "Gastritis"
    return None

recs, cur = [], None
with open(sys.argv[1], encoding="utf-8", errors="replace") as fh:
    for line in fh:
        line = line.rstrip("\n")
        if line.startswith("^SAMPLE"):
            if cur: recs.append(cur)
            cur = {"gsm": line.split("=")[1].strip()}
        elif cur is None:
            continue
        elif line.startswith("!Sample_title"):
            cur["title"] = line.split("=", 1)[1].strip()
        elif line.startswith("!Sample_characteristics_ch1"):
            v = line.split("=", 1)[1].strip()
            if ":" in v:
                k, val = v.split(":", 1)
                cur[k.strip().lower()] = val.strip()
        elif line.startswith("!Sample_description"):
            cur["description"] = line.split("=", 1)[1].strip()
        elif line.startswith("!Sample_data_row_count"):
            cur["n_probes"] = int(line.split("=", 1)[1].strip())
        elif line.startswith("!Sample_platform_id"):
            cur["platform"] = line.split("=", 1)[1].strip()
if cur: recs.append(cur)

df = pd.DataFrame(recs)
df["stage"] = df["title"].map(stage_of)
df["patient"] = df["title"].str.extract(r"^\s*([A-Za-z]+\d+)\s*-\s*\d+")
df["biopsy_no"] = df["title"].str.extract(r"^\s*[A-Za-z]+\d+\s*-\s*(\d+)")
df["age"] = pd.to_numeric(df.get("age"), errors="coerce")
df["stage_ord"] = df["stage"].map({s: i for i, s in enumerate(STAGES)})
df = df.set_index("gsm")

print(f"samples parsed      : {len(df)}")
print(f"probes per sample   : {sorted(df['n_probes'].unique())}")
print(f"platform            : {df['platform'].unique().tolist()}")
print(f"unique patients     : {df['patient'].nunique()}")
print(f"samples per patient : {sorted(df['patient'].value_counts().unique().tolist())}")
print(f"explicit 'paired control' statements: "
      f"{df['description'].astype(str).str.contains('paired control', case=False).sum()}")

print("\n--- stage counts (all 94 samples) ---")
print(df['stage'].value_counts().reindex(STAGES).to_string())

# Lesion stage per patient (the -2 biopsy); -1 is always the paired gastritis control
lesion = df[df['biopsy_no'] == "2"]
print("\n--- lesion stage per patient (n=47) ---")
print(lesion['stage'].value_counts().reindex(STAGES).dropna().astype(int).to_string())

print("\n=== TABLE 1: cohort characteristics, GSE130823 ===")
t1 = df.groupby("stage", dropna=True).agg(
    n=("title", "size"),
    age_mean=("age", "mean"),
    age_sd=("age", "std"),
    age_min=("age", "min"),
    age_max=("age", "max"),
    female=("gender", lambda s: (s.str.lower() == "female").sum()),
    male=("gender", lambda s: (s.str.lower() == "male").sum()),
).reindex(STAGES)
t1["age_mean"] = t1["age_mean"].round(1); t1["age_sd"] = t1["age_sd"].round(1)
print(t1.to_string())

print("\n=== Patient-level demographics (one row per patient, from the -2 biopsy) ===")
t2 = lesion.groupby("stage").agg(
    patients=("patient", "nunique"),
    age_mean=("age", "mean"), age_sd=("age", "std"),
    female=("gender", lambda s: (s.str.lower() == "female").sum()),
    male=("gender", lambda s: (s.str.lower() == "male").sum()),
).reindex(["LGIN", "HGIN", "Cancer"])
t2["age_mean"] = t2["age_mean"].round(1); t2["age_sd"] = t2["age_sd"].round(1)
print(t2.to_string())

df.to_csv("results/GSE130823_clinical_meta.csv")
t1.to_csv("results/table1_sample_level.csv")
t2.to_csv("results/table1_patient_level.csv")
print("\nSaved: results/GSE130823_clinical_meta.csv, table1_sample_level.csv, table1_patient_level.csv")
