# Gene panels for staging gastric precancerous lesions vs. random genes

Code for: A. Misra, "Gene Panels for Staging Gastric Precancerous Lesions Rarely Beat Random
Genes or a Single Immune Score."

## Data

All data are public on GEO. Place the files below in `data/` (they are not tracked):

| File | Source |
|---|---|
| `GSE130823_series_matrix.txt.gz` | GEO series matrix |
| `GSE55696_series_matrix.txt.gz` | GEO series matrix |
| `GSE87666_series_matrix.txt.gz` | GEO series matrix |
| `GPL17077.txt` | GEO platform annotation |
| `ext/GSE78523_series_matrix.txt.gz` | GEO series matrix (independent cohort) |
| `ext/GPL18990-3879.txt` | GEO platform annotation (used to build `ext/GPL18990_symbols.csv`) |

`geo_search_table.csv` lists every series returned by the GEO search (run 22 Nov 2026) and why
it was included or excluded.

`data/sig/` holds the ESTIMATE immune/stromal gene sets and MCP-counter marker genes.

## Pipeline

Run from the repository root with `python src/<script>.py`.

1. `00`-`02`: parse GEO files and preprocess (shared probes, median centring, sex-probe removal).
2. `22_duplicate_threshold.py`, `23_cross_duplicates.py`: within-cohort correlation reference
   and cross-cohort duplicate detection (mutual best match, r >= 0.75, margin >= 0.15).
   Output: `results/duplicate_pairs.csv`, `results/duplicate_samples_removed.csv`.
3. `v5data.py`, `common.py`: deduplicated data, two platform groups
   (GPL17077 = GSE130823 + GSE87666; GPL6480 = GSE55696). Set `DEDUP=0` to disable deduplication.
4. Analyses (leave-one-platform-out, pooled predictions):
   - `v5_core.py`: ordinal vs multinomial / RF / XGBoost, bootstrap CIs, label permutation
   - `v5_nulls.py`: published panels and panel-size sweep vs three random-gene nulls
   - `v5_immune.py`: ESTIMATE and MCP-counter baselines, Jonckheere-Terpstra trends
   - `v5_stability.py`, `v5_combat.py`, `v5_clinical.py`: robustness checks
   - `v5_external.py`, `v5_external_prog.py`: independent cohort GSE78523
5. `v5_figs.py`: figures.

Scripts `03`-`21` are from the earlier, non-deduplicated three-cohort analysis and are kept
for reference.

## Requirements

`pip install -r requirements.txt` (Python 3.11+).
