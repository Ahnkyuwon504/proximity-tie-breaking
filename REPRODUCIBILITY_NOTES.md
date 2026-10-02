# Reproducibility notes

## Verifier-consistent variant: reported SD on Assembled-500

For the verifier-consistent variant on Assembled-500, the manuscript reports
an SD of 0.46 percentage points. Recomputing from the full-precision per-seed
results gives 0.4545368, which rounds to 0.45. The reported 0.46 is reproduced
when per-seed percentages are rounded to two decimals before computing the SD.
This reporting discrepancy does not alter the interpretation of this
sensitivity check or the primary analysis.

## IFEval per-run aggregates

`results/per_run_ifeval.csv` exports per-run IFEval aggregates (macro SSR and
HSR per model, variant, and seed; `seed = -1` denotes the untrained
checkpoint) from the original evaluation records. Rows with
`in_table_iv = yes` are the records that reproduce the IFEval column of
Table IV: for trained runs, the per-seed values whose mean and sample SD
match the table; for untrained checkpoints, the entries of the original
evaluation ledger. The evaluation archive also contains later
re-evaluations of some runs under the same configuration (vLLM batched
inference is not bit-deterministic across runs; the manuscript documents
re-run variation of up to about one point); these are retained with
`in_table_iv = no` together with their source paths.

## Figures

`code/figures.py` is a plotting helper kept from an earlier notebook revision
for provenance; it is not the generation path of the current Fig. 4, which is
computed from the per-cell covered-minus-uncovered differentials derived from
`results/per_class_rates.csv` (the same input as `stats/run_stats.py`).
