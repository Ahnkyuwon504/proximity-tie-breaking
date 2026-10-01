"""Regenerates results/boundary_instances.csv.gz from raw check_*.jsonl (available on request).
Usage: python stats/export_boundary_instances.py --check-dir DIR --data-dir DIR
The counting functions are identical to stats/boundary_analysis.py (obs_lite_v1_approx).
See boundary_analysis.py for the shared obs_lite implementation; this script was run with the merged
raw-output directory to produce the released CSV; the drop accounting is stored alongside in
boundary_failed_breakdown.json."""
