# Weld inspection export — 2026-06-12

One row per weld. `train.csv` is welds the ultrasonic inspector has already signed off,
with the finding recorded. `test_features.csv` is the held-back batch — same process
columns, findings not included.

## Columns

| column | source | description |
|---|---|---|
| `weld_id` | cell log | weld identifier (held-back batch only) |
| `heat_input_kj_mm` | cell log | arc heat input per mm of travel |
| `travel_speed_mm_s` | cell log | torch travel speed |
| `gas_flow_l_min` | cell log | shielding gas flow rate |
| `root_gap_mm` | fit-up check | measured root gap before the pass |
| `plate_thk_mm` | job card | plate thickness |
| `preheat_c` | job card | preheat temperature |
| `wire_lot` | consumables | filler wire lot identifier |
| `defect` | UT inspection | what the inspector found: `none`, `porosity`, `undercut`, `lack_of_fusion` or `crack` |

## Notes

Findings are the inspector's call from the ultrasonic report, one per weld — where a
weld showed more than one indication the report records the most severe. Process values
are the cell's own logged averages over the pass. Both cohorts come off the same cells
on the same job mix.
