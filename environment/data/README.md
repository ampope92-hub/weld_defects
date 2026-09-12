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
| `ut_scan_minutes` | UT work order | scan minutes booked against the weld |
| `repair_pass_count` | UT work order | repair passes booked against the weld |
| `consequence_costs.csv` | contracts | what each outcome costs us, in pounds per weld (see below) |
| `defect` | UT inspection | what the inspector found: `none`, `porosity`, `undercut`, `lack_of_fusion` or `crack` |

## Notes

Findings are the inspector's call from the ultrasonic report, one per weld — where a
weld showed more than one indication the report records the most severe. Process values
are the cell's own logged averages over the pass. Both cohorts come off the same cells
on the same job mix.

Both cohorts were pulled in the same export run off the inspection database. The signed-off
cohort was taken after its UT work orders had been reconciled against the inspection reports;
the held-back batch was still with the scheduler at that point and was taken as it stood.

## What a call costs us

`consequence_costs.csv` is the table our contracts desk keeps. One row per finding the weld
actually had, one column per finding we called it. The diagonal is zero — we got it right.

The numbers are not symmetric and they are not close. Shipping a weld that turns out to have a
crack costs 900: that is a field failure, a shutdown and a re-lay. Shipping one with porosity
costs 25, because porosity is usually found at the next hydrotest. Pulling a sound weld off the
line costs 40 whatever we thought was wrong with it — that is a scanner's morning and a hold on
the spool. Pulling a weld that really was defective but writing up the wrong finding costs a
quarter of the miss, because the repair crew arrives with the wrong procedure and comes back.

What we are judged on at the end of the job is the total this table adds up to across the batch,
divided by the number of welds.
