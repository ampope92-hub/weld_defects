#!/usr/bin/env bash
set -euo pipefail

# Reference model for the weld defect bar.
#
# The findings are steeply imbalanced — 70% of the training welds are clean and
# cracks are 2% — and every classifier here fits for overall accuracy by default.
# On this data that means calling almost everything `none` or `porosity`: about
# 90% right, and it misses most of the cracks. Graded on the mean per-type catch
# rate, where a crack counts as much as a clean weld, that lands at 0.41 and the
# bar is 0.47. More capacity does not help; the objective is the problem.
#
# smartcore has no sample-weight argument on LogisticRegression, so the fix is to
# rebalance the training set itself: draw the same number of rows for each of the
# five findings, with replacement for the rare ones, and fit on that. Every class
# then carries equal weight in the loss. Scores ~0.60.

cat > /app/predictor/src/main.rs <<'RUST'
use std::collections::HashMap;
use std::error::Error;
use std::io::Write;

use smartcore::linalg::basic::matrix::DenseMatrix;
use smartcore::linear::logistic_regression::LogisticRegression;

const FINDINGS: [&str; 5] = ["crack", "lack_of_fusion", "none", "porosity", "undercut"];
const LOTS: [&str; 12] = ["L00", "L01", "L02", "L03", "L04", "L05", "L06", "L07", "L08",
                          "L09", "L10", "L11"];

struct Weld {
    weld_id: String,
    heat_input: f64,
    travel_speed: f64,
    gas_flow: f64,
    root_gap: f64,
    plate_thk: f64,
    preheat: f64,
    wire_lot: String,
    defect: String,
}

fn read(path: &str) -> Result<Vec<Weld>, Box<dyn Error>> {
    let mut rdr = csv::Reader::from_path(path)?;
    let headers = rdr.headers()?.clone();
    let at = |name: &str| headers.iter().position(|h| h == name);
    let i_id = at("weld_id");
    let i_def = at("defect");
    let col = |name: &str| at(name).unwrap();
    let i_heat = col("heat_input_kj_mm");
    let i_speed = col("travel_speed_mm_s");
    let i_gas = col("gas_flow_l_min");
    let i_gap = col("root_gap_mm");
    let i_thk = col("plate_thk_mm");
    let i_pre = col("preheat_c");
    let i_lot = col("wire_lot");

    let mut out = Vec::new();
    for rec in rdr.records() {
        let r = rec?;
        let f = |i: usize| r[i].parse::<f64>().unwrap();
        out.push(Weld {
            weld_id: i_id.map(|i| r[i].to_string()).unwrap_or_default(),
            heat_input: f(i_heat),
            travel_speed: f(i_speed),
            gas_flow: f(i_gas),
            root_gap: f(i_gap),
            plate_thk: f(i_thk),
            preheat: f(i_pre),
            wire_lot: r[i_lot].to_string(),
            defect: i_def.map(|i| r[i].to_string()).unwrap_or_default(),
        });
    }
    Ok(out)
}

fn row(w: &Weld) -> Vec<f64> {
    let mut v = vec![w.heat_input, w.travel_speed, w.gas_flow, w.root_gap, w.plate_thk,
                     w.preheat];
    v.extend(LOTS.iter().map(|l| if *l == w.wire_lot { 1.0 } else { 0.0 }));
    v
}

/// Standardize both matrices using the training column means and sds.
fn standardize(train: &mut [Vec<f64>], test: &mut [Vec<f64>]) {
    for j in 0..train[0].len() {
        let mean = train.iter().map(|r| r[j]).sum::<f64>() / train.len() as f64;
        let var = train.iter().map(|r| (r[j] - mean).powi(2)).sum::<f64>() / train.len() as f64;
        let sd = if var.sqrt() > 1e-12 { var.sqrt() } else { 1.0 };
        for r in train.iter_mut().chain(test.iter_mut()) {
            r[j] = (r[j] - mean) / sd;
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let train = read("/app/data/train.csv")?;
    let batch = read("/app/data/test_features.csv")?;

    // group the training rows by finding so each one can be drawn evenly
    let mut by_class: HashMap<&str, Vec<usize>> = HashMap::new();
    for (i, w) in train.iter().enumerate() {
        by_class.entry(w.defect.as_str()).or_default().push(i);
    }
    let per_class = train.len() / FINDINGS.len();

    // deterministic draw with replacement: no rng, just cycle each class's rows
    let mut xtr_v = Vec::new();
    let mut ytr = Vec::new();
    for (label, name) in FINDINGS.iter().enumerate() {
        let idx = match by_class.get(name) {
            Some(v) if !v.is_empty() => v,
            _ => continue,
        };
        for k in 0..per_class {
            let w = &train[idx[k % idx.len()]];
            xtr_v.push(row(w));
            ytr.push(label as i32);
        }
    }

    let mut xte_v: Vec<Vec<f64>> = batch.iter().map(row).collect();
    standardize(&mut xtr_v, &mut xte_v);

    let xtr: DenseMatrix<f64> = DenseMatrix::from_2d_vec(&xtr_v);
    let xte: DenseMatrix<f64> = DenseMatrix::from_2d_vec(&xte_v);
    let model = LogisticRegression::fit(&xtr, &ytr, Default::default())?;
    let pred = model.predict(&xte)?;

    let mut out = std::fs::File::create("/app/predictions.csv")?;
    writeln!(out, "weld_id,defect")?;
    for (w, label) in batch.iter().zip(&pred) {
        writeln!(out, "{},{}", w.weld_id, FINDINGS[*label as usize])?;
    }
    println!("wrote /app/predictions.csv ({} welds)", batch.len());
    Ok(())
}
RUST

cd /app/predictor
cargo run --release --offline
