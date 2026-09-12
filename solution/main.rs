use std::error::Error;
use std::io::Write;

use smartcore::linalg::basic::arrays::Array;
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

/// The contracts table: costs[actual][called], both indexed as FINDINGS.
fn read_costs(path: &str) -> Result<Vec<Vec<f64>>, Box<dyn Error>> {
    let text = std::fs::read_to_string(path)?;
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header: Vec<&str> = lines.next().ok_or("empty cost table")?.split(',').collect();
    let mut costs = vec![vec![0.0; FINDINGS.len()]; FINDINGS.len()];
    for line in lines {
        let cells: Vec<&str> = line.split(',').collect();
        let actual = FINDINGS.iter().position(|f| *f == cells[0]).ok_or("unknown finding")?;
        for (col, name) in header.iter().enumerate().skip(1) {
            let called = FINDINGS.iter().position(|f| f == name).ok_or("unknown call")?;
            costs[actual][called] = cells[col].trim().parse()?;
        }
    }
    Ok(costs)
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

    // No rebalancing. The decision below needs posteriors that carry the population's own
    // class priors: resampling the training set shifts those priors, and applying the cost
    // table to shifted posteriors is worse than ignoring costs altogether. (Rebalancing and
    // then dividing the priors back out also works; not rebalancing is simply the shorter road.)
    // The classes the model knows are the ones the training file holds. A batch whose training
    // file never records some finding cannot be called that finding, and the fit is over the
    // findings present, mapped to a compact index, so a missing class does not leave a hole.
    let mut present: Vec<usize> = Vec::new();
    for w in train.iter() {
        if let Some(c) = FINDINGS.iter().position(|f| *f == w.defect) {
            if !present.contains(&c) {
                present.push(c);
            }
        }
    }
    present.sort_unstable();
    let mut xtr_v = Vec::new();
    let mut ytr = Vec::new();
    for w in train.iter() {
        if let Some(c) = FINDINGS.iter().position(|f| *f == w.defect) {
            xtr_v.push(row(w));
            ytr.push(present.iter().position(|p| *p == c).unwrap() as i32);
        }
    }

    let mut xte_v: Vec<Vec<f64>> = batch.iter().map(row).collect();
    standardize(&mut xtr_v, &mut xte_v);

    let xtr: DenseMatrix<f64> = DenseMatrix::from_2d_vec(&xtr_v);
    let xte: DenseMatrix<f64> = DenseMatrix::from_2d_vec(&xte_v);
    let model = LogisticRegression::fit(&xtr, &ytr, Default::default())?;

    // smartcore's LogisticRegression exposes no predict_proba, and predict() takes the argmax
    // of the class scores — which is the wrong decision here. The scores are recoverable from
    // the fitted parameters: for class c, score = x . coefficients[c] + intercept[c], laid out
    // exactly as predict() consumes them. Softmax turns those into posteriors.
    let coef = model.coefficients();
    let icept = model.intercept();
    let k = present.len();
    let nfeat = xte_v[0].len();

    let costs = read_costs("/app/data/consequence_costs.csv")?;

    let mut out = std::fs::File::create("/app/predictions.csv")?;
    writeln!(out, "weld_id,defect")?;
    for (i, w) in batch.iter().enumerate() {
        // With exactly two findings present smartcore fits one logit (class 1 against class 0)
        // and stores a single coefficient row; with three or more it stores one row per class.
        let p: Vec<f64> = if k == 2 {
            let mut acc = *icept.get((0, 0));
            for j in 0..nfeat {
                acc += xte_v[i][j] * *coef.get((0, j));
            }
            let p1 = 1.0 / (1.0 + (-acc).exp());
            vec![1.0 - p1, p1]
        } else {
            let mut score = vec![0.0f64; k];
            for c in 0..k {
                let mut acc = *icept.get((c, 0));
                for j in 0..nfeat {
                    acc += xte_v[i][j] * *coef.get((c, j));
                }
                score[c] = acc;
            }
            let hi = score.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let ex: Vec<f64> = score.iter().map(|s| (s - hi).exp()).collect();
            let z: f64 = ex.iter().sum();
            ex.iter().map(|e| e / z).collect()
        };

        // Call the finding with the lowest EXPECTED cost, not the most likely one. The posterior
        // is over the findings present in the training file; the call can be any finding.
        let mut best = 0usize;
        let mut best_cost = f64::INFINITY;
        for call in 0..FINDINGS.len() {
            let exp_cost: f64 = (0..k).map(|a| p[a] * costs[present[a]][call]).sum();
            if exp_cost < best_cost {
                best_cost = exp_cost;
                best = call;
            }
        }
        writeln!(out, "{},{}", w.weld_id, FINDINGS[best])?;
    }
    println!("wrote /app/predictions.csv ({} welds)", batch.len());
    Ok(())
}
