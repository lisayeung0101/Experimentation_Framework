import argparse
import os
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd

""" 
-----------------------------
Configurable experiment scenarios
-----------------------------
Each scenario defines the "true" effect for treatment vs control.
- type: "proportion" (conversion only), "revenue" (continuous), or "both"
- baseline_*: control-group baselines
- lift_*: absolute or relative lifts for treatment
- cuped_r2: how predictive pre_metric is of the outcome (higher -> stronger CUPED benefit)
- srm_jitter: small randomization jitter to simulate mild assignment imbalance
"""

SCENARIOS = [
dict(
experiment_id="exp_paywall_A",
start="2025-03-01T09:00:00Z",
type="both",
baseline_conv=0.05, # 5% conversion in control
abs_lift_conv=0.008, # +0.8pp lift in treatment (to 5.8%)
baseline_revenue_mean=18.0, # average revenue for converters (post-paywall)
revenue_sd=6.0,
rev_uplift_rel=0.08, # +8% revenue uplift in treatment
cuped_r2=0.25,
srm_jitter=0.01,
desc="Paywall copy and plan messaging test targeting conversion and ARPU.",
),
dict(
experiment_id="exp_onboarding_B",
start="2025-04-05T10:00:00Z",
type="proportion",
baseline_conv=0.12,
abs_lift_conv=0.015, # +1.5pp activation lift
baseline_revenue_mean=0.0, # no revenue outcome used; included for schema
revenue_sd=0.0,
rev_uplift_rel=0.0,
cuped_r2=0.15,
srm_jitter=0.0,
desc="Onboarding coachmarks and guided setup aiming to increase activation.",
),
dict(
experiment_id="exp_pricing_C",
start="2025-05-01T11:00:00Z",
type="both",
baseline_conv=0.07,
abs_lift_conv=0.000, # intentionally 0 lift: negative/control-like example
baseline_revenue_mean=22.0,
revenue_sd=8.0,
rev_uplift_rel=0.00, # no revenue effect either
cuped_r2=0.30,
srm_jitter=0.02,
desc="Pricing page layout variants; expected neutral impact (control-like) for realism.",
),
]

def parse_args():
    ap = argparse.ArgumentParser(description="Generate synthetic experiment seeds.")
    ap.add_argument("--rows-per-experiment", type=int, default=10000, help="Users per experiment")
    ap.add_argument("--experiments", type=int, default=len(SCENARIOS), help="How many scenarios to generate (prefix of SCENARIOS)")
    ap.add_argument("--seed", type=int, default=42, help="Random seed")
    ap.add_argument("--outdir", type=str, default="dbt_project/seeds", help="Output directory for CSVs")
    return ap.parse_args()

def iso_ts(base: datetime, i: int, jitter_seconds: int = 300):
    # Base plus i seconds + small jitter, ISO8601 Zulu
    jitter = np.random.randint(0, jitter_seconds)
    ts = base + timedelta(seconds=i + jitter)
    return ts.replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z")

def generate_assignments_and_outcomes(n: int, scenario: dict, uid_prefix: str):
    """
    Returns:
    assignments_df: user_id, experiment_id, variant, assigned_at
    outcomes_df: user_id, experiment_id, conversion, revenue, pre_metric, event_ts
    """
    exp_id = scenario["experiment_id"]
    start_ts = datetime.fromisoformat(scenario["start"].replace("Z", "+00:00"))
    srm_jitter = scenario.get("srm_jitter", 0.0)
    # Slight SRM jitter: treatment share deviates from 0.5 by uniform(-srm_jitter, +srm_jitter)
    treat_share = 0.5 + np.random.uniform(-srm_jitter, srm_jitter)
    treat_share = float(np.clip(treat_share, 0.45, 0.55))

    # Assign variants
    variants = np.where(np.random.rand(n) < treat_share, "treatment", "control")

    # User ids
    user_ids = [f"{uid_prefix}_{i+1:06d}" for i in range(n)]

    # Assignment timestamps
    assigned_at = [iso_ts(start_ts, i) for i in range(n)]

    assignments_df = pd.DataFrame({
        "user_id": user_ids,
        "experiment_id": exp_id,
        "variant": variants,
        "assigned_at": assigned_at,
    })

    # Generate pre_metric (CUPED covariate): Poisson/Gamma mix mapped to approx normal-ish scale
    # Higher pre_metric correlates with higher outcome, controlled by cuped_r2
    cuped_r2 = scenario.get("cuped_r2", 0.2)
    # Generate latent engagement, then add noise to reach desired R^2 with outcome drivers
    latent_engagement = np.random.gamma(shape=2.0, scale=5.0, size=n)  # mean ~10
    pre_metric = latent_engagement + np.random.normal(0, (1 - cuped_r2) * 3.0, size=n)
    pre_metric = np.maximum(pre_metric, 0.0).round(2)

    # Conversion generation
    baseline_conv = scenario["baseline_conv"]
    abs_lift_conv = scenario["abs_lift_conv"]
    # Map pre_metric to a small logit shift to correlate with conversion
    conv_logit = np.log(baseline_conv / (1 - baseline_conv)) + 0.05 * (pre_metric - pre_metric.mean())
    p_control = 1 / (1 + np.exp(-conv_logit))
    p_treat = np.clip(p_control + abs_lift_conv, 0.0001, 0.9999)

    is_treat = (variants == "treatment")
    if scenario["type"] in ("proportion", "both"):
        conv_prob = np.where(is_treat, p_treat, p_control)
        conversion = (np.random.rand(n) < conv_prob).astype(int)
    else:
        conversion = np.zeros(n, dtype=int)

    # Revenue generation (only meaningful when conversion==1)
    base_mean = scenario["baseline_revenue_mean"]
    rev_sd = scenario["revenue_sd"]
    rev_uplift_rel = scenario["rev_uplift_rel"]

    if scenario["type"] in ("both", "revenue"):
        mean_control = base_mean * (1 + 0.01 * (pre_metric - pre_metric.mean()))
        mean_treat = mean_control * (1 + rev_uplift_rel)
        means = np.where(is_treat, mean_treat, mean_control)

        # Draw revenue only for converters; otherwise 0
        revenue = np.where(conversion == 1, np.maximum(0.0, np.random.normal(means, rev_sd)), 0.0)
        revenue = np.round(revenue, 2)
    else:
        revenue = np.zeros(n).round(2)

    # Event timestamps (outcomes) lag assignments by minutes to days
    event_ts = [iso_ts(start_ts + timedelta(hours=1), i, jitter_seconds=60*60) for i in range(n)]

    outcomes_df = pd.DataFrame({
        "user_id": user_ids,
        "experiment_id": exp_id,
        "conversion": conversion.astype(int),
        "revenue": revenue.astype(float),
        "pre_metric": pre_metric.astype(float),
        "event_ts": event_ts,
    })

    return assignments_df, outcomes_df

def main():
    args = parse_args()
    np.random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    assignments_all = []
    outcomes_all = []

    for idx, scenario in enumerate(SCENARIOS[: args.experiments]):
        uid_prefix = f"u{idx+1:02d}"
        a_df, o_df = generate_assignments_and_outcomes(
            n=args.rows_per_experiment,
            scenario=scenario,
            uid_prefix=uid_prefix
        )
        assignments_all.append(a_df)
        outcomes_all.append(o_df)

    assignments = pd.concat(assignments_all, ignore_index=True)
    outcomes = pd.concat(outcomes_all, ignore_index=True)

    # Shuffle rows a bit for realism
    assignments = assignments.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    outcomes = outcomes.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    # Write CSVs
    assign_path = os.path.join(args.outdir, "assignments_seed.csv")
    out_path = os.path.join(args.outdir, "outcomes_seed.csv")
    assignments.to_csv(assign_path, index=False)
    outcomes.to_csv(out_path, index=False)

    print(f"Wrote {assign_path} ({len(assignments):,} rows)")
    print(f"Wrote {out_path} ({len(outcomes):,} rows)")


if __name__ == "__main__":
    main()