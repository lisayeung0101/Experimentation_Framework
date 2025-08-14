import os
import math
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# ---------------------
# Tunable parameters
# ---------------------
N_USERS = 30000  # total users
START_DATE = datetime.today().date() - timedelta(days=27)  # ~28-day window
DAYS = 28

# Baseline (Control) rates
P_TRIAL_C = 0.12           # 12% start trial
P_T2P_C = 0.20             # 20% of trial starters become paid
P_REFUND_FIRST_CYCLE_C = 0.015  # 1.5% refund in first cycle
P_EARLY_CHURN_30D_C = 0.09      # 9% cancel within 30 days of paid

# Treatment lifts
REL_LIFT_TRIAL = 0.15      # +15% relative on trial_start
REL_LIFT_T2P   = 0.08      # +8% relative on trial_to_paid
DELTA_REFUND   = 0.0000    # guardrail difference (keep flat by default)
DELTA_CHURN    = 0.0000    # guardrail difference (keep flat by default)

# Pre-engagement (CUPED covariate)
# Control mean λ=1.6, Treatment λ=1.7 (slightly higher intent with improved onboarding)
LAMBDA_PRE_C = 1.6
LAMBDA_PRE_T = 1.7
WINSOR_PCTL = 0.99  # winsorize extreme pre-engagement values

# Time-to-subscribe distributions (days) for those who pay
# Treatment shorter decision time (7-day trial vs 14-day)
TT_SUB_CONTROL_MEAN = 10.0
TT_SUB_TREAT_MEAN   = 7.0
TT_SUB_SHAPE = 2.0  # gamma shape

# Randomization options
PLATFORMS = ["web", "ios", "android"]
CHANNELS = ["paid", "organic", "referral", "partner"]

rng = np.random.default_rng(42)
random.seed(42)

def gamma_days(mean_days, shape, size):
    scale = mean_days / shape
    return rng.gamma(shape, scale, size=size)

def bounded_date(i):
    # Spread users uniformly over DAYS for assigned_at/event_date
    day_offset = rng.integers(0, DAYS)
    return START_DATE + timedelta(days=int(day_offset))

def main():
    os.makedirs("dbt_project/seeds", exist_ok=True)

    # Assign users and variants (50/50)
    user_ids = np.arange(1, N_USERS + 1, dtype=int)
    variants = rng.choice(["control", "treatment"], size=N_USERS, replace=True)
    platforms = rng.choice(PLATFORMS, size=N_USERS, replace=True)
    channels = rng.choice(CHANNELS, size=N_USERS, replace=True)
    assigned_dates = [bounded_date(i) for i in range(N_USERS)]
    assigned_times = [datetime.combine(d, datetime.min.time()) + timedelta(
        seconds=int(rng.integers(0, 24*3600))) for d in assigned_dates]

    # Pre-engagement (sessions in prior 30d)
    pre_engagement = np.zeros(N_USERS, dtype=float)
    mask_t = variants == "treatment"
    mask_c = ~mask_t
    pre_engagement[mask_c] = rng.poisson(LAMBDA_PRE_C, size=mask_c.sum())
    pre_engagement[mask_t] = rng.poisson(LAMBDA_PRE_T, size=mask_t.sum())

    # Winsorize pre_engagement
    cutoff = np.quantile(pre_engagement, WINSOR_PCTL)
    pre_engagement = np.clip(pre_engagement, 0, cutoff)

    # Trial start probabilities
    p_trial = np.where(mask_t, P_TRIAL_C * (1 + REL_LIFT_TRIAL), P_TRIAL_C)
    trial_start = rng.binomial(1, p_trial)

    # Conditional trial-to-paid probabilities
    p_t2p = np.where(mask_t, P_T2P_C * (1 + REL_LIFT_T2P), P_T2P_C)

    # Paid only among trial starters
    paid_subscriber = np.zeros(N_USERS, dtype=int)
    idx_trial = np.where(trial_start == 1)[0]
    paid_subscriber[idx_trial] = rng.binomial(1, p_t2p[idx_trial])

    # Refund and early churn among payers
    p_refund = np.where(mask_t, P_REFUND_FIRST_CYCLE_C + DELTA_REFUND, P_REFUND_FIRST_CYCLE_C)
    p_churn = np.where(mask_t, P_EARLY_CHURN_30D_C + DELTA_CHURN, P_EARLY_CHURN_30D_C)

    refund_in_first_cycle = np.zeros(N_USERS, dtype=int)
    early_churn_30d = np.zeros(N_USERS, dtype=int)

    idx_paid = np.where(paid_subscriber == 1)[0]
    refund_in_first_cycle[idx_paid] = rng.binomial(1, np.clip(p_refund[idx_paid], 0, 1))
    early_churn_30d[idx_paid] = rng.binomial(1, np.clip(p_churn[idx_paid], 0, 1))

    # trial_start_at and paid_at timestamps
    trial_start_at = [None] * N_USERS
    paid_at = [None] * N_USERS
    time_to_subscribe_days = np.full(N_USERS, np.nan, dtype=float)

    # Time to trial start: beta skew in first few days after assignment
    tt_trial_days = rng.gamma(2.0, 2.0, size=idx_trial.size)  # mean ~4 days
    for j, i in enumerate(idx_trial):
        tdays = float(tt_trial_days[j])
        trial_dt = assigned_times[i] + timedelta(days=tdays)
        trial_start_at[i] = trial_dt

    # Time to subscribe distribution (for paid users)
    # Control longer mean, Treatment shorter mean
    tt_paid_days = np.zeros(idx_paid.size)
    for j, i in enumerate(idx_paid):
        if variants[i] == "treatment":
            tt = gamma_days(TT_SUB_TREAT_MEAN, TT_SUB_SHAPE, 1)[0]
        else:
            tt = gamma_days(TT_SUB_CONTROL_MEAN, TT_SUB_SHAPE, 1)[0]
        # Ensure paid_at after trial_start_at if trial exists, else after assignment
        anchor = trial_start_at[i] if trial_start_at[i] is not None else assigned_times[i]
        paid_dt = anchor + timedelta(days=float(tt))
        paid_at[i] = paid_dt
        time_to_subscribe_days[i] = (paid_dt - assigned_times[i]).days + ((paid_dt - assigned_times[i]).seconds / 86400.0)

    # event_date anchor (use assignment date for daily grouping)
    event_dates = [dt.date() for dt in assigned_times]

    # Build DataFrames
    assignments = pd.DataFrame({
        "user_id": user_ids,
        "variant": variants,
        "assigned_at": assigned_times,
        "platform": platforms,
        "acquisition_channel": channels,
    })
    # Ensure ISO formatting for timestamps
    assignments["assigned_at"] = assignments["assigned_at"].map(lambda x: x.isoformat())

    outcomes = pd.DataFrame({
        "user_id": user_ids,
        "event_date": event_dates,  # date anchor for grouping/plots
        "trial_start": trial_start,
        "trial_start_at": [ts.isoformat() if isinstance(ts, datetime) else "" for ts in trial_start_at],
        "paid_subscriber": paid_subscriber,
        "paid_at": [ts.isoformat() if isinstance(ts, datetime) else "" for ts in paid_at],
        "refund_in_first_cycle": refund_in_first_cycle,
        "early_churn_30d": early_churn_30d,
        "time_to_subscribe_days": np.round(time_to_subscribe_days, 3),
        "pre_engagement_30d": pre_engagement.astype(int),
    })

    # Write CSVs
    a_path = "dbt_project/seeds/assignments_seed.csv"
    o_path = "dbt_project/seeds/outcomes_seed.csv"
    assignments.to_csv(a_path, index=False)
    outcomes.to_csv(o_path, index=False)

    print(f"Wrote: {a_path} ({len(assignments):,} rows)")
    print(f"Wrote: {o_path} ({len(outcomes):,} rows)")

if __name__ == "__main__":
    main()
