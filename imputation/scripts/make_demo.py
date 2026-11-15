"""Generate a synthetic enterprise operating panel with missing data.

The data are simulated from the same bivariate Gaussian regression family that
the imputer fits, with a strong positive revenue/cost correlation.  We then
introduce the three real-world defects described in the task:

1. missing revenue/cost cells,
2. interval-only disclosures (rounded bands instead of exact values),
3. published exact profits and regional revenue totals, which are kept as
   separate observations and must never be overwritten.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(seed: int = 7, n: int = 24) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    industries = np.array(["制造", "零售", "科技", "物流"])
    regions = np.array(["华东", "华南", "华北"])

    industry = rng.choice(industries, size=n, p=[0.4, 0.3, 0.2, 0.1])
    region = rng.choice(regions, size=n)
    employees = np.clip(rng.lognormal(mean=5.5, sigma=0.7, size=n), 30, 9000)
    assets = np.clip(employees * rng.uniform(0.8, 2.5, size=n) * 10, 500, None)
    year = rng.choice([2023, 2024], size=n)

    # Design: intercept + log employees + industry effects (制造 baseline).
    x_emp = (np.log(employees) - np.log(employees).mean()) / np.log(employees).std()
    ind_tech = (industry == "科技").astype(float)
    ind_retail = (industry == "零售").astype(float)
    ind_logi = (industry == "物流").astype(float)
    X = np.column_stack([np.ones(n), x_emp, ind_tech, ind_retail, ind_logi])

    beta_r = np.array([500.0, 260.0, 180.0, -60.0, -40.0])
    beta_c = np.array([420.0, 210.0, 120.0, -55.0, -30.0])
    mu_r, mu_c = X @ beta_r, X @ beta_c
    sr, sc, rho = 90.0, 80.0, 0.85
    Sigma = np.array([[sr**2, rho * sr * sc], [rho * sr * sc, sc**2]])
    eps = rng.multivariate_normal([0, 0], Sigma, size=n)
    # Round the latent values to two decimals up front so that exact cells,
    # published profits and regional totals are all mutually consistent at the
    # disclosed precision (the imputer must reconcile data at that precision).
    revenue = np.round(np.maximum(mu_r + eps[:, 0], 20.0), 2)
    cost = np.round(np.maximum(mu_c + eps[:, 1], 15.0), 2)
    profit = np.round(revenue - cost, 2)

    r_cell = [f"{v:.2f}" for v in revenue]
    c_cell = [f"{v:.2f}" for v in cost]

    # --- introduce defects deterministically ---
    missing_r = rng.choice(n, size=4, replace=False)
    for i in missing_r:
        r_cell[i] = ""
    missing_c = rng.choice(np.setdiff1d(np.arange(n), missing_r), size=3, replace=False)
    for i in missing_c:
        c_cell[i] = ""
    interval_r = rng.choice(np.setdiff1d(np.arange(n), missing_r), size=4, replace=False)
    for i in interval_r:
        width = 0.15 * revenue[i]
        # Round outward so the disclosed band always contains the latent truth.
        lo = np.floor((revenue[i] - width) * 100) / 100
        hi = np.ceil((revenue[i] + width) * 100) / 100
        r_cell[i] = f"[{lo:.2f}, {hi:.2f}]"
    interval_c = rng.choice(
        np.setdiff1d(np.arange(n), np.concatenate([missing_c, interval_r])),
        size=3, replace=False)
    for i in interval_c:
        width = 0.12 * cost[i]
        lo = np.floor((cost[i] - width) * 100) / 100
        hi = np.ceil((cost[i] + width) * 100) / 100
        c_cell[i] = f"[{lo:.2f}, {hi:.2f}]"

    # Published profit: available for 8 rows, including some with bad cells.
    profit_rows = rng.choice(n, size=8, replace=False)
    profit_cell = [np.nan] * n
    for i in profit_rows:
        profit_cell[i] = round(profit[i], 2)

    df = pd.DataFrame({
        "id": [f"E{idx:03d}" for idx in range(n)],
        "year": year,
        "region": region,
        "industry": industry,
        "employees": employees.round(1),
        "assets": assets.round(1),
        "revenue": r_cell,
        "cost": c_cell,
        "profit": profit_cell,
        # Ground truth kept only for validation, never read by the imputer.
        "_true_revenue": revenue.round(4),
        "_true_cost": cost.round(4),
    })

    # Regional totals over the full panel (the aggregate that was published).
    totals = (
        df.assign(_tr=revenue, _tc=cost)
          .groupby("region", as_index=False)
          .agg(total_revenue=("_tr", "sum"), total_cost=("_tc", "sum"))
    )
    totals = totals.round(2)
    return df, totals


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="/workspace/data/demo")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n", type=int, default=24)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    panel, totals = generate(seed=args.seed, n=args.n)
    panel.to_csv(out / "panel.csv", index=False)
    totals.to_csv(out / "totals.csv", index=False)
    print(f"wrote {out / 'panel.csv'} ({len(panel)} rows)")
    print(f"wrote {out / 'totals.csv'} ({len(totals)} regions)")


if __name__ == "__main__":
    main()
