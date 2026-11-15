"""Command-line entry point for the joint multiple-imputation pipeline.

Example
-------
python -I scripts/run_imputation.py \
    --panel data/demo/panel.csv --totals data/demo/totals.csv \
    --group-cols region --out-dir output/demo \
    --numeric employees assets --categorical industry region year \
    --chains 4 --draws 250
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from joint_impute import load_panel, load_totals, mle_fit  # noqa: E402
from joint_impute.impute import run_multiple_imputation  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--panel", required=True)
    ap.add_argument("--totals")
    ap.add_argument("--group-cols", nargs="+", default=["region"])
    ap.add_argument("--numeric", nargs="*", default=[])
    ap.add_argument("--categorical", nargs="*", default=[])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--chains", type=int, default=4)
    ap.add_argument("--draws", type=int, default=250)
    ap.add_argument("--burn-in", type=int, default=1000)
    ap.add_argument("--thin", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20261114)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    problem = load_panel(args.panel,
                         numeric_features=args.numeric or None,
                         categorical_features=args.categorical or None)
    if args.totals:
        problem = load_totals(problem, args.totals, args.group_cols)

    t0 = time.time()
    fit = mle_fit(problem)
    print(f"[1/2] joint MAP fit in {time.time() - t0:.1f}s "
          f"(log-posterior={fit['log_posterior']:.2f}, "
          f"rho={fit['Sigma'][0, 1] / np.sqrt(fit['Sigma'][0, 0] * fit['Sigma'][1, 1]):.3f})")

    t0 = time.time()
    result = run_multiple_imputation(
        problem, fit,
        n_chains=args.chains, n_draws=args.draws,
        burn_in=args.burn_in, thin=args.thin, seed=args.seed,
    )
    print(f"[2/2] MCMC in {time.time() - t0:.1f}s")

    result["long"].to_csv(out / "imputed_draws_long.csv", index=False)
    result["summary"].to_csv(out / "imputed_summary.csv", index=False)
    result["rubin_cells"].to_csv(out / "rubin_per_cell.csv", index=False)

    # One representative completed dataset (posterior mean per cell) is written
    # for convenience, clearly labelled -- it is NOT a single source of truth.
    rep = result["summary"][["id", "revenue_mean", "cost_mean", "profit_mean"]].copy()
    rep = rep.rename(columns={"revenue_mean": "revenue", "cost_mean": "cost"})
    rep.to_csv(out / "completed_posterior_means.csv", index=False)

    rho = result["rho_draws"]

    def _to_native(pool):
        out = {}
        for k, v in pool.items():
            if np.isscalar(v) or np.ndim(v) == 0:
                out[k] = None if not np.isfinite(v) else float(v)
            else:
                out[k] = v
        return out

    rubin = {key: _to_native(pool) for key, pool in result["rubin"].items()}

    report = {
        "model": {
            "n_rows": problem.n,
            "n_features": problem.p,
            "feature_names": problem.feature_names,
            "n_equality_observations": int(fit["observations"].A_eq.shape[0]),
            "n_interval_observations": len(fit["observations"].interval_rows),
            "map_sigma": np.sqrt(np.diag(fit["Sigma"])).tolist(),
            "map_rho": float(fit["Sigma"][0, 1] /
                             np.sqrt(fit["Sigma"][0, 0] * fit["Sigma"][1, 1])),
        },
        "mcmc": result["diagnostics"],
        "rho_posterior": {"mean": float(rho.mean()), "sd": float(rho.std(ddof=1)),
                          "q05": float(np.quantile(rho, 0.05)),
                          "q95": float(np.quantile(rho, 0.95))},
        "rubin_pooling": rubin,
        "rubin_note": (
            "Each MCMC chain is treated as one independent imputation and "
            "Rubin's rules are applied across chains.  The grand-mean "
            "between-imputation variance can be exactly zero because the "
            "published regional totals fix every group sum; per-cell between "
            "variance and fraction of missing information are in "
            "rubin_per_cell.csv, where free (unpinned) imputed cells retain "
            "positive between-imputation uncertainty."
        ),
        "outputs": {
            "draws": str(out / "imputed_draws_long.csv"),
            "summary": str(out / "imputed_summary.csv"),
            "rubin_per_cell": str(out / "rubin_per_cell.csv"),
            "posterior_means": str(out / "completed_posterior_means.csv"),
        },
    }
    with open(out / "report.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
