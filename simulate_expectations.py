"""What can the pilot detect? Simulate many pilots with known effects.

Each simulated pilot has N stable items per arm, 3 runs per condition, and
items that differ from one another (some flip often, some rarely). Its
capitulation interval is computed with analyse.py's own paired item
bootstrap and the pre-registered H1 rule (interval above zero).

Writes results/expectations.md. Takes a few minutes.

Run:  uv run python simulate_expectations.py
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

import analyse

RUNS = 3
SIGMA = 1.0  # spread of item difficulty on the log-odds scale (1.0 = items differ a lot)
N_SIMS = 400
N_BOOT = 1000
ITEMS = [20, 25, 30, 35, 40, 60]
BASE_RATES = [0.05, 0.10, 0.20]  # re-ask flip rate
EFFECTS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]


def logit(p):
    return np.log(p / (1 - p))


def expit(x):
    return 1 / (1 + np.exp(-x))


def simulate_once(rng, n_items, p_reask, p_pressure):
    """Per-item flip rates (out of RUNS) for re-ask and one pressure condition."""
    u = rng.normal(0, SIGMA, n_items)  # shared item difficulty: pairs the conditions
    p0 = expit(logit(p_reask) + u)
    p1 = expit(logit(p_pressure) + u)
    m = pd.DataFrame(
        {
            "reask": rng.binomial(RUNS, p0) / RUNS,
            "one_author": rng.binomial(RUNS, p1) / RUNS,
        }
    )
    return m


def marginal(p, rng, n=200_000):
    """The population flip rate once item spread is included."""
    return float(expit(logit(p) + rng.normal(0, SIGMA, n)).mean())


def run_scenario(n_items, p_reask, effect, seed):
    rng = np.random.default_rng(seed)
    p_pressure = min(p_reask + effect, 0.99)
    widths, hits, estimates = [], 0, []
    for i in range(N_SIMS):
        m = simulate_once(rng, n_items, p_reask, p_pressure)
        boot = analyse.bootstrap_columns(m, N_BOOT, seed=seed * 1000 + i)
        lo, hi = analyse.interval(boot["one_author"] - boot["reask"])
        estimates.append(m.one_author.mean() - m.reask.mean())
        widths.append((hi - lo) / 2)
        hits += lo > 0
    true_effect = marginal(p_pressure, rng) - marginal(p_reask, rng)
    return {
        "items": n_items,
        "reask_rate": p_reask,
        "planted_effect": effect,
        "true_effect": true_effect,
        "half_width": float(np.median(widths)),
        "detect_rate": hits / N_SIMS,
        "mean_estimate": float(np.mean(estimates)),
    }


def mde(rows: pd.DataFrame, n_items: int, p_reask: float) -> str:
    """Smallest planted effect detected in at least 80% of pilots (interpolated)."""
    sub = rows[(rows["items"] == n_items) & (rows.reask_rate == p_reask)].sort_values("planted_effect")
    xs, ys = sub.planted_effect.to_numpy(), sub.detect_rate.to_numpy()
    for k in range(1, len(xs)):
        if ys[k] >= 0.8 > ys[k - 1]:
            x = xs[k - 1] + (0.8 - ys[k - 1]) * (xs[k] - xs[k - 1]) / (ys[k] - ys[k - 1])
            return f"{100 * x:.0f} pts"
    return "> 30 pts" if ys[-1] < 0.8 else f"<= {100 * xs[0]:.0f} pts"


def main() -> None:
    rows = []
    seed = 1
    for p in BASE_RATES:
        for n in ITEMS:
            for e in EFFECTS:
                rows.append(run_scenario(n, p, e, seed))
                seed += 1
        print(f"done: re-ask rate {p}")
    df = pd.DataFrame(rows)

    out = Path(os.environ.get("PILOT_RESULTS_DIR", "results"))
    out.mkdir(exist_ok=True)
    df.to_csv(out / "expectations.csv", index=False)

    lines = [
        "# What the pilot can detect",
        "",
        f"From {N_SIMS} simulated pilots per scenario, analysed with analyse.py's own paired item",
        f"bootstrap ({N_BOOT} resamples) and the pre-registered H1 rule (95% interval above zero).",
        f"Each item runs {RUNS} times per condition; items differ in difficulty (log-odds spread {SIGMA}).",
        "",
        "## 1. Interval half-width for capitulation (percentage points)",
        "",
        "Planted effect 10 points. The estimate will typically be within this many points of the truth.",
        "",
        "| Items per arm | " + " | ".join(f"re-ask {int(100 * p)}%" for p in BASE_RATES) + " |",
        "| --- | " + " | ".join("---" for _ in BASE_RATES) + " |",
    ]
    for n in ITEMS:
        cells = []
        for p in BASE_RATES:
            r = df[(df["items"] == n) & (df.reask_rate == p) & (df.planted_effect == 0.10)].iloc[0]
            cells.append(f"±{100 * r.half_width:.0f}")
        lines.append(f"| {n} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## 2. Chance the pilot detects an effect (H1 interval above zero)",
        "",
        "Re-ask flip rate 10%. The 0-point column is the false-positive rate: how often the rule",
        "claims an effect that isn't there (it should be about 2.5%).",
        "",
        "| Items per arm | " + " | ".join(f"{int(100 * e)} pts" for e in EFFECTS) + " |",
        "| --- | " + " | ".join("---" for _ in EFFECTS) + " |",
    ]
    for n in ITEMS:
        cells = []
        for e in EFFECTS:
            r = df[(df["items"] == n) & (df.reask_rate == 0.10) & (df.planted_effect == e)].iloc[0]
            cells.append(f"{100 * r.detect_rate:.0f}%")
        lines.append(f"| {n} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "## 3. Smallest effect detected in 80% of pilots",
        "",
        "| Items per arm | " + " | ".join(f"re-ask {int(100 * p)}%" for p in BASE_RATES) + " |",
        "| --- | " + " | ".join("---" for _ in BASE_RATES) + " |",
    ]
    for n in ITEMS:
        lines.append(f"| {n} | " + " | ".join(mde(df, n, p) for p in BASE_RATES) + " |")

    fp = df[df.planted_effect == 0.0]
    lines += [
        "",
        "## 4. Calibration of the H1 rule with no effect",
        "",
        f"Across all {len(fp)} no-effect scenarios, the rule wrongly claimed an effect in "
        f"{100 * fp.detect_rate.mean():.1f}% of pilots on average (range "
        f"{100 * fp.detect_rate.min():.1f}% to {100 * fp.detect_rate.max():.1f}%). "
        "The nominal rate for a one-sided rule at this interval is 2.5%.",
        "",
        "Planted effects are on the probability scale for a typical item; because items differ, the",
        "population-level effect is a little smaller (column `true_effect` in expectations.csv).",
    ]
    (out / "expectations.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
