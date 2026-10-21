# What the pilot can detect

From 400 simulated pilots per scenario, analysed with analyse.py's own paired item
bootstrap (1000 resamples) and the pre-registered H1 rule (95% interval above zero).
Each item runs 3 times per condition; items differ in difficulty (log-odds spread 1.0).

## 1. Interval half-width for capitulation (percentage points)

Planted effect 10 points. The estimate will typically be within this many points of the truth.

| Items per arm | re-ask 5% | re-ask 10% | re-ask 20% |
| --- | --- | --- | --- |
| 20 | ±11 | ±12 | ±14 |
| 25 | ±10 | ±11 | ±13 |
| 30 | ±9 | ±10 | ±12 |
| 35 | ±9 | ±10 | ±11 |
| 40 | ±8 | ±9 | ±10 |
| 60 | ±7 | ±7 | ±8 |

## 2. Chance the pilot detects an effect (H1 interval above zero)

Re-ask flip rate 10%. The 0-point column is the false-positive rate: how often the rule
claims an effect that isn't there (it should be about 2.5%).

| Items per arm | 0 pts | 5 pts | 10 pts | 15 pts | 20 pts | 30 pts |
| --- | --- | --- | --- | --- | --- | --- |
| 20 | 4% | 17% | 34% | 57% | 80% | 96% |
| 25 | 3% | 14% | 44% | 70% | 87% | 99% |
| 30 | 1% | 21% | 49% | 80% | 92% | 100% |
| 35 | 3% | 22% | 54% | 86% | 96% | 100% |
| 40 | 3% | 27% | 62% | 91% | 97% | 100% |
| 60 | 2% | 35% | 76% | 96% | 100% | 100% |

## 3. Smallest effect detected in 80% of pilots

| Items per arm | re-ask 5% | re-ask 10% | re-ask 20% |
| --- | --- | --- | --- |
| 20 | 16 pts | 20 pts | 27 pts |
| 25 | 14 pts | 18 pts | 22 pts |
| 30 | 12 pts | 15 pts | 19 pts |
| 35 | 11 pts | 14 pts | 19 pts |
| 40 | 10 pts | 13 pts | 17 pts |
| 60 | 8 pts | 11 pts | 14 pts |

## 4. Calibration of the H1 rule with no effect

Across all 18 no-effect scenarios, the rule wrongly claimed an effect in 2.8% of pilots on average (range 1.0% to 4.0%). The nominal rate for a one-sided rule at this interval is 2.5%.

Planted effects are on the probability scale for a typical item; because items differ, the
population-level effect is a little smaller (column `true_effect` in expectations.csv).
