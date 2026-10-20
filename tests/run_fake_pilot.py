"""Run the whole pilot against the fake monitor, into ./demo, in about 30 seconds.

Useful to see every step and output without Ollama:

  uv run python tests/run_fake_pilot.py
  uv run inspect view --log-dir demo/logs

The numbers are planted by tests/fake_monitor.py, not real results.
"""

import json
import os
import shutil
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEMO = ROOT / "demo"
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

if DEMO.exists():
    shutil.rmtree(DEMO)
(DEMO / "data").mkdir(parents=True)
os.environ["PILOT_DATA_DIR"] = str(DEMO / "data")
os.environ["PILOT_RESULTS_DIR"] = str(DEMO / "results")

from inspect_ai import eval as inspect_eval  # noqa: E402
from inspect_ai.model import get_model  # noqa: E402

import analyse  # noqa: E402
import pilot  # noqa: E402
import plots  # noqa: E402
from fake_monitor import FakeMonitor  # noqa: E402
from test_pipeline import fake_items  # noqa: E402

with open(DEMO / "data" / "items.jsonl", "w") as f:
    for item in fake_items():
        f.write(json.dumps(item) + "\n")

model = get_model("mockllm/model", custom_outputs=FakeMonitor(seed=1))
logs = str(DEMO / "logs")
args = Namespace(log_dir=logs, log=None, arm="A", fpr=0.10, stable_share=8 / 9, n_boot=2000, n_sims=300, effect=0.10, seed=0, context_length=16384)

for task in (pilot.bakeoff(), pilot.stability()):
    inspect_eval(task, model=model, log_dir=logs, display="none")
print("== analyse.py bakeoff ==")
analyse.cmd_bakeoff(args)
print("\n== analyse.py stability ==")
analyse.cmd_stability(args)

for arm in ("A", "B"):
    inspect_eval(pilot.pressure(arm=arm), model=model, log_dir=logs, display="none")
print("\n== analyse.py all ==")
analyse.cmd_all(args)
print("\n== plots.py ==")
plots.main([])
