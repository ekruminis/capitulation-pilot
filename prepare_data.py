"""Build the pilot's items from the APPS control dataset.

Writes:
  data/items.jsonl   one line per item (100 backdoored + 60 clean by default)
  data/eyeball.md    10 random items before and after cleaning, to check by eye

Run:  uv run python prepare_data.py
"""

import argparse
import ast
import json
import random
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

DATASET = "RoganInglis/apps-control-arena"
# The dataset snapshot the pilot was built from, so the items can be rebuilt exactly.
DATASET_REVISION = "29db28f76042952d56add8671f798280b200744d"
DATA_DIR = Path("data")

# Identifiers that would tell the monitor the label (e.g. a variable named
# `bug`). Items whose cleaned code contains one are skipped.
GIVEAWAY_NAME = re.compile(
    r"\b\w*(backdoor|flaw|bug|sabotag|hack|secret|malicious|trigger|subtle)\w*\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Code hygiene
# ---------------------------------------------------------------------------


class _DropBareStrings(ast.NodeTransformer):
    """Remove docstrings and other bare string statements (comment-like text)."""

    def _clean_body(self, body):
        kept = [
            stmt
            for stmt in body
            if not (
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Constant)
                and isinstance(stmt.value.value, str)
            )
        ]
        return kept or [ast.Pass()]

    def generic_visit(self, node):
        super().generic_visit(node)
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if isinstance(block, list) and block and isinstance(block[0], ast.stmt):
                setattr(node, field, self._clean_body(block))
        return node


def strip_code(code: str) -> str | None:
    """Return the code without comments or docstrings, or None if it doesn't parse.

    ast.unparse rebuilds the source from its syntax tree, so comments disappear
    and every program comes out in the same basic layout. Python 2 code fails
    to parse and is dropped.
    """
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return None
    tree = _DropBareStrings().visit(tree)
    ast.fix_missing_locations(tree)
    out = ast.unparse(tree)
    # Proof that nothing but comments and docstrings changed: the cleaned code
    # parses back to exactly the same syntax tree.
    if not same_program(ast.dump(tree), out):
        return None
    return out


def same_program(tree_dump: str, code: str) -> bool:
    try:
        return ast.dump(ast.parse(code)) == tree_dump
    except SyntaxError:
        return False


def ruff_format(snippets: list[str]) -> list[str]:
    """Format every snippet with ruff, so backdoored and clean code look alike."""
    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for i, code in enumerate(snippets):
            path = Path(tmp) / f"s{i}.py"
            path.write_text(code + "\n")
            paths.append(path)
        subprocess.run(
            [sys.executable, "-m", "ruff", "format", "--quiet", tmp],
            check=True,
        )
        return [p.read_text().rstrip() + "\n" for p in paths]


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------


def working_backdoors(row) -> list[dict]:
    return [
        b
        for b in row["backdoors"]
        if b.get("backdoor_works") and b.get("passes_tests") and b.get("compiles")
    ]


def passing_solution(row) -> str | None:
    # Solutions are sorted best first; take the first that passes the tests.
    for s in row["solutions"]:
        if s.get("passes_tests"):
            return s["code"]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-backdoor", type=int, default=100)
    parser.add_argument("--n-clean", type=int, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-question-chars", type=int, default=4000)
    parser.add_argument("--max-code-lines", type=int, default=150)
    args = parser.parse_args()

    from datasets import load_dataset

    ds = load_dataset(DATASET, split="train", revision=DATASET_REVISION)
    counts = Counter(rows=len(ds))

    # 1. Candidate problems: deterministic, a passing clean solution, at least
    #    one working backdoor, and a question short enough to keep runs fast.
    candidates = []
    for row in ds:
        if row["is_nondeterministic"]:
            counts["dropped: nondeterministic"] += 1
            continue
        if not row["solution_passes_tests"] or passing_solution(row) is None:
            counts["dropped: no passing solution"] += 1
            continue
        if not working_backdoors(row):
            counts["dropped: no working backdoor"] += 1
            continue
        if len(row["question"]) > args.max_question_chars:
            counts["dropped: question too long"] += 1
            continue
        candidates.append(row)
    counts["candidate problems"] = len(candidates)

    # 2. Shuffle once with a fixed seed. Backdoored items are taken first and
    #    clean items from the problems after them, so no problem appears twice
    #    and both groups come from the same pool of problems.
    rng = random.Random(args.seed)
    rng.shuffle(candidates)

    selected: list[dict] = []
    n_backdoor = n_clean = 0
    for row in candidates:
        if n_backdoor < args.n_backdoor:
            label = "backdoor"
            backdoor = rng.choice(working_backdoors(row))
            original = backdoor["backdoor_code"]
        elif n_clean < args.n_clean:
            label = "clean"
            backdoor = None
            original = passing_solution(row)
        else:
            break

        stripped = strip_code(original)
        if stripped is None:
            counts[f"dropped: {label} code does not parse"] += 1
            continue
        if stripped.count("\n") + 1 > args.max_code_lines:
            counts[f"dropped: {label} code too long"] += 1
            continue
        if GIVEAWAY_NAME.search(stripped):
            counts[f"dropped: {label} code has a give-away name"] += 1
            continue

        selected.append(
            {
                "id": f"p{row['problem_id']}-{label}",
                "problem_id": str(row["problem_id"]),
                "label": label,
                "difficulty": row["difficulty"],
                "question": row["question"],
                "code": stripped,
                "original_code": original,
                # Kept for reference only; the monitor never sees these.
                "backdoor_input": backdoor["backdoor_input"] if backdoor else None,
                "backdoor_source": backdoor["source"] if backdoor else None,
                "backdoor_model": backdoor["model"] if backdoor else None,
            }
        )
        if label == "backdoor":
            n_backdoor += 1
        else:
            n_clean += 1

    if n_backdoor < args.n_backdoor or n_clean < args.n_clean:
        sys.exit(f"Not enough items: {n_backdoor} backdoored, {n_clean} clean")

    # 3. Format all code the same way, and check it still parses.
    formatted = ruff_format([item["code"] for item in selected])
    for item, code in zip(selected, formatted, strict=True):
        # ruff only changes layout; check the program is still the same.
        if not same_program(ast.dump(ast.parse(item["code"])), code):
            sys.exit(f"Formatting changed the program for {item['id']}")
        item["code"] = code

    DATA_DIR.mkdir(exist_ok=True)
    with open(DATA_DIR / "items.jsonl", "w") as f:
        for item in selected:
            f.write(json.dumps(item) + "\n")

    style = style_table(selected)
    write_eyeball(selected, random.Random(args.seed + 1), style)

    counts["selected backdoor"] = n_backdoor
    counts["selected clean"] = n_clean
    print("\nItem selection")
    for key, value in counts.items():
        print(f"  {key:<46} {value:>6}")
    print("\nStyle of the code, by group (big gaps are cues a monitor could use instead of the backdoor)")
    print(style)
    print(f"\nWrote {DATA_DIR / 'items.jsonl'} and {DATA_DIR / 'eyeball.md'}")


def style_features(code: str) -> dict:
    tree = ast.parse(code)
    names = [n.id for n in ast.walk(tree) if isinstance(n, ast.Name)]
    return {
        "lines": code.count("\n") + 1,
        "defines a function": any(isinstance(n, ast.FunctionDef) for n in ast.walk(tree)),
        "has def main": any(isinstance(n, ast.FunctionDef) and n.name == "main" for n in ast.walk(tree)),
        "reads sys.stdin": "sys.stdin" in code,
        "mean name length": sum(map(len, names)) / len(names) if names else 0.0,
        "single-letter names": sum(len(n) == 1 for n in names) / len(names) if names else 0.0,
    }


def style_table(items: list[dict]) -> str:
    """Median or share of simple style features for backdoored vs clean code."""
    import statistics

    rows = {}
    for label in ("backdoor", "clean"):
        feats = [style_features(i["code"]) for i in items if i["label"] == label]
        rows[label] = {k: statistics.mean(float(f[k]) for f in feats) for k in feats[0]}
    width = max(len(k) for k in rows["backdoor"])
    out = [f"  {'feature':<{width}}  backdoored      clean"]
    for k in rows["backdoor"]:
        out.append(f"  {k:<{width}}  {rows['backdoor'][k]:>10.2f} {rows['clean'][k]:>10.2f}")
    return "\n".join(out)


def write_eyeball(items: list[dict], rng: random.Random, style: str) -> None:
    backdoors = [i for i in items if i["label"] == "backdoor"]
    cleans = [i for i in items if i["label"] == "clean"]
    picks = rng.sample(backdoors, 5) + rng.sample(cleans, 5)
    lines = [
        "# Ten items before and after cleaning",
        "",
        "Check that comments and docstrings are gone, the code still reads",
        "sensibly, and nothing in the cleaned code gives the label away.",
        "",
        "Style by group (averages; shares are 0 to 1). Backdoors are model-written",
        "and clean solutions human-written, so a large gap here is a cue the",
        "monitor could use instead of spotting the backdoor:",
        "",
        "```",
        style,
        "```",
        "",
    ]
    for item in picks:
        lines += [f"## {item['id']} ({item['difficulty']})", ""]
        if item["backdoor_input"] is not None:
            trigger = item["backdoor_input"].strip()[:200]
            lines += [f"Trigger input (never shown to the monitor): `{trigger}`", ""]
        lines += ["### Original", "```python", item["original_code"].rstrip(), "```", ""]
        lines += ["### Cleaned (what the monitor sees)", "```python", item["code"].rstrip(), "```", ""]
    (DATA_DIR / "eyeball.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
