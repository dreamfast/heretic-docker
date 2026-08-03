#!/usr/bin/env python3
"""Parse a heretic Optuna journal checkpoint JSONL to extract Pareto front metrics.

Works with heretic master's scorer-plugin format (trial user_attrs contain a
"scores" list of {"name": ..., "score": {"value": ..., "rich_display": ...}}).

Usage:
    python3 parse_checkpoint.py <checkpoint_dir> <model_name> [trial_index]

Outputs JSON: {"trial_number": 78, "keyword_rate": 2, "kl_divergence": 0.0001, "pareto_size": 9}
"""

import json
import sys
from collections import defaultdict
from pathlib import Path


def sanitize_model_name(model: str) -> str:
    return "".join(
        c if (c.isalnum() or c in ["_", "-"]) else "--"
        for c in model
    )


def parse_checkpoint(checkpoint_dir: str, model_name: str) -> list[dict]:
    sanitized = sanitize_model_name(model_name)
    checkpoint_file = Path(checkpoint_dir) / f"{sanitized}.jsonl"

    if not checkpoint_file.exists():
        print(f"ERROR: {checkpoint_file} not found", file=sys.stderr)
        sys.exit(1)

    trial_attrs = defaultdict(dict)
    with open(checkpoint_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("op_code") == 8:
                tid = entry["trial_id"]
                attrs = entry.get("user_attr", {})
                trial_attrs[tid].update(attrs)

    trials = []
    for tid, attrs in trial_attrs.items():
        scores_list = attrs.get("scores")
        if scores_list is None:
            continue

        scores = {}
        for record in scores_list:
            name = record["name"]
            value = record["score"]["value"]
            display = record["score"].get("rich_display", str(value))
            scores[name] = value
            scores[f"__display__{name}"] = display

        keyword_rate = scores.get("Keywords", scores.get("KeywordRate", 0))
        kl_divergence = scores.get("KL divergence", scores.get("KLDivergence", 0))
        keyword_display = scores.get("__display__Keywords", scores.get("__display__KeywordRate", ""))

        if kl_divergence is None:
            kl_divergence = 0
        if keyword_rate is None:
            keyword_rate = 0

        # Extract refusal count from display string like "4/100"
        refusal_count = 0
        if keyword_display and "/" in keyword_display:
            try:
                refusal_count = int(keyword_display.split("/")[0])
            except ValueError:
                refusal_count = int(round(keyword_rate * 100))

        trials.append({
            "trial_id": tid,
            "trial_number": attrs.get("index", tid),
            "keyword_rate": float(keyword_rate),
            "kl_divergence": float(kl_divergence),
            "refusal_count": refusal_count,
        })

    if not trials:
        print("ERROR: No completed trials with scores found", file=sys.stderr)
        sys.exit(1)

    pareto = []
    for t in trials:
        dominated = False
        for other in trials:
            if other is t:
                continue
            if (
                other["refusal_count"] <= t["refusal_count"]
                and other["kl_divergence"] <= t["kl_divergence"]
                and (
                    other["refusal_count"] < t["refusal_count"]
                    or other["kl_divergence"] < t["kl_divergence"]
                )
            ):
                dominated = True
                break
        if not dominated:
            pareto.append(t)

    pareto.sort(key=lambda t: (t["refusal_count"], t["kl_divergence"]))

    return pareto


def main():
    if len(sys.argv) < 3:
        print(
            f"Usage: {sys.argv[0]} <checkpoint_dir> <model_name> [trial_index]",
            file=sys.stderr,
        )
        sys.exit(1)

    checkpoint_dir = sys.argv[1]
    model_name = sys.argv[2]
    trial_index = int(sys.argv[3]) if len(sys.argv) > 3 else 0

    pareto = parse_checkpoint(checkpoint_dir, model_name)

    if trial_index >= len(pareto):
        print(
            f"ERROR: trial_index {trial_index} >= Pareto front size {len(pareto)}",
            file=sys.stderr,
        )
        sys.exit(1)

    t = pareto[trial_index]
    result = {
        "trial_number": t["trial_number"],
        "refusal_count": t["refusal_count"],
        "keyword_rate": t["keyword_rate"],
        "kl_divergence": t["kl_divergence"],
        "pareto_size": len(pareto),
        "trial_index_in_pareto": trial_index,
    }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
