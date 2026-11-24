#!/usr/bin/env python3
"""GQA balanced exact-match accuracy and official type breakdowns.

This mirrors the accuracy-related part of GQA's official evaluator. It does not
claim validity, plausibility, consistency or grounding metrics.
"""

import argparse
from collections import defaultdict
import json
from pathlib import Path


def percentages(groups):
    return {name: {"accuracy": 100.0 * sum(values) / len(values), "questions": len(values)}
            for name, values in sorted(groups.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", required=True,
                        help="raw *_balanced_questions.json dictionary")
    parser.add_argument("--predictions", required=True,
                        help="JSON list with questionId and prediction")
    parser.add_argument("--output")
    args = parser.parse_args()
    questions = json.loads(Path(args.questions).read_text())
    rows = json.loads(Path(args.predictions).read_text())
    predictions = {str(row["questionId"]): str(row["prediction"]) for row in rows}
    if len(predictions) != len(rows):
        raise ValueError("duplicate question IDs in predictions")
    balanced = {str(qid): row for qid, row in questions.items()
                if row.get("isBalanced", True)}
    if not balanced:
        raise ValueError("no balanced questions to evaluate")
    missing = sorted(set(balanced) - set(predictions))
    extra = sorted(set(predictions) - set(balanced))
    if missing or extra:
        raise ValueError("prediction coverage mismatch: missing=%d extra=%d" %
                         (len(missing), len(extra)))
    structural, semantic, binary_open = defaultdict(list), defaultdict(list), defaultdict(list)
    scores = []
    for qid, question in balanced.items():
        score = float(predictions[qid] == str(question["answer"]))
        scores.append(score)
        structure = question["types"]["structural"]
        structural[structure].append(score)
        semantic[question["types"]["semantic"]].append(score)
        binary_open["open" if structure == "query" else "binary"].append(score)
    result = {"overall": 100.0 * sum(scores) / len(scores),
              "questions": len(scores), "binary_open": percentages(binary_open),
              "structural": percentages(structural), "semantic": percentages(semantic),
              "scope": "official GQA balanced exact-match accuracy fields only",
              "not_computed": ["validity", "plausibility", "consistency", "grounding"]}
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
