import json
import re
import sys
from collections import defaultdict

from crag_graph import NOT_FOUND, app

ROWS_PATH = "eval/questions.jsonl"
RESULTS_PATH = "eval/results_step2.jsonl"
CITATION = re.compile(r"\[([a-z_]+)\]")
VERDICTS = ["CORRECT", "AMBIGUOUS", "INCORRECT"]
PREFIX = "Partly covered by the recipes:"


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def cited_ids(answer):
    return CITATION.findall(answer)

def check_row(row, result):
    cited = cited_ids(result["answer"])
    checks = {
        "verdict": result["verdict"] == row["expected_verdict"],
        "no false citations": all(recipe_id in result["good_ids"] for recipe_id in cited),
    }
    if row["group"] in ("library", "half"):
        checks["right recipe"] = row["recipe_id"] in cited
    if row["group"] == "half":
        checks["prefix"] = result["answer"].startswith(PREFIX)
    if row["group"] == "general":
        checks["not found"] = result["answer"] == NOT_FOUND
    return checks

def run_eval(rows):
    results = []
    for row in rows:
        config = {"run_name": "recipe-ask-eval", "tags": ["recipe-ask", "eval"],
                  "metadata": {"row_id": row["id"], "group": row["group"]}}
        output = app.invoke({"question": row["question"]}, config=config)
        result = {
            "id": row["id"],
            "group": row["group"],
            "question": row["question"],
            "expected_verdict": row["expected_verdict"],
            "verdict": output["verdict"],
            "scores": output["scores"],
            "good_ids": [doc.metadata["id"] for doc in output["good_docs"]],
            "answer": output["answer"],
        }
        result["checks"] = check_row(row, result)
        results.append(result)
    return results

def save_results(results, path):
    with open(path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

def print_report(results):
    correct_verdicts = sum(result["checks"]["verdict"] for result in results)
    print(f"verdicts: {correct_verdicts}/{len(results)}")

    print(f"\n{'expected → got':16}" + "".join(f"{verdict:>12}" for verdict in VERDICTS))
    for expected in VERDICTS:
        counts = [sum(r["expected_verdict"] == expected and r["verdict"] == got for r in results) for got in VERDICTS]
        print(f"{expected:16}" + "".join(f"{count:>12}" for count in counts))

    passed = defaultdict(int)
    applied = defaultdict(int)
    for result in results:
        for name, ok in result["checks"].items():
            applied[name] += 1
            passed[name] += ok
    print("\nchecks: " + " · ".join(f"{name} {passed[name]}/{applied[name]}" for name in applied))

    failures = [result for result in results if not all(result["checks"].values())]
    if not failures:
        print("\nno failures")
        return
    print("\nFAILED")
    for result in failures:
        failed = [name for name, ok in result["checks"].items() if not ok]
        print(f"{result['id']}  {result['group']:8} failed: {', '.join(failed)}  "
              f"(verdict {result['verdict']}, expected {result['expected_verdict']})")
        print(f"     Q: {result['question']}")
        print("     scores: " + ", ".join(f"{recipe_id} {score:.2f}" for recipe_id, score, _ in result["scores"]))
        print(f"     A: {result['answer']}\n")


if __name__ == "__main__":
    print("--- checks (free) ---")
    print("ok  " if cited_ids("300 g [palak_paneer]. Also [red_lentil_soup].") == ["palak_paneer", "red_lentil_soup"] else "FAIL",
          "citations found")
    print("ok  " if cited_ids("Score [0.5] and step [1].") == [] else "FAIL", "numbers in brackets are not citations")
    library_row = {"group": "library", "expected_verdict": "CORRECT", "recipe_id": "palak_paneer"}
    good = {"verdict": "CORRECT", "good_ids": ["palak_paneer"], "answer": "300 g [palak_paneer]."}
    false_cite = {"verdict": "CORRECT", "good_ids": ["palak_paneer"], "answer": "300 g [chickpea_spinach_curry]."}
    print("ok  " if all(check_row(library_row, good).values()) else "FAIL", "a good library answer passes:", check_row(library_row, good))
    print("ok  " if not check_row(library_row, false_cite)["no false citations"] else "FAIL", "a citation outside good_docs fails")
    general_row = {"group": "general", "expected_verdict": "INCORRECT", "recipe_id": None}
    print("ok  " if "right recipe" not in check_row(general_row, {"verdict": "INCORRECT", "good_ids": [], "answer": NOT_FOUND})
          else "FAIL", "right-recipe check skipped for general rows")

    rows = load_rows(ROWS_PATH)
    print("ok  " if len(rows) == 17 else "FAIL", "questions loaded:", len(rows))

    if len(sys.argv) > 1 and sys.argv[1] == "run":
        print("\n--- step 2 eval: 17 questions (about 85 model calls) ---")
        results = run_eval(rows)
        save_results(results, RESULTS_PATH)
        print_report(results)
        print(f"\nsaved to {RESULTS_PATH}")