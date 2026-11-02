import json
import re
import sys
from collections import defaultdict

from crag_graph import CITATION, NOT_FOUND, app

ROWS_PATH = "eval/questions.jsonl"
RESULTS_PATH = "eval/results_step3.jsonl"
VERDICTS = ["CORRECT", "AMBIGUOUS", "INCORRECT"]
PREFIX = "Partly covered by the recipes:"


def load_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

def body(answer):
    return answer.split("\n\nSources:")[0]

def cited_ids(answer):
    return CITATION.findall(body(answer))

def separate_citations(answer):
    sentences = re.split(r"(?<=[.!?])\s+", body(answer))
    recipe_only = web_only = False
    for sentence in sentences:
        ids = CITATION.findall(sentence)
        has_web = any(i.startswith("web_") for i in ids)
        has_recipe = any(not i.startswith("web_") for i in ids)
        recipe_only = recipe_only or (has_recipe and not has_web)
        web_only = web_only or (has_web and not has_recipe)
    return recipe_only and web_only

def check_row(row, result):
    cited = cited_ids(result["answer"])
    web_cited = [i for i in cited if i.startswith("web_")]
    recipe_cited = [i for i in cited if not i.startswith("web_")]
    checks = {
        "verdict": result["verdict"] == row["expected_verdict"],
        "no false citations": all(i in result["good_ids"] + result["web_ids"] for i in cited),
    }
    if row["group"] in ("library", "half"):
        checks["right recipe"] = row["recipe_id"] in recipe_cited
    if row["group"] == "library":
        checks["library only"] = not web_cited
    if row["group"] == "general":
        checks["cites the web"] = bool(web_cited) and not recipe_cited
    if row["group"] == "half":
        checks["cites the web"] = bool(web_cited)
        checks["prefix"] = result["answer"].startswith(PREFIX)
        checks["separate citations"] = separate_citations(result["answer"])
    return checks

def run_eval(rows):
    results = []
    for row in rows:
        config = {"run_name": "recipe-ask-eval", "tags": ["recipe-ask", "eval", "step3"],
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
            "web_query": output.get("web_query", ""),
            "web_ids": [doc.metadata["id"] for doc in output.get("web_docs", [])],
            "kept": output.get("kept_strips", []),
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

    not_found = sum(result["answer"] == NOT_FOUND for result in results)
    print(f"not found: {not_found}/{len(results)}")

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
        if result["web_query"]:
            print(f"     web query: {result['web_query']}")
        print("     kept: " + " | ".join(f"[{source_id}] {text[:60]}" for source_id, text in result["kept"]))
        print(f"     A: {body(result['answer'])}\n")


if __name__ == "__main__":
    print("--- checks (free) ---")
    print("ok  " if cited_ids("300 g [palak_paneer]. Also [red_lentil_soup].") == ["palak_paneer", "red_lentil_soup"] else "FAIL",
          "citations found")
    print("ok  " if cited_ids("Score [0.5] and step [1].") == [] else "FAIL", "numbers in brackets are not citations")

    library_row = {"group": "library", "expected_verdict": "CORRECT", "recipe_id": "palak_paneer"}
    good = {"verdict": "CORRECT", "good_ids": ["palak_paneer"], "web_ids": [], "answer": "300 g [palak_paneer]."}
    false_cite = {"verdict": "CORRECT", "good_ids": ["palak_paneer"], "web_ids": [], "answer": "300 g [chickpea_spinach_curry]."}
    print("ok  " if all(check_row(library_row, good).values()) else "FAIL", "a good library answer passes:", check_row(library_row, good))
    print("ok  " if not check_row(library_row, false_cite)["no false citations"] else "FAIL", "a citation outside the sources fails")

    general_row = {"group": "general", "expected_verdict": "INCORRECT", "recipe_id": None}
    web_answer = {"verdict": "INCORRECT", "good_ids": [], "web_ids": ["web_a"],
                  "answer": "165 °F [web_a].\n\nSources:\n- web_a: https://example.com"}
    print("ok  " if all(check_row(general_row, web_answer).values()) else "FAIL", "a cited web answer passes:", check_row(general_row, web_answer))

    print("ok  " if separate_citations("It freezes well [red_lentil_soup]. It keeps 3 months [web_a].") else "FAIL",
          "one fact per source passes")
    print("ok  " if not separate_citations("It freezes and keeps 3 months [red_lentil_soup][web_a].") else "FAIL",
          "pooled citations fail")

    rows = load_rows(ROWS_PATH)
    print("ok  " if len(rows) == 17 else "FAIL", "questions loaded:", len(rows))

    if len(sys.argv) > 1 and sys.argv[1] == "run":
        group = sys.argv[2] if len(sys.argv) > 2 else None
        selected = [row for row in rows if row["group"] == group] if group else rows
        if not selected:
            sys.exit(f"No rows in group '{group}'. Groups: {sorted({row['group'] for row in rows})}")
        path = RESULTS_PATH.replace(".jsonl", f"_{group}.jsonl") if group else RESULTS_PATH
        print(f"\n--- step 3 eval: {len(selected)} questions{f' (group: {group})' if group else ''} ---")
        results = run_eval(selected)
        save_results(results, path)
        print_report(results)
        print(f"\nsaved to {path}")