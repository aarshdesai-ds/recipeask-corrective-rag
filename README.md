# RecipeAsk: Corrective RAG over a Recipe Library

A LangGraph question-answering workflow built on the Corrective RAG (CRAG) pattern. It answers cooking questions from a library of 30 recipes. Before answering, it grades each retrieved recipe and decides, in code, whether the library answers the question fully, partly or not at all. When the library falls short, it searches the web, filters the results down to the lines that help, and cites every fact to the recipe or web page it came from.

The project was built in three steps: plain RAG, then grading, then the web fallback. Each step was measured on a labelled question set, and the answers were read as well as counted. The sections below report the results of each step, including the changes that made no difference.

*This is an independent learning project. The recipes and questions are synthetic and were written for this project. Web results come from live searches.*

---

## Table of Contents

- [What It Does](#what-it-does)
- [How It Works](#how-it-works)
- [The Recipe Library](#the-recipe-library)
- [Step 1: The Baseline](#step-1-the-baseline)
- [Step 2: Grading and the Verdict](#step-2-grading-and-the-verdict)
- [Step 3: The Web Fallback](#step-3-the-web-fallback)
- [Design Choices](#design-choices)
- [What the Project Taught](#what-the-project-taught)
- [Project Structure](#project-structure)
- [How to Run](#how-to-run)
- [Known Limitations](#known-limitations)
- [Ideas for Extension](#ideas-for-extension)

---

## What It Does

Three questions, three paths:

```
Q: How much spinach goes in the palak paneer?
verdict: CORRECT  (answered from the library)
A: The palak paneer recipe calls for 300 g of spinach [palak_paneer].

Q: What internal temperature should a roast turkey reach?
verdict: INCORRECT  (no turkey recipe, so answered from the web)
A: A roast turkey should reach a minimum internal temperature of 165 °F [web_a][web_b][web_c].

   Sources:
   - web_a: http://www.fsis.usda.gov/food-safety/safe-food-handling-and-preparation/poultry/lets-talk-turkey-roasting
   - web_b: https://shawano.extension.wisc.edu/files/2010/11/Timetable-for-Turkey-Roasting-Handout-2012.pdf
   - web_c: https://www.eatturkey.org/cooking-turkey

Q: Can I freeze the red lentil soup, and for how many months does it keep?
verdict: AMBIGUOUS  (the recipe says it freezes, the web says for how long)
A: Partly covered by the recipes: Yes, you can freeze the red lentil soup. It keeps well in the
   freezer for up to 6 months according to multiple sources [red_lentil_soup][web_a][web_b].
```

---

## How It Works

```
START ──► retrieve ──► grade ──route_after_grade──┬── CORRECT ───────────────────────────────┐
                                                  │                                          ▼
                                                  └── AMBIGUOUS / INCORRECT ──► rewrite ──► web_search ──► refine
                                                                                                           │
                                                                                      route_after_refine ──┤
                                                                                         ├── kept lines ──► generate  ──► END
                                                                                         └── none ────────► not_found ──► END
```

| Node | Does | Decided by |
|---|---|---|
| `retrieve` | The 4 nearest recipes from Chroma. One document per recipe | Code |
| `grade` | Scores each recipe from 0 to 1 against the question, one call per recipe, with a short reason | `gpt-4o-mini`, structured output |
| verdict | CORRECT if any score is above 0.7, INCORRECT if all are below 0.3, otherwise AMBIGUOUS | **Code** (`verdict_for`) |
| `rewrite` | Writes a 4–10 word web query for the part the recipes don't cover. It is given the best recipe's grading reason, so it searches for the gap, not the whole question | Model, structured output |
| `web_search` | Tavily, 3 results. Each becomes a document with an id (`web_a`, `web_b`, `web_c`) and its url | Code |
| `refine` | Splits the sources into strips: recipes into lines, web pages into sentences. One model call returns the numbers of the strips that help. The numbers are range-checked in code | Model chooses, code validates |
| `generate` | Answers only from the kept strips, citing the id of each line used. The "Partly covered" prefix and the Sources footer (urls of the cited web pages only) are added in code | Model writes, code adds |
| `not_found` | A fixed reply when nothing useful is left | Code |
| LangSmith | Every run is traced with `run_name`, `tags` and `metadata` (eval row id and group) | — |

Which sources `refine` uses depends on the verdict: the graded recipes for CORRECT, the web for INCORRECT, and both (recipes first) for AMBIGUOUS.

---

## The Recipe Library

There are 30 synthetic recipes, one Markdown file each, with front matter:

```
id: thai_veggie_noodles
title: Thai Veggie Noodles
diet: vegan
time_minutes: 20
allergens: peanuts, soy, gluten
cuisine: thai
style: quick
servings: 2
```

`ingest.py` stores each file whole as one document, with a header line at the top of its text: "Thai Veggie Noodles (vegan, 20 min, quick); allergens: peanuts, soy, gluten". There is no splitter, because recipes are short and a split would separate an ingredient list from its dish. The ids are the recipe ids, so re-running the ingest replaces documents instead of duplicating them.

**A data fix before step 1.** The allergens were at first only in the document's metadata. The grader and the answer only ever see the text, so "Which allergens are in the pesto gnocchi?" could not have been answered from the library, and would have looked like a retrieval failure. The allergens were added to the header line. If it isn't in the text, the graph can't see it.

---

## Step 1: The Baseline

Plain RAG: retrieve 4 recipes, then answer with citations. There was no grading and no "say you don't know" rule.

| Question | Answer | Problem |
|---|---|---|
| How much spinach in the palak paneer? | "300 g of spinach [palak_paneer]" | None |
| What temperature should a roast turkey reach? | "165°F (74°C)… this specific temperature is not mentioned in the provided recipes." | The right fact, from the model's memory, with no source |
| Can I freeze the lentil soup, and for how long? | "keeps well for up to **3 months** [red_lentil_soup]" | **A false citation.** The recipe only says "Freezes well." The number came from the model, cited as if the recipe said it |

The lentil soup answer is the failure CRAG is for: half the answer is from the library and half from the model, under one citation.

---

## Step 2: Grading and the Verdict

The grader scores each retrieved recipe, and `verdict_for` turns the four scores into one verdict. INCORRECT went to a fixed "not in the library" reply, because there was no web search yet. The answer was built only from recipes scoring above 0.3, with a rule not to answer uncovered parts from the model's own knowledge.

### The question set

`eval/questions.jsonl` has 17 labelled questions:

| Group | Rows | Expected verdict |
|---|---|---|
| library | 6 | CORRECT: the recipe answers it |
| general | 7 | INCORRECT: no recipe answers any part of it |
| half | 4 | AMBIGUOUS: a recipe answers part of it |

`evaluate.py` scores each answer in code: the verdict, that every citation points to a recipe the grader kept, the right recipe id, and the "Partly covered" prefix.

### Results

| Run | Verdicts | Note |
|---|---|---|
| First run, 15 questions | 13 of 15 | A03 ("What wine goes with the duck?") and A04 ("A vegan substitute for fish sauce?") came back INCORRECT, not AMBIGUOUS |
| After relabelling A03 and A04, and adding A05 and A06 | **17 of 17** | No false citations, 17 of 17 |

**The two misses were labelling mistakes.** Both questions name a library dish, but the recipe answers no part of either question. By the rule that AMBIGUOUS means "the library answers part of it", they are INCORRECT. They were moved to the general group, with a note in the file. Two new half questions (A05, A06) were written before the next run and passed on their first run.

**The lentil soup answer became honest:** "Yes, you can freeze the red lentil soup. However, the recipe does not specify how many months it keeps."

**The grader only uses three numbers.** Every score in every run was 1.0, 0.5 or 0.0, the three anchors in its prompt. The 0.7 and 0.3 thresholds therefore do less work than they appear to: any threshold between 0.5 and 1.0 gives the same verdicts.

---

## Step 3: The Web Fallback

AMBIGUOUS and INCORRECT now go through `rewrite`, `web_search` and `refine`, and the answer cites web pages by id, with their urls listed underneath.

| Check | Result |
|---|---|
| Verdicts | 16 of 17 |
| No false citations | 17 of 17 |
| Library questions answered without touching the web | 6 of 6 |
| Web-path questions citing at least one web page | 11 of 11 |
| Answers that ended in "not found" | 0 of 17 (in step 2, 7 of 17) |
| Right recipe cited | 9 of 10 |
| Half answers citing the recipe and the web in **separate sentences** | 2 of 4 |

The seven general questions went from "That isn't in the recipe library" to cited answers. The turkey answer has the same fact as the baseline, now with the USDA page as its source.

### Three findings

**Pooled citations.** On half questions, the model often puts every id at the end of the answer, "[red_lentil_soup][web_a][web_b]", so a reader can't tell which fact came from the recipe and which from the web. One line was added to the answer prompt: "Put each citation right after the sentence it supports." Over three runs on the half questions, separate citations stayed at 2 of 4. In one run the web citations were placed correctly, and the answer showed that two sources disagreed (6 months against 3), but the recipe citation was dropped. The line was kept, and it made no measurable difference.

**A borderline row flips.** A01 ("The risotto uses parmesan. Which cheese can a vegetarian use instead?") was AMBIGUOUS in step 2 and INCORRECT in 3 of the next 4 runs, with the same prompts and model. The recipe only helps through its note that parmesan uses animal rennet, and a grader that only gives 0, 0.5 or 1 flips that judgement all the way.

**A real citation can still be wrong.** In one run, A01's answer said "feta and mature cheddar are also mentioned as alternatives [web_c]". web_c was a video transcript in which a cook made one risotto with feta and another with cheddar: it never offered them as parmesan substitutes, and many fetas and cheddars are made with animal rennet themselves. The citation existed, so every check passed. The transcript had no full stops, so it became one 900-character strip, and the model took a phrase out of the middle. Web strips are now capped at 300 characters, but the underlying problem, a cited source that doesn't support the claim, is not checked by anything in this project. That is what Self-RAG's support check is for.

**Web results change between runs.** The same question returned different pages on different days. On four questions, a change in a web row can come from a prompt change or from the search, and the two can't be told apart.

---

## Design Choices

The pattern follows standard Corrective RAG (score each chunk, verdict by threshold, rewrite, web search, refine, generate), with these changes:

| Common CRAG setup | RecipeAsk | Why |
|---|---|---|
| PDF chunks | Whole recipes, one document each | Recipes are short, and refine does the cutting |
| Sentences under 20 characters dropped | Recipes split into lines, all kept | "- spinach: 300 g" and "Freezes well." are short and are often the answer |
| One filter call per sentence | One call for all strips, returning their numbers | About 1/25th of the calls, with the numbers checked in code |
| The rewrite sees the question | It also sees what the best recipe already covers | So the search targets the gap |
| Answers without sources | Every fact cited by id, with urls listed in code | So the answer can be checked by a reader and by the eval |
| Verdict logic inside the node | `verdict_for` as its own function | Tested for free, without model calls |

---

## What the Project Taught

- **A false citation is worse than no citation.** The baseline's "3 months [red_lentil_soup]" looked trustworthy because it was cited.
- **What isn't in the text doesn't exist for the model.** The allergens were in the metadata and invisible to every node.
- **Read the misses before trusting the labels.** Two of the first eval's failures were labelling mistakes.
- **Coarse scores make thresholds decorative.** A grader that only says 0, 0.5 or 1 turns borderline rows into coin flips.
- **A citation check is not a support check.** Every cited id was real. One cited page didn't say what the answer claimed.
- **The web makes evaluation harder.** Live search results change, so small differences between runs can't be attributed to a prompt.
- **Check the shape of what a tool returns.** The old Tavily tool returned errors as text, and the new one returns a dictionary around the results. Both broke the same loop in different ways.

---

## Project Structure

```
recipe-ask/
│
├── README.md
├── requirements.txt
├── .env                    # Not committed: OPENAI_API_KEY and TAVILY_API_KEY (and LangSmith keys, if used)
│
├── data/
│   └── recipes/            # 30 synthetic recipes with front matter
│
├── ingest.py               # Recipes → Chroma (kb_db/), one document per recipe, allergens in the text
├── crag_graph.py           # State, prompts, all nodes, routers and the graph, with free checks and test modes
├── evaluate.py             # Runs the labelled questions and scores verdicts and citations in code
│
└── eval/
    ├── questions.jsonl          # 17 labelled questions (library, general, half)
    ├── results_step2.jsonl      # Step 2 run: 17 of 17 verdicts
    ├── results_step3.jsonl      # Step 3 run: 16 of 17 verdicts, 11 of 11 web-cited
    └── results_step3_half.jsonl # The latest run on the half questions only
```

`kb_db/` is created when the project runs, and is not committed.

---

## How to Run

1. Create a virtual environment and install the dependencies:

```bash
python -m venv venv
```

```bash
venv\Scripts\activate
```

```bash
python -m pip install -r requirements.txt
```

2. Create a `.env` file with your `OPENAI_API_KEY` and `TAVILY_API_KEY`.

3. Build the recipe store (one embedding call):

```bash
python ingest.py
```

4. Run the free checks (no model calls):

```bash
python crag_graph.py
```

5. Run the full graph on three example questions (about 21 `gpt-4o-mini` calls and 2 web searches):

```bash
python crag_graph.py step3
```

6. Run the evaluation (about 120 model calls and 11 web searches), or one group of questions:

```bash
python evaluate.py run
```

```bash
python evaluate.py run half
```

---

## Known Limitations

- **Cited sources aren't checked for support.** The checks confirm a cited id exists and was kept, not that its text says what the answer claims.
- **Half answers often pool their citations.** 2 of 4 separated the recipe and the web across three runs.
- **The grader is coarse.** It only scores 0, 0.5 or 1, so borderline questions like A01 flip between AMBIGUOUS and INCORRECT.
- **Web results aren't reproducible.** The same query returns different pages on different runs, and the eval doesn't save them for re-scoring.
- **Disagreeing sources aren't resolved.** The answer may report 6 months from one page and 3 from another, or pick one.
- **Web pages can contain markdown, transcripts and adverts.** Refine filters most of it, but junk like "## Preparation Methods" sometimes survives.
- **"Not found" now means neither the library nor the web helped,** but the message still says "That isn't in the recipe library."
- **The evaluation is small and synthetic.** 17 questions, written by the same hand as the recipes, gpt-4o-mini at temperature 0, each run once. One row on the half questions is 25%.

---

## Ideas for Extension

1. **A support check (Self-RAG's IsSUP):** after generating, check each sentence against the strips it cites, and revise or drop unsupported claims.
2. **Save web results per question,** so prompt changes can be compared on the same pages and re-scored for free.
3. **A finer grader,** for example asking which part of the question the recipe answers, so AMBIGUOUS is decided from content rather than a coarse score.
4. **Report conflicts explicitly** when web sources disagree, instead of leaving it to the model.
5. **Use it in MealMate,** as the recipe question-answering step of the weekly planner.
