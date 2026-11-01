from typing import TypedDict, List, Literal
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from ingest import load_store
import re
import string
from langchain_tavily import TavilySearch


load_dotenv()

UPPER = 0.7
LOWER = 0.3
NOT_FOUND = "That isn't in the recipe library."

store = load_store()
retriever = store.as_retriever(search_kwargs = {"k" : 4})
llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)
answer_prompt = ChatPromptTemplate([
        ("system", "You are a cooking assistant. Answer the question using the recipes below. Cite the recipe id you used in square brackets, like [palak_paneer]. Keep it short. If the recipes don't answer part of the question, say which part, and don't answer that part from your own knowledge."),
        ("human", "Question: {question}\n\nRecipes:\n{context}")
    ])
answer_chain = answer_prompt | llm

class DocScore(BaseModel):
    score: float = Field(... , description="How well this recipe alone answers the question. 1.0: it answers it fully. 0.0: it's unrelated", ge=0, le=1)
    reason: str = Field(..., description="One short line explaining the score")

grading_model = llm.with_structured_output(DocScore)

grading_prompt = ChatPromptTemplate([
    ("system", """You grade whether ONE recipe answers a cooking question. Return a score from 0.0 to 1.0 and a short reason.
    
    - 1.0: the recipe alone answers the question fully.
    - About 0.5: the recipe is about the right dish, but doesn't answer the whole question.
    - 0.0: the recipe is unrelated, or is about a different dish.
    
    Be conservative with high scores. Use only the recipe text."""), 
    ("human", "Question: {question}\n\nRecipe:\n{recipe}")
])

grading_chain = grading_prompt | grading_model

class WebQuery(BaseModel):
    query: str = Field(... , description = "4-10 keywords for a web search. No full sentence, and no answer")

rewrite_prompt = ChatPromptTemplate([
    ("system", """You write web search queries for a cooking assistant.
    You get a question and a note on what the recipe library already covers.
    Write a query for only the part the recipes do NOT cover.

    - Use 4-10 keywords, not a sentence.
    - Keep the dish or ingredient names from the question.
    - Do not answer the question."""),
    ("human", "Question: {question}\n\nWhat the recipes cover: {covered}")
])
rewrite_model = llm.with_structured_output(WebQuery)
rewrite_chain = rewrite_prompt | rewrite_model

tavily_tool = TavilySearch(max_results=3)

def web_ids(count):
    return [f"web_{letter}" for letter in string.ascii_lowercase[:count]]

class GraphState(TypedDict):
    question: str
    docs: List[Document]
    answer: str
    scores: list
    good_docs: List[Document]
    verdict: Literal['CORRECT', 'INCORRECT', 'AMBIGUOUS']
    web_query: str
    web_docs: List[Document]
    strips: List[str]
    kept_strips: List[str]


def format_docs(docs):
    blocks = []
    for doc in docs:
        block = f"[{doc.metadata['id']}]\n{doc.page_content}"
        blocks.append(block)
    return "\n\n---\n\n".join(blocks)

def retrieve(state: GraphState):
    docs = retriever.invoke(state['question'])
    return {"docs": docs}

def verdict_for(scores) -> Literal['CORRECT', 'INCORRECT', 'AMBIGUOUS']:
    if any(score > UPPER for score in scores):
        return "CORRECT"
    elif all(score < LOWER for score in scores):
        return "INCORRECT" 
    return "AMBIGUOUS"

def grade(state: GraphState):
    good_docs = []
    numeric_scores= []
    scores = []
    docs = state['docs']
    for doc in docs:
        grade_output = grading_chain.invoke({"question": state['question'], "recipe": doc.page_content})
        scores.append((doc.metadata['id'], grade_output.score, grade_output.reason))
        numeric_scores.append(grade_output.score)
        if grade_output.score > LOWER:
            good_docs.append(doc)
    verdict = verdict_for(numeric_scores)
    return {"scores": scores, "good_docs": good_docs, "verdict": verdict}

def generate(state: GraphState):
    context = format_docs(state['good_docs'])
    result = answer_chain.invoke({"question":state['question'], "context": context}).content
    if state['verdict'] == "AMBIGUOUS":
        result = f"Partly covered by the recipes: {result}"

    return {"answer":result}


def not_found(state: GraphState):
    return {"answer": NOT_FOUND}
    
def route_after_grade(state:GraphState) -> Literal['generate', "not_found"]:
    if state['verdict'] == "INCORRECT":
        return "not_found"
    return "generate"

def covered_note(scores):
    best_recipe = max(scores, key = lambda x: x[1])
    score = best_recipe[1]
    if score > LOWER:
        covered = f"Recipe {best_recipe[0]}: {best_recipe[2]}"
    else:
        covered = "No recipe covers it."
    return covered

def rewrite(state: GraphState):
    scores = state['scores']
    covered = covered_note(scores)
    web_query = rewrite_chain.invoke({"question": state['question'], "covered": covered}).query
    return {"web_query": web_query}

def web_search(state: GraphState):
    q = state.get("web_query") or state['question']
    response = tavily_tool.invoke({"query": q})
    if isinstance(response, dict) and response.get("error"):
        raise RuntimeError(f"Tavily search failed: {response['error']}")
    results = response.get("results", []) if isinstance(response, dict) else []

    web_docs: List[Document] = []
    for web_id, r in zip(web_ids(len(results)), results):
        title = r.get("title", "")
        url = r.get("url", "")
        content = r.get("content", "") or r.get("snippet", "")
        text = f"TITLE: {title}\nCONTENT:\n{content}"
        web_docs.append(Document(page_content=text, metadata={"url": url, "id": web_id}))
    return {"web_docs": web_docs}

graph = StateGraph(GraphState)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)
graph.add_node("grade", grade)
graph.add_node("not_found", not_found)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "grade")
graph.add_conditional_edges("grade", route_after_grade,{
    "generate":"generate",
    "not_found":"not_found"
})

graph.add_edge("not_found", END)
graph.add_edge("generate", END)

app = graph.compile()

if __name__ == "__main__":
    import sys

    print("--- graph (free) ---")
    print(sorted(app.get_graph().nodes))

    print("\n--- verdict_for (free) ---")
    for scores, expected in [([0.9, 0.1, 0.0, 0.2], "CORRECT"), ([0.1, 0.2, 0.0, 0.25], "INCORRECT"),
                             ([0.5, 0.4, 0.1, 0.0], "AMBIGUOUS"), ([0.7, 0.3, 0.3, 0.3], "AMBIGUOUS")]:
        verdict = verdict_for(scores)
        print("ok  " if verdict == expected else "FAIL", f"{scores} -> {verdict}")

    print("\n--- covered_note (free) ---")
    note = covered_note([("beef_chilli", 0.0, "unrelated"), ("red_lentil_soup", 0.5, "freezes well, not how long")])
    print("ok  " if note == "Recipe red_lentil_soup: freezes well, not how long" else "FAIL",
          "uses the best recipe, not the first:", note)
    note = covered_note([("duck_with_orange", 0.0, "unrelated to turkey")])
    print("ok  " if note == "No recipe covers it." else "FAIL", "nothing above LOWER:", note)

    print("\n--- web_ids (free) ---")
    print("ok  " if web_ids(3) == ["web_a", "web_b", "web_c"] else "FAIL", "three letter ids:", web_ids(3))
    print("ok  " if web_ids(0) == [] else "FAIL", "no results, no ids")

    if len(sys.argv) > 1 and sys.argv[1] == "grader":
        print("\n--- grader on 3 questions (about 15 model calls) ---")
        for question, expected in [("How much spinach goes in the palak paneer?", "CORRECT"),
                                   ("What internal temperature should a roast turkey reach?", "INCORRECT"),
                                   ("Can I freeze the red lentil soup, and for how many months does it keep?", "AMBIGUOUS")]:
            result = app.invoke({"question": question},
                                config={"run_name": "recipe-ask", "tags": ["recipe-ask", "test", "grader"]})
            print(f"\nQ: {question}")
            for recipe_id, score, reason in result["scores"]:
                print(f"   {score:.2f}  {recipe_id:28} {reason}")
            print("ok  " if result["verdict"] == expected else "FAIL", "verdict:", result["verdict"], "| expected:", expected)
            print("answer:", result["answer"])

    if len(sys.argv) > 1 and sys.argv[1] == "web":
        print("\n--- rewrite + web_search, called directly (2 model calls, 2 web searches) ---")
        cases = [("Can I freeze the red lentil soup, and for how many months does it keep?",
                  [("red_lentil_soup", 0.5, "mentions it freezes well but not how long"), ("beef_chilli", 0.0, "unrelated")]),
                 ("What internal temperature should a roast turkey reach?",
                  [("duck_with_orange", 0.0, "a duck recipe, not turkey")])]
        for question, scores in cases:
            query = rewrite({"question": question, "scores": scores})["web_query"]
            print(f"\nQ: {question}")
            print("covered:", covered_note(scores))
            print("query:", query, f"({len(query.split())} words)")
            web_docs = web_search({"question": question, "web_query": query})["web_docs"]
            print("ok  " if web_docs and all(doc.metadata["id"].startswith("web_") and doc.metadata["url"] for doc in web_docs)
                  else "FAIL", "results tagged with a web id and a url:", len(web_docs))
            for doc in web_docs:
                print(f"   [{doc.metadata['id']}] {doc.metadata['url']}")
                print(f"        {doc.page_content[:120]!r}")