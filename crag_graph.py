from typing import TypedDict, List, Literal
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from ingest import load_store

load_dotenv()

UPPER = 0.7
LOWER = 0.3

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

structured_model = llm.with_structured_output(DocScore)

grading_prompt = ChatPromptTemplate([
    ("system", """You grade whether ONE recipe answers a cooking question. Return a score from 0.0 to 1.0 and a short reason.
    
    - 1.0: the recipe alone answers the question fully.
    - About 0.5: the recipe is about the right dish, but doesn't answer the whole question.
    - 0.0: the recipe is unrelated, or is about a different dish.
    
    Be conservative with high scores. Use only the recipe text."""), 
    ("human", "Question: {question}\n\nRecipe:\n{recipe}")
])

grading_chain = grading_prompt | structured_model


class GraphState(TypedDict):
    question: str
    docs: List[Document]
    answer: str
    scores: list
    good_docs: List[Document]
    verdict: Literal['CORRECT', 'INCORRECT', 'AMBIGUOUS']


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
    return {"answer": "That isn't in the recipe library."}
    
def route_after_grade(state:GraphState) -> Literal['generate', "not_found"]:
    if state['verdict'] == "INCORRECT":
        return "not_found"
    return "generate"


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
    print("--- graph (free) ---")
    print(sorted(app.get_graph().nodes))
    for scores, expected in [([0.9, 0.1, 0.0, 0.2], "CORRECT"), ([0.1, 0.2, 0.0, 0.25], "INCORRECT"),
                             ([0.5, 0.4, 0.1, 0.0], "AMBIGUOUS"), ([0.7, 0.3, 0.3, 0.3], "AMBIGUOUS")]:
        verdict = verdict_for(scores)
        print("ok  " if verdict == expected else "FAIL", f"{scores} -> {verdict}")

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