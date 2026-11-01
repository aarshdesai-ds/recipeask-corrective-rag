from typing import TypedDict, List
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from ingest import load_store

load_dotenv()

store = load_store()
retriever = store.as_retriever(search_kwargs = {"k" : 4})
llm = ChatOpenAI(model = "gpt-4o-mini", temperature = 0)
answer_prompt = ChatPromptTemplate([
        ("system", "You are a cooking assistant. Answer the question using the recipes below. Cite the recipe id you used in square brackets, like [palak_paneer]. Keep it short."),
        ("human", "Question: {question}\n\nRecipes:\n{context}")
    ])
answer_chain = answer_prompt | llm

class GraphState(TypedDict):
    question: str
    docs: List[Document]
    answer: str

def format_docs(docs):
    blocks = []
    for doc in docs:
        block = f"[{doc.metadata['id']}]\n{doc.page_content}"
        blocks.append(block)
    return "\n\n---\n\n".join(blocks)

def retrieve(state: GraphState):
    docs = retriever.invoke(state['question'])
    return {"docs": docs}

def generate(state: GraphState):
    context = format_docs(state['docs'])
    result = answer_chain.invoke({"question":state['question'], "context": context}).content
    return {"answer":result}

graph = StateGraph(GraphState)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)

graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

app = graph.compile()

if __name__ == "__main__":
    print("--- graph (free) ---")
    print(sorted(app.get_graph().nodes))
    sample = retriever.invoke("how much spinach in palak paneer")
    formatted = format_docs(sample)
    print("ok  " if "[palak_paneer]" in formatted else "FAIL", "recipes are labelled with their ids")
    print("ok  " if "allergens:" in formatted else "FAIL", "allergens are in the text the model sees")

    print("\n--- baseline on 3 questions (3 model calls) ---")
    for question in ["How much spinach goes in the palak paneer?",
                     "What internal temperature should a roast turkey reach?",
                     "Can I freeze the red lentil soup, and for how many months does it keep?"]:
        result = app.invoke({"question": question},
                            config={"run_name": "recipe-ask", "tags": ["recipe-ask", "test", "baseline"]})
        print(f"\nQ: {question}")
        print("retrieved:", [doc.metadata["id"] for doc in result["docs"]])
        print("answer:", result["answer"])