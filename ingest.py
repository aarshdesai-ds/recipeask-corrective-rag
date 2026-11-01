from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv
import os
from pathlib import Path

load_dotenv()

def parse_front_matter(text):
    metadata = {}
    if not text.startswith("---"):
        return ({},text)
    splits = text.split("---", maxsplit = 2)
    header = splits[1]
    body = splits[2]

    for line in header.strip().splitlines():
        if ": " in line:
            pair = line.split(": ", maxsplit = 1)
            attribute, value = pair[0], pair[1]
            metadata[attribute.strip()] = value.strip()
    return (metadata, body.strip())

def load_recipes(folder = "data/recipes"):
    docs = []
    files = os.listdir(folder)
    for file in files:
        if not file.endswith(".md"):
            continue
        path = os.path.join(folder,file)
        text = open(path, encoding="utf-8").read()
        metadata, body = parse_front_matter(text)
        header = f"{metadata['title']} ({metadata['diet']}, {metadata['time_minutes']} min, {metadata['style']}); allergens: {metadata['allergens']}"
        docs.append(Document(page_content=f"{header}\n{body}", metadata=metadata))
    return docs

def build_store():
    docs = load_recipes()
    embedding_model = OpenAIEmbeddings(model = "text-embedding-3-small")
    ids = [doc.metadata['id'] for doc in docs]
    store = Chroma(
        collection_name = "recipes",
        embedding_function=embedding_model,
        persist_directory="kb_db"
    )
    store.add_documents(docs, ids = ids)
    return store

def load_store():
    embedding_model = OpenAIEmbeddings(model = "text-embedding-3-small")
    store = Chroma(
            collection_name = "recipes",
            embedding_function=embedding_model,
            persist_directory="kb_db"
        )
    return store

if __name__ == "__main__":
    store = build_store()
    print(f"Built the recipe store: {len(store.get()['ids'])} recipes in kb_db.")

    opened = load_store()
    print("ok  " if len(opened.get()["ids"]) == 30 else "FAIL", "load_store sees the same 30, without adding")

    for query, expected in [("how much spinach in palak paneer", "palak_paneer"),
                            ("what goes on the thai noodles", "thai_veggie_noodles"),
                            ("roast turkey temperature", None)]:
        hits = opened.similarity_search(query, k=3)
        top = [hit.metadata["id"] for hit in hits]
        if expected:
            print("ok  " if top[0] == expected else "FAIL", f"{query!r} -> {top}")
        else:
            print("info", f"{query!r} -> {top}   (no turkey recipe: CRAG's grader must reject these)")