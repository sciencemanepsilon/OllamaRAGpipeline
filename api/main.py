import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Path
from pdf_input import Pdf_DataUpdater
from vector_db import refDB, catDB, ChromaNative

from my_init import (
    chat_model, embedding_model, match_threshold,
    not_relevant_threshold, lock, olla, sys_promt
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not (0 <= match_threshold < not_relevant_threshold <= 2):
        raise ValueError("Require 0 <= MATCH_THRESHOLD < NOT_RELEVANT_THRESHOLD <= 2")

    # next, use env vars to set m1 and m2
    await asyncio.gather(
        Pdf_DataUpdater.OverWriteByLocalFile(
            'Adams_Furniture_Store.pdf', m1='Splitter/TableSimple', m2='Splitter/nlp'
        ),
        olla.pull(chat_model, stream=False),
        olla.pull(embedding_model, stream=False)
    )
    # next, use chromadb async client
    refDB.add(
        documents=Pdf_DataUpdater.chunks,
        ids=ChromaNative.GenIds(Pdf_DataUpdater.chunks),
        metadatas=Pdf_DataUpdater.metadata
    )
    catDB.add(
        documents=Pdf_DataUpdater.chunks,
        ids=ChromaNative.GenIds(Pdf_DataUpdater.chunks),
        metadatas=Pdf_DataUpdater.metadata
    )
    yield
    await asyncio.gather(olla.close(), Pdf_DataUpdater.hpx.aclose())


app = FastAPI(
    title="Furniture Vector API",
    description="Semantic furniture classification and deduplication",
    lifespan=lifespan
)
@app.get("/health")
def health():
    return {"status":"ok", "items":catDB.count()}


@app.get("/add/{item}")
async def add_item(item: str = Path(min_length=1, max_length=160)):
    item = item.strip()
    print(f"Start: item {item}")
    if not item: raise HTTPException(status_code=422, detail="item cannot be blank")

    async with lock: ref = refDB.query(query_texts=[item], n_results=1)
    assert ref is not None
    distance, docu = ChromaNative.GetQueryResults(ref)
    print(f"refDB.query, doc {docu} dist {distance}")

    if distance > not_relevant_threshold:
        print("not relevant, End")
        return {
            "status":"probably not furniture",
            "item":item, "distance":round(distance, 3), "closest_reference":docu
        }
    async with lock:
        catalog_match = catDB.query(
            query_texts=[item],
            n_results=1,
            include=['distances','documents','embeddings']
        )
    distance, docu, vector = ChromaNative.GetQueryResults(catalog_match, vector=True)
    print(f"catDB.query, doc {docu} dist {distance}")

    if catalog_match is None or distance >= match_threshold:
        async with lock:
            # next, add metadata to the catalog VectorDB
            catDB.add(documents=[item], embeddings=[vector], ids=ChromaNative.GenIds([item]))
        ApiResponse = {"status":"item added", "item":item, "total_items":catDB.count()}
        print("item added, End")
        if catalog_match is None: return ApiResponse
        ApiResponse.update(similar_to=docu, distance=distance)
        return ApiResponse
    
    if distance < match_threshold:
        interpretation = await olla.chat(
            model=chat_model, options={"temperature": 0}, messages=[
            {"role": "system", "content": sys_promt},
            {"role":"user", "content":f"Use this information: {docu!r}, to respond to this prompt: {item!r}"}
        ])
        print(f"chat response: {interpretation.message.content}, End")
        return {
            "status":"already similar (RAG Response)",
            "item":item, "similar_to":docu, "distance":round(distance, 3),
            "meaning":interpretation.message.content.strip() or "Chat failed"
        }