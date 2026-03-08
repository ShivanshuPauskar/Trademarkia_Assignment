from fastapi import FastAPI
from pydantic import BaseModel

from semantic_engine import process_query, cache_stats, clear_cache

app = FastAPI()


class QueryRequest(BaseModel):
    query: str


@app.post("/query")
def query_endpoint(request: QueryRequest):

    return process_query(request.query)


@app.get("/cache/stats")
def cache_statistics():

    return cache_stats()


@app.delete("/cache")
def clear_cache_endpoint():

    clear_cache()

    return {"message": "Cache cleared"}