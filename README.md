# Trademarkia AI/ML Engineer Assignment

## Overview
This project implements a lightweight semantic search system using the 20 Newsgroups dataset.

The system includes:

• Document embeddings using Sentence Transformers  
• Vector similarity search using FAISS  
• Fuzzy clustering of documents  
• Semantic cache to avoid recomputation  
• FastAPI service exposing the system as an API  

---

## Architecture

1. Documents are embedded using the model:


sentence-transformers/all-MiniLM-L6-v2


2. Embeddings are reduced using PCA.

3. Fuzzy clustering groups documents into semantic clusters.

4. A FAISS vector index enables efficient similarity search.

5. A semantic cache stores query embeddings and results to avoid recomputation.

---

## API Endpoints

### POST `/query`

Request:

```json
{
 "query": "space shuttle launch"
}

Response:

{
 "query": "...",
 "cache_hit": true,
 "matched_query": "...",
 "similarity_score": 0.91,
 "result": "...",
 "dominant_cluster": 3
}
GET /cache/stats

Returns cache statistics:

{
 "total_entries": 42,
 "hit_count": 17,
 "miss_count": 25,
 "hit_rate": 0.40
}
DELETE /cache

Clears the semantic cache.

Running the Project

Install dependencies:

pip install -r requirements.txt

Run server:

uvicorn main:app --reload

Open API docs:

http://127.0.0.1:8000/docs
Dataset

20 Newsgroups dataset from UCI Machine Learning Repository.

Author

Shivanshu Pauskar