# Trademarkia AI/ML Engineer Assignment

## Overview
This project implements a **lightweight semantic search system** using the **20 Newsgroups dataset**.  
The system combines **vector embeddings, fuzzy clustering, semantic caching, and a FastAPI service** to efficiently retrieve semantically related documents while avoiding redundant computations.

The goal of the system is to demonstrate how modern NLP systems can:

- Represent text using semantic embeddings
- Cluster documents into overlapping semantic groups
- Efficiently retrieve similar documents
- Cache results for semantically similar queries
- Serve the system through a REST API

---

# System Architecture

The system consists of the following pipeline:

1. **Text Embedding**
   - Model used:  
   `sentence-transformers/all-MiniLM-L6-v2`
   - Converts documents and queries into dense semantic vectors.

2. **Dimensionality Reduction**
   - PCA reduces embedding dimensionality.
   - Improves clustering efficiency.

3. **Fuzzy Clustering**
   - Fuzzy C-Means clustering.
   - Documents belong to multiple clusters with different probabilities.

4. **Vector Database**
   - FAISS is used for fast similarity search.
   - Enables efficient nearest neighbor retrieval.

5. **Semantic Cache**
   - Avoids recomputing results for similar queries.
   - Uses cosine similarity between query embeddings.

6. **FastAPI Service**
   - Exposes the system via REST API endpoints.

---

# API Endpoints

## 1️⃣ POST `/query`

Accepts a natural language query and returns semantic search results.

### Request

```json
{
  "query": "recent nasa rocket launches"
}
```

### Example Queries

First Query (Cache Miss)

```json
{
  "query": "space shuttle launch"
}
```

Example Response

```json
{
  "query": "space shuttle launch",
  "cache_hit": false,
  "matched_query": null,
  "similarity_score": 0,
  "result": "...",
  "dominant_cluster": 1
}
```

Second Query (Semantic Cache Hit)

```json
{
  "query": "recent nasa rocket launches"
}
```

Example Response

```json
{
  "query": "recent nasa rocket launches",
  "cache_hit": true,
  "matched_query": "space shuttle launch",
  "similarity_score": 0.7083759307861328,
  "result": "['sci.space', 'sci.space', 'sci.space']",
  "dominant_cluster": 1
}
```

### Response Fields

| Field | Description |
|------|-------------|
| query | Input query |
| cache_hit | Whether result was retrieved from cache |
| matched_query | Previously cached similar query |
| similarity_score | Cosine similarity between queries |
| result | Retrieved semantic results |
| dominant_cluster | Cluster with highest membership |

---

## 2️⃣ GET `/cache/stats`

Returns statistics about the semantic cache.

### Example Response

```json
{
  "total_entries": 42,
  "hit_count": 17,
  "miss_count": 25,
  "hit_rate": 0.40
}
```

---

## 3️⃣ DELETE `/cache`

Clears the semantic cache and resets statistics.

---

# Running the Project

## 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

## 2️⃣ Run the FastAPI Server

```bash
uvicorn main:app --reload
```

## 3️⃣ Open API Documentation

Open the following URL in your browser:

```
http://127.0.0.1:8000/docs
```

The interactive API interface allows testing all endpoints.

---

# Dataset

Dataset used: **20 Newsgroups**

- ~20,000 Usenet posts
- 20 topic categories
- Source: UCI Machine Learning Repository

Categories include topics such as:

- politics
- religion
- sports
- technology
- hardware
- space
- automobiles

---

# Key Technologies

| Technology | Purpose |
|-----------|--------|
| Sentence Transformers | Semantic text embeddings |
| FAISS | Fast vector similarity search |
| Fuzzy C-Means | Soft document clustering |
| PCA | Dimensionality reduction |
| FastAPI | API service |
| Python | Core implementation |

---

# Semantic Cache Design

Traditional caching only works when queries are identical.

This project implements a **semantic cache** that recognizes when two queries are *similar in meaning*.

Example:

```
space shuttle launch
rocket launch nasa
```

Even though the wording differs, the semantic meaning is similar.  
The cache detects this using cosine similarity between embeddings.

---

# Cache Similarity Threshold

A tunable parameter controls when queries are considered similar enough to reuse cached results.

```python
SIMILARITY_THRESHOLD = 0.75
```

### Observations

| Threshold | Behavior |
|----------|----------|
| 0.6 | More cache hits but lower precision |
| 0.75 | Balanced reuse and accuracy |
| 0.9 | Strict matching, fewer cache hits |

Final choice: **0.75** as a balanced trade-off.

---

# Project Structure

```
Trademarkia_Assignment
│
├── main.py
├── semantic_engine.py
├── faiss_index.pkl
├── kmeans.pkl
├── pca.pkl
├── labels.pkl
├── requirements.txt
└── README.md
```

---

# Author

**Shivanshu Pauskar**




