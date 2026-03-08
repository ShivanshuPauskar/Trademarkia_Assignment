import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# load trained artifacts
with open("kmeans.pkl", "rb") as f:
    kmeans = pickle.load(f)

with open("pca.pkl", "rb") as f:
    pca = pickle.load(f)

with open("faiss_index.pkl", "rb") as f:
    index = pickle.load(f)

with open("labels.pkl", "rb") as f:
    labels = pickle.load(f)

semantic_cache = []

hit_count = 0
miss_count = 0

SIMILARITY_THRESHOLD = 0.70


def process_query(query):

    global hit_count, miss_count

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding)

    query_embedding_reduced = pca.transform(query_embedding)

    query_cluster = kmeans.predict(query_embedding_reduced)[0]

    best_similarity = 0
    best_match = None

    for entry in semantic_cache:

        if entry["cluster"] != query_cluster:
            continue

        sim = cosine_similarity(
            query_embedding,
            entry["embedding"].reshape(1, -1)
        )[0][0]

        if sim > best_similarity:
            best_similarity = sim
            best_match = entry

    if best_similarity >= SIMILARITY_THRESHOLD:

        hit_count += 1

        return {
            "query": query,
            "cache_hit": True,
            "matched_query": best_match["query"],
            "similarity_score": float(best_similarity),
            "result": best_match["result"],
            "dominant_cluster": int(best_match["cluster"])
        }

    miss_count += 1

    query_embedding_faiss = query_embedding.astype("float32")

    distances, indices = index.search(query_embedding_faiss, 3)

    results = [labels[i] for i in indices[0]]

    result_text = str(results)

    semantic_cache.append({
        "query": query,
        "embedding": query_embedding[0],
        "result": result_text,
        "cluster": int(query_cluster)
    })

    return {
        "query": query,
        "cache_hit": False,
        "matched_query": None,
        "similarity_score": 0,
        "result": result_text,
        "dominant_cluster": int(query_cluster)
    }


def cache_stats():

    total = len(semantic_cache)

    hit_rate = hit_count / (hit_count + miss_count) if (hit_count + miss_count) > 0 else 0

    return {
        "total_entries": total,
        "hit_count": hit_count,
        "miss_count": miss_count,
        "hit_rate": hit_rate
    }


def clear_cache():

    global semantic_cache, hit_count, miss_count

    semantic_cache = []
    hit_count = 0
    miss_count = 0