import os
from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import (normalize_scores, 
                           hybrid_score,
                           load_movies, 
                           DEFAULT_SEARCH_LIMIT)

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha=0.5, limit=5):
        bm25_results = self._bm25_search(query=query, limit=500*limit)
        chunked_sem_results = self.semantic_search.search_chunks(query=query, limit=500*limit)
        bm25_scores = [r["score"] for r in bm25_results]
        sem_scores = [r["score"] for r in chunked_sem_results]

        normalized_bm25 = normalize_scores(bm25_scores)
        normalized_sem = normalize_scores(sem_scores)
        
        docs: dict[str, dict] = {}

        for result, norm in zip(bm25_results, normalized_bm25):
            doc_id = result["id"]
            docs[doc_id] = {
                "id": doc_id,
                "title": result["title"],
                "document": result["document"],
                "bm25": norm,
                "semantic": 0.0,
            }

        for result, norm in zip(chunked_sem_results, normalized_sem):
            doc_id = result['id']
            if doc_id in docs:
                docs[doc_id]["semantic"] = norm
            else:
                docs[doc_id] = {
                    "id": doc_id,
                    "title": result["title"],
                    "document": result["document"],
                    "bm25": 0.0,
                    "semantic": norm}
                
        # compute hybrid scores        
        for doc in docs.values():
            bm25 = doc['bm25']
            semantic = doc['semantic']
            doc['hybrid'] = hybrid_score(bm25, semantic, alpha)
        
        # sort by hybrid score, descending
        sorted_docs = sorted(
            docs.values(),
            key=lambda d: d['hybrid'],
            reverse=True,
        )

        return sorted_docs[:limit]

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
    
def weighted_search_command(query: str, alpha: float = 0.5, limit: int = DEFAULT_SEARCH_LIMIT) -> dict:
    # 1. load movies
    movies = load_movies()
    # 2. create HybridSearch(movies)
    hybrid_search = HybridSearch(movies)
    # 3. call hybrid.weighted_search(query, alpha, limit *  ???)
    results = hybrid_search.weighted_search(query, alpha, limit)
    # 4. wrap the results + query + alpha into a dict the CLI can print from
    return {
        "query": query,
        "alpha": alpha,
        "results": results,
    }
