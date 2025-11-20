import os
from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import normalize_scores, hybrid_score

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
        chunked_sem_results = self.semantic_search.search(query=query, limit=500*limit)
        bm25_scores = [r["score"] for r in bm25_results]
        sem_scores = [r["score"] for r in chunked_sem_results]

        normalized_bm25 = normalize_scores(bm25_scores)
        normalized_sem = normalize_scores(sem_scores)
        
        
    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
    
