import os
from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import (normalize_scores, 
                           hybrid_score,
                           load_movies, 
                           DEFAULT_SEARCH_LIMIT)

import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import time


load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
print(f"Using key {api_key[:6]}...")

client = genai.Client(api_key=api_key)

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

    def rrf_search(self, query, k, limit=10, rerank_method=None):
        bm25_results = self._bm25_search(query=query, limit=500*limit)
        chunked_sem_results = self.semantic_search.search_chunks(query=query, limit=500*limit)

        docs: dict[str, dict] = {}

        for rank, result in enumerate(bm25_results, start=1):
            doc_id = result['id']
            if result["id"] not in docs:
                docs[doc_id] = {
                    "id": doc_id,
                    "title": result["title"],
                    "document": result["document"],
                    "bm25_rank": rank,
                    "semantic_rank": None,
                    "rrf_score": 0.0}
                
        for rank, result in enumerate(chunked_sem_results, start=1):
            doc_id = result['id']
            if doc_id in docs:
                docs[doc_id]["semantic_rank"] = rank
            else:
                docs[doc_id] = {
                    "id": doc_id,
                    "title": result["title"],
                    "document": result["document"],
                    "bm25_rank": None,
                    "semantic_rank": rank,
                    "rrf_score": 0.0}
                
        # compute rrf        
        for doc in docs.values():
            bm25_rank = doc['bm25_rank']
            semantic_rank = doc['semantic_rank']
            if bm25_rank is not None:
                doc['rrf_score'] += 1 / (k + bm25_rank)
            if semantic_rank is not None:
                doc['rrf_score'] += 1 / (k + semantic_rank)

        sorted_docs = sorted(docs.values(), 
                             key= lambda d: d['rrf_score'], 
                             reverse = True)
        
        return sorted_docs[:limit]

    def _rerank_individual(self, query: str, docs: list[dict]):
        print("Reranking results with individual LLM calls...")

        for doc in docs:
            system_instruction = f"""Rate how well this movie matches the search query.

            Query: "{query}"
            Movie: {doc.get("title", "")} - {doc.get("document", "")}

            Consider:
            - Direct relevance to query
            - User intent (what they're looking for)
            - Content appropriateness

            Rate 0-10 (10 = perfect match).
            Give me ONLY the number in your response, no other text or explanation.

            Score:"""

            try:
                # Call the LLM
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    config = types.GenerateContentConfig(
                        system_instruction = system_instruction
                    ),
                    contents = query
                )

                # Extract and parse the score
                score_text = response.text.strip()
                try:
                    score = float(score_text)

                except ValueError:
                    print(f"Warning: Could not parse LLM score for {doc['title']}. Received: '{score_text}'. Defaulting to 0.0.")
                    score = 0.0
            except Exception as e:
                print(f"Error calling LLM for {doc['title']}: {e}. Defaulting to 0.0.")
                score = 0.0 # Default score on API failure
                
                doc['rerank_score'] = score
                
                # Sleep for 3 seconds to avoid rate limits
                time.sleep(3)

        return docs

    
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

def rrf_search_command(query: str, k: int = 60, limit:int = DEFAULT_SEARCH_LIMIT, enhance: str = None, rerank_method: str=None):
    # 1. load movies
    movies = load_movies()
    # 2. create hybrid search
    hybrid_search = HybridSearch(movies)
    # 3. call rrf
    if enhance == None:
        results = hybrid_search.rrf_search(query, k, limit)
        # 4. return results in a dictionary
        return {
            "original_query": query,
            'k': k,
            'method': None,
            'enhanced_query': None,
            "results": results

        }
    
    elif enhance == "expand":
        system_instruction = f"""Expand this movie search query with related terms.

            Add synonyms and related concepts that might appear in movie descriptions.
            Keep expansions relevant and focused.
            This will be appended to the original query.

            Examples:

            - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
            - "action movie with bear" -> "action thriller bear chase fight adventure"
            - "comedy with bear" -> "comedy funny bear humor lighthearted"
            - "math movie" -> "mathematics science genius smart intelligence algebra physics"

            Query: "{query}"
            """

        response = client.models.generate_content(
            model='gemini-2.0-flash-001',
            config = types.GenerateContentConfig(
                system_instruction = system_instruction
            ),
            contents = query)
        enhanced_query = response.text.strip()
        method = "expand"
        results = hybrid_search.rrf_search(enhanced_query, k, limit)
        return {
            "original_query": query,
            'k': k,
            'method': method,
            'enhanced_query': enhanced_query,
            "results": results
        }


    elif enhance == "rewrite":
        system_instruction = f"""Rewrite this movie search query to be more specific and searchable.

        Original: "{query}"

        Consider:
        - Common movie knowledge (famous actors, popular films)
        - Genre conventions (horror = scary, animation = cartoon)
        - Keep it concise (under 10 words)
        - It should be a google style search query that's very specific
        - Don't use boolean logic

        Examples:

        - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
        - "movie about bear in london with marmalade" -> "Paddington London marmalade"
        - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

        Rewritten query:"""

        response = client.models.generate_content(
            model='gemini-2.0-flash-001',
            config = types.GenerateContentConfig(
                system_instruction = system_instruction
            ),
            contents = query)
        enhanced_query = response.text.strip()
        method = "rewrite"
        results = hybrid_search.rrf_search(enhanced_query, k, limit)
        return {
            "original_query": query,
            'k': k,
            'method': method,
            'enhanced_query': enhanced_query,
            "results": results
        }

    elif enhance == "spell":
        system_instruction = f"""Fix any spelling errors in this movie search query.

            Only correct obvious typos. Don't change correctly spelled words.

            Query: "{query}"

            If no errors, return the original query.
            Corrected:"""
        response = client.models.generate_content(
            model='gemini-2.0-flash-001',
            config = types.GenerateContentConfig(
                system_instruction = system_instruction
            ),
            contents = query)
        enhanced_query = response.text.strip()
        method = "spell"
        results = hybrid_search.rrf_search(enhanced_query, k, limit)
        return {
            "original_query": query,
            'k': k,
            'method': method,
            'enhanced_query': enhanced_query,
            "results": results
        }     

        
