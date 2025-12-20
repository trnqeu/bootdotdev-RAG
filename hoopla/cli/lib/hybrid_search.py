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
import json
import time
from sentence_transformers import CrossEncoder


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
        if rerank_method:
            fetch_limit = limit * 5
        else:
            fetch_limit = limit
        
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
        
        # Truncate to the appropriate limit for the re-rank phase
        rrf_results = sorted_docs[:fetch_limit]

        print(f"DEBUG: Results after RRF search: {[d['title'] for d in rrf_results[:5]]}")

        # If reranking is requested, rerank and re-sort
        if rerank_method == 'individual':
            reranked_docs = self._rerank_individual(query, rrf_results)

            final_docs = sorted(
                reranked_docs,
                key = lambda d: d.get('rerank_score', 0.0),
                reverse = True,
            )

            print(f"DEBUG: Final results after re-ranking: {[d['title'] for d in reranked_docs[:5]]}")
            return final_docs[:limit]
        
        elif rerank_method == 'batch':
            reranked_docs = self._rerank_batch(query, rrf_results)
            return reranked_docs[:limit]
        
        elif rerank_method == 'cross_encoder':
            reranked_docs = self._rerank_cross_encoder(query, rrf_results)
            return reranked_docs[:limit]
        
        return rrf_results[:limit]
    
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
                    model='gemini-2.0-flash-001',
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

    def _rerank_batch(self, query: str, docs: list[dict]) -> list[dict]:
        print(f"Reranking top {len(docs)} results using batch method...")

        doc_list_str = "\n"
        for i, doc in enumerate(docs, start=1):
            doc_list_str += f"ID: {doc['id']}\nTitle: {doc.get('title')}\nSummary: {doc.get('document', '')}\n---\n"

        system_instruction = f"""Rank these movies by relevance to the search query.

        Query: "{query}"

        Movies:
        {doc_list_str}

        Return ONLY the IDs in order of relevance (best match first). 
        Return a valid JSON list, nothing else. 
        For example:

        [75, 12, 34, 2, 1]

        """
        try:
            response = client.models.generate_content(
                model = 'gemini-2.0-flash-001',
                config = types.GenerateContentConfig(
                    system_instruction = system_instruction
                ),
                contents = query
            )


            raw_response = response.text.strip()

            if raw_response.startswith('```'):
                # Split lines, remove the first (```json) and the last (```) line
                lines = raw_response.split('\n')
                # Join the lines back together, skipping the first and last
                json_string = '\n'.join(lines[1:-1]).strip()
            else:
                # Assume it's a plain JSON string if it doesn't start with fences
                json_string = raw_response
            
            # 2. Check if the cleaned string is empty or invalid (optional, but robust)
            if not json_string.startswith('['):
                 raise ValueError(f"Cleaned response is not a JSON list: {json_string}")
                
            rank_id_list = json.loads(json_string) # Load the cleaned string

        except Exception as e:
            print(f"Error calling LLM for batch reranking or parsing JSON: {e}")
            print("Defaulting to original RRF order.")
            rank_id_list = [doc['id'] for doc in docs]

        # Assign a new rerank to each document
        # map: doc_id -> new_rank
        id_to_rank = {doc_id: rank + 1 for rank, doc_id in enumerate(rank_id_list)}

        for doc in docs:
            doc['rerank_rank'] = id_to_rank.get(doc['id'], len(docs) + 1)

        # sort the list by the new rerank_rank
        reranked_docs = sorted(
            docs,
            key=lambda d: d.get('rerank_rank', len(docs) + 1),
            reverse = False
        )

        return reranked_docs

    def _rerank_cross_encoder(self, query: str, docs: list[dict]) -> list[dict]:
        print(f"Reranking top {len(docs)} results using cross_encoder method...")
        cross_encoder = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L2-v2')
        pairs = []
        for doc in docs:
            doc_string = f"{doc.get('title', '')} - {doc.get("document", '')}"
            pairs.append([query, doc_string])

        scores = cross_encoder.predict(pairs)

        for doc, score in zip(docs, scores):
            doc['cross_encoder_score'] = score

        reranked_docs = sorted(
            docs,
            key=lambda d: d.get('cross_encoder_score', -100.0),
            reverse = True
        )

        return reranked_docs


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

def rrf_search_command(query: str, 
                       k: int = 60, 
                       limit:int = DEFAULT_SEARCH_LIMIT, 
                       enhance: str = None, 
                       rerank_method: str=None,
                       evaluate: bool=False):
    # 1. load movies
    movies = load_movies()
    # 2. create hybrid search
    hybrid_search = HybridSearch(movies)

    print(f"DEBUG: Original Query: {query}")

    final_query = query
    method = None
    enhanced_query = None
    # 3. call rrf

    if enhance == None:
        print(f"DEBUG: Enhanced Query: {final_query}")
        results = hybrid_search.rrf_search(query, k, limit, rerank_method = rerank_method)
        # 4. return results in a dictionary
        # return {
        #     "original_query": query,
        #     'k': k,
        #     'method': None,
        #     'enhanced_query': None,
        #     "results": results,
        #     "rerank_method": rerank_method

        # }
    
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
        final_query = enhanced_query
        method = "expand"
        results = hybrid_search.rrf_search(final_query, k, limit)
        # return {
        #     "original_query": query,
        #     'k': k,
        #     'method': method,
        #     'enhanced_query': enhanced_query,
        #     "results": results
        # }


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
        final_query = enhanced_query
        method = "rewrite"
        results = hybrid_search.rrf_search(final_query, k, limit)
        # return {
        #     "original_query": query,
        #     'k': k,
        #     'method': method,
        #     'enhanced_query': enhanced_query,
        #     "results": results
        # }

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
        final_query = enhanced_query
        method = "spell"
        results = hybrid_search.rrf_search(final_query, k, limit)

    if evaluate:
        # Prepariamo la lista dei risultati formattata
        formatted_results = [f"Title: {doc['title']}\nSummary: {doc['document']}\n---" for doc in results]
        
        prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers out than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""

        try:
            response = client.models.generate_content(
                model='gemini-2.0-flash-001',
                contents=prompt
            )
            
            raw_response = response.text.strip()
            # Pulizia JSON (Markdown fences)
            if raw_response.startswith('```'):
                lines = raw_response.split('\n')
                json_string = '\n'.join(lines[1:-1]).strip()
            else:
                json_string = raw_response
            
            scores = json.loads(json_string)

            # Stampa il report finale richiesto
            print("\nEvaluation Report:")
            for i, (doc, score) in enumerate(zip(results, scores), 1):
                print(f"{i}. {doc['title']}: {score}/3")
                
        except Exception as e:
            print(f"Error during LLM evaluation: {e}")

    return {
    "original_query": query,
    'k': k,
    'method': method,
    'enhanced_query': enhanced_query,
    "results": results,
    "rerank_method": rerank_method # Fondamentale aggiungerlo qui!
}     

        
