from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import os

from .search_utils import (
    load_movies
)

class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if not text or text.isspace():
            raise ValueError('Invalid values')
        else:
            return self.model.encode([text])[0]
        
    def build_embeddings(self, documents, cache_path='cache/movie_embeddings.npy'):
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc['id']] = doc
        docs_list = [f"{doc['title']}: {doc['description']}" for doc in self.documents]
        self.embeddings = self.model.encode(docs_list, show_progress_bar=True)
        np.save('cache/movie_embeddings.npy', self.embeddings)
        # 4. Save to disk (Ensure directory exists)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.save(cache_path, self.embeddings)
        print(f"Embeddings saved to {cache_path}")
        return self.embeddings
    
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc['id']] = doc
        embeddings_path = 'cache/movie_embeddings.npy'
        path = Path(embeddings_path)

        if path.is_file():
            self.embeddings = np.load(path)
            if len(self.embeddings) == len(self.documents):
                return self.embeddings
            else:
                raise ValueError('len(self.embeddings) != len(self.documents)')
        else:
            return self.build_embeddings(documents)

    def search(self, query, limit=5):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        else:
            embedded_query = self.generate_embedding(query)
            scores = []
            for i,emb in enumerate(self.embeddings):
                score = cosine_similarity(embedded_query, emb)
                scores.append((self.documents[i]['id'], score))

            sorted_scores = sorted(scores, key=lambda tup: tup[1], reverse = True)
            sorted_scores = sorted_scores[:limit]

        results = []
        for doc_id, score in sorted_scores:
            doc = self.document_map[doc_id]
            results.append({
                "score": score,
                "title": doc['title'],
                "description": doc['description']
            })
        return results

def create_chunks(self, text:str, chunk_size:int =200) -> list[str]:
    words = text.split(" ")
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i: i + chunk_size]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)
    total_chars = len(text)
    print(f"Chunking {total_chars} characters (Chunk Size: {chunk_size} words)")
    
    # 3. Print each chunk in the numbered format
    for i, chunk in enumerate(chunks):
        print(f"{i+1}. {chunk}")

def verify_model():
    search = SemanticSearch()
    print(f"Model loaded: {search.model}")
    print(f"Max sequence length: {search.model.max_seq_length}")

def embed_text(text):
    # creates an instance of the class
    searcher = SemanticSearch()
    # get the embedding
    embedding = searcher.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def embed_query_text(query):
    search = SemanticSearch()
    embedded_query = search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedded_query[:5]}")
    print(f"Shape: {embedded_query.shape}")

def verify_embeddings():
    search = SemanticSearch()
    documents = load_movies()
    search.load_or_create_embeddings(documents)
    embeddings_shape = search.embeddings.shape
    print(f"Loaded {len(documents)} movie documents.")
    print(f"{embeddings_shape[0]} vectors in {embeddings_shape[1]} dimensions")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

