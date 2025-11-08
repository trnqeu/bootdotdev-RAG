from sentence_transformers import SentenceTransformer
import numpy as np
from pathlib import Path
import os
import re
import json

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


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name='all-MiniLM-L6-v2') -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents, cache_path='cache/chunk_embeddings.npy', metadata_path='cache/chunk_metadata.json'):
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc['id']] = doc
        all_chunks = []
        self.chunk_metadata = []
        for doc_idx, doc in enumerate(self.documents):
                
                description = doc.get('description', '').strip() 

                if not description: # Skips truly empty or whitespace-only descriptions
                    continue

                # Pass the clean, stripped description to the chunker
                description_chunks = create_semantic_chunks(
                    description, 
                    max_chunk_size = 4,
                    overlap = 1
                    )
                total_chunks = len(description_chunks)

                for chunk_idx, chunk_text in enumerate(description_chunks):
                    all_chunks.append(chunk_text)

                    metadata = {
                        "movie_idx": doc_idx,      # not doc['id']
                        "chunk_idx": chunk_idx,
                        "total_chunks": total_chunks,
}
                    self.chunk_metadata.append(metadata)
        print(f"Generating embeddings for {len(all_chunks)} chunks...")
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)

        # check if directory exists
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        # save the embeddings
        np.save(cache_path, self.chunk_embeddings)
        print(f"Chunk embeddings saved to {cache_path}")

        metadata_dict = {
            "chunks": self.chunk_metadata,
            "total_chunks": len(all_chunks)
        }

        # check path
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

        with open(metadata_path, 'w') as f:
            json.dump(metadata_dict, f, indent=2)

        print(f"Chunk metadata saved to {metadata_path}")

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc['id']] = doc
        # Define the paths for the chunk files
        embeddings_path = 'cache/chunk_embeddings.npy'
        metadata_path = 'cache/chunk_metadata.json'

        # Create Path objects
        embeddings_file = Path(embeddings_path)
        metadata_file = Path(metadata_path)

        if embeddings_file.is_file() and metadata_file.is_file():
            print(f"Chunk cache found! Loading embeddings and metadata...")
            
            # Load embeddings into self.chunk_embeddings
            self.chunk_embeddings = np.load(embeddings_file)
            
            # Load metadata JSON and extract chunks into self.chunk_metadata
            with open(metadata_file, 'r') as f:
                metadata_content = json.load(f)
                # The 'chunks' key holds the list of metadata dictionaries
                self.chunk_metadata = metadata_content['chunks']

            # Check for consistency (important!)
            if len(self.chunk_embeddings) == len(self.chunk_metadata):
                print(f"Loaded {len(self.chunk_embeddings)} chunk embeddings from cache.")
                # --- FIX APPLIED HERE ---
                return self.chunk_embeddings 
            else:
                print("Cache mismatch (embeddings/metadata length). Forcing rebuild...")

        # If cache is not found or mismatch occurred, build and return the result
        print("Cache not found or corrupted. Building and caching chunk embeddings...")
        return self.build_chunk_embeddings(
            documents, 
            cache_path=embeddings_path, 
            metadata_path=metadata_path
        )


def create_chunks(text:str, chunk_size:int, overlap:int) -> list[str]:
    # Placeholder/example logic for word-based chunking
    words = text.split(" ") 
    chunks = []
    start_index = 0
    step = chunk_size - overlap
    if step < 1:
        step = 1
    while start_index < len(words):
        end_index = start_index + chunk_size
        chunk_words = words[start_index: end_index]
        chunk = " ".join(chunk_words)
        chunks.append(chunk)
   
        # Check if we've processed all words or if the last chunk is small enough
        if end_index >= len(words):
            break
        
        start_index += step

    total_chars = len(text)
    print(f"Chunking {total_chars} characters (Chunk Size: {chunk_size} words)")
    
    # Print each chunk in the numbered format
    for i, chunk in enumerate(chunks):
        print(f"{i+1}. {chunk}")
    return chunks

def create_semantic_chunks(text:str, max_chunk_size:int, overlap:int) -> list[str]:
    # 1. Split the input into individual sentences by using the nasty regex.
    # The regex r"(?<=[.!?])\s+" splits by space immediately following a period,
    # question mark, or exclamation point, keeping the punctuation with the sentence.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    # The re.split might leave an empty string at the end if the text ends with the delimiter pattern
    sentences = [s for s in sentences if s] 

    chunks = []
    start_index = 0
    step = max_chunk_size - overlap
    # Ensure step is at least 1 to avoid an infinite loop or redundant step if overlap >= max_chunk_size
    step = max(1, step)
    
    # 2. Chunk sentences with max_chunk_size and overlap
    while start_index < len(sentences):
        end_index = start_index + max_chunk_size
        chunk_sentences = sentences[start_index: end_index]
        # Join sentences back into a chunk string. We add a space after the period/punctuation
        # because the re.split removes the trailing space from the sentence.
        chunk = " ".join(chunk_sentences)
        chunks.append(chunk)
        
        # Move the start index for the next chunk
        start_index += step
        
        # Stop if the next start_index would result in a chunk with fewer sentences than the overlap,
        # or if we've passed the end of the list. The last chunk should already be added.
        if start_index >= len(sentences):
            break

    
    return chunks

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