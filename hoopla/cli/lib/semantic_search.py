from sentence_transformers import SentenceTransformer
import numpy as np
import os


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

# text = 'pippo beveva la piooggia'

# print(searcher.generate_embedding(text))