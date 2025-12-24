import os
from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .search_utils import (normalize_scores, 
                           hybrid_score,
                           load_movies, 
                           DEFAULT_SEARCH_LIMIT)
from dotenv import load_dotenv
from google import genai
from google.genai import types
import json
import time
from sentence_transformers import CrossEncoder
from .hybrid_search import HybridSearch, weighted_search_command, rrf_search_command


load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
print(f"Using key {api_key[:6]}...")

client = genai.Client(api_key=api_key)

def rag_command(query):
    movies = load_movies()
    search = HybridSearch(movies)
    docs = search.rrf_search(query, 60, 5)
    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

    Query: {query}

    Documents:
    {docs}

    Provide a comprehensive answer that addresses the query:"""

    response = client.models.generate_content(
            model='gemini-2.0-flash-001',
            contents = prompt)
    
    return docs, response