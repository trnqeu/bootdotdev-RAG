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
    pass