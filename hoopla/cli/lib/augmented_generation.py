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

def summarize_command(query, limit=5):
    movies = load_movies()
    search = HybridSearch(movies)
    docs = search.rrf_search(query, 60, limit)
    results_text = ""
    for doc in docs:
        results_text += f"\n- Title: {doc['title']}\n  Genre: {doc.get('genres', 'N/A')}\n  Plot: {doc.get('description', '')}\n"

    prompt = f"""
        Provide information useful to this query by synthesizing information from multiple search results in detail.
        The goal is to provide comprehensive information so that users know what their options are.
        Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
        This should be tailored to Hoopla users. Hoopla is a movie streaming service.
        Query: {query}
        Search Results:{results_text}
        Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:
        """
    
    response = client.models.generate_content(
            model='gemini-2.0-flash-001',
            contents = [prompt])
    
    return docs, response.text
    
def rag_command(query):
    movies = load_movies()
    search = HybridSearch(movies)
    docs = search.rrf_search(query, 60, 5)

    context_text = "\n".join([f"- {doc['title']}: {doc.get('description', '')}" for doc in docs])

    prompt = f"""Answer the question or provide information based on the provided documents. 
    This should be tailored to Hoopla users. Hoopla is a movie streaming service.

    Query: {query}

    Documents:
    {context_text}

    Provide a comprehensive answer that addresses the query:"""

    response = client.models.generate_content(
            model='gemini-2.0-flash-001',
            contents = prompt)
    
    return docs, response.text


def citations_command(query, limit=5):
    movies = load_movies()
    search = HybridSearch(movies)
    
    # Perform search
    docs = search.rrf_search(query, 60, limit)

    # Format documents with indices [1], [2]
    formatted_docs = ""

    for i, doc in enumerate(docs, 1):
        text_content = doc.get('document', 'No document available.')
        formatted_docs += f"\n[{i}] Title: {doc['title']}\n Overview: {text_content}\n"
    
    prompt = f"""Answer the question or provide information based on the provided documents.

    This should be tailored to Hoopla users. Hoopla is a movie streaming service.

    If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

    Query: {query}

    Documents:{formatted_docs}

    Instructions:
    - Provide a comprehensive answer that addresses the query
    - Cite sources using [1], [2], etc. format when referencing information
    - If sources disagree, mention the different viewpoints
    - If the answer isn't in the documents, say "I don't have enough information"
    - Be direct and informative

    Answer:"""

    response = client.models.generate_content(
            model='gemini-2.0-flash-001',
            contents = prompt)
    
    return docs, response.text

def question_command(question, limit=5):
    movies = load_movies()
    search = HybridSearch(movies)

    # Perform search
    docs = search.rrf_search(question, 60, limit)

    context_text = "\n".join([f"- {doc['title']}: {doc.get('document', '')}" for doc in docs])

    prompt = f"""Answer the user's question based on the provided movies that are available on Hoopla.

    This should be tailored to Hoopla users. Hoopla is a movie streaming service.

    Question: {question}

    Documents:
    {context_text}

    Instructions:
    - Answer questions directly and concisely
    - Be casual and conversational
    - Don't be cringe or hype-y
    - Talk like a normal person would in a chat conversation

    Answer:"""

    response = client.models.generate_content(
            model='gemini-2.0-flash-001',
            contents = prompt)
    
    return docs, response.text