#!/usr/bin/env python3
import argparse
from lib.semantic_search import (
    SemanticSearch,
    ChunkedSemanticSearch,
    verify_model, 
    embed_text, 
    verify_embeddings, 
    embed_query_text,
    cosine_similarity,
    create_chunks,
    create_semantic_chunks
)

from lib.search_utils import load_movies
def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest='command', help='available commands')
    verify_parser = subparsers.add_parser("verify", help="Verify that the embedding model is loaded")
    embeddings_parser = subparsers.add_parser("verify_embeddings", help="Load or create embeddings and print details")
    embedding_query_parser = subparsers.add_parser("embedquery", help = "Insert your query")
    embedding_query_parser.add_argument("query", type=str, help = "query to embed")
    text_embed_parser = subparsers.add_parser("embed_text", help='text to be embedded')
    text_embed_parser.add_argument("text", type=str, help='text to be embedded')
    search_parser = subparsers.add_parser("search", help="performs semantic search")
    search_parser.add_argument("query", type=str, help='text to search')
    search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")
    chunk_parser = subparsers.add_parser("chunk", help="chunks the text")
    chunk_parser.add_argument("text", type=str, help = "text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, default=200, help = "size of the chunk")
    chunk_parser.add_argument("--overlap", type=int, default=20, help = "chunk overlapping size")
    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="chunks the text")
    semantic_chunk_parser.add_argument("text", type=str, help = "text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help = "size of the chunk")
    semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help = "chunk overlapping size")
    embed_chunk_parser = subparsers.add_parser('embed_chunks', help="Load or create chunk embeddings")
    

    args = parser.parse_args()

    match args.command:
        case 'verify':
            verify_model()

        case 'embed_text':
            embed_text(args.text)

        case 'verify_embeddings':
            verify_embeddings()
        
        case 'embedquery':
            embed_query_text(args.query)

        case 'chunk':
            create_chunks(args.text, args.chunk_size, args.overlap)

        case 'semantic_chunk':
            create_semantic_chunks(args.text, args.max_chunk_size, args.overlap)

        case 'search':
            searcher = SemanticSearch()
            movies = load_movies()
            searcher.load_or_create_embeddings(movies)
            results = searcher.search(args.query, limit=args.limit)
            for i, result in enumerate(results):
                print(f"{i+1}. {result['title']} (score: {result['score']:.4f})")
                print(f"    {result['description']}")
                print()

        case 'embed_chunks':
            movies = load_movies()
            chunk_searcher = ChunkedSemanticSearch()
            embeddings = chunk_searcher.load_or_create_chunk_embeddings(movies)
            print(f"Generated {len(embeddings)} chunked embeddings")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()