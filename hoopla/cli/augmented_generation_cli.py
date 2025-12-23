import argparse
from lib.search_utils import load_movies
from lib.hybrid_search import HybridSearch, weighted_search_command, rrf_search_command



def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            # do RAG stuff here
            movies = load_movies()
            result = rrf_search_command(
                args.query, 
                args.k, 
                args.limit, 
                args.enhance, 
                args.rerank_method,
                evaluate = args.evaluate
                )
            
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()