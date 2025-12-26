import argparse
from lib.search_utils import load_movies
from lib.hybrid_search import HybridSearch, weighted_search_command, rrf_search_command
from lib.augmented_generation import rag_command


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
            docs, answer = rag_command(args.query)
            
            print("Search Results:")
            for doc in docs:
                print(f" - {doc['title']}")

            print("\nRAG Response:")
            print(answer)
            
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()