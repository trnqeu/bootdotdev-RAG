import argparse
from lib.search_utils import load_movies
from lib.augmented_generation import rag_command, summarize_command, citations_command, question_command


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    sum_parser = subparsers.add_parser("summarize", help="Summarize multiple search results")
    sum_parser.add_argument("query", type=str, help="Search query for summarization")
    sum_parser.add_argument("--limit", type=int, default=5, help="Number of results to summarize")

    # citation parser
    citation_parser = subparsers.add_parser("citations", help="Answer with citations")
    citation_parser.add_argument("query", type=str, help="Search query for citation")
    citation_parser.add_argument("--limit", type=int, default=5, help="Number of results to cite")

    # question parser
    question_parser = subparsers.add_parser("question", help="answer questions about movies")
    question_parser.add_argument("question", type=str, help="Type a question for the LLM")
    question_parser.add_argument("--limit", type=int, default=5, help="Number of results to cite")
    
    args = parser.parse_args()

    match args.command:
        case "rag":
            docs, answer = rag_command(args.query)
            
            print("Search Results:")
            for doc in docs:
                print(f" - {doc['title']}")

            print("\nRAG Response:")
            print(answer)

        case "summarize":
            docs, summary = summarize_command(args.query, args.limit)

            print("Search Results:")

            for doc in docs:
                print(f"  - {doc['title']}")
            
            print("\nLLM Summary:")
            print(summary)

        case "citations":
            docs, answer = citations_command(args.query, args.limit)

            print("Search Results:")
            for doc in docs:
                print(f"  - {doc['title']}")
            
            print("\nLLM Answer:")
            print(answer)

        case "question":
            docs, answer = question_command(args.question, args.limit)

            print("Search Results:")
            for doc in docs:
                print(f"  - {doc['title']}")
            
            print("\nAnswer:")
            print(answer)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()