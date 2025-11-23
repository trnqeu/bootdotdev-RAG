import argparse
from lib.search_utils import normalize_scores
from lib.hybrid_search import HybridSearch, weighted_search_command, rrf_search_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    # parser.add_subparsers(dest="command", help="Available commands")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    normalize_parser = subparsers.add_parser("normalize", help = "normalize the scores")
    normalize_parser.add_argument("scores",
                                  nargs = "+",
                                  type=float,
                                  help="list of scores to be normalized")
    
    weighted_parser = subparsers.add_parser(
        "weighted-search",
        help="Perform weighted hybrid search",
    )
    weighted_parser.add_argument("query", type=str, help="Search query")
    weighted_parser.add_argument("--alpha", type=float, default=0.5)
    weighted_parser.add_argument("--limit", type=int, default=5)

    rff_parser = subparsers.add_parser(
        "rrf-search",
        help = "performs reciprocal rank fusion"
    )

    rff_parser.add_argument("query", type=str, help="Search query")
    rff_parser.add_argument("--k", type=int, default=60)
    rff_parser.add_argument("--limit", type=int, default=5)
    rff_parser.add_argument("--enhance",
                            type=str,
                            choices=["spell"],
                            help="Query enhancement method")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized_scores = normalize_scores(args.scores)
            for n in normalized_scores:
                print(f"* {n:.4f}")

        case "weighted-search":
            result = weighted_search_command(args.query, args.alpha, args.limit)
            for i, doc in enumerate(result["results"], start=1):
                print(f"{i}. {doc['title']}")
                print(f"   Hybrid Score: {doc['hybrid']:.3f}")
                print(f"   BM25: {doc['bm25']:.3f}, Semantic: {doc['semantic']:.3f}")
                print(f"   {doc['document'][:100]}...")
                print()

        case "rrf-search":
            result = rrf_search_command(args.query, args.k, args.limit, args.enhance)
            if result['enhanced_query']:
                print( f"Enhanced query ({result['method']}): '{result['original_query']}' -> '{result['enhanced_query']}'\n")
                for i, doc in enumerate(result["results"], start=1):
                    print(f"{i}. {doc['title']}")
                    print(f"   RRF Score: {doc['rrf_score']:.3f}")
                    print(f"   BM25 Rank: {doc['bm25_rank']}, Semantic rank: {doc['semantic_rank']}")
                    print(f"   {doc['document'][:100]}...")
                    print()
            else:
                for i, doc in enumerate(result["results"], start=1):
                    print(f"{i}. {doc['title']}")
                    print(f"   RRF Score: {doc['rrf_score']:.3f}")
                    print(f"   BM25 Rank: {doc['bm25_rank']}, Semantic rank: {doc['semantic_rank']}")
                    print(f"   {doc['document'][:100]}...")
                    print()

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()