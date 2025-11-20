import argparse
from lib.search_utils import normalize_scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    # parser.add_subparsers(dest="command", help="Available commands")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    normalize_parser = subparsers.add_parser("normalize", help = "normalize the scores")
    normalize_parser.add_argument("scores",
                                  nargs = "+",
                                  type=float,
                                  help="list of scores to be normalized")
    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized_scores = normalize_scores(args.scores)
            for n in normalized_scores:
                print(f"* {n:.4f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()