#!/usr/bin/env python3

import argparse
import json

with open('./data/movies.json', 'r') as f:
    movies_dict = json.load(f)

movies_list = movies_dict['movies']



def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    results = []

    match args.command:
        case "search":
            query = args.query
            # print the search query here
            print(f"Searching for: {query}")

            for movie in movies_list:                
                if query.lower() in movie['title'].lower():
                    results.append(movie)
              

            results.sort(key= lambda movie: movie['id'], reverse=False)

            final_results = results[:5]

            if final_results:
                for index, movie in enumerate(final_results):
                    print(f"{index+1}. {movie['title']}")
        case _:
            parser.print_help()

    


if __name__ == "__main__":
    main()