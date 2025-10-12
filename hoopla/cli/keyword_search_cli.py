#!/usr/bin/env python3

import argparse
import json
import string

with open('./data/movies.json', 'r') as f:
    movies_dict = json.load(f)

movies_list = movies_dict['movies']

PUNCTUATION_REMOVER = str.maketrans("", "", string.punctuation)

with open('data/stopwords.txt', 'r') as f:
    text = f.read()

STOPWORDS_SET = set(text.splitlines())

def remove_punctuation(text: str) -> str:
    # Remove punctuation from a string
    return text.translate(PUNCTUATION_REMOVER)

def tokenizer(text: str, stopwords: set[str]) -> set[str]:
    # word tokenizer
    processed_text = remove_punctuation(text).lower()
    
    # split by whitespace to get tokens
    tokens = {token for token in processed_text.split() if token}

    # use set difference to remove stopwords
    return tokens - stopwords



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
            # processed_query = remove_punctuation(query).lower()
            query_tokens = tokenizer(query, STOPWORDS_SET)
            
            
            # print the search query here
            print(f"Searching for: {query}")

            if not query_tokens:
                print("Invalid search")
                return

            for movie in movies_list: 
                # remove punctuation from title
                # processed_title = remove_punctuation(movie['title']).lower()              
                title_tokens = tokenizer(movie['title'], STOPWORDS_SET)
                
                # set.intersection() returns a new set containing elements common to both sets.
                # If the resulting set is non-empty, there is at least one matching token.
                # matching_tokens = [query_tokens.intersection(title_tokens) ]
                # new matching logic to match substrings, like 'shot' in 'killshot'
                is_match = False

                for q_token in query_tokens:
                    for t_token in title_tokens:
                        if q_token in t_token:
                            is_match = True
                            break
                    if is_match:
                        break

                if is_match:
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