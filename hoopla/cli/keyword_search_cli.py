#!/usr/bin/env python3

import argparse
import json
import string
import os
import pickle
from nltk.stem import PorterStemmer

with open('./data/movies.json', 'r') as f:
    movies_dict = json.load(f)

movies_list = movies_dict['movies']

PUNCTUATION_REMOVER = str.maketrans("", "", string.punctuation)

# create an instance of the Stemmer
stemmer = PorterStemmer()

with open('data/stopwords.txt', 'r') as f:
    text = f.read()

STOPWORDS_SET = set(text.splitlines())

class InvertedIndex:
    def __init__(self):
        # index: token (string) -> set of document IDs (integers)
        self.index: dict[str, set[int]] = {}
        # docmap: document ID (integer) -> document object (dict)
        self.docmap: dict[int, object] = {}
    
    def __add_document_(self, doc_id: int, text: str):
        '''
        Tokenize the input text, 
        then add each token to the index 
        with the document ID.
        '''
        # use the existing tokenizer
        tokens = tokenizer(text, STOPWORDS_SET)

        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_documents(self, term: str) -> list[int]:
        '''
        Get the set of documents for a given token, 
        and return them as a list, 
        sorted in ascending order by document ID.
        Process the input term (lowercase, stem) to match index keys 
        '''
        processed_term = stemmer.stem(term.lower())

        # Get the set of document IDs, or an empty set if the term isn't found
        doc_ids_set = self.index.get(processed_term, set())

        # Convert the set to a list and sort it
        return sorted(list(doc_ids_set))


    def build(self, movies_list: list[dict]):
        print("Builing inverted index...")
        for movie in movies_list:
            doc_id = movie['id']
            # Concatenate title and description for indexing
            text_to_index = f"{movie['title']} {movie['description']}"

            # Add to the index
            self.__add_document_(doc_id, text_to_index) 

            # Add the full document object to the docmap
            self.docmap[doc_id] = movie

        print("Index build complete.")

    def save(self):
        cache_dir = "cache"
        os.makedirs(cache_dir, exist_ok=True) # create the directory if doesn't exist

        index_path = os.path.join(cache_dir, "index.pkl")
        docmap_path = os.path.join(cache_dir, "docmap.pkl")

        # Save the index
        with open(index_path, 'wb') as f:
            pickle.dump(self.index, f)

        # Save the docmap
        with open(docmap_path, 'wb') as f:
            pickle.dump(self.docmap, f)
        print(f"Document map saved to {docmap_path}")

    def load(self):
        cache_dir = "cache"

        index_path = os.path.join(cache_dir, "index.pkl")
        docmap_path = os.path.join(cache_dir, "docmap.pkl")

        # Load the index
        try:
            with open(index_path, 'rb') as f:
                self.index = pickle.load(f)
            print('index loaded')
        except Exception as e:
            raise FileNotFoundError(f"Error loading index file: {e}")


        # Load the docmap
        try:
            with open(docmap_path, 'rb') as f:
                self.docmap = pickle.load(f)
            print('docmap loaded')
        except Exception as e:
            raise FileNotFoundError(f"Error loading the docmap: {e}")
             
    

def remove_punctuation(text: str) -> str:
    # Remove punctuation from a string
    return text.translate(PUNCTUATION_REMOVER)

def tokenizer(text: str, stopwords: set[str]) -> set[str]:
    # word tokenizer
    processed_text = remove_punctuation(text).lower()
    
    # split by whitespace to get tokens
    tokens = {stemmer.stem(token) for token in processed_text.split() if token}


    # use set difference to remove stopwords
    return tokens - stopwords



def main() -> None:
    # create InvertedIndex instance
    index = InvertedIndex()
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    # build command parser
    subparsers.add_parser("build", help="Build and save inverted index")
    args = parser.parse_args()

    results = []

    match args.command:
        case "search":
            query = args.query
            # processed_query = remove_punctuation(query).lower()
            
            # print the search query here
            print(f"Searching for: {query}")

            try: 
                index.load()
            except FileNotFoundError as e:
                print(f"\n{e}")

            query_tokens = tokenizer(query, STOPWORDS_SET)
            if not query_tokens:
                print("Invalid search")
                return
            # search using the index
            
            # crea set vuoto degli id univoci
            results_ids = set()

            # variabile massimo numero di risultati = 5
            MAX_RESULTS = 5

            # itera su ogni token della query, trova gli id dei documenti, 
            for token in query_tokens:
                # use the index to get doc ids for this token
                doc_ids = index.get_documents(token)

                for doc_id in doc_ids:
                    if doc_id not in results_ids:
                        results_ids.add(doc_id)

                        if len(results_ids) >= MAX_RESULTS:
                            break

                if len(results_ids) >= MAX_RESULTS:
                    break

            print(f"\nFound {len(results_ids)} unique results:")

            sorted_doc_ids = sorted(list(results_ids))

            for index_num, doc_id in enumerate(results_ids):
                # use the docmap to retrieve titles
                movie = index.docmap.get(doc_id)
                if movie:
                    print(f"{index_num + 1}: ID: {doc_id} - Title: {movie['title']}")
               

            # for movie in movies_list: 
            #     # remove punctuation from title
            #     # processed_title = remove_punctuation(movie['title']).lower()              
            #     title_tokens = tokenizer(movie['title'], STOPWORDS_SET)
                
            #     # set.intersection() returns a new set containing elements common to both sets.
            #     # If the resulting set is non-empty, there is at least one matching token.
            #     # matching_tokens = [query_tokens.intersection(title_tokens) ]
            #     # new matching logic to match substrings, like 'shot' in 'killshot'
            #     is_match = False

            #     for q_token in query_tokens:
            #         for t_token in title_tokens:
            #             if q_token in t_token:
            #                 is_match = True
            #                 break
            #         if is_match:
            #             break

            #     if is_match:
            #         results.append(movie)
          

            # results.sort(key= lambda movie: movie['id'], reverse=False)

            # final_results = results[:5]

            # if final_results:
            #     for index, movie in enumerate(final_results):
            #         print(f"{index+1}. {movie['title']}")
        
        case "build":
            # build the index
            index.build(movies_list)

            # save the index
            index.save()



        
        case _:
            parser.print_help()

    


if __name__ == "__main__":
    main()