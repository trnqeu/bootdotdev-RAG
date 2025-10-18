#!/usr/bin/env python3
import argparse
import json
import string
import os
import pickle
import math
from collections import Counter
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
        self.term_frequencies: dict[int, Counter] = {}
        self.length: int

        
    def __add_document_(self, doc_id: int, text: str):
        '''
        Tokenize the input text, 
        then add each token to the index 
        with the document ID.
        '''
        # use the existing tokenizer
        tokens = tokenizer(text, STOPWORDS_SET)

        self.term_frequencies[doc_id] = Counter(tokens) 

        for token in set(tokens):
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

    def get_document_count(self) -> int:
        # returns the number of documents in the index
        return len(self.docmap)        
    

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
    
    def get_df(self, term: str) -> int:
        '''
        get the term frequency
        '''
        # Process the input term (lowercase, stem) to match index keys
        processed_term = stemmer.stem(term.lower())
        return len(self.index.get(processed_term, set()))


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
        term_frequency_path = os.path.join(cache_dir, 'term_frequency.pkl')


        # Save the index
        with open(index_path, 'wb') as f:
            pickle.dump(self.index, f)

        # Save the docmap
        with open(docmap_path, 'wb') as f:
            pickle.dump(self.docmap, f)
        print(f"Document map saved to {docmap_path}")

        # Save the term-frequency index
        with open(term_frequency_path, 'wb') as f:
            pickle.dump(self.term_frequencies, f)
        print(f"Document term-frequency index saved to {term_frequency_path}")

    def load(self):
        cache_dir = "cache"

        index_path = os.path.join(cache_dir, "index.pkl")
        docmap_path = os.path.join(cache_dir, "docmap.pkl")
        term_frequency_path = os.path.join(cache_dir, 'term_frequency.pkl')

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

        # Load the term-frequency-index
        try:
            with open(term_frequency_path, 'rb') as f:
                self.term_frequencies = pickle.load(f)
            print('term-frequency loaded')
        except Exception as e:
            raise FileNotFoundError(f"Error loading the term-frequency index: {e}")
        
    def get_tf(self, doc_id: int, term: str) -> int:
        """
        Return the times the token appears in the document with the given ID.
        """
        # tokenize input
        tokenized_terms = tokenizer(term, STOPWORDS_SET)

        # Enforce the single-token requirement
        if len(tokenized_terms) == 0:
            return 0
        elif len(tokenized_terms) > 1:
            raise ValueError(f"Expected a single term. Input: {term}")

        single_token = tokenized_terms[0]

        # get the document tf
        doc_term_counts = self.term_frequencies.get(doc_id)

        if doc_term_counts is None:
            return 0

        return doc_term_counts.get(single_token, 0)     

def remove_punctuation(text: str) -> str:
    # Remove punctuation from a string
    return text.translate(PUNCTUATION_REMOVER)

def tokenizer(text: str, stopwords: list[str]) -> set[str]:
    # word tokenizer
    processed_text = remove_punctuation(text).lower()
    
    # split by whitespace to get tokens
    tokens = [
        stemmer.stem(token) 
        for token in processed_text.split() 
        if token and token not in stopwords]

    # use set difference to remove stopwords
    return tokens

def main() -> None:
    # create InvertedIndex instance
    index = InvertedIndex()
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    # build command parser
    subparsers.add_parser("build", help="Build and save inverted index")
    
    # tf parser
    tf_parser = subparsers.add_parser("tf", help="Prints term frequency in a document")
    tf_parser.add_argument("doc_id", type=int, help="The id of the document searched")
    tf_parser.add_argument("term", type=str, help="The term to search")

    # idf parser
    idf_parser = subparsers.add_parser("idf", help="Prints term inverse frequency")
    idf_parser.add_argument("idf_term", type=str, help="The term to search")

    # tfidf parser
    tfidf_parser = subparsers.add_parser("tfidf", help="Prints term inverse frequency")
    tfidf_parser.add_argument("doc_id", type=int, help="The id of the document searched")
    tfidf_parser.add_argument("term", type=str, help="The term to search")


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
        
        case "build":
            # build the index
            index.build(movies_list)

            # save the index
            index.save()

        case "tf":
            doc_id = args.doc_id
            term = args.term

            try: 
                # load the index
                index.load()
            except FileNotFoundError as e:
                print(f"\n{e}")


            try: 
                frequency = index.get_tf(doc_id, term)
                print(frequency)
            except ValueError as e:
                # This catches the exception raised in get_tf if the input term tokenizes into multiple words
                print(f"Error: {e}")
            except KeyError:
                # Catches if doc_id doesn't exist (though get_tf should handle it gracefully)
                print(0)

        case "idf":
            idf_term = args.idf_term

            try: 
                # load the index
                index.load()
            except FileNotFoundError as e:
                print(f"\n{e}")
                return

            doc_count = index.get_document_count()
            term_doc_count = index.get_df(idf_term)

            idf = math.log((doc_count + 1) / (term_doc_count + 1))

            print(f"Inverse document frequency of '{args.idf_term}': {idf:.2f}")

        case "tfidf":
            doc_id = args.doc_id
            term = args.term

            try: 
                # load the index
                index.load()
            except FileNotFoundError as e:
                print(f"\n{e}")
                return
            
            try: 
                tf = index.get_tf(doc_id, term)
            except ValueError as e:
                # This catches the exception raised in get_tf if the input term tokenizes into multiple words
                print(f"Error: {e}")
            except KeyError:
                # Catches if doc_id doesn't exist (though get_tf should handle it gracefully)
                print(0)

            try:
                doc_count = index.get_document_count()
                term_doc_count = index.get_df(term)

                idf = math.log((doc_count + 1) / (term_doc_count + 1))
            
            except KeyError:
                print(0)

            try:
                tfidf = tf*idf
                print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tfidf:.2f}")
            except KeyError:
                print(0)

            
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()