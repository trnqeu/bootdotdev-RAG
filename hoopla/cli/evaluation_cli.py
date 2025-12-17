import argparse
import json
from lib.search_utils import normalize_scores
from lib.hybrid_search import HybridSearch, weighted_search_command, rrf_search_command
from lib.search_utils import load_movies

RRF_K_PARAM = 60

def calculate_precision_at_k(retrieved_titles: set, relevant_titles: set) -> float:
    """Calculates Precision@k."""
    if not retrieved_titles:
        return 0.0
        
    # Intersection is the set of relevant items that were also retrieved (True Positives)
    true_positives = retrieved_titles.intersection(relevant_titles)
    
    # Precision@k = |True Positives| / |Retrieved|
    return len(true_positives) / len(retrieved_titles)

def calculate_recall_at_k(retrieved_titles: set, relevant_titles: set) -> float:
    """
    Calculates Recall@k.
    Recall@k = |True Positives| / |Total Relevant Items|
    """
    if not retrieved_titles:
        return 1.0
    else:
        true_positives = retrieved_titles.intersection(relevant_titles)
        return len(true_positives) / len(relevant_titles)

def calculate_f1_score(precision: float, recall: float) -> float:
    """Calculates the F1 score (harmonic mean of precision and recall)."""
    if (precision + recall) == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    print(f"k={limit}\n")

    # load data
    movies = load_movies()

    # create instance of HybridSearch
    hybrid_search = HybridSearch(movies)

    # run evaluation logic here
    with open('./data/golden_dataset.json', 'r') as f:
        eval_data = json.load(f)

    test_cases_list = eval_data.get("test_cases", [])

    # run RRF search and calculate metrics for each test case
    for test_case in test_cases_list:
        query = test_case['query']

        relevant_titles = set(test_case['relevant_docs'])

        rrf_result_dict = hybrid_search.rrf_search(
            query = query,
            k=RRF_K_PARAM,
            limit = limit,
            rerank_method=None
        )

        retrieved_titles_list = [doc['title'] for doc in rrf_result_dict]
        retrieved_titles = set(retrieved_titles_list)

        # calculate precision@k
        precision_at_k = calculate_precision_at_k(retrieved_titles, relevant_titles)

        # calculate recall@k
        recall_at_k = calculate_recall_at_k(retrieved_titles, relevant_titles)

        # f1 score
        f1 = calculate_f1_score(precision_at_k, recall_at_k)

        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision_at_k:.4f}")

        print(f"  - Recall@{limit}: {recall_at_k:.4f}")

        print(f"  - F1 Score: {f1:.4f}")

        print(f"  - Retrieved: {', '.join(retrieved_titles_list)}")

        print(f"  - Relevant: {', '.join(sorted(list(relevant_titles)))}") # Sort for consistent printing
        print()

if __name__ == "__main__":
    main()