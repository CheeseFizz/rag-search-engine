#!/usr/bin/env python3

import argparse, json, math
from pathlib import Path

import lib.search as ls

# Create database for searching
with open(Path(__file__).parent.parent.joinpath("data","movies.json")) as f:
    MDB = json.load(f)

# Load stopwords
with open(Path(__file__).parent.parent.joinpath("data","stopwords.txt")) as f:
    SWL = f.read().splitlines()



def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    search_parser = subparsers.add_parser("build", help="Builds the search index")

    search_parser = subparsers.add_parser("tf", help="Gets the count of term in the document")
    search_parser.add_argument("doc_id", type=int, help="Document ID")
    search_parser.add_argument("term", type=str, help="Search term")

    search_parser = subparsers.add_parser("idf", help="Gets the inverse document frequency of term in the database")
    search_parser.add_argument("term", type=str, help="Search term")

    search_parser = subparsers.add_parser("tfidf", help="Gets the inverse document frequency of term in the database")
    search_parser.add_argument("doc_id", type=int, help="Document ID")
    search_parser.add_argument("term", type=str, help="Search term")

    args = parser.parse_args()

    inv_index = ls.InvertedIndex(SWL)
    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            inv_index.load()
            results = ls.keyword_search(args.query, inv_index, SWL)
            for i in range(len(results)):
                m = inv_index.docmap[results[i]]
                print(f"{i+1}. {m['title']}")
            print("")
        case "build":
            inv_index.build(MDB["movies"])
            inv_index.save()
        case "tf":
            inv_index.load()
            results = inv_index.get_tf(args.doc_id, args.term)
            print(results)
        case "idf":
            inv_index.load()
            total_doc_count = len(inv_index.docmap)
            term_match_doc_count = len(inv_index.get_documents(ls.tokenize(args.term,SWL)[0]))
            idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            inv_index.load()
            total_doc_count = len(inv_index.docmap)
            term_match_doc_count = len(inv_index.get_documents(ls.tokenize(args.term,SWL)[0]))
            idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
            tf = inv_index.get_tf(args.doc_id, args.term)
            tf_idf = tf*idf
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()