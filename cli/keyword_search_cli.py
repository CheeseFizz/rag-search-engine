#!/usr/bin/env python3

import argparse, json
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
    search_parser.add_argument("document_id", type=int, help="Search query")
    search_parser.add_argument("term", type=str, help="Search query")

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
            results = inv_index.get_tf(args.document_id, args.term)
            print(results)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()