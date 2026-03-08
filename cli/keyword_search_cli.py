#!/usr/bin/env python3

import argparse, json
from pathlib import Path

import lib.search as ls

# Create database for searching
with open(Path(__file__).parent.parent.joinpath("data","movies.json")) as f:
    MDB = json.load(f)



def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = ls.keyword_search(args.query, MDB)
            for i in range(5):
                print(f"{i+1}. {results[i]["title"]}")
            print("")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()