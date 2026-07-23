#!/usr/bin/env python3

from pathlib import Path

import argparse
import json

import lib.semantic_search as ss

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparsers.add_parser("verify", help="Verify semantic search model")
    
    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Test the embeddings functionality")
    
    embed_text_parser = subparsers.add_parser("embed_text", help="Get a sense of the embedding with the input text")
    embed_text_parser.add_argument("text", type=str, help="text to embed")

    embed_query_parser = subparsers.add_parser("embed_query", help="Embed query text")
    embed_query_parser.add_argument("query", type=str, help="text to embed")

    search_parser = subparsers.add_parser("search", help="Semantic search for documents based on query")
    search_parser.add_argument("query", type=str, help="search query")
    search_parser.add_argument("--limit", type=int, help="Max number of search results to return", default=5)
    
    chunk_parser = subparsers.add_parser("chunk", help="chunk text into --chunk-size number of words")
    chunk_parser.add_argument("text", type=str, help="text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, help="Number of words per chunk", default=200)
    chunk_parser.add_argument("--overlap", type=int, help="Number of words to overlap from the previous chunk", default=0)

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="chunk text into --max-chunk-size number of sentences or less")
    semantic_chunk_parser.add_argument("text", type=str, help="text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, help="Number of sentences per chunk", default=200)
    semantic_chunk_parser.add_argument("--overlap", type=int, help="Number of sentences to overlap from the previous chunk", default=0)

    embed_chunks_parser = subparsers.add_parser("embed_chunks", help="embed chunks")
    

    args = parser.parse_args()

    match args.command:
        case "verify":
            ss.verify_model()
        case "verify_embeddings":
            ss.verify_embeddings()
        case "embed_text":
            ss.embed_text(args.text)
        case "embed_query":
            ss.embed_query_text(args.query)
        case "search":
            s = ss.SemanticSearch()
            doc = Path(__file__).parent.parent.joinpath("data","movies.json")
            with open(doc, "r") as f:
                documents = json.load(f)
            embeddings = s.load_or_create_embeddings(documents['movies'])
            results = s.search(args.query, args.limit)
            if len(results) == 0:
                print(f"No results returned for query: {args.query}")
            for i, res in enumerate(results):
                print(f"{i+1}. {res["title"]} (score: {res["score"]})")
                print(f"\t{res["description"]}\n")
        case "chunk":
            results = ss.chunk(args.text, args.chunk_size, args.overlap)
            print(f"Chunking {results['numchar']} characters")
            for n, line in enumerate(results['lines']):
                print(f"{n+1}. {line}")
        case "semantic_chunk":
            results = ss.semantic_chunk(args.text, args.max_chunk_size, args.overlap)
            print(f"Semantically chunking {results['numchar']} characters")
            for n, line in enumerate(results['chunks']):
                print(f"{n+1}. {line}")
        case "embed_chunks":
            doc = Path(__file__).parent.parent.joinpath("data","movies.json")
            with open(doc, "r") as f:
                documents = json.load(f)
            chunk_ss = ss.ChunkedSemanticSearch()
            embeddings = chunk_ss.load_or_create_chunk_embeddings(documents['movies'])
            print(f"Generated {len(embeddings)} chunked embeddings")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()