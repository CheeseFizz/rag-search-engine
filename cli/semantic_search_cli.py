#!/usr/bin/env python3

import argparse

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
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()