#!/usr/bin/env python3
import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest='command', help='available commands')
    verify_parser = subparsers.add_parser("verify", help="Verify that the embedding model is loaded")
    embeddings_parser = subparsers.add_parser("verify_embeddings", help="Load or create embeddings and print details")
    text_embed_parser = subparsers.add_parser("embed_text", help='text to be embedded')
    text_embed_parser.add_argument("text", type=str, help='text to be embedded')

    args = parser.parse_args()

    match args.command:
        case 'verify':
            verify_model()

        case 'embed_text':
            embed_text(args.text)

        case 'verify_embeddings':
            verify_embeddings()

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()