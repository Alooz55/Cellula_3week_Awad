"""Entry point: python app.py [--rebuild] [--github-user NAME] [--repos a b] [-q "question"]"""
import argparse
import sys

from src.loaders import load_all
from src.splitter import split_documents
from src.embeddings import get_embeddings
from src.vector_store import build_vector_store, load_vector_store, index_exists
from src.rag import PersonalRAG, get_llm


def main():
    ap = argparse.ArgumentParser(description="Personal RAG over your professional data")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--rebuild", action="store_true", help="re-ingest data and rebuild the index")
    ap.add_argument("--github-user", default="", help="GitHub username to ingest")
    ap.add_argument("--repos", nargs="*", default=None, help="specific repos (default: all public)")
    ap.add_argument("-q", "--question", help="ask one question and exit")
    args = ap.parse_args()

    embeddings = get_embeddings()

    if args.rebuild or not index_exists():
        docs = load_all(args.data_dir, args.github_user, args.repos)
        if not docs:
            sys.exit("No documents found. Add files under data/ or pass --github-user.")
        chunks = split_documents(docs)
        print(f"Loaded {len(docs)} documents -> {len(chunks)} chunks")
        db = build_vector_store(chunks, embeddings)
    else:
        db = load_vector_store(embeddings)

    rag = PersonalRAG(db, get_llm())

    def show(question):
        out = rag.ask(question)
        print(f"\nAnswer: {out['answer']}\n\nSources: {', '.join(out['sources'])}")

    if args.question:
        show(args.question)
        return

    print("Ask about the person (type 'exit' to quit).")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in ("", "exit", "quit"):
            break
        show(q)


if __name__ == "__main__":
    main()
