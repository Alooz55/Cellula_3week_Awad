"""FAISS vector store: build, save, load."""
import os

from langchain_community.vectorstores import FAISS

INDEX_DIR = "faiss_index"


def build_vector_store(chunks, embeddings, index_dir=INDEX_DIR):
    if not chunks:
        raise ValueError("No chunks to index. Add files to data/ or set a GitHub user.")
    db = FAISS.from_documents(chunks, embeddings)
    db.save_local(index_dir)
    return db


def load_vector_store(embeddings, index_dir=INDEX_DIR):
    # The index is created locally by this project, so deserialization is safe.
    return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)


def index_exists(index_dir=INDEX_DIR):
    return os.path.isfile(os.path.join(index_dir, "index.faiss"))
