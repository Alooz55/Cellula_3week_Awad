"""Chroma vector store: build, save, load."""
import os
import shutil

from langchain_chroma import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata

INDEX_DIR = "chroma_db"
COLLECTION = "personal_rag"
_META = {"hnsw:space": "cosine"}


def build_vector_store(chunks, embeddings, index_dir=INDEX_DIR):
    if not chunks:
        raise ValueError("No chunks to index. Add files to data/ or set a GitHub user.")
    if os.path.isdir(index_dir):
        shutil.rmtree(index_dir)  # rebuild from scratch so chunks are not duplicated
    chunks = filter_complex_metadata(chunks)  # Chroma only accepts str/int/float/bool metadata
    return Chroma.from_documents(
        chunks,
        embeddings,
        collection_name=COLLECTION,
        persist_directory=index_dir,
        collection_metadata=_META,
    )


def load_vector_store(embeddings, index_dir=INDEX_DIR):
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=embeddings,
        persist_directory=index_dir,
        collection_metadata=_META,
    )


def index_exists(index_dir=INDEX_DIR):
    return os.path.isfile(os.path.join(index_dir, "chroma.sqlite3"))
