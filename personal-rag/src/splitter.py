"""Chunking."""
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(docs, chunk_size=500, chunk_overlap=80):
    """Split documents into overlapping chunks; metadata (source, type) is kept."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(docs)
