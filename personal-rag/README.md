# Personal RAG

A Retrieval-Augmented Generation system over a person's professional information
(CV, projects, certifications, GitHub). Answers are grounded in the stored sources
and always list which sources were used.

## Pipeline

```
load (loaders.py) -> chunk (splitter.py) -> embed (embeddings.py)
   -> FAISS store (vector_store.py) -> retrieve + LLM answer (rag.py)
```

## Structure

```
personal-rag/
├── data/
│   ├── cv.pdf
│   ├── projects/          # .pdf / .txt / .md, one file per project
│   └── certifications/    # .pdf / .txt / .md
├── src/
│   ├── loaders.py         # CV, folders, GitHub
│   ├── splitter.py        # RecursiveCharacterTextSplitter (500 / 80)
│   ├── embeddings.py      # all-MiniLM-L6-v2
│   ├── vector_store.py    # FAISS build/load
│   └── rag.py             # prompt, LLM (OpenRouter), PersonalRAG
├── app.py
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY="your-key"     # Windows PowerShell: $env:OPENROUTER_API_KEY="your-key"
```

Never hard-code the key in the code or commit it to GitHub.
Optional: `export GITHUB_TOKEN=...` to avoid GitHub API rate limits.

## Usage

```bash
# first run: ingest data + GitHub, build the index, start chat
python app.py --rebuild --github-user YOUR_USERNAME

# only some repos
python app.py --rebuild --github-user YOUR_USERNAME --repos repo1 repo2

# later runs reuse the saved index (faiss_index/)
python app.py

# one-shot question
python app.py -q "What projects has this person built?"
```

Run with `--rebuild` whenever you add or change files in `data/`.

## Notes
- Low LLM temperature (0.2) and a strict prompt reduce hallucinations.
- If the answer is not in the sources, the system says so instead of guessing.
- LinkedIn: export your profile as PDF and put it in `data/` (or `data/projects/`).
