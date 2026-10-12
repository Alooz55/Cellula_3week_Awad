# Personal RAG

A Retrieval-Augmented Generation system over a person's professional information:
LinkedIn profile, GitHub repositories, CVs / resumes, portfolio, projects,
certifications and professional experience. Answers are grounded in the stored
sources and always list which sources were used.

## Pipeline

```
load (loaders.py) -> chunk (splitter.py) -> embed (embeddings.py)
   -> Chroma store (vector_store.py) -> retrieve + LLM answer (rag.py)
```

## Structure

```
personal-rag/
├── data/
│   ├── cv.pdf             # optional: loose cv.* / resume.* files are picked up too
│   ├── cv/                # CVs (pdf / docx / txt / md)
│   ├── resumes/           # resumes (same formats)
│   ├── linkedin/          # LinkedIn profile "Save to PDF" and/or data-export CSVs
│   ├── portfolio/         # portfolio documents (pdf / docx / txt / md)
│   ├── projects/          # one file per project
│   ├── certifications/    # certificates / credential lists
│   └── experience/        # job descriptions, achievements, references
├── src/
│   ├── loaders.py         # files, CSV, portfolio URLs, GitHub
│   ├── splitter.py        # RecursiveCharacterTextSplitter (500 / 80)
│   ├── embeddings.py      # all-MiniLM-L6-v2
│   ├── vector_store.py    # Chroma build/load (persisted in chroma_db/)
│   └── rag.py             # prompt, LLM (OpenRouter), PersonalRAG
├── app.py
├── requirements.txt
└── README.md
```

Supported file types: `.pdf`, `.docx`, `.txt`, `.md`, `.csv`. Every chunk keeps its
`source` and `type` (cv, linkedin, portfolio, project, certification, experience, github).

## Data sources

| Source | How to add it |
|---|---|
| **LinkedIn** | Profile -> More -> *Save to PDF* -> put it in `data/linkedin/`. Or request *Get a copy of your data* and drop `Positions.csv`, `Education.csv`, `Certifications.csv`, `Skills.csv`, `Profile.csv` there (`Connections.csv` and `Messages.csv` are ignored on purpose). LinkedIn pages are not scraped (against their terms). |
| **GitHub** | `--github-user NAME` (profile + all public repos + READMEs), optionally `--repos a b`. |
| **CVs / resumes** | Files in `data/cv/`, `data/resumes/` (or `data/cv.pdf`). |
| **Portfolio** | Files in `data/portfolio/` and/or `--portfolio-url https://...`. |
| **Projects / certifications / experience** | Files in the matching folder. |

## Setup

```bash
pip install -r requirements.txt
export OPENROUTER_API_KEY="your-key"     # Windows PowerShell: $env:OPENROUTER_API_KEY="your-key"
```

Never hard-code the key in the code or commit it to GitHub.
Optional: `export GITHUB_TOKEN=...` to avoid GitHub API rate limits.

## Usage

```bash
# first run: ingest everything, build the index, start chat
python app.py --rebuild --github-user YOUR_USERNAME --portfolio-url https://your-site.com

# only some repos
python app.py --rebuild --github-user YOUR_USERNAME --repos repo1 repo2

# later runs reuse the saved index (chroma_db/)
python app.py

# one-shot question
python app.py -q "What projects has this person built?"
```

Run with `--rebuild` whenever you add or change files in `data/` (the old index is
deleted and rebuilt, so nothing is duplicated).

## Notes
- Low LLM temperature (0.2) and a strict prompt reduce hallucinations.
- If the answer is not in the sources, the system says so instead of guessing.
- If the CV and LinkedIn disagree, the answer points out the difference.
- `PersonalRAG.ask(question, doc_type="github")` restricts retrieval to one source type.
- `data/` contains personal information: keep it out of public repos.
