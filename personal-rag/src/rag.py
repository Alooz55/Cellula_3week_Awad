"""Retrieval + grounded generation with source citations."""
import os

from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

PROMPT = PromptTemplate.from_template(
    """You are an assistant that answers questions about a person's professional background
(CV, projects, certifications, GitHub).

Rules:
- Answer ONLY from the context below.
- If the answer is not in the context, say: "I don't know based on the available sources."
- Do not invent facts, dates, employers or skills.
- Be concise (under 200 words) and mention the source names you relied on.

## Context
{context}

## Question
{question}

## Answer"""
)


def get_llm(model="openrouter/free", temperature=0.2):
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("Set the OPENROUTER_API_KEY environment variable (see README).")
    return ChatOpenAI(
        api_key=key,
        base_url="https://openrouter.ai/api/v1",
        model=model,
        temperature=temperature,  # low temperature = fewer hallucinations
    )


class PersonalRAG:
    def __init__(self, vector_store, llm, k=4):
        self.db = vector_store
        self.llm = llm
        self.k = k

    def retrieve(self, question):
        return self.db.similarity_search_with_score(question, k=self.k)

    def ask(self, question):
        results = self.retrieve(question)
        context = "\n\n".join(
            f"[Source: {d.metadata.get('source')}]\n{d.page_content}" for d, _ in results
        )
        answer = self.llm.invoke(PROMPT.format(context=context, question=question)).content
        sources = sorted({d.metadata.get("source", "unknown") for d, _ in results})
        return {"answer": answer, "sources": sources, "chunks": results}
