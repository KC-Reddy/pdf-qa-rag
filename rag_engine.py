"""
RAG (Retrieval-Augmented Generation) engine for PDF Q&A.

Pipeline overview:
  PDF → extract text → chunk → TF-IDF index → retrieve top-k chunks → Claude API → answer
"""

import os
import numpy as np
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import anthropic


# ---------------------------------------------------------------------------
# Step 1: Text Extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text from every page of a PDF, keeping page numbers for citation.

    Returns a list of dicts: {"page": int, "text": str}
    Page numbers are 1-indexed to match what users see in PDF viewers.
    """
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:  # skip blank pages
            pages.append({"page": i, "text": text})
    return pages


# ---------------------------------------------------------------------------
# Step 2: Chunking
# ---------------------------------------------------------------------------

def chunk_text(pages: list[dict], chunk_size: int = 500, chunk_overlap: int = 100) -> list[dict]:
    """
    Split page text into overlapping chunks so that context is not cut at boundaries.

    chunk_size=500 keeps each chunk within typical LLM prompt budgets while
    preserving enough context for coherent answers.
    chunk_overlap=100 ensures sentences that span a boundary appear in both
    neighbouring chunks, preventing retrieval gaps.

    Returns chunks as dicts: {"page": int, "chunk_index": int, "text": str}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Try to split on paragraph/sentence boundaries before falling back to chars.
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for page_data in pages:
        splits = splitter.split_text(page_data["text"])
        for idx, split in enumerate(splits):
            chunks.append({
                "page": page_data["page"],
                "chunk_index": idx,
                "text": split.strip(),
            })
    return chunks


# ---------------------------------------------------------------------------
# Step 3: Vector Store (TF-IDF + cosine similarity)
# ---------------------------------------------------------------------------

class VectorStore:
    """
    Lightweight retrieval index backed by TF-IDF vectors.

    Why TF-IDF instead of dense embeddings?
    - Zero external API calls for indexing (no embedding model needed).
    - Fast, deterministic, and works well for factual/keyword-heavy PDFs.
    - Suitable for documents where the exact terminology in the question
      is likely to appear in the relevant passage.

    For semantic (paraphrase) queries, a dense embedding model would be
    more powerful, but TF-IDF is a good baseline that runs fully offline.
    """

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),   # unigrams + bigrams capture phrase context
            stop_words="english",
            min_df=1,
            sublinear_tf=True,    # log-scale TF dampens very frequent terms
        )
        self.matrix = None   # shape: (n_chunks, n_features)
        self.chunks: list[dict] = []

    def build(self, chunks: list[dict]) -> None:
        """Fit the vectorizer and store the TF-IDF matrix for all chunks."""
        self.chunks = chunks
        texts = [c["text"] for c in chunks]
        self.matrix = self.vectorizer.fit_transform(texts)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Return the top_k chunks most similar to the query.

        Cosine similarity is used because it is length-invariant — a short
        chunk and a long chunk with the same term distribution score equally.
        """
        if self.matrix is None:
            raise RuntimeError("VectorStore.build() must be called before retrieve().")

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self.matrix).flatten()

        # argsort is ascending; take from the end for top scores
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk = dict(self.chunks[idx])
            chunk["score"] = float(scores[idx])
            results.append(chunk)
        return results


# ---------------------------------------------------------------------------
# Step 4: Answer Generation via Claude
# ---------------------------------------------------------------------------

def generate_answer(
    question: str,
    context_chunks: list[dict],
    api_key: str | None = None,
    model: str = "claude-sonnet-4-20250514",
) -> dict:
    """
    Send the retrieved chunks and the user's question to Claude and return
    a structured answer with source citations.

    The system prompt instructs Claude to answer ONLY from the provided
    context so the response is grounded in the document, not Claude's
    parametric knowledge.  Source page numbers are embedded in the context
    so Claude can cite them naturally.
    """
# MOCK MODE: if no API key is available, skip the LLM call and return
    # the retrieved context directly. This lets the retrieval pipeline
    # (chunking, TF-IDF, ranking) be demoed end-to-end without API costs.
    effective_key = (api_key or "").strip() or os.environ.get("ANTHROPIC_API_KEY", "").strip()
    mock_mode = (
        not effective_key
        or os.environ.get("MOCK_MODE", "").strip().lower() in {"1", "true", "yes"}
    )
    if mock_mode:
        preview = "\n\n---\n\n".join(
            f"[Source {i} — Page {chunk['page']}]\n{chunk['text'][:300]}"
            for i, chunk in enumerate(context_chunks, start=1)
        )
        return {
            "answer": (
                "(Running in offline/demo mode — no API key provided, "
                "so the LLM generation step is skipped.)\n\n"
                "Here are the most relevant passages retrieved for your question:\n\n"
                f"{preview}"
            ),
            "sources": context_chunks,
            "model": "mock",
            "usage": {},
        }
    client = anthropic.Anthropic(api_key=effective_key)

    # Build a readable context block with page labels for citation
    context_parts = []
    for i, chunk in enumerate(context_chunks, start=1):
        context_parts.append(
            f"[Source {i} — Page {chunk['page']}]\n{chunk['text']}"
        )
    context_text = "\n\n---\n\n".join(context_parts)

    system_prompt = (
        "You are a precise document assistant. "
        "Answer the user's question using ONLY the context passages provided below. "
        "Do not use any outside knowledge. "
        "Always cite your sources by referencing the [Source N — Page P] labels. "
        "If the answer is not present in the context, say: "
        "'I could not find an answer to that question in the provided document.'"
    )

    user_message = (
        f"Context:\n\n{context_text}\n\n"
        f"Question: {question}"
    )

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    return {
        "answer": response.content[0].text,
        "sources": context_chunks,
        "model": model,
        "usage": {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        },
    }


# ---------------------------------------------------------------------------
# Step 5: High-level Pipeline Class
# ---------------------------------------------------------------------------

class PDFQuestionAnswering:
    """
    End-to-end PDF Q&A pipeline.

    Usage:
        qa = PDFQuestionAnswering(api_key="sk-ant-...")
        qa.load_pdf("report.pdf")
        result = qa.ask("What is the main finding?")
        print(result["answer"])
    """

    def __init__(
        self,
        api_key: str | None = None,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        top_k: int = 5,
        model: str = "claude-sonnet-4-20250514",
    ):
        self.api_key = api_key
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.model = model

        self.store = VectorStore()
        self._loaded = False
        self.pdf_path: str | None = None
        self.chunks: list[dict] = []

    def _is_mock_mode(self) -> bool:
        effective_key = (self.api_key or "").strip() or os.environ.get("ANTHROPIC_API_KEY", "").strip()
        return not effective_key or os.environ.get("MOCK_MODE", "").strip().lower() in {"1", "true", "yes"}

    def load_pdf(self, pdf_path: str) -> int:
        """
        Extract, chunk, and index a PDF.

        Returns the total number of chunks indexed so callers can give
        the user feedback about document size.
        """
        self.pdf_path = pdf_path

        # Extract raw page text
        pages = extract_text_from_pdf(pdf_path)
        if not pages:
            raise ValueError(f"No extractable text found in '{pdf_path}'.")

        # Split into overlapping chunks
        self.chunks = chunk_text(pages, self.chunk_size, self.chunk_overlap)
        if not self.chunks:
            raise ValueError("Text extraction produced no chunks.")

        # Build the TF-IDF index
        self.store.build(self.chunks)
        self._loaded = True

        return len(self.chunks)

    def ask(self, question: str) -> dict:
        """
        Answer a question about the loaded PDF.

        Returns a dict with keys: answer, sources, model, usage.
        """
        if not self._loaded:
            raise RuntimeError("No PDF loaded. Call load_pdf() first.")

        # Retrieve the most relevant chunks
        context_chunks = self.store.retrieve(question, top_k=self.top_k)
        if not context_chunks:
            return {
                "answer": "No relevant content found in the document for your question.",
                "sources": [],
                "model": self.model,
                "usage": {},
            }

        if self._is_mock_mode():
            return generate_answer(
                question=question,
                context_chunks=context_chunks,
                api_key=None,
                model=self.model,
            )

        # Generate a grounded answer
        return generate_answer(
            question=question,
            context_chunks=context_chunks,
            api_key=self.api_key,
            model=self.model,
        )
