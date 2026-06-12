"""
CLI interface for the PDF Q&A system.

Usage:
    python app.py --pdf path/to/document.pdf
    python app.py --pdf report.pdf --top-k 3 --chunk-size 400
"""

import argparse
import os
import sys
from rag_engine import PDFQuestionAnswering


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ask questions about a PDF using RAG + Claude."
    )
    parser.add_argument("--pdf", required=True, help="Path to the PDF file.")
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="Number of context chunks to retrieve per question (default: 5).",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=500,
        help="Character size of each text chunk (default: 500).",
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=100,
        help="Overlap between consecutive chunks (default: 100).",
    )
    parser.add_argument(
        "--model", default="claude-sonnet-4-20250514",
        help="Claude model to use for answer generation.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Validate PDF path
    if not os.path.isfile(args.pdf):
        print(f"Error: file not found — '{args.pdf}'")
        sys.exit(1)

    # Use API key if available; otherwise run in offline/demo mode.
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(
            "Warning: ANTHROPIC_API_KEY is not set. Running in offline/demo mock mode."
        )

    print(f"\nLoading '{args.pdf}' ...")
    qa = PDFQuestionAnswering(
        api_key=api_key,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        model=args.model,
    )

    try:
        n_chunks = qa.load_pdf(args.pdf)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    print(f"Document indexed — {n_chunks} chunks ready.")
    print("Type your question and press Enter. Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        print("\nThinking...\n")
        try:
            result = qa.ask(question)
        except Exception as e:
            print(f"Error generating answer: {e}\n")
            continue

        print(f"Answer:\n{result['answer']}\n")

        # Show source citations
        if result.get("sources"):
            print("Sources:")
            for i, src in enumerate(result["sources"], start=1):
                score_str = f"{src.get('score', 0):.3f}"
                preview = src["text"][:120].replace("\n", " ")
                print(f"  [{i}] Page {src['page']} (score {score_str}): {preview}...")

        usage = result.get("usage", {})
        if usage:
            print(
                f"\n  Tokens — in: {usage.get('input_tokens', '?')},"
                f" out: {usage.get('output_tokens', '?')}"
            )
        print()


if __name__ == "__main__":
    main()
