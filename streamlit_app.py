"""
Streamlit web UI for the PDF Q&A system.

Run with:
    streamlit run streamlit_app.py
"""

import streamlit as st
from rag_engine import PDFQuestionAnswering
import tempfile
import os

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="PDF Q&A — RAG",
    page_icon="📄",
    layout="wide",
)

st.title("📄 PDF Question & Answer")
st.caption("Upload a PDF, ask questions, and get answers grounded in the document.")

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------

if "qa" not in st.session_state:
    st.session_state.qa = None          # PDFQuestionAnswering instance
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None    # filename for display
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {role, content, sources}
if "n_chunks" not in st.session_state:
    st.session_state.n_chunks = 0

# ---------------------------------------------------------------------------
# Sidebar — settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")

    api_key = st.text_input(
        "Anthropic API Key",
        type="password",
        value=os.environ.get("ANTHROPIC_API_KEY", ""),
        help="Your Anthropic API key. Never stored permanently.",
    )

    if not api_key:
        st.info(
            "No Anthropic API key provided. The app will run in offline/demo mode "
            "and return retrieved passages without calling Claude."
        )

    st.divider()
    st.subheader("Chunking")

    chunk_size = st.slider(
        "Chunk size (characters)",
        min_value=100,
        max_value=2000,
        value=500,
        step=50,
        help="Larger chunks preserve more context but may dilute relevance.",
    )
    chunk_overlap = st.slider(
        "Chunk overlap (characters)",
        min_value=0,
        max_value=500,
        value=100,
        step=25,
        help="Overlap prevents context loss at chunk boundaries.",
    )

    st.subheader("Retrieval")
    top_k = st.slider(
        "Top-K chunks",
        min_value=1,
        max_value=15,
        value=5,
        help="How many chunks to send to Claude as context.",
    )

    st.divider()
    st.subheader("Upload PDF")

    uploaded_file = st.file_uploader("Choose a PDF", type=["pdf"])

    if uploaded_file:
        if True:
            # Re-index if the file or any setting changed
            settings_key = (uploaded_file.name, chunk_size, chunk_overlap, top_k)
            if st.session_state.get("_settings_key") != settings_key:
                with st.spinner("Indexing PDF…"):
                    # Write upload to a temp file so pypdf can read it
                    with tempfile.NamedTemporaryFile(
                        suffix=".pdf", delete=False
                    ) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name

                    try:
                        qa = PDFQuestionAnswering(
                            api_key=api_key,
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap,
                            top_k=top_k,
                        )
                        n_chunks = qa.load_pdf(tmp_path)
                        st.session_state.qa = qa
                        st.session_state.pdf_name = uploaded_file.name
                        st.session_state.n_chunks = n_chunks
                        st.session_state.chat_history = []
                        st.session_state["_settings_key"] = settings_key
                        st.success(f"Indexed {n_chunks} chunks from '{uploaded_file.name}'")
                    except Exception as e:
                        st.error(f"Failed to index PDF: {e}")
                    finally:
                        os.unlink(tmp_path)

    if st.session_state.pdf_name:
        st.info(
            f"Active: **{st.session_state.pdf_name}**  \n"
            f"{st.session_state.n_chunks} chunks indexed"
        )

    if st.button("Clear chat history"):
        st.session_state.chat_history = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main — chat interface
# ---------------------------------------------------------------------------

if st.session_state.qa is None:
    st.info("Upload a PDF in the sidebar to get started.")
    st.stop()

# Render existing chat messages
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"📎 {len(msg['sources'])} source(s) cited"):
                for i, src in enumerate(msg["sources"], start=1):
                    score_pct = f"{src.get('score', 0) * 100:.1f}%"
                    st.markdown(f"**Source {i} — Page {src['page']}** (similarity: {score_pct})")
                    st.caption(src["text"])

# Chat input
if question := st.chat_input("Ask a question about the document…"):
    # Show user bubble immediately
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.chat_history.append({"role": "user", "content": question})

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer…"):
            try:
                result = st.session_state.qa.ask(question)
                answer = result["answer"]
                sources = result.get("sources", [])

                st.markdown(answer)

                if sources:
                    with st.expander(f"📎 {len(sources)} source(s) cited"):
                        for i, src in enumerate(sources, start=1):
                            score_pct = f"{src.get('score', 0) * 100:.1f}%"
                            st.markdown(
                                f"**Source {i} — Page {src['page']}** (similarity: {score_pct})"
                            )
                            st.caption(src["text"])

                usage = result.get("usage", {})
                if usage:
                    st.caption(
                        f"Tokens — in: {usage.get('input_tokens', '?')},"
                        f" out: {usage.get('output_tokens', '?')}"
                    )

                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except Exception as e:
                err_msg = f"Error: {e}"
                st.error(err_msg)
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": err_msg,
                    "sources": [],
                })
