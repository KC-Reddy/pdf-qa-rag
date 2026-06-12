# PDF Q&A — RAG System

Ask natural-language questions about any PDF document. Answers are grounded exclusively in the document content, with cited page references.

Built by **Krishna Chaitanya Reddy Kothakapu** — [github.com/KC-Reddy](https://github.com/KC-Reddy)

---

## Architecture

```
┌──────────────┐
│   PDF File   │
└──────┬───────┘
       │  pypdf
       ▼
┌──────────────┐
│  Page Text   │  (per-page, with page numbers retained)
└──────┬───────┘
       │  LangChain RecursiveCharacterTextSplitter
       │  chunk_size=500, overlap=100
       ▼
┌──────────────┐
│    Chunks    │  (text + page number + chunk index)
└──────┬───────┘
       │  scikit-learn TfidfVectorizer
       ▼
┌──────────────────────┐
│  TF-IDF Vector Index │
└──────┬───────────────┘
       │  cosine_similarity  ◄── User Question
       ▼
┌──────────────┐
│  Top-K       │  most relevant chunks
│  Chunks      │
└──────┬───────┘
       │  Anthropic Claude API
       │  (answer ONLY from context, cite sources)
       ▼
┌──────────────┐
│   Answer +   │
│   Citations  │
└──────────────┘
```

---

## Tech Stack

| Component        | Library / Service                          |
|------------------|--------------------------------------------|
| PDF parsing      | `pypdf`                                    |
| Text chunking    | `langchain-text-splitters`                 |
| Vector retrieval | `scikit-learn` (TF-IDF + cosine similarity)|
| LLM              | Anthropic Claude (`claude-sonnet-4-20250514`) |
| Web UI           | `streamlit`                                |
| CLI              | Python `argparse`                          |

---

## Setup

### 1. Clone / enter the project directory

```bash
cd pdf-qa-rag
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your Anthropic API key

```bash
# macOS / Linux
export ANTHROPIC_API_KEY="sk-ant-..."

# Windows (Command Prompt)
set ANTHROPIC_API_KEY=sk-ant-...

# Windows (PowerShell)
$env:ANTHROPIC_API_KEY="sk-ant-..."
```

If you do not provide `ANTHROPIC_API_KEY`, the app can still run in offline/demo mode. It will skip the Claude call and return the retrieved passages directly.

To force mock mode even if a key is present, set:

```bash
export MOCK_MODE=1
```

---

## Usage

### CLI

```bash
python app.py --pdf path/to/document.pdf
```

Optional flags:

| Flag              | Default | Description                             |
|-------------------|---------|-----------------------------------------|
| `--top-k`         | 5       | Number of context chunks per question   |
| `--chunk-size`    | 500     | Characters per chunk                    |
| `--chunk-overlap` | 100     | Overlap between consecutive chunks     |
| `--model`         | claude-sonnet-4-20250514 | Claude model to use     |

Example:

```bash
python app.py --pdf annual_report.pdf --top-k 3 --chunk-size 600
```

Then type your questions interactively. Type `quit` or `exit` to stop.

---

### Streamlit Web UI

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501` in your browser.

1. Enter your Anthropic API key in the sidebar.
2. Adjust chunk size, overlap, and top-K sliders as desired.
3. Upload a PDF using the file uploader.
4. Type questions in the chat box at the bottom.
5. Expand the **Sources** section under each answer to see cited passages.

---

## Project Structure

```
pdf-qa-rag/
├── rag_engine.py        # Core RAG pipeline (extraction → chunking → retrieval → generation)
├── app.py               # CLI interface
├── streamlit_app.py     # Streamlit web UI
├── requirements.txt     # Python dependencies
├── .gitignore
└── README.md
```

---

## Notes

- The system answers **only** from the uploaded document. If the answer is not present, Claude will say so.
- TF-IDF retrieval works best when the question uses similar terminology to the document. For highly paraphrased queries, consider upgrading to a dense embedding model.
- No data is stored persistently; everything lives in memory for the duration of the session.
