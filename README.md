# The Reading Room — PDF RAG Chat

A retrieval-augmented generation (RAG) app that lets you ask questions about a PDF and get answers grounded **only** in that document. Built with LangChain, Mistral AI (embeddings + LLM), Chroma as the vector store, and a Streamlit front end.

## How it works

1. **`create_db.py`** loads a PDF, splits it into overlapping chunks, embeds each chunk with Mistral's embedding model, and stores the vectors in a local Chroma database.
2. **`streamlit_app.py`** retrieves the most relevant chunks for a question using MMR (Maximal Marginal Relevance) search, stuffs them into a prompt as context, and asks the Mistral LLM to answer using only that context.

```
PDF file --> chunk --> embed --> Chroma DB --> retrieve --> LLM answer
```

## Project structure

```
.
├── create_db.py        # One-time script: builds the Chroma vector store from a PDF
├── streamlit_app.py     # Chat UI for querying the vector store
├── chroma_db/            # Generated vector store (created by create_db.py)
├── .env                  # Your API keys (not committed)
└── README.md
```

> ⚠️ **Path consistency**: `create_db.py` must persist to the same directory that `streamlit_app.py` reads from. Both are set to `chroma_db` — if you rename one, rename the other (`PERSIST_DIR` at the top of `streamlit_app.py`).

## Setup

### 1. Clone / copy the project files

Make sure `create_db.py`, `streamlit_app.py`, and `requirements.txt` (below) are in the same folder.

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install streamlit langchain langchain-mistralai langchain-community chromadb python-dotenv pypdf
```

Or, if you save this as `requirements.txt`:

```
streamlit
langchain
langchain-mistralai
langchain-community
chromadb
python-dotenv
pypdf
```

then run:

```bash
pip install -r requirements.txt
```

### 4. Add your API key

Create a `.env` file in the project root:

```
MISTRAL_API_KEY=your_mistral_api_key_here
```

Get a key from the [Mistral AI console](https://console.mistral.ai/).

### 5. Point `create_db.py` at your PDF

Open `create_db.py` and update the file path:

```python
data = PyPDFLoader(r"C:\path\to\your\document.pdf")
```

## Usage

### Step 1 — Build the vector store (run once per document)

```bash
python create_db.py
```

This creates a `chroma_db/` folder containing your document's embeddings. Re-run this any time you change the source PDF or want to rebuild the index.

### Step 2 — Launch the chat UI

```bash
streamlit run streamlit_app.py
```

This opens the app in your browser (usually `http://localhost:8501`). Ask questions in the chat box — each answer includes a "catalog drawer" you can open to see exactly which passages from the document were used to generate it.

## Configuration (in the sidebar)

| Setting | What it does |
|---|---|
| **Model** | Which Mistral chat model answers your question |
| **Slips returned (k)** | Number of chunks retrieved and passed to the LLM as context |
| **Candidates scanned (fetch_k)** | Pool size MMR selects from before picking the top `k` |
| **Relevance ↔ Diversity (lambda_mult)** | 1.0 = pure relevance, 0.0 = maximum diversity among results |

## Notes & troubleshooting

- **"Could not open the archive" error**: usually means `chroma_db/` doesn't exist yet or the path doesn't match — run `create_db.py` first and check `PERSIST_DIR`.
- **Empty or irrelevant answers**: try increasing `k` or `fetch_k`, or rebuild the index with a smaller `chunk_size` for finer-grained retrieval.
- **Multiple PDFs**: loop `PyPDFLoader` over a folder of files in `create_db.py`, combine all resulting chunks, and pass the full list to `Chroma.from_documents`.
- **Answers not grounded**: the system prompt instructs the model to answer only from context and say so when it can't — if you see it going off-document, check that retrieval is actually returning relevant chunks (visible in the catalog drawer).
