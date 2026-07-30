import os

os.environ.setdefault("USER_AGENT", "coursemate-ai/1.0")

import shutil
import tempfile
from collections import defaultdict

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

import chromadb

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "rag_collection"
MODEL_OPTIONS = [
    "mistral-small-2506",
    "mistral-small-latest",
    "mistral-medium-latest",
    "mistral-large-latest",
]
SUGGESTED_QUESTIONS = [
    "Summarize this document.",
    "What are the key takeaways?",
    "Explain the main topic simply.",
    "List the most important points.",
    "What questions does this raise?",
    "Are there any limitations or gaps?",
    "Give me a quick overview.",
    "Explain like I'm new to this.",
    "What should I remember most?",
]

st.set_page_config(page_title="CourseMateAi", page_icon="🖋️", layout="wide")

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant.
            Use ONLY the provided context to answer the question.
            If the answer is not present in the context, say: "I could not find the answer in the document."
            """,
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
""",
        ),
    ]
)

# ---------------------------------------------------------------------------
# Theme — dark navy field notebook with a gold workflow rail
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,500;0,6..72,600;1,6..72,500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    :root {
        --bg: #1B222C;
        --panel: #1F2733;
        --panel-2: #171D26;
        --gold: #D3A360;
        --gold-dim: rgba(211,163,96,0.35);
        --text: #E8E6E1;
        --text-dim: #8B93A1;
        --line: rgba(211,163,96,0.22);
    }

    [data-testid="stAppViewContainer"] { background-color: var(--bg); }
    [data-testid="stHeader"] { background-color: transparent; }
    html, body, [class*="css"] { font-family: 'IBM Plex Sans', sans-serif; color: var(--text); }
    .block-container { padding-top: 2rem; }

    /* ---------- Sidebar ---------- */
    [data-testid="stSidebar"] {
        background-color: var(--panel);
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] * { color: var(--text) !important; }
    [data-testid="stSidebar"] hr { border-color: var(--line); }

    /* Numbered workflow steps */
    .idx-step { display: flex; align-items: center; gap: 0.55rem; margin: 0.2rem 0 0.35rem 0; }
    .idx-step-num {
        font-family: 'Newsreader', serif;
        font-size: 1rem;
        color: var(--gold);
        border: 1px solid var(--gold);
        border-radius: 50%;
        width: 1.6rem; height: 1.6rem; min-width: 1.6rem;
        display: flex; align-items: center; justify-content: center;
    }
    .idx-step-title { font-family: 'Newsreader', serif; font-size: 1.1rem; font-weight: 600; color: var(--gold); }
    .idx-step-sub {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.66rem;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        color: var(--text-dim) !important;
        margin: 0.1rem 0 0.7rem 2.15rem;
    }

    /* Tabs (PDF / URL input) */
    [data-testid="stSidebar"] .stTabs [data-baseweb="tab-list"] { gap: 4px; border-bottom: 1px solid var(--line); }
    [data-testid="stSidebar"] .stTabs [data-baseweb="tab"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: var(--text-dim) !important;
        padding: 0.4rem 0.6rem;
    }
    [data-testid="stSidebar"] .stTabs [aria-selected="true"] {
        color: var(--gold) !important;
        border-bottom: 2px solid var(--gold);
    }

    /* Inputs, number inputs, selects — dark fill, gold hairline */
    div[data-baseweb="input"], div[data-baseweb="select"] > div, div[data-baseweb="base-input"] {
        background-color: var(--panel-2) !important;
        border: 1px solid var(--gold-dim) !important;
        border-radius: 5px !important;
    }
    [data-testid="stTextArea"] textarea { background-color: var(--panel-2) !important; border: 1px solid var(--gold-dim) !important; }
    [data-testid="stFileUploaderDropzone"] {
        background-color: var(--panel-2);
        border: 1px dashed var(--gold-dim);
    }

    /* Slider labels */
    [data-testid="stSlider"] label p, [data-testid="stNumberInput"] label p {
        font-family: 'IBM Plex Mono', monospace !important;
        font-size: 0.68rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--text-dim) !important;
    }

    /* Buttons — gold outline, transparent fill */
    .stButton>button {
        background-color: transparent;
        color: var(--text) !important;
        border: 1px solid var(--gold);
        border-radius: 6px;
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 500;
        transition: background-color 0.15s ease;
    }
    .stButton>button:hover { background-color: rgba(211,163,96,0.12); border-color: var(--gold); }
    .stButton>button:disabled { border-color: rgba(211,163,96,0.25); color: var(--text-dim) !important; }

    /* Vector store tag */
    .idx-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: var(--gold);
        border: 1px solid var(--gold-dim);
        border-radius: 4px;
        padding: 0.15rem 0.55rem;
        margin: 0.4rem 0;
    }

    /* ---------- Index cards (catalog entries) ---------- */
    .idx-card {
        position: relative;
        background: var(--panel-2);
        border: 1px solid var(--line);
        border-radius: 4px;
        padding: 0.55rem 0.7rem 0.5rem 0.7rem;
        margin-bottom: 0.5rem;
    }
    .idx-card-num { font-family: 'IBM Plex Mono', monospace; font-size: 0.66rem; color: var(--gold); }
    .idx-card-stamp {
        float: right;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.6rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--gold);
        border: 1px solid var(--gold);
        border-radius: 20px;
        padding: 0.03rem 0.4rem;
    }
    .idx-card-name { font-weight: 600; font-size: 0.87rem; margin-top: 0.15rem; word-break: break-word; }
    .idx-card-meta { font-family: 'IBM Plex Mono', monospace; font-size: 0.66rem; color: var(--text-dim); margin-top: 0.1rem; }

    /* Status pill */
    .idx-status {
        display: inline-flex; align-items: center; gap: 0.4rem;
        font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem;
        padding: 0.25rem 0.7rem; border-radius: 20px; border: 1px solid var(--line);
    }
    .idx-status-dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; }

    /* ---------- Main: hero card ---------- */
    .idx-hero-card {
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1.2rem;
        background: var(--panel);
    }
    .idx-hero-eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: var(--gold);
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.45rem;
    }
    .idx-hero-title { font-family: 'Newsreader', serif; font-size: 1.6rem; font-weight: 600; color: var(--text); margin-bottom: 0.35rem; }
    .idx-hero-sub { color: var(--text-dim); font-size: 0.95rem; max-width: 50rem; }

    /* Stat strip */
    .idx-stats { display: flex; gap: 0.8rem; margin-bottom: 1.2rem; flex-wrap: wrap; }
    .idx-stat { background: var(--panel); border: 1px solid var(--line); border-radius: 6px; padding: 0.5rem 0.85rem; min-width: 8rem; }
    .idx-stat-num { font-family: 'Newsreader', serif; font-size: 1.3rem; font-weight: 600; color: var(--gold); line-height: 1; }
    .idx-stat-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.66rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.2rem; }

    /* Empty state */
    .idx-empty {
        border: 1px dashed var(--gold-dim);
        border-radius: 8px;
        padding: 1.8rem 1.4rem;
        text-align: center;
        color: var(--text-dim);
        background-color: rgba(211,163,96,0.04);
    }
    .idx-empty b { color: var(--gold); }

    /* Suggested questions */
    .idx-suggest-label {
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 600;
        font-size: 0.95rem;
        color: var(--text);
        margin: 0.4rem 0 0.6rem 0;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background-color: var(--panel);
        border: 1px solid var(--line);
        border-radius: 6px;
        padding: 0.5rem 0.75rem;
        margin-bottom: 0.7rem;
    }
    [data-testid="stChatMessage"] p, [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span, [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
        color: var(--text) !important;
        opacity: 1 !important;
        font-size: 1rem;
        line-height: 1.55;
    }

    /* Footnote badges */
    .idx-footnotes { margin-top: 0.5rem; padding-top: 0.4rem; border-top: 1px dashed var(--line); }
    .idx-footnote-label {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.7rem; color: var(--text-dim);
        margin-right: 0.4rem; text-transform: uppercase; letter-spacing: 0.05em;
    }
    .idx-badge {
        display: inline-block;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.74rem;
        background-color: rgba(211,163,96,0.1);
        border: 1px solid var(--gold-dim);
        color: var(--gold);
        padding: 0.05rem 0.5rem;
        border-radius: 20px;
        margin-right: 0.3rem;
        margin-bottom: 0.3rem;
    }
    .idx-no-answer { font-family: 'IBM Plex Mono', monospace; font-size: 0.75rem; color: var(--text-dim); font-style: italic; }

    /* Bottom chat-input bar */
    [data-testid="stBottom"] > div { background-color: var(--bg); border-top: 1px solid var(--line); }
    [data-testid="stChatInput"] textarea { background-color: var(--panel-2); color: var(--text); }
    [data-testid="stChatInput"] { border: 1px solid var(--line); border-radius: 6px; }
    [data-testid="stChatInput"]:focus-within { border-color: var(--gold); box-shadow: 0 0 0 1px var(--gold); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "messages" not in st.session_state:
    st.session_state.messages = []  # each: {role, content, sources (optional)}
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []  # list of {"name", "type", "chunks"}
if "active_suggestion" not in st.session_state:
    st.session_state.active_suggestion = None


def _set_active_suggestion(question):
    st.session_state.active_suggestion = question


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return MistralAIEmbeddings()


@st.cache_resource(show_spinner=False)
def get_llm(model_name: str, temperature: float):
    return ChatMistralAI(model=model_name, temperature=temperature)


def clear_chroma_system_cache():
    """
    ChromaDB keeps a process-wide cache of client instances keyed by path
    (chromadb.api.client.SharedSystemClient). If the persist directory is
    deleted and recreated while an old client for that same path is still
    cached, ChromaDB hands back the stale client instead of a fresh one,
    which causes errors on the next add/query. Clearing this cache before
    creating a new client fixes it.
    """
    try:
        chromadb.api.client.SharedSystemClient.clear_system_cache()
    except Exception:
        pass


def load_existing_vectorstore():
    if os.path.isdir(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        clear_chroma_system_cache()
        try:
            return Chroma(
                persist_directory=PERSIST_DIR,
                embedding_function=get_embedding_model(),
                collection_name=COLLECTION_NAME,
            )
        except Exception:
            return None
    return None


if st.session_state.vectorstore is None:
    st.session_state.vectorstore = load_existing_vectorstore()


def _add_chunks_to_store(chunks):
    """Create the vectorstore on first ingest, or append to it afterwards."""
    if not chunks:
        return
    clear_chroma_system_cache()
    embedding_model = get_embedding_model()

    if st.session_state.vectorstore is None:
        st.session_state.vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embedding_model,
            persist_directory=PERSIST_DIR,
            collection_name=COLLECTION_NAME,
        )
    else:
        st.session_state.vectorstore.add_documents(chunks)


def process_documents(uploaded_files, urls, chunk_size, chunk_overlap):
    """
    Load, split, embed and persist any combination of uploaded PDFs and/or
    URLs into Chroma. Either argument may be empty — pass PDFs only, URLs
    only, or both together in a single call.

    Returns (added, skipped) where skipped items carry a human-readable
    reason, so a failure like "this PDF has no extractable text" is visible
    instead of a generic "nothing was added".
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    added = []
    skipped = []

    # --- PDFs ---------------------------------------------------------------
    for uploaded_file in uploaded_files or []:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
        except Exception as e:
            skipped.append({"name": uploaded_file.name, "reason": f"could not read the PDF ({e})"})
            os.remove(tmp_path)
            continue
        os.remove(tmp_path)

        total_chars = sum(len(d.page_content.strip()) for d in docs)
        if not docs or total_chars == 0:
            skipped.append({
                "name": uploaded_file.name,
                "reason": "no extractable text found — this is likely a scanned/image-only PDF and needs OCR first",
            })
            continue

        chunks = splitter.split_documents(docs)

        # tag with a friendly source name so citations show the real filename
        for c in chunks:
            c.metadata["source"] = uploaded_file.name
            c.metadata["doc_type"] = "pdf"

        if chunks:
            _add_chunks_to_store(chunks)
            added.append({"name": uploaded_file.name, "type": "pdf", "chunks": len(chunks)})
        else:
            skipped.append({"name": uploaded_file.name, "reason": "text was extracted but produced no chunks after splitting"})

    # --- URLs -----------------------------------------------------------------
    if urls:
        try:
            loader = WebBaseLoader(urls)
            web_docs = loader.load()
        except Exception as e:
            skipped.append({"name": ", ".join(urls), "reason": f"could not load the URL(s) ({e})"})
            web_docs = []

        if web_docs:
            chunks = splitter.split_documents(web_docs)
            for c in chunks:
                c.metadata["doc_type"] = "url"

            if chunks:
                _add_chunks_to_store(chunks)
                counts = defaultdict(int)
                for c in chunks:
                    counts[c.metadata.get("source", "unknown")] += 1
                for src, count in counts.items():
                    added.append({"name": src, "type": "url", "chunks": count})
            else:
                skipped.append({"name": ", ".join(urls), "reason": "page loaded but produced no usable text"})

    return added, skipped


def clear_database():
    """Fully wipe the vector store: collection, on-disk files, and Chroma's
    internal client cache, so a subsequent upload starts completely clean."""
    vs = st.session_state.vectorstore
    if vs is not None:
        try:
            vs.delete_collection()
        except Exception:
            pass
        st.session_state.vectorstore = None

    clear_chroma_system_cache()

    if os.path.isdir(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR, ignore_errors=True)

    clear_chroma_system_cache()

    st.session_state.messages = []
    st.session_state.processed_files = []


def get_retriever(k, fetch_k, lambda_mult):
    return st.session_state.vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
    )


def answer_question(query, k, fetch_k, lambda_mult, model_name, temperature):
    retriever = get_retriever(k, fetch_k, lambda_mult)
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)
    final_prompt = PROMPT.invoke({"context": context, "question": query})
    response = get_llm(model_name, temperature).invoke(final_prompt)

    seen = set()
    sources = []
    for doc in docs:
        source = doc.metadata.get("source")
        page = doc.metadata.get("page")
        if source is None:
            continue
        label = f"{source} — p.{page + 1}" if page is not None else source
        if label not in seen:
            seen.add(label)
            sources.append(label)

    return response.content, sources


def handle_query(query, k, fetch_k, lambda_mult, model_name, temperature):
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="🧭"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="🖋️"):
        if st.session_state.vectorstore is None:
            answer, sources = "Add a source to the archive first.", []
            st.markdown(answer)
        else:
            with st.spinner("Turning pages..."):
                try:
                    answer, sources = answer_question(query, k, fetch_k, lambda_mult, model_name, temperature)
                except Exception as e:
                    answer, sources = f"Something went wrong: {e}", []
                st.markdown(answer)
                if sources:
                    badges = " ".join(f'<span class="idx-badge">{s}</span>' for s in sources)
                    st.markdown(
                        f'<div class="idx-footnotes"><span class="idx-footnote-label">Found in:</span> {badges}</div>',
                        unsafe_allow_html=True,
                    )

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        '<div class="idx-step"><span class="idx-step-num">①</span>'
        '<span class="idx-step-title">Add your sources</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="idx-step-sub">PDF(s) and/or URLs — mix and match</div>',
        unsafe_allow_html=True,
    )

    tab_pdf, tab_url = st.tabs(["📄 PDF", "🌐 URL"])
    with tab_pdf:
        uploaded_files = st.file_uploader(
            "Drop PDF file(s)", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed"
        )
    with tab_url:
        url_text = st.text_area(
            "One URL per line",
            placeholder="https://example.com/article\nhttps://example.com/docs",
            height=100,
            label_visibility="collapsed",
        )

    col_cs, col_co = st.columns(2)
    with col_cs:
        chunk_size = st.number_input("Chunk size", min_value=200, max_value=4000, value=1000, step=100)
    with col_co:
        chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=1000, value=200, step=50)

    index_clicked = st.button("📥 Index documents", use_container_width=True)
    if index_clicked:
        urls = [u.strip() for u in url_text.splitlines() if u.strip()]
        if not uploaded_files and not urls:
            st.warning("Nothing to index — add a PDF or a URL first.")
        else:
            with st.spinner("Reading and indexing..."):
                try:
                    added, skipped = process_documents(uploaded_files, urls, chunk_size, chunk_overlap)
                    st.session_state.processed_files.extend(added)
                    if added:
                        st.success(f"Indexed {len(added)} item(s).")
                    for s in skipped:
                        st.error(f"**{s['name']}** — {s['reason']}")
                    if not added and not skipped:
                        st.warning("Nothing was added — check the source and try again.")
                except Exception as e:
                    st.error(f"Could not process: {e}")

    if st.session_state.processed_files:
        st.markdown("**Catalog**")
        for i, f in enumerate(st.session_state.processed_files, start=1):
            tag = "URL" if f.get("type") == "url" else "PDF"
            st.markdown(
                f"""
                <div class="idx-card">
                    <span class="idx-card-stamp">indexed</span>
                    <span class="idx-card-num">{i:03d} · {tag}</span>
                    <div class="idx-card-name">{f['name']}</div>
                    <div class="idx-card-meta">{f['chunks']} passages</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.divider()

    st.markdown(
        '<div class="idx-step"><span class="idx-step-num">②</span>'
        '<span class="idx-step-title">Analysis settings</span></div>',
        unsafe_allow_html=True,
    )

    model_name = st.selectbox("Model", MODEL_OPTIONS, index=0)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    k = st.slider("Chunks returned (k)", 1, 30, 10, 1)
    fetch_k = st.slider("Candidates scanned (fetch_k)", 10, 200, 100, 10)
    lambda_mult = st.slider("Relevance ↔ Diversity", 0.0, 1.0, 0.5, 0.05)

    st.markdown(f'<span class="idx-tag">Vector store: {PERSIST_DIR}</span>', unsafe_allow_html=True)

    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col_b:
        confirm_wipe = st.checkbox("Confirm wipe", label_visibility="collapsed", help="Tick to enable Reset archive")
        if st.button("Reset archive", use_container_width=True, disabled=not confirm_wipe):
            with st.spinner("Clearing the archive..."):
                try:
                    clear_database()
                    st.success("Archive cleared.")
                    st.rerun()
                except Exception as e:
                    st.error(f"Could not clear: {e}")

    st.divider()
    ready = st.session_state.vectorstore is not None
    status_label = "Ready" if ready else "Empty"
    dot_color = "#D3A360" if ready else "#8B93A1"
    st.markdown(
        f'<span class="idx-status"><span class="idx-status-dot" style="background:{dot_color}"></span>{status_label}</span>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# A suggestion click sets st.session_state.active_suggestion via its on_click
# callback; pick it up here and answer it in this same run.
active_query = None
if st.session_state.active_suggestion:
    active_query = st.session_state.active_suggestion
    st.session_state.active_suggestion = None

st.markdown(
    """
    <div class="idx-hero-card">
        <div class="idx-hero-eyebrow">Reading room · grounded in your sources only</div>
        <div class="idx-hero-title">CourseMate-Ai</div>
        <div class="idx-hero-sub">Upload PDFs and/or URLs, then ask anything about them —
        every answer is backed by a cited passage.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if st.session_state.processed_files:
    pdf_count = sum(1 for f in st.session_state.processed_files if f.get("type") != "url")
    url_count = sum(1 for f in st.session_state.processed_files if f.get("type") == "url")
    chunk_count = sum(f.get("chunks", 0) for f in st.session_state.processed_files)
    st.markdown(
        f"""
        <div class="idx-stats">
            <div class="idx-stat"><div class="idx-stat-num">{pdf_count}</div><div class="idx-stat-label">PDFs</div></div>
            <div class="idx-stat"><div class="idx-stat-num">{url_count}</div><div class="idx-stat-label">URLs</div></div>
            <div class="idx-stat"><div class="idx-stat-num">{chunk_count}</div><div class="idx-stat-label">Passages indexed</div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

if st.session_state.vectorstore is None:
    st.markdown(
        """
        <div class="idx-empty">
            <b>No sources on file yet.</b><br/>
            Add a PDF or a URL in the sidebar and click <b>Index documents</b> to begin.
        </div>
        """,
        unsafe_allow_html=True,
    )
else:
    if not st.session_state.messages and active_query is None:
        st.markdown('<div class="idx-suggest-label">Suggested questions</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        for i, question in enumerate(SUGGESTED_QUESTIONS):
            with cols[i % 3]:
                st.button(
                    question,
                    use_container_width=True,
                    key=f"suggest_{i}",
                    on_click=_set_active_suggestion,
                    args=(question,),
                )

    for msg in st.session_state.messages:
        avatar = "🧭" if msg["role"] == "user" else "🖋️"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["content"])
            sources = msg.get("sources")
            if sources:
                badges = " ".join(f'<span class="idx-badge">{s}</span>' for s in sources)
                st.markdown(
                    f'<div class="idx-footnotes"><span class="idx-footnote-label">Found in:</span> {badges}</div>',
                    unsafe_allow_html=True,
                )
            elif msg["role"] == "assistant" and "sources" in msg:
                st.markdown('<div class="idx-no-answer">No matching passage in the archive.</div>', unsafe_allow_html=True)

    if active_query:
        handle_query(active_query, k, fetch_k, lambda_mult, model_name, temperature)

    typed_query = st.chat_input("Ask something about your documents...")
    if typed_query:
        handle_query(typed_query, k, fetch_k, lambda_mult, model_name, temperature)
