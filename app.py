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
PERSIST_DIR = "chroma_db"           # single, consistent directory name everywhere
COLLECTION_NAME = "rag_collection"  # fixed collection name

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
# Theme — "The Index": a research desk, not a chat app
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,500;0,6..72,600;1,6..72,500&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

    :root {
        --paper: #F3ECDD;
        --card: #FFFDF7;
        --ink: #23201B;
        --ink-soft: #6B6355;
        --rust: #BB4430;
        --rust-soft: rgba(187,68,48,0.1);
        --sage: #5C7A5A;
        --sage-soft: rgba(92,122,90,0.12);
        --line: #D8CBAA;
    }

    [data-testid="stAppViewContainer"] { background-color: var(--paper); }
    [data-testid="stHeader"] { background-color: transparent; }
    html, body, [class*="css"] {
        font-family: 'IBM Plex Sans', sans-serif;
        color: var(--ink);
    }
    /* tighten default top padding */
    .block-container { padding-top: 2.2rem; }

    /* ---------- Sidebar: "The Desk" ---------- */
    [data-testid="stSidebar"] {
        background-color: var(--card);
        border-right: 1px solid var(--line);
    }
    [data-testid="stSidebar"] * { color: var(--ink) !important; }
    .idx-desk-title {
        font-family: 'Newsreader', serif;
        font-size: 1.3rem;
        font-weight: 600;
        letter-spacing: 0.01em;
        margin-bottom: 0.15rem;
    }
    .idx-desk-sub {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.72rem;
        color: var(--ink-soft) !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-bottom: 1rem;
    }
    [data-testid="stSidebar"] hr { border-color: var(--line); }

    /* Tabs (PDF / URL input) */
    [data-testid="stSidebar"] .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        border-bottom: 1px solid var(--line);
    }
    [data-testid="stSidebar"] .stTabs [data-baseweb="tab"] {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: var(--ink-soft) !important;
        padding: 0.4rem 0.6rem;
    }
    [data-testid="stSidebar"] .stTabs [aria-selected="true"] {
        color: var(--rust) !important;
        border-bottom: 2px solid var(--rust);
    }

    /* Hero */
    .idx-hero-eyebrow {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: var(--rust);
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.3rem;
    }
    .idx-hero-title {
        font-family: 'Newsreader', serif;
        font-size: 2.7rem;
        font-weight: 600;
        color: var(--ink);
        margin-bottom: 0.2rem;
        letter-spacing: -0.01em;
        line-height: 1.1;
    }
    .idx-hero-sub {
        color: var(--ink-soft);
        font-size: 1.02rem;
        margin-bottom: 1.4rem;
        max-width: 46rem;
    }

    /* Stat strip */
    .idx-stats { display: flex; gap: 0.9rem; margin-bottom: 1.6rem; flex-wrap: wrap; }
    .idx-stat {
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 6px;
        padding: 0.55rem 0.9rem;
        min-width: 8.5rem;
    }
    .idx-stat-num {
        font-family: 'Newsreader', serif;
        font-size: 1.4rem;
        font-weight: 600;
        color: var(--ink);
        line-height: 1;
    }
    .idx-stat-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        color: var(--ink-soft);
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.2rem;
    }

    /* ---------- Index cards (catalog entries) ---------- */
    .idx-card {
        position: relative;
        background: var(--card);
        border: 1px solid var(--line);
        border-radius: 3px;
        padding: 0.65rem 0.75rem 0.55rem 0.75rem;
        margin-bottom: 0.6rem;
        background-image: repeating-linear-gradient(
            to bottom, transparent, transparent 20px, var(--line) 20px, var(--line) 21px
        );
        background-position: 0 34px;
    }
    .idx-card::before {
        content: "";
        position: absolute;
        top: 0; left: 12px; right: 12px;
        height: 1px;
        background: repeating-linear-gradient(to right, var(--ink-soft) 0, var(--ink-soft) 3px, transparent 3px, transparent 7px);
        opacity: 0.4;
    }
    .idx-card-num {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        color: var(--rust);
        letter-spacing: 0.04em;
    }
    .idx-card-stamp {
        float: right;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.62rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--sage);
        border: 1px solid var(--sage);
        border-radius: 20px;
        padding: 0.05rem 0.45rem;
        transform: rotate(-2deg);
    }
    .idx-card-name {
        font-weight: 600;
        font-size: 0.9rem;
        margin-top: 0.2rem;
        word-break: break-word;
    }
    .idx-card-meta {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.68rem;
        color: var(--ink-soft);
        margin-top: 0.1rem;
    }

    /* Status pill (bottom of sidebar) */
    .idx-status {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        padding: 0.25rem 0.7rem;
        border-radius: 20px;
        border: 1px solid var(--line);
    }
    .idx-status-dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; }

    /* Buttons */
    .stButton>button {
        background-color: var(--rust);
        color: var(--card) !important;
        border: none;
        border-radius: 3px;
        font-family: 'IBM Plex Sans', sans-serif;
        font-weight: 600;
        font-size: 0.9rem;
        transition: opacity 0.15s ease;
    }
    .stButton>button:hover { opacity: 0.88; color: var(--card) !important; }
    .stButton>button:disabled { background-color: rgba(187,68,48,0.25); }
    [data-testid="stSidebar"] .stButton>button * { color: var(--card) !important; }

    /* Secondary (housekeeping) buttons */
    [data-testid="stSidebar"] div[data-testid="stVerticalBlock"] .stButton>button {
        background-color: var(--ink);
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background-color: var(--card);
        border: 1px solid var(--line);
        border-radius: 4px;
        padding: 0.5rem 0.75rem;
        margin-bottom: 0.7rem;
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
        color: var(--ink) !important;
        opacity: 1 !important;
        font-size: 1rem;
        line-height: 1.55;
    }

    /* Footnote badges under an answer */
    .idx-footnotes { margin-top: 0.5rem; padding-top: 0.4rem; border-top: 1px dashed var(--line); }
    .idx-footnote-label {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.7rem;
        color: var(--ink-soft);
        margin-right: 0.4rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .idx-badge {
        display: inline-block;
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.74rem;
        background-color: var(--rust-soft);
        border: 1px solid var(--rust);
        color: var(--rust);
        padding: 0.05rem 0.5rem;
        border-radius: 20px;
        margin-right: 0.3rem;
        margin-bottom: 0.3rem;
    }
    .idx-no-answer {
        font-family: 'IBM Plex Mono', monospace;
        font-size: 0.75rem;
        color: var(--ink-soft);
        font-style: italic;
    }

    /* Empty state */
    .idx-empty {
        border: 1px dashed var(--rust);
        border-radius: 4px;
        padding: 2rem 1.4rem;
        text-align: center;
        color: var(--ink-soft);
        background-color: rgba(187,68,48,0.04);
    }
    .idx-empty-mark {
        font-family: 'Newsreader', serif;
        font-size: 2rem;
        color: var(--rust);
        margin-bottom: 0.3rem;
    }
    .idx-empty-title {
        font-family: 'Newsreader', serif;
        font-size: 1.25rem;
        color: var(--ink);
        margin-bottom: 0.3rem;
    }
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


@st.cache_resource(show_spinner=False)
def get_embedding_model():
    return MistralAIEmbeddings()


@st.cache_resource(show_spinner=False)
def get_llm():
    return ChatMistralAI(model="mistral-small-2506")


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


def process_documents(uploaded_files, urls):
    """
    Load, split, embed and persist any combination of uploaded PDFs and/or
    URLs into Chroma. Either argument may be empty — pass PDFs only, URLs
    only, or both together in a single call.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    added = []

    # --- PDFs -------------------------------------------------------------
    for uploaded_file in uploaded_files or []:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            chunks = splitter.split_documents(docs)
        finally:
            os.remove(tmp_path)

        # tag with a friendly source name so citations show the real filename
        for c in chunks:
            c.metadata["source"] = uploaded_file.name
            c.metadata["doc_type"] = "pdf"

        if chunks:
            _add_chunks_to_store(chunks)
            added.append({"name": uploaded_file.name, "type": "pdf", "chunks": len(chunks)})

    # --- URLs ---------------------------------------------------------------
    if urls:
        try:
            loader = WebBaseLoader(urls)
            web_docs = loader.load()
        except Exception as e:
            st.error(f"Could not load one or more URLs: {e}")
            web_docs = []

        if web_docs:
            chunks = splitter.split_documents(web_docs)
            for c in chunks:
                c.metadata["doc_type"] = "url"

            if chunks:
                _add_chunks_to_store(chunks)

                # group chunk counts per URL for the catalog display
                counts = defaultdict(int)
                for c in chunks:
                    counts[c.metadata.get("source", "unknown")] += 1
                for src, count in counts.items():
                    added.append({"name": src, "type": "url", "chunks": count})

    return added


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


def get_retriever():
    return st.session_state.vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 10, "fetch_k": 100, "lambda_mult": 0.5},
    )


def answer_question(query: str):
    retriever = get_retriever()
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)
    final_prompt = PROMPT.invoke({"context": context, "question": query})
    response = get_llm().invoke(final_prompt)

    # Build citation badges: "filename — p.N" for PDFs, bare URL for web pages
    seen = set()
    sources = []
    for doc in docs:
        source = doc.metadata.get("source")
        page = doc.metadata.get("page")
        if source is None:
            continue
        if page is not None:
            label = f"{source} — p.{page + 1}"
        else:
            label = source
        if label not in seen:
            seen.add(label)
            sources.append(label)

    return response.content, sources


# ---------------------------------------------------------------------------
# Sidebar — "The Archive"
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="idx-desk-title">The Desk</div>', unsafe_allow_html=True)
    st.markdown('<div class="idx-desk-sub">Add sources · then ask</div>', unsafe_allow_html=True)

    tab_pdf, tab_url = st.tabs(["📄 PDF", "🌐 URL"])

    with tab_pdf:
        uploaded_files = st.file_uploader(
            "Drop PDF file(s)", type=["pdf"], accept_multiple_files=True,
            label_visibility="collapsed",
        )
        pdf_clicked = st.button("Catalogue PDF(s)", use_container_width=True, key="pdf_go")

    with tab_url:
        url_text = st.text_area(
            "One URL per line",
            placeholder="https://example.com/article\nhttps://example.com/docs",
            height=110,
            label_visibility="collapsed",
        )
        url_clicked = st.button("Catalogue URL(s)", use_container_width=True, key="url_go")

    if pdf_clicked or url_clicked:
        urls = [u.strip() for u in url_text.splitlines() if u.strip()] if url_clicked else []
        files = uploaded_files if pdf_clicked else []
        if not files and not urls:
            st.warning("Nothing to add — choose a PDF or paste a URL first.")
        else:
            with st.spinner("Reading and indexing..."):
                try:
                    added = process_documents(files, urls)
                    st.session_state.processed_files.extend(added)
                    if added:
                        st.success(f"Catalogued {len(added)} item(s).")
                    else:
                        st.warning("Nothing was added — check the source and try again.")
                except Exception as e:
                    st.error(f"Could not process: {e}")

    st.divider()
    st.markdown("**Catalog**")
    if st.session_state.processed_files:
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
    else:
        st.caption("Nothing catalogued yet.")

    st.divider()
    st.markdown("**Housekeeping**")

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    confirm_wipe = st.checkbox("Confirm permanent deletion of all catalogued documents")
    if st.button("Clear entire archive", use_container_width=True, disabled=not confirm_wipe):
        with st.spinner("Clearing the archive..."):
            try:
                clear_database()
                st.success("Archive cleared. Add a new source to begin again.")
                st.rerun()
            except Exception as e:
                st.error(f"Could not clear: {e}")

    st.divider()
    ready = st.session_state.vectorstore is not None
    status_label = "Ready" if ready else "Empty"
    dot_color = "#5C7A5A" if ready else "#BB4430"
    st.markdown(
        f'<span class="idx-status"><span class="idx-status-dot" '
        f'style="background:{dot_color}"></span>{status_label}</span>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main — "The Reading Room"
# ---------------------------------------------------------------------------
st.markdown('<div class="marg-hero-title">CourseMate-Ai</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="marg-hero-sub">Ask questions. Answers stay within what your PDFs and URLs actually say — '
    "and every answer cites the source it came from.</div>",
    unsafe_allow_html=True,
)

if st.session_state.vectorstore is None:
    st.markdown(
        """
        <div class="marg-empty">
            <div class="marg-empty-title">The archive is empty</div>
            Add a PDF and/or a URL from The Archive panel on the left to start a conversation.
        </div>
        """,
        unsafe_allow_html=True,
    )

for msg in st.session_state.messages:
    avatar = "🧭" if msg["role"] == "user" else "🖋️"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        sources = msg.get("sources")
        if sources:
            badges = "".join(f'<span class="marg-badge">{s}</span>' for s in sources)
            st.markdown(
                f'<div class="marg-footnotes"><span class="marg-footnote-label">Found in</span>{badges}</div>',
                unsafe_allow_html=True,
            )
        elif msg["role"] == "assistant" and "sources" in msg:
            st.markdown(
                '<div class="marg-no-answer">No matching passage in the archive.</div>',
                unsafe_allow_html=True,
            )

query = st.chat_input("Ask something about your documents...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="🧭"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="🖋️"):
        if st.session_state.vectorstore is None:
            answer, sources = "Add a document to the archive first.", []
            st.markdown(answer)
        else:
            with st.spinner("Turning pages..."):
                try:
                    answer, sources = answer_question(query)
                except Exception as e:
                    answer, sources = f"Something went wrong: {e}", []
                st.markdown(answer)
                if sources:
                    badges = "".join(f'<span class="marg-badge">{s}</span>' for s in sources)
                    st.markdown(
                        f'<div class="marg-footnotes"><span class="marg-footnote-label">Found in</span>{badges}</div>',
                        unsafe_allow_html=True,
                    )

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
