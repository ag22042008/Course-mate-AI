import os
import shutil
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
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
# Theme — "Marginalia": answers as margin notes on your own document
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=Source+Sans+3:wght@400;600&family=JetBrains+Mono:wght@400;600&display=swap');

    :root {
        --ink: #1C2B4A;
        --parchment: #F7F2E9;
        --brass: #A9814B;
        --brass-dark: #8C6A3D;
        --charcoal: #2A2A28;
        --slate: #64707D;
        --rule: #D9CFBB;
    }

    [data-testid="stAppViewContainer"] {
        background-color: var(--parchment);
    }

    [data-testid="stHeader"] {
        background-color: transparent;
    }

    html, body, [class*="css"] {
        font-family: 'Source Sans 3', sans-serif;
        color: var(--charcoal);
    }

    /* Sidebar = "The Archive" */
    [data-testid="stSidebar"] {
        background-color: var(--ink);
        color: var(--parchment);
    }
    [data-testid="stSidebar"] * {
        color: var(--parchment) !important;
    }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        font-family: 'Fraunces', serif;
        letter-spacing: 0.03em;
    }
    [data-testid="stSidebar"] hr {
        border-color: rgba(247,242,233,0.25);
    }

    /* Hero */
    .marg-hero-title {
        font-family: 'Fraunces', serif;
        font-size: 2.6rem;
        font-weight: 600;
        color: var(--ink);
        margin-bottom: 0.1rem;
        letter-spacing: -0.01em;
    }
    .marg-hero-sub {
        font-family: 'Source Sans 3', sans-serif;
        color: var(--slate);
        font-size: 1.02rem;
        margin-bottom: 1.6rem;
        border-left: 3px solid var(--brass);
        padding-left: 0.7rem;
    }

    /* Catalog entries in the sidebar */
    .marg-catalog-entry {
        border: 1px solid rgba(247,242,233,0.25);
        border-left: 3px solid var(--brass);
        padding: 0.5rem 0.7rem;
        margin-bottom: 0.5rem;
        border-radius: 2px;
        font-size: 0.88rem;
    }
    .marg-catalog-num {
        font-family: 'JetBrains Mono', monospace;
        color: var(--brass);
        font-size: 0.75rem;
        display: block;
        margin-bottom: 0.15rem;
    }
    .marg-catalog-name {
        font-weight: 600;
        word-break: break-word;
    }
    .marg-catalog-meta {
        font-family: 'JetBrains Mono', monospace;
        color: rgba(247,242,233,0.65);
        font-size: 0.72rem;
    }

    /* Status pill */
    .marg-status {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.78rem;
        padding: 0.15rem 0.6rem;
        border-radius: 20px;
        border: 1px solid var(--brass);
    }

    /* Buttons */
    .stButton>button {
        background-color: var(--brass);
        color: var(--parchment);
        border: none;
        border-radius: 3px;
        font-family: 'Source Sans 3', sans-serif;
        font-weight: 600;
        transition: background-color 0.15s ease;
    }
    .stButton>button:hover {
        background-color: var(--brass-dark);
        color: var(--parchment);
    }
    .stButton>button:disabled {
        background-color: rgba(169,129,79,0.3);
        color: rgba(247,242,233,0.5);
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background-color: #FFFFFF;
        border: 1px solid var(--rule);
        border-left: 3px solid var(--brass);
        border-radius: 4px;
        padding: 0.4rem 0.6rem;
        margin-bottom: 0.6rem;
    }
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] {
        color: var(--charcoal) !important;
        opacity: 1 !important;
        font-size: 1rem;
        line-height: 1.55;
    }

    /* Footnote badges under an answer */
    .marg-footnotes {
        margin-top: 0.5rem;
        padding-top: 0.4rem;
        border-top: 1px dashed var(--rule);
    }
    .marg-footnote-label {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.72rem;
        color: var(--slate);
        margin-right: 0.4rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .marg-badge {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        background-color: var(--parchment);
        border: 1px solid var(--brass);
        color: var(--brass-dark);
        padding: 0.05rem 0.5rem;
        border-radius: 20px;
        margin-right: 0.3rem;
    }
    .marg-no-answer {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: var(--slate);
        font-style: italic;
    }

    /* Empty state */
    .marg-empty {
        border: 1px dashed var(--brass);
        border-radius: 4px;
        padding: 1.4rem;
        text-align: center;
        color: var(--slate);
        font-family: 'Source Sans 3', sans-serif;
        background-color: rgba(169,129,79,0.05);
    }
    .marg-empty-title {
        font-family: 'Fraunces', serif;
        font-size: 1.2rem;
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
    st.session_state.messages = []  # each: {role, content, pages (optional)}
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []  # list of {"name", "chunks"}


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


def process_pdfs(uploaded_files):
    """Load, split, embed and persist the uploaded PDFs into Chroma."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    added = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        try:
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            chunks = splitter.split_documents(docs)
        finally:
            os.remove(tmp_path)

        if not chunks:
            continue

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

        added.append({"name": uploaded_file.name, "chunks": len(chunks)})

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

    pages = sorted(
        {
            doc.metadata.get("page")
            for doc in docs
            if doc.metadata.get("page") is not None
        }
    )
    return response.content, pages


# ---------------------------------------------------------------------------
# Sidebar — "The Archive"
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🗂️ The Archive")
    st.caption("Every document you add is catalogued and indexed here.")

    uploaded_files = st.file_uploader(
        "Add document(s)", type=["pdf"], accept_multiple_files=True, label_visibility="visible"
    )

    process_clicked = st.button("📥 Catalogue PDF(s)", use_container_width=True)

    if process_clicked:
        if not uploaded_files:
            st.warning("Choose at least one PDF first.")
        else:
            with st.spinner("Reading and indexing..."):
                try:
                    added = process_pdfs(uploaded_files)
                    st.session_state.processed_files.extend(added)
                    st.success(f"Catalogued {len(added)} document(s).")
                except Exception as e:
                    st.error(f"Could not process: {e}")

    st.divider()
    st.markdown("### Catalog")
    if st.session_state.processed_files:
        for i, f in enumerate(st.session_state.processed_files, start=1):
            st.markdown(
                f"""
                <div class="marg-catalog-entry">
                    <span class="marg-catalog-num">No. {i:03d}</span>
                    <span class="marg-catalog-name">{f['name']}</span><br/>
                    <span class="marg-catalog-meta">{f['chunks']} passages indexed</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
    else:
        st.caption("The archive is empty.")

    st.divider()
    st.markdown("### Housekeeping")

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    confirm_wipe = st.checkbox("Confirm permanent deletion of all catalogued documents")
    if st.button("🗑️ Clear entire archive", use_container_width=True, disabled=not confirm_wipe):
        with st.spinner("Clearing the archive..."):
            try:
                clear_database()
                st.success("Archive cleared. Add a new document to begin again.")
                st.rerun()
            except Exception as e:
                st.error(f"Could not clear: {e}")

    st.divider()
    ready = st.session_state.vectorstore is not None
    status_label = "Ready" if ready else "Empty"
    st.markdown(
        f'<span class="marg-status">● {status_label}</span>',
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------------------------
# Main — "The Reading Room"
# ---------------------------------------------------------------------------
st.markdown('<div class="marg-hero-title">CourseMate-Ai</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="marg-hero-sub">Ask questions. Answers stay within what the document actually says — '
    "and every answer cites the page it came from.</div>",
    unsafe_allow_html=True,
)

if st.session_state.vectorstore is None:
    st.markdown(
        """
        <div class="marg-empty">
            <div class="marg-empty-title">The archive is empty</div>
            Add a PDF from The Archive panel on the left to start a conversation.
        </div>
        """,
        unsafe_allow_html=True,
    )

for msg in st.session_state.messages:
    avatar = "🧭" if msg["role"] == "user" else "🖋️"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        pages = msg.get("pages")
        if pages:
            badges = "".join(f'<span class="marg-badge">p. {p + 1}</span>' for p in pages)
            st.markdown(
                f'<div class="marg-footnotes"><span class="marg-footnote-label">Found on</span>{badges}</div>',
                unsafe_allow_html=True,
            )
        elif msg["role"] == "assistant" and "pages" in msg:
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
            answer, pages = "Add a document to the archive first.", []
            st.markdown(answer)
        else:
            with st.spinner("Turning pages..."):
                try:
                    answer, pages = answer_question(query)
                except Exception as e:
                    answer, pages = f"Something went wrong: {e}", []
                st.markdown(answer)
                if pages:
                    badges = "".join(f'<span class="marg-badge">p. {p + 1}</span>' for p in pages)
                    st.markdown(
                        f'<div class="marg-footnotes"><span class="marg-footnote-label">Found on</span>{badges}</div>',
                        unsafe_allow_html=True,
                    )

    st.session_state.messages.append({"role": "assistant", "content": answer, "pages": pages})
