import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

st.set_page_config(page_title="CourseMate AI", page_icon="📚", layout="wide")

PERSIST_DIR = "chroma_db"


def get_mistral_api_key():
    key = None
    try:
        key = st.secrets.get("MISTRAL_API_KEY")
    except Exception:
        key = None
    if not key:
        key = os.environ.get("MISTRAL_API_KEY")
    if key:
        key = key.strip()
        os.environ["MISTRAL_API_KEY"] = key
    return key


MISTRAL_API_KEY = get_mistral_api_key()

if not MISTRAL_API_KEY:
    st.error(
        "MISTRAL_API_KEY is missing. Locally: add it to a .env file. "
        "On Streamlit Cloud: go to Manage app > Settings > Secrets and add "
        "MISTRAL_API_KEY = your_key, then reboot the app."
    )
    st.stop()

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

# ==================================================================================
# STYLE — "The Reading Room": an ink-and-paper archive, sources shown as catalog slips
# ==================================================================================
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,500;0,600;1,500&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --ink: #1B2430;
    --ink-soft: #232E3D;
    --paper: #EDE7D9;
    --paper-dim: #E2DBC9;
    --brass: #B08D57;
    --moss: #5C7A6A;
    --chalk: #F2EFE6;
    --rust: #A6452C;
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(180deg, var(--ink) 0%, var(--ink-soft) 100%);
    color: var(--chalk);
}

#MainMenu, footer, header { visibility: hidden; }

.rr-header {
    border: 1px solid rgba(176, 141, 87, 0.4);
    border-radius: 2px;
    padding: 28px 32px;
    margin-bottom: 28px;
    background: repeating-linear-gradient(180deg, rgba(176,141,87,0.03) 0px, rgba(176,141,87,0.03) 2px, transparent 2px, transparent 4px), var(--ink-soft);
    position: relative;
}
.rr-header::before {
    content: "";
    position: absolute;
    inset: 6px;
    border: 1px solid rgba(176, 141, 87, 0.25);
    pointer-events: none;
}
.rr-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    font-size: 11px;
    color: var(--brass);
    margin-bottom: 6px;
}
.rr-title {
    font-family: 'Lora', serif;
    font-weight: 600;
    font-size: 34px;
    color: var(--chalk);
    margin: 0;
}
.rr-sub {
    font-family: 'Inter', sans-serif;
    font-size: 14px;
    color: rgba(242,239,230,0.55);
    margin-top: 6px;
}

section[data-testid="stSidebar"] {
    background: var(--ink-soft);
    border-right: 1px solid rgba(176,141,87,0.25);
}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
    font-family: 'Lora', serif;
    color: var(--brass);
}
section[data-testid="stSidebar"] .stCaption, section[data-testid="stSidebar"] p {
    color: rgba(242,239,230,0.7) !important;
}
section[data-testid="stSidebar"] label p {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: rgba(242,239,230,0.75) !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

div[data-baseweb="slider"] div[role="slider"] { background-color: var(--brass) !important; }
div[data-testid="stSliderThumbValue"] { color: var(--brass) !important; }

section[data-testid="stSidebar"] button {
    background: transparent !important;
    border: 1px solid var(--brass) !important;
    color: var(--brass) !important;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    letter-spacing: 0.05em;
    border-radius: 2px !important;
}
section[data-testid="stSidebar"] button:hover {
    background: rgba(176,141,87,0.12) !important;
}

div[data-testid="stFileUploaderDropzone"] {
    background: rgba(176,141,87,0.06) !important;
    border: 1px dashed rgba(176,141,87,0.5) !important;
}

div[data-testid="stChatMessage"] { background: transparent; padding: 0; }
div[data-testid="stChatMessageContent"] { border-radius: 4px; padding: 4px 2px; }

div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarUser"]) div[data-testid="stChatMessageContent"] {
    background: rgba(176, 141, 87, 0.08);
    border: 1px solid rgba(176, 141, 87, 0.3);
    border-radius: 4px;
    padding: 14px 18px;
    font-family: 'Inter', sans-serif;
    color: var(--chalk);
}

div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) div[data-testid="stChatMessageContent"] {
    background: var(--paper);
    border: 1px solid var(--paper-dim);
    border-radius: 4px;
    padding: 16px 20px;
    font-family: 'Lora', serif;
    color: var(--ink);
    font-size: 16px;
    line-height: 1.6;
}
div[data-testid="stChatMessage"]:has(div[data-testid="stChatMessageAvatarAssistant"]) div[data-testid="stChatMessageContent"] p {
    color: var(--ink) !important;
}

.rr-catalog-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--moss);
    margin: 10px 0 6px 2px;
}
.rr-slip {
    background: var(--paper);
    border: 1px solid var(--paper-dim);
    border-left: 3px solid var(--brass);
    border-radius: 2px;
    padding: 12px 14px;
    margin-bottom: 8px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12.5px;
    color: var(--ink);
    line-height: 1.55;
}
.rr-slip-num {
    display: inline-block;
    background: var(--brass);
    color: var(--ink);
    font-weight: 700;
    font-size: 11px;
    padding: 1px 7px;
    border-radius: 2px;
    margin-right: 8px;
}
.rr-slip-meta { color: var(--moss); font-size: 11px; margin-top: 6px; opacity: 0.85; }

div[data-testid="stExpander"] {
    background: transparent;
    border: 1px solid rgba(176,141,87,0.35);
    border-radius: 3px;
}
div[data-testid="stExpander"] summary {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 12px;
    color: var(--brass);
    letter-spacing: 0.04em;
}

div[data-testid="stChatInput"] {
    background: var(--ink-soft);
    border: 1px solid rgba(176,141,87,0.4);
    border-radius: 4px;
}
div[data-testid="stChatInput"] textarea {
    color: var(--chalk) !important;
    font-family: 'Inter', sans-serif !important;
}

.rr-project-tag {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--chalk);
    background: var(--brass);
    display: inline-block;
    padding: 4px 14px;
    border-radius: 2px;
    margin-bottom: 10px;
}

.rr-rule { border: none; border-top: 1px dashed rgba(176,141,87,0.3); margin: 18px 0; }

.rr-empty {
    border: 1px dashed rgba(176,141,87,0.4);
    border-radius: 4px;
    padding: 40px 24px;
    text-align: center;
    color: rgba(242,239,230,0.6);
    font-family: 'Lora', serif;
    font-size: 17px;
    margin-top: 10px;
}
</style>
""",
    unsafe_allow_html=True,
)


# ---------------- Cached / persistent resources ----------------
@st.cache_resource
def get_embedding_model():
    return MistralAIEmbeddings(api_key=MISTRAL_API_KEY)


def get_llm(model_name):
    return ChatMistralAI(model=model_name, api_key=MISTRAL_API_KEY)


def index_files(uploaded_files, chunk_size, chunk_overlap):
    """Chunk uploaded PDFs, embed them, and store/update them in the Chroma vector store."""
    embedding_model = get_embedding_model()
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    all_chunks = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for uploaded_file in uploaded_files:
            tmp_path = os.path.join(tmp_dir, uploaded_file.name)
            with open(tmp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            loader = PyPDFLoader(tmp_path)
            docs = loader.load()
            # Fix source metadata to the real filename rather than the temp path
            for d in docs:
                d.metadata["source"] = uploaded_file.name
            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)

    if not all_chunks:
        return 0

    if not os.path.exists(PERSIST_DIR):
    os.makedirs(PERSIST_DIR)

vectorstore = Chroma(
    persist_directory=PERSIST_DIR,
    embedding_function=embedding_model
)
    vectorstore.add_documents(all_chunks)
    st.session_state.vectorstore_ready = True
    st.session_state.indexed_files = st.session_state.get("indexed_files", set()) | {
        f.name for f in uploaded_files
    }
    return len(all_chunks)


def load_existing_vectorstore():
    """Reuse a chroma_db folder from a previous session/run, if present."""
    if os.path.isdir(PERSIST_DIR) and os.listdir(PERSIST_DIR):
        embedding_model = get_embedding_model()
        vs = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_model)
        try:
            if vs._collection.count() > 0:
                return True
        except Exception:
            pass
    return False


def get_retriever(k, fetch_k, lambda_mult):
    embedding_model = get_embedding_model()
    vectorstore = Chroma(persist_directory=PERSIST_DIR, embedding_function=embedding_model)
    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
    )


def answer_query(query, retriever, llm):
    docs = retriever.invoke(query)
    context = "\n\n".join(doc.page_content for doc in docs)
    final_prompt = PROMPT.invoke({"context": context, "question": query})
    response = llm.invoke(final_prompt)
    return response.content, docs


def render_slips(docs):
    st.markdown('<div class="rr-catalog-label">◆ Consulted passages</div>', unsafe_allow_html=True)
    with st.expander(f"Open catalog drawer ({len(docs)} slips)"):
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get("page", "—")
            source = doc.metadata.get("source", "document")
            snippet = doc.page_content.strip().replace("\n", " ")
            if len(snippet) > 480:
                snippet = snippet[:480].rsplit(" ", 1)[0] + " …"
            st.markdown(
                f"""<div class="rr-slip">
                <span class="rr-slip-num">{i:02d}</span>{snippet}
                <div class="rr-slip-meta">source: {source} · page {page}</div>
                </div>""",
                unsafe_allow_html=True,
            )


# ---------------- Session state init ----------------
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vectorstore_ready" not in st.session_state:
    st.session_state.vectorstore_ready = load_existing_vectorstore()
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()

# ---------------- Sidebar ----------------
with st.sidebar:
    st.markdown("## ① Add documents")
    uploaded_files = st.file_uploader(
        "Upload PDF(s) to add to the archive",
        type=["pdf"],
        accept_multiple_files=True,
    )
    chunk_size = st.number_input("Chunk size", min_value=200, max_value=4000, value=1000, step=100)
    chunk_overlap = st.number_input("Chunk overlap", min_value=0, max_value=1000, value=200, step=50)

    if st.button("📥 Index documents", disabled=not uploaded_files):
        with st.spinner("Splitting, embedding, and filing into the archive..."):
            n_chunks = index_files(uploaded_files, chunk_size, chunk_overlap)
        st.success(f"Indexed {n_chunks} chunks from {len(uploaded_files)} file(s).")

    if st.session_state.indexed_files:
        st.caption("Filed in this session: " + ", ".join(sorted(st.session_state.indexed_files)))

    st.markdown('<hr class="rr-rule">', unsafe_allow_html=True)
    st.markdown("## ② Retrieval settings")
    model_name = st.selectbox("Model", ["mistral-small-2506", "mistral-large-latest"], index=0)
    k = st.slider("Slips returned (k)", 1, 20, 10)
    fetch_k = st.slider("Candidates scanned (fetch_k)", 10, 200, 100, step=10)
    lambda_mult = st.slider("Relevance ↔ Diversity", 0.0, 1.0, 0.5)

    st.markdown('<hr class="rr-rule">', unsafe_allow_html=True)
    st.caption(f"ARCHIVE: `{PERSIST_DIR}`")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear chat"):
            st.session_state.messages = []
            st.rerun()
    with col2:
    if st.button("Reset archive"):
        import shutil
        import os

        try:
            shutil.rmtree(PERSIST_DIR, ignore_errors=True)
        except:
            pass

        os.makedirs(PERSIST_DIR, exist_ok=True)

        st.cache_resource.clear()      # <-- ADD THIS
        st.session_state.vectorstore_ready = False
        st.session_state.indexed_files = set()
        st.session_state.messages = []

        st.rerun()

# ---------------- Header ----------------
st.markdown('<div class="rr-project-tag">CourseMate AI</div>', unsafe_allow_html=True)
st.markdown(
    """
<div class="rr-header">
    <div class="rr-eyebrow">Est. for grounded answers only</div>
    <p class="rr-title">The Reading Room</p>
    <p class="rr-sub">Upload a document, let it be filed and embedded, then ask a question — every answer is drawn strictly from what's on record.</p>
</div>
""",
    unsafe_allow_html=True,
)

# ---------------- Main: chat or empty state ----------------
if not st.session_state.vectorstore_ready:
    st.markdown(
        '<div class="rr-empty">The archive is empty.<br>Upload a PDF in the sidebar and click '
        '<b>Index documents</b> to begin.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

try:
    retriever = get_retriever(k, fetch_k, lambda_mult)
    llm = get_llm(model_name)
except Exception as e:
    st.error(f"Could not open the archive: {e}")
    st.stop()

for msg in st.session_state.messages:
    avatar = "🖋️" if msg["role"] == "user" else "📖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            render_slips(msg["sources"])

query = st.chat_input("Ask the archive...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user", avatar="🖋️"):
        st.markdown(query)

    with st.chat_message("assistant", avatar="📖"):
        with st.spinner("Searching the stacks..."):
            try:
                answer, docs = answer_query(query, retriever, llm)
            except Exception as e:
                answer, docs = f"The archive could not be reached: {e}", []
        st.markdown(answer)
        if docs:
            render_slips(docs)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": docs})
