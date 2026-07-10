import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="CourseMate AI", page_icon="🎓", layout="wide")

# ---------------------------------------------------------------------------
# Visual identity — "annotated textbook" theme
# ---------------------------------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=Inter:wght@400;500;600&family=IBM+Plex+Mono:wght@500&display=swap');

    :root {
        --ink: #1C2541;
        --parchment: #F6F1E1;
        --paper: #FFFDF7;
        --highlighter: #F4D35E;
        --graphite: #4A4E69;
        --rust: #C1502E;
        --line: #E4DBC4;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stApp { background: var(--parchment); }

    [data-testid="stSidebar"] { background: var(--ink); border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] * { color: #EDE7D3 !important; }
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
        font-family: 'Fraunces', serif; font-weight: 700; letter-spacing: -0.01em;
    }
    [data-testid="stSidebar"] hr { border-color: rgba(237,231,211,0.15); }
    [data-testid="stSidebar"] .stSlider label, [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stTextInput label, [data-testid="stSidebar"] .stCheckbox label,
    [data-testid="stSidebar"] .stFileUploader label {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem;
        text-transform: uppercase; letter-spacing: 0.06em; color: #B9B190 !important;
    }
    [data-testid="stSidebar"] .stButton>button {
        background: transparent; border: 1px solid #B9B190; color: #EDE7D3 !important;
        border-radius: 6px; font-family: 'IBM Plex Mono', monospace; font-size: 0.8rem;
    }
    [data-testid="stSidebar"] .stButton>button:hover { border-color: var(--highlighter); color: var(--highlighter) !important; }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background: rgba(244,211,94,0.06); border: 1px dashed #B9B190; border-radius: 8px;
    }

    .hero-wrap { display: flex; align-items: baseline; gap: 0.6rem; margin-bottom: 0.1rem; }
    .hero-title {
        font-family: 'Fraunces', serif; font-weight: 700; font-size: 2.7rem;
        color: var(--ink); letter-spacing: -0.02em; margin: 0;
    }
    .hero-title .swipe {
        background-image: linear-gradient(var(--highlighter), var(--highlighter));
        background-repeat: no-repeat; background-size: 100% 38%; background-position: 0 88%;
        padding: 0 0.1em;
    }
    .hero-tag { font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; color: var(--graphite); }
    .hero-sub { color: var(--graphite); font-size: 1.02rem; margin: 0.35rem 0 1.6rem 0; max-width: 40rem; }
    .hero-rule { border: none; border-top: 1px dashed var(--line); margin: 0 0 1.6rem 0; }

    .msg-row { display: flex; margin-bottom: 1.1rem; animation: fadein 0.25s ease-out; }
    .msg-row.user { justify-content: flex-end; }
    .msg-row.ai { justify-content: flex-start; }
    .card { max-width: 70%; padding: 0.9rem 1.15rem; border-radius: 10px; font-size: 0.96rem; line-height: 1.55; box-shadow: 0 1px 3px rgba(28,37,65,0.08); }
    .card.user { background: var(--ink); color: var(--paper); border-top-right-radius: 2px; }
    .card.ai { background: var(--paper); color: var(--ink); border-left: 4px solid var(--highlighter); border-top-left-radius: 2px; }
    .card-label { font-family: 'IBM Plex Mono', monospace; font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.08em; opacity: 0.6; margin-bottom: 0.35rem; display: block; }

    .flags { margin: 0.5rem 0 0 0.1rem; }
    .flag { display: inline-block; font-family: 'IBM Plex Mono', monospace; font-size: 0.68rem; background: var(--highlighter); color: var(--ink); border-radius: 3px; padding: 0.12rem 0.4rem; margin-right: 0.3rem; font-weight: 500; }

    @keyframes fadein { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }

    [data-testid="stChatInput"] { background: var(--paper); border: 1px solid var(--line); border-radius: 10px; }
    [data-testid="stExpander"] { background: var(--paper); border: 1px dashed var(--line); border-radius: 8px; }
    [data-testid="stExpander"] summary { font-family: 'IBM Plex Mono', monospace; font-size: 0.78rem; }

    .empty-state { border: 1px dashed var(--line); border-radius: 10px; padding: 2.2rem; text-align: center; color: var(--graphite); background: rgba(255,253,247,0.5); }
    .empty-state .glyph { font-size: 1.8rem; margin-bottom: 0.4rem; }

    .lib-item {
        font-family: 'IBM Plex Mono', monospace; font-size: 0.72rem; color: #B9B190 !important;
        padding: 0.25rem 0; border-bottom: 1px dashed rgba(237,231,211,0.15);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Opening your notes...")
def load_vector_store(persist_directory: str):
    embedding_model = MistralAIEmbeddings()
    return Chroma(persist_directory=persist_directory, embedding_function=embedding_model)


@st.cache_resource(show_spinner=False)
def load_llm(model_name: str):
    return ChatMistralAI(model=model_name)


def get_prompt():
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a helpful AI assistant.
                Use ONLY the provided context to answer the question.
                If the answer is not present in the context, say: "I could not find the answer in the document."
                """,
            ),
            ("human", "Context:\n{context}\n\nQuestion:\n{question}\n"),
        ]
    )


def ingest_pdf(uploaded_file, vector_store, chunk_size=1000, chunk_overlap=200) -> int:
    """Save an uploaded PDF to disk, split it, and add it to the vector store."""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        pages = PyPDFLoader(tmp_path).load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = splitter.split_documents(pages)
        for chunk in chunks:
            chunk.metadata["source"] = uploaded_file.name
        if chunks:
            vector_store.add_documents(chunks)
        return len(chunks)
    finally:
        os.remove(tmp_path)


def list_indexed_sources(vector_store):
    """Return the distinct filenames currently indexed, and total chunk count."""
    try:
        data = vector_store.get(include=["metadatas"])
        metadatas = data.get("metadatas") or []
        sources = sorted({m.get("source", "unknown") for m in metadatas if m})
        return sources, len(metadatas)
    except Exception:
        return [], 0


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## Study Settings")
    st.caption("Add your course material and tune how CourseMate searches it.")
    st.markdown("---")

    persist_directory = st.text_input("Notes folder (Chroma DB)", value="chroma_db")
    model_name = st.selectbox(
        "Model",
        options=["mistral-small-2506", "mistral-large-latest", "mistral-medium-latest"],
        index=0,
    )

    vector_store = load_vector_store(persist_directory)

    st.markdown("---")
    st.markdown("### Upload course material")
    uploaded_files = st.file_uploader(
        "Drop PDFs here — lecture notes, slides, readings",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if "ingested_names" not in st.session_state:
        st.session_state.ingested_names = set()

    if uploaded_files:
        new_files = [f for f in uploaded_files if f.name not in st.session_state.ingested_names]
        if new_files:
            with st.spinner(f"Reading {len(new_files)} file(s)..."):
                total_chunks = 0
                for f in new_files:
                    total_chunks += ingest_pdf(f, vector_store)
                    st.session_state.ingested_names.add(f.name)
            st.success(f"Indexed {len(new_files)} file(s) — {total_chunks} passages added.")
            st.rerun()

    sources, chunk_count = list_indexed_sources(vector_store)
    if sources:
        with st.expander(f"Indexed material ({len(sources)} file, {chunk_count} passages)" if len(sources) == 1
                          else f"Indexed material ({len(sources)} files, {chunk_count} passages)"):
            for s in sources:
                st.markdown(f'<div class="lib-item">{s}</div>', unsafe_allow_html=True)
    else:
        st.caption("No material indexed yet — upload a PDF to get started.")

    st.markdown("---")
    st.markdown("### Search depth")
    k = st.slider("Passages returned", min_value=1, max_value=20, value=10)
    fetch_k = st.slider("Passages considered", min_value=10, max_value=200, value=100, step=10)
    lambda_mult = st.slider("Focus vs. variety", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

    st.markdown("---")
    show_sources = st.checkbox("Show passages behind each answer", value=True)

    st.markdown("---")
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------

st.markdown(
    """
    <div class="hero-wrap">
        <p class="hero-title">Course<span class="swipe">Mate</span> AI</p>
        <span class="hero-tag">/ study companion</span>
    </div>
    <p class="hero-sub">Upload your course material and ask a question — every answer is pulled straight from the text, nothing invented.</p>
    <hr class="hero-rule" />
    """,
    unsafe_allow_html=True,
)

if "messages" not in st.session_state:
    st.session_state.messages = []

try:
    llm = load_llm(model_name)
    prompt = get_prompt()
except Exception as e:
    st.error(f"Couldn't load the model: {e}")
    st.stop()

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
)

# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

def render_user(text: str):
    st.markdown(
        f'<div class="msg-row user"><div class="card user"><span class="card-label">You</span>{text}</div></div>',
        unsafe_allow_html=True,
    )


def render_ai(text: str, sources: list[str] | None = None):
    st.markdown(
        f'<div class="msg-row ai"><div class="card ai"><span class="card-label">CourseMate</span>{text}</div></div>',
        unsafe_allow_html=True,
    )
    if sources:
        flags_html = "".join(f'<span class="flag">Passage {i}</span>' for i in range(1, len(sources) + 1))
        st.markdown(f'<div class="flags">{flags_html}</div>', unsafe_allow_html=True)
        with st.expander("View passages behind this answer"):
            for i, src in enumerate(sources, start=1):
                st.markdown(f"**Passage {i}**")
                st.text(src)

# ---------------------------------------------------------------------------
# Conversation
# ---------------------------------------------------------------------------

no_material = chunk_count == 0

if not st.session_state.messages:
    if no_material:
        st.markdown(
            """
            <div class="empty-state">
                <div class="glyph">📎</div>
                <div>Nothing indexed yet — upload a PDF from the sidebar to start asking questions.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="empty-state">
                <div class="glyph">📖</div>
                <div>No questions yet — ask something about your course material to get started.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            render_user(msg["content"])
        else:
            render_ai(msg["content"], msg.get("sources") if show_sources else None)

query = st.chat_input(
    "Ask about your course material..." if not no_material else "Upload a PDF first to ask questions",
    disabled=no_material,
)

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    render_user(query)

    with st.spinner("Searching your notes..."):
        docs = retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in docs)
        final_prompt = prompt.invoke({"context": context, "question": query})
        response = llm.invoke(final_prompt)

    sources = [doc.page_content for doc in docs]
    render_ai(response.content, sources if show_sources else None)

    st.session_state.messages.append(
        {"role": "assistant", "content": response.content, "sources": sources}
    )
