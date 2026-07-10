import streamlit as st
from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="RAG Chatbot", page_icon="📄", layout="wide")

# ---------------------------------------------------------------------------
# Cached resources — loaded once per session, not on every rerun
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model & vector store...")
def load_vector_store(persist_directory: str):
    embedding_model = MistralAIEmbeddings()
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
    )
    return vector_store


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
# Sidebar — retriever & model settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("⚙️ Settings")

    persist_directory = st.text_input("Chroma persist directory", value="chroma_db")
    model_name = st.selectbox(
        "Mistral model",
        options=["mistral-small-2506", "mistral-large-latest", "mistral-medium-latest"],
        index=0,
    )

    st.subheader("Retriever (MMR)")
    k = st.slider("k (results returned)", min_value=1, max_value=20, value=10)
    fetch_k = st.slider("fetch_k (candidates considered)", min_value=10, max_value=200, value=100, step=10)
    lambda_mult = st.slider("lambda_mult (relevance vs diversity)", min_value=0.0, max_value=1.0, value=0.5, step=0.05)

    show_sources = st.checkbox("Show retrieved source chunks", value=False)

    if st.button("🗑️ Clear chat history"):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------------------------
# Main chat UI
# ---------------------------------------------------------------------------

st.title("📄 RAG Chatbot")
st.caption("Ask questions about your document — answers are grounded only in the retrieved context.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Load resources (cached — cheap after first run)
try:
    vector_store = load_vector_store(persist_directory)
    llm = load_llm(model_name)
    prompt = get_prompt()
except Exception as e:
    st.error(f"Failed to initialize RAG pipeline: {e}")
    st.stop()

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": k, "fetch_k": fetch_k, "lambda_mult": lambda_mult},
)

# Replay chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander("📚 Sources used"):
                for i, src in enumerate(msg["sources"], start=1):
                    st.markdown(f"**Chunk {i}**")
                    st.text(src)

# Chat input
query = st.chat_input("Enter your question...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving context and generating answer..."):
            docs = retriever.invoke(query)
            context = "\n\n".join(doc.page_content for doc in docs)

            final_prompt = prompt.invoke({"context": context, "question": query})
            response = llm.invoke(final_prompt)

        st.markdown(response.content)

        sources = [doc.page_content for doc in docs]
        if show_sources and sources:
            with st.expander("📚 Sources used"):
                for i, src in enumerate(sources, start=1):
                    st.markdown(f"**Chunk {i}**")
                    st.text(src)

    st.session_state.messages.append(
        {"role": "assistant", "content": response.content, "sources": sources}
    )
