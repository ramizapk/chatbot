import io
import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import plotly.express as px
import streamlit as st
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langdetect import detect
from openai import OpenAI
from pypdf import PdfReader

# -----------------------------------------------------------------------------
# Path & configuration helpers
# -----------------------------------------------------------------------------
DATA_DIR = Path("app_state")
UPLOAD_DIR = DATA_DIR / "uploaded_files"
INDEX_DIR = DATA_DIR / "faiss_index"
FILES_DB = DATA_DIR / "files.json"
CONVERSATIONS_DB = DATA_DIR / "conversations.json"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

for path in (DATA_DIR, UPLOAD_DIR):
    path.mkdir(parents=True, exist_ok=True)

st.set_page_config(
    page_title="SalaSa Knowledge Assistant",
    layout="wide",
    page_icon="💬",
)

# -----------------------------------------------------------------------------
# Session bootstrap
# -----------------------------------------------------------------------------
def _load_json(path: Path, default):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return default
    return default


def _save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def bootstrap_state() -> None:
    if "files_state" not in st.session_state:
        st.session_state.files_state = _load_json(FILES_DB, [])
    if "conversation_state" not in st.session_state:
        st.session_state.conversation_state = _load_json(CONVERSATIONS_DB, [])
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "is_admin" not in st.session_state:
        st.session_state.is_admin = False
    if "index_updated_at" not in st.session_state:
        st.session_state.index_updated_at = datetime.utcnow().timestamp() if INDEX_DIR.exists() else 0.0


bootstrap_state()

# -----------------------------------------------------------------------------
# Caches
# -----------------------------------------------------------------------------
@st.cache_resource(show_spinner=True)
def get_embeddings():
    return HuggingFaceEmbeddings(model_name=MODEL_NAME)


@st.cache_resource(show_spinner=False)
def load_vector_store(timestamp: float):
    if not INDEX_DIR.exists():
        return None
    if not any(INDEX_DIR.iterdir()):
        return None
    return FAISS.load_local(
        str(INDEX_DIR),
        get_embeddings(),
        allow_dangerous_deserialization=True,
    )


# -----------------------------------------------------------------------------
# Storage helpers
# -----------------------------------------------------------------------------
def get_file_records() -> List[Dict]:
    return st.session_state.files_state


def set_file_records(records: List[Dict]) -> None:
    st.session_state.files_state = records
    _save_json(FILES_DB, records)


def get_conversations() -> List[Dict]:
    return st.session_state.conversation_state


def set_conversations(records: List[Dict]) -> None:
    st.session_state.conversation_state = records
    _save_json(CONVERSATIONS_DB, records)


def persist_file(content: bytes, original_name: str) -> str:
    safe_name = Path(original_name).name
    target = UPLOAD_DIR / safe_name
    if target.exists():
        stem = target.stem
        suffix = target.suffix
        safe_name = f"{stem}_{uuid.uuid4().hex[:4]}{suffix}"
        target = UPLOAD_DIR / safe_name
    target.write_bytes(content)
    return safe_name


# -----------------------------------------------------------------------------
# Document preparation & vector store management
# -----------------------------------------------------------------------------
def extract_text(file_bytes: bytes, filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(io.BytesIO(file_bytes))
        text_parts = []
        for page in reader.pages:
            text_parts.append(page.extract_text() or "")
        text = "\n".join(text_parts)
        if not text.strip():
            raise ValueError("No text detected in the PDF. Scanned files are not supported in this deployment.")
        return text
    if suffix in {".txt", ".md", ".csv"}:
        return file_bytes.decode("utf-8", errors="ignore")
    if suffix == ".json":
        data = json.loads(file_bytes.decode("utf-8", errors="ignore"))
        return json.dumps(data, ensure_ascii=False, indent=2)
    raise ValueError("Unsupported file format. Please upload PDF, TXT, MD, CSV, or JSON files.")


splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


def make_documents(text: str, stored_name: str, display_name: str) -> List[Document]:
    chunks = splitter.split_text(text)
    return [
        Document(page_content=chunk, metadata={"source": stored_name, "display_name": display_name})
        for chunk in chunks
    ]


def bump_index_timestamp() -> None:
    st.session_state.index_updated_at = datetime.utcnow().timestamp()


def append_to_index(docs: List[Document]) -> None:
    embeddings = get_embeddings()
    if INDEX_DIR.exists() and any(INDEX_DIR.iterdir()):
        store = FAISS.load_local(str(INDEX_DIR), embeddings, allow_dangerous_deserialization=True)
        store.add_documents(docs)
    else:
        store = FAISS.from_documents(docs, embeddings)
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
    store.save_local(str(INDEX_DIR))
    bump_index_timestamp()


def rebuild_index_from_disk() -> None:
    docs: List[Document] = []
    for record in get_file_records():
        path = UPLOAD_DIR / record["stored_name"]
        if not path.exists():
            continue
        try:
            text = extract_text(path.read_bytes(), record["original_name"])
            docs.extend(make_documents(text, record["stored_name"], record["original_name"]))
        except Exception:
            continue
    if docs:
        embeddings = get_embeddings()
        store = FAISS.from_documents(docs, embeddings)
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        store.save_local(str(INDEX_DIR))
    elif INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR)
    bump_index_timestamp()


# -----------------------------------------------------------------------------
# Conversations & analytics
# -----------------------------------------------------------------------------
def log_conversation(question: str, answer: str, sources: List[str], language: str) -> None:
    record = {
        "id": str(uuid.uuid4()),
        "session_id": st.session_state.session_id,
        "question": question,
        "answer": answer,
        "language": language,
        "sources": sources,
        "timestamp": datetime.utcnow().isoformat(),
    }
    conversations = get_conversations()
    conversations.append(record)
    set_conversations(conversations)


def conversations_dataframe() -> pd.DataFrame:
    records = get_conversations()
    if not records:
        return pd.DataFrame(columns=["timestamp", "question", "answer", "language", "sources"])
    df = pd.DataFrame(records)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df.sort_values("timestamp")


# -----------------------------------------------------------------------------
# LLM helper
# -----------------------------------------------------------------------------
def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        return "arabic" if lang.startswith("ar") else "english"
    except Exception:
        return "english"


def call_llm(question: str, context: str, language: str) -> Tuple[str, str]:
    api_key = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
    model = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        fallback = "Missing OPENAI_API_KEY. Showing the most relevant context instead.\n\n" + context
        return fallback, "missing_api_key"

    client = OpenAI(api_key=api_key)
    system_message = (
        "You are a helpful university assistant that responds using only the supplied context. "
        "If an answer is missing, say that it was not found."
    )
    if language == "arabic":
        system_message += " Always respond in Modern Standard Arabic."
    else:
        system_message += " Always respond in English."

    user_prompt = f"Context:\n{context or 'No context available.'}\n\nQuestion:\n{question.strip()}"
    try:
        response = client.chat.completions.create(
            model=model,
            temperature=0.1,
            max_tokens=600,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": user_prompt},
            ],
        )
        return response.choices[0].message.content.strip(), "ok"
    except Exception as exc:
        return f"Could not reach the language model: {exc}", "error"


# -----------------------------------------------------------------------------
# Chat pipeline
# -----------------------------------------------------------------------------
def answer_question(question: str) -> Tuple[str, List[str]]:
    vector_store = load_vector_store(st.session_state.index_updated_at)
    if vector_store is None:
        raise RuntimeError("No knowledge base has been created yet. Upload documents first.")

    docs = vector_store.similarity_search(question, k=3)
    used_sources = []
    context_chunks = []
    for doc in docs:
        display_name = doc.metadata.get("display_name") or doc.metadata.get("source", "Unknown file")
        used_sources.append(display_name)
        snippet = doc.page_content.strip()
        context_chunks.append(f"Source: {display_name}\n{snippet}")

    context = "\n\n".join(context_chunks)
    language = detect_language(question)
    answer, status = call_llm(question, context, language)
    log_conversation(question, answer, used_sources, language)
    st.session_state.chat_history.append({
        "question": question,
        "answer": answer,
        "language": language,
        "sources": used_sources,
        "status": status,
    })
    return answer, used_sources


# -----------------------------------------------------------------------------
# File management
# -----------------------------------------------------------------------------
def add_file_record(record: Dict) -> None:
    records = get_file_records()
    records.append(record)
    set_file_records(records)


def delete_file(record_id: str) -> None:
    records = get_file_records()
    target = next((item for item in records if item["id"] == record_id), None)
    if not target:
        return
    path = UPLOAD_DIR / target["stored_name"]
    if path.exists():
        path.unlink()
    records = [item for item in records if item["id"] != record_id]
    set_file_records(records)
    rebuild_index_from_disk()


def ingest_uploaded_file(uploaded_file) -> Tuple[bool, str]:
    try:
        raw_bytes = uploaded_file.getvalue()
        stored_name = persist_file(raw_bytes, uploaded_file.name)
        text = extract_text(raw_bytes, uploaded_file.name)
        docs = make_documents(text, stored_name, uploaded_file.name)
        append_to_index(docs)
        record = {
            "id": str(uuid.uuid4()),
            "stored_name": stored_name,
            "original_name": uploaded_file.name,
            "uploaded_at": datetime.utcnow().isoformat(),
            "size_bytes": len(raw_bytes),
            "char_count": len(text),
        }
        add_file_record(record)
        return True, f"Added {uploaded_file.name} ({len(text)} characters)."
    except Exception as exc:
        return False, str(exc)


# -----------------------------------------------------------------------------
# UI helpers
# -----------------------------------------------------------------------------
def render_chat() -> None:
    st.subheader("Chat with your knowledge base")
    if not (INDEX_DIR.exists() and any(INDEX_DIR.iterdir())):
        st.info("Upload documents in the Admin tab to build the knowledge base.")

    for message in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(message["question"])
        with st.chat_message("assistant"):
            st.markdown(message["answer"])
            if message["sources"]:
                st.caption("Sources: " + ", ".join(message["sources"]))

    user_prompt = st.chat_input("Ask about the uploaded files...")
    if user_prompt:
        try:
            answer, sources = answer_question(user_prompt)
            with st.chat_message("assistant"):
                st.markdown(answer)
                if sources:
                    st.caption("Sources: " + ", ".join(sources))
        except RuntimeError as err:
            st.error(str(err))


def render_file_manager(is_admin: bool) -> None:
    st.subheader("Knowledge base files")
    records = sorted(get_file_records(), key=lambda item: item["uploaded_at"], reverse=True)
    if not records:
        st.info("No files uploaded yet.")
    else:
        table = [
            {
                "Original name": item["original_name"],
                "Stored name": item["stored_name"],
                "Uploaded at": item["uploaded_at"],
                "Size (KB)": round(item["size_bytes"] / 1024, 1),
                "Characters": item["char_count"],
            }
            for item in records
        ]
        st.dataframe(pd.DataFrame(table), use_container_width=True)

        selected = st.selectbox(
            "Choose a file to download or delete",
            options=[item["id"] for item in records],
            format_func=lambda rid: next(item["original_name"] for item in records if item["id"] == rid),
        )
        target = next(item for item in records if item["id"] == selected)
        file_bytes = (UPLOAD_DIR / target["stored_name"]).read_bytes() if (UPLOAD_DIR / target["stored_name"]).exists() else b""
        st.download_button(
            label="Download selected file",
            data=file_bytes,
            file_name=target["original_name"],
            use_container_width=True,
        )
        if is_admin and st.button("Delete selected file", type="primary"):
            delete_file(selected)
            st.success("File removed and index rebuilt.")
            st.experimental_rerun()

    if is_admin:
        st.markdown("---")
        uploads = st.file_uploader(
            "Upload PDF/TXT/MD/JSON files",
            accept_multiple_files=True,
            type=["pdf", "txt", "md", "json", "csv"],
        )
        if uploads:
            for uploaded in uploads:
                ok, msg = ingest_uploaded_file(uploaded)
                (st.success if ok else st.error)(msg)

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Rebuild index", help="Recreate the FAISS index from disk"):
                rebuild_index_from_disk()
                st.success("Vector index rebuilt.")
        with col2:
            if st.button("Clear chat history"):
                st.session_state.chat_history = []
                st.success("Cleared the in-session chat history.")


def render_analytics() -> None:
    st.subheader("Usage analytics")
    df = conversations_dataframe()
    total_questions = len(df)
    st.metric("Total questions", total_questions)
    st.metric("Uploaded files", len(get_file_records()))

    if total_questions == 0:
        st.info("Ask a question to start collecting analytics.")
        return

    lang_counts = df["language"].value_counts().to_dict()
    st.write("Language breakdown:", lang_counts)

    df_daily = df.copy()
    df_daily["date"] = df_daily["timestamp"].dt.date
    daily_counts = df_daily.groupby("date").size().reset_index(name="questions")
    st.plotly_chart(px.line(daily_counts, x="date", y="questions", markers=True, title="Questions per day"), use_container_width=True)

    top_questions = df["question"].value_counts().head(5).reset_index()
    top_questions.columns = ["question", "count"]
    st.plotly_chart(px.bar(top_questions, x="question", y="count", title="Most common questions"), use_container_width=True)

    csv_data = df[["timestamp", "session_id", "question", "answer", "language", "sources"]]
    st.download_button(
        "Download conversation log (CSV)",
        data=csv_data.to_csv(index=False).encode("utf-8"),
        file_name="conversation_log.csv",
    )


# -----------------------------------------------------------------------------
# Sidebar authentication
# -----------------------------------------------------------------------------
def render_sidebar() -> bool:
    st.sidebar.header("Admin login")
    if st.session_state.is_admin:
        st.sidebar.success("Logged in as admin")
        if st.sidebar.button("Log out"):
            st.session_state.is_admin = False
        return True

    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    expected_user = st.secrets.get("ADMIN_USERNAME", "admin")
    expected_pass = st.secrets.get("ADMIN_PASSWORD", "adminpass")

    if st.sidebar.button("Log in"):
        if username == expected_user and password == expected_pass:
            st.session_state.is_admin = True
            st.sidebar.success("Authentication successful")
        else:
            st.sidebar.error("Invalid credentials")
    return st.session_state.is_admin


# -----------------------------------------------------------------------------
# Main layout
# -----------------------------------------------------------------------------
is_admin = render_sidebar()

st.title("SalaSa Knowledge Assistant")
st.caption("This build stores files locally and relies on OpenAI for answers, making it Streamlit Cloud friendly.")

chat_tab, files_tab, analytics_tab = st.tabs(["Chat", "Knowledge Base", "Analytics"])
with chat_tab:
    render_chat()
with files_tab:
    render_file_manager(is_admin)
with analytics_tab:
    render_analytics()
