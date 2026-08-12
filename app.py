import os
import uuid
from pathlib import Path
from typing import Any

import chromadb
import gradio as gr
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

_embeddings = None
_generator = None
_vectorstore = None
_chroma_client = None
_current_pdf = None

def get_embeddings():
    global _embeddings
    if _embeddings is None:
        # Outsourcing embeddings to Google
        _embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    return _embeddings

def get_generator():
    global _generator
    if _generator is None:
        # Outsourcing text generation to Gemini 1.5 Flash
        _generator = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    return _generator

def build_vectorstore(pdf_path: str):
    global _vectorstore
    global _chroma_client
    global _current_pdf

    if not pdf_path:
        raise ValueError("PDF ഫയൽ ലഭിച്ചില്ല.")

    print(f"[INDEX] PDF ലഭിച്ചു: {pdf_path}")

    documents = PyPDFLoader(pdf_path).load()
    print(f"[INDEX] പേജുകൾ: {len(documents)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    print(f"[INDEX] chunks: {len(chunks)}")

    _chroma_client = chromadb.EphemeralClient()
    collection_name = f"day11_rag_{uuid.uuid4().hex[:12]}"

    _vectorstore = Chroma(
        client=_chroma_client,
        collection_name=collection_name,
        embedding_function=get_embeddings(),
    )

    _vectorstore.add_documents(chunks)
    _current_pdf = Path(pdf_path).name
    print(f"[INDEX] വിജയിച്ചു: {_current_pdf}")

    return (
        f"✅ **PDF വിജയകരമായി Index ചെയ്തു!**\n\n"
        f"📄 ഫയൽ: **{_current_pdf}**\n\n"
        f"📑 പേജുകൾ: **{len(documents)}**\n\n"
        f"🧩 Chunks: **{len(chunks)}**"
    )

def format_history(history: list[dict[str, Any]]) -> str:
    if not history:
        return "(മുൻ സംഭാഷണം ഇല്ല)"

    lines = []
    for item in history[-6:]:
        role = item.get("role", "")
        content = item.get("content", "")
        if role in {"user", "assistant"} and content:
            lines.append(f"{role.title()}: {content}")

    return "\n".join(lines) or "(മുൻ സംഭാഷണം ഇല്ല)"

def generate_answer(prompt: str) -> str:
    generator = get_generator()
    
    system_instruction = (
        "You are a PDF question-answering assistant. "
        "Answer only from the supplied context. "
        "If the context does not contain the answer, say: "
        "I don't know based on the uploaded PDF. "
        "Do not invent facts.\n\n"
    )
    
    # Combine instructions with the prompt
    full_prompt = system_instruction + prompt
    
    response = generator.invoke(full_prompt)
    return response.content.strip()

def ask_pdf(question: str, history: list[dict[str, Any]]):
    history = history or []

    if not question or not question.strip():
        return history

    if _vectorstore is None:
        answer = "⚠️ ആദ്യം PDF upload ചെയ്ത് **Index PDF** അമർത്തുക."
        new_history = list(history)
        new_history.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ])
        return new_history

    print(f"[ASK] Question: {question}")
    history_text = format_history(history)

    retrieval_query = f"Conversation:\n{history_text}\n\nCurrent question: {question}"
    collection_count = _vectorstore._collection.count()
    k = min(4, collection_count)

    docs = _vectorstore.similarity_search(retrieval_query, k=max(k, 1))

    context_parts = []
    sources = []

    for doc in docs:
        page = doc.metadata.get("page")
        source = Path(doc.metadata.get("source", _current_pdf or "uploaded PDF")).name

        if isinstance(page, int):
            label = f"{source}, page {page + 1}"
        else:
            label = source

        context_parts.append(doc.page_content)
        sources.append(label)

    context = "\n\n---\n\n".join(context_parts)

    prompt = (
        "Use the following context from the uploaded PDF to answer the current question.\n\n"
        f"Context:\n{context}\n\n"
        f"Recent conversation:\n{history_text}\n\n"
        f"Current question: {question}\n\n"
        "Give a concise answer based only on the supplied PDF context."
    )

    answer = generate_answer(prompt)

    unique_sources = list(dict.fromkeys(sources))
    if unique_sources:
        answer += "\n\n**Sources:** " + ", ".join(unique_sources)

    new_history = list(history)
    new_history.extend([
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ])

    return new_history

def index_pdf(pdf_file):
    print("[BUTTON] Index PDF clicked")
    if pdf_file is None:
        return "⚠️ ആദ്യം ഒരു PDF upload ചെയ്യുക."
    try:
        print(f"[BUTTON] File: {pdf_file}")
        result = build_vectorstore(pdf_file)
        return result
    except Exception as exc:
        print(f"[ERROR] Indexing failed: {exc}")
        return f"❌ **Indexing പരാജയപ്പെട്ടു**\n\nError: `{exc}`"

def clear_chat():
    return [], []

with gr.Blocks(title="PDF RAG Assistant") as demo:

    gr.Markdown(
        """
        # 📚 PDF RAG Assistant
        PDF upload ചെയ്യുക → Index ചെയ്യുക → ചോദ്യങ്ങൾ ചോദിക്കുക.
        ChromaDB retrieval + conversational memory ഉപയോഗിക്കുന്നു.
        """
    )

    history = gr.State([])

    with gr.Row():
        pdf = gr.File(label="📄 Upload PDF", file_types=[".pdf"], type="filepath")
        index_button = gr.Button("📚 Index PDF", variant="primary")

    status = gr.Markdown("⬆️ PDF upload ചെയ്ത് **Index PDF** അമർത്തുക.")
    chatbot = gr.Chatbot(label="💬 Conversation", height=450)
    question = gr.Textbox(label="❓ Question", placeholder="PDF-നെക്കുറിച്ച് ഒരു ചോദ്യം ചോദിക്കുക...")

    with gr.Row():
        ask_button = gr.Button("💬 Ask", variant="primary")
        clear_button = gr.Button("🗑️ Clear Conversation")

    def ask_for_chat(question_text, history_state):
        new_history = ask_pdf(question_text, history_state or [])
        return new_history, new_history

    index_button.click(fn=index_pdf, inputs=pdf, outputs=status)

    ask_button.click(
        fn=ask_for_chat,
        inputs=[question, history],
        outputs=[chatbot, history],
    ).then(lambda: "", outputs=question)

    question.submit(
        fn=ask_for_chat,
        inputs=[question, history],
        outputs=[chatbot, history],
    ).then(lambda: "", outputs=question)

    clear_button.click(fn=clear_chat, outputs=[chatbot, history])

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 10000)),
    )