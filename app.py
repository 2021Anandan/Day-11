import uuid
from pathlib import Path
from typing import Any

import chromadb
import gradio as gr
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import pipeline

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

_embeddings = None
_generator = None
_vectorstore = None
_chroma_client = None
_current_pdf = None


def get_embeddings():
    global _embeddings
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings


def get_generator():
    global _generator
    if _generator is None:
        _generator = pipeline(
            "text-generation",
            model=LLM_MODEL,
            tokenizer=LLM_MODEL,
            device_map="auto",
            torch_dtype="auto",
        )
    return _generator


def build_vectorstore(pdf_path: str):
    global _vectorstore, _chroma_client, _current_pdf

    if not pdf_path:
        raise ValueError("Please upload a PDF first.")

    documents = PyPDFLoader(pdf_path).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    # Create an explicit ephemeral Chroma client so local runs do not depend on
    # a previously persisted/default tenant or stale collections.
    _chroma_client = chromadb.EphemeralClient()
    collection_name = f"day11_rag_{uuid.uuid4().hex[:12]}"
    _vectorstore = Chroma(
        client=_chroma_client,
        collection_name=collection_name,
        embedding_function=get_embeddings(),
    )
    _vectorstore.add_documents(chunks)
    _current_pdf = Path(pdf_path).name

    return f"Indexed **{_current_pdf}**: {len(documents)} page(s), {len(chunks)} chunk(s)."


def format_history(history: list[dict[str, Any]]) -> str:
    if not history:
        return "(no previous conversation)"
    lines = []
    for item in history[-6:]:
        role = item.get("role", "")
        content = item.get("content", "")
        if role in {"user", "assistant"} and content:
            lines.append(f"{role.title()}: {content}")
    return "\n".join(lines) or "(no previous conversation)"


def generate_answer(prompt: str) -> str:
    generator = get_generator()
    messages = [
        {
            "role": "system",
            "content": (
                "You are a PDF question-answering assistant. "
                "Answer only from the supplied context. If the context does not contain the answer, "
                "say: I don't know based on the uploaded PDF. Do not invent facts."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    output = generator(messages, max_new_tokens=220, do_sample=False)
    generated = output[0]["generated_text"]
    if isinstance(generated, list):
        return generated[-1]["content"].strip()
    return str(generated).strip()


def ask_pdf(question: str, history: list[dict[str, Any]]):
    history = history or []

    if not question or not question.strip():
        return history

    if _vectorstore is None:
        answer = "Please upload and index a PDF first."
        new_history = list(history)
        new_history.extend([
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ])
        return new_history

    history_text = format_history(history)
    retrieval_query = f"Conversation:\n{history_text}\n\nCurrent question: {question}"
    k = min(4, _vectorstore._collection.count())
    docs = _vectorstore.similarity_search(retrieval_query, k=max(k, 1))

    context_parts = []
    sources = []
    for doc in docs:
        page = doc.metadata.get("page")
        source = Path(doc.metadata.get("source", _current_pdf or "uploaded PDF")).name
        label = f"{source}, page {page + 1}" if isinstance(page, int) else source
        context_parts.append(doc.page_content)
        sources.append(label)

    context = "\n\n---\n\n".join(context_parts)
    prompt = (
        "Use the following context from the uploaded PDF to answer the current question.\n\n"
        f"Context:\n{context}\n\n"
        f"Recent conversation:\n{history_text}\n\n"
        f"Current question: {question}\n\n"
        "Give a concise, helpful answer grounded only in the context."
    )

    answer = generate_answer(prompt)
    unique_sources = list(dict.fromkeys(sources))
    answer += "\n\n**Sources:** " + ", ".join(unique_sources)

    new_history = list(history)
    new_history.extend([
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer},
    ])
    return new_history


def index_pdf(pdf_file):
    if pdf_file is None:
        return "Please upload a PDF."
    try:
        return build_vectorstore(pdf_file)
    except Exception as exc:
        return f"Indexing failed: {exc}"


def clear_chat():
    return [], []


with gr.Blocks(title="PDF RAG Assistant") as demo:
    gr.Markdown(
        "# 📚 PDF RAG Assistant\n"
        "Upload a PDF, index it into ChromaDB, and ask grounded questions with conversational memory."
    )

    history = gr.State([])

    with gr.Row():
        pdf = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")
        index_button = gr.Button("Index PDF", variant="primary")

    status = gr.Markdown("Upload a PDF and click **Index PDF**.")
    chatbot = gr.Chatbot(label="Conversation", height=450)
    question = gr.Textbox(
        label="Question",
        placeholder="Ask a question about the uploaded PDF...",
    )
    with gr.Row():
        ask_button = gr.Button("Ask", variant="primary")
        clear_button = gr.Button("Clear Conversation")

    def ask_for_chat(question_text, history_state):
        new_history = ask_pdf(question_text, history_state or [])
        return new_history, new_history

    index_button.click(index_pdf, inputs=pdf, outputs=status)
    ask_button.click(ask_for_chat, inputs=[question, history], outputs=[chatbot, history]).then(
        lambda: "", outputs=question
    )
    question.submit(ask_for_chat, inputs=[question, history], outputs=[chatbot, history]).then(
        lambda: "", outputs=question
    )
    clear_button.click(clear_chat, outputs=[chatbot, history])


if __name__ == "__main__":
    demo.launch()
