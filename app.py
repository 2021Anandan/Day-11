import os
from pathlib import Path
from typing import Any

import gradio as gr
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

APP_DIR = Path(__file__).resolve().parent
CHROMA_DIR = APP_DIR / "chroma_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

_embeddings = None
_generator = None
_vectorstore = None
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
    global _vectorstore, _current_pdf

    if not pdf_path:
        raise ValueError("Please upload a PDF first.")

    documents = PyPDFLoader(pdf_path).load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    _vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=get_embeddings(),
        collection_name="day11_rag",
        persist_directory=str(CHROMA_DIR),
    )
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
    if not question or not question.strip():
        return "Please enter a question.", history
    if _vectorstore is None:
        return "Please upload and index a PDF first.", history

    # Memory-aware retrieval: include recent conversation so follow-up questions
    # such as “What about the second option?” retain their context.
    history_text = format_history(history)
    retrieval_query = f"Conversation:\n{history_text}\n\nCurrent question: {question}"
    docs = _vectorstore.similarity_search(retrieval_query, k=4)

    context_parts = []
    sources = []
    for doc in docs:
        page = doc.metadata.get("page")
        source = doc.metadata.get("source", _current_pdf or "uploaded PDF")
        label = f"{Path(source).name}, page {page + 1}" if isinstance(page, int) else Path(source).name
        context_parts.append(doc.page_content)
        sources.append(label)

    context = "\n\n---\n\n".join(context_parts)
    prompt = (
        f"Use the following context from the uploaded PDF to answer the current question.\n\n"
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
    return answer, new_history


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

    chatbot = gr.Chatbot(label="Conversation", type="messages", height=450)
    question = gr.Textbox(
        label="Question",
        placeholder="Ask a question about the uploaded PDF...",
    )
    with gr.Row():
        ask_button = gr.Button("Ask", variant="primary")
        clear_button = gr.Button("Clear Conversation")

    index_button.click(index_pdf, inputs=pdf, outputs=status)
    ask_button.click(ask_pdf, inputs=[question, history], outputs=[chatbot, history]).then(
        lambda: "", outputs=question
    )
    question.submit(ask_pdf, inputs=[question, history], outputs=[chatbot, history]).then(
        lambda: "", outputs=question
    )
    clear_button.click(clear_chat, outputs=[chatbot, history])


if __name__ == "__main__":
    demo.launch()
