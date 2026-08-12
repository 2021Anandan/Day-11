# 📚 Day 11 — PDF RAG Assistant

Epochs '26 Assignment 11: Retrieval-Augmented Generation (RAG) & Vector Databases.

## Participant

- **Name:** Anandan M A
- **MUID:** _Add your Epochs MUID before submission_

## Project Overview

A PDF Question Answering application that loads an uploaded PDF, splits it into chunks, creates embeddings, stores them in ChromaDB, retrieves relevant context, and generates grounded answers with a small local instruction-tuned LLM. Recent conversation turns are retained so follow-up questions can use previous context.

## Architecture

```text
PDF Upload
   ↓
PyPDFLoader
   ↓
Recursive Text Chunking
   ↓
Sentence-Transformer Embeddings
   ↓
ChromaDB Vector Store
   ↓
Similarity Retrieval (Top-K)
   ↓
Conversation Context + Retrieved Context
   ↓
Qwen2.5-0.5B-Instruct
   ↓
Grounded Answer + Source Pages
```

## Technologies Used

- Python
- LangChain
- PyPDFLoader / pypdf
- RecursiveCharacterTextSplitter
- Sentence Transformers (`all-MiniLM-L6-v2`)
- ChromaDB
- Hugging Face Transformers
- Qwen2.5-0.5B-Instruct
- Gradio

## Memory Implementation

The application maintains recent user/assistant messages in Gradio state. For follow-up questions, recent conversation history is included in the retrieval query and generation prompt. This makes follow-up questions such as "What about the second option?" retain conversational context.

## How to Run Locally

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Then open the local Gradio URL printed in the terminal.

## How to Use

1. Upload a PDF.
2. Click **Index PDF**.
3. Ask a question about the document.
4. Ask a follow-up question to test conversation memory.
5. Check the source page references shown with the answer.

## Challenges Faced

- Keeping the application lightweight enough for local execution.
- Connecting PDF parsing, chunking, embeddings, vector retrieval, and generation into one flow.
- Preserving conversation context for follow-up questions.
- Preventing unsupported answers by explicitly instructing the generator to rely only on retrieved context.

## Future Improvements

- Add hybrid BM25 + dense retrieval.
- Add cross-encoder reranking.
- Add RAGAS-based automated evaluation.
- Add streaming responses.
- Add multi-PDF collections and document management.
- Deploy a public demo on Hugging Face Spaces or another supported platform.

## Submission Checklist

- [ ] Add Epochs MUID.
- [ ] Verify the complete notebook/source files are in the repository.
- [ ] Deploy the application publicly.
- [ ] Add the public deployment URL here.
- [ ] Submit the GitHub repository and deployment URL in `#epochs-submissions` using `#evn-ds-epochs26-day11`.
