# Document Q&A Chatbot (RAG)

A simple Retrieval-Augmented Generation (RAG) chatbot that answers questions
about any PDF document. It uses LangChain to orchestrate retrieval, FAISS
as a local vector store, and Google's Gemini model to generate answers
grounded in the document's actual content — not just general model knowledge.

## How it works

1. **Load** — The PDF is loaded and split into overlapping text chunks.
2. **Embed** — Each chunk is converted into a vector using Gemini's
   embedding model.
3. **Store** — Vectors are stored locally in a FAISS index for fast
   similarity search.
4. **Retrieve** — When you ask a question, the most relevant chunks are
   retrieved from FAISS.
5. **Generate** — Those chunks are passed to the Gemini LLM as context,
   so the answer is grounded in the document rather than hallucinated.

## Setup

1. Clone this repo and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Get a free API key from [Google AI Studio](https://aistudio.google.com).

3. Copy `.env.example` to `.env` and add your key:
   ```bash
   cp .env.example .env
   ```

4. Run it on any PDF:
   ```bash
   python rag_bot.py path/to/document.pdf
   ```

5. Ask questions in the terminal. Type `exit` to quit.

## Example

```
You: What is the main topic of this document?
Bot: The document discusses ...
(Answer grounded in page(s): [1, 2])
```

## Tech stack

- Python
- LangChain
- Google Gemini (LLM + embeddings)
- FAISS (vector store)
- PyPDF (document loading)

## Possible improvements

- Support multiple documents at once
- Add a simple web UI with Streamlit
- Add conversation memory for follow-up questions
- Swap FAISS for a hosted vector DB (e.g., Pinecone) for larger datasets
