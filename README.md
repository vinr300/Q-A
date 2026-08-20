# Document Q&A Chatbot (RAG + Agentic Routing)

A Retrieval-Augmented Generation (RAG) chatbot that answers questions
about any PDF document, with a lightweight agentic routing layer on top.
It uses LangChain to orchestrate retrieval, FAISS as a local vector store,
and Google's Gemini model to generate answers grounded in the document's
actual content — not just general model knowledge.

## How it works

1. **Load** — The PDF is loaded and split into overlapping text chunks.
2. **Embed** — Each chunk is converted into a vector using Gemini's
   embedding model.
3. **Store** — Vectors are stored locally in a FAISS index for fast
   similarity search.
4. **Route (agentic step)** — Before answering, a small "Router Agent"
   decides whether the user's message actually needs document retrieval
   (e.g. a factual question) or can be answered directly (e.g. "hi",
   "what can you do?"). This avoids unnecessary retrieval calls and is
   a basic example of agent decision-making rather than a fixed pipeline.
5. **Retrieve** — If routed to RETRIEVE, the most relevant chunks are
   pulled from FAISS.
6. **Generate** — Those chunks are passed to the Gemini LLM as context,
   so the answer is grounded in the document rather than hallucinated.
   If routed to DIRECT, the LLM answers the message on its own without
   touching the document.

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
You: hi
[router decision: DIRECT]
Bot: Hello! Ask me anything about the document you uploaded.

You: What is the main topic of this document?
[router decision: RETRIEVE]
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
- Add a second "verifier" agent that checks the RAG answer is actually
  supported by the retrieved chunks before showing it to the user
- Explore structuring this project's planning with agentic dev frameworks
  like BMAD-METHOD or GSD
