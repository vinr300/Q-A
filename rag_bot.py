"""
Document Q&A Chatbot using Retrieval-Augmented Generation (RAG)
with a simple Agentic Routing step
-----------------------------------------------------------------
Loads a PDF, splits it into chunks, embeds those chunks, stores them
in a local FAISS vector index, and answers questions using the
Gemini LLM.

Agentic behavior: before answering, a lightweight "Router Agent"
decides whether the question actually needs document retrieval
(RAG) or can be answered directly (e.g. greetings, meta questions
like "what can you do?"). This avoids unnecessary retrieval calls
and demonstrates basic agent decision-making on top of plain RAG.

Usage:
    python rag_bot.py path/to/document.pdf
"""

import os
import sys
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA


ROUTER_PROMPT = """You are a routing agent for a document Q&A assistant.
Decide whether the user's message requires searching the uploaded document
to answer accurately, or whether it can be answered directly without it.

Answer with exactly one word: "RETRIEVE" or "DIRECT".

Use "DIRECT" for:
- Greetings (hi, hello, thanks, bye)
- Meta questions about the assistant itself (what can you do, how do you work)
- General knowledge questions clearly unrelated to a specific document

Use "RETRIEVE" for:
- Any question that could be about the content, facts, or details of the uploaded document

User message: {message}

Answer (one word only):"""


def load_and_split(pdf_path: str):
    """Load a PDF and split it into overlapping text chunks."""
    print(f"Loading document: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    )
    chunks = splitter.split_documents(pages)
    print(f"Split into {len(chunks)} chunks.")
    return chunks


def build_vector_store(chunks, api_key: str):
    """Embed chunks and store them in a local FAISS index."""
    print("Generating embeddings and building FAISS index...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=api_key,
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def build_qa_chain(vector_store, api_key: str):
    """Connect the retriever to the Gemini LLM via a RetrievalQA chain."""
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0.2,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
    )
    return qa_chain


def build_router_llm(api_key: str):
    """A separate lightweight LLM call used purely for routing decisions."""
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        google_api_key=api_key,
        temperature=0,
    )


def route_query(router_llm, user_message: str) -> str:
    """
    Agentic routing step.
    Returns 'RETRIEVE' or 'DIRECT' depending on whether the query
    needs the document search pipeline.
    """
    prompt = ROUTER_PROMPT.format(message=user_message)
    response = router_llm.invoke(prompt)
    decision = response.content.strip().upper()

    # Fail-safe: if the router gives an unexpected output, default to RETRIEVE
    # so we never silently skip grounding on document content.
    if "DIRECT" in decision:
        return "DIRECT"
    return "RETRIEVE"


def answer_direct(router_llm, user_message: str) -> str:
    """Answer simple/meta queries without touching the document at all."""
    prompt = (
        "You are a friendly document Q&A assistant. "
        "Respond briefly and naturally to this message, without inventing "
        "any document content:\n\n"
        f"{user_message}"
    )
    response = router_llm.invoke(prompt)
    return response.content.strip()


def main():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        print("ERROR: GOOGLE_API_KEY not found. Add it to a .env file.")
        sys.exit(1)

    if len(sys.argv) < 2:
        print("Usage: python rag_bot.py path/to/document.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    if not os.path.exists(pdf_path):
        print(f"ERROR: File not found: {pdf_path}")
        sys.exit(1)

    chunks = load_and_split(pdf_path)
    vector_store = build_vector_store(chunks, api_key)
    qa_chain = build_qa_chain(vector_store, api_key)
    router_llm = build_router_llm(api_key)

    print("\nDocument loaded. Ask questions about it (type 'exit' to quit).\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if not query:
            continue

        # --- Agentic routing step ---
        decision = route_query(router_llm, query)
        print(f"[router decision: {decision}]")

        if decision == "DIRECT":
            answer = answer_direct(router_llm, query)
            print(f"\nBot: {answer}\n")
            continue

        # --- RAG path ---
        result = qa_chain.invoke({"query": query})
        print(f"\nBot: {result['result']}\n")

        sources = result.get("source_documents", [])
        if sources:
            pages_used = sorted({doc.metadata.get("page", "?") for doc in sources})
            print(f"(Answer grounded in page(s): {pages_used})\n")


if __name__ == "__main__":
    main()
