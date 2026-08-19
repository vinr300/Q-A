"""
Document Q&A Chatbot using Retrieval-Augmented Generation (RAG)
-----------------------------------------------------------------
Loads a PDF, splits it into chunks, embeds those chunks, stores them
in a local FAISS vector index, and answers questions using the
Gemini LLM grounded in the document's actual content.

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

    print("\nDocument loaded. Ask questions about it (type 'exit' to quit).\n")

    while True:
        query = input("You: ").strip()
        if query.lower() in ("exit", "quit"):
            print("Goodbye!")
            break
        if not query:
            continue

        result = qa_chain.invoke({"query": query})
        print(f"\nBot: {result['result']}\n")

        # Optional: show which chunks were used to ground the answer
        sources = result.get("source_documents", [])
        if sources:
            pages_used = sorted({doc.metadata.get("page", "?") for doc in sources})
            print(f"(Answer grounded in page(s): {pages_used})\n")


if __name__ == "__main__":
    main()
