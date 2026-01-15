from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from config import DATA_PATH


def load_pdfs(data_path: str = DATA_PATH) -> list[Document]:
    """Load all PDFs from directory"""
    pdf_loader = PyPDFDirectoryLoader(data_path)
    documents = pdf_loader.load()
    print(f"Loaded {len(documents)} PDF documents")
    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    """Split documents into overlapping chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} chunks")
    return chunks


def add_chunk_ids(chunks: list[Document]) -> list[Document]:
    """Add unique IDs: filename:page:chunk_index"""
    last_page_id = None
    chunk_idx = 0

    for chunk in chunks:
        source = Path(chunk.metadata.get("source", "")).name
        page = chunk.metadata.get("page", 0)
        page_id = f"{source}:{page}"

        if page_id == last_page_id:
            chunk_idx += 1
        else:
            chunk_idx = 0

        chunk.metadata["chunk_id"] = f"{page_id}:{chunk_idx}"
        last_page_id = page_id

    print(f"Added IDs to {len(chunks)} chunks")
    return chunks


def process_pdfs(data_path: str = DATA_PATH) -> list[Document]:
    """Complete pipeline: PDFs → ready chunks"""
    docs = load_pdfs(data_path)
    chunks = split_documents(docs)
    return add_chunk_ids(chunks)
