from __future__ import annotations

import faiss
import numpy as np
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from model import get_embedding, get_llm


def _get_dim() -> int:
    embedding = get_embedding()
    test_vec = embedding.embed_query("dimension probe")
    return len(test_vec)


def create_vector_store(
    docs: list[Document],
    m: int = 32,
    ef_construction: int = 200,
) -> FAISS:
    embedding = get_embedding()
    dim = _get_dim()

    index = faiss.IndexHNSWFlat(dim, m)
    index.hnsw.efConstruction = ef_construction  # type: ignore[attr-defined]

    texts = [d.page_content for d in docs]
    vectors = [embedding.embed_query(t) for t in texts]
    arr = np.array(vectors, dtype="float32")
    index.add(arr)

    ids = [str(i) for i in range(len(docs))]
    docstore = InMemoryDocstore(dict(zip(ids, docs)))
    index_to_docstore_id = {i: ids[i] for i in range(len(ids))}

    store = FAISS(
        embedding_function=embedding,
        index=index,
        docstore=docstore,
        index_to_docstore_id=index_to_docstore_id,
    )
    print(
        f"Created HNSW FAISS store with {len(docs)} docs "
        f"(dim={dim}, M={m}, efConstruction={ef_construction})"
    )
    return store


def save_vector_store(store: FAISS, path: str = "faiss_hnsw_index") -> None:
    store.save_local(path)
    print(f"Saved FAISS HNSW index to {path}")


def load_vector_store(path: str = "faiss_hnsw_index") -> FAISS:
    embedding = get_embedding()
    store = FAISS.load_local(
        path,
        embedding,
        allow_dangerous_deserialization=True,
    )
    print(f"Loaded FAISS HNSW index from {path}")
    return store


def create_rag_chain(store: FAISS):
    llm = get_llm()
    retriever = store.as_retriever(search_kwargs={"k": 4})

    def format_docs(docs: list[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_template(
        """
        You are a concise assistant. Use only the provided context.

        Context:
        {context}

        Question: {question}

        Answer (do not mention missing context, just say you don't know if needed):
        """
    )

    rag_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | get_llm()
        | StrOutputParser()
    )
    return rag_chain


def query_vector_store(store: FAISS, question: str) -> str:
    chain = create_rag_chain(store)
    return chain.invoke(question)


def query_with_sources(store: FAISS, question: str) -> dict[str, object]:
    retriever = store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)

    llm = get_llm()
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = f"""
    You are a helpful assistant. Answer using only the context.

    Context:
    {context}

    Question: {question}

    Answer:
    """

    answer = llm.invoke(prompt)

    return {
        "answer": answer,
        "sources": [doc.metadata.get("chunk_id", "N/A") for doc in docs],
        "context": context,
    }
