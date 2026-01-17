"""Test script for retrieval debugging"""

from vector_store import (
    load_vector_store, 
    hybrid_retrieve, 
    rerank_documents, 
    compute_evidence_score,
    QueryClassifier,
    fetch_sibling_chunks,
    RERANK_TOP_K
)
from langchain_core.documents import Document

def test_tif_chunks():
    """List all TIF chunks"""
    store = load_vector_store()
    
    # Get all TIF chunks
    results = store.get(
        where={"file_name": "TATA Innovation Fellowship (TIF) Programme _ Department of Biotechnology_[L2].pdf"},
        include=["documents", "metadatas"]
    )
    
    print(f"Total TIF chunks: {len(results['ids'])}")
    print()
    
    for i, (content, meta) in enumerate(zip(results["documents"], results["metadatas"])):
        chunk_id = meta.get("chunk_id", "?")
        has_phone = "011-" in content
        has_email = "[at]" in content
        has_name = "Dr." in content
        preview = content[:150].replace("\n", " ")
        
        print(f"Chunk {i}: {chunk_id}")
        print(f"  Has phone: {has_phone}, Has email: {has_email}, Has Dr.: {has_name}")
        print(f"  Preview: {preview}...")
        print()


def test_retrieval():
    """Test retrieval for contact query"""
    store = load_vector_store()
    query = "Who to contact for TIF Programme?"
    
    print(f"Query: {query}")
    print(f"Query type: {QueryClassifier.classify(query)}")
    print()
    
    # Hybrid retrieve
    docs = hybrid_retrieve(store, query, user_level=5, vector_k=12, bm25_k=6)
    print(f"Hybrid retrieve returned: {len(docs)} docs")
    
    # Check for TIF contact chunk
    target_chunk = "TATA Innovation Fellowship (TIF) Programme _ Department of Biotechnology_[L2].pdf:p0:c6"
    
    for doc in docs:
        cid = doc.metadata.get("chunk_id", "")
        if "TIF" in cid or "TATA" in cid:
            has_phone = "011-" in doc.page_content
            has_email = "[at]" in doc.page_content
            evidence = compute_evidence_score("contact", doc.page_content)
            print(f"  TIF chunk: {cid}")
            print(f"    Has phone: {has_phone}, Has email: {has_email}, Evidence: {evidence:.2f}")
    
    print()
    
    # Rerank
    reranked = rerank_documents(query, docs, top_k=RERANK_TOP_K)
    print(f"After reranking: {len(reranked)} docs")
    
    for i, doc in enumerate(reranked):
        cid = doc.metadata.get("chunk_id", "")[:60]
        evidence = compute_evidence_score("contact", doc.page_content)
        print(f"  {i+1}. {cid} (evidence: {evidence:.2f})")


if __name__ == "__main__":
    print("=" * 60)
    print("TEST 1: List all TIF chunks")
    print("=" * 60)
    test_tif_chunks()
    
    print()
    print("=" * 60)
    print("TEST 2: Test sibling chunk fetching")
    print("=" * 60)
    store = load_vector_store()
    
    # Create a fake doc with TIF chunk c5
    doc = Document(
        page_content="test",
        metadata={"chunk_id": "TATA Innovation Fellowship (TIF) Programme _ Department of Biotechnology_[L2].pdf:p0:c5"}
    )
    
    siblings = fetch_sibling_chunks(store, [doc], user_level=5)
    print(f"Found {len(siblings)} siblings from c5:")
    for s in siblings:
        cid = s.metadata.get("chunk_id", "?")
        has_phone = "011-" in s.page_content
        has_email = "[at]" in s.page_content
        print(f"  - {cid}")
        print(f"    Has phone: {has_phone}, Has email: {has_email}")
    
    print()
    print("=" * 60)
    print("TEST 3: Test full retrieval for contact query")
    print("=" * 60)
    test_retrieval()
