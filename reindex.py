"""
Re-index all documents with improved table-aware chunking.
Run this after updating ocr.py with new chunking logic.
"""

import sys
from pathlib import Path

from ocr import process_documents, DATA_PATH
from vector_store import load_vector_store, create_vector_store, DB_PATH

def reindex_all():
    """Re-index all documents with new chunking strategy."""
    print("=" * 60)
    print("RE-INDEXING WITH TABLE-AWARE CHUNKING")
    print("=" * 60)
    
    # Backup info
    print(f"\nData path: {DATA_PATH}")
    print(f"DB path: {DB_PATH}")
    
    # Confirm
    response = input("\nThis will delete the existing vector store and re-index all documents.\nContinue? (y/n): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Delete existing DB
    import shutil
    db_path = Path(DB_PATH)
    if db_path.exists():
        print(f"\nDeleting existing database at {db_path}...")
        shutil.rmtree(db_path)
        print("Deleted.")
    
    # Process all documents with new chunking
    print(f"\nProcessing documents from {DATA_PATH}...")
    docs = process_documents(
        data_path=DATA_PATH,
        chunk_size=512,
        chunk_overlap=128,
        max_workers=4
    )
    
    print(f"\nTotal documents processed: {len(docs)}")
    
    # Count contact blocks
    contact_blocks = [d for d in docs if d.metadata.get('is_contact_block')]
    print(f"Contact block chunks: {len(contact_blocks)}")
    
    # Create new vector store
    print("\nCreating vector store...")
    store = create_vector_store(docs)
    
    print("\n" + "=" * 60)
    print("RE-INDEXING COMPLETE")
    print("=" * 60)
    print(f"Total chunks: {len(docs)}")
    print(f"Contact chunks: {len(contact_blocks)}")
    
    # Show sample contact chunks
    if contact_blocks:
        print("\nSample contact chunks:")
        for cb in contact_blocks[:3]:
            print(f"  - {cb.metadata.get('chunk_id', '?')}")
            print(f"    {cb.page_content[:100]}...")


def reindex_single(filename: str):
    """Re-index a single file."""
    from ocr import DoclingOCREngine, RAGDocumentProcessor
    from vector_store import load_vector_store, delete_documents_by_filename, add_documents_to_store
    
    data_path = Path(DATA_PATH)
    files = list(data_path.glob(f"*{filename}*"))
    
    if not files:
        print(f"No files matching '{filename}' found in {DATA_PATH}")
        return
    
    file_path = files[0]
    print(f"Re-indexing: {file_path.name}")
    
    # Load store
    store = load_vector_store()
    
    # Delete existing chunks for this file
    delete_documents_by_filename(store, file_path.name)
    
    # Re-process file
    engine = DoclingOCREngine()
    result = engine.process_file(file_path)
    
    if result.success:
        processor = RAGDocumentProcessor()
        docs = processor.process_extracted_content(result.content)
        
        # Add new chunks
        add_documents_to_store(store, docs)
        
        contact_blocks = [d for d in docs if d.metadata.get('is_contact_block')]
        print(f"Added {len(docs)} chunks ({len(contact_blocks)} contact blocks)")
    else:
        print(f"Error processing file: {result.error}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Re-index single file
        reindex_single(sys.argv[1])
    else:
        # Re-index all
        reindex_all()
