import sys
import argparse

from audio import process_audio
from image import process_images
from text import process_pdfs
from ocr import (
    process_documents,
    process_pdfs_with_ocr,
    process_office_documents,
    process_images_with_ocr_and_caption,
    get_file_stats,
)
from vector_store import (
    create_vector_store,
    load_vector_store,
    query_vector_store,
    save_vector_store,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Scoped RAG - Document Q&A System")
    parser.add_argument("question", nargs="?", default="summarize documents", help="Question to ask")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild the index")
    parser.add_argument("--use-docling", action="store_true", help="Use Docling OCR for document processing")
    parser.add_argument("--data-path", default="data", help="Path to documents directory")
    return parser.parse_args()


def build_index_legacy(data_path: str = "data") -> list:
    """Legacy index building using original processors"""
    print("Using legacy document processors...")
    
    # Process PDFs
    pdf_chunks = process_pdfs(data_path)
    
    # Process Images (PNG, JPEG, JPG)
    image_chunks = process_images(data_path)
    
    # Process Audio (MP3, WAV, etc.)
    audio_chunks = process_audio(data_path)
    
    # Combine all chunks
    all_chunks = pdf_chunks + image_chunks + audio_chunks
    print(
        f"Total chunks: {len(all_chunks)} (PDFs: {len(pdf_chunks)}, Images: {len(image_chunks)}, Audio: {len(audio_chunks)})"
    )
    
    return all_chunks


def build_index_docling(data_path: str = "data") -> list:
    """
    Build index using Docling OCR engine
    Supports: PDF, DOCX, PPTX, XLSX, and Images with OCR + Vision Captioning
    """
    print("Using Docling OCR for document processing...")
    
    # Show file statistics
    stats = get_file_stats(data_path)
    print(f"Files found: {stats}")
    
    # Process all documents with Docling
    # This handles PDFs, DOCX, PPTX, XLSX with OCR and table extraction
    doc_chunks = process_documents(data_path)
    
    # Process Audio (MP3, WAV, etc.) - still use legacy for audio
    audio_chunks = process_audio(data_path)
    
    # Combine all chunks
    all_chunks = doc_chunks + audio_chunks
    print(
        f"Total chunks: {len(all_chunks)} (Documents: {len(doc_chunks)}, Audio: {len(audio_chunks)})"
    )
    
    return all_chunks


def main():
    args = parse_args()
    question = args.question
    
    # Check if we need to rebuild or load existing index
    need_rebuild = args.rebuild
    store = None
    
    if not need_rebuild:
        try:
            store = load_vector_store()
            print("Loaded existing index.")
        except Exception as e:
            print(f"No existing index found: {e}")
            need_rebuild = True
    
    if need_rebuild:
        print("Creating index...")
        
        # Choose processing method
        if args.use_docling:
            all_chunks = build_index_docling(args.data_path)
        else:
            all_chunks = build_index_legacy(args.data_path)
        
        if not all_chunks:
            print("No documents found to process!")
            return
        
        store = create_vector_store(all_chunks)
        save_vector_store(store)
    
    # Answer question
    print(f"\nQuestion: {question}")
    print("-" * 50)
    answer = query_vector_store(store, question)
    print(answer)


if __name__ == "__main__":
    main()
