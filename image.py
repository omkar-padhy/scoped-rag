"""
Image Processing Module
Supports: PNG, JPG, JPEG with OCR and Vision Model Captioning
Uses Qwen3-VL via Ollama for text extraction and image description
"""

import base64
from pathlib import Path
from typing import Optional

import ollama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from config import DATA_PATH, VISION_MODEL

from model import get_image_description

# Vision model for OCR and captioning
# VISION_MODEL imported from config
SUPPORTED_FORMATS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}


def encode_image_base64(image_path: Path) -> str:
    """Encode image to base64 string for Ollama vision model"""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_image_format(image_path: Path) -> str:
    """Get image format from file extension"""
    suffix = image_path.suffix.lower()
    if suffix == ".jpg":
        return "JPEG"
    return suffix[1:].upper()


def extract_ocr_and_description(image_path: Path) -> dict:
    """
    Use Qwen3-VL via Ollama to extract OCR text and generate description.
    Returns dict with: file_name, format, ocr_content, description
    
    This combines OCR text extraction with semantic image captioning
    for RAG-friendly document representation.
    """
    image_base64 = encode_image_base64(image_path)
    
    # OCR extraction prompt - optimized for accuracy
    ocr_prompt = """Extract ALL text visible in this image using OCR.
Return ONLY the extracted text in a clean, readable format.
Preserve the original structure, formatting, and layout where possible.
Include all text from headers, body, tables, labels, and captions.
If no text is found, return 'NO_TEXT_FOUND'."""
    
    # Description/Caption prompt - optimized for RAG retrieval
    description_prompt = """Provide a comprehensive description of this image for document retrieval:

1. **Document Type**: What kind of document/image is this? (invoice, receipt, diagram, photo, chart, form, etc.)
2. **Main Subject**: What is the primary content or subject matter?
3. **Key Information**: List the most important facts, numbers, or data points visible.
4. **Visual Elements**: Describe diagrams, charts, tables, or graphical elements.
5. **Context & Purpose**: What is this document likely used for?

Be detailed but concise. Focus on searchable, retrievable information."""
    
    ocr_content = ""
    description = ""
    
    try:
        # Extract OCR content
        ocr_response = ollama.chat(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": ocr_prompt,
                "images": [image_base64]
            }]
        )
        ocr_content = ocr_response["message"]["content"].strip()
        
        # Generate semantic description/caption
        desc_response = ollama.chat(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": description_prompt,
                "images": [image_base64]
            }]
        )
        description = desc_response["message"]["content"].strip()
        
    except Exception as e:
        print(f"Error processing {image_path.name}: {e}")
        ocr_content = "ERROR_EXTRACTING_TEXT"
        description = f"Image file: {image_path.name} (processing failed)"
    
    return {
        "file_name": image_path.name,
        "file_path": str(image_path),
        "format": get_image_format(image_path),
        "ocr_content": ocr_content,
        "description": description
    }


def load_images(data_path: str = DATA_PATH) -> list[dict]:
    """Load all supported images from directory and extract OCR + description"""
    data_dir = Path(data_path)
    image_data = []
    
    if not data_dir.exists():
        print(f"Data path {data_path} does not exist")
        return image_data
    
    image_files = [
        f for f in data_dir.iterdir() 
        if f.is_file() and f.suffix.lower() in SUPPORTED_FORMATS
    ]
    
    print(f"Found {len(image_files)} images to process")
    
    for img_path in image_files:
        print(f"Processing: {img_path.name}")
        extracted = extract_ocr_and_description(img_path)
        image_data.append(extracted)
        
        # Show OCR status
        has_text = extracted['ocr_content'] not in ["NO_TEXT_FOUND", "ERROR_EXTRACTING_TEXT", ""]
        ocr_status = "✓" if has_text else "✗"
        print(f"  {ocr_status} Format: {extracted['format']}, OCR: {len(extracted['ocr_content'])} chars")
    
    print(f"Loaded {len(image_data)} image documents")
    return image_data


def create_image_documents(image_data: list[dict]) -> list[Document]:
    """
    Convert extracted image data to LangChain Documents
    Combines OCR content and vision caption for RAG-friendly output
    """
    documents = []
    
    for data in image_data:
        # Build comprehensive page content combining OCR and description
        content_parts = []
        
        # Header with metadata
        content_parts.append(f"# Image: {data['file_name']}")
        content_parts.append(f"[Format: {data['format']}]")
        content_parts.append("")
        
        # Vision model description/caption
        if data.get("description"):
            content_parts.append("## Image Description")
            content_parts.append(data["description"])
            content_parts.append("")
        
        # OCR extracted text
        has_ocr = data["ocr_content"] and data["ocr_content"] not in ["NO_TEXT_FOUND", "ERROR_EXTRACTING_TEXT"]
        if has_ocr:
            content_parts.append("## Extracted Text (OCR)")
            content_parts.append(data["ocr_content"])
        
        page_content = "\n".join(content_parts)
        
        doc = Document(
            page_content=page_content,
            metadata={
                "source": data.get("file_path", data["file_name"]),
                "file_name": data["file_name"],
                "format": data["format"],
                "type": "image",
                "doc_type": "image",
                "has_ocr": has_ocr,
                "has_caption": bool(data.get("description")),
            }
        )
        documents.append(doc)
    
    return documents


def split_image_documents(documents: list[Document]) -> list[Document]:
    """Split image documents into chunks (same as text processing)"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Created {len(chunks)} image chunks")
    return chunks


def add_image_chunk_ids(chunks: list[Document]) -> list[Document]:
    """Add unique IDs: filename:chunk_index"""
    last_source = None
    chunk_idx = 0
    
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        
        if source == last_source:
            chunk_idx += 1
        else:
            chunk_idx = 0
        
        chunk.metadata["chunk_id"] = f"{source}:{chunk_idx}"
        last_source = source
    
    print(f"Added IDs to {len(chunks)} image chunks")
    return chunks


def process_images(data_path: str = DATA_PATH) -> list[Document]:
    """Complete pipeline: Images → OCR + Description → ready chunks"""
    image_data = load_images(data_path)
    
    if not image_data:
        print("No images found to process")
        return []
    
    documents = create_image_documents(image_data)
    chunks = split_image_documents(documents)
    return add_image_chunk_ids(chunks)


if __name__ == "__main__":
    # Test the image processing pipeline
    chunks = process_images()
    
    if chunks:
        print("\n--- Sample Output ---")
        for i, chunk in enumerate(chunks[:2]):
            print(f"\nChunk {i + 1}:")
            print(f"  ID: {chunk.metadata.get('chunk_id')}")
            print(f"  Type: {chunk.metadata.get('type')}")
            print(f"  Format: {chunk.metadata.get('format')}")
            print(f"  Has OCR: {chunk.metadata.get('has_ocr')}")
            print(f"  Content preview: {chunk.page_content[:200]}...")
