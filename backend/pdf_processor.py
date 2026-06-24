import fitz  # PyMuPDF
import pdfplumber

def extract_text_from_pdf(pdf_path: str) -> str:
    """Try pdfplumber first, fallback to PyMuPDF."""
    text = ""

    # Method 1: pdfplumber (excellent for structured tabular layouts or certificates)
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text:
                    text += f"\n[Page {i+1}]\n{page_text}"
    except Exception as e:
        print(f"[pdf_processor] pdfplumber failed: {e}, trying PyMuPDF...")

    # Method 2: PyMuPDF fallback
    if len(text.strip()) < 50:
        text = ""
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc):
            text += f"\n[Page {i+1}]\n{page.get_text()}"
        doc.close()

    print(f"[pdf_processor] Extracted {len(text)} characters")
    return text


def split_into_chunks(text: str, chunk_size: int = 600, overlap: int = 150) -> list[dict]:
    """Split text into larger overlapping chunks to ensure metadata and dates stay bound together."""
    words = text.split()
    if not words:
        return []
    chunks = []
    start = 0
    while start < len(words):
        chunk_text = " ".join(words[start:start + chunk_size])
        chunks.append({"index": len(chunks), "text": chunk_text})
        start += chunk_size - overlap
    return chunks


def process_pdf(pdf_path: str) -> list[dict]:
    """Orchestrate extraction and expanded semantic grouping."""
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        raise ValueError("No text found in PDF. It may be a scanned image-only PDF.")
    chunks = split_into_chunks(text)
    print(f"[pdf_processor] Created {len(chunks)} chunks from {pdf_path}")
    return chunks