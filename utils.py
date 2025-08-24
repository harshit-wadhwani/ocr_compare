import io
import base64
import hashlib
from typing import List, Optional
from PIL import Image
import fitz  # PyMuPDF


def hash_bytes(b: bytes) -> str:
    """Generate SHA256 hash of bytes."""
    return hashlib.sha256(b).hexdigest()


def pdf_to_images(pdf_bytes: bytes, dpi: int = 180, max_pages: Optional[int] = None) -> List[Image.Image]:
    """Render PDF pages to PIL images using PyMuPDF (no external system deps)."""
    images = []
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page_count = len(doc)
    if max_pages:
        page_count = min(page_count, max_pages)
    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    for i in range(page_count):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(img)
    return images


def pil_to_data_uri(img: Image.Image, fmt="PNG") -> str:
    """Convert PIL Image to data URI."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{b64}"