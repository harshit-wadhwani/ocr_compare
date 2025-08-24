"""OCR backend implementations."""
import os
import io
import base64
import json
import re
import tempfile
from typing import Dict, List, Optional
from PIL import Image
import streamlit as st
import google.generativeai as genai
from docling.document_converter import DocumentConverter

from config import (
    DOC_LING_AVAILABLE, OPENAI_AVAILABLE, ANTHROPIC_AVAILABLE,
    GEMINI_AVAILABLE, MISTRAL_AVAILABLE,
    openai_client, anthropic_client, mistral_client
)
from utils import pil_to_data_uri


def get_default_prompt() -> str:
    """Get the default OCR prompt for markdown conversion."""
    return (
        "You are an OCR engine. Read the page image and output ONLY Markdown that "
        "faithfully represents the content, preserving headings, lists, tables, and code blocks. "
        "Do not add commentary or extra text. If a table exists, render as GitHub Markdown table. "
        "If there are images, include a short alt text placeholder. Return Markdown only."
    )


def backend_docling(pdf_bytes: bytes) -> str:
    """Process PDF using Docling for direct PDF to Markdown conversion."""
    if not DOC_LING_AVAILABLE:
        raise RuntimeError("Docling not installed. pip install docling")

    conv = DocumentConverter()

    # Convert from a temp file (works with most versions)
    # Use delete=False to avoid permission issues on Windows
    tmp = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    try:
        tmp.write(pdf_bytes)
        tmp.flush()
        tmp.close()  # Close the file handle before conversion
        res = conv.convert(tmp.name)
    finally:
        # Clean up manually
        try:
            os.unlink(tmp.name)
        except Exception:
            pass  # Ignore cleanup errors

    # Some versions return a result with .document; in others, the document is the result
    doc = getattr(res, "document", res)

    # Try multiple known export method names on both doc and result
    def try_export_md(obj):
        for name in ("export_markdown", "export_to_markdown", "to_markdown", "as_markdown"):
            fn = getattr(obj, name, None)
            if callable(fn):
                try:
                    return fn()
                except TypeError:
                    # Some APIs expect a format arg
                    try:
                        return fn("markdown")
                    except Exception:
                        pass
        # Generic export(format=...) style
        fn = getattr(obj, "export", None)
        if callable(fn):
            for fmt in ("markdown", "md"):
                try:
                    return fn(fmt)
                except TypeError:
                    try:
                        return fn(format=fmt)
                    except Exception:
                        pass
        return None

    md = try_export_md(doc) or try_export_md(res)
    if isinstance(md, str) and md.strip():
        return md

    # If we get here, your Docling version doesn't expose a direct Markdown exporter.
    # Suggest upgrading or enabling extras.
    raise RuntimeError(
        "Docling Markdown export method not found in your installed version. "
        "Try: pip install --upgrade docling or pip install 'docling[extras]'. "
        "If it still fails, run 'pip show docling' and share the version."
    )


def backend_openai_page(img: Image.Image, model_name: str = "gpt-4o", custom_prompt: Optional[str] = None) -> str:
    """Process single page with OpenAI vision model."""
    if not OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI not installed or OPENAI_API_KEY missing.")
    data_uri = pil_to_data_uri(img, "PNG")
    prompt = custom_prompt if custom_prompt else get_default_prompt()
    # Using Chat Completions with vision
    resp = openai_client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ]}
        ],
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


def backend_claude_page(img: Image.Image, model_name: str = "claude-3-5-sonnet-latest", custom_prompt: Optional[str] = None) -> str:
    """Process single page with Claude vision model."""
    if not ANTHROPIC_AVAILABLE:
        raise RuntimeError("Anthropic not installed or ANTHROPIC_API_KEY missing.")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    prompt = custom_prompt if custom_prompt else get_default_prompt()
    resp = anthropic_client.messages.create(
        model=model_name,
        max_tokens=4000,
        temperature=0.0,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": b64}},
                ],
            }
        ],
    )
    return resp.content[0].text.strip()


def backend_gemini_page(img: Image.Image, model_name: str = "gemini-2.5-pro", custom_prompt: Optional[str] = None) -> str:
    """Process single page with Gemini vision model."""
    if not GEMINI_AVAILABLE or not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("google-generativeai not installed or GOOGLE_API_KEY missing.")
    
    model = genai.GenerativeModel(model_name)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    prompt = custom_prompt if custom_prompt else get_default_prompt()
    # Gemini accepts a list of parts; image bytes go as a dict with mime_type
    result = model.generate_content(
        [prompt, {"mime_type": "image/png", "data": buf.getvalue()}],
        generation_config={"temperature": 0.0},
    )
    return (result.text or "").strip()


def run_llm_ocr_over_pages(images: List[Image.Image], per_page_fn, cache_key_prefix: str, model_name: Optional[str] = None, custom_prompt: Optional[str] = None) -> str:
    """Process multiple pages with an LLM OCR backend."""
    outputs = []
    progress = st.progress(0.0)
    total = len(images)
    # Cache per page to avoid re-running on same PDF
    for idx, img in enumerate(images):
        page_key = f"{cache_key_prefix}-page-{idx}"
        if model_name:
            page_key = f"{cache_key_prefix}-{model_name}-page-{idx}"
        if custom_prompt:
            # Include a hash of the prompt in cache key to handle prompt changes
            import hashlib
            prompt_hash = hashlib.md5(custom_prompt.encode()).hexdigest()[:8]
            page_key = f"{page_key}-prompt-{prompt_hash}"
        
        md = st.session_state.get(page_key)
        if not md:
            try:
                if model_name and custom_prompt:
                    md = per_page_fn(img, model_name, custom_prompt)
                elif model_name:
                    md = per_page_fn(img, model_name)
                elif custom_prompt:
                    md = per_page_fn(img, custom_prompt=custom_prompt)
                else:
                    md = per_page_fn(img)
            except Exception as e:
                md = f"<!-- Error on page {idx+1}: {e} -->\n"
            st.session_state[page_key] = md
        outputs.append(f"\n\n<!-- Page {idx+1} -->\n{md}")
        progress.progress((idx + 1) / total)
    return "\n".join(outputs)


def backend_mistral_pdf2md(pdf_bytes: bytes, model_name: str = "mistral-ocr-latest") -> str:
    """
    Process PDF using Mistral OCR API with proper client implementation.
    Requires MISTRAL_API_KEY environment variable.
    Note: Mistral OCR doesn't support custom prompts as it's a specialized OCR model.
    """
    if not MISTRAL_AVAILABLE:
        raise RuntimeError("mistralai package not installed. pip install mistralai")
    
    if not mistral_client:
        raise RuntimeError("Set MISTRAL_API_KEY environment variable.")

    # Encode PDF to base64
    base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')

    # Process the document with Mistral OCR
    ocr_response = mistral_client.ocr.process(
        model=model_name,
        document={
            "type": "document_url",
            "document_url": f"data:application/pdf;base64,{base64_pdf}" 
        },
        include_image_base64=True
    )
    
    # Convert response to dict
    response_dict = json.loads(ocr_response.model_dump_json())
    
    # Process the response to markdown with image handling
    return _convert_mistral_response_to_markdown(response_dict)


def _convert_mistral_response_to_markdown(response_dict: Dict) -> str:
    """
    Convert Mistral OCR response to markdown with proper image handling.
    This function handles the image extraction and markdown processing.
    """
    md = ""
    
    # Dictionary to keep track of image IDs and their paths
    # Since we're in Streamlit, we'll embed images as base64 data URIs
    image_map = {}
    
    # First pass: process all images and create the mapping
    for page in response_dict.get("pages", []):
        if page.get("images"):
            for image in page["images"]:
                # Get image ID or generate one if not available
                image_id = image.get("id", f"img-{page['index']}.jpeg")
                
                # Ensure the image has a file extension
                if not any(image_id.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif']):
                    image_id = f"{image_id}.jpeg"
                
                # Extract the base64 data
                img_data = image["image_base64"]
                
                # Check if the base64 string has a data:image prefix, if so, remove it
                if "," in img_data:
                    img_data = img_data.split(",", 1)[1]
                
                # Create a data URI for the image (for Streamlit display)
                data_uri = f"data:image/jpeg;base64,{img_data}"
                image_map[image_id] = data_uri
    
    # Second pass: process the markdown with image references replaced
    for page in response_dict.get("pages", []):
        # Get the markdown content for this page
        page_md = page.get("markdown", "")
        
        # Replace image references in the markdown content
        # Use regex to find and replace image references like ![img-name.ext](img-name.ext)
        for img_id, img_path in image_map.items():
            pattern = f"!\\[{re.escape(img_id)}\\]\\({re.escape(img_id)}\\)"
            replacement = f"![{img_id}]({img_path})"
            page_md = re.sub(pattern, replacement, page_md)
        
        md += page_md + "\n"
        
        # Add any images that weren't referenced in the markdown
        if page.get("images"):
            for image in page["images"]:
                image_id = image.get("id", f"img-{page['index']}.jpeg")
                if not any(image_id.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif']):
                    image_id = f"{image_id}.jpeg"
                    
                # Check if this image was already referenced in the markdown
                pattern = f"!\\[{re.escape(image_id)}\\]\\("
                if not re.search(pattern, page_md):
                    # Add the unreferenced image
                    md += f"\n![{image_id}]({image_map.get(image_id, '')})\n"
    
    return md.strip()


# For backward compatibility, keep the old function names
def backend_gpt4o_page(img: Image.Image) -> str:
    """Backward compatibility wrapper for OpenAI GPT-4o."""
    return backend_openai_page(img, "gpt-4o")