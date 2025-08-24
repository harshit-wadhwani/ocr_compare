"""Main Streamlit application for PDF to Markdown OCR comparison."""
import os
import time
import streamlit as st

from config import (
    DOC_LING_AVAILABLE, OPENAI_AVAILABLE, ANTHROPIC_AVAILABLE,
    GEMINI_AVAILABLE, MISTRAL_AVAILABLE, mistral_client
)
from utils import hash_bytes, pdf_to_images
from ocr_backend import (
    backend_docling, backend_openai_page, backend_claude_page,
    backend_gemini_page, backend_mistral_pdf2md, run_llm_ocr_over_pages,
    get_default_prompt
)


def get_available_providers():
    """Get list of available OCR providers based on installed packages and API keys."""
    providers = []
    
    if DOC_LING_AVAILABLE:
        providers.append("Docling")
    
    if MISTRAL_AVAILABLE and mistral_client:
        providers.append("Mistral OCR")
    
    if OPENAI_AVAILABLE:
        providers.append("OpenAI")
        
    if ANTHROPIC_AVAILABLE:
        providers.append("Claude")
        
    if GEMINI_AVAILABLE and os.getenv("GOOGLE_API_KEY"):
        providers.append("Gemini")
    
    return providers


def get_default_model(provider: str) -> str:
    """Get default model for each provider."""
    defaults = {
        "OpenAI": "gpt-4o",
        "Claude": "claude-3-5-sonnet-latest", 
        "Gemini": "gemini-2.5-pro"
    }
    return defaults.get(provider, "")


def process_pdf_with_backend(provider: str, model_name: str, pdf_bytes: bytes, pdf_hash: str, custom_prompt: str = None, images: list = None):
    """Process PDF with the specified backend provider and model."""
    t0 = time.time()
    md_output = None
    error = None

    try:
        if provider == "Docling":
            with st.spinner("Converting via Docling..."):
                cache_key = f"{pdf_hash}-docling"
                md_output = st.session_state.get(cache_key)
                if not md_output:
                    md_output = backend_docling(pdf_bytes)
                    st.session_state[cache_key] = md_output

        elif provider == "Mistral OCR":
            with st.spinner("Processing with Mistral OCR..."):
                cache_key = f"{pdf_hash}-mistral-ocr"
                md_output = st.session_state.get(cache_key)
                if not md_output:
                    md_output = backend_mistral_pdf2md(pdf_bytes)
                    st.session_state[cache_key] = md_output

        elif provider == "OpenAI" and images:
            with st.spinner(f"OCR with {model_name}..."):
                cache_key_prefix = f"{pdf_hash}-openai"
                md_output = run_llm_ocr_over_pages(images, backend_openai_page, cache_key_prefix, model_name, custom_prompt)

        elif provider == "Gemini" and images:
            with st.spinner(f"OCR with {model_name}..."):
                cache_key_prefix = f"{pdf_hash}-gemini"
                md_output = run_llm_ocr_over_pages(images, backend_gemini_page, cache_key_prefix, model_name, custom_prompt)

        elif provider == "Claude" and images:
            with st.spinner(f"OCR with {model_name}..."):
                cache_key_prefix = f"{pdf_hash}-claude"
                md_output = run_llm_ocr_over_pages(images, backend_claude_page, cache_key_prefix, model_name, custom_prompt)

        else:
            error = "Backend not available or no images could be rendered."
            
    except NotImplementedError as e:
        error = str(e)
    except Exception as e:
        error = f"Error: {e}"

    elapsed = time.time() - t0
    return md_output, error, elapsed


def main():
    """Main Streamlit application."""
    st.set_page_config(page_title="PDF → Markdown OCR Comparison", layout="wide")
    st.title("PDF → Markdown OCR Comparison")

    # Sidebar settings
    with st.sidebar:
        st.header("Settings")
        max_pages = st.number_input("Max pages (for quick tests)", min_value=1, max_value=200, value=5, step=1)
        dpi = st.slider("Render DPI for LLM OCR (higher -> better, slower)", min_value=120, max_value=300, value=180, step=10)
        
        st.header("Model Configuration")
        
        # Provider and model selection
        available_providers = get_available_providers()
        selected_providers = st.multiselect(
            "Choose OCR Providers", 
            options=available_providers,
            default=available_providers[:2] if len(available_providers) >= 2 else available_providers
        )
        
        # Model configuration for each selected provider
        provider_models = {}
        for provider in selected_providers:
            if provider in ["Docling", "Mistral OCR"]:
                provider_models[provider] = None  # These don't need model selection
            else:
                default_model = get_default_model(provider)
                model_name = st.text_input(
                    f"{provider} Model", 
                    value=default_model,
                    key=f"{provider}_model",
                    help=f"Enter the model name for {provider} (e.g., {default_model})"
                )
                provider_models[provider] = model_name
        
        # Prompt customization section
        st.header("Prompt Customization")
        
        # Check if any vision providers are selected
        vision_providers = ["OpenAI", "Claude", "Gemini"]
        has_vision_providers = any(provider in vision_providers for provider in selected_providers)
        
        if has_vision_providers:
            use_custom_prompt = st.checkbox(
                "Use Custom Prompt", 
                value=False,
                help="Enable to customize the OCR prompt for vision models (OpenAI, Claude, Gemini)"
            )
            
            if use_custom_prompt:
                custom_prompt = st.text_area(
                    "Custom OCR Prompt",
                    value=get_default_prompt(),
                    height=150,
                    help="This prompt will be sent to the vision models. Customize it to change how the models interpret and extract content from your PDFs.",
                    placeholder="Enter your custom OCR prompt here..."
                )
            else:
                custom_prompt = None
                
            # Show default prompt for reference
            with st.expander("View Default Prompt"):
                st.code(get_default_prompt(), language="text")
        else:
            custom_prompt = None
            if selected_providers:
                st.info("Custom prompts only apply to vision models (OpenAI, Claude, Gemini). Currently selected providers use built-in processing.")

    # File upload
    uploaded = st.file_uploader("Upload a PDF", type=["pdf"])

    if uploaded is not None:
        pdf_bytes = uploaded.read()
        pdf_hash = hash_bytes(pdf_bytes)
        st.write(f"PDF loaded. Size: {len(pdf_bytes)/1024:.1f} KB")

        # Pre-render pages once for LLM OCR paths
        vision_providers = ["OpenAI", "Claude", "Gemini"]
        need_images = any(provider in vision_providers for provider in selected_providers)
        images = []
        if need_images:
            with st.spinner("Rendering PDF pages to images..."):
                try:
                    images = pdf_to_images(pdf_bytes, dpi=dpi, max_pages=max_pages)
                    st.write(f"Rendered {len(images)} page(s).")
                except Exception as e:
                    st.error(f"Failed to render PDF pages: {e}")

        # Process with selected backends
        if len(selected_providers) > 0:
            cols = st.columns(len(selected_providers))

            for i, provider in enumerate(selected_providers):
                with cols[i]:
                    model_name = provider_models.get(provider)
                    display_name = f"{provider}" + (f" ({model_name})" if model_name else "")
                    if custom_prompt and provider in vision_providers:
                        display_name += " [Custom Prompt]"
                    st.subheader(display_name)
                    
                    # Validate model name for providers that need it
                    if provider not in ["Docling", "Mistral OCR"] and not model_name:
                        st.error(f"Please enter a model name for {provider}")
                        continue
                    
                    # Pass custom prompt only to vision providers
                    prompt_to_use = custom_prompt if provider in vision_providers else None
                    
                    md_output, error, elapsed = process_pdf_with_backend(
                        provider, model_name, pdf_bytes, pdf_hash, prompt_to_use, images
                    )

                    if md_output:
                        st.caption(f"Done in {elapsed:.1f}s")
                        filename_suffix = f"{provider.lower().replace(' ', '_')}" + (f"_{model_name}" if model_name else "")
                        if custom_prompt and provider in vision_providers:
                            filename_suffix += "_custom_prompt"
                        st.download_button(
                            "Download Markdown", 
                            data=md_output.encode("utf-8"), 
                            file_name=f"{os.path.splitext(uploaded.name)[0]}_{filename_suffix}.md", 
                            mime="text/markdown"
                        )
                        st.text_area("Preview (Markdown)", value=md_output[:20000], height=300, key=f"preview_{i}")
                    else:
                        st.error(error or "No output")
        else:
            st.warning("Please select at least one OCR provider from the sidebar.")

    else:
        st.info("Upload a PDF to begin.")

    # Prompt examples
    with st.expander("Custom Prompt Examples"):
        st.write("**Extract Only Tables:**")
        st.code("""Extract only tables from this image. Convert each table to GitHub Markdown format. 
If no tables are found, return "No tables found on this page".""")
        
        st.write("**Focus on Specific Content:**")
        st.code("""Extract only headings and bullet points from this document page. 
Format as Markdown with proper heading levels (# ## ###) and bullet lists. 
Ignore images, tables, and regular paragraphs.""")
        
        st.write("**Detailed Extraction:**")
        st.code("""You are a detailed OCR engine. Extract ALL content from this page including:
- All text in reading order
- Table structures with proper formatting
- Image descriptions and captions
- Any mathematical formulas or equations
- Preserve exact formatting and layout
Output in clean Markdown format.""")


if __name__ == "__main__":
    main()