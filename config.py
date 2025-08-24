import os
from dotenv import load_dotenv

load_dotenv()


DOC_LING_AVAILABLE = False
OPENAI_AVAILABLE = False  
ANTHROPIC_AVAILABLE = False
GEMINI_AVAILABLE = False
MISTRAL_AVAILABLE = False

openai_client = None
anthropic_client = None
mistral_client = None


def initialize_clients():
    global DOC_LING_AVAILABLE, OPENAI_AVAILABLE, ANTHROPIC_AVAILABLE
    global GEMINI_AVAILABLE, MISTRAL_AVAILABLE
    global openai_client, anthropic_client, mistral_client
    
    try:
        from docling.document_converter import DocumentConverter
        DOC_LING_AVAILABLE = True
    except Exception:
        DOC_LING_AVAILABLE = False

    try:
        from openai import OpenAI
        OPENAI_AVAILABLE = True
        openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    except Exception:
        OPENAI_AVAILABLE = False
        openai_client = None

    try:
        import anthropic
        ANTHROPIC_AVAILABLE = True
        anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    except Exception:
        ANTHROPIC_AVAILABLE = False
        anthropic_client = None

    try:
        import google.generativeai as genai
        if os.getenv("GOOGLE_API_KEY"):
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        GEMINI_AVAILABLE = True
    except Exception:
        GEMINI_AVAILABLE = False

    try:
        from mistralai import Mistral
        MISTRAL_AVAILABLE = True
        mistral_client = Mistral(api_key=os.getenv("MISTRAL_API_KEY")) if os.getenv("MISTRAL_API_KEY") else None
    except Exception:
        MISTRAL_AVAILABLE = False
        mistral_client = None

# Initialize on import
initialize_clients()