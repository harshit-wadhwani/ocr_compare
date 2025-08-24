# PDF → Markdown OCR Comparison Tool

A  Streamlit application for comparing different OCR (Optical Character Recognition) services when converting PDF documents to Markdown format. This tool allows you to test and compare the performance of various AI-powered OCR providers side-by-side.

## 📋 Supported OCR Providers

### Direct PDF Processing (No Model Selection Required)
- **Docling** 
- **Mistral OCR**

### Vision-Based Processing (Custom Model Names Supported)
- **OpenAI**
- **Claude**
- **Gemini**

## 🛠️ Installation

### Prerequisites
- Python 3.11
- API keys for the services you want to use

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/harshit-wadhwani/ocr_compare.git
   cd ocr_compare
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure API keys**
   
   Create a `.env` file in the project root:
   ```env
   OPENAI_API_KEY=your_openai_api_key_here
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   GOOGLE_API_KEY=your_google_api_key_here
   MISTRAL_API_KEY=your_mistral_api_key_here
   ```

## 🚀 Usage

1. **Start the application**
   ```bash
   streamlit run app.py
   ```

2. **Configure your setup**
   - Select OCR providers from the sidebar
   - For vision-based providers, specify model names
   - Adjust processing settings (DPI, max pages)

3. **Upload and process**
   - Upload a PDF file
   - View results from all selected providers
   - Download Markdown outputs

