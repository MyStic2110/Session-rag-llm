# LumeHealth | LLM Intelligence Node

The specialized microservice powering LumeHealth's medical and insurance analysis pipeline. This service manages high-complexity reasoning, PII scrubbing, and clinical-to-policy logic mapping.

## 🧠 Intelligence Architecture

This service implements an **Agency specialist Pipeline**:
- **Optical Data Specialist**: Neural OCR processing.
- **Extraction Architect**: Parsing medical parameters.
- **Logic Mapping Engine**: Comparing health status to insurance triggers.
- **Audit Logger**: Real-time logging of session results to MongoDB.

## 🛠️ Configuration

The service uses a global fallback chain to ensure high availability:
1. **Primary**: Grok 4.20 (via OpenRouter)
2. **Fallback**: Qwen 3.5
3. **Backup**: Specialist Google Gemma Nodes

### Environment Variables
```env
MISTRAL_API_KEY=your_key
OPENROUTER_API_KEY=your_key
MONGO_URI=your_uri
ANALYSIS_MODEL=google/gemma-4-31b-it:free
```

## 🚀 API Endpoints

### `POST /analyze`
Comprehensive analysis trigger. Supports:
- Streaming events for real-time UI updates.
- Automatic PII scrubbing.
- Pydantic schema validation.

### `GET /health`
Returns system status, active LLM model, and OCR engine version.

## 📦 Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the service:
   ```bash
   python llm_app.py
   ```

---
*Enterprise-Grade Medical AI Hub*
