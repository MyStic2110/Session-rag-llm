import os
import tempfile
import json
import re
from typing import Dict, Any, cast, List, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from mistralai.client import Mistral
import contextlib
import httpx
import hashlib
import logging
import uuid
import time
from datetime import datetime
from motor.motor_asyncio import AsyncIOMotorClient
load_dotenv(override=True)

# --- Enterprise Shield: Audit Logger ---
logging.basicConfig(
    filename="audit.log",
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
audit_logger = logging.getLogger("LumeHealthAudit")

# --- Enterprise Shield: MongoDB Connectivity & Lifespan ---
db = None
db_client = None

@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    global db, db_client

    # --- Startup: Validate Required Environment Variables ---
    required_vars = ["MISTRAL_API_KEY", "OPENROUTER_API_KEY"]
    for var in required_vars:
        if not os.environ.get(var):
            print(f"[!!!] [LLM Service] CRITICAL: Required environment variable '{var}' is NOT SET. Service may not function.")

    mongo_uri = os.environ.get("MONGO_URI")
    if mongo_uri:
        try:
            masked_uri = mongo_uri.split("@")[-1] if "@" in mongo_uri else "HIDDEN"
            print(f"[*] [LLM Service] Attempting MongoDB initialization (Host: {masked_uri})...")
            
            db_client = AsyncIOMotorClient(
                mongo_uri, 
                serverSelectionTimeoutMS=5000, 
                connectTimeoutMS=10000,
                tlsAllowInvalidCertificates=True
            )
            
            # Connectivity check (forced)
            await db_client.admin.command('ping')
            db = db_client["lumehealth"]
            
            # Write test for production verification
            await db.startup_check.insert_one({"status": "active", "ts": datetime.utcnow()})
            print(f"[OK] [LLM Service] MongoDB initialized and writable: {db.name}")
        except Exception as e:
            print(f"[!] [LLM Service] MongoDB Startup Connection Failed: {str(e)}")
            db = None
            db_client = None
    else:
        print("[!] [LLM Service] MONGO_URI environment variable is MISSING. Audit logging will use local file fallback.")
    
    yield
    
    if db_client:
        print("[*] [LLM Service] Closing MongoDB connection...")
        db_client.close()

async def log_to_db(audit_data: Dict[str, Any]):
    """Logs audit data to MongoDB with a local file fallback."""
    global db
    try:
        if db is not None:
            # Automatic collection handling
            await db.audit_logs.insert_one({
                **audit_data,
                "timestamp": datetime.utcnow()
            })
            print(f"[OK] [LLM Service] Audit entry saved to DB (AuditID: {audit_data.get('audit_id')})")
            return True
        else:
            print(f"[!] [LLM Service] MongoDB not initialized. Falling back to local log.")
    except Exception as e:
        print(f"[!] [LLM Service] MongoDB Logging Failed (AuditID: {audit_data.get('audit_id')}): {str(e)}")
        if "timeout" in str(e).lower():
            print("[TIP] This looks like a network timeout. Check MongoDB Atlas IP Whitelisting (0.0.0.0/0).")
    
    # Fallback to local log for resilience
    audit_logger.info(f"DB_FALLBACK - {json.dumps(audit_data)}")
    return False

# --- Enterprise Shield: Result Cache ---
ANALYSIS_CACHE: Dict[str, Any] = {}

app = FastAPI(title="LumeHealth - LLM Microservice", lifespan=lifespan)

# --- Global Exception Handler: Always return clean JSON ---
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    err_str = str(exc)
    print(f"[!!!] [LLM Service] Unhandled Exception on {request.url.path}: {err_str}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred in the Intelligence Engine. Please try again."},
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    db_status = "disconnected"
    global db
    if db is not None:
        try:
            await db.command("ping")
            db_status = "connected"
        except Exception as e:
            print(f"[!] [LLM Service] Health Check DB Ping Failed: {str(e)}")
            db_status = "error"

    return {
        "status": "online",
        "service": "LumeHealth LLM Microservice",
        "mongodb": db_status,
        "model": os.environ.get("ANALYSIS_MODEL", "google/gemma-4-26b-a4b-it:free"),
        "ocr_engine": "Mistral OCR 2512"
    }

def get_mistral_client() -> Mistral:
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY is not set")
    return Mistral(api_key=api_key)

# Decoy Alias Map to hide LLM names from UI
AGENT_ALIAS_MAP = {
    "mistral-small-latest": "High-Speed Logic Node",
    "x-ai/grok-4.20-beta": "Primary Cognition Architect",
    "qwen/qwen3.5-397b-a17b": "Deep Reasoning Master",
    "qwen/qwen3.5-35b-a3b": "Balanced Logic Engine",
    "nvidia/nemotron-3-super-120b:free": "Neural Efficiency Node",
    "openrouter/free": "Resilient Backup Node",
    "google/gemma-4-31b-it:free": "Secondary Contextual Analyst",
    "default": "Specialist Backup Node"
}

# Global Fallback Chain (OpenRouter Only)
FALLBACK_MODELS = [
    "x-ai/grok-4.20-beta",
    "qwen/qwen3.5-397b-a17b",
    "qwen/qwen3.5-35b-a3b",
    "nvidia/nemotron-3-super-120b:free",
    "google/gemma-4-31b-it:free",
    "openrouter/free"
]

async def call_openrouter(messages: List[Dict[str, Any]], stream: bool = False, on_retry: Optional[callable] = None):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    primary_model = os.environ.get("ANALYSIS_MODEL", FALLBACK_MODELS[0])
    
    # Construct chain starting with preferred model
    MODELS_CHAIN = list(dict.fromkeys([primary_model] + FALLBACK_MODELS))

    if not api_key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY is not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://lume.up.railway.app/", # Required by OpenRouter for free models
        "X-Title": "LumeHealth Debug"
    }
    
    last_error = ""
    for idx, model in enumerate(MODELS_CHAIN):
        try:
            if idx > 0 and on_retry:
                alias = AGENT_ALIAS_MAP.get(model, AGENT_ALIAS_MAP["default"])
                await on_retry(alias)

            print(f"[*] [LLM Service] Attempting OpenRouter call with: {model} (Streaming Mode)")
            payload = {
                "model": model,
                "messages": messages,
                "stream": True, 
                "temperature": 0.0, # Shield 1: Determinism
                "top_p": 1.0,
                "seed": 42, # Shield 1: Reproducibility
                "reasoning": {"enabled": True},
                "response_format": {"type": "json_object"}
            }
            
            full_content = ""
            usage_data = {}
            reasoning_data = ""

            async with httpx.AsyncClient(timeout=45.0) as client:
                async with client.stream("POST", "https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload) as response:
                    if response.status_code in [429, 404, 503]:
                        resp_json = await response.json()
                        err_msg = resp_json.get("error", {}).get("message", "Intelligence Node busy")
                        print(f"[!] [LLM Service] Model {model} failed ({response.status_code}): {err_msg}")
                        last_error = f"{model}: {err_msg}"
                        continue
                    
                    if response.status_code != 200:
                        err_text = await response.read()
                        last_error = f"{model} Error: {err_text.decode()}"
                        continue

                    async for line in response.aiter_lines():
                        if not line or line.startswith(":"): # Ignore heartbeats/comments
                            continue
                        
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if data_str == "[DONE]":
                                break
                            
                            chunk = json.loads(data_str)
                            delta = chunk['choices'][0].get('delta', {})
                            
                            # Extract content
                            if 'content' in delta and delta['content']:
                                full_content += delta['content']
                            
                            # Extract reasoning
                            if 'reasoning' in delta and delta['reasoning']:
                                reasoning_data += delta['reasoning']
                            elif 'reasoning_details' in delta and delta['reasoning_details']:
                                # Some providers put it here
                                if isinstance(delta['reasoning_details'], str):
                                    reasoning_data += delta['reasoning_details']
                                elif isinstance(delta['reasoning_details'], list):
                                    for rd in delta['reasoning_details']:
                                        if rd.get('type') == 'reasoning.text':
                                            reasoning_data += rd.get('text', '')

                            # Extract usage (usually in final chunk)
                            if 'usage' in chunk:
                                usage_data = chunk['usage']

            print(f"[OK] [LLM Service] Success with model: {model} (Reconstructed {len(full_content)} chars)")
            
            # Return in standard OpenRouter format but include our alias
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": full_content,
                        "reasoning_details": reasoning_data
                    }
                }],
                "usage": usage_data,
                "agent_alias": AGENT_ALIAS_MAP.get(model, AGENT_ALIAS_MAP["default"])
            }

        except Exception as e:
            print(f"[!] [LLM Service] SSE Exception with {model}: {str(e)}")
            last_error = str(e)
            continue
            
    raise HTTPException(status_code=503, detail=f"All OpenRouter models failed. Logs: {last_error}")

async def call_mistral_direct(messages: List[Dict[str, Any]]):
    """Fallback to Mistral Direct API if OpenRouter fails."""
    try:
        print(f"[*] [LLM Service] Critical Fallback: Attempting Mistral Direct API...")
        client = get_mistral_client()
        
        # Strip reasoning_details if present in messages as Mistral doesn't support them
        clean_messages = []
        for msg in messages:
            m = {k: v for k, v in msg.items() if k != "reasoning_details" and k != "reasoning"}
            clean_messages.append(m)
            
        response = client.chat.complete(
            model="mistral-small-latest",
            messages=clean_messages,
            temperature=0, # Shield 1: Determinism
            response_format={"type": "json_object"}
        )
        
        # Format response to match OpenRouter's structure for consistency
        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": response.choices[0].message.content
                }
            }],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
    except Exception as e:
        print(f"[!] [LLM Service] Mistral Direct Error: {str(e)}")
        raise HTTPException(status_code=503, detail=f"Mistral Fallback failed: {str(e)}")

def clean_json_response(content: str) -> str:
    """Strips reasoning/thought tags from content to extract pure JSON."""
    if not content: return "{}"
    # Remove <thought>...</thought> or <think>...</think>
    content = re.sub(r'<(thought|think)>.*?</\1>', '', content, flags=re.DOTALL)
    # Remove any markdown code blocks
    content = re.sub(r'```json\s*', '', content)
    content = re.sub(r'```', '', content)
    return content.strip()

def get_token_estimate(text: str) -> int:
    """Fallback token estimation (approx 4 chars per token)."""
    if not text: return 0
    return max(1, len(text) // 4)

class AnalyzePayload(BaseModel):
    health_text: str
    policy_text: str
    health_filename: Optional[str] = None
    policy_filename: Optional[str] = None

    @validator('health_text')
    def health_text_not_empty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError('Health report text cannot be empty.')
        if len(v) < 50:
            raise ValueError('Health report text is too short to be a valid document.')
        return v

    @validator('policy_text')
    def policy_text_not_empty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError('Insurance policy text cannot be empty.')
        if len(v) < 50:
            raise ValueError('Insurance policy text is too short to be a valid document.')
        return v

# --- Guardrail Schemas ---

class AbnormalExplanation(BaseModel):
    parameter: str
    explanation: str

class RiskOutlook(BaseModel):
    short_term: str
    medium_term: str
    long_term: str
    short_term_multiplier: str
    medium_term_multiplier: str
    long_term_multiplier: str

class ContextualGuardrail(BaseModel):
    category: str
    limit_details: str
    waiting_period: str
    red_lining_risk: str
    source_citation: str
    exact_quote: Optional[str] = "N/A"

class InsuranceInfo(BaseModel):
    covered: List[str]
    conditional: List[str]
    not_covered: List[str]
    future_cost_awareness: str
    potential_out_of_pocket_increase: str
    contextual_guardrails: List[ContextualGuardrail] = []

class FutureMapping(BaseModel):
    pattern: str
    future_condition: str
    timeframe_years: str
    coverage_status: str
    coverage_gap_risk: str
    severity_trend: str
    source_proof: str
    red_line_risk: str
    intent_clarity_explanation: str
    exact_quote: Optional[str] = "N/A"

class ComprehensiveAnalysisResponse(BaseModel):
    summary: str
    abnormal_explanations: List[AbnormalExplanation]
    pattern_explanation: List[str]
    risk_outlook: RiskOutlook
    recommendations: List[str]
    insurance: InsuranceInfo
    future_coverage_mapping: List[FutureMapping]
    health_metrics: Dict[str, int] = Field(default_factory=dict)
    confidence_score: float = 0.95
    disclaimer: str
    validation_warnings: List[str] = []
    status: Optional[str] = "success"

# --- Utility Functions ---

def scrub_pii(text: str) -> str:
    """Redacts patient personal details using regex heuristics."""
    if not text:
        return ""
    # 1. Email
    text = re.sub(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', '[REDACTED_EMAIL]', text)
    # 2. Phone (various formats)
    text = re.sub(r'(\+?\d{1,3}[\s-]?)?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}', '[REDACTED_PHONE]', text)
    # 3. Specific Personal IDs (SSN, common medical ID formats)
    text = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED_ID]', text)
    # 4. Dates of Birth (DOB) and other sensitive dates (e.g., 12/04/1985, 12-04-1985)
    text = re.sub(r'\b(0?[1-9]|1[012])[- /.](0?[1-9]|[12][0-9]|3[01])[- /.](19|20)\d\d\b', '[REDACTED_DATE]', text)
    # 5. Alphanumeric Medical Record Numbers (MRN) - e.g. MRN123456
    text = re.sub(r'\b(MRN|ID|Patient|Member)\s*[:#-]?\s*[A-Z0-9]{5,}\b', r'\1: [REDACTED_CODE]', text, flags=re.IGNORECASE)
    # 6. Patient Name Heuristics (matches phrases like "Patient Name: John Doe")
    text = re.sub(r'(?i)(patient\s*name|name|subject|client)\s*[:=-]\s*([A-Z][a-z]+(\s+[A-Z][a-z]+)+)', r'\1: [REDACTED_NAME]', text)
    return text

const_safety_disclaimer = (
    "This is not a medical diagnosis or insurance advice. LumeHealth provides AI-driven explanations "
    "based on the documents provided. Please consult a qualified healthcare professional and your "
    "insurance provider for definitive guidance."
)

@app.post("/ocr")
async def process_ocr(doc_type: str = Form(...), file: UploadFile = File(...)):
    # Validate doc_type
    if doc_type not in ("health", "policy"):
        raise HTTPException(status_code=400, detail="Invalid document type. Must be 'health' or 'policy'.")

    print(f"[*] [LLM Service] Processing OCR for: {file.filename} of type {doc_type}")
    
    engine = os.environ.get("OCR_ENGINE", "mistral").lower()
    print(f"[*] [LLM Service] Using OCR Engine: {engine}")

    raw_content = await file.read()
    content_bytes = cast(bytes, raw_content)
    
    if len(content_bytes) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large. Maximum allowed size is 5MB.")
    
    # --- PDF Magic Byte Validation (%PDF header) ---
    if not content_bytes.startswith(b'%PDF'):
        raise HTTPException(
            status_code=415,
            detail="Invalid file format. Please upload a valid PDF document from your medical provider or insurance company."
        )

    if len(content_bytes) < 100:
        raise HTTPException(status_code=400, detail="The uploaded file appears to be empty or corrupted.")

    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content_bytes)
            tmp_path = tmp.name
        return await run_mistral_ocr_process(tmp_path, file.filename)
            
    except HTTPException:
        raise
    except Exception as e:
        err_msg = str(e)
        print(f"[!] [LLM Service] OCR ERROR: {err_msg}")
        if "api" in err_msg.lower() or "mistral" in err_msg.lower():
            raise HTTPException(status_code=503, detail="Document scanner is temporarily unavailable. Please try again in a moment.")
        raise HTTPException(status_code=500, detail="Failed to process the document. Please ensure the PDF is not password-protected or corrupted.")
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

async def run_mistral_ocr_process(tmp_path: str, filename: str):
    print(f"[*] [LLM Service] Running Mistral OCR...")
    client = get_mistral_client()
    with open(tmp_path, "rb") as f:
        uploaded_file = client.files.upload(
            file={"file_name": filename, "content": f},
            purpose="ocr"
        )
    
    file_id = uploaded_file.id
    signed_url = client.files.get_signed_url(file_id=file_id)
    ocr_response = client.ocr.process(
        model="mistral-ocr-2512",
        document={"type": "document_url", "document_url": signed_url.url}
    )
    
    full_text = ""
    for page in ocr_response.pages:
        full_text += page.markdown + "\n\n"
    
    print(f"[OK] [LLM Service] Mistral OCR COMPLETE: {len(full_text)} chars extracted.")
    return {"status": "success", "text": full_text, "file_id": file_id, "engine": "mistral"}

@app.delete("/file/{file_id}")
async def delete_file(file_id: str):
    try:
        client = get_mistral_client()
        client.files.delete(file_id=file_id)
        print(f"[*] [LLM Service] Deleted remote file {file_id}")
        return {"status": "success"}
    except Exception as e:
        print(f"[!] [LLM Service] Error deleting file {file_id}: {str(e)}")
        # We don't raise error to keep cleanup non-blocking
        return {"status": "error", "detail": str(e)}

@app.post("/analyze")
async def analyze_coverage(payload: AnalyzePayload):
    start_time = time.time()
    audit_id = str(uuid.uuid4())
    cache_key = hashlib.md5((payload.health_text + payload.policy_text).encode()).hexdigest()
    
    print(f"[*] [LLM Service] Starting Analysis (AuditID: {audit_id})...")
    
    # Base report
    report = {
        "audit_id": audit_id,
        "doc_fingerprint": cache_key,
        "files": {"health": payload.health_filename, "policy": payload.policy_filename},
        "performance": {"cache_hit": False},
        "nodes": {"ocr_engine": os.environ.get("OCR_ENGINE", "mistral")}
    }

    if cache_key in ANALYSIS_CACHE:
        print(f"[OK] Cache Hit: {cache_key}")
        report["performance"]["cache_hit"] = True
        report["performance"]["latency_ms"] = int((time.time() - start_time) * 1000)
        await log_to_db(report)
        return ANALYSIS_CACHE[cache_key]

    client = get_mistral_client()
    
    try:
        extraction_prompt = f"""
        Extract deterministic data from the following health and insurance texts.
        Output ONLY valid JSON.
        
        HEALTH TEXT:
        {payload.health_text[:20000]}
        
        POLICY TEXT:
        {payload.policy_text[:20000]}
        
        JSON Structure:
        {{
            "health": {{
                "abnormal_parameters": ["string: parameter name only"],
                "domain_scores": {{ "cardio": 0-100, "liver": 0-100, "respiratory": 0-100, "metabolic": 0-100 }},
                "detected_patterns": ["string: identified trend"],
                "risk_projection": {{ "short": "string", "medium": "string", "long": "string" }},
                "overall_risk": "low|moderate|high"
            }},
            "insurance": {{
                "matched_policy_items": ["string: benefit name"],
                "coverage_details": {{ "covered": ["string"], "conditional": ["string"], "excluded": ["string"] }},
                "waiting_periods": ["string: period description"],
                "contextual_guardrails": [
                    {{
                        "category": "Maternity|Oncology|Cardiology|Other",
                        "limit_details": "string: sub-limits, caps, co-pays",
                        "waiting_period": "string",
                        "red_lining_risk": "High|Moderate|Low",
                        "source_citation": "string: precise policy section",
                        "exact_quote": "string: verbatim text snippet as direct evidence"
                    }}
                ]
            }},
            "identity": {{
                "patient_name": "string",
                "policy_holder_name": "string"
            }}
        }}
        STRICT RULES: 
        1. Contextual Guardrails: ONLY populate if the Health Report indicates a relevant condition (e.g. identify 'Maternity' details ONLY if pregnancy is mentioned).
        2. Identity Check: Carefully extract the Patient Name from the Health Text and the Policy Holder/Member Name from the Policy Text.
        3. NO conversational text. 
        3. NO markdown formatting outside the JSON block.
        4. All numeric scores must be INTEGERS.
        5. Do NOT hallucinate data not present in texts.
        """
        # Layer 2: Extraction using OpenRouter with Reasoning
        messages_layer2 = [{"role": "user", "content": extraction_prompt}]
        extract_res = await call_openrouter(messages_layer2)
        
        message_l2 = extract_res['choices'][0]['message']
        content_l2 = clean_json_response(message_l2.get('content', '{}'))
        deterministic_data = json.loads(content_l2)
        reasoning_l2 = message_l2.get('reasoning_details')

        system_prompt = """
        You are a health and insurance explanation assistant.
        You DO NOT perform any medical analysis, scoring, inference, or insurance eligibility decisions.
        All health analysis and insurance interpretations are provided to you as deterministic data.
        Your role is to:
        1. Explain the health data in simple terms
        2. Explain how health status relates to policy coverage
        3. Clearly present coverage strictly based on provided policy mapping
        STRICT RULES:
        - DO NOT calculate or override data.
        - DO NOT diagnose or predict specific diseases.
        - DO NOT give financial or medical advice.
        - Use CLEAR, CALM, and SUPPORTIVE tone.
        - Output STRICT JSON format as specified.
        SAFETY STATEMENT (MANDATORY):
        "This is not a medical diagnosis or insurance advice. Please consult a qualified healthcare professional and your insurance provider for detailed guidance."
        """
        final_prompt = f"""
        INPUT DATA: {json.dumps(deterministic_data)}
        TASK: Generate the explanation following the 7 sections. 
        Section: "Future Coverage Mapping"
        For each map, use an "Intelligent Re-analysis" tone.
        Explain the mapping as: "Your insurance will cover this if you are within the policy period, otherwise you will pay from your pocket."
        Be specific about WHY (e.g. waiting periods, exclusions).
        STRICT SCHEME ENFORCEMENT:
        Every field below MUST be a STRING. Do NOT return sub-objects or arrays where a string is expected.
        STRICT SCHEME ENFORCEMENT:
        Every field below MUST be exactly as specified. 
        Strings must be meaningful explanations, not just "N/A" unless truly missing.
        REQUIRED OUTPUT JSON FORMAT:
        {{
            "summary": "1-2 paragraph executive summary",
            "abnormal_explanations": [{{ "parameter": "name", "explanation": "clear medical explanation" }}],
            "pattern_explanation": ["explanation of trend 1", "explanation of trend 2"],
            "risk_outlook": {{ 
                "short_term": "Optimistic|Stable|Concerning", 
                "medium_term": "Optimistic|Stable|Concerning", 
                "long_term": "Optimistic|Stable|Concerning", 
                "short_term_multiplier": "+0% to +100%", 
                "medium_term_multiplier": "+0% to +100%", 
                "long_term_multiplier": "+0% to +100%" 
            }},
            "recommendations": ["Actionable step 1", "Actionable step 2"],
            "insurance": {{ 
                "covered": ["Policy Item A", "Policy Item B"], 
                "conditional": ["Condition X", "Condition Y"], 
                "not_covered": ["Exclusion Z"], 
                "future_cost_awareness": "Detailed impact on future premiums/costs", 
                "potential_out_of_pocket_increase": "Percentage string",
                "contextual_guardrails": [{{
                    "category": "Type",
                    "limit_details": "Full explanation of sub-limits/caps",
                    "waiting_period": "X months/years",
                    "red_lining_risk": "High|Moderate|Low",
                    "source_citation": "Policy Ref"
                }}]
            }},
            "future_coverage_mapping": [{{ 
                "pattern": "Health Trend", 
                "future_condition": "Likely Diagnosis", 
                "timeframe_years": "Estimated timeframe (e.g., 2 Years, 10 Years)",
                "coverage_status": "Covered|Excluded|Partial", 
                "source_proof": "Policy Section X / Evidence text",
                "red_line_risk": "High|Moderate|Low",
                "intent_clarity_explanation": "How precisely the policy covers this intent",
                "coverage_gap_risk": "High|Medium|Low", 
                "severity_trend": "Increasing|Stable|Decreasing",
                "exact_quote": "string: verbatim source text supporting this mapping"
            }}],
            "health_metrics": {{ "cardio": 0-100, "liver": 0-100, "respiratory": 0-100, "metabolic": 0-100 }},
            "confidence_score": 0.0-1.0,
            "disclaimer": "Safety statement"
        }}
        """
        # Layer 3: Explanation using OpenRouter with Reasoning Pass-back
        messages_layer3 = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": extraction_prompt},
            {
                "role": "assistant", 
                "content": json.dumps(deterministic_data),
                "reasoning_details": reasoning_l2
            },
            {"role": "user", "content": final_prompt}
        ]
        
        final_res = await call_openrouter(messages_layer3)
        content_l3 = clean_json_response(final_res['choices'][0]['message'].get('content', '{}'))
        analysis_data = json.loads(content_l3)
        
        print(f"[OK] [LLM Service] OpenRouter ANALYSIS COMPLETE.")
        
        # Identity Cross-Check Logic
        validation_warnings = []
        identity = deterministic_data.get("identity", {})
        p_name = identity.get("patient_name", "").strip().lower()
        h_name = identity.get("policy_holder_name", "").strip().lower()
        
        if p_name and h_name:
            # Simple fuzzy check: if one isn't contained in the other and they are reasonably long
            if p_name not in h_name and h_name not in p_name and len(p_name) > 3 and len(h_name) > 3:
                msg = f"Identity Mismatch Detected: Health Report belongs to '{identity.get('patient_name')}' but Policy belongs to '{identity.get('policy_holder_name')}'."
                print(f"[WARNING] {msg}")
                validation_warnings.append(msg)
        
        analysis_data["validation_warnings"] = validation_warnings
        analysis_data["audit_id"] = audit_id
        
        # --- Finalize Master Report ---
        report["performance"]["latency_ms"] = int((time.time() - start_time) * 1000)
        report["performance"]["token_count"] = analysis_data.get("tokens", {}).get("total", 0)
        report["nodes"]["primary_agent"] = analysis_data.get("agent_alias", "Intelligence Node")
        report["result"] = analysis_data # Store actual results in the audit log
        
        ANALYSIS_CACHE[cache_key] = analysis_data
        await log_to_db(report)
        
        return analysis_data
    except Exception as e:
        audit_logger.error(f"AuditID: {audit_id} - ANALYSIS ERROR: {str(e)}")
        print(f"[!] [LLM Service] ANALYSIS ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/stream")
async def analyze_coverage_stream(payload: AnalyzePayload):
    start_time = time.time()
    audit_id = str(uuid.uuid4())
    cache_key = hashlib.md5((payload.health_text + payload.policy_text).encode()).hexdigest()
    
    print(f"[*] [LLM Service] Starting Analysis Stream (AuditID: {audit_id})...")
    
    report = {
        "audit_id": audit_id,
        "doc_fingerprint": cache_key,
        "files": {"health": payload.health_filename, "policy": payload.policy_filename},
        "performance": {"cache_hit": False, "is_stream": True},
        "nodes": {"ocr_engine": os.environ.get("OCR_ENGINE", "mistral")}
    }

    if cache_key in ANALYSIS_CACHE:
        print(f"[OK] Cache Hit: {cache_key}")
        report["performance"]["cache_hit"] = True
        report["performance"]["latency_ms"] = int((time.time() - start_time) * 1000)
        await log_to_db(report)
        async def cached_generator():
            yield f"event: result\ndata: {json.dumps(ANALYSIS_CACHE[cache_key])}\n\n"
        return StreamingResponse(cached_generator(), media_type="text/event-stream")
    
    # --- Shield 2: Input Sanitization ---
    payload.health_text = scrub_pii(payload.health_text)
    payload.policy_text = scrub_pii(payload.policy_text)
    
    client = get_mistral_client()
    
    async def event_generator():
        total_tokens = {"prompt": 0, "completion": 0, "total": 0}
        try:
            yield f"event: step\ndata: {json.dumps({'message': 'Extracting deterministic facts (Layer 2)', 'progress': 30})}\n\n"
            
            extraction_prompt = f"""
            Extract deterministic data from the following health and insurance texts.
            Output ONLY valid JSON.
            
            HEALTH TEXT:
            {payload.health_text[:20000]}
            
            POLICY TEXT:
            {payload.policy_text[:20000]}
            
            JSON Structure:
            {{
                "health": {{
                    "abnormal_parameters": ["string: parameter name only"],
                    "domain_scores": {{ "cardio": 0-100, "liver": 0-100, "respiratory": 0-100, "metabolic": 0-100 }},
                    "detected_patterns": ["string: identified trend"],
                    "risk_projection": {{ "short": "string", "medium": "string", "long": "string" }},
                    "overall_risk": "low|moderate|high"
                }},
                "insurance": {{
                    "matched_policy_items": ["string: benefit name"],
                    "coverage_details": {{ "covered": ["string"], "conditional": ["string"], "excluded": ["string"] }},
                    "waiting_periods": ["string: period description"],
                    "contextual_guardrails": [
                        {{
                            "category": "Maternity|Oncology|Cardiology|Other",
                            "limit_details": "string: sub-limits, caps, co-pays",
                            "waiting_period": "string",
                            "red_lining_risk": "High|Moderate|Low",
                            "source_citation": "string: precise policy section",
                            "exact_quote": "string: verbatim text snippet as direct evidence"
                        }}
                    ]
                }},
                "identity": {{
                    "patient_name": "string",
                    "policy_holder_name": "string"
                }}
            }}
            STRICT RULES: 
            1. Contextual Guardrails: ONLY populate if the Health Report indicates a relevant condition (e.g. identify 'Maternity' details ONLY if pregnancy is mentioned).
            2. Identity Check: Carefully extract the Patient Name from the Health Text and the Policy Holder/Member Name from the Policy Text.
            3. NO conversational text. 
            3. NO markdown formatting outside the JSON block.
            4. All numeric scores must be INTEGERS.
            5. Do NOT hallucinate data not present in texts.
            """
            import asyncio
            # Notify UI of Agent Selection
            async def on_retry_l2(alias: str):
                return f"event: retry\ndata: {json.dumps({'agent_alias': alias})}\n\n"

            # Layer 2: Extraction using OpenRouter with Reasoning
            messages_l2 = [{"role": "user", "content": extraction_prompt}]
            
            extract_res = None
            try:
                # Primary Entry: Mistral Direct (High-Speed Logic Node)
                extract_res = await call_mistral_direct(messages_l2)
            except Exception as e:
                print(f"[!] Mistral Direct failed (L2). engaging Advanced OpenRouter Chain...")
                yield f"event: retry\ndata: {json.dumps({'agent_alias': 'Advanced Reasoning Chain'})}\n\n"
                extract_res = await call_openrouter(messages_l2, stream=False)
            
            if not extract_res:
                raise HTTPException(status_code=503, detail="Analytical Layer 2 failed entirely.")
            
            message_l2 = extract_res['choices'][0]['message']
            content_l2 = clean_json_response(message_l2.get('content', '{}'))
            deterministic_data = json.loads(content_l2)
            
            # Extract reasoning or reasoning_details per spec
            reasoning_l2 = message_l2.get('reasoning') or message_l2.get('reasoning_details')
            if not reasoning_l2:
                # Some models put it at the choice level
                reasoning_l2 = extract_res['choices'][0].get('reasoning')
            
            # Track Tokens Layer 2 with Fallback Estimation
            usage = extract_res.get('usage', {})
            p_tokens = usage.get('prompt_tokens') or get_token_estimate(extraction_prompt)
            c_tokens = usage.get('completion_tokens') or get_token_estimate(message_l2.get('content', ''))
            
            total_tokens["prompt"] += p_tokens
            total_tokens["completion"] += c_tokens
            total_tokens["total"] += (p_tokens + c_tokens)
            
            yield f"event: token\ndata: {json.dumps(total_tokens)}\n\n"

            yield f"event: step\ndata: {json.dumps({'message': 'Formulating explanation parameters (Layer 3)', 'progress': 60})}\n\n"

            system_prompt = """
            You are a health and insurance explanation assistant.
            You DO NOT perform any medical analysis, scoring, inference, or insurance eligibility decisions.
            All health analysis and insurance interpretations are provided to you as deterministic data.
            Your role is to:
            1. Explain the health data in simple terms
            2. Explain how health status relates to policy coverage
            3. Clearly present coverage strictly based on provided policy mapping
            STRICT RULES:
            - DO NOT calculate or override data.
            - DO NOT diagnose or predict specific diseases.
            - DO NOT give financial or medical advice.
            - Use CLEAR, CALM, and SUPPORTIVE tone.
            - Output STRICT JSON format as specified.
            SAFETY STATEMENT (MANDATORY):
            "This is not a medical diagnosis or insurance advice. Please consult a qualified healthcare professional and your insurance provider for detailed guidance."
            """
            final_prompt = f"""
            INPUT DATA: {json.dumps(deterministic_data)}
            TASK: Generate the explanation following the 7 sections. 
            Section: "Future Coverage Mapping"
            For each map, use an "Intelligent Re-analysis" tone.
            Explain the mapping as: "Your insurance will cover this if you are within the policy period, otherwise you will pay from your pocket."
            Be specific about WHY (e.g. waiting periods, exclusions).
            STRICT SCHEME ENFORCEMENT:
            Every field below MUST be a STRING. Do NOT return sub-objects or arrays where a string is expected.
            STRICT SCHEME ENFORCEMENT:
            Every field below MUST be exactly as specified. 
            Strings must be meaningful explanations, not just "N/A" unless truly missing.
            REQUIRED OUTPUT JSON FORMAT:
            {{
                "summary": "1-2 paragraph executive summary",
                "abnormal_explanations": [{{ "parameter": "name", "explanation": "clear medical explanation" }}],
                "pattern_explanation": ["explanation of trend 1", "explanation of trend 2"],
                "risk_outlook": {{ 
                    "short_term": "Optimistic|Stable|Concerning", 
                    "medium_term": "Optimistic|Stable|Concerning", 
                    "long_term": "Optimistic|Stable|Concerning", 
                    "short_term_multiplier": "+0% to +100%", 
                    "medium_term_multiplier": "+0% to +100%", 
                    "long_term_multiplier": "+0% to +100%" 
                }},
                "recommendations": ["Actionable step 1", "Actionable step 2"],
                "insurance": {{ 
                    "covered": ["Policy Item A", "Policy Item B"], 
                    "conditional": ["Condition X", "Condition Y"], 
                    "not_covered": ["Exclusion Z"], 
                    "future_cost_awareness": "Detailed impact on future premiums/costs", 
                    "potential_out_of_pocket_increase": "Percentage string",
                    "contextual_guardrails": [{{
                        "category": "Type",
                        "limit_details": "Full explanation of sub-limits/caps",
                        "waiting_period": "X months/years",
                        "red_lining_risk": "High|Moderate|Low",
                        "source_citation": "Policy Ref"
                    }}]
                }},
                "future_coverage_mapping": [{{ 
                    "pattern": "Health Trend", 
                    "future_condition": "Likely Diagnosis", 
                    "timeframe_years": "Estimated timeframe (e.g., 2 Years, 10 Years)",
                    "coverage_status": "Covered|Excluded|Partial", 
                    "source_proof": "Policy Section X / Evidence text",
                    "red_line_risk": "High|Moderate|Low",
                    "intent_clarity_explanation": "How precisely the policy covers this intent",
                    "coverage_gap_risk": "High|Medium|Low", 
                    "severity_trend": "Increasing|Stable|Decreasing",
                    "exact_quote": "string: verbatim source text supporting this mapping"
                }}],
                "health_metrics": {{ "cardio": 0-100, "liver": 0-100, "respiratory": 0-100, "metabolic": 0-100 }},
                "confidence_score": 0.0-1.0,
                "disclaimer": "Safety statement"
            }}
            """
            
            # Layer 3: Explanation using OpenRouter with Reasoning Pass-back
            messages_l3 = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": extraction_prompt},
                {
                    "role": "assistant", 
                    "content": json.dumps(deterministic_data),
                    "reasoning_details": reasoning_l2
                },
                {"role": "user", "content": final_prompt}
            ]
            
            async def on_retry_l3(alias: str):
                return f"event: retry\ndata: {json.dumps({'agent_alias': alias})}\n\n"

            final_res = None
            try:
                # Primary Entry: Mistral Direct (High-Speed Logic Node)
                final_res = await call_mistral_direct(messages_l3)
            except Exception as e:
                print(f"[!] Mistral Direct failed (L3). engaging Advanced OpenRouter Chain...")
                yield f"event: retry\ndata: {json.dumps({'agent_alias': 'Advanced Reasoning Chain'})}\n\n"
                final_res = await call_openrouter(messages_l3, stream=False)

            if not final_res:
                raise HTTPException(status_code=503, detail="Analytical Layer 3 failed entirely.")
            content_l3 = clean_json_response(final_res['choices'][0]['message'].get('content', '{}'))
            analysis_data = json.loads(content_l3)
            analysis_data["status"] = "success"

            # Identity Cross-Check Logic
            validation_warnings = []
            identity = deterministic_data.get("identity", {})
            p_name = identity.get("patient_name", "").strip().lower()
            h_name = identity.get("policy_holder_name", "").strip().lower()
            
            if p_name and h_name:
                if p_name not in h_name and h_name not in p_name and len(p_name) > 3 and len(h_name) > 3:
                    msg = f"Identity Mismatch Detected: Health Report belongs to '{identity.get('patient_name')}' but Policy belongs to '{identity.get('policy_holder_name')}'."
                    print(f"[WARNING] {msg}")
                    validation_warnings.append(msg)
            
            analysis_data["validation_warnings"] = validation_warnings

            # Track Tokens Layer 3 with Fallback Estimation
            usage_final = final_res.get('usage', {})
            p_tokens_l3 = usage_final.get('prompt_tokens') or get_token_estimate(str(messages_l3))
            c_tokens_l3 = usage_final.get('completion_tokens') or get_token_estimate(content_l3)
            
            total_tokens["prompt"] += p_tokens_l3
            total_tokens["completion"] += c_tokens_l3
            total_tokens["total"] += (p_tokens_l3 + c_tokens_l3)
            
            yield f"event: token\ndata: {json.dumps(total_tokens)}\n\n"
            
            analysis_data["agent_alias"] = final_res.get("agent_alias", "Intelligence Node")
            analysis_data["audit_id"] = audit_id
            
            # --- Finalize Master Report ---
            report["performance"]["latency_ms"] = int((time.time() - start_time) * 1000)
            report["performance"]["token_count"] = total_tokens.get("total", 0)
            report["nodes"]["primary_agent"] = analysis_data.get("agent_alias", "Intelligence Node")
            report["result"] = analysis_data # Store actual results in the audit log
            
            ANALYSIS_CACHE[cache_key] = analysis_data
            await log_to_db(report)
            
            yield f"event: result\ndata: {json.dumps(analysis_data)}\n\n"
            print("[OK] [LLM Service] ANALYSIS STREAM COMPLETE.")
            
        except Exception as e:
            print(f"[!] [LLM Service] ANALYSIS STREAM ERROR: {str(e)}")
            yield f"event: error\ndata: {json.dumps({'detail': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    # Standard Railway environment uses PORT variable
    port = int(os.environ.get("PORT", 8001))
    print(f"[*] Starting LLM service on port {port}...")
    uvicorn.run("llm_app:app", host="0.0.0.0", port=port, reload=False)
