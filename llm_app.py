import os
import tempfile
import json
import re
from typing import Dict, Any, cast, List, Optional

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from mistralai.client import Mistral
from io import StringIO
import sys
import contextlib
import httpx

load_dotenv(override=True)

app = FastAPI(title="LumeHealth - LLM Microservice")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "service": "LumeHealth LLM Microservice",
        "model": os.environ.get("ANALYSIS_MODEL", "google/gemma-4-26b-a4b-it:free"),
        "ocr_engine": "Mistral OCR 2512"
    }

def get_mistral_client() -> Mistral:
    api_key = os.environ.get("MISTRAL_API_KEY")
    if not api_key:
        raise ValueError("MISTRAL_API_KEY is not set")
    return Mistral(api_key=api_key)

async def call_openrouter(messages: List[Dict[str, Any]], stream: bool = False):
    api_key = os.environ.get("OPENROUTER_API_KEY")
    model = os.environ.get("ANALYSIS_MODEL", "google/gemma-4-26b-a4b-it:free")
    
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY is not set")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "reasoning": {"enabled": True},
        "response_format": {"type": "json_object"}
    }
    
    if stream:
        return httpx.AsyncClient().stream("POST", "https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=120.0)
    else:
        async with httpx.AsyncClient() as client:
            resp = await client.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload, timeout=120.0)
            if resp.status_code != 200:
                print(f"[!] OpenRouter Error: {resp.text}")
                raise HTTPException(status_code=resp.status_code, detail=f"OpenRouter Error: {resp.text}")
            return resp.json()

class AnalyzePayload(BaseModel):
    health_text: str
    policy_text: str

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
    coverage_status: str
    coverage_gap_risk: str
    severity_trend: str
    source_proof: str
    red_line_risk: str
    intent_clarity_explanation: str

class ComprehensiveAnalysisResponse(BaseModel):
    summary: str
    abnormal_explanations: List[AbnormalExplanation]
    pattern_explanation: List[str]
    risk_outlook: RiskOutlook
    recommendations: List[str]
    insurance: InsuranceInfo
    future_coverage_mapping: List[FutureMapping]
    disclaimer: str
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
    # 4. Patient Name Heuristics (matches phrases like "Patient Name: John Doe")
    text = re.sub(r'(?i)(patient\s*name|name|subject|client)\s*[:=-]\s*([A-Z][a-z]+(\s+[A-Z][a-z]+)+)', r'\1: [REDACTED_NAME]', text)
    return text

const_safety_disclaimer = (
    "This is not a medical diagnosis or insurance advice. LumeHealth provides AI-driven explanations "
    "based on the documents provided. Please consult a qualified healthcare professional and your "
    "insurance provider for definitive guidance."
)

@app.post("/ocr")
async def process_ocr(doc_type: str = Form(...), file: UploadFile = File(...)):
    print(f"[*] [LLM Service] Processing OCR for: {file.filename} of type {doc_type}")
    
    engine = os.environ.get("OCR_ENGINE", "mistral").lower()
    print(f"[*] [LLM Service] Using OCR Engine: {engine}")

    raw_content = await file.read()
    content_bytes = cast(bytes, raw_content)
    
    if len(content_bytes) > 5 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 5MB.")
    
    tmp_path = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(content_bytes)
            tmp_path = tmp.name
        return await run_mistral_ocr_process(tmp_path, file.filename)
            
    except Exception as e:
        print(f"[!] [LLM Service] OCR ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")
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
    print(f"[*] [LLM Service] Starting Analysis...")
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
                        "source_citation": "string: precise policy section"
                    }}
                ]
            }}
        }}
        STRICT RULES: 
        1. Contextual Guardrails: ONLY populate if the Health Report indicates a relevant condition (e.g. identify 'Maternity' details ONLY if pregnancy is mentioned).
        2. NO conversational text. 
        3. NO markdown formatting outside the JSON block.
        4. All numeric scores must be INTEGERS.
        5. Do NOT hallucinate data not present in texts.
        """
        # Layer 2: Extraction using OpenRouter with Reasoning
        messages_layer2 = [{"role": "user", "content": extraction_prompt}]
        extract_res = await call_openrouter(messages_layer2)
        
        message_l2 = extract_res['choices'][0]['message']
        deterministic_data = json.loads(message_l2.get('content', '{}'))
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
                "coverage_status": "Covered|Excluded|Partial", 
                "source_proof": "Policy Section X / Evidence text",
                "red_line_risk": "High|Moderate|Low",
                "intent_clarity_explanation": "How precisely the policy covers this intent",
                "coverage_gap_risk": "High|Medium|Low", 
                "severity_trend": "Increasing|Stable|Decreasing" 
            }}],
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
        analysis_data = json.loads(final_res['choices'][0]['message'].get('content', '{}'))
        
        print(f"[OK] [LLM Service] OpenRouter ANALYSIS COMPLETE.")
        # Debug log reasoning presence
        if reasoning_l2:
            print("[INFO] [LLM Service] Layer 2 Reasoning preserved in Layer 3.")
            
        return analysis_data
    except Exception as e:
        print(f"[!] [LLM Service] ANALYSIS ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze/stream")
async def analyze_coverage_stream(payload: AnalyzePayload):
    print(f"[*] [LLM Service] Starting Analysis Stream...")
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
                            "source_citation": "string: precise policy section"
                        }}
                    ]
                }}
            }}
            STRICT RULES: 
            1. Contextual Guardrails: ONLY populate if the Health Report indicates a relevant condition (e.g. identify 'Maternity' details ONLY if pregnancy is mentioned).
            2. NO conversational text. 
            3. NO markdown formatting outside the JSON block.
            4. All numeric scores must be INTEGERS.
            5. Do NOT hallucinate data not present in texts.
            """
            import asyncio
            # Layer 2: Extraction using OpenRouter with Reasoning
            messages_l2 = [{"role": "user", "content": extraction_prompt}]
            extract_res = await call_openrouter(messages_l2)
            
            message_l2 = extract_res['choices'][0]['message']
            deterministic_data = json.loads(message_l2.get('content', '{}'))
            reasoning_l2 = message_l2.get('reasoning_details')
            
            # Track Tokens Layer 2
            usage = extract_res.get('usage', {})
            total_tokens["prompt"] += usage.get('prompt_tokens', 0)
            total_tokens["completion"] += usage.get('completion_tokens', 0)
            total_tokens["total"] += usage.get('total_tokens', 0)
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
                    "coverage_status": "Covered|Excluded|Partial", 
                    "source_proof": "Policy Section X / Evidence text",
                    "red_line_risk": "High|Moderate|Low",
                    "intent_clarity_explanation": "How precisely the policy covers this intent",
                    "coverage_gap_risk": "High|Medium|Low", 
                    "severity_trend": "Increasing|Stable|Decreasing" 
                }}],
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
            
            final_res = await call_openrouter(messages_l3)
            analysis_data = json.loads(final_res['choices'][0]['message'].get('content', '{}'))
            analysis_data["status"] = "success"

            # Track Tokens Layer 3
            usage_final = final_res.get('usage', {})
            total_tokens["prompt"] += usage_final.get('prompt_tokens', 0)
            total_tokens["completion"] += usage_final.get('completion_tokens', 0)
            total_tokens["total"] += usage_final.get('total_tokens', 0)
            yield f"event: token\ndata: {json.dumps(total_tokens)}\n\n"
            
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
