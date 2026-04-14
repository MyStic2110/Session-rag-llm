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

load_dotenv(override=True)

app = FastAPI(title="LumeHealth - LLM Microservice")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_mistral_client() -> Mistral:
    key = os.environ.get("MISTRAL_API_KEY")
    if not key:
        print("[!] ERROR: Mistral API key not configured")
        raise HTTPException(status_code=500, detail="Mistral API key not configured")
    return Mistral(api_key=key)

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

class InsuranceInfo(BaseModel):
    covered: List[str]
    conditional: List[str]
    not_covered: List[str]
    future_cost_awareness: str
    potential_out_of_pocket_increase: str

class FutureMapping(BaseModel):
    pattern: str
    future_condition: str
    coverage_status: str
    coverage_gap_risk: str
    severity_trend: str

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
    
    if len(content_bytes) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB.")
    
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
                "waiting_periods": ["string: period description"]
            }}
        }}
        STRICT RULES: 
        1. NO conversational text. 
        2. NO markdown formatting outside the JSON block.
        3. All numeric scores must be INTEGERS.
        4. Do NOT hallucinate data not present in texts.
        """
        extract_res = client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "user", "content": extraction_prompt}],
            response_format={"type": "json_object"}
        )
        deterministic_data = json.loads(extract_res.choices[0].message.content)

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
                "potential_out_of_pocket_increase": "Percentage string" 
            }},
            "future_coverage_mapping": [{{ 
                "pattern": "Health Trend", 
                "future_condition": "Likely Diagnosis", 
                "coverage_status": "Covered|Excluded|Partial", 
                "coverage_gap_risk": "High|Medium|Low", 
                "severity_trend": "Increasing|Stable|Decreasing" 
            }}],
            "disclaimer": "Safety statement"
        }}
        """
        final_res = client.chat.complete(
            model="mistral-large-latest",
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": final_prompt}],
            response_format={"type": "json_object"}
        )
        analysis_data = json.loads(final_res.choices[0].message.content)
        print("[OK] [LLM Service] ANALYSIS COMPLETE.")
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
                    "waiting_periods": ["string: period description"]
                }}
            }}
            STRICT RULES: 
            1. NO conversational text. 
            2. NO markdown formatting outside the JSON block.
            3. All numeric scores must be INTEGERS.
            4. Do NOT hallucinate data not present in texts.
            """
            import asyncio
            def call_mistral():
                return client.chat.complete(
                    model="mistral-large-latest",
                    messages=[{"role": "user", "content": extraction_prompt}],
                    response_format={"type": "json_object"}
                )
            
            extract_res = await asyncio.to_thread(call_mistral)
            deterministic_data = json.loads(extract_res.choices[0].message.content)
            
            # Track Tokens Layer 2
            usage = extract_res.usage
            total_tokens["prompt"] += usage.prompt_tokens
            total_tokens["completion"] += usage.completion_tokens
            total_tokens["total"] += usage.total_tokens
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
                    "potential_out_of_pocket_increase": "Percentage string" 
                }},
                "future_coverage_mapping": [{{ 
                    "pattern": "Health Trend", 
                    "future_condition": "Likely Diagnosis", 
                    "coverage_status": "Covered|Excluded|Partial", 
                    "coverage_gap_risk": "High|Medium|Low", 
                    "severity_trend": "Increasing|Stable|Decreasing" 
                }}],
                "disclaimer": "Safety statement"
            }}
            """
            
            def call_mistral_final():
                return client.chat.complete(
                    model="mistral-large-latest",
                    messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": final_prompt}],
                    response_format={"type": "json_object"}
                )
            
            final_res = await asyncio.to_thread(call_mistral_final)
            analysis_data = json.loads(final_res.choices[0].message.content)
            analysis_data["status"] = "success"

            # Track Tokens Layer 3
            usage_final = final_res.usage
            total_tokens["prompt"] += usage_final.prompt_tokens
            total_tokens["completion"] += usage_final.completion_tokens
            total_tokens["total"] += usage_final.total_tokens
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
