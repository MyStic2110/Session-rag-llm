import requests
import json
import os
import sys

# Requirements: pip install requests
# Run: python debug/test_analysis.py <health_pdf_path> <policy_pdf_path>

BASE_URL = "http://localhost:8001"

def test_full_flow(health_path, policy_path):
    print(f"[*] Starting test with:")
    print(f"    Health PDF: {health_path}")
    print(f"    Policy PDF: {policy_path}")
    
    if not os.path.exists(health_path) or not os.path.exists(policy_path):
        print("[!] ERROR: PDF files not found.")
        return

    # 1. OCR Health Report
    print("[*] Performing OCR on Health Report...")
    with open(health_path, "rb") as f:
        resp = requests.post(f"{BASE_URL}/ocr", files={"file": f}, data={"doc_type": "health"})
    
    if resp.status_code != 200:
        print(f"[!] OCR Failed: {resp.text}")
        return
    
    health_text = resp.json().get("text", "")
    print(f"[OK] Health OCR Complete ({len(health_text)} chars)")

    # 2. OCR Insurance Policy
    print("[*] Performing OCR on Insurance Policy...")
    with open(policy_path, "rb") as f:
        resp = requests.post(f"{BASE_URL}/ocr", files={"file": f}, data={"doc_type": "policy"})
    
    if resp.status_code != 200:
        print(f"[!] OCR Failed: {resp.text}")
        return
    
    policy_text = resp.json().get("text", "")
    print(f"[OK] Policy OCR Complete ({len(policy_text)} chars)")

    # 3. Analyze Stream
    print("[*] Initiating Analysis Stream...")
    payload = {
        "health_text": health_text,
        "policy_text": policy_text
    }
    
    # Using streaming response
    with requests.post(f"{BASE_URL}/analyze/stream", json=payload, stream=True) as resp:
        if resp.status_code != 200:
            print(f"[!] Analysis Failed: {resp.text}")
            return
            
        for line in resp.iter_lines():
            if line:
                decoded_line = line.decode('utf-8')
                print(f"STREAM >> {decoded_line}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python debug/test_analysis.py <health_pdf> <policy_pdf>")
    else:
        test_full_flow(sys.argv[1], sys.argv[2])
