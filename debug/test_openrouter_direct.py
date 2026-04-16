import os
import httpx
import json
import sys
from dotenv import load_dotenv

# Ensure we can load .env from parent dir
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"), override=True)

def test_openrouter():
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print("[ERROR] OPENROUTER_API_KEY NOT FOUND IN ENV")
        return

    print(f"[*] Testing OpenRouter with Key: {api_key[:10]}...{api_key[-4:]}")
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://lume.up.railway.app/", # Required by OpenRouter for free models
        "X-Title": "LumeHealth Debug"
    }
    
    # Testing with the new Qwen 3.5 model
    payload = {
        "model": "qwen/qwen3.5-35b-a3b",
        "messages": [{"role": "user", "content": "Say 'Healthy' in a JSON object with key 'status'"}],
        "stream": True,
        "response_format": {"type": "json_object"}
    }
    
    try:
        with httpx.Client(timeout=45.0) as client:
            print(f"[*] Sending request to OpenRouter with model {payload['model']}...")
            full_content = ""
            
            with client.stream("POST", "https://openrouter.ai/api/v1/chat/completions", headers=headers, json=payload) as resp:
                print(f"[*] Status Code: {resp.status_code}")
                
                if resp.status_code != 200:
                    print(f"[FAIL] API Error: {resp.read().decode()}")
                    return

                print("[*] Receiving SSE stream...")
                for line in resp.iter_lines():
                    if not line or line.startswith(":"):
                        continue
                    
                    if line.startswith("data: "):
                        data_str = line[6:].strip()
                        if data_str == "[DONE]":
                            print("[*] Stream [DONE]")
                            break
                        
                        chunk = json.loads(data_str)
                        delta = chunk['choices'][0].get('delta', {})
                        if 'content' in delta and delta['content']:
                            full_content += delta['content']
                            # Small print for progress
                            sys.stdout.write(".")
                            sys.stdout.flush()

            print("\n[OK] Connection & Stream Successful!")
            print(f"[*] Reconstructed Content: {full_content}")
                
    except httpx.ConnectError:
        print("[FAIL] Network Connection Error.")
    except Exception as e:
        print(f"[EXCEPTION] {type(e).__name__}: {str(e)}")

if __name__ == "__main__":
    test_openrouter()
