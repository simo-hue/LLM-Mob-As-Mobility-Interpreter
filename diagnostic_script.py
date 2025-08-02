#!/usr/bin/env python3
"""
Script di diagnostica per debug problemi Ollama
"""
import requests
import json
import time
import sys
from pathlib import Path

def read_ollama_port():
    """Legge la porta da ollama_port.txt"""
    try:
        with open("ollama_port.txt", "r") as f:
            port = f.read().strip()
        return f"http://127.0.0.1:{port}"
    except FileNotFoundError:
        print("❌ File ollama_port.txt non trovato")
        sys.exit(1)

def test_ollama_endpoints(base_url):
    """Testa tutti gli endpoint principali di Ollama"""
    print(f"🔍 Testing Ollama su {base_url}")
    print("=" * 60)
    
    # Test 1: /api/tags
    print("1️⃣ Test /api/tags")
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=10)
        print(f"   Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            models = data.get('models', [])
            print(f"   Modelli disponibili: {len(models)}")
            for model in models:
                print(f"   - {model.get('name', 'Unknown')}")
        else:
            print(f"   Errore: {resp.text}")
    except Exception as e:
        print(f"   ❌ Errore: {e}")
    
    print()
    
    # Test 2: /api/version
    print("2️⃣ Test /api/version")
    try:
        resp = requests.get(f"{base_url}/api/version", timeout=10)
        print(f"   Status: {resp.status_code}")
        if resp.status_code == 200:
            print(f"   Version: {resp.json()}")
        else:
            print(f"   Errore: {resp.text}")
    except Exception as e:
        print(f"   ❌ Errore: {e}")
    
    print()
    
    # Test 3: /api/generate (endpoint diverso da /api/chat)
    print("3️⃣ Test /api/generate")
    payload = {
        "model": "llama3.1:8b",
        "prompt": "Say hello",
        "stream": False
    }
    
    try:
        resp = requests.post(
            f"{base_url}/api/generate",
            json=payload,
            timeout=60,
            headers={'Content-Type': 'application/json'}
        )
        print(f"   Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            print(f"   Risposta completa: {data.get('done', False)}")
            print(f"   Contenuto: {data.get('response', 'VUOTO')[:100]}...")
            print(f"   Tokens: {data.get('eval_count', 'N/A')}")
        else:
            print(f"   Errore: {resp.text}")
    except Exception as e:
        print(f"   ❌ Errore: {e}")
    
    print()
    
    # Test 4: /api/chat (quello che usa il tuo script)
    print("4️⃣ Test /api/chat")
    payload = {
        "model": "llama3.1:8b",
        "messages": [{"role": "user", "content": "Say hello"}],
        "stream": False
    }
    
    try:
        resp = requests.post(
            f"{base_url}/api/chat",
            json=payload,
            timeout=60,
            headers={'Content-Type': 'application/json'}
        )
        print(f"   Status: {resp.status_code}")
        print(f"   Headers: {dict(resp.headers)}")
        if resp.status_code == 200:
            try:
                data = resp.json()
                print(f"   Risposta completa: {data.get('done', False)}")
                message = data.get('message', {})
                content = message.get('content', 'VUOTO')
                print(f"   Contenuto: {content[:100]}...")
                print(f"   Struttura completa: {json.dumps(data, indent=2)[:500]}...")
            except json.JSONDecodeError:
                print(f"   ❌ Risposta non è JSON valido: {resp.text[:200]}")
        else:
            print(f"   Errore: {resp.text}")
    except Exception as e:
        print(f"   ❌ Errore: {e}")
    
    print()
    
    # Test 5: Verifica formato prompt utilizzato nel tuo script
    print("5️⃣ Test con payload originale del tuo script")
    original_payload = {
        "model": "llama3.1:8b",
        "prompt": "Say hello",  # Il tuo script usa 'prompt', non 'messages'!
        "stream": False,
        "options": {
            "temperature": 0.1,
            "top_p": 0.9,
            "top_k": 40,
            "num_ctx": 4096,
            "num_predict": 200,
            "stop": ["\n\n", "```"],
            "num_thread": 16,
            "repeat_penalty": 1.1
        }
    }
    
    try:
        resp = requests.post(
            f"{base_url}/api/chat",
            json=original_payload,
            timeout=60,
            headers={'Content-Type': 'application/json'}
        )
        print(f"   Status: {resp.status_code}")
        if resp.status_code == 200:
            try:
                data = resp.json()
                print(f"   Risposta completa: {data.get('done', False)}")
                print(f"   Contenuto: {data.get('response', 'VUOTO')[:100]}...")
                print(f"   Struttura: {list(data.keys())}")
            except json.JSONDecodeError:
                print(f"   ❌ Risposta non è JSON: {resp.text[:200]}")
        else:
            print(f"   Errore: {resp.text}")
    except Exception as e:
        print(f"   ❌ Errore: {e}")

def main():
    base_url = read_ollama_port()
    test_ollama_endpoints(base_url)
    
    print("\n" + "=" * 60)
    print("🔧 RACCOMANDAZIONI:")
    print("1. Se /api/generate funziona ma /api/chat no, usa /api/generate")
    print("2. Se /api/chat richiede 'messages' invece di 'prompt', aggiorna il payload")
    print("3. Controlla i log del server Ollama per dettagli")
    print("4. Verifica che il modello sia effettivamente caricato")

if __name__ == "__main__":
    main()