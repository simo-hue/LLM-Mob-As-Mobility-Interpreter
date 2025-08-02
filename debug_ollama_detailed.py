#!/usr/bin/env python3
"""
Script diagnostico dettagliato per Ollama
"""
import requests
import json
import time
import sys
from pathlib import Path

def main():
    print("🔍 DIAGNOSTICA OLLAMA DETTAGLIATA")
    print("=" * 50)
    
    # Leggi porta
    try:
        with open("ollama_port.txt") as f:
            port = f.read().strip()
        base_url = f"http://127.0.0.1:{port}"
        print(f"🌐 URL base: {base_url}")
    except FileNotFoundError:
        print("❌ File ollama_port.txt non trovato")
        return False
    
    success_count = 0
    total_tests = 6
    
    # === TEST 1: /api/tags ===
    print(f"\n1️⃣ Test /api/tags")
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=10)
        print(f"   Status: {resp.status_code}")
        if resp.status_code == 200:
            data = resp.json()
            models = data.get('models', [])
            print(f"   ✅ Modelli: {len(models)}")
            for model in models:
                name = model.get('name', 'Unknown')
                size = model.get('size', 0) // (1024*1024*1024)  # GB
                print(f"      - {name} ({size}GB)")
            success_count += 1
        else:
            print(f"   ❌ Errore: {resp.text[:100]}")
    except Exception as e:
        print(f"   ❌ Eccezione: {e}")
    
    # === TEST 2: /api/version ===
    print(f"\n2️⃣ Test /api/version")
    try:
        resp = requests.get(f"{base_url}/api/version", timeout=5)
        print(f"   Status: {resp.status_code}")
        if resp.status_code == 200:
            print(f"   ✅ Versione: {resp.json()}")
            success_count += 1
        else:
            print(f"   ❌ Errore: {resp.text[:100]}")
    except Exception as e:
        print(f"   ❌ Eccezione: {e}")
    
    # === TEST 3: /api/generate (formato originale) ===
    print(f"\n3️⃣ Test /api/generate")
    payload = {
        "model": "llama3.1:8b",
        "prompt": "Rispondi solo: TEST_OK",
        "stream": False,
        "options": {"num_predict": 10, "temperature": 0.1}
    }
    
    try:
        print(f"   📤 Payload: {json.dumps(payload, indent=4)}")
        resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=45)
        print(f"   📥 Status: {resp.status_code}")
        print(f"   📥 Content-Length: {len(resp.content)}")
        
        if resp.status_code == 200:
            if resp.content:
                try:
                    data = resp.json()
                    print(f"   📋 JSON keys: {list(data.keys())}")
                    print(f"   📋 Done: {data.get('done', 'MISSING')}")
                    response_text = data.get('response', 'MISSING_FIELD')
                    print(f"   📋 Response: '{response_text}'")
                    print(f"   📋 Response length: {len(str(response_text))}")
                    
                    if data.get('done') and response_text and response_text != 'MISSING_FIELD':
                        print(f"   ✅ /api/generate FUNZIONA!")
                        success_count += 1
                    else:
                        print(f"   ⚠️  Risposta incompleta o vuota")
                        
                except json.JSONDecodeError as e:
                    print(f"   ❌ JSON malformato: {e}")
                    print(f"   📄 Raw: {resp.text[:200]}")
            else:
                print(f"   ❌ Corpo risposta completamente vuoto")
        else:
            print(f"   ❌ HTTP Error: {resp.text[:200]}")
            
    except requests.exceptions.Timeout:
        print(f"   ❌ Timeout (45s)")
    except Exception as e:
        print(f"   ❌ Eccezione: {e}")
    
    # === TEST 4: /api/chat (formato corretto) ===
    print(f"\n4️⃣ Test /api/chat")
    chat_payload = {
        "model": "llama3.1:8b",
        "messages": [{"role": "user", "content": "Rispondi solo: CHAT_OK"}],
        "stream": False,
        "options": {"num_predict": 10, "temperature": 0.1}
    }
    
    try:
        print(f"   📤 Payload: {json.dumps(chat_payload, indent=4)}")
        resp = requests.post(f"{base_url}/api/chat", json=chat_payload, timeout=45)
        print(f"   📥 Status: {resp.status_code}")
        print(f"   📥 Content-Length: {len(resp.content)}")
        
        if resp.status_code == 200:
            if resp.content:
                try:
                    data = resp.json()
                    print(f"   📋 JSON keys: {list(data.keys())}")
                    print(f"   📋 Done: {data.get('done', 'MISSING')}")
                    
                    message = data.get('message', {})
                    print(f"   📋 Message keys: {list(message.keys()) if message else 'NO_MESSAGE'}")
                    
                    content = message.get('content', 'MISSING_CONTENT') if message else 'NO_MESSAGE_FIELD'
                    print(f"   📋 Content: '{content}'")
                    print(f"   📋 Content length: {len(str(content))}")
                    
                    if data.get('done') and content and content not in ['MISSING_CONTENT', 'NO_MESSAGE_FIELD']:
                        print(f"   ✅ /api/chat FUNZIONA!")
                        success_count += 1
                    else:
                        print(f"   ⚠️  Risposta incompleta o vuota")
                        
                except json.JSONDecodeError as e:
                    print(f"   ❌ JSON malformato: {e}")
                    print(f"   📄 Raw: {resp.text[:200]}")
            else:
                print(f"   ❌ Corpo risposta completamente vuoto")
        else:
            print(f"   ❌ HTTP Error: {resp.text[:200]}")
            
    except requests.exceptions.Timeout:
        print(f"   ❌ Timeout (45s)")
    except Exception as e:
        print(f"   ❌ Eccezione: {e}")
    
    # === TEST 5: Payload sbagliato (/api/chat con "prompt") ===
    print(f"\n5️⃣ Test payload sbagliato (/api/chat con 'prompt')")
    wrong_payload = {
        "model": "llama3.1:8b",
        "prompt": "Test sbagliato",  # SBAGLIATO per /api/chat
        "stream": False
    }
    
    try:
        resp = requests.post(f"{base_url}/api/chat", json=wrong_payload, timeout=30)
        print(f"   Status: {resp.status_code}")
        print(f"   Content-Length: {len(resp.content)}")
        if resp.content:
            print(f"   Response: {resp.text[:200]}")
        else:
            print(f"   ❌ Risposta vuota (come nel tuo script!)")
            success_count += 1  # Questo è il comportamento che vediamo
    except Exception as e:
        print(f"   Eccezione: {e}")
    
    # === TEST 6: Stress test veloce ===
    print(f"\n6️⃣ Stress test (3 richieste rapide)")
    working_endpoint = None
    working_payload = None
    
    # Determina quale endpoint funziona
    if success_count >= 2:  # Se almeno generate o chat funziona
        working_endpoint = f"{base_url}/api/generate"
        working_payload = {
            "model": "llama3.1:8b",
            "prompt": "OK",
            "stream": False,
            "options": {"num_predict": 5}
        }
    
    if working_endpoint:
        successful_requests = 0
        for i in range(3):
            try:
                resp = requests.post(working_endpoint, json=working_payload, timeout=30)
                if resp.status_code == 200 and resp.content:
                    data = resp.json()
                    if data.get('done') and data.get('response'):
                        successful_requests += 1
                        print(f"   ✅ Richiesta {i+1}/3 OK")
                    else:
                        print(f"   ⚠️  Richiesta {i+1}/3 incompleta")
                else:
                    print(f"   ❌ Richiesta {i+1}/3 fallita")
            except Exception as e:
                print(f"   ❌ Richiesta {i+1}/3 errore: {e}")
        
        if successful_requests >= 2:
            success_count += 1
            print(f"   ✅ Stress test superato ({successful_requests}/3)")
        else:
            print(f"   ❌ Stress test fallito ({successful_requests}/3)")
    else:
        print(f"   ⏭️  Saltato (nessun endpoint funzionante)")
    
    # === RISULTATO FINALE ===
    print(f"\n" + "=" * 50)
    print(f"📊 RISULTATO: {success_count}/{total_tests} test superati")
    
    if success_count >= 4:
        print(f"✅ OLLAMA FUNZIONA CORRETTAMENTE")
        print(f"💡 Raccomandazione: Usa /api/generate nel tuo script")
        return True
    elif success_count >= 2:
        print(f"⚠️  OLLAMA FUNZIONA PARZIALMENTE")
        print(f"💡 Raccomandazione: Verifica payload e endpoint")
        return True
    else:
        print(f"❌ OLLAMA NON FUNZIONA")
        print(f"💡 Raccomandazione: Controlla configurazione server")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
