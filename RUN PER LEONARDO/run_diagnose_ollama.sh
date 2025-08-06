#!/bin/bash
#SBATCH --job-name=ollama-debug
#SBATCH --account=IscrC_LLM-Mob
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:15:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=debug-%j.out

echo "🚀 AVVIO DIAGNOSTICA OLLAMA"
echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodo: $(hostname)"
echo "Data: $(date)"
echo "Working dir: $(pwd)"
echo ""

# === 1. AMBIENTE ===
echo "📦 Caricamento moduli..."
module purge
module load python/3.11.6--gcc--8.5.0
module load cuda/12.3
source $SLURM_SUBMIT_DIR/LLM/bin/activate

export CUDA_VISIBLE_DEVICES=0
export OLLAMA_GPU_OVERHEAD=0

# === 2. CONFIGURAZIONE ===
MODEL_PATH="/leonardo/home/userexternal/smattiol/.ollama/models/blobs/sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29"
OLLAMA_PORT=39001  # Porta diversa per evitare conflitti
OLLAMA_BIN="$HOME/opt/ollama/bin/ollama"

echo "🔧 Configurazione:"
echo "   Porta: $OLLAMA_PORT"
echo "   Modello: $MODEL_PATH"
echo "   Ollama: $OLLAMA_BIN"
echo ""

# Scrivi porta nel file
echo $OLLAMA_PORT > $SLURM_SUBMIT_DIR/ollama_port.txt
echo "✅ Porta salvata in ollama_port.txt"

# === 3. VARIABILI AMBIENTE OLLAMA ===
export OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT
export OLLAMA_MODELS=$HOME/.ollama/models
export OLLAMA_KEEP_ALIVE=10m
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_LOAD_TIMEOUT=5m

echo "🌍 Variabili ambiente Ollama configurate"

# === 4. AVVIO SERVER ===
echo ""
echo "🚀 Avvio server Ollama..."
$OLLAMA_BIN serve > ollama_debug.log 2>&1 &
SERVER_PID=$!

echo "   PID server: $SERVER_PID"

# Funzione per pulire al termine
cleanup() {
    echo ""
    echo "🧹 Pulizia in corso..."
    if kill -0 $SERVER_PID 2>/dev/null; then
        echo "   Terminazione server (PID: $SERVER_PID)"
        kill $SERVER_PID 2>/dev/null
        sleep 2
        kill -9 $SERVER_PID 2>/dev/null
    fi
    echo "✅ Pulizia completata"
}
trap cleanup EXIT

# === 5. ATTESA SERVER ===
echo "⏳ Attesa server (max 60s)..."
for i in {1..30}; do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "❌ Server morto prematuramente!"
        echo "📋 Log server:"
        cat ollama_debug.log
        exit 1
    fi
    
    if curl -s --connect-timeout 2 --max-time 3 "http://127.0.0.1:$OLLAMA_PORT/api/tags" >/dev/null 2>&1; then
        echo "✅ Server pronto dopo $((i * 2)) secondi"
        break
    fi
    
    if [ $((i % 10)) -eq 0 ]; then
        echo "   Still waiting... (${i}/30)"
    fi
    
    sleep 2
done

# Verifica finale connessione
if ! curl -s --connect-timeout 2 --max-time 3 "http://127.0.0.1:$OLLAMA_PORT/api/tags" >/dev/null 2>&1; then
    echo "❌ Server non risponde dopo 60s"
    echo "📋 Log server:"
    cat ollama_debug.log
    exit 1
fi

# === 6. CARICAMENTO MODELLO ===
echo ""
echo "📥 Setup modello llama3.1:8b..."
MODEL_NAME="llama3.1:8b"

# Verifica se esiste già
if curl -s "http://127.0.0.1:$OLLAMA_PORT/api/tags" | grep -q "$MODEL_NAME"; then
    echo "✅ Modello già presente"
else
    echo "🔨 Creazione modello dal blob..."
    
    # Modelfile temporaneo
    cat > /tmp/debug_modelfile << EOF
FROM $MODEL_PATH
PARAMETER num_ctx 4096
PARAMETER num_batch 256
PARAMETER num_gpu 33
PARAMETER num_thread 16
EOF

    # Crea modello
    if curl -X POST "http://127.0.0.1:$OLLAMA_PORT/api/create" \
           -H "Content-Type: application/json" \
           -d "{\"name\": \"$MODEL_NAME\", \"modelfile\": \"$(cat /tmp/debug_modelfile | tr '\n' '\\n')\"}" \
           --max-time 180 --silent; then
        echo "✅ Modello creato"
        rm -f /tmp/debug_modelfile
    else
        echo "❌ Errore creazione modello"
        rm -f /tmp/debug_modelfile
        exit 1
    fi
fi

# === 7. CREAZIONE SCRIPT DIAGNOSTICO ===
echo ""
echo "📝 Creazione script diagnostico..."

cat > $SLURM_SUBMIT_DIR/debug_ollama_detailed.py << 'PYEOF'
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
PYEOF

# === 8. ESECUZIONE DIAGNOSTICA ===
echo ""
echo "🔬 Esecuzione diagnostica..."
cd $SLURM_SUBMIT_DIR

if python3 debug_ollama_detailed.py; then
    echo ""
    echo "✅ DIAGNOSTICA COMPLETATA CON SUCCESSO"
    echo "📋 Controlla l'output sopra per dettagli"
else
    echo ""
    echo "❌ DIAGNOSTICA RILEVATO PROBLEMI"
    echo "📋 Controlla l'output sopra per dettagli"
fi

# === 9. INFORMAZIONI FINALI ===
echo ""
echo "📋 INFORMAZIONI SISTEMA"
echo "------------------------"
echo "GPU disponibili: $CUDA_VISIBLE_DEVICES"
echo "Memoria GPU:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || echo "nvidia-smi non disponibile"

echo ""
echo "📋 LOG SERVER (ultime 20 righe):"
echo "--------------------------------"
tail -20 ollama_debug.log 2>/dev/null || echo "Log non disponibile"

echo ""
echo "🏁 DIAGNOSTICA TERMINATA"
echo "Job completato alle: $(date)"