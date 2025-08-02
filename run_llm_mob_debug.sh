#!/bin/bash
#SBATCH --job-name=llm-mob
#SBATCH --account=IscrC_LLM-Mob
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=slurm-%j.out

# === 1. AMBIENTE ===
module purge
module load python/3.11.6--gcc--8.5.0
module load cuda/12.3
source $SLURM_SUBMIT_DIR/LLM/bin/activate

export CUDA_VISIBLE_DEVICES=0
export OLLAMA_GPU_OVERHEAD=0

# === 2. CONFIGURAZIONE MODELLO ===
MODEL_PATH="/leonardo/home/userexternal/smattiol/.ollama/models/blobs/sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29"
OLLAMA_PORT=39000
OLLAMA_BIN="$HOME/opt/ollama/bin/ollama"

echo $OLLAMA_PORT > $SLURM_SUBMIT_DIR/ollama_port.txt
echo "✅ Ho scritto la porta in ollama_port.txt: $(cat $SLURM_SUBMIT_DIR/ollama_port.txt)"

# === 3. AVVIO SERVER OLLAMA ===
echo "▶ Avvio server Ollama sulla porta $OLLAMA_PORT..."
echo "▶ Nodo di esecuzione: $(hostname)"
echo "▶ Working directory: $(pwd)"

# Configura le variabili d'ambiente per Ollama
export OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT
export OLLAMA_MODELS=$HOME/.ollama/models
export OLLAMA_KEEP_ALIVE=30m
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_LOAD_TIMEOUT=10m

# Avvia il server Ollama in background
echo "▶ Comando: $OLLAMA_BIN serve"
$OLLAMA_BIN serve > ollama_server.log 2>&1 &
SERVER_PID=$!

echo "▶ Server PID: $SERVER_PID"

# === 4. ATTESA SERVER E SETUP MODELLO ===
echo "▶ Attesa che il server Ollama sia pronto..."
MAX_WAIT=30
WAIT_INTERVAL=2

for i in $(seq 1 $MAX_WAIT); do
  # Controlla se il processo è ancora attivo
  if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "❌ Il server Ollama si è fermato inaspettatamente!"
    echo "▶ Log del server:"
    cat ollama_server.log
    exit 1
  fi
  
  # Prova a connettersi
  if curl -s --connect-timeout 3 --max-time 5 "http://127.0.0.1:$OLLAMA_PORT/api/tags" >/dev/null 2>&1; then
    echo "✓ Server Ollama attivo dopo $((i * WAIT_INTERVAL)) secondi"
    break
  fi
  
  if [ $((i % 5)) -eq 0 ]; then
    echo "⏳ Attesa server... ($((i * WAIT_INTERVAL))s / $((MAX_WAIT * WAIT_INTERVAL))s)"
  fi
  
  sleep $WAIT_INTERVAL
done

# Verifica finale della connessione
if ! curl -s --connect-timeout 3 --max-time 5 "http://127.0.0.1:$OLLAMA_PORT/api/tags" >/dev/null 2>&1; then
  echo "❌ Server non risponde dopo $((MAX_WAIT * WAIT_INTERVAL)) secondi"
  echo "▶ Log del server:"
  cat ollama_server.log
  kill $SERVER_PID 2>/dev/null
  exit 1
fi

# === 5. CARICAMENTO MODELLO ===
echo "▶ Caricamento del modello LLaMA..."
MODEL_NAME="llama3.1:8b"

# Se il modello non è disponibile, crealo dal blob
if ! curl -s "http://127.0.0.1:$OLLAMA_PORT/api/tags" | grep -q "$MODEL_NAME"; then
  echo "▶ Creazione modello $MODEL_NAME dal blob..."
  
  # Crea un Modelfile temporaneo
  cat > /tmp/Modelfile << EOF
FROM $MODEL_PATH
PARAMETER num_ctx 8192
PARAMETER num_batch 512
PARAMETER num_gpu 33
PARAMETER num_thread 32
EOF

  # Crea il modello
  if curl -X POST "http://127.0.0.1:$OLLAMA_PORT/api/create" \
       -H "Content-Type: application/json" \
       -d "{\"name\": \"$MODEL_NAME\", \"modelfile\": \"$(cat /tmp/Modelfile | tr '\n' '\\n')\"}" \
       --max-time 300; then
    echo "✓ Modello creato con successo"
  else
    echo "❌ Errore nella creazione del modello"
    kill $SERVER_PID 2>/dev/null
    exit 1
  fi
  
  rm -f /tmp/Modelfile
fi

# === 6. TEST DIAGNOSTICI DETTAGLIATI ===
echo "▶ Esecuzione test diagnostici..."

# Test 1: /api/tags dettagliato
echo "🔍 Test 1: /api/tags dettagliato"
curl -s "http://127.0.0.1:$OLLAMA_PORT/api/tags" | python3 -m json.tool || echo "❌ /api/tags fallito"

# Test 2: /api/generate semplice
echo "🔍 Test 2: /api/generate"
curl -X POST "http://127.0.0.1:$OLLAMA_PORT/api/generate" \
     -H "Content-Type: application/json" \
     -d '{"model": "'$MODEL_NAME'", "prompt": "Say hello", "stream": false}' \
     --max-time 60 | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f'✓ Done: {data.get(\"done\", False)}')
    print(f'✓ Response: \"{data.get(\"response\", \"EMPTY\")[:50]}...\"')
except Exception as e:
    print(f'❌ Error: {e}')
" || echo "❌ /api/generate fallito"

# Test 3: /api/chat
echo "🔍 Test 3: /api/chat"
curl -X POST "http://127.0.0.1:$OLLAMA_PORT/api/chat" \
     -H "Content-Type: application/json" \
     -d '{"model": "'$MODEL_NAME'", "messages": [{"role": "user", "content": "Say hello"}], "stream": false}' \
     --max-time 60 | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f'✓ Done: {data.get(\"done\", False)}')
    message = data.get('message', {})
    print(f'✓ Content: \"{message.get(\"content\", \"EMPTY\")[:50]}...\"')
except Exception as e:
    print(f'❌ Error: {e}')
" || echo "❌ /api/chat fallito"

# Test 4: Verifica log errori server
echo "🔍 Test 4: Ultimi log server"
echo "--- INIZIO LOG ---"
tail -20 ollama_server.log
echo "--- FINE LOG ---"

# === 7. CREAZIONE SCRIPT DIAGNOSTICO SEPARATO ===
echo "▶ Creazione script diagnostico..."
cat > debug_ollama.py << 'PYEOF'
#!/usr/bin/env python3
import requests
import json
import sys

def read_port():
    with open("ollama_port.txt") as f:
        return f.read().strip()

port = read_port()
base_url = f"http://127.0.0.1:{port}"

print(f"🔍 Testing {base_url}")

# Test minimale
payload = {"model": "llama3.1:8b", "prompt": "Hello", "stream": False}
try:
    resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=30)
    print(f"Status: {resp.status_code}")
    print(f"Content-Length: {len(resp.content)}")
    if resp.content:
        try:
            data = resp.json()
            print(f"JSON keys: {list(data.keys())}")
            print(f"Done: {data.get('done')}")
            print(f"Response: '{data.get('response', 'MISSING')[:100]}'")
        except json.JSONDecodeError:
            print(f"❌ Not JSON: {resp.text[:200]}")
    else:
        print("❌ Empty response body")
except Exception as e:
    print(f"❌ Error: {e}")
PYEOF

python3 debug_ollama.py

# === 8. SCRIPT PYTHON PRINCIPALE ===
cd $SLURM_SUBMIT_DIR
echo "▶ Lancio script Python alle $(date)..."

# Esegui con debug limitato
if python veronacard_mob_with_geom.py --max-users 5; then
  echo "✅ Script Python completato con successo"
else
  EXIT_CODE=$?
  echo "❌ Script Python fallito con codice $EXIT_CODE"
  echo "▶ Ultimi log server:"
  tail -20 ollama_server.log
fi

# === 9. CHIUSURA ===
echo "▶ Chiusura server alle $(date)..."
kill $SERVER_PID 2>/dev/null
wait $SERVER_PID 2>/dev/null

echo "▶ Log finale server:"
tail -20 ollama_server.log 2>/dev/null || echo "Log non disponibile"

echo "✅ Job completato!"