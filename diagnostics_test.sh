#!/bin/bash
#SBATCH --job-name=llm-mob-fixed
#SBATCH --account=IscrC_LLM-Mob
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=slurm-%j.out

echo "🚀 LLM-MOB CON OLLAMA 0.3.14 AGGIORNATO"
echo "========================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodo: $(hostname)"
echo "Data: $(date)"

# === 1. AMBIENTE ===
module purge
module load python/3.11.6--gcc--8.5.0
module load cuda/12.3
source $SLURM_SUBMIT_DIR/LLM/bin/activate

# Debug GPU
echo "🔍 INFO GPU:"
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# CORREZIONE CRITICA 1: Path Ollama aggiornato
OLLAMA_BIN="/leonardo/home/userexternal/smattiol/opt/bin/ollama"

# Verifica che la versione corretta sia disponibile
if [ ! -f "$OLLAMA_BIN" ]; then
    echo "❌ Ollama non trovato in $OLLAMA_BIN"
    echo "Contenuto directory opt:"
    ls -la /leonardo/home/userexternal/smattiol/opt/bin/
    exit 1
fi

OLLAMA_VERSION=$($OLLAMA_BIN --version 2>&1 | grep -o "0\.[0-9]\+\.[0-9]\+" || echo "unknown")
echo "📦 Versione Ollama: $OLLAMA_VERSION"

if [[ ! "$OLLAMA_VERSION" =~ ^0\.3\.[0-9]+$ ]]; then
    echo "⚠️ Versione Ollama non ottimale: $OLLAMA_VERSION (attesa: 0.3.x)"
    echo "Continuo comunque..."
fi

# CORREZIONE CRITICA 2: Variabili ambiente per Ollama 0.3.x
export CUDA_VISIBLE_DEVICES=0
export OLLAMA_DEBUG=1
export OLLAMA_HOST=127.0.0.1
export OLLAMA_ORIGINS="*"
export OLLAMA_MODELS="$HOME/.ollama/models"

# CORREZIONE CRITICA 3: Configurazioni moderne per 0.3.x
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_FLASH_ATTENTION=0
export OLLAMA_KEEP_ALIVE="30m"

# Rimuovi vecchie variabili non supportate in 0.3.x
unset OLLAMA_GPU_OVERHEAD
unset OLLAMA_HOST_GPU
unset OLLAMA_RUNNER_TIMEOUT
unset OLLAMA_LOAD_TIMEOUT
unset OLLAMA_REQUEST_TIMEOUT
unset OLLAMA_COMPLETION_TIMEOUT
unset OLLAMA_CONTEXT_TIMEOUT

# === 2. CONFIGURAZIONE ===
OLLAMA_PORT=39003
echo $OLLAMA_PORT > $SLURM_SUBMIT_DIR/ollama_port.txt
echo "✅ Porta: $OLLAMA_PORT"

# Pulizia processi precedenti
pkill -f "ollama serve" 2>/dev/null || true
sleep 3

# === 3. AVVIO SERVER ===
echo "🚀 Avvio server Ollama 0.3.x..."

# Avvio semplificato per 0.3.x (senza variabili deprecate)
OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT $OLLAMA_BIN serve > ollama_server.log 2>&1 &
SERVER_PID=$!

echo "   Comando: OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT $OLLAMA_BIN serve"
echo "   PID server: $SERVER_PID"

# Cleanup function
cleanup() {
    echo "🧹 Cleanup server..."
    if kill -0 $SERVER_PID 2>/dev/null; then
        kill -TERM $SERVER_PID 2>/dev/null
        sleep 3
        if kill -0 $SERVER_PID 2>/dev/null; then
            kill -KILL $SERVER_PID 2>/dev/null
        fi
    fi
    pkill -f "ollama" 2>/dev/null || true
}
trap cleanup EXIT

# === 4. ATTESA SERVER ===
echo "⏳ Attesa server..."
MAX_WAIT=30
WAIT_INTERVAL=2

for i in $(seq 1 $MAX_WAIT); do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "❌ Server morto prematurely!"
        echo "--- LOG SERVER ---"
        cat ollama_server.log
        exit 1
    fi
    
    # Test connessione API
    if curl -s --connect-timeout 2 --max-time 5 "http://127.0.0.1:$OLLAMA_PORT/api/tags" >/dev/null 2>&1; then
        echo "✅ Server attivo dopo $((i * WAIT_INTERVAL))s"
        break
    fi
    
    if [ $((i % 5)) -eq 0 ]; then
        echo "   Attesa... ($((i * WAIT_INTERVAL))s)"
        # Mostra ultimi log se disponibili
        if [ -f ollama_server.log ]; then
            echo "   Ultimi log:"
            tail -3 ollama_server.log | sed 's/^/     /'
        fi
    fi
    
    sleep $WAIT_INTERVAL
done

# Verifica finale connessione
if ! curl -s --max-time 10 "http://127.0.0.1:$OLLAMA_PORT/api/tags" >/dev/null 2>&1; then
    echo "❌ Server non risponde dopo $((MAX_WAIT * WAIT_INTERVAL))s"
    echo "--- LOG COMPLETO SERVER ---"
    cat ollama_server.log
    exit 1
fi

# === 5. VERIFICA MODELLO ===
echo "📥 Verifica modello..."
MODEL_NAME="llama3.1:8b"

# Lista modelli disponibili
echo "Modelli disponibili:"
MODELS_RESPONSE=$(curl -s "http://127.0.0.1:$OLLAMA_PORT/api/tags" || echo '{"models":[]}')
echo "$MODELS_RESPONSE" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    models = data.get('models', [])
    if models:
        for model in models:
            print(f\"  - {model.get('name', 'unknown')}\")
    else:
        print('  Nessun modello trovato')
except:
    print('  Errore nel parsing modelli')
"

# Controlla se il modello esiste
MODEL_EXISTS=$(echo "$MODELS_RESPONSE" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    models = [m.get('name', '') for m in data.get('models', [])]
    print('true' if '$MODEL_NAME' in models else 'false')
except:
    print('false')
")

if [ "$MODEL_EXISTS" = "false" ]; then
    echo "❌ Modello $MODEL_NAME non trovato!"
    echo "⚠️ Per Ollama 0.3.x, devi prima fare il pull del modello:"
    echo "   $OLLAMA_BIN pull $MODEL_NAME"
    
    # Tenta di fare il pull automaticamente
    echo "🔄 Tentativo pull automatico..."
    if timeout 300s $OLLAMA_BIN pull $MODEL_NAME; then
        echo "✅ Modello scaricato con successo"
    else
        echo "❌ Fallito pull del modello"
        echo "--- CONTENUTO DIRECTORY MODELLI ---"
        ls -la $HOME/.ollama/models/manifests/registry.ollama.ai/library/ 2>/dev/null || echo "Directory manifests non trovata"
        exit 1
    fi
else
    echo "✅ Modello $MODEL_NAME già disponibile"
fi

# === 6. TEST INFERENZA ===
echo "🧪 Test finale inferenza..."

# Test semplificato per 0.3.x
TEST_PAYLOAD='{
    "model": "'$MODEL_NAME'",
    "prompt": "Hello",
    "stream": false,
    "options": {
        "num_predict": 5,
        "temperature": 0.1
    }
}'

echo "   Payload test: $TEST_PAYLOAD"

# Test con timeout ragionevole
INFERENCE_RESULT=$(timeout 60s curl -X POST "http://127.0.0.1:$OLLAMA_PORT/api/generate" \
    -H "Content-Type: application/json" \
    -d "$TEST_PAYLOAD" \
    --max-time 55 \
    --connect-timeout 10 \
    -s 2>&1)

CURL_EXIT=$?

if [ $CURL_EXIT -eq 0 ]; then
    echo "$INFERENCE_RESULT" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    if data.get('done', False):
        response = data.get('response', '')
        print(f'✅ Test riuscito! Response: \"{response}\"')
        print('🎉 OLLAMA FUNZIONA CORRETTAMENTE!')
        sys.exit(0)
    else:
        print('❌ Test incompleto:', data)
        sys.exit(1)
except Exception as e:
    print(f'❌ Errore parsing risposta: {e}')
    print('Raw response:', repr(sys.stdin.read()[:200]))
    sys.exit(1)
" <<< "$INFERENCE_RESULT"
    
    if [ $? -eq 0 ]; then
        OLLAMA_READY=true
        echo "✅ Ollama pronto per lo script Python"
    else
        OLLAMA_READY=false
        echo "⚠️ Test fallito, uso modalità conservativa"
    fi
else
    echo "❌ Test fallito con codice $CURL_EXIT"
    echo "Raw result: $INFERENCE_RESULT"
    OLLAMA_READY=false
    
    echo "--- LOG RECENTE OLLAMA ---"
    tail -10 ollama_server.log
fi

# === 7. ESECUZIONE SCRIPT PYTHON ===
echo "🐍 Preparazione script Python..."

# Crea configurazione per Python script
cat > ollama_config.json << EOF
{
    "endpoint": "http://127.0.0.1:$OLLAMA_PORT",
    "model": "$MODEL_NAME",
    "timeout": 120,
    "max_retries": 3,
    "ollama_ready": $OLLAMA_READY
}
EOF

# Determina parametri in base al risultato del test
if [ "$OLLAMA_READY" = "true" ]; then
    echo "✅ Uso modalità normale"
    MAX_USERS="--max-users 25"
else
    echo "⚠️ Uso modalità conservativa (test fallito)"
    MAX_USERS="--max-users 3"
fi

# Aggiungi variabili environment per Python
export OLLAMA_ENDPOINT="http://127.0.0.1:$OLLAMA_PORT"
export OLLAMA_MODEL="$MODEL_NAME"

echo "🐍 Avvio script Python..."
cd $SLURM_SUBMIT_DIR

# Assicurati che requests sia installato
python3 -c "import requests" 2>/dev/null || pip3 install --user requests

# Esegui script con timeout
if timeout 20m python3 veronacard_mob_with_geom.py $MAX_USERS; then
    echo "✅ Script Python completato!"
    PYTHON_SUCCESS=true
else
    EXIT_CODE=$?
    echo "❌ Script Python fallito (exit code: $EXIT_CODE)"
    PYTHON_SUCCESS=false
fi

# === 8. STATISTICHE FINALI ===
echo ""
echo "📊 STATISTICHE FINALI"
echo "---------------------"
echo "Durata job: $SECONDS secondi"
echo "Ollama version: $OLLAMA_VERSION"
echo "Ollama ready: $OLLAMA_READY"
echo "Python success: $PYTHON_SUCCESS"

echo ""
echo "Risultati generati:"
ls -la results/ 2>/dev/null | head -5 || echo "Nessuna directory results"

echo ""
echo "Memoria GPU finale:"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader

echo ""
echo "--- STATO PROCESSO OLLAMA ---"
if kill -0 $SERVER_PID 2>/dev/null; then
    echo "Server ancora attivo"
else
    echo "Server terminato"
fi

echo ""
echo "--- ULTIMI LOG OLLAMA ---"
tail -20 ollama_server.log 2>/dev/null || echo "Nessun log disponibile"

echo ""
if [ "$PYTHON_SUCCESS" = "true" ]; then
    echo "🎉 JOB COMPLETATO CON SUCCESSO"
else
    echo "⚠️ JOB COMPLETATO CON ERRORI"
fi