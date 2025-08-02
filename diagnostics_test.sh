#!/bin/bash
#SBATCH --job-name=llm-mob-timeout-fixed
#SBATCH --account=IscrC_LLM-Mob
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=slurm-%j.out

echo "🚀 LLM-MOB CON CORREZIONE TIMEOUT DEFINITIVA"
echo "=============================================="
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

# CORREZIONE PRINCIPALE: Configurazioni anti-timeout
export CUDA_VISIBLE_DEVICES=0
export OLLAMA_GPU_OVERHEAD=0
export OLLAMA_HOST_GPU=1
export OLLAMA_DEBUG=1

# NUOVE VARIABILI CRITICHE PER TIMEOUT
export OLLAMA_RUNNER_TIMEOUT=600s          # 10 minuti per runner
export OLLAMA_LOAD_TIMEOUT=300s            # 5 minuti per caricamento modello
export OLLAMA_REQUEST_TIMEOUT=600s         # 10 minuti per singola richiesta
export OLLAMA_COMPLETION_TIMEOUT=600s      # 10 minuti per completion
export OLLAMA_KEEP_ALIVE=30m               # Mantieni modello in memoria 30 min
export OLLAMA_CONTEXT_TIMEOUT=600s         # Timeout per context switching

# Ottimizzazioni GPU
export OLLAMA_FLASH_ATTENTION=0            # Disabilita flash attention (problematico su alcuni setup)
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=1

# === 2. CONFIGURAZIONE ===
MODEL_PATH="/leonardo/home/userexternal/smattiol/.ollama/models/blobs/sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29"
OLLAMA_PORT=39003  # Nuova porta
OLLAMA_BIN="$HOME/opt/ollama/bin/ollama"

echo $OLLAMA_PORT > $SLURM_SUBMIT_DIR/ollama_port.txt
echo "✅ Porta: $OLLAMA_PORT"

export OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT
export OLLAMA_MODELS=$HOME/.ollama/models

# Pulizia
pkill -f "ollama serve" 2>/dev/null || true
sleep 3

# === 3. AVVIO SERVER CON CONFIGURAZIONI ANTI-TIMEOUT ===
echo "🚀 Avvio server con timeout estesi..."

# Avvia con parametri espliciti per timeout
OLLAMA_RUNNER_TIMEOUT=600s \
OLLAMA_LOAD_TIMEOUT=300s \
OLLAMA_REQUEST_TIMEOUT=600s \
OLLAMA_COMPLETION_TIMEOUT=600s \
OLLAMA_KEEP_ALIVE=30m \
$OLLAMA_BIN serve > ollama_server_fixed.log 2>&1 &
SERVER_PID=$!

echo "   PID server: $SERVER_PID"

# Cleanup potenziato
cleanup() {
    echo "🧹 Cleanup con terminazione gentile..."
    if kill -0 $SERVER_PID 2>/dev/null; then
        # Prima terminazione gentile
        kill -TERM $SERVER_PID 2>/dev/null
        sleep 5
        # Se ancora vivo, forza
        if kill -0 $SERVER_PID 2>/dev/null; then
            kill -KILL $SERVER_PID 2>/dev/null
        fi
    fi
    # Pulizia processi orfani
    pkill -f "ollama runner" 2>/dev/null || true
}
trap cleanup EXIT

# === 4. ATTESA SERVER ESTESA ===
echo "⏳ Attesa server (fino a 3 minuti)..."
MAX_WAIT=60  # 3 minuti
WAIT_INTERVAL=3

for i in $(seq 1 $MAX_WAIT); do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "❌ Server morto!"
        echo "--- LOG SERVER ---"
        cat ollama_server_fixed.log
        exit 1
    fi
    
    if curl -s --connect-timeout 2 --max-time 5 "http://127.0.0.1:$OLLAMA_PORT/api/tags" >/dev/null 2>&1; then
        echo "✅ Server attivo dopo $((i * WAIT_INTERVAL))s"
        break
    fi
    
    if [ $((i % 10)) -eq 0 ]; then
        echo "   Attesa... ($((i * WAIT_INTERVAL))s / $((MAX_WAIT * WAIT_INTERVAL))s)"
        tail -3 ollama_server_fixed.log 2>/dev/null || echo "   (nessun log)"
    fi
    
    sleep $WAIT_INTERVAL
done

if ! curl -s "http://127.0.0.1:$OLLAMA_PORT/api/tags" >/dev/null 2>&1; then
    echo "❌ Server non risponde"
    echo "--- LOG COMPLETO ---"
    cat ollama_server_fixed.log
    exit 1
fi

# === 5. SETUP MODELLO MINIMALISTA (RIMUOVE PARAMETRI PROBLEMATICI) ===
echo "📥 Setup modello anti-timeout..."
MODEL_NAME="llama3.1:8b"

if ! curl -s "http://127.0.0.1:$OLLAMA_PORT/api/tags" | grep -q "$MODEL_NAME"; then
    echo "🔨 Creazione modello MINIMALISTA..."
    
    # CORREZIONE CRITICA: Modelfile minimalista senza parametri che causano timeout
    cat > /tmp/Modelfile_minimal << EOF
FROM $MODEL_PATH

# Solo parametri essenziali anti-timeout
PARAMETER num_ctx 2048
PARAMETER num_batch 128
PARAMETER num_gpu 33
PARAMETER num_thread 4
PARAMETER temperature 0.7
PARAMETER top_p 0.9

# Template semplificato
TEMPLATE """<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
EOF
    
    echo "Modelfile contenuto:"
    cat /tmp/Modelfile_minimal
    echo ""
    
    # Crea con timeout molto lungo
    echo "Creando modello (timeout 10 min)..."
    if curl -X POST "http://127.0.0.1:$OLLAMA_PORT/api/create" \
           -H "Content-Type: application/json" \
           -d "{\"name\": \"$MODEL_NAME\", \"modelfile\": \"$(cat /tmp/Modelfile_minimal | sed 's/"/\\"/g' | tr '\n' '\\n')\"}" \
           --max-time 600 \
           --connect-timeout 30 \
           --retry 1; then
        echo "✅ Modello creato"
    else
        echo "❌ Errore creazione modello"
        tail -20 ollama_server_fixed.log
        exit 1
    fi
    
    rm -f /tmp/Modelfile_minimal
else
    echo "✅ Modello già presente"
fi

# === 6. TEST PROGRESSIVO CON TIMEOUT CUSTOM ===
echo "🧪 Test progressivo anti-timeout..."

# Test 1: Micro (1 token, timeout corto)
echo "Test 1: Micro response"
micro_payload='{
    "model": "'$MODEL_NAME'",
    "prompt": "Hi",
    "stream": false,
    "options": {
        "num_predict": 1,
        "temperature": 0.1
    }
}'

echo "   Payload: $micro_payload"

if timeout 30s curl -X POST "http://127.0.0.1:$OLLAMA_PORT/api/generate" \
    -H "Content-Type: application/json" \
    -d "$micro_payload" \
    --max-time 25 \
    --connect-timeout 5 \
    -s | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f'✅ Micro test: done={data.get(\"done\")}, response=\"{data.get(\"response\", \"\")}\"')
    if data.get('done'):
        sys.exit(0)
    else:
        sys.exit(1)
except Exception as e:
    print(f'❌ Micro test failed: {e}')
    sys.exit(1)
"; then
    echo "✅ Micro test OK, procedo con test completo"
    
    # Test 2: Risposta normale (solo se micro OK)
    echo "Test 2: Risposta normale"
    normal_payload='{
        "model": "'$MODEL_NAME'",
        "prompt": "What is Rome?",
        "stream": false,
        "options": {
            "num_predict": 20,
            "temperature": 0.3
        }
    }'
    
    echo "   Timeout: 2 minuti"
    if timeout 120s curl -X POST "http://127.0.0.1:$OLLAMA_PORT/api/generate" \
        -H "Content-Type: application/json" \
        -d "$normal_payload" \
        --max-time 115 \
        --connect-timeout 10 \
        -s | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    response = data.get('response', '')
    print(f'✅ Normal test: done={data.get(\"done\")}, response_len={len(response)}')
    if data.get('done') and len(response) > 5:
        print(f'   Response preview: \"{response[:50]}...\"')
        print('🎉 OLLAMA FUNZIONA CORRETTAMENTE!')
        sys.exit(0)
    else:
        print('❌ Risposta incompleta o vuota')
        sys.exit(1)
except Exception as e:
    print(f'❌ Normal test failed: {e}')
    sys.exit(1)
    "; then
        echo "✅ Test completo OK!"
        OLLAMA_READY=true
    else
        echo "⚠️ Test normale fallito, uso modalità conservativa"
        OLLAMA_READY=false
        echo "--- LOG RECENTE ---"
        tail -10 ollama_server_fixed.log
    fi
else
    echo "❌ Micro test fallito, problema grave"
    OLLAMA_READY=false
    echo "--- LOG COMPLETO ---"
    tail -20 ollama_server_fixed.log
fi

# === 7. CONFIGURAZIONE PYTHON SCRIPT ===
echo "🐍 Configurazione per script Python..."
cd $SLURM_SUBMIT_DIR

# Crea file di configurazione per il Python script con timeout custom
cat > ollama_config.json << EOF
{
    "host": "127.0.0.1",
    "port": $OLLAMA_PORT,
    "model": "$MODEL_NAME",
    "timeout": 300,
    "retry_attempts": 3,
    "retry_delay": 10,
    "request_params": {
        "temperature": 0.3,
        "num_predict": 100,
        "top_p": 0.9
    }
}
EOF

echo "✅ Config salvata in ollama_config.json"

# Determina parametri per Python script
if [ "$OLLAMA_READY" = "true" ]; then
    echo "✅ Uso modalità normale (50 utenti)"
    MAX_USERS_ARG="--max-users 50"
    PYTHON_TIMEOUT="25m"
else
    echo "⚠️ Uso modalità ridotta (5 utenti per test)"
    MAX_USERS_ARG="--max-users 5"
    PYTHON_TIMEOUT="10m"
fi

# === 8. ESECUZIONE PYTHON SCRIPT ===
echo "🐍 Avvio script Python con timeout $PYTHON_TIMEOUT..."

# Aggiungi variabili d'ambiente per il Python script
export OLLAMA_ENDPOINT="http://127.0.0.1:$OLLAMA_PORT"
export OLLAMA_MODEL="$MODEL_NAME"
export OLLAMA_TIMEOUT=300
export OLLAMA_RETRIES=3

# Esegui con monitoring
if timeout $PYTHON_TIMEOUT python veronacard_mob_with_geom.py $MAX_USERS_ARG --config ollama_config.json --force; then
    echo "✅ Script Python completato!"
    PYTHON_SUCCESS=true
else
    EXIT_CODE=$?
    echo "❌ Script Python fallito (codice: $EXIT_CODE)"
    PYTHON_SUCCESS=false
    
    # Debug dettagliato
    echo "--- STATO SERVER ---"
    if kill -0 $SERVER_PID 2>/dev/null; then
        echo "Server ancora attivo"
        echo "Modelli caricati:"
        curl -s "http://127.0.0.1:$OLLAMA_PORT/api/tags" | grep -o '"name":"[^"]*"' || echo "Errore nel recuperare modelli"
    else
        echo "Server morto durante esecuzione"
    fi
    
    echo "--- ULTIMI 20 LOG OLLAMA ---"
    tail -20 ollama_server_fixed.log
    
    echo "--- MEMORIA GPU ---"
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
fi

# === 9. STATISTICHE E CLEANUP ===
echo ""
echo "📊 STATISTICHE FINALI"
echo "====================="
echo "Durata job: $SECONDS secondi"
echo "Ollama ready: $OLLAMA_READY"
echo "Python success: $PYTHON_SUCCESS"

echo ""
echo "Risultati generati:"
ls -la results/ 2>/dev/null | tail -5 || echo "Nessun risultato trovato"

echo ""
echo "Memoria GPU finale:"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader

echo ""
echo "File di config e log generati:"
ls -la ollama_*.{json,log,txt} 2>/dev/null || echo "Nessun file di config"

echo ""
if [ "$PYTHON_SUCCESS" = "true" ]; then
    echo "🎉 JOB COMPLETATO CON SUCCESSO!"
else
    echo "⚠️ JOB COMPLETATO CON ERRORI - Controllare i log"
fi