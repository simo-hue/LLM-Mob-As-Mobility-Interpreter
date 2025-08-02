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

echo "🚀 AVVIO LLM-MOB CON CORREZIONI GPU"
echo "===================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodo: $(hostname)"
echo "Data: $(date)"

# === 1. AMBIENTE E DEBUG GPU ===
module purge
module load python/3.11.6--gcc--8.5.0
module load cuda/12.3
source $SLURM_SUBMIT_DIR/LLM/bin/activate

# Debug GPU iniziale
echo "🔍 INFO GPU:"
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu --format=csv
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# CORREZIONE 1: Forzare GPU 0 e configurazioni CUDA
export CUDA_VISIBLE_DEVICES=0
export OLLAMA_GPU_OVERHEAD=0
export OLLAMA_HOST_GPU=1  # Forza uso GPU
export OLLAMA_DEBUG=1     # Debug mode

# CORREZIONE 2: Configurazioni aggressive per GPU
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=f16
export OLLAMA_RUNNERS_DIR=/tmp/ollama_runners_$$  # Runners isolati per questo job

# === 2. CONFIGURAZIONE MODELLO ===
MODEL_PATH="/leonardo/home/userexternal/smattiol/.ollama/models/blobs/sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29"
OLLAMA_PORT=39002  # Porta diversa per evitare conflitti
OLLAMA_BIN="$HOME/opt/ollama/bin/ollama"

echo $OLLAMA_PORT > $SLURM_SUBMIT_DIR/ollama_port.txt
echo "✅ Porta: $OLLAMA_PORT"

# === 3. CONFIGURAZIONI OLLAMA OTTIMIZZATE ===
export OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT
export OLLAMA_MODELS=$HOME/.ollama/models
export OLLAMA_KEEP_ALIVE=45m          # Più lungo per batch processing
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_LOAD_TIMEOUT=15m        # Più lungo per GPU lenta
export OLLAMA_REQUEST_TIMEOUT=300s    # Timeout richieste lunghe

# CORREZIONE 3: Pulizia ambiente precedente
pkill -f "ollama serve" 2>/dev/null || true
sleep 2

# === 4. AVVIO SERVER CON MONITORING ===
echo "🚀 Avvio server Ollama..."
echo "   Comando: $OLLAMA_BIN serve"

# Avvia server con log dettagliato
$OLLAMA_BIN serve > ollama_server.log 2>&1 &
SERVER_PID=$!

echo "   PID server: $SERVER_PID"

# Cleanup function
cleanup() {
    echo "🧹 Cleanup server..."
    if kill -0 $SERVER_PID 2>/dev/null; then
        kill $SERVER_PID 2>/dev/null
        sleep 3
        kill -9 $SERVER_PID 2>/dev/null
    fi
    # Pulisci runners temp
    rm -rf /tmp/ollama_runners_$$ 2>/dev/null || true
}
trap cleanup EXIT

# === 5. ATTESA SERVER MIGLIORATA ===
echo "⏳ Attesa server..."
MAX_WAIT=60  # Più tempo per GPU
WAIT_INTERVAL=3

for i in $(seq 1 $MAX_WAIT); do
    # Verifica processo
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "❌ Server morto!"
        echo "--- LOG SERVER ---"
        cat ollama_server.log
        exit 1
    fi
    
    # Test connessione
    if curl -s --connect-timeout 2 --max-time 3 "http://127.0.0.1:$OLLAMA_PORT/api/tags" >/dev/null 2>&1; then
        echo "✅ Server attivo dopo $((i * WAIT_INTERVAL))s"
        break
    fi
    
    if [ $((i % 10)) -eq 0 ]; then
        echo "   Attesa... ($((i * WAIT_INTERVAL))s / $((MAX_WAIT * WAIT_INTERVAL))s)"
        # Mostra gli ultimi log ogni 30s
        echo "   Ultimi log:"
        tail -3 ollama_server.log 2>/dev/null || echo "   (nessun log)"
    fi
    
    sleep $WAIT_INTERVAL
done

# Verifica finale
if ! curl -s "http://127.0.0.1:$OLLAMA_PORT/api/tags" >/dev/null 2>&1; then
    echo "❌ Server non risponde"
    echo "--- LOG COMPLETO ---"
    cat ollama_server.log
    exit 1
fi

# === 6. SETUP MODELLO CON PARAMETRI OTTIMIZZATI ===
echo "📥 Setup modello..."
MODEL_NAME="llama3.1:8b"

if ! curl -s "http://127.0.0.1:$OLLAMA_PORT/api/tags" | grep -q "$MODEL_NAME"; then
    echo "🔨 Creazione modello ottimizzato per GPU..."
    
    # CORREZIONE 4: Modelfile ottimizzato per problemi GPU
    cat > /tmp/Modelfile_optimized << EOF
FROM $MODEL_PATH

# Parametri ottimizzati per GPU con problemi
PARAMETER num_ctx 4096
PARAMETER num_batch 256
PARAMETER num_gpu 33
PARAMETER num_thread 8
PARAMETER rope_frequency_base 500000
PARAMETER rope_frequency_scale 1.0

# Parametri per ridurre timeout
PARAMETER num_predict 512
PARAMETER top_k 40
PARAMETER top_p 0.9
PARAMETER temperature 0.1
PARAMETER repeat_penalty 1.1

# Template più esplicito
TEMPLATE """<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
EOF
    
    # Crea modello con timeout lungo
    if curl -X POST "http://127.0.0.1:$OLLAMA_PORT/api/create" \
           -H "Content-Type: application/json" \
           -d "{\"name\": \"$MODEL_NAME\", \"modelfile\": \"$(cat /tmp/Modelfile_optimized | tr '\n' '\\n')\"}" \
           --max-time 600 --show-error; then  # 10 minuti per creazione
        echo "✅ Modello creato"
    else
        echo "❌ Errore creazione modello"
        echo "--- LOG SERVER ---"
        tail -20 ollama_server.log
        exit 1
    fi
    
    rm -f /tmp/Modelfile_optimized
else
    echo "✅ Modello già presente"
fi

# === 7. TEST FINALE PRE-PYTHON ===
echo "🧪 Test finale inferenza..."

# Test con timeout molto lungo e parametri ridotti
test_payload='{
    "model": "'$MODEL_NAME'",
    "prompt": "Hello",
    "stream": false,
    "options": {
        "num_predict": 5,
        "temperature": 0.1,
        "num_ctx": 2048
    }
}'

echo "   Payload test: $test_payload"

# Test con curl e timeout custom
if curl -X POST "http://127.0.0.1:$OLLAMA_PORT/api/generate" \
        -H "Content-Type: application/json" \
        -d "$test_payload" \
        --max-time 120 \
        --connect-timeout 10 \
        --retry 2 \
        --retry-delay 5 \
        -v | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(f'✅ Test OK: done={data.get(\"done\")}, response_len={len(str(data.get(\"response\", \"\")))}')
    if data.get('done') and data.get('response'):
        print('🎉 OLLAMA PRONTO PER PYTHON SCRIPT!')
        sys.exit(0)
    else:
        print('❌ Risposta incompleta')
        sys.exit(1)
except Exception as e:
    print(f'❌ Test fallito: {e}')
    sys.exit(1)
"; then
    echo "✅ Pre-test superato"
    TEST_OK=true
else
    echo "⚠️  Pre-test fallito, ma provo comunque con Python"
    TEST_OK=false
    echo "--- LOG RECENTE ---"
    tail -10 ollama_server.log
fi

# === 8. PYTHON SCRIPT CON RETRY AGGRESSIVO ===
echo "🐍 Avvio script Python..."
cd $SLURM_SUBMIT_DIR

# Se il test preliminare è fallito, usa parametri molto conservativi
if [ "$TEST_OK" = "false" ]; then
    echo "⚠️  Uso modalità conservativa (max 3 utenti)"
    MAX_USERS_ARG="--max-users 3"
else
    echo "✅ Uso modalità normale"
    MAX_USERS_ARG="--max-users 50"
fi

# Esegui script con gestione errori
if timeout 25m python veronacard_mob_with_geom.py $MAX_USERS_ARG --force; then
    echo "✅ Script Python completato!"
else
    EXIT_CODE=$?
    echo "❌ Script Python fallito (codice: $EXIT_CODE)"
    
    # Debug finale
    echo "--- STATO PROCESSO OLLAMA ---"
    if kill -0 $SERVER_PID 2>/dev/null; then
        echo "Server ancora attivo"
    else
        echo "Server morto durante esecuzione"
    fi
    
    echo "--- ULTIMI LOG OLLAMA ---"
    tail -15 ollama_server.log
    
    echo "--- MEMORIA GPU ---"
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv
fi

# === 9. STATISTICHE FINALI ===
echo ""
echo "📊 STATISTICHE FINALI"
echo "---------------------"
echo "Durata job: $SECONDS secondi"
echo "Risultati generati:"
ls -la results/ 2>/dev/null | tail -5 || echo "Nessun risultato"

echo "Memoria GPU finale:"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader

echo ""
echo "🏁 JOB COMPLETATO"