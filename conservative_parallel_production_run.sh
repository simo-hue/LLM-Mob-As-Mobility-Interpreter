#!/bin/bash
#SBATCH --job-name=conservative
#SBATCH --account=IscrC_LLM-Mob
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=00:45:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm-%j.out

echo "🚀 LLM-MOB CONSERVATIVE PRODUCTION RUN CON LIMITE 1000 UTENTI"
echo "====================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodo: $(hostname)"
echo "Data: $(date)"
echo "💡 MODALITÀ: Richiesta risorse minime per schedulazione rapida"
echo ""

# === RESTO DELLO SCRIPT IDENTICO AL SEQUENZIALE ===
# (Il contenuto è identico allo script sequenziale originale)

# === 1. AMBIENTE ===
echo "📦 Caricamento moduli e ambiente..."
module purge
module load python/3.11.6--gcc--8.5.0
module load cuda/12.3
source $WORK/venv/bin/activate

echo "✅ Python: $(python3 --version)"
echo "✅ Virtual env: $VIRTUAL_ENV"
echo "✅ CUDA: $(nvcc --version | grep release)"

# Debug GPU
echo ""
echo "🔍 INFO GPU:"
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# === 2. CONFIGURAZIONE OLLAMA ===
echo ""
echo "⚙️ Configurazione Ollama..."

OLLAMA_BIN="/leonardo_work/IscrC_LLM-Mob/opt/bin/ollama"

# Verifica binario
if [ ! -f "$OLLAMA_BIN" ]; then
    echo "❌ ERRORE CRITICO: Ollama non trovato in $OLLAMA_BIN"
    echo "Contenuto directory opt:"
    ls -la /leonardo/home/userexternal/smattiol/opt/bin/
    exit 1
fi

OLLAMA_VERSION=$($OLLAMA_BIN --version 2>&1 | grep -o "0\.[0-9]\+\.[0-9]\+" || echo "unknown")
echo "✅ Versione Ollama: $OLLAMA_VERSION"

# Variabili ambiente conservative
export CUDA_VISIBLE_DEVICES=0
export OLLAMA_DEBUG=0
export OLLAMA_HOST=127.0.0.1
export OLLAMA_ORIGINS="*"
export OLLAMA_MODELS="$WORK/.ollama/models"

# Configurazioni conservative per schedulazione rapida
export OLLAMA_NUM_PARALLEL=2
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_FLASH_ATTENTION=0
export OLLAMA_KEEP_ALIVE="2h"

# === 3. AVVIO SERVER ===
echo ""
echo "🚀 Avvio server Ollama (modalità conservative)..."

OLLAMA_PORT=39003
echo $OLLAMA_PORT > $SLURM_SUBMIT_DIR/ollama_port.txt
echo "✅ Porta server: $OLLAMA_PORT"

# Cleanup processi precedenti
echo "🧹 Pulizia processi precedenti..."
pkill -f "ollama serve" 2>/dev/null || true
sleep 5

# Avvio server
OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT $OLLAMA_BIN serve > ollama_conservative.log 2>&1 &
SERVER_PID=$!

echo "✅ Server PID: $SERVER_PID"
echo "✅ Log file: ollama_conservative.log"

# Cleanup function
cleanup() {
    echo ""
    echo "🧹 CLEANUP..."
    echo "⏱️ Tempo totale job: $SECONDS secondi ($(($SECONDS / 3600))h $(($SECONDS % 3600 / 60))m)"
    
    if [ -d "results/" ]; then
        TOTAL_RESULTS=$(ls -1 results/*.csv 2>/dev/null | wc -l)
        echo "📁 Risultati generati: $TOTAL_RESULTS files"
        echo "💾 Dimensione risultati: $(du -sh results/ 2>/dev/null | cut -f1)"
    fi
    
    # Graceful shutdown Ollama
    if kill -0 $SERVER_PID 2>/dev/null; then
        echo "🔄 Shutdown graceful Ollama..."
        kill -TERM $SERVER_PID 2>/dev/null
        for i in {1..10}; do
            if ! kill -0 $SERVER_PID 2>/dev/null; then break; fi
            sleep 1
        done
        if kill -0 $SERVER_PID 2>/dev/null; then
            echo "⚡ Force kill Ollama..."
            kill -KILL $SERVER_PID 2>/dev/null
        fi
    fi
    
    pkill -f "ollama" 2>/dev/null || true
    echo "✅ Cleanup completato"
}
trap cleanup EXIT

# === 4. ATTESA E VERIFICA SERVER ===
echo ""
echo "⏳ Attesa avvio server (max 60s)..."

MAX_WAIT=30
WAIT_INTERVAL=2

for i in $(seq 1 $MAX_WAIT); do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "❌ ERRORE: Server terminato prematuramente!"
        echo "--- LOG SERVER ---"
        cat ollama_conservative.log
        exit 1
    fi
    
    # Test connessione
    if curl -s --connect-timeout 3 --max-time 5 "http://127.0.0.1:$OLLAMA_PORT/api/tags" >/dev/null 2>&1; then
        echo "✅ Server operativo dopo $((i * WAIT_INTERVAL))s"
        break
    fi
    
    if [ $((i % 10)) -eq 0 ]; then
        echo "   ⏱️ Attesa... ($((i * WAIT_INTERVAL))s)"
        if [ -f ollama_conservative.log ]; then
            tail -2 ollama_conservative.log | sed 's/^/     /'
        fi
    fi
    
    sleep $WAIT_INTERVAL
done

# Verifica finale
if ! curl -s --max-time 15 "http://127.0.0.1:$OLLAMA_PORT/api/tags" >/dev/null 2>&1; then
    echo "❌ ERRORE CRITICO: Server non risponde"
    echo "--- LOG COMPLETO ---"
    cat ollama_conservative.log
    exit 1
fi

# === 5. VERIFICA MODELLO ===
echo ""
echo "🔥 Preparazione modello..."

MODEL_NAME="llama3.1:8b"

# Lista modelli
echo "📋 Modelli disponibili:"
MODELS_RESPONSE=$(curl -s "http://127.0.0.1:$OLLAMA_PORT/api/tags" || echo '{"models":[]}')
echo "$MODELS_RESPONSE" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    models = data.get('models', [])
    if models:
        for model in models:
            print(f\"  ✅ {model.get('name', 'unknown')}\")
    else:
        print('  ⚠️ Nessun modello trovato')
except:
    print('  ❌ Errore parsing modelli')
"

# === 6. ESECUZIONE CONSERVATIVE ===
echo ""
echo "🎯 AVVIO PRODUZIONE CONSERVATIVE"
echo "================================"

# Configurazione
cat > ollama_config.json << EOF
{
    "endpoint": "http://127.0.0.1:$OLLAMA_PORT",
    "model": "$MODEL_NAME",
    "timeout": 180,
    "max_retries": 5,
    "production_mode": true,
    "conservative_mode": true,
    "ollama_ready": true
}
EOF

export OLLAMA_ENDPOINT="http://127.0.0.1:$OLLAMA_PORT"
export OLLAMA_MODEL="$MODEL_NAME"
export PRODUCTION_MODE=1

echo "✅ Modalità conservative attiva per schedulazione rapida"
echo "💡 Elaborazione con risorse minime ma efficace"
echo ""

cd /leonardo_work/IscrC_LLM-Mob/LLM-Mob-As-Mobility-Interpreter

# Verifica dipendenze
echo "📦 Verifica dipendenze Python..."
python3 -c "import requests, json, csv, pandas, numpy" 2>/dev/null || {
    echo "⚠️ Installazione dipendenze mancanti..."
    pip3 install --user requests pandas numpy
}

mkdir -p results/
echo "📁 Directory risultati: $(pwd)/results/"

# Esecuzione script con limite utenti per rispettare tempo
echo "🚀 AVVIO SCRIPT CONSERVATIVE..."
echo "💡 Limite 1000 utenti per rispettare finestra temporale"
echo ""

PYTHON_START=$(date +%s)

if python3 $WORK/LLM-Mob-As-Mobility-Interpreter/veronacard_mob_with_geom.py --append --max-users 1000; then
    PYTHON_END=$(date +%s)
    PYTHON_TIME=$((PYTHON_END - PYTHON_START))
    
    echo ""
    echo "🎉 PRODUZIONE CONSERVATIVE COMPLETATA!"
    echo "⏱️ Tempo totale Python: $PYTHON_TIME secondi"
    PYTHON_SUCCESS=true
else
    PYTHON_EXIT=$?
    echo ""
    echo "❌ ERRORE IN PRODUZIONE (exit code: $PYTHON_EXIT)"
    PYTHON_SUCCESS=false
fi

# === 7. REPORT FINALE ===
echo ""
echo "📋 REPORT FINALE CONSERVATIVE"
echo "============================"

TOTAL_JOB_TIME=$SECONDS
echo "⏱️ Tempo totale job: $TOTAL_JOB_TIME secondi"
echo "🔧 Modalità: Conservative (schedulazione rapida)"
echo "✅ Python success: $PYTHON_SUCCESS"

if [ -d "results/" ]; then
    FINAL_COUNT=$(ls -1 results/*.csv 2>/dev/null | wc -l)
    FINAL_SIZE=$(du -sh results/ 2>/dev/null | cut -f1 || echo "N/A")
    echo "📊 Files: $FINAL_COUNT, Size: $FINAL_SIZE"
fi

echo ""
if [ "$PYTHON_SUCCESS" = "true" ]; then
    echo "🎉 JOB CONSERVATIVE COMPLETATO!"
    echo "💡 Schedulazione rapida con risorse minime"
else
    echo "⚠️ JOB TERMINATO CON ERRORI"
fi

echo ""
echo "🏁 Fine job: $(date)"
echo "============================"