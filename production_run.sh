#!/bin/bash
#SBATCH --job-name=llm-mob-production
#SBATCH --account=IscrC_LLM-Mob
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=slurm-%j.out

echo "🚀 LLM-MOB PRODUCTION RUN - TUTTI GLI UTENTI"
echo "============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodo: $(hostname)"
echo "Data: $(date)"
echo "💡 MODALITÀ: Processamento completo senza limiti utenti"
echo ""

# === 1. AMBIENTE ===
echo "📦 Caricamento moduli e ambiente..."
module purge
module load python/3.11.6--gcc--8.5.0
module load cuda/12.3
source $WORK/venv/bin/activate

echo "✓ Python: $(python3 --version)"
echo "✓ Virtual env: $VIRTUAL_ENV"
echo "✓ CUDA: $(nvcc --version | grep release)"

# Debug GPU
echo ""
echo "🔍 INFO GPU:"
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# === 2. CONFIGURAZIONE OLLAMA ===
echo ""
echo "⚙️  Configurazione Ollama..."

OLLAMA_BIN="/leonardo_work/IscrC_LLM-Mob/opt/bin/ollama"

# Verifica binario
if [ ! -f "$OLLAMA_BIN" ]; then
    echo "❌ ERRORE CRITICO: Ollama non trovato in $OLLAMA_BIN"
    echo "Contenuto directory opt:"
    ls -la /leonardo/home/userexternal/smattiol/opt/bin/
    exit 1
fi

OLLAMA_VERSION=$($OLLAMA_BIN --version 2>&1 | grep -o "0\.[0-9]\+\.[0-9]\+" || echo "unknown")
echo "✓ Versione Ollama: $OLLAMA_VERSION"

# Variabili ambiente ottimizzate per produzione
export CUDA_VISIBLE_DEVICES=0
export OLLAMA_DEBUG=0  # Disabilita debug per performance
export OLLAMA_HOST=127.0.0.1
export OLLAMA_ORIGINS="*"
export OLLAMA_MODELS="$WORK/.ollama/models"

# Configurazioni performance per produzione
export OLLAMA_NUM_PARALLEL=2  # Aumentato per production
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_FLASH_ATTENTION=0
export OLLAMA_KEEP_ALIVE="24h"  # Mantiene modello in memoria più a lungo

# Pulizia variabili deprecate
unset OLLAMA_GPU_OVERHEAD OLLAMA_HOST_GPU OLLAMA_RUNNER_TIMEOUT
unset OLLAMA_LOAD_TIMEOUT OLLAMA_REQUEST_TIMEOUT OLLAMA_COMPLETION_TIMEOUT OLLAMA_CONTEXT_TIMEOUT

# === 3. AVVIO SERVER ===
echo ""
echo "🚀 Avvio server Ollama per produzione..."

OLLAMA_PORT=39003
echo $OLLAMA_PORT > $SLURM_SUBMIT_DIR/ollama_port.txt
echo "✓ Porta server: $OLLAMA_PORT"

# Cleanup processi precedenti
echo "🧹 Pulizia processi precedenti..."
pkill -f "ollama serve" 2>/dev/null || true
sleep 5

# Avvio server
OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT $OLLAMA_BIN serve > ollama_production.log 2>&1 &
SERVER_PID=$!

echo "✓ Server PID: $SERVER_PID"
echo "✓ Log file: ollama_production.log"

# Cleanup function per produzione
cleanup() {
    echo ""
    echo "🧹 CLEANUP PRODUZIONE..."
    echo "🕐 Tempo totale job: $SECONDS secondi ($(($SECONDS / 3600))h $(($SECONDS % 3600 / 60))m)"
    
    # Salva statistiche finali
    echo "📊 Stato GPU finale:" 
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv
    
    # Conta risultati
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
    echo "✓ Cleanup completato"
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
        cat ollama_production.log
        exit 1
    fi
    
    # Test connessione
    if curl -s --connect-timeout 3 --max-time 5 "http://127.0.0.1:$OLLAMA_PORT/api/tags" >/dev/null 2>&1; then
        echo "✅ Server operativo dopo $((i * WAIT_INTERVAL))s"
        break
    fi
    
    if [ $((i % 10)) -eq 0 ]; then
        echo "   ⏱️  Attesa... ($((i * WAIT_INTERVAL))s) - controllo log:"
        if [ -f ollama_production.log ]; then
            tail -2 ollama_production.log | sed 's/^/     /'
        fi
    fi
    
    sleep $WAIT_INTERVAL
done

# Verifica finale
if ! curl -s --max-time 15 "http://127.0.0.1:$OLLAMA_PORT/api/tags" >/dev/null 2>&1; then
    echo "❌ ERRORE CRITICO: Server non risponde dopo $((MAX_WAIT * WAIT_INTERVAL))s"
    echo "--- LOG COMPLETO ---"
    cat ollama_production.log
    exit 1
fi

# === 5. VERIFICA E WARM-UP MODELLO ===
echo ""
echo "🔥 Preparazione modello per produzione..."

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
            print(f\"  ✓ {model.get('name', 'unknown')}\")
    else:
        print('  ⚠️ Nessun modello trovato')
except:
    print('  ❌ Errore parsing modelli')
"

# Verifica modello target
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
    echo "❌ ERRORE CRITICO: Modello $MODEL_NAME non trovato!"
    echo "🔄 Tentativo download automatico..."
    if timeout 600s $OLLAMA_BIN pull $MODEL_NAME; then
        echo "✅ Modello scaricato"
    else
        echo "❌ Download fallito"
        exit 1
    fi
else
    echo "✅ Modello $MODEL_NAME disponibile"
fi

# === 6. ESECUZIONE PRODUZIONE ===
echo ""
echo "🎯 AVVIO PRODUZIONE - PROCESSAMENTO COMPLETO"
echo "============================================="

# Configurazione per Python
cat > ollama_config.json << EOF
{
    "endpoint": "http://127.0.0.1:$OLLAMA_PORT",
    "model": "$MODEL_NAME",
    "timeout": 180,
    "max_retries": 5,
    "production_mode": true,
    "ollama_ready": true
}
EOF

# Variabili ambiente per Python
export OLLAMA_ENDPOINT="http://127.0.0.1:$OLLAMA_PORT"
export OLLAMA_MODEL="$MODEL_NAME"
export PRODUCTION_MODE=1

echo "✅ Configurazione produzione attiva"
echo "📊 Nessun limite utenti - processamento completo"
echo "⏱️  Tempo stimato: variabile (dipende dai dati)"
echo "🔄 Retry automatici: 5 tentativi per richiesta"
echo ""

cd /leonardo_work/IscrC_LLM-Mob/LLM-Mob-As-Mobility-Interpreter

# Verifica dipendenze
echo "📦 Verifica dipendenze Python..."
python3 -c "import requests, json, csv, pandas, numpy" 2>/dev/null || {
    echo "⚠️ Installazione dipendenze mancanti..."
    pip3 install --user requests pandas numpy
}

# Crea directory risultati se non esiste
mkdir -p results/
echo "📁 Directory risultati: $(pwd)/results/"

# Monitoring function
monitor_progress() {
    while true; do
        sleep 300  # Ogni 5 minuti
        if [ -d "results/" ]; then
            CURRENT_FILES=$(ls -1 results/*.csv 2>/dev/null | wc -l)
            CURRENT_SIZE=$(du -sh results/ 2>/dev/null | cut -f1 || echo "0")
            echo "📊 [$(date '+%H:%M:%S')] Progresso: $CURRENT_FILES files, $CURRENT_SIZE"
            
            # Monitor GPU
            GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
            GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
            echo "🔧 [$(date '+%H:%M:%S')] GPU: ${GPU_MEM}MB mem, ${GPU_UTIL}% util"
        fi
        
        # Controlla se il processo Python è ancora attivo
        if ! pgrep -f "veronacard_mob_with_geom.py" >/dev/null; then
            break
        fi
    done
}

# Avvio monitoring in background
monitor_progress &
MONITOR_PID=$!

# Avvio script principale SENZA limiti utenti
echo "🚀 AVVIO SCRIPT PRODUZIONE..."
echo "📝 Log dettagliato disponibile in tempo reale"
echo ""

PYTHON_START=$(date +%s)

# NESSUN parametro max-users = processamento completo
if python3 $WORK/LLM-Mob-As-Mobility-Interpreter/veronacard_mob_with_geom.py --file data/verona/dataset_veronacard_2014_2020/dati_2016.csv; then
    PYTHON_END=$(date +%s)
    PYTHON_TIME=$((PYTHON_END - PYTHON_START))
    
    echo ""
    echo "🎉 PRODUZIONE COMPLETATA CON SUCCESSO!"
    echo "⏱️  Tempo totale Python: $PYTHON_TIME secondi ($(($PYTHON_TIME / 3600))h $(($PYTHON_TIME % 3600 / 60))m)"
    PYTHON_SUCCESS=true
else
    PYTHON_EXIT=$?
    PYTHON_END=$(date +%s)
    PYTHON_TIME=$((PYTHON_END - PYTHON_START))
    
    echo ""
    echo "❌ ERRORE IN PRODUZIONE (exit code: $PYTHON_EXIT)"
    echo "⏱️  Tempo prima del fallimento: $PYTHON_TIME secondi"
    
    # Analisi tipo errore
    case $PYTHON_EXIT in
        124) echo "⚠️  Timeout - possibili risultati parziali" ;;
        130) echo "⚠️  Interruzione manuale (Ctrl+C)" ;;
        137) echo "⚠️  Killed (OOM o sistema)" ;;
        *) echo "⚠️  Errore generico - controllare log" ;;
    esac
    
    PYTHON_SUCCESS=false
fi

# Stop monitoring
kill $MONITOR_PID 2>/dev/null || true

# === 7. REPORT FINALE ===
echo ""
echo "📋 REPORT FINALE PRODUZIONE"
echo "============================"

TOTAL_JOB_TIME=$SECONDS
echo "⏱️  Tempo totale job: $TOTAL_JOB_TIME secondi ($(($TOTAL_JOB_TIME / 3600))h $(($TOTAL_JOB_TIME % 3600 / 60))m)"
echo "🔧 Versione Ollama: $OLLAMA_VERSION"
echo "📊 Modalità: Produzione completa (tutti gli utenti)"
echo "✅ Python success: $PYTHON_SUCCESS"

echo ""
echo "📁 RISULTATI GENERATI:"
if [ -d "results/" ]; then
    FINAL_COUNT=$(ls -1 results/*.csv 2>/dev/null | wc -l)
    FINAL_SIZE=$(du -sh results/ 2>/dev/null | cut -f1 || echo "N/A")
    echo "   📊 Total files: $FINAL_COUNT"
    echo "   💾 Total size: $FINAL_SIZE"
    echo ""
    echo "   📋 File generati:"
    ls -lah results/*.csv 2>/dev/null | tail -10 | while read line; do
        echo "     $line"
    done
    
    if [ $FINAL_COUNT -gt 10 ]; then
        echo "     ... (mostrati ultimi 10 di $FINAL_COUNT)"
    fi
else
    echo "   ⚠️ Nessun risultato trovato"
fi

echo ""
echo "🔧 STATO FINALE SISTEMA:"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv

echo ""
echo "📊 LOG OLLAMA (ultimi 20 righe):"
tail -20 ollama_production.log 2>/dev/null || echo "   Log non disponibile"

echo ""
if [ "$PYTHON_SUCCESS" = "true" ]; then
    echo "🎉 JOB PRODUZIONE COMPLETATO CON SUCCESSO!"
    echo "✅ Tutti i dataset sono stati processati senza limiti utenti"
else
    echo "⚠️ JOB PRODUZIONE TERMINATO CON ERRORI"
    echo "📋 Verificare log per dettagli specifici"
fi

echo ""
echo "🏁 Fine job: $(date)"
echo "============================================="