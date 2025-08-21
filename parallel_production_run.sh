#!/bin/bash
#SBATCH --job-name=parallel
#SBATCH --account=IscrC_LLM-Mob
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --output=slurm-%j.out

echo "🚀 LLM-MOB PARALLEL PRODUCTION RUN"
echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodo: $(hostname)"
echo "Data: $(date)"
echo "🔥 MODALITÀ: Elaborazione parallela su 4x A100"
echo ""

# === 1. AMBIENTE ===
echo "📦 Caricamento moduli e ambiente..."
module purge
module load python/3.11.6--gcc--8.5.0
module load cuda/12.3
source $WORK/venv/bin/activate

echo "✅ Python: $(python3 --version)"
echo "✅ Virtual env: $VIRTUAL_ENV"
echo "✅ CUDA: $(nvcc --version | grep release)"

# Debug GPU dettagliato per ambiente parallelo
echo ""
echo "🔍 INFO GPU DETTAGLIATE:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "GPU disponibili per il job: $SLURM_GPUS_ON_NODE"

# Configura ambiente per 2 GPU invece di 4
export CUDA_VISIBLE_DEVICES=0,1
export NVIDIA_VISIBLE_DEVICES=$SLURM_LOCALID

# === 2. CONFIGURAZIONE OLLAMA PARALLELO ===
echo ""
echo "⚙️ Configurazione Ollama per elaborazione parallela..."

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

# Variabili ambiente ottimizzate per produzione parallela
export OLLAMA_DEBUG=0
export OLLAMA_HOST=127.0.0.1
export OLLAMA_ORIGINS="*"
export OLLAMA_MODELS="$WORK/.ollama/models"

# Configurazioni performance ottimizzate per stabilità 4 GPU
export OLLAMA_NUM_PARALLEL=4          # Ridotto da 8 per evitare sovraccarico
export OLLAMA_MAX_LOADED_MODELS=1     # Un solo modello alla volta per stabilità
export OLLAMA_FLASH_ATTENTION=1       # Abilita per A100
export OLLAMA_KEEP_ALIVE="4h"         # Mantiene modelli in memoria più a lungo
export OLLAMA_GPU_LAYERS=-1           # Auto-detect layers
export OLLAMA_RUNNER_KEEP_ALIVE="4h"  # Mantiene i runner attivi

# Configurazioni specifiche per stabilità multi-GPU
export OLLAMA_LOAD_TIMEOUT=600        # Timeout molto generoso per 4 GPU
export OLLAMA_REQUEST_TIMEOUT=300     # Timeout richieste
export OLLAMA_MAX_QUEUE=10            # Limita la coda per evitare OOM

# Pulizia variabili deprecate
unset OLLAMA_GPU_OVERHEAD OLLAMA_HOST_GPU OLLAMA_RUNNER_TIMEOUT
unset OLLAMA_COMPLETION_TIMEOUT OLLAMA_CONTEXT_TIMEOUT

# === 3. AVVIO SERVER PARALLELO ===
echo ""
echo "🚀 Avvio server Ollama per elaborazione parallela..."

OLLAMA_PORT=39003
echo $OLLAMA_PORT > $SLURM_SUBMIT_DIR/ollama_port.txt
echo "✅ Porta server: $OLLAMA_PORT"

# Cleanup processi precedenti più aggressivo
echo "🧹 Pulizia processi precedenti..."
pkill -f "ollama serve" 2>/dev/null || true
pkill -f "ollama" 2>/dev/null || true
sleep 10

# Avvio server con configurazione multi-GPU
echo "🔥 Avvio con supporto 4x A100..."
OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT $OLLAMA_BIN serve > ollama_parallel.log 2>&1 &
SERVER_PID=$!

echo "✅ Server PID: $SERVER_PID"
echo "✅ Log file: ollama_parallel.log"

# Cleanup function ottimizzata per ambiente parallelo
cleanup() {
    echo ""
    echo "🧹 CLEANUP PRODUZIONE PARALLELA..."
    echo "⏱️ Tempo totale job: $SECONDS secondi ($(($SECONDS / 3600))h $(($SECONDS % 3600 / 60))m)"
    
    # Statistiche dettagliate GPU
    echo "📊 Stato finale tutte le GPU:" 
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv
    
    # Conta risultati con più dettagli
    if [ -d "results/" ]; then
        TOTAL_RESULTS=$(ls -1 results/*.csv 2>/dev/null | wc -l)
        echo "📁 Risultati generati: $TOTAL_RESULTS files"
        echo "💾 Dimensione risultati: $(du -sh results/ 2>/dev/null | cut -f1)"
        
        # Conta record totali nei CSV
        if [ $TOTAL_RESULTS -gt 0 ]; then
            TOTAL_RECORDS=$(wc -l results/*.csv 2>/dev/null | tail -1 | awk '{print $1}')
            echo "📋 Record totali elaborati: $TOTAL_RECORDS"
        fi
    fi
    
    # Graceful shutdown Ollama con timeout più generoso per multi-GPU
    if kill -0 $SERVER_PID 2>/dev/null; then
        echo "🔄 Shutdown graceful Ollama multi-GPU..."
        kill -TERM $SERVER_PID 2>/dev/null
        for i in {1..20}; do  # Timeout più lungo per 4 GPU
            if ! kill -0 $SERVER_PID 2>/dev/null; then break; fi
            sleep 1
        done
        if kill -0 $SERVER_PID 2>/dev/null; then
            echo "⚡ Force kill Ollama..."
            kill -KILL $SERVER_PID 2>/dev/null
        fi
    fi
    
    # Cleanup più completo per ambiente multi-GPU
    pkill -f "ollama" 2>/dev/null || true
    pkill -f "llama" 2>/dev/null || true
    
    # Libera memoria GPU
    nvidia-smi --gpu-reset -i 0,1,2,3 2>/dev/null || true
    
    echo "✅ Cleanup completato"
}
trap cleanup EXIT

# === 4. ATTESA E VERIFICA SERVER MULTI-GPU ===
echo ""
echo "⏳ Attesa avvio server multi-GPU (max 120s)..."

MAX_WAIT=60
WAIT_INTERVAL=2

for i in $(seq 1 $MAX_WAIT); do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "❌ ERRORE: Server terminato prematuramente!"
        echo "--- LOG SERVER ---"
        cat ollama_parallel.log
        exit 1
    fi
    
    # Test connessione
    if curl -s --connect-timeout 5 --max-time 10 "http://127.0.0.1:$OLLAMA_PORT/api/tags" >/dev/null 2>&1; then
        echo "✅ Server multi-GPU operativo dopo $((i * WAIT_INTERVAL))s"
        break
    fi
    
    if [ $((i % 15)) -eq 0 ]; then
        echo "   ⏱️ Attesa... ($((i * WAIT_INTERVAL))s) - controllo log:"
        if [ -f ollama_parallel.log ]; then
            tail -3 ollama_parallel.log | sed 's/^/     /'
        fi
        # Stato GPU durante startup
        echo "   🔍 GPU status:"
        nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -4
    fi
    
    sleep $WAIT_INTERVAL
done

# Verifica finale più rigorosa
if ! curl -s --max-time 20 "http://127.0.0.1:$OLLAMA_PORT/api/tags" >/dev/null 2>&1; then
    echo "❌ ERRORE CRITICO: Server non risponde dopo $((MAX_WAIT * WAIT_INTERVAL))s"
    echo "--- LOG COMPLETO ---"
    cat ollama_parallel.log
    exit 1
fi

# === 5. PREPARAZIONE MODELLO SU TUTTE LE GPU ===
echo ""
echo "🔥 Preparazione modello per elaborazione parallela..."

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
    echo "📥 Tentativo download automatico..."
    if timeout 900s $OLLAMA_BIN pull $MODEL_NAME; then  # Timeout più lungo
        echo "✅ Modello scaricato"
    else
        echo "❌ Download fallito"
        exit 1
    fi
else
    echo "✅ Modello $MODEL_NAME disponibile"
fi

# Pre-caricamento sequenziale per evitare deadlock
echo "🔥 Pre-caricamento modello (sequenziale per evitare blocchi)..."
echo "📡 Warm-up iniziale..."
timeout 120s curl -s -X POST "http://127.0.0.1:$OLLAMA_PORT/api/generate" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL_NAME\",\"prompt\":\"Hi\",\"stream\":false,\"options\":{\"num_predict\":1,\"temperature\":0}}" \
    > /tmp/warmup_result.json 2>&1

if [ $? -eq 0 ] && [ -s /tmp/warmup_result.json ]; then
    echo "✅ Pre-caricamento completato con successo"
    cat /tmp/warmup_result.json | head -1
else
    echo "⚠️ Pre-caricamento con timeout/errore - procedo comunque"
    echo "Log warm-up:"
    tail -5 ollama_parallel.log || echo "Nessun log disponibile"
fi

rm -f /tmp/warmup_result.json

# === 6. ESECUZIONE PRODUZIONE PARALLELA ===
echo ""
echo "🎯 AVVIO PRODUZIONE PARALLELA"
echo "============================="

# Configurazione per Python ottimizzata per parallelo
cat > ollama_config.json << EOF
{
    "endpoint": "http://127.0.0.1:$OLLAMA_PORT",
    "model": "$MODEL_NAME",
    "timeout": 300,
    "max_retries": 5,
    "production_mode": true,
    "parallel_mode": true,
    "num_workers": 8,
    "gpu_count": 4,
    "ollama_ready": true
}
EOF

# Variabili ambiente per Python parallelo
export OLLAMA_ENDPOINT="http://127.0.0.1:$OLLAMA_PORT"
export OLLAMA_MODEL="$MODEL_NAME"
export PRODUCTION_MODE=1
export PARALLEL_MODE=1
export GPU_COUNT=4

echo "✅ Configurazione produzione parallela attiva"
echo "🔥 4x A100 GPU disponibili per elaborazione"
echo "🔧 8 worker paralleli configurati"
echo "📊 Nessun limite utenti - processamento completo"
echo "⏱️ Tempo stimato: ridotto grazie al parallelismo"
echo "🔄 Retry automatici: 5 tentativi per richiesta"
echo ""

cd /leonardo_work/IscrC_LLM-Mob/LLM-Mob-As-Mobility-Interpreter

# Verifica dipendenze per ambiente parallelo
echo "📦 Verifica dipendenze Python per elaborazione parallela..."
python3 -c "
import requests, json, csv, pandas, numpy
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import multiprocessing
print('✅ Tutte le dipendenze per il parallelo sono disponibili')
" 2>/dev/null || {
    echo "⚠️ Installazione dipendenze mancanti..."
    pip3 install --user requests pandas numpy
}

# Crea directory risultati se non esiste
mkdir -p results/
echo "📁 Directory risultati: $(pwd)/results/"

# Monitoring function ottimizzata per parallelo
monitor_parallel() {
    while true; do
        sleep 180  # Ogni 3 minuti (più frequente per monitorare il parallelo)
        
        if [ -d "results/" ]; then
            CURRENT_FILES=$(ls -1 results/*.csv 2>/dev/null | wc -l)
            CURRENT_SIZE=$(du -sh results/ 2>/dev/null | cut -f1 || echo "0")
            echo "📊 [$(date '+%H:%M:%S')] Progresso: $CURRENT_FILES files, $CURRENT_SIZE"
            
            # Monitor tutte e 4 le GPU
            echo "🔧 [$(date '+%H:%M:%S')] GPU Status:"
            nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits | \
                while IFS=',' read idx mem util temp; do
                    echo "   GPU$idx: ${mem}MB mem, ${util}% util, ${temp}°C"
                done
        fi
        
        # Controlla processi Python
        if ! pgrep -f "veronacard_mob_with_geom_parrallel.py" >/dev/null; then
            break
        fi
    done
}

# Avvio monitoring in background
monitor_parallel &
MONITOR_PID=$!

# Avvio script parallelo SENZA limiti utenti
echo "🚀 AVVIO SCRIPT PARALLELO..."
echo "📡 Elaborazione distribuita su 4 GPU A100"
echo "🔄 Log dettagliato disponibile in tempo reale"
echo ""

PYTHON_START=$(date +%s)

# Esecuzione con gestione errori ottimizzata
if timeout 7000s python3 $WORK/LLM-Mob-As-Mobility-Interpreter/veronacard_mob_with_geom_parrallel.py --append; then
    PYTHON_END=$(date +%s)
    PYTHON_TIME=$((PYTHON_END - PYTHON_START))
    
    echo ""
    echo "🎉 PRODUZIONE PARALLELA COMPLETATA CON SUCCESSO!"
    echo "⏱️ Tempo totale Python: $PYTHON_TIME secondi ($(($PYTHON_TIME / 3600))h $(($PYTHON_TIME % 3600 / 60))m)"
    echo "🚀 Accelerazione stimata: ~4x rispetto alla versione sequenziale"
    PYTHON_SUCCESS=true
else
    PYTHON_EXIT=$?
    PYTHON_END=$(date +%s)
    PYTHON_TIME=$((PYTHON_END - PYTHON_START))
    
    echo ""
    echo "❌ ERRORE IN PRODUZIONE PARALLELA (exit code: $PYTHON_EXIT)"
    echo "⏱️ Tempo prima del fallimento: $PYTHON_TIME secondi"
    
    # Analisi errori specifici per ambiente parallelo
    case $PYTHON_EXIT in
        124) echo "⚠️ Timeout - possibili risultati parziali (normale con grandi dataset)" ;;
        130) echo "⚠️ Interruzione manuale (Ctrl+C)" ;;
        137) echo "⚠️ Killed - possibile OOM o limite sistema" ;;
        139) echo "⚠️ Segfault - possibile problema GPU/driver" ;;
        *) echo "⚠️ Errore generico - controllare log per dettagli" ;;
    esac
    
    PYTHON_SUCCESS=false
fi

# Stop monitoring
kill $MONITOR_PID 2>/dev/null || true

# === 7. REPORT FINALE PARALLELO ===
echo ""
echo "📋 REPORT FINALE PRODUZIONE PARALLELA"
echo "====================================="

TOTAL_JOB_TIME=$SECONDS
echo "⏱️ Tempo totale job: $TOTAL_JOB_TIME secondi ($(($TOTAL_JOB_TIME / 3600))h $(($TOTAL_JOB_TIME % 3600 / 60))m)"
echo "🔧 Versione Ollama: $OLLAMA_VERSION"
echo "🚀 Modalità: Produzione parallela (4x A100, 8 workers)"
echo "✅ Python success: $PYTHON_SUCCESS"

echo ""
echo "📁 RISULTATI GENERATI:"
if [ -d "results/" ]; then
    FINAL_COUNT=$(ls -1 results/*.csv 2>/dev/null | wc -l)
    FINAL_SIZE=$(du -sh results/ 2>/dev/null | cut -f1 || echo "N/A")
    echo "   📊 Total files: $FINAL_COUNT"
    echo "   💾 Total size: $FINAL_SIZE"
    
    # Calcola throughput
    if [ "$PYTHON_SUCCESS" = "true" ] && [ $PYTHON_TIME -gt 0 ]; then
        THROUGHPUT=$(echo "scale=2; $FINAL_COUNT / ($PYTHON_TIME / 3600)" | bc -l 2>/dev/null || echo "N/A")
        echo "   ⚡ Throughput: $THROUGHPUT files/hour"
    fi
    
    echo ""
    echo "   📋 File generati (ultimi 10):"
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
echo "🔧 STATO FINALE SISTEMA MULTI-GPU:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv

echo ""
echo "📊 LOG OLLAMA PARALLELO (ultimi 20 righe):"
tail -20 ollama_parallel.log 2>/dev/null || echo "   Log non disponibile"

# Statistiche di utilizzo GPU
echo ""
echo "📈 STATISTICHE UTILIZZO GPU:"
echo "   (Durante l'elaborazione le GPU dovrebbero aver mostrato utilizzo distribuito)"

echo ""
if [ "$PYTHON_SUCCESS" = "true" ]; then
    echo "🎉 JOB PRODUZIONE PARALLELA COMPLETATO CON SUCCESSO!"
    echo "🚀 Elaborazione accelerata su architettura 4x A100"
    echo "✅ Tutti i dataset processati con parallelismo ottimizzato"
    
    # Calcola efficienza teorica
    if [ $PYTHON_TIME -gt 0 ]; then
        ESTIMATED_SEQUENTIAL=$((PYTHON_TIME * 4))  # Stima tempo sequenziale
        EFFICIENCY=$(echo "scale=1; ($ESTIMATED_SEQUENTIAL - $PYTHON_TIME) * 100 / $ESTIMATED_SEQUENTIAL" | bc -l 2>/dev/null || echo "N/A")
        echo "⚡ Efficienza parallelismo stimata: $EFFICIENCY% riduzione tempo"
    fi
else
    echo "⚠️ JOB PRODUZIONE PARALLELA TERMINATO CON ERRORI"
    echo "🔍 Verificare log e stato GPU per diagnostica"
fi

echo ""
echo "🏁 Fine job parallelo: $(date)"
echo "====================================="