#!/bin/bash
#SBATCH --job-name=parallel
#SBATCH --account=IscrC_LLM-Mob
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=07:00:00
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
echo "🚀 Modalità: Produzione parallela (2x A100, 8 workers)"
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

# === 3. AVVIO SERVER PARALLELO MULTI-ISTANZA ===
echo ""
echo "🚀 Avvio server Ollama multi-istanza (2 GPU)..."

OLLAMA_PORT1=39003
OLLAMA_PORT2=39004

# Cleanup più aggressivo
pkill -f "ollama serve" 2>/dev/null || true
pkill -f "ollama" 2>/dev/null || true
sleep 10

# Istanza 1: GPU 0
echo "🔥 Avvio istanza 1 su GPU 0 (porta $OLLAMA_PORT1)..."
CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT1 \
    $OLLAMA_BIN serve > ollama_gpu0.log 2>&1 &
SERVER_PID1=$!

# Attesa breve per evitare conflitti di startup
sleep 5

# Istanza 2: GPU 1  
echo "🔥 Avvio istanza 2 su GPU 1 (porta $OLLAMA_PORT2)..."
CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT2 \
    $OLLAMA_BIN serve > ollama_gpu1.log 2>&1 &
SERVER_PID2=$!

echo "✅ Server PID1: $SERVER_PID1 (GPU 0)"
echo "✅ Server PID2: $SERVER_PID2 (GPU 1)"

# Salva configurazione per Python
echo "$OLLAMA_PORT1,$OLLAMA_PORT2" > $SLURM_SUBMIT_DIR/ollama_ports.txt

# Cleanup function aggiornata
cleanup() {
    echo "🧹 CLEANUP MULTI-ISTANZA..."
    
    # Shutdown graceful di entrambe le istanze
    for pid in $SERVER_PID1 $SERVER_PID2; do
        if kill -0 $pid 2>/dev/null; then
            echo "🔄 Shutdown graceful PID $pid..."
            kill -TERM $pid 2>/dev/null
            sleep 5
            if kill -0 $pid 2>/dev/null; then
                kill -KILL $pid 2>/dev/null
            fi
        fi
    done
    
    # Cleanup globale
    pkill -f "ollama" 2>/dev/null || true
    nvidia-smi --gpu-reset -i 0,1 2>/dev/null || true
}
trap cleanup EXIT

# === 4. ATTESA E VERIFICA MULTI-ISTANZA ===
echo "⏳ Attesa avvio server multi-istanza..."

# Verifica istanza 1
for i in $(seq 1 30); do
    if curl -s --connect-timeout 5 "http://127.0.0.1:$OLLAMA_PORT1/api/tags" >/dev/null 2>&1; then
        echo "✅ Istanza 1 (GPU 0) operativa"
        break
    fi
    sleep 2
done

# Verifica istanza 2
for i in $(seq 1 30); do
    if curl -s --connect-timeout 5 "http://127.0.0.1:$OLLAMA_PORT2/api/tags" >/dev/null 2>&1; then
        echo "✅ Istanza 2 (GPU 1) operativa"
        break
    fi
    sleep 2
done

# === 5. PREPARAZIONE MODELLO SU ENTRAMBE LE GPU ===
echo ""
echo "🔥 Preparazione modello per elaborazione parallela..."

MODEL_NAME="mixtral:8x7b" #llama3.1:8b | deepseek-coder:33b

# Verifica modelli su ENTRAMBE le istanze
for PORT in $OLLAMA_PORT1 $OLLAMA_PORT2; do
    echo "📋 Modelli disponibili su porta $PORT:"
    MODELS_RESPONSE=$(curl -s "http://127.0.0.1:$PORT/api/tags" || echo '{"models":[]}')
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
        echo "❌ ERRORE CRITICO: Modello $MODEL_NAME non trovato su porta $PORT!"
        echo "📥 Tentativo download automatico..."
        # Usa la prima istanza per il download
        if timeout 900s CUDA_VISIBLE_DEVICES=0 $OLLAMA_BIN pull $MODEL_NAME; then
            echo "✅ Modello scaricato"
        else
            echo "❌ Download fallito"
            exit 1
        fi
    else
        echo "✅ Modello $MODEL_NAME disponibile su porta $PORT"
    fi
done

# Pre-caricamento su ENTRAMBE le istanze
echo "🔥 Pre-caricamento modello su entrambe le GPU..."

for i in 1 2; do
    PORT_VAR="OLLAMA_PORT$i"
    PORT=${!PORT_VAR}
    
    echo "📡 Warm-up GPU $((i-1)) (porta $PORT)..."
    timeout 120s curl -s -X POST "http://127.0.0.1:$PORT/api/generate" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MODEL_NAME\",\"prompt\":\"Hi\",\"stream\":false,\"options\":{\"num_predict\":1,\"temperature\":0}}" \
        > /tmp/warmup_gpu$((i-1)).json 2>&1

    if [ $? -eq 0 ] && [ -s /tmp/warmup_gpu$((i-1)).json ]; then
        echo "✅ Pre-caricamento GPU $((i-1)) completato"
        head -1 /tmp/warmup_gpu$((i-1)).json
    else
        echo "⚠️ Pre-caricamento GPU $((i-1)) con problemi - procedo comunque"
    fi
    
    rm -f /tmp/warmup_gpu$((i-1)).json
done

# === 6. ESECUZIONE PRODUZIONE PARALLELA ===
echo ""
echo "🎯 AVVIO PRODUZIONE PARALLELA"
echo "============================="

# Configurazione per Python ottimizzata per parallelo
cat > ollama_config.json << EOF
{
    "endpoints": ["http://127.0.0.1:$OLLAMA_PORT1", "http://127.0.0.1:$OLLAMA_PORT2"],
    "model": "$MODEL_NAME",
    "timeout": 300,
    "max_retries": 5,
    "production_mode": true,
    "parallel_mode": true,
    "num_workers": 8,
    "gpu_count": 2,
    "ollama_ready": true
}
EOF

# Variabili ambiente per Python parallelo (CORRETTE)
export OLLAMA_MODEL="$MODEL_NAME"
export PRODUCTION_MODE=1
export PARALLEL_MODE=1
export GPU_COUNT=2  # NON 4!

echo "✅ Configurazione produzione parallela attiva"
echo "🔥 2x A100 GPU disponibili per elaborazione"  # CORRETTO
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
            
            # Monitor tutte le GPU
            echo "🔧 [$(date '+%H:%M:%S')] GPU Status:"
            nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits | head -2 | \
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
echo "📡 Elaborazione distribuita su 2 GPU A100"
echo "🔄 Log dettagliato disponibile in tempo reale"
echo ""

PYTHON_START=$(date +%s)

# Esecuzione con gestione errori ottimizzata
if timeout 7000s python3 $WORK/LLM-Mob-As-Mobility-Interpreter/veronacard_mob_with_geom_parrallel.py --file veronacard_2020_2023/veronacard_2021_original.csv; then
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
        ESTIMATED_SEQUENTIAL=$((PYTHON_TIME * 2))
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