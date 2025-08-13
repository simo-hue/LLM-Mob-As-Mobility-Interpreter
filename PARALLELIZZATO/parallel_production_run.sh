#!/bin/bash
#SBATCH --job-name=llm-mob-parallel
#SBATCH --account=IscrC_LLM-Mob
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --output=slurm-%j.out

echo "🚀 LLM-MOB PARALLEL PRODUCTION RUN - VERSIONE OTTIMIZZATA HPC"
echo "=============================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodo: $(hostname)"
echo "Data: $(date)"
echo "💡 MODALITÀ: Processamento parallelo completo"
echo "🔧 CPUs disponibili: $SLURM_CPUS_PER_TASK"
echo "💾 Memoria allocata: 128GB"
echo ""

# === 1. AMBIENTE ===
echo "📦 Caricamento moduli e ambiente..."
module purge
module load python/3.11.6--gcc--8.5.0
module load cuda/12.3
source $SLURM_SUBMIT_DIR/llm/bin/activate

echo "✔ Python: $(python3 --version)"
echo "✔ Virtual env: $VIRTUAL_ENV"
echo "✔ CUDA: $(nvcc --version | grep release)"

# Installa/verifica psutil per monitoring risorse
pip install -q psutil 2>/dev/null || true

# Debug GPU e sistema
echo ""
echo "🔍 INFO SISTEMA:"
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "CPU cores fisici: $(lscpu | grep '^CPU(s):' | awk '{print $2}')"
echo "Memoria totale: $(free -h | grep Mem | awk '{print $2}')"

# === 2. CONFIGURAZIONE OLLAMA ===
echo ""
echo "⚙️  Configurazione Ollama per parallelismo..."

OLLAMA_BIN="/leonardo/home/userexternal/smattiol/opt/bin/ollama"

# Verifica binario
if [ ! -f "$OLLAMA_BIN" ]; then
    echo "❌ ERRORE CRITICO: Ollama non trovato in $OLLAMA_BIN"
    echo "Contenuto directory opt:"
    ls -la /leonardo/home/userexternal/smattiol/opt/bin/
    exit 1
fi

OLLAMA_VERSION=$($OLLAMA_BIN --version 2>&1 | grep -o "0\.[0-9]\+\.[0-9]\+" || echo "unknown")
echo "✔ Versione Ollama: $OLLAMA_VERSION"

# Variabili ambiente ottimizzate per parallelismo
export CUDA_VISIBLE_DEVICES=0
export OLLAMA_DEBUG=0
export OLLAMA_HOST=127.0.0.1
export OLLAMA_ORIGINS="*"
export OLLAMA_MODELS="$HOME/.ollama/models"

# Configurazioni ottimizzate per workload parallelo
export OLLAMA_NUM_PARALLEL=8  # Aumentato per supportare thread multipli
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_FLASH_ATTENTION=0
export OLLAMA_KEEP_ALIVE="24h"
export OLLAMA_MAX_QUEUE=100  # Queue più grande per richieste parallele

# Pulizia variabili deprecate
unset OLLAMA_GPU_OVERHEAD OLLAMA_HOST_GPU OLLAMA_RUNNER_TIMEOUT
unset OLLAMA_LOAD_TIMEOUT OLLAMA_REQUEST_TIMEOUT OLLAMA_COMPLETION_TIMEOUT OLLAMA_CONTEXT_TIMEOUT

# === 3. CONFIGURAZIONE PARALLELISMO ===
echo ""
echo "🔧 Configurazione parallelismo ottimale..."

# Calcolo automatico workers ottimali
AVAILABLE_CORES=$SLURM_CPUS_PER_TASK
OPTIMAL_FILE_WORKERS=$((AVAILABLE_CORES / 8))  # 1 worker ogni 8 cores
OPTIMAL_FILE_WORKERS=$((OPTIMAL_FILE_WORKERS > 4 ? 4 : OPTIMAL_FILE_WORKERS))  # Max 4
OPTIMAL_FILE_WORKERS=$((OPTIMAL_FILE_WORKERS < 1 ? 1 : OPTIMAL_FILE_WORKERS))  # Min 1

OPTIMAL_LLM_THREADS=8  # Thread per richieste LLM per processo
OPTIMAL_BATCH_SIZE=50   # Utenti per batch

echo "📊 Configurazione calcolata:"
echo "   - File workers: $OPTIMAL_FILE_WORKERS (processi paralleli)"
echo "   - LLM threads: $OPTIMAL_LLM_THREADS (per processo)"
echo "   - Batch size: $OPTIMAL_BATCH_SIZE utenti"
echo "   - Max parallelismo totale: $((OPTIMAL_FILE_WORKERS * OPTIMAL_LLM_THREADS)) richieste LLM"

# === 4. AVVIO SERVER OLLAMA ===
echo ""
echo "🚀 Avvio server Ollama ottimizzato per parallelismo..."

OLLAMA_PORT=39003
echo $OLLAMA_PORT > $SLURM_SUBMIT_DIR/ollama_port.txt
echo "✔ Porta server: $OLLAMA_PORT"

# Cleanup processi precedenti
echo "🧹 Pulizia processi precedenti..."
pkill -f "ollama serve" 2>/dev/null || true
sleep 5

# Avvio server con log dedicato
OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT $OLLAMA_BIN serve > ollama_parallel.log 2>&1 &
SERVER_PID=$!

echo "✔ Server PID: $SERVER_PID"
echo "✔ Log file: ollama_parallel.log"

# Cleanup function avanzata
cleanup() {
    echo ""
    echo "🧹 CLEANUP PRODUZIONE PARALLELA..."
    echo "🕐 Tempo totale job: $SECONDS secondi ($(($SECONDS / 3600))h $(($SECONDS % 3600 / 60))m)"
    
    # Salva statistiche finali
    echo "📊 Stato GPU finale:" 
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv
    
    # Statistiche CPU
    echo "📊 Utilizzo CPU finale:"
    top -bn1 | head -5
    
    # Conta risultati
    if [ -d "results/" ]; then
        TOTAL_RESULTS=$(ls -1 results/*.csv 2>/dev/null | wc -l)
        echo "📄 Risultati generati: $TOTAL_RESULTS files"
        echo "💾 Dimensione risultati: $(du -sh results/ 2>/dev/null | cut -f1)"
        
        # Analisi risultati per file
        echo "📊 Dettaglio per file:"
        for f in results/*.csv; do
            if [ -f "$f" ]; then
                LINES=$(wc -l < "$f")
                SIZE=$(du -h "$f" | cut -f1)
                echo "   - $(basename $f): $LINES righe, $SIZE"
            fi
        done | head -20
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
    
    # Kill monitoring se ancora attivo
    kill $MONITOR_PID 2>/dev/null || true
    
    pkill -f "ollama" 2>/dev/null || true
    pkill -f "veronacard_parallel" 2>/dev/null || true
    echo "✔ Cleanup completato"
}
trap cleanup EXIT

# === 5. ATTESA E VERIFICA SERVER ===
echo ""
echo "⏳ Attesa avvio server (max 60s)..."

MAX_WAIT=30
WAIT_INTERVAL=2

for i in $(seq 1 $MAX_WAIT); do
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "❌ ERRORE: Server terminato prematuramente!"
        echo "--- LOG SERVER ---"
        cat ollama_parallel.log
        exit 1
    fi
    
    # Test connessione
    if curl -s --connect-timeout 3 --max-time 5 "http://127.0.0.1:$OLLAMA_PORT/api/tags" >/dev/null 2>&1; then
        echo "✅ Server operativo dopo $((i * WAIT_INTERVAL))s"
        break
    fi
    
    if [ $((i % 10)) -eq 0 ]; then
        echo "   ⏱️  Attesa... ($((i * WAIT_INTERVAL))s)"
    fi
    
    sleep $WAIT_INTERVAL
done

# Verifica finale
if ! curl -s --max-time 15 "http://127.0.0.1:$OLLAMA_PORT/api/tags" >/dev/null 2>&1; then
    echo "❌ ERRORE CRITICO: Server non risponde dopo $((MAX_WAIT * WAIT_INTERVAL))s"
    echo "--- LOG COMPLETO ---"
    cat ollama_parallel.log
    exit 1
fi

# === 6. VERIFICA E WARM-UP MODELLO ===
echo ""
echo "🔥 Preparazione modello per produzione parallela..."

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
            print(f\"  ✔ {model.get('name', 'unknown')}\")
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
    echo "❌ ERRORE: Modello $MODEL_NAME non trovato!"
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

# Warm-up modello con test request
echo "🔥 Warm-up modello con test request..."
curl -s -X POST "http://127.0.0.1:$OLLAMA_PORT/api/chat" \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$MODEL_NAME\", \"messages\": [{\"role\": \"user\", \"content\": \"test\"}], \"stream\": false}" \
    --max-time 30 > /dev/null 2>&1 && echo "✅ Modello caricato in memoria" || echo "⚠️ Warm-up fallito ma continuo"

# === 7. ESECUZIONE PRODUZIONE PARALLELA ===
echo ""
echo "🎯 AVVIO PRODUZIONE PARALLELA - PROCESSAMENTO COMPLETO"
echo "======================================================"

# Crea directory risultati e logs
mkdir -p results/
mkdir -p logs/
echo "📁 Directory risultati: $(pwd)/results/"
echo "📁 Directory logs: $(pwd)/logs/"

# Monitoring function avanzata
monitor_progress() {
    local last_count=0
    local stall_count=0
    
    while true; do
        sleep 180  # Ogni 3 minuti
        
        if [ -d "results/" ]; then
            CURRENT_FILES=$(ls -1 results/*.csv 2>/dev/null | wc -l)
            CURRENT_SIZE=$(du -sh results/ 2>/dev/null | cut -f1 || echo "0")
            
            # Calcola velocità
            if [ $last_count -gt 0 ]; then
                FILES_DELTA=$((CURRENT_FILES - last_count))
                if [ $FILES_DELTA -eq 0 ]; then
                    stall_count=$((stall_count + 1))
                    echo "⚠️  [$(date '+%H:%M:%S')] Nessun progresso da $((stall_count * 3)) minuti"
                else
                    stall_count=0
                    echo "📊 [$(date '+%H:%M:%S')] Progresso: $CURRENT_FILES files (+$FILES_DELTA), $CURRENT_SIZE"
                fi
            else
                echo "📊 [$(date '+%H:%M:%S')] Progresso: $CURRENT_FILES files, $CURRENT_SIZE"
            fi
            last_count=$CURRENT_FILES
            
            # Monitor risorse
            GPU_MEM=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
            GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits | head -1)
            CPU_LOAD=$(uptime | awk -F'load average:' '{print $2}')
            echo "🔧 [$(date '+%H:%M:%S')] GPU: ${GPU_MEM}MB, ${GPU_UTIL}% | CPU load:${CPU_LOAD}"
            
            # Monitor processi Python
            PYTHON_PROCS=$(pgrep -c -f "veronacard_parallel.py" || echo "0")
            if [ $PYTHON_PROCS -gt 0 ]; then
                echo "🐍 [$(date '+%H:%M:%S')] Processi Python attivi: $PYTHON_PROCS"
            fi
            
            # Alert se stallo prolungato
            if [ $stall_count -ge 5 ]; then
                echo "⚠️  ATTENZIONE: Possibile stallo - nessun progresso da 15 minuti"
            fi
        fi
        
        # Controlla se il processo principale è ancora attivo
        if ! pgrep -f "veronacard_parallel.py" >/dev/null; then
            echo "📊 [$(date '+%H:%M:%S')] Processo principale terminato"
            break
        fi
    done
}

# Avvio monitoring in background
monitor_progress &
MONITOR_PID=$!

echo "🚀 AVVIO SCRIPT PARALLELO..."
echo "🔧 Parametri di esecuzione:"
echo "   --parallel-files $OPTIMAL_FILE_WORKERS"
echo "   --parallel-llm $OPTIMAL_LLM_THREADS"
echo "   --batch-size $OPTIMAL_BATCH_SIZE"
echo "   --append (riprende da dove interrotto)"
echo ""

PYTHON_START=$(date +%s)

# Esecuzione con parametri ottimizzati per parallelismo
if timeout 82800 python3 PARALLELIZZATO/veronacard_parallel.py --max-users 1000 --batch-size 100 \
    --parallel-files $OPTIMAL_FILE_WORKERS \
    --parallel-llm $OPTIMAL_LLM_THREADS \
    --batch-size $OPTIMAL_BATCH_SIZE \
    --append 2>&1 | tee -a logs/parallel_execution.log; then
    
    PYTHON_END=$(date +%s)
    PYTHON_TIME=$((PYTHON_END - PYTHON_START))
    
    echo ""
    echo "🎉 PRODUZIONE PARALLELA COMPLETATA CON SUCCESSO!"
    echo "⏱️  Tempo totale Python: $PYTHON_TIME secondi ($(($PYTHON_TIME / 3600))h $(($PYTHON_TIME % 3600 / 60))m)"
    echo "🚀 Speedup stimato: ${OPTIMAL_FILE_WORKERS}x rispetto a versione seriale"
    PYTHON_SUCCESS=true
else
    PYTHON_EXIT=$?
    PYTHON_END=$(date +%s)
    PYTHON_TIME=$((PYTHON_END - PYTHON_START))
    
    echo ""
    echo "❌ ERRORE IN PRODUZIONE PARALLELA (exit code: $PYTHON_EXIT)"
    echo "⏱️  Tempo prima del fallimento: $PYTHON_TIME secondi"
    
    # Analisi tipo errore
    case $PYTHON_EXIT in
        124) echo "⚠️  Timeout dopo 23 ore - risultati parziali disponibili" ;;
        130) echo "⚠️  Interruzione manuale (Ctrl+C)" ;;
        137) echo "⚠️  Killed (OOM o sistema)" ;;
        *) echo "⚠️  Errore generico - controllare logs/parallel_execution.log" ;;
    esac
    
    PYTHON_SUCCESS=false
fi

# Stop monitoring
kill $MONITOR_PID 2>/dev/null || true

# === 8. REPORT FINALE DETTAGLIATO ===
echo ""
echo "📋 REPORT FINALE PRODUZIONE PARALLELA"
echo "====================================="

TOTAL_JOB_TIME=$SECONDS
echo "⏱️  Tempo totale job: $TOTAL_JOB_TIME secondi ($(($TOTAL_JOB_TIME / 3600))h $(($TOTAL_JOB_TIME % 3600 / 60))m)"
echo "🔧 Versione Ollama: $OLLAMA_VERSION"
echo "📊 Modalità: Produzione parallela completa"
echo "🔢 Configurazione parallelismo:"
echo "   - File workers utilizzati: $OPTIMAL_FILE_WORKERS"
echo "   - LLM threads per worker: $OPTIMAL_LLM_THREADS"
echo "   - Batch size: $OPTIMAL_BATCH_SIZE"
echo "✅ Python success: $PYTHON_SUCCESS"

echo ""
echo "📄 RISULTATI GENERATI:"
if [ -d "results/" ]; then
    FINAL_COUNT=$(ls -1 results/*.csv 2>/dev/null | wc -l)
    FINAL_SIZE=$(du -sh results/ 2>/dev/null | cut -f1 || echo "N/A")
    TOTAL_ROWS=0
    
    for f in results/*.csv; do
        if [ -f "$f" ]; then
            ROWS=$(wc -l < "$f")
            TOTAL_ROWS=$((TOTAL_ROWS + ROWS - 1))  # -1 per header
        fi
    done
    
    echo "   📊 Files totali: $FINAL_COUNT"
    echo "   📝 Righe totali processate: ~$TOTAL_ROWS"
    echo "   💾 Dimensione totale: $FINAL_SIZE"
    echo ""
    echo "   📋 Ultimi file generati:"
    ls -lah results/*.csv 2>/dev/null | tail -10 | while read line; do
        echo "     $line"
    done
    
    if [ $FINAL_COUNT -gt 10 ]; then
        echo "     ... (mostrati ultimi 10 di $FINAL_COUNT)"
    fi
    
    # Calcolo throughput
    if [ $PYTHON_TIME -gt 0 ] && [ $TOTAL_ROWS -gt 0 ]; then
        THROUGHPUT=$((TOTAL_ROWS * 3600 / PYTHON_TIME))
        echo ""
        echo "   ⚡ Throughput: ~$THROUGHPUT utenti/ora"
    fi
else
    echo "   ⚠️ Nessun risultato trovato"
fi

echo ""
echo "🔧 STATO FINALE SISTEMA:"
echo "GPU:"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv
echo ""
echo "CPU e Memoria:"
free -h
echo ""
uptime

echo ""
echo "📊 LOG OLLAMA (ultimi 30 righe):"
tail -30 ollama_parallel.log 2>/dev/null || echo "   Log non disponibile"

echo ""
echo "📊 LOG PYTHON (ultimi errori/warning):"
grep -E "(ERROR|WARNING|CRITICAL)" logs/parallel_execution.log 2>/dev/null | tail -20 || echo "   Nessun errore rilevante"

# Analisi performance finale
echo ""
echo "📈 ANALISI PERFORMANCE:"
if [ -f "logs/parallel_execution.log" ]; then
    # Estrai metriche di performance dal log
    SUCCESS_RATE=$(grep -o "success rate: [0-9.]*%" logs/parallel_execution.log | tail -1 | cut -d' ' -f3)
    AVG_TIME=$(grep -o "avg response time: [0-9.]*s" logs/parallel_execution.log | tail -1 | cut -d' ' -f4)
    
    if [ -n "$SUCCESS_RATE" ]; then
        echo "   📊 Success rate LLM: $SUCCESS_RATE"
    fi
    if [ -n "$AVG_TIME" ]; then
        echo "   ⏱️  Tempo medio risposta LLM: $AVG_TIME"
    fi
fi

echo ""
if [ "$PYTHON_SUCCESS" = "true" ]; then
    echo "🎉 JOB PRODUZIONE PARALLELA COMPLETATO CON SUCCESSO!"
    echo "✅ Tutti i dataset processati con ottimizzazione parallela"
    echo "🚀 Performance migliorate di ~${OPTIMAL_FILE_WORKERS}x"
else
    echo "⚠️ JOB PRODUZIONE PARALLELA TERMINATO CON ERRORI"
    echo "📋 Verificare logs/parallel_execution.log per dettagli"
    echo "💡 Suggerimento: rilanciare con --append per riprendere"
fi

echo ""
echo "🕐 Fine job: $(date)"
echo "======================================================"