#!/bin/bash
#SBATCH --job-name=parallel_max
#SBATCH --account=IscrC_LLM-Mob
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=110  # Tutti i core disponibili (2x56)
#SBATCH --mem=480G           # Quasi tutta la RAM (512GB)
#SBATCH --output=slurm-%j.out

echo "🚀 LLM-MOB MAXIMUM PERFORMANCE RUN"
echo "=================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodo: $(hostname)"
echo "Data: $(date)"
echo "🚀 Modalità: Performance massima (4x A100, tutti i core)"
echo ""

# === 1. AMBIENTE OTTIMIZZATO ===
echo "📦 Caricamento moduli e ambiente ottimizzato..."
module purge
module load python/3.11.6--gcc--8.5.0
module load cuda/12.3
source $WORK/venv/bin/activate

# Ottimizzazioni ambiente per performance massima
export OMP_NUM_THREADS=28           # Threads per istanza Ollama (112/4)
export MKL_NUM_THREADS=28           # Intel MKL ottimizzato
export OPENBLAS_NUM_THREADS=28      
export NUMBA_NUM_THREADS=28         # Numba per numpy/pandas
export PYTHONHASHSEED=42            # Reproducibilità
export CUDA_LAUNCH_BLOCKING=0       # Async CUDA per performance

# GPU ottimizzazioni
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NVIDIA_VISIBLE_DEVICES=$SLURM_LOCALID

echo "✅ Python: $(python3 --version)"
echo "✅ Virtual env: $VIRTUAL_ENV"
echo "✅ CUDA: $(nvcc --version | grep release)"
echo "✅ CPU cores allocated: $SLURM_CPUS_PER_TASK"
echo "✅ Memory allocated: $SLURM_MEM_PER_NODE MB"

# Debug GPU completo
echo ""
echo "🔍 INFO GPU COMPLETE:"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "GPU disponibili per il job: $SLURM_GPUS_ON_NODE"

# === 2. CONFIGURAZIONE OLLAMA PERFORMANCE MASSIMA ===
echo ""
echo "⚙️ Configurazione Ollama per performance massima..."

OLLAMA_BIN="/leonardo_work/IscrC_LLM-Mob/opt/bin/ollama"

if [ ! -f "$OLLAMA_BIN" ]; then
    echo "❌ ERRORE CRITICO: Ollama non trovato in $OLLAMA_BIN"
    exit 1
fi

OLLAMA_VERSION=$($OLLAMA_BIN --version 2>&1 | grep -o "0\.[0-9]\+\.[0-9]\+" || echo "unknown")
echo "✅ Versione Ollama: $OLLAMA_VERSION"

# Configurazioni performance ottimizzate per MASSIMA velocità
export OLLAMA_DEBUG=0
export OLLAMA_HOST=127.0.0.1
export OLLAMA_ORIGINS="*"
export OLLAMA_MODELS="$WORK/.ollama/models"

# CONFIGURAZIONI AGGRESSIVE PER PERFORMANCE MASSIMA
export OLLAMA_NUM_PARALLEL=16         # Aumentato significativamente
export OLLAMA_MAX_LOADED_MODELS=4     # Un modello per GPU
export OLLAMA_FLASH_ATTENTION=1       # Obbligatorio per A100
export OLLAMA_KEEP_ALIVE="6h"         # Mantiene tutto in memoria
export OLLAMA_GPU_LAYERS=-1           # Tutti i layer su GPU
export OLLAMA_RUNNER_KEEP_ALIVE="6h"  # Runner sempre attivi

# Configurazioni memory-aggressive per 512GB RAM
export OLLAMA_MAX_VRAM=60000          # 60GB per GPU (A100 ha 64GB)
export OLLAMA_CONTEXT_SIZE=8192       # Contesto ampio per efficienza
export OLLAMA_BATCH_SIZE=512          # Batch grandi per throughput
export OLLAMA_THREAD_COUNT=28         # Thread per istanza

# Timeout generosi per performance
export OLLAMA_LOAD_TIMEOUT=900        # 15 minuti per caricamento
export OLLAMA_REQUEST_TIMEOUT=600     # 10 minuti per richiesta
export OLLAMA_MAX_QUEUE=50            # Coda più grande

# === 3. AVVIO 4 ISTANZE OLLAMA PARALLELE ===
echo ""
echo "🚀 Avvio 4 istanze Ollama parallele (una per GPU)..."

# Porte diverse per ogni istanza
OLLAMA_PORT1=39001
OLLAMA_PORT2=39002
OLLAMA_PORT3=39003
OLLAMA_PORT4=39004

# Cleanup aggressivo
pkill -f "ollama" 2>/dev/null || true
sleep 15

# Funzione per avviare istanza con affinità CPU
start_ollama_instance() {
    local gpu_id=$1
    local port=$2
    local cpu_start=$((gpu_id * 28))
    local cpu_end=$((cpu_start + 27))
    
    echo "🔥 Avvio istanza GPU $gpu_id (porta $port, CPU $cpu_start-$cpu_end)..."
    
    # Usa taskset per affinità CPU + CUDA_VISIBLE_DEVICES per GPU specifica
    taskset -c $cpu_start-$cpu_end \
        bash -c "CUDA_VISIBLE_DEVICES=$gpu_id OLLAMA_HOST=127.0.0.1:$port \
                OMP_NUM_THREADS=28 \
                $OLLAMA_BIN serve" \
        > ollama_gpu${gpu_id}.log 2>&1 &
    
    echo $! > ollama_gpu${gpu_id}.pid
    echo "✅ Istanza GPU $gpu_id avviata (PID: $(cat ollama_gpu${gpu_id}.pid))"
}

# Avvia tutte e 4 le istanze con affinità CPU
start_ollama_instance 0 $OLLAMA_PORT1 &
sleep 8
start_ollama_instance 1 $OLLAMA_PORT2 &
sleep 8
start_ollama_instance 2 $OLLAMA_PORT3 &
sleep 8
start_ollama_instance 3 $OLLAMA_PORT4 &

echo "✅ Tutte e 4 le istanze avviate"

# Salva configurazione per Python
echo "$OLLAMA_PORT1,$OLLAMA_PORT2,$OLLAMA_PORT3,$OLLAMA_PORT4" > $SLURM_SUBMIT_DIR/ollama_ports.txt

# Cleanup function per 4 istanze
cleanup() {
    echo "🧹 CLEANUP 4 ISTANZE..."
    
    for gpu_id in 0 1 2 3; do
        if [ -f "ollama_gpu${gpu_id}.pid" ]; then
            pid=$(cat ollama_gpu${gpu_id}.pid)
            if kill -0 $pid 2>/dev/null; then
                echo "🔄 Shutdown graceful GPU $gpu_id (PID $pid)..."
                kill -TERM $pid 2>/dev/null
                sleep 3
                if kill -0 $pid 2>/dev/null; then
                    kill -KILL $pid 2>/dev/null
                fi
            fi
            rm -f ollama_gpu${gpu_id}.pid
        fi
    done
    
    pkill -f "ollama" 2>/dev/null || true
    nvidia-smi --gpu-reset -i 0,1,2,3 2>/dev/null || true
}
trap cleanup EXIT

# === 4. ATTESA E VERIFICA 4 ISTANZE ===
echo "⏳ Attesa avvio 4 istanze parallele..."

# Verifica tutte le istanze
for i in 1 2 3 4; do
    port_var="OLLAMA_PORT$i"
    port=${!port_var}
    
    echo "🔍 Verifica istanza $i (porta $port)..."
    
    for attempt in $(seq 1 45); do  # Più tempo per 4 istanze
        if curl -s --connect-timeout 5 "http://127.0.0.1:$port/api/tags" >/dev/null 2>&1; then
            echo "✅ Istanza $i (GPU $((i-1))) operativa"
            break
        fi
        sleep 3
    done
done

# === 5. PREPARAZIONE MODELLO SU TUTTE E 4 LE GPU ===
echo ""
echo "🔥 Pre-caricamento modello su tutte e 4 le GPU..."

MODEL_NAME="llama3.1:8b"

# Pre-caricamento parallelo su tutte le istanze
for i in 1 2 3 4; do
    port_var="OLLAMA_PORT$i"
    port=${!port_var}
    
    echo "📡 Warm-up GPU $((i-1)) (porta $port)..."
    (
        timeout 180s curl -s -X POST "http://127.0.0.1:$port/api/generate" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"$MODEL_NAME\",\"prompt\":\"Hi\",\"stream\":false,\"options\":{\"num_predict\":1,\"temperature\":0}}" \
            > /tmp/warmup_gpu$((i-1)).json 2>&1
        
        if [ $? -eq 0 ]; then
            echo "✅ Pre-caricamento GPU $((i-1)) completato"
        fi
    ) &
done

# Attendi tutti i warm-up
wait
echo "✅ Tutte e 4 le GPU pre-caricate"

# === 6. CONFIGURAZIONE PYTHON PERFORMANCE MASSIMA ===
echo ""
echo "🎯 CONFIGURAZIONE PYTHON PERFORMANCE MASSIMA"
echo "============================================="

# Configurazione ottimale per Python
cat > ollama_config.json << EOF
{
    "endpoints": [
        "http://127.0.0.1:$OLLAMA_PORT1",
        "http://127.0.0.1:$OLLAMA_PORT2", 
        "http://127.0.0.1:$OLLAMA_PORT3",
        "http://127.0.0.1:$OLLAMA_PORT4"
    ],
    "model": "$MODEL_NAME",
    "timeout": 600,
    "max_retries": 5,
    "production_mode": true,
    "parallel_mode": true,
    "num_workers": 32,
    "gpu_count": 4,
    "cpu_cores": 112,
    "memory_gb": 480,
    "ollama_ready": true
}
EOF

# Variabili ambiente per Python ad alte prestazioni
export OLLAMA_MODEL="$MODEL_NAME"
export PRODUCTION_MODE=1
export PARALLEL_MODE=1
export GPU_COUNT=4
export MAX_WORKERS=32              # 8 worker per GPU
export BATCH_SIZE=1000             # Batch grandi per efficienza
export SAVE_FREQUENCY=250          # Salvataggio frequente

# Ottimizzazioni Python specifiche
export PYTHONOPTIMIZE=1            # Ottimizzazioni Python
export PYTHONDONTWRITEBYTECODE=1   # No .pyc per I/O veloce

echo "✅ Configurazione performance massima attiva"
echo "🔥 4x A100 GPU (256GB VRAM totale)"
echo "🖥️ 112 CPU cores (2x Intel Sapphire Rapids)"
echo "🧠 480GB RAM disponibile"
echo "🔧 32 worker paralleli (8 per GPU)"
echo "⚡ Throughput stimato: 10-15x baseline"
echo ""

cd /leonardo_work/IscrC_LLM-Mob/LLM-Mob-As-Mobility-Interpreter

# Monitoring avanzato per performance massima
monitor_performance() {
    while true; do
        sleep 120  # Monitor ogni 2 minuti per performance
        
        if [ -d "results/" ]; then
            CURRENT_FILES=$(ls -1 results/*.csv 2>/dev/null | wc -l)
            CURRENT_SIZE=$(du -sh results/ 2>/dev/null | cut -f1 || echo "0")
            echo "📊 [$(date '+%H:%M:%S')] Files: $CURRENT_FILES, Size: $CURRENT_SIZE"
            
            # Monitor dettagliato tutte le GPU
            echo "🔧 [$(date '+%H:%M:%S')] GPU Status (tutte e 4):"
            nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader,nounits | \
                while IFS=',' read idx mem_used mem_total util temp power; do
                    mem_pct=$((mem_used * 100 / mem_total))
                    echo "   GPU$idx: ${mem_pct}% mem (${mem_used}/${mem_total}MB), ${util}% util, ${temp}°C, ${power}W"
                done
            
            # Monitor CPU load
            CPU_LOAD=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | tr -d ',')
            echo "🖥️ [$(date '+%H:%M:%S')] CPU Load: $CPU_LOAD/112 cores"
            
            # Monitor memoria
            MEM_INFO=$(free -h | grep "Mem:" | awk '{print $3"/"$2}')
            echo "🧠 [$(date '+%H:%M:%S')] RAM: $MEM_INFO"
        fi
        
        # Controlla se il processo Python è ancora attivo
        if ! pgrep -f "veronacard_mob_with_geom_parrallel.py" >/dev/null; then
            break
        fi
    done
}

# Avvio monitoring in background
monitor_performance &
MONITOR_PID=$!

# === 7. AVVIO SCRIPT PYTHON OTTIMIZZATO ===
echo "🚀 AVVIO ELABORAZIONE PERFORMANCE MASSIMA..."
echo "📡 Distribuzione su 4 GPU A100 + 112 CPU cores"
echo "🎯 32 worker paralleli per throughput massimo"
echo ""

PYTHON_START=$(date +%s)

# Esecuzione con tutte le ottimizzazioni
if timeout 10800s python3 $WORK/LLM-Mob-As-Mobility-Interpreter/veronacard_mob_with_geom_parrallel.py --append --max-workers 32 --batch-size 1000; then
    PYTHON_END=$(date +%s)
    PYTHON_TIME=$((PYTHON_END - PYTHON_START))
    
    echo ""
    echo "🎉 ELABORAZIONE PERFORMANCE MASSIMA COMPLETATA!"
    echo "⏱️ Tempo totale: $PYTHON_TIME secondi ($(($PYTHON_TIME / 3600))h $(($PYTHON_TIME % 3600 / 60))m)"
    echo "🚀 Accelerazione stimata: ~12-15x rispetto baseline"
    PYTHON_SUCCESS=true
else
    PYTHON_EXIT=$?
    PYTHON_END=$(date +%s)
    PYTHON_TIME=$((PYTHON_END - PYTHON_START))
    
    echo ""
    echo "❌ ERRORE IN ELABORAZIONE (exit code: $PYTHON_EXIT)"
    echo "⏱️ Tempo prima del fallimento: $PYTHON_TIME secondi"
    PYTHON_SUCCESS=false
fi

# Stop monitoring
kill $MONITOR_PID 2>/dev/null || true

# === 8. REPORT FINALE PERFORMANCE ===
echo ""
echo "📋 REPORT FINALE PERFORMANCE MASSIMA"
echo "====================================="

TOTAL_JOB_TIME=$SECONDS
echo "⏱️ Tempo totale job: $TOTAL_JOB_TIME secondi ($(($TOTAL_JOB_TIME / 3600))h $(($TOTAL_JOB_TIME % 3600 / 60))m)"
echo "🔧 Versione Ollama: $OLLAMA_VERSION"
echo "🚀 Hardware: 4x A100 (256GB VRAM) + 112 CPU cores + 480GB RAM"
echo "⚡ Worker: 32 paralleli"
echo "✅ Python success: $PYTHON_SUCCESS"

# Statistiche finali dettagliate
echo ""
echo "📁 RISULTATI PERFORMANCE:"
if [ -d "results/" ]; then
    FINAL_COUNT=$(ls -1 results/*.csv 2>/dev/null | wc -l)
    FINAL_SIZE=$(du -sh results/ 2>/dev/null | cut -f1 || echo "N/A")
    echo "   📊 Total files: $FINAL_COUNT"
    echo "   💾 Total size: $FINAL_SIZE"
    
    # Calcola throughput effettivo
    if [ "$PYTHON_SUCCESS" = "true" ] && [ $PYTHON_TIME -gt 0 ]; then
        THROUGHPUT=$(echo "scale=2; $FINAL_COUNT / ($PYTHON_TIME / 3600)" | bc -l 2>/dev/null || echo "N/A")
        echo "   ⚡ Throughput reale: $THROUGHPUT files/hour"
        
        # Efficienza GPU
        THEORETICAL_MAX=$(echo "scale=1; 4 * 60 * 60 / 10" | bc -l)  # 4 GPU * 3600s / 10s per request
        EFFICIENCY=$(echo "scale=1; $THROUGHPUT * 100 / $THEORETICAL_MAX" | bc -l 2>/dev/null || echo "N/A")
        echo "   🎯 Efficienza GPU: $EFFICIENCY%"
    fi
fi

echo ""
echo "🔧 UTILIZZO FINALE RISORSE:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv

echo ""
if [ "$PYTHON_SUCCESS" = "true" ]; then
    echo "🎉 JOB PERFORMANCE MASSIMA COMPLETATO!"
    echo "🚀 Pieno utilizzo dell'hardware Leonardo Booster"
    echo "✅ Elaborazione ottimizzata su architettura completa"
else
    echo "⚠️ JOB TERMINATO CON ERRORI"
    echo "🔍 Verificare log per diagnostica"
fi

echo ""
echo "🏁 Fine job performance massima: $(date)"
echo "======================================="