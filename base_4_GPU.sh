#!/bin/bash
#SBATCH --job-name=r1_B
#SBATCH --account=IscrC_LLM-Mob
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_lprod
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --output=r1-%j.out

RES_DIR="$WORK/results/deepseek-r1_32b/base_version/"
mkdir -p "$RES_DIR"  # Crea la directory se non esiste

echo "🚀 VERONA CARD - r1 LONG"
echo "================================================"
echo "⚠️ ATTENZIONE: Questo script aspetterà INDEFINITAMENTE il caricamento"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodo: $(hostname)"
echo "Data: $(date)"
echo "Salvo risultati: $RES_DIR"
echo ""

# ============= SETUP AMBIENTE =============
echo "📦 Setup ambiente HPC..."
module purge
module load python/3.11.6--gcc--8.5.0
module load cuda/12.3
source $WORK/venv/bin/activate

echo "✅ Python: $(python3 --version)"
echo "✅ CUDA: $(nvcc --version | grep release)"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NVIDIA_VISIBLE_DEVICES=0,1,2,3

# Debug GPU iniziale
echo ""
echo "🔍 GPU DETECTION:"
nvidia-smi --query-gpu=index,name,memory.total,temperature.gpu --format=csv,noheader
echo ""

# ============= SETUP DIRECTORY TEMPORANEA =============
CUSTOM_TMP="$WORK/tmp_ollama_$SLURM_JOB_ID"
mkdir -p "$CUSTOM_TMP"
chmod 700 "$CUSTOM_TMP"

# Export variabili temporanee
export TMPDIR="$CUSTOM_TMP"
export TMP="$CUSTOM_TMP"
export TEMP="$CUSTOM_TMP"
export OLLAMA_TMPDIR="$CUSTOM_TMP"

echo "📁 Directory temporanea: $CUSTOM_TMP"
WORK_AVAILABLE=$(df "$WORK" | tail -1 | awk '{print $4}')
WORK_AVAILABLE_GB=$((WORK_AVAILABLE / 1024 / 1024))
echo "💾 Spazio disponibile: ${WORK_AVAILABLE_GB}GB"

if [ $WORK_AVAILABLE_GB -lt 30 ]; then
    echo "❌ ERRORE: Spazio insufficiente (${WORK_AVAILABLE_GB}GB < 30GB)"
    exit 1
fi

# ============= CONFIGURAZIONE OLLAMA =============
OLLAMA_BIN="/leonardo_work/IscrC_LLM-Mob/opt/bin/ollama"

if [ ! -f "$OLLAMA_BIN" ]; then
    echo "❌ ERRORE: Ollama non trovato in $OLLAMA_BIN"
    exit 1
fi

export OLLAMA_DEBUG=0
export OLLAMA_MODELS="$WORK/.ollama/models"
export OLLAMA_CACHE_DIR="$WORK/.ollama/cache"
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_KEEP_ALIVE="8h"
export OLLAMA_LLM_LIBRARY="cuda_v12"
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_MAX_QUEUE=10
export OLLAMA_CONCURRENT_REQUESTS=1

# 🔴 RIMOZIONE DI TUTTI I TIMEOUT OLLAMA
unset OLLAMA_LOAD_TIMEOUT
unset OLLAMA_REQUEST_TIMEOUT
unset OLLAMA_SERVER_TIMEOUT

# ============= CLEANUP PREVENTIVO =============
echo ""
echo "🧹 Cleanup preventivo..."
pkill -f ollama 2>/dev/null || true
sleep 20

# Cleanup vecchie directory temporanee
find "$WORK" -maxdepth 1 -name "tmp_ollama_*" -type d -user $(whoami) -mmin +120 -exec rm -rf {} + 2>/dev/null || true

# ============= DEFINIZIONE VARIABILI GLOBALI =============
SERVER_PID1=""
SERVER_PID2=""
SERVER_PID3=""
SERVER_PID4=""

# ============= FUNZIONE DI CLEANUP PER EXIT =============
cleanup() {
    echo ""
    echo "🧹 Cleanup finale..."
    
    # Kill processi Ollama
    for pid in $SERVER_PID1 $SERVER_PID2 $SERVER_PID3 $SERVER_PID4; do
        if [ -n "$pid" ] && kill -0 $pid 2>/dev/null; then
            echo "Stopping PID $pid..."
            kill -TERM $pid 2>/dev/null
        fi
    done
    
    sleep 10
    pkill -f ollama 2>/dev/null || true
    
    # Rimuovi directory temporanea
    if [ -n "$CUSTOM_TMP" ] && [ -d "$CUSTOM_TMP" ]; then
        echo "Removing $CUSTOM_TMP..."
        rm -rf "$CUSTOM_TMP"
    fi
    
    echo "✅ Cleanup completato"
}
trap cleanup EXIT

# ============= FUNZIONE DI AVVIO SENZA TIMEOUT =============
start_ollama_gpu() {
    local gpu_id=$1
    local port=$2
    local is_master=$3
    
    echo ""
    echo "🔧 Avvio GPU $gpu_id su porta $port..."
    
    # Crea cache directory dedicata
    local gpu_cache="$OLLAMA_CACHE_DIR/gpu${gpu_id}"
    mkdir -p "$gpu_cache"
    
    # 🔴 CRITICO: Nessun comando timeout, processo libero di vivere
    CUDA_VISIBLE_DEVICES=$gpu_id \
    OLLAMA_HOST=127.0.0.1:$port \
    OLLAMA_MAX_LOADED_MODELS=1 \
    OLLAMA_TMPDIR="$CUSTOM_TMP" \
    OLLAMA_CACHE_DIR="$gpu_cache" \
    $OLLAMA_BIN serve > ollama_gpu${gpu_id}.log 2>&1 &
    
    local pid=$!
    echo "✅ GPU $gpu_id PID: $pid (NO TIMEOUT)"
    
    # Salva PID globalmente
    eval "SERVER_PID$((gpu_id+1))=$pid"
    
    # Verifica che il processo sia vivo
    sleep 5
    if ! kill -0 $pid 2>/dev/null; then
        echo "❌ Processo GPU $gpu_id morto immediatamente!"
        tail -20 ollama_gpu${gpu_id}.log
        return 1
    fi
    
    # Se è la GPU master, aspetta il caricamento completo
    if [ "$is_master" = "true" ]; then
        echo "⏳ GPU $gpu_id è MASTER - attesa caricamento modello SENZA LIMITI..."
        
        local attempts=0
        while true; do
            ((attempts++))
            
            # Check processo ancora vivo
            if ! kill -0 $pid 2>/dev/null; then
                echo "❌ Processo GPU $gpu_id terminato inaspettatamente!"
                echo "📜 Ultimi log:"
                tail -30 ollama_gpu${gpu_id}.log
                return 1
            fi
            
            # Test API
            if curl -s --connect-timeout 5 "http://127.0.0.1:$port/api/tags" >/dev/null 2>&1; then
                echo "   🌐 API risponde, test modello..."
                
                # Test caricamento modello
                local test_response=$(curl -s -X POST \
                    --connect-timeout 10 \
                    --max-time 120 \
                    "http://127.0.0.1:$port/api/generate" \
                    -H "Content-Type: application/json" \
                    -d '{
                        "model":"deepseek-r1:32b",
                        "prompt":"Hi",
                        "stream":false,
                        "options":{"num_predict":1}
                    }' 2>&1)
                
                if echo "$test_response" | grep -q '"done":true'; then
                    echo "   ✅ GPU $gpu_id PRONTA dopo $attempts tentativi!"
                    return 0
                elif echo "$test_response" | grep -q "model.*not found"; then
                    echo "   ⚠️ Modello non trovato, potrebbe essere in download..."
                fi
            fi
            
            # Feedback periodico
            if [ $((attempts % 10)) -eq 0 ]; then
                echo "   ⏳ Tentativo $attempts - GPU $gpu_id ancora in caricamento..."
                echo "   📊 Memoria GPU:"
                nvidia-smi --id=$gpu_id --query-gpu=memory.used,memory.total --format=csv,noheader
                
                # Check log per progresso
                local progress=$(grep "model load progress" ollama_gpu${gpu_id}.log | tail -1)
                [ -n "$progress" ] && echo "   📈 $progress"
            fi
            
            sleep 30  # Check ogni 30 secondi
            
            # Safety check dopo 30 minuti
            if [ $attempts -gt 60 ]; then
                echo "   ⚠️ GPU $gpu_id impiega più di 30 minuti..."
                echo "   Continuo ad aspettare (Ctrl+C per interrompere)..."
            fi
        done
    fi
    
    return 0
}

# ============= AVVIO SEQUENZIALE CONTROLLATO =============
echo ""
echo "🚀 AVVIO SISTEMA OLLAMA"
echo "========================"

# 1. AVVIA GPU 0 COME MASTER (carica il modello)
if ! start_ollama_gpu 0 39001 true; then
    echo "❌ ERRORE CRITICO: GPU 0 fallita"
    exit 1
fi

echo ""
echo "✅ GPU 0 completamente operativa con modello caricato"
echo "⏳ Pausa 60s per stabilizzazione..."
sleep 60

# 2. AVVIA ALTRE GPU (che riuseranno il modello già in cache)
echo ""
echo "🚀 Avvio GPU secondarie..."

for gpu_id in 1 2 3; do
    port=$((39001 + gpu_id))
    start_ollama_gpu $gpu_id $port false
    sleep 30
done

echo "⏳ Attesa finale stabilizzazione sistema per DeepSeek R1 32B (120s)..."
sleep 120

# ============= VERIFICA FINALE =============
echo ""
echo "🔍 VERIFICA FINALE SISTEMA"
echo "==========================="

WORKING_GPUS=0
WORKING_PORTS=""

for i in 0 1 2 3; do
    port=$((39001 + i))
    
    echo -n "GPU $i (porta $port): "
    
    # Test completo
    if curl -s "http://127.0.0.1:$port/api/tags" >/dev/null 2>&1; then
        test_resp=$(curl -s -X POST \
            "http://127.0.0.1:$port/api/chat" \
            -H "Content-Type: application/json" \
            -d '{
                "model":"deepseek-r1:32b",
                "messages":[{"role":"user","content":"Say OK"}],
                "stream":false,
                "options":{"num_predict":2}
            }' 2>&1)
        
        if echo "$test_resp" | grep -q '"done":true'; then
            echo "✅ OPERATIVA"
            ((WORKING_GPUS++))
            [ -z "$WORKING_PORTS" ] && WORKING_PORTS="$port" || WORKING_PORTS="$WORKING_PORTS,$port"
        else
            echo "⚠️ API risponde ma modello non pronto"
        fi
    else
        echo "❌ NON RISPONDE"
    fi
done

echo ""
echo "📊 RISULTATO: $WORKING_GPUS/4 GPU operative"

if [ $WORKING_GPUS -eq 0 ]; then
    echo "❌ ERRORE: Nessuna GPU operativa!"
    for i in 0 1 2 3; do
        echo ""
        echo "=== Log GPU $i (ultime 30 righe) ==="
        tail -30 ollama_gpu${i}.log 2>/dev/null || echo "Log non disponibile"
    done
    exit 1
fi

# Salva porte funzionanti
echo "$WORKING_PORTS" > ollama_ports.txt
echo "✅ Porte salvate: $WORKING_PORTS"

# ============= MONITORING AVANZATO GPU =============
advanced_gpu_monitor() {
    echo "📊 Starting Advanced GPU Monitor (ogni 60s)"
    
    while true; do
        sleep 180
        
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "📊 GPU STATUS - $(date '+%Y-%m-%d %H:%M:%S')"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        
        # Mostra utilizzo GPU dettagliato
        nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,temperature.gpu,power.draw --format=csv,noheader,nounits | \
        while IFS=',' read -r idx name util_gpu util_mem mem_used mem_total temp power; do
            # Calcola percentuale memoria
            mem_percent=$(echo "scale=1; $mem_used * 100 / $mem_total" | bc -l 2>/dev/null || echo "0")
            
            # Colori per output (se supportato)
            if [ "$util_gpu" -gt 80 ]; then
                status="🔥 HIGH"
            elif [ "$util_gpu" -gt 50 ]; then
                status="✅ GOOD"
            elif [ "$util_gpu" -gt 10 ]; then
                status="⚡ LOW"
            else
                status="💤 IDLE"
            fi
            
            printf "GPU %s: %s\n" "$idx" "$status"
            printf "  Compute: %3d%% | Memory: %3d%% (%s/%s MB)\n" \
                   "$util_gpu" "$util_mem" "$mem_used" "$mem_total"
            printf "  Temp: %d°C | Power: %s W\n" "$temp" "$power"
            echo ""
        done
        
        # Mostra processi Ollama
        echo "🔄 Processi Ollama:"
        for i in 0 1 2 3; do
            eval "pid=\$SERVER_PID$((i+1))"
            port=$((39001 + i))
            
            if [ -n "$pid" ] && kill -0 $pid 2>/dev/null; then
                # CPU usage del processo
                cpu_usage=$(ps -p $pid -o %cpu= 2>/dev/null | tr -d ' ' || echo "0")
                # Memoria del processo
                mem_usage=$(ps -p $pid -o rss= 2>/dev/null | awk '{printf "%.1f", $1/1024/1024}' || echo "0")
                
                echo "  GPU $i (PID $pid): ✅ CPU: ${cpu_usage}% | RAM: ${mem_usage}GB | Port: $port"
                
                # Test veloce della porta
                if timeout 2s curl -s "http://127.0.0.1:$port/api/tags" >/dev/null 2>&1; then
                    echo "    └─ API: ✅ Responsive"
                else
                    echo "    └─ API: ⚠️ Slow/Unresponsive"
                fi
            else
                echo "  GPU $i: ❌ Process not running"
            fi
        done
        
        # Statistiche Python se in esecuzione
        if pgrep -f "veronacard_mob_versione_base_parrallel" >/dev/null; then
            echo ""
            echo "🐍 Python Processing:"
            
            # Conta file risultati
            if [ -d $RES_DIR ]; then
                result_count=$(ls -1 $RES_DIR/*.csv 2>/dev/null | wc -l)
                echo "  Output files: $result_count"
                
                # Ultimo file modificato
                latest=$(ls -t $RES_DIR*.csv 2>/dev/null | head -1)
                if [ -n "$latest" ]; then
                    size=$(du -h "$latest" | cut -f1)
                    echo "  Latest: $(basename $latest) ($size)"
                fi
            fi
            
            # Linee processate dal log
            if [ -f "r1_base_version_python_execution.log" ]; then
                processed=$(grep -c "Processing card" r1_base_version_python_execution.log 2>/dev/null || echo "0")
                errors=$(grep -c "ERROR\|Error" r1_base_version_python_execution.log 2>/dev/null || echo "0")
                echo "  Cards processed: $processed"
                echo "  Errors: $errors"
                echo "  Dir RESULTS: $RES_DIR"
            fi
        else
            echo ""
            echo "🐍 Python: Not running or completed"
        fi
        
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    done
}

# Avvia monitor avanzato in background
advanced_gpu_monitor &
ADV_MONITOR_PID=$!

# ============= ESECUZIONE PYTHON =============
cd /leonardo_work/IscrC_LLM-Mob/LLM-Mob-As-Mobility-Interpreter

echo ""
echo "🐍 AVVIO PYTHON"
echo "==============="
echo ""

if [ -f "data/verona/vc_site.csv" ]; then
    python3 -u veronacard_mob_versione_base_parrallel.py \
        --append 2>&1 | tee r1_base_version_python_execution.log
    PYTHON_EXIT=$?
else
    echo "❌ File non trovato!"
    PYTHON_EXIT=1
fi

# Stop monitors
kill $ADV_MONITOR_PID 2>/dev/null || true

echo ""
echo "============================================"
echo "📊 JOB COMPLETATO"
echo "Exit code Python: $PYTHON_EXIT"
echo "GPU utilizzate: $WORKING_GPUS"
echo "Tempo totale: $SECONDS secondi"
echo "============================================"

exit $PYTHON_EXIT