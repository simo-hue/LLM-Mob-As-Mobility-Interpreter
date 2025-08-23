#!/bin/bash
#SBATCH --job-name=verona-opt
#SBATCH --account=IscrC_LLM-Mob
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --output=slurm-%j.out

echo "🚀 VERONA CARD HPC OTTIMIZZATO - VERSIONE STABILE"
echo "================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodo: $(hostname)"
echo "Data: $(date)"
echo "🎯 Modalità: Produzione stabile (4x A100, anti-contention)"
echo ""

# === AMBIENTE E MODULI ===
echo "📦 Setup ambiente HPC..."
module purge
module load python/3.11.6--gcc--8.5.0
module load cuda/12.3
source $WORK/venv/bin/activate

echo "✅ Python: $(python3 --version)"
echo "✅ CUDA: $(nvcc --version | grep release)"

# Configura tutte e 4 le GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NVIDIA_VISIBLE_DEVICES=0,1,2,3

# Debug GPU iniziale
echo ""
echo "🔍 GPU DETECTION:"
nvidia-smi --query-gpu=index,name,memory.total,temperature.gpu --format=csv,noheader
echo "SLURM GPUs: $SLURM_GPUS_ON_NODE"

echo ""
echo "💾 LEONARDO BOOSTER: Setup spazio temporaneo ottimizzato..."

# Su Leonardo Booster, /tmp è sempre limitato a ~10GB
TMP_AVAILABLE=$(df /tmp | tail -1 | awk '{print $4}')
TMP_AVAILABLE_GB=$((TMP_AVAILABLE / 1024 / 1024))
echo "📊 Spazio /tmp Leonardo Booster: ${TMP_AVAILABLE_GB}GB disponibili su ~10GB totali"

# SEMPRE usa directory custom su $WORK per Leonardo Booster
# Mixtral:8x7b richiede ~26GB di spazio temporaneo
CUSTOM_TMP="$WORK/tmp_ollama_$SLURM_JOB_ID"
mkdir -p "$CUSTOM_TMP"
chmod 700 "$CUSTOM_TMP"  # Sicurezza aggiuntiva

# Override TUTTE le variabili temporanee
export TMPDIR="$CUSTOM_TMP"
export TMP="$CUSTOM_TMP" 
export TEMP="$CUSTOM_TMP"
export OLLAMA_TMPDIR="$CUSTOM_TMP"

# Verifica spazio $WORK
WORK_AVAILABLE=$(df "$WORK" | tail -1 | awk '{print $4}')
WORK_AVAILABLE_GB=$((WORK_AVAILABLE / 1024 / 1024))

echo "✅ Configurazione Leonardo Booster:"
echo "   📁 /tmp originale: ${TMP_AVAILABLE_GB}GB (TROPPO PICCOLO per Mixtral)"
echo "   📁 Directory custom: $CUSTOM_TMP"
echo "   💾 Spazio \$WORK disponibile: ${WORK_AVAILABLE_GB}GB"
echo "   🎯 Mixtral richiede ~26GB temporanei - OK con \$WORK"

# Verifica che abbiamo abbastanza spazio su $WORK
if [ $WORK_AVAILABLE_GB -lt 30 ]; then
    echo "❌ ERRORE: Spazio insufficiente anche su \$WORK (${WORK_AVAILABLE_GB}GB)"
    echo "   Mixtral:8x7b richiede almeno 30GB di spazio temporaneo"
    exit 1
fi

# === CONFIGURAZIONI OLLAMA OTTIMIZZATE PER LEONARDO BOOSTER ===
export OLLAMA_DEBUG=0
export OLLAMA_ORIGINS="*"
export OLLAMA_MODELS="$WORK/.ollama/models"
export OLLAMA_CACHE_DIR="$WORK/.ollama/cache"

# Configurazioni anti-contention specifiche per A100
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KEEP_ALIVE="1h"
export OLLAMA_LOAD_TIMEOUT=1200          # 20 minuti per Leonardo Booster
export OLLAMA_REQUEST_TIMEOUT=300
export OLLAMA_MAX_QUEUE=2
export OLLAMA_MAX_VRAM_USAGE=0.75
export OLLAMA_MAX_CONCURRENT_DOWNLOADS=1
export OLLAMA_RUNNER_CACHE_SIZE="3GB"

# Assicurati che le directory esistano
mkdir -p "$OLLAMA_MODELS"
mkdir -p "$OLLAMA_CACHE_DIR"

echo "📋 Variabili temporanee configurate:"
echo "   TMPDIR=$TMPDIR"
echo "   OLLAMA_TMPDIR=$OLLAMA_TMPDIR" 
echo "   OLLAMA_CACHE_DIR=$OLLAMA_CACHE_DIR"

# === CLEANUP SPECIFICO LEONARDO BOOSTER ===
leonardo_cleanup() {
    echo "🧹 CLEANUP LEONARDO BOOSTER..."
    
    # Graceful shutdown Ollama
    for pid in $SERVER_PID1 $SERVER_PID2 $SERVER_PID3 $SERVER_PID4; do
        if [ -n "$pid" ] && kill -0 $pid 2>/dev/null; then
            echo "🔄 Shutdown graceful PID $pid..."
            kill -TERM $pid 2>/dev/null
        fi
    done
    
    sleep 30
    pkill -f ollama 2>/dev/null || true
    
    # CLEANUP DIRECTORY TEMPORANEA CUSTOM
    if [ -n "$CUSTOM_TMP" ] && [ -d "$CUSTOM_TMP" ]; then
        echo "🗑️ Cleanup directory temporanea Leonardo: $CUSTOM_TMP"
        TEMP_SIZE=$(du -sh "$CUSTOM_TMP" 2>/dev/null | cut -f1 || echo "N/A")
        echo "   📊 Dimensione da rimuovere: $TEMP_SIZE"
        rm -rf "$CUSTOM_TMP" 2>/dev/null || true
        echo "   ✅ Spazio liberato su \$WORK"
    fi
    
    # Cleanup anche vecchie directory temporanee
    echo "🧹 Cleanup vecchie directory temporanee..."
    find "$WORK" -maxdepth 1 -name "tmp_ollama_*" -type d -user $(whoami) -mmin +120 -exec rm -rf {} + 2>/dev/null || true
    
    echo "✅ Cleanup Leonardo Booster completato"
}

trap leonardo_cleanup EXIT

# === CLEANUP PREVENTIVO ===
echo ""
echo "🧹 Cleanup preventivo Leonardo Booster..."
pkill -f ollama 2>/dev/null || true

# Cleanup vecchie directory temporanee su $WORK
find "$WORK" -maxdepth 1 -name "tmp_ollama_*" -type d -user $(whoami) -mmin +60 -exec rm -rf {} + 2>/dev/null || true

# Cleanup residui /tmp (anche se piccolo)
find /tmp -maxdepth 1 -name "ollama*" -type d -user $(whoami) -mmin +30 -exec rm -rf {} + 2>/dev/null || true

sleep 20

echo "🔍 Stato spazio post-cleanup:"
echo "💾 /tmp: $(df -h /tmp | tail -1 | awk '{print $4}') disponibili"
echo "💾 \$WORK: $(df -h $WORK | tail -1 | awk '{print $4}') disponibili"
echo "📂 Directory temporanea: $(ls -lah $CUSTOM_TMP 2>/dev/null || echo 'Creata ma vuota')"


# === CONFIGURAZIONE OLLAMA STABILIZZATA ===
OLLAMA_BIN="/leonardo_work/IscrC_LLM-Mob/opt/bin/ollama"

# Verifica binario
if [ ! -f "$OLLAMA_BIN" ]; then
    echo "❌ ERRORE: Ollama non trovato in $OLLAMA_BIN"
    exit 1
fi

# Variabili ottimizzate per stabilità massima
export OLLAMA_DEBUG=0
export OLLAMA_ORIGINS="*"
export OLLAMA_MODELS="$WORK/.ollama/models"

# CONFIGURAZIONI ANTI-CONTENTION
export OLLAMA_NUM_PARALLEL=1              # UNA richiesta per volta per GPU
export OLLAMA_MAX_LOADED_MODELS=1         # Un modello per volta
export OLLAMA_FLASH_ATTENTION=1        
export OLLAMA_KEEP_ALIVE="2h"             # Riduci per liberare memoria
export OLLAMA_LOAD_TIMEOUT=600            # 10 minuti per caricamento
export OLLAMA_REQUEST_TIMEOUT=300         # 5 minuti per richiesta
export OLLAMA_MAX_QUEUE=2                 # Coda ridotta
export OLLAMA_MAX_VRAM_USAGE=0.8          # Limita VRAM al 80%

# === CLEANUP PREVENTIVO AGGRESSIVO ===
echo ""
echo "🧹 Cleanup preventivo..."
pkill -f ollama 2>/dev/null || true
sleep 20  # Attesa più lunga per cleanup completo

echo "🔍 Controllo stato GPU post-cleanup..."
nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu --format=csv,noheader
nvidia-smi pmon -c 1 2>/dev/null || echo "Monitoring non disponibile"

# === AVVIO MULTI-ISTANZA SEQUENZIALE ===
echo ""
echo "🚀 Avvio server Ollama con anti-contention..."

# Configurazione porte
OLLAMA_PORT1=39001
OLLAMA_PORT2=39002  
OLLAMA_PORT3=39003
OLLAMA_PORT4=39004

# Funzione di avvio ottimizzata per Leonardo Booster
start_ollama_instance() {
    local gpu_id=$1
    local port=$2
    local log_file="ollama_gpu${gpu_id}.log"
    
    echo "🔥 Avvio istanza GPU $gpu_id su porta $port (Leonardo Booster)..."
    
    # Cleanup specifico per questa GPU
    pkill -f "OLLAMA_HOST=127.0.0.1:$port" 2>/dev/null || true
    sleep 2
    
    # Verifica spazio $WORK (non /tmp che è troppo piccolo)
    WORK_SPACE=$(df "$WORK" | tail -1 | awk '{print $4}')
    WORK_SPACE_GB=$((WORK_SPACE / 1024 / 1024))
    
    if [ $WORK_SPACE_GB -lt 25 ]; then
        echo "❌ GPU $gpu_id: spazio \$WORK insufficiente (${WORK_SPACE_GB}GB < 25GB)"
        return 1
    fi
    
    echo "✅ GPU $gpu_id: spazio \$WORK OK (${WORK_SPACE_GB}GB disponibili)"
    
    # Crea directory cache dedicata per questa GPU
    GPU_CACHE_DIR="$OLLAMA_CACHE_DIR/gpu${gpu_id}"
    mkdir -p "$GPU_CACHE_DIR"
    
    # Avvio con tutte le variabili personalizzate
    CUDA_VISIBLE_DEVICES=$gpu_id \
    OLLAMA_HOST=127.0.0.1:$port \
    OLLAMA_MAX_LOADED_MODELS=1 \
    OLLAMA_TMPDIR="$CUSTOM_TMP" \
    OLLAMA_CACHE_DIR="$GPU_CACHE_DIR" \
    TMPDIR="$CUSTOM_TMP" \
    TMP="$CUSTOM_TMP" \
    TEMP="$CUSTOM_TMP" \
    $OLLAMA_BIN serve > $log_file 2>&1 &
    
    local pid=$!
    echo "✅ GPU $gpu_id PID: $pid"
    echo "   📁 Cache dedicata: $GPU_CACHE_DIR"
    echo "   📁 Temp directory: $CUSTOM_TMP"
    
    return $pid
}

# Avvio sequenziale con attese lunghe per evitare race conditions
echo "📡 Avvio istanza 1/4..."
start_ollama_instance 0 $OLLAMA_PORT1; SERVER_PID1=$?
sleep 40

echo "📡 Avvio istanza 2/4..."
start_ollama_instance 1 $OLLAMA_PORT2; SERVER_PID2=$?
sleep 40

echo "📡 Avvio istanza 3/4..."
start_ollama_instance 2 $OLLAMA_PORT3; SERVER_PID3=$?
sleep 40

echo "📡 Avvio istanza 4/4..."  
start_ollama_instance 3 $OLLAMA_PORT4; SERVER_PID4=$?
sleep 40

echo "✅ Tutte le istanze avviate:"
echo "   GPU 0: PID $SERVER_PID1 (porta $OLLAMA_PORT1)"
echo "   GPU 1: PID $SERVER_PID2 (porta $OLLAMA_PORT2)" 
echo "   GPU 2: PID $SERVER_PID3 (porta $OLLAMA_PORT3)"
echo "   GPU 3: PID $SERVER_PID4 (porta $OLLAMA_PORT4)"

# Salva configurazione per Python
echo "$OLLAMA_PORT1,$OLLAMA_PORT2,$OLLAMA_PORT3,$OLLAMA_PORT4" > ollama_ports.txt

# Funzione di cleanup migliorata
cleanup() {
    echo "🧹 CLEANUP STABILIZZATO..."
    
    # Graceful shutdown con timeout
    for pid in $SERVER_PID1 $SERVER_PID2 $SERVER_PID3 $SERVER_PID4; do
        if [ -n "$pid" ] && kill -0 $pid 2>/dev/null; then
            echo "🔄 Shutdown graceful PID $pid..."
            kill -TERM $pid 2>/dev/null
        fi
    done
    
    # Attesa shutdown più lunga
    echo "⏳ Attesa shutdown (30s)..."
    sleep 30
    
    # Force kill se necessario
    echo "🔨 Force cleanup..."
    pkill -f ollama 2>/dev/null || true
    
    echo "✅ Cleanup completato"
}
trap cleanup EXIT

# === HEALTH CHECK APPROFONDITO ===
echo ""
echo "⏳ Health check approfondito multi-istanza..."

check_instance_health() {
    local port=$1
    local gpu_id=$2
    local max_attempts=15  # Ridotto ma con timeout maggiori
    
    echo "🔍 Check approfondito GPU $gpu_id (porta $port)..."
    
    for i in $(seq 1 $max_attempts); do
        # Test 1: endpoint tags con timeout lungo
        if ! timeout 20s curl -s "http://127.0.0.1:$port/api/tags" >/dev/null 2>&1; then
            echo "   Tentativo $i/$max_attempts: endpoint non risponde"
            sleep 8
            continue
        fi
        
        # Test 2: mini-inference per verificare funzionalità reale
        local test_response=$(timeout 45s curl -s -X POST "http://127.0.0.1:$port/api/generate" \
            -H "Content-Type: application/json" \
            -d '{"model":"mixtral:8x7b","prompt":"Test","stream":false,"options":{"num_predict":1,"temperature":0}}' 2>/dev/null)
        
        if echo "$test_response" | grep -q '"done":true' && echo "$test_response" | grep -q '"response"'; then
            echo "✅ GPU $gpu_id completamente operativa"
            return 0
        fi
        
        echo "   Tentativo $i/$max_attempts: inference test fallito"
        
        if [ $i -eq $max_attempts ]; then
            echo "❌ GPU $gpu_id FALLITA dopo test approfonditi"
            echo "🔍 Log GPU $gpu_id (ultime 5 righe):"
            tail -5 ollama_gpu${gpu_id}.log 2>/dev/null || echo "Log non disponibile"
            return 1
        fi
        
        sleep 10
    done
}

# Health check di tutte le istanze
HEALTHY_COUNT=0
check_instance_health $OLLAMA_PORT1 0 && ((HEALTHY_COUNT++))
check_instance_health $OLLAMA_PORT2 1 && ((HEALTHY_COUNT++))
check_instance_health $OLLAMA_PORT3 2 && ((HEALTHY_COUNT++))
check_instance_health $OLLAMA_PORT4 3 && ((HEALTHY_COUNT++))

echo "📊 Health check: $HEALTHY_COUNT/4 istanze operative"

if [ $HEALTHY_COUNT -lt 2 ]; then
    echo "❌ ERRORE CRITICO: Troppe poche istanze funzionanti ($HEALTHY_COUNT/4)"
    echo "🔍 Debug completo logs:"
    for i in 0 1 2 3; do
        echo "--- GPU $i log (ultime 10 righe) ---"
        tail -10 ollama_gpu${i}.log 2>/dev/null || echo "Log non disponibile"
        echo ""
    done
    
    # Tentativo recovery
    echo "🔄 Tentativo recovery rapido..."
    cleanup
    sleep 30
    exit 1
fi

# === PREPARAZIONE MODELLO SEQUENZIALE ===
echo ""
echo "📥 Preparazione modello con prevenzione contention..."

MODEL_NAME="mixtral:8x7b"

# Verifica modello disponibile
echo "🔍 Verifica modello $MODEL_NAME..."
MODELS_RESPONSE=$(timeout 45s curl -s "http://127.0.0.1:$OLLAMA_PORT1/api/tags" || echo '{"models":[]}')

MODEL_EXISTS=$(echo "$MODELS_RESPONSE" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    models = [m.get('name', '') for m in data.get('models', [])]
    print('true' if '$MODEL_NAME' in models else 'false')
except:
    print('false')
" 2>/dev/null)

if [ "$MODEL_EXISTS" != "true" ]; then
    echo "📥 Download modello $MODEL_NAME..."
    if timeout 2400s CUDA_VISIBLE_DEVICES=0 $OLLAMA_BIN pull $MODEL_NAME; then
        echo "✅ Modello scaricato"
    else
        echo "❌ Download fallito - abort"
        exit 1
    fi
else
    echo "✅ Modello già presente"
fi

# Pre-caricamento SEQUENZIALE per evitare contention
echo "🔥 Pre-caricamento sequenziale anti-contention..."

preload_gpu_sequential() {
    local port=$1
    local gpu_id=$2
    
    echo "⚡ Warm-up GPU $gpu_id (sequenziale)..."
    timeout 180s curl -s -X POST "http://127.0.0.1:$port/api/generate" \
        -H "Content-Type: application/json" \
        -d '{"model":"'$MODEL_NAME'","prompt":"warmup","stream":false,"options":{"num_predict":1,"temperature":0}}' \
        >/dev/null 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✅ GPU $gpu_id pronta"
        return 0
    else
        echo "⚠️ GPU $gpu_id warm-up parziale"
        return 1
    fi
}

# Preload sequenziale con pausa anti-contention
READY_GPUS=0
for i in 0 1 2 3; do
    eval "port=\$OLLAMA_PORT$((i+1))"
    echo "🎯 Warm-up GPU $i..."
    if preload_gpu_sequential $port $i; then
        ((READY_GPUS++))
    fi
    
    # Pausa critica per evitare contention memoria GPU
    if [ $i -lt 3 ]; then
        echo "⏸️ Pausa anti-contention (15s)..."
        sleep 15
    fi
done

echo "✅ GPU pronte per produzione: $READY_GPUS/4"
HEALTHY_COUNT=$READY_GPUS

# === VERIFICA FILE INPUT ===
echo ""
echo "📂 Verifica file di input..."

cd /leonardo_work/IscrC_LLM-Mob/LLM-Mob-As-Mobility-Interpreter

TARGET_FILE="data/verona/veronacard_2020_2023/veronacard_2022_original.csv"
if [ ! -f "$TARGET_FILE" ]; then
    echo "❌ ERRORE CRITICO: File non trovato: $TARGET_FILE"
    echo "📁 Contenuto directory corrente:"
    ls -la
    echo "📁 Contenuto veronacard_2020_2023/:"
    ls -la veronacard_2020_2023/ 2>/dev/null || echo "Directory non trovata"
    exit 1
fi

FILE_SIZE=$(du -h "$TARGET_FILE" | cut -f1)
LINE_COUNT=$(wc -l < "$TARGET_FILE" 2>/dev/null || echo "N/A")
echo "✅ File input verificato: $TARGET_FILE"
echo "   📊 Dimensione: $FILE_SIZE"
echo "   📊 Righe: $LINE_COUNT"

# Verifica dipendenze Python
echo ""
echo "🐍 Verifica ambiente Python..."
python3 -c "
import sys
print('Python:', sys.executable)
try:
    import pandas, requests, sklearn, numpy
    print('✅ Dipendenze principali OK')
except ImportError as e:
    print('❌ Dipendenza mancante:', e)
    sys.exit(1)
"

if [ $? -ne 0 ]; then
    echo "❌ Ambiente Python non valido"
    exit 1
fi

# Crea directory risultati
mkdir -p results/
echo "📂 Directory results: $(ls -ld results/)"

# === DEBUG PRE-ESECUZIONE ===
echo ""
echo "🔍 DEBUG: Stato sistema pre-esecuzione"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Uptime job: $SECONDS secondi"
echo "Memoria disponibile: $(free -h | grep '^Mem' | awk '{print $7}')"
echo "Spazio disco: $(df -h . | tail -1 | awk '{print $4}')"

# Verifica processi Ollama
echo "🔍 Processi Ollama attivi:"
for pid in $SERVER_PID1 $SERVER_PID2 $SERVER_PID3 $SERVER_PID4; do
    if [ -n "$pid" ] && kill -0 $pid 2>/dev/null; then
        echo "✅ PID $pid attivo"
    else
        echo "❌ PID $pid terminato"
    fi
done

# Test connettività finale
echo "🌐 Test connettività finale:"
WORKING_PORTS=0
for port in $OLLAMA_PORT1 $OLLAMA_PORT2 $OLLAMA_PORT3 $OLLAMA_PORT4; do
    if timeout 10s curl -s "http://127.0.0.1:$port/api/tags" >/dev/null 2>&1; then
        echo "✅ Porta $port operativa"
        ((WORKING_PORTS++))
    else
        echo "❌ Porta $port non risponde"
    fi
done

echo "📊 Porte operative: $WORKING_PORTS/4"

if [ $WORKING_PORTS -lt 2 ]; then
    echo "❌ Troppo poche porte operative - abort"
    exit 1
fi

# === MONITORING OTTIMIZZATO ===
monitor_system() {
    while true; do
        sleep 300  # Ogni 5 minuti
        
        # Statistiche essenziali
        if [ -d "results/" ]; then
            TOTAL_FILES=$(ls -1 results/*.csv 2>/dev/null | wc -l) 
            LATEST_SIZE=$(du -sh results/ 2>/dev/null | cut -f1 || echo "0B")
            echo "📊 [$(date '+%H:%M:%S')] Files: $TOTAL_FILES, Size: $LATEST_SIZE"
        fi
        
        # Check processo Python
        if ! pgrep -f "veronacard_mob_with_geom" >/dev/null; then
            echo "ℹ️ [$(date '+%H:%M:%S')] Processo Python terminato"
            break
        fi
        
        # GPU health compatto (solo temperature critiche)
        HOT_GPUS=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | awk '$1 > 85 {print NR-1}')
        if [ -n "$HOT_GPUS" ]; then
            echo "🌡️ [$(date '+%H:%M:%S')] GPU calde: $HOT_GPUS"
        fi
    done
}

# Controlla se i processi Ollama sono in esecuzione
ps aux | grep ollama

# Verifica che le istanze rispondano
curl -s http://127.0.0.1:39001/api/tags
curl -s http://127.0.0.1:39002/api/tags
curl -s http://127.0.0.1:39003/api/tags
curl -s http://127.0.0.1:39004/api/tags

# === ESECUZIONE PYTHON STABILIZZATA ===
echo ""
echo "🎯 AVVIO ELABORAZIONE STABILIZZATA"
echo "================================="

# Configurazione finale per Python
export OLLAMA_MODEL="$MODEL_NAME"
export PRODUCTION_MODE=1
export GPU_COUNT=4
export MAX_CONCURRENT_REQUESTS=$WORKING_PORTS

echo "📊 Configurazione finale:"
echo "   🎯 GPU operative: $WORKING_PORTS/4"
echo "   ⚡ Rate limit: $WORKING_PORTS richieste concurrent"
echo "   💾 Salvataggio: ogni 500 carte"
echo "   🔄 Timeout richiesta: 300s"
echo "   🛡️ Modalità stabile: attiva"
echo ""

# Avvio monitoring in background
monitor_system &
MONITOR_PID=$!

# Esecuzione con gestione errori robusta
PYTHON_START=$(date +%s)

echo "🚀 Avvio script Python stabilizzato..."
echo "📄 Target: $TARGET_FILE"
echo "⏰ Inizio: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

# Esecuzione con timeout di 3.5 ore (lascia margine per cleanup)
if timeout 12600s python3 -u veronacard_mob_with_geom_parrallel.py \
    --file "$TARGET_FILE" \
    --append 2>&1 | tee python_execution.log; then
    
    PYTHON_END=$(date +%s)
    PYTHON_TIME=$((PYTHON_END - PYTHON_START))
    
    echo ""
    echo "🎉 ELABORAZIONE COMPLETATA CON SUCCESSO!"
    echo "⏱️ Tempo Python: $PYTHON_TIME sec ($(($PYTHON_TIME / 3600))h $(($PYTHON_TIME % 3600 / 60))m)"
    PYTHON_SUCCESS=true
    
else
    PYTHON_EXIT=$?
    PYTHON_END=$(date +%s) 
    PYTHON_TIME=$((PYTHON_END - PYTHON_START))
    
    echo ""
    echo "❌ ERRORE ELABORAZIONE (exit: $PYTHON_EXIT)"
    echo "⏱️ Tempo prima fallimento: $PYTHON_TIME sec"
    echo "📜 Diagnostica errore:"
    
    case $PYTHON_EXIT in
        124) echo "⚠️ Timeout 3.5h - risultati parziali potrebbero essere salvati" ;;
        130) echo "⚠️ Interruzione manuale (SIGINT)" ;;
        137) echo "⚠️ Kill signal (SIGKILL) - possibile OOM o kill del sistema" ;;
        139) echo "⚠️ Segfault - problema GPU/driver" ;;
        1)   echo "⚠️ Errore Python - controllare log dettagliato" ;;
        *)   echo "⚠️ Exit code inatteso: $PYTHON_EXIT" ;;
    esac
    
    echo ""
    echo "📜 Ultime 15 righe del log Python:"
    tail -15 python_execution.log 2>/dev/null || echo "Log non disponibile"
    
    PYTHON_SUCCESS=false
fi

# Stop monitoring
kill $MONITOR_PID 2>/dev/null || true
wait $MONITOR_PID 2>/dev/null || true

# === REPORT FINALE DETTAGLIATO ===
echo ""
echo "📋 REPORT FINALE HPC STABILIZZATO"
echo "================================="

TOTAL_TIME=$SECONDS
echo "⏱️ Tempo totale job: $TOTAL_TIME sec ($(($TOTAL_TIME / 3600))h $(($TOTAL_TIME % 3600 / 60))m)"
echo "   📡 Setup sistema: $(($PYTHON_START - 0)) sec"
echo "   🐍 Elaborazione Python: $PYTHON_TIME sec"
echo "   🧹 Cleanup: $((TOTAL_TIME - PYTHON_END)) sec"
echo "🚀 Modalità: HPC stabilizzato (4x A100 anti-contention)"
echo "✅ Esito Python: $PYTHON_SUCCESS"

# Statistiche risultati dettagliate
echo ""
echo "📊 ANALISI RISULTATI:"
if [ -d "results/" ]; then
    FINAL_FILES=$(ls -1 results/*.csv 2>/dev/null | wc -l)
    FINAL_SIZE=$(du -sh results/ 2>/dev/null | cut -f1 || echo "N/A")
    
    echo "   📁 File CSV generati: $FINAL_FILES"
    echo "   💾 Dimensione totale: $FINAL_SIZE" 
    
    # Calcolo throughput
    if [ "$PYTHON_SUCCESS" = "true" ] && [ $PYTHON_TIME -gt 0 ]; then
        if command -v bc >/dev/null 2>&1; then
            THROUGHPUT=$(echo "scale=2; $FINAL_FILES * 3600 / $PYTHON_TIME" | bc -l 2>/dev/null || echo "N/A")
            echo "   ⚡ Throughput: $THROUGHPUT files/hour"
            
            # Stima carte processate (assumendo media 1000 carte per file)
            ESTIMATED_CARDS=$((FINAL_FILES * 1000))
            CARD_RATE=$(echo "scale=1; $ESTIMATED_CARDS / $PYTHON_TIME" | bc -l 2>/dev/null || echo "N/A")
            echo "   🎯 Rate stimato: $CARD_RATE carte/sec"
        fi
    fi
    
    # File più recenti con timestamp
    echo ""
    echo "   📋 File recenti generati:"
    ls -laht results/*.csv 2>/dev/null | head -3 | while read line; do
        echo "     $line"
    done
else
    echo "   ❌ Nessun risultato trovato in results/"
fi

# Stato GPU finale dettagliato
echo ""
echo "🔧 STATO GPU FINALE:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu,power.draw --format=csv,noheader | \
    while IFS=',' read idx name mem_used mem_total util temp power; do
        mem_percent=$(echo "scale=1; $mem_used * 100 / $mem_total" | bc -l 2>/dev/null || echo "N/A")
        printf "GPU%s (%s): Mem %s%% Util %s%% Temp %s°C Power %sW\n" \
               "$idx" "${name:0:12}" "$mem_percent" "$util" "$temp" "$power"
    done

# Analisi log Ollama (sample per troubleshooting)
echo ""
echo "📜 SAMPLE LOG OLLAMA (troubleshooting):"
for i in 0 1 2 3; do
    if [ -f "ollama_gpu${i}.log" ]; then
        ERROR_COUNT=$(grep -i error ollama_gpu${i}.log | wc -l)
        WARN_COUNT=$(grep -i warn ollama_gpu${i}.log | wc -l)
        echo "GPU $i: $ERROR_COUNT errors, $WARN_COUNT warnings"
        if [ $ERROR_COUNT -gt 0 ]; then
            echo "   Ultimi errori:"
            grep -i error ollama_gpu${i}.log | tail -2 | sed 's/^/     /'
        fi
    else
        echo "GPU $i: log non disponibile"
    fi
done

# Conclusione con raccomandazioni
echo ""
if [ "$PYTHON_SUCCESS" = "true" ]; then
    echo "🎉 JOB HPC COMPLETATO CON SUCCESSO!"
    echo "⚡ Elaborazione stabilizzata su 4x A100 completata senza errori critici"
    
    if [ $PYTHON_TIME -gt 0 ] && [ $FINAL_FILES -gt 0 ]; then
        echo "📈 Performance: sistema stabile e produttivo"
        
        # Suggerimenti per run futuri
        if [ $PYTHON_TIME -gt 7200 ]; then  # > 2 ore
            echo "💡 Suggerimento: per dataset più grandi considera --time=06:00:00"
        fi
        
        if [ $WORKING_PORTS -lt 4 ]; then
            echo "💡 Nota: solo $WORKING_PORTS/4 GPU utilizzate - verifica configurazione per run futuri"
        fi
    fi
else
    echo "⚠️ JOB TERMINATO CON PROBLEMI"
    echo ""
    echo "🔍 TROUBLESHOOTING SUGGERITO:"
    
    if [ $PYTHON_EXIT -eq 124 ]; then
        echo "   • Timeout: aumenta --time nel job SLURM per dataset grandi"
        echo "   • Verifica risultati parziali in results/ (potrebbero essere utili)"
    elif [ $PYTHON_EXIT -eq 137 ]; then
        echo "   • OOM o kill: aumenta --mem nel job SLURM"
        echo "   • Riduce --max-users nel script Python per test"
    else
        echo "   • Controlla log Python completo: cat python_execution.log"
        echo "   • Verifica log Ollama per errori GPU specifici"
        echo "   • Considera restart del job con stesso --append per continuare"
    fi
    
    echo ""
    echo "📄 Log disponibili per analisi:"
    echo "   • python_execution.log (output Python completo)"
    for i in 0 1 2 3; do
        if [ -f "ollama_gpu${i}.log" ]; then
            echo "   • ollama_gpu${i}.log (log GPU $i)"
        fi
    done
fi

echo ""
echo "🏁 Fine job: $(date '+%Y-%m-%d %H:%M:%S')"
echo "📊 Job ID: $SLURM_JOB_ID"
echo "🖥️ Nodo: $(hostname)"
echo "================================="