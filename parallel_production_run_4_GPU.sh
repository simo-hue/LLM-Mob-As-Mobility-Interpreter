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

export OLLAMA_DEBUG=1                      # ✅ ABILITA debug per diagnostica
export OLLAMA_VERBOSE=1                    # ✅ Log verbosi temporanei
export OLLAMA_ORIGINS="*"
export OLLAMA_MODELS="$WORK/.ollama/models"
export OLLAMA_CACHE_DIR="$WORK/.ollama/cache"

# 🚨 CONFIGURAZIONI CRITICHE POTENZIATE
export OLLAMA_NUM_PARALLEL=1              # Confermato: 1 richiesta per volta
export OLLAMA_MAX_LOADED_MODELS=1         # Confermato: 1 modello per volta
export OLLAMA_KEEP_ALIVE="4h"             # ✅ AUMENTATO: da 1h a 4h
export OLLAMA_LOAD_TIMEOUT=3600           # ✅ AUMENTATO: da 20min a 60min !!!
export OLLAMA_REQUEST_TIMEOUT=600         # ✅ AUMENTATO: da 5min a 10min
export OLLAMA_MAX_QUEUE=1                 # ✅ RIDOTTO: da 2 a 1 (più conservativo)

# 🎯 CONFIGURAZIONI A100-SPECIFIC POTENZIATE
export OLLAMA_MAX_VRAM_USAGE=0.85         # ✅ AUMENTATO: da 75% a 85%
export OLLAMA_RUNNER_CACHE_SIZE="8GB"     # ✅ AUMENTATO: da 3GB a 8GB
export OLLAMA_MAX_CONCURRENT_DOWNLOADS=1  # Confermato
export OLLAMA_FLASH_ATTENTION=1           # Confermato

# 🔥 NUOVE CONFIGURAZIONI CRITICHE
export OLLAMA_LLM_LIBRARY="cuda_v12"      # ✅ NUOVO: Forza CUDA 12
export OLLAMA_CUDA_MEMORY_FRACTION=0.85   # ✅ NUOVO: Controlla allocazione CUDA
export OLLAMA_GPU_LAYERS=-1               # ✅ NUOVO: Tutte le layer su GPU
export OLLAMA_BATCH_SIZE=1024              # ✅ NUOVO: Batch size ottimizzato A100
export OLLAMA_CONTEXT_SIZE=4096           # ✅ NUOVO: Context size maggiore
export OLLAMA_PREDICTION_TOKENS=512       # ✅ NUOVO: Token predizione maggiore

# 📊 LOG delle configurazioni per debug
echo "🔍 Configurazioni Ollama attive:"
env | grep OLLAMA_ | sort

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
    local startup_timeout=900  # 15 minuti per startup !!!
    
    echo "🔥 Avvio ROBUSTO istanza GPU $gpu_id su porta $port..."
    
    # 🧹 Cleanup specifico porta
    echo "🧹 Cleanup specifico porta $port..."
    pkill -f "OLLAMA_HOST=127.0.0.1:$port" 2>/dev/null || true
    sleep 5
    
    # 🔍 Verifica porta libera
    if netstat -tuln 2>/dev/null | grep -q ":$port "; then
        echo "⚠️ Porta $port ancora occupata, cleanup forzato..."
        fuser -k $port/tcp 2>/dev/null || true
        sleep 10
    fi
    
    # 💾 Verifica spazio WORK con margine maggiore
    WORK_SPACE=$(df "$WORK" | tail -1 | awk '{print $4}')
    WORK_SPACE_GB=$((WORK_SPACE / 1024 / 1024))
    
    if [ $WORK_SPACE_GB -lt 35 ]; then  # ✅ AUMENTATO: da 25GB a 35GB
        echo "❌ GPU $gpu_id: spazio $WORK insufficiente (${WORK_SPACE_GB}GB < 35GB)"
        return 1
    fi
    
    # 📁 Cache dedicata con cleanup preventivo
    GPU_CACHE_DIR="$OLLAMA_CACHE_DIR/gpu${gpu_id}"
    rm -rf "$GPU_CACHE_DIR" 2>/dev/null || true  # ✅ NUOVO: cleanup cache
    mkdir -p "$GPU_CACHE_DIR"
    chmod 755 "$GPU_CACHE_DIR"
    
    echo "✅ GPU $gpu_id: spazio OK (${WORK_SPACE_GB}GB), cache: $GPU_CACHE_DIR"
    
    # 🚀 Lancio con configurazioni potenziate
    echo "🚀 Avvio Ollama GPU $gpu_id con timeout esteso..."
    CUDA_VISIBLE_DEVICES=$gpu_id \
    OLLAMA_HOST=127.0.0.1:$port \
    OLLAMA_MAX_LOADED_MODELS=1 \
    OLLAMA_LOAD_TIMEOUT=3600 \
    OLLAMA_REQUEST_TIMEOUT=600 \
    OLLAMA_TMPDIR="$CUSTOM_TMP" \
    OLLAMA_CACHE_DIR="$GPU_CACHE_DIR" \
    TMPDIR="$CUSTOM_TMP" \
    TMP="$CUSTOM_TMP" \
    TEMP="$CUSTOM_TMP" \
    timeout $startup_timeout $OLLAMA_BIN serve > $log_file 2>&1 &
    
    local pid=$!
    eval "SERVER_PID$((gpu_id+1))=$pid"
    
    echo "✅ GPU $gpu_id PID: $pid (timeout startup: ${startup_timeout}s)"
    echo "   📁 Cache: $GPU_CACHE_DIR"
    echo "   📄 Log: $log_file"
    
    # 🔍 Verifica immediata che il processo sia avviato
    sleep 5
    if ! kill -0 $pid 2>/dev/null; then
        echo "❌ GPU $gpu_id: processo terminato immediatamente"
        echo "🔍 Ultimi 10 righe log:"
        tail -10 $log_file 2>/dev/null || echo "Log non disponibile"
        return 1
    fi
    
    echo "✅ GPU $gpu_id: processo avviato correttamente"
    return 0
}

# Avvio sequenziale con attese lunghe per evitare race conditions
echo ""
echo "🚀 Avvio server Ollama con anti-contention POTENZIATO..."

# 🎯 ATTESE DRASTICAMENTE AUMENTATE
echo "🔡 Avvio istanza 1/4 (MASTER)..."
if ! start_ollama_instance 0 $OLLAMA_PORT1; then
    echo "❌ Fallito avvio GPU 0 - abort"
    exit 1
fi

echo "⏳ Attesa caricamento modello su GPU 0 MASTER (180s)..."  # ✅ AUMENTATO: da 90s a 180s
sleep 180

echo "🔡 Avvio istanza 2/4..."
start_ollama_instance 1 $OLLAMA_PORT2
echo "⏳ Attesa stabilizzazione GPU 1 (90s)..."  # ✅ AUMENTATO: da 40s a 90s
sleep 90

echo "🔡 Avvio istanza 3/4..."
start_ollama_instance 2 $OLLAMA_PORT3  
echo "⏳ Attesa stabilizzazione GPU 2 (90s)..."  # ✅ AUMENTATO: da 40s a 90s
sleep 90

echo "🔡 Avvio istanza 4/4..."
start_ollama_instance 3 $OLLAMA_PORT4
echo "⏳ Attesa stabilizzazione GPU 3 (90s)..."  # ✅ AUMENTATO: da 40s a 90s
sleep 90

echo "⏳ Attesa stabilizzazione COMPLETA sistema (120s)..."  # ✅ AUMENTATO: da 60s a 120s
sleep 120

# 📊 Verifica processi attivi prima di health check
echo "🔍 Verifica processi Ollama attivi:"
ACTIVE_PROCESSES=0
for i in 1 2 3 4; do
    eval "pid=\$SERVER_PID$i"
    if [ -n "$pid" ] && kill -0 $pid 2>/dev/null; then
        echo "✅ GPU $((i-1)): PID $pid attivo"
        ((ACTIVE_PROCESSES++))
    else
        echo "❌ GPU $((i-1)): PID $pid NON attivo"
        echo "🔍 Log GPU $((i-1)):"
        tail -5 "ollama_gpu$((i-1)).log" 2>/dev/null || echo "Log non disponibile"
    fi
done

if [ $ACTIVE_PROCESSES -lt 2 ]; then
    echo "❌ ERRORE: Troppo pochi processi attivi ($ACTIVE_PROCESSES/4)"
    exit 1
fi

# === HEALTH CHECK APPROFONDITO ===
echo ""
echo "🔍 Health check SUPER-ROBUSTO multi-istanza..."

HEALTHY_COUNT=0
RETRY_COUNT=0
MAX_RETRIES=2

# Prima passata health check
echo "📋 Prima passata health check..."
check_instance_health $OLLAMA_PORT1 0 && ((HEALTHY_COUNT++)) || echo "GPU 0 fallita prima passata"
check_instance_health $OLLAMA_PORT2 1 && ((HEALTHY_COUNT++)) || echo "GPU 1 fallita prima passata"  
check_instance_health $OLLAMA_PORT3 2 && ((HEALTHY_COUNT++)) || echo "GPU 2 fallita prima passata"
check_instance_health $OLLAMA_PORT4 3 && ((HEALTHY_COUNT++)) || echo "GPU 3 fallita prima passata"

echo "📊 Prima passata: $HEALTHY_COUNT/4 istanze operative"

# 🔄 RETRY per GPU fallite
if [ $HEALTHY_COUNT -lt 4 ] && [ $HEALTHY_COUNT -ge 1 ]; then
    echo "🔄 Retry health check per GPU fallite..."
    sleep 60  # Attesa aggiuntiva
    
    # Retry solo GPU fallite
    [ $HEALTHY_COUNT -lt 4 ] && ! check_instance_health $OLLAMA_PORT1 0 >/dev/null 2>&1 && \
        check_instance_health $OLLAMA_PORT1 0 && ((HEALTHY_COUNT++))
        
    [ $HEALTHY_COUNT -lt 4 ] && ! check_instance_health $OLLAMA_PORT2 1 >/dev/null 2>&1 && \
        check_instance_health $OLLAMA_PORT2 1 && ((HEALTHY_COUNT++))
        
    [ $HEALTHY_COUNT -lt 4 ] && ! check_instance_health $OLLAMA_PORT3 2 >/dev/null 2>&1 && \
        check_instance_health $OLLAMA_PORT3 2 && ((HEALTHY_COUNT++))
        
    [ $HEALTHY_COUNT -lt 4 ] && ! check_instance_health $OLLAMA_PORT4 3 >/dev/null 2>&1 && \
        check_instance_health $OLLAMA_PORT4 3 && ((HEALTHY_COUNT++))
fi

echo "📊 Health check finale: $HEALTHY_COUNT/4 istanze operative"

# ============= GESTIONE RISULTATO HEALTH CHECK =============
if [ $HEALTHY_COUNT -eq 0 ]; then
    echo "❌ ERRORE CRITICO: Nessuna istanza funzionante"
    echo ""
    echo "🔍 DIAGNOSI DETTAGLIATA:"
    echo "Tutti i processi Ollama sono avviati ma non rispondono alle richieste."
    echo "Cause possibili:"
    echo "1. Modello Mixtral:8x7b non scaricato/non trovato"
    echo "2. GPU memory insufficiente per caricare il modello"
    echo "3. Timeout durante caricamento modello (26GB)"
    echo ""
    echo "🛠️ DEBUG NECESSARIO:"
    for i in 0 1 2 3; do
        echo "--- GPU $i DIAGNOSI ---"
        local pid_var="SERVER_PID$((i+1))"
        local pid=$(eval echo \$$pid_var)
        echo "PID: $pid"
        echo "Processo attivo: $(kill -0 $pid 2>/dev/null && echo 'SI' || echo 'NO')"
        echo "Log (ultime 10 righe):"
        tail -10 "ollama_gpu${i}.log" 2>/dev/null | sed 's/^/  /' || echo "  Log non disponibile"
        echo ""
    done
    exit 1

elif [ $HEALTHY_COUNT -eq 1 ]; then
    echo "⚠️ WARNING: Solo 1 GPU operativa - continuiamo con performance ridotte"
    echo "🎯 Sistema degradato ma funzionale"
    
elif [ $HEALTHY_COUNT -eq 2 ]; then
    echo "⚠️ WARNING: Solo 2 GPU operative - performance parziali"
    echo "🎯 Sistema accettabile per testing"
elif [ $HEALTHY_COUNT -eq 3 ]; then
    echo "⚠️ WARNING: Solo 3 GPU operative - performance parziali"
    echo "🎯 Sistema quasi PRODUCTION"
    
else
    echo "✅ Sistema OTTIMO: $HEALTHY_COUNT GPU operative"
fi

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

check_instance_health() {
    local port=$1
    local gpu_id=$2
    local max_attempts=20      # Ridotto per primo test
    local wait_time=10         # 10 secondi tra tentativi
    
    echo "🔍 Health check GPU $gpu_id (porta $port)..."
    
    # GPU 0 ha timeout esteso (è la master)
    if [ $gpu_id -eq 0 ]; then
        max_attempts=30
        echo "👑 GPU $gpu_id MASTER - timeout esteso (30 tentativi)"
    fi
    
    for i in $(seq 1 $max_attempts); do
        echo "   🔄 Tentativo $i/$max_attempts..."
        
        # Test 1: Verifica processo ancora attivo
        local pid_var="SERVER_PID$((gpu_id+1))"
        local pid=$(eval echo \$$pid_var)
        if [ -n "$pid" ] && ! kill -0 $pid 2>/dev/null; then
            echo "   ❌ Processo GPU $gpu_id (PID $pid) terminato"
            echo "   🔍 Ultime righe log:"
            tail -5 "ollama_gpu${gpu_id}.log" 2>/dev/null || echo "Log non disponibile"
            return 1
        fi
        
        # Test 2: Connection test con timeout
        echo "   🌐 Test connessione..."
        if timeout 20s curl -s --connect-timeout 5 --max-time 20 \
             "http://127.0.0.1:$port/api/tags" >/dev/null 2>&1; then
            echo "   ✅ Connessione OK, test funzionalità..."
            
            # Test 3: Mini-inference test
            local test_response=$(timeout 60s curl -s --connect-timeout 10 --max-time 60 \
                -X POST "http://127.0.0.1:$port/api/generate" \
                -H "Content-Type: application/json" \
                -d '{
                    "model":"mixtral:8x7b",
                    "prompt":"Test",
                    "stream":false,
                    "options":{
                        "num_predict":1,
                        "temperature":0,
                        "num_ctx":512
                    }
                }' 2>/dev/null)
            
            # Verifica risposta valida
            if echo "$test_response" | grep -q '"done":true' && \
               echo "$test_response" | grep -q '"response"'; then
                echo "   ✅ GPU $gpu_id COMPLETAMENTE OPERATIVA dopo $i tentativi"
                return 0
            fi
            
            echo "   ⚠️ Test inference fallito, risposta parziale: $(echo "$test_response" | head -c 100)..."
        else
            echo "   ⏳ Connessione non pronta, attesa ${wait_time}s..."
        fi
        
        # Log diagnostico ogni 5 tentativi
        if [ $((i % 5)) -eq 0 ]; then
            echo "   📊 Diagnostica tentativo $i:"
            echo "   🔍 Processo: $(kill -0 $pid 2>/dev/null && echo 'ATTIVO' || echo 'MORTO')"
            echo "   🔍 Porta: $(netstat -tuln 2>/dev/null | grep ":$port " | wc -l) listener(s)"
            echo "   🔍 Log recenti:"
            tail -2 "ollama_gpu${gpu_id}.log" 2>/dev/null | sed 's/^/        /' || echo "        Log non disponibile"
        fi
        
        sleep $wait_time
    done
    
    echo "   ❌ GPU $gpu_id FALLITA dopo $max_attempts tentativi"
    return 1
}

# === PREPARAZIONE MODELLO SEQUENZIALE ===
echo ""
echo "📥 Preparazione modello con prevenzione contention..."

MODEL_NAME="mixtral:8x7b"

# ⚠️ PUNTO CRITICO: Verifica che il modello esista
echo "🔍 Verifica modello $MODEL_NAME..."

# Usa GPU 0 (master) per check modello
MODELS_RESPONSE=""
MODEL_CHECK_SUCCESS=false

# Prova con tutte le porte finché una non risponde
for port in $OLLAMA_PORT1 $OLLAMA_PORT2 $OLLAMA_PORT3 $OLLAMA_PORT4; do
    echo "🔍 Tentativo check modello su porta $port..."
    MODELS_RESPONSE=$(timeout 30s curl -s "http://127.0.0.1:$port/api/tags" 2>/dev/null || echo '{"models":[]}')
    
    if echo "$MODELS_RESPONSE" | grep -q '"models"'; then
        MODEL_CHECK_SUCCESS=true
        echo "✅ Connessione modello OK su porta $port"
        break
    else
        echo "⚠️ Porta $port non risponde per check modello"
    fi
done

if [ "$MODEL_CHECK_SUCCESS" = "false" ]; then
    echo "❌ ERRORE: Impossibile verificare modelli su nessuna porta"
    echo "🔍 Response sample: $MODELS_RESPONSE"
    echo "Probabilmente Ollama non è ancora pronto per servire richieste"
    exit 1
fi

# Parse del JSON per verificare presenza modello
MODEL_EXISTS=$(echo "$MODELS_RESPONSE" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    models = [m.get('name', '') for m in data.get('models', [])]
    print('true' if '$MODEL_NAME' in models else 'false')
except Exception as e:
    print('false')
" 2>/dev/null)

echo "🔍 Risultato check modello: MODEL_EXISTS='$MODEL_EXISTS'"

if [ "$MODEL_EXISTS" != "true" ]; then
    echo "🔥 DOWNLOAD modello $MODEL_NAME (questo può richiedere 20-40 minuti)..."
    echo "📊 Modello Mixtral:8x7b = ~26GB download"
    
    # Usa solo GPU 0 per download per evitare conflitti
    if timeout 3600s CUDA_VISIBLE_DEVICES=0 \
       OLLAMA_MODELS="$WORK/.ollama/models" \
       $OLLAMA_BIN pull $MODEL_NAME; then
        echo "✅ Modello scaricato con successo"
    else
        echo "❌ Download fallito dopo 60 minuti - abort"
        echo "🔍 Possibili cause:"
        echo "  • Connessione internet instabile"
        echo "  • Spazio disco insufficiente su $WORK"
        echo "  • Timeout download (modello molto grande)"
        exit 1
    fi
else
    echo "✅ Modello $MODEL_NAME già presente"
fi

preload_gpu_sequential() {
    local port=$1
    local gpu_id=$2
    local max_attempts=3
    
    echo "⚡ Warm-up ROBUSTO GPU $gpu_id..."
    
    for attempt in $(seq 1 $max_attempts); do
        echo "   🔄 Tentativo warm-up $attempt/$max_attempts..."
        
        local warmup_response=$(timeout 300s curl -s -X POST \
            "http://127.0.0.1:$port/api/chat" \
            -H "Content-Type: application/json" \
            -d '{
                "model":"mixtral:8x7b",
                "messages":[{"role":"user","content":"warmup test"}],
                "stream":false,
                "options":{
                    "num_ctx":2048,
                    "num_predict":2,
                    "num_batch":512,
                    "temperature":0,
                    "num_thread":32
                }
            }' 2>&1)
        
        if echo "$warmup_response" | grep -q '"done":true'; then
            echo "   ✅ GPU $gpu_id warm-up SUCCESSO tentativo $attempt"
            return 0
        else
            echo "   ⚠️ GPU $gpu_id warm-up fallito tentativo $attempt"
            echo "   📄 Risposta: $(echo "$warmup_response" | head -c 200)"
            if [ $attempt -lt $max_attempts ]; then
                echo "   ⏳ Attesa 30s prima retry..."
                sleep 30
            fi
        fi
    done
    
    echo "   ⚠️ GPU $gpu_id warm-up parziale (falliti $max_attempts tentativi)"
    return 1
}

# Warm-up con attese estese
echo ""
echo "🔥 Pre-caricamento sequenziale ROBUSTO..."
READY_GPUS=0

for i in 0 1 2 3; do
    eval "port=\$OLLAMA_PORT$((i+1))"
    
    # Solo se GPU ha passato health check
    if check_instance_health $port $i >/dev/null 2>&1; then
        echo "🎯 Warm-up GPU $i..."
        if preload_gpu_sequential $port $i; then
            ((READY_GPUS++))
        fi
    else
        echo "⏭️ Skip warm-up GPU $i (health check fallito)"
    fi
    
    # Pause anti-contention estese
    if [ $i -lt 3 ]; then
        echo "⏸️ Pausa anti-contention estesa (30s)..."  # ✅ AUMENTATO: da 15s a 30s
        sleep 30
    fi
done

echo "✅ GPU pronte per produzione: $READY_GPUS/4"
HEALTHY_COUNT=$READY_GPUS

# === VERIFICA FILE INPUT ===
echo ""
echo "📂 Verifica file di input..."

cd /leonardo_work/IscrC_LLM-Mob/LLM-Mob-As-Mobility-Interpreter

TARGET_FILE="data/verona/veronacard_2022_original.csv"
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
    local check_interval=300
    local error_threshold=10
    local consecutive_errors=0
    
    echo "🔍 Monitor sistema POTENZIATO avviato (intervallo: ${check_interval}s)"
    
    while true; do
        sleep $check_interval
        
        local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        
        # 📊 Statistiche risultati
        if [ -d "results/" ]; then
            local total_files=$(ls -1 results/*.csv 2>/dev/null | wc -l)
            local latest_size=$(du -sh results/ 2>/dev/null | cut -f1 || echo "0B")
            echo "📊 [$timestamp] Files: $total_files, Size: $latest_size"
        fi
        
        # 🔍 Check processo Python
        if ! pgrep -f "veronacard_mob_with_geom" >/dev/null; then
            echo "ℹ️ [$timestamp] Processo Python terminato - stopping monitor"
            break
        fi
        
        # 🌡️ Monitoring GPU avanzato
        local hot_gpus=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | \
                        awk '$1 > 85 {print NR-1}')
        if [ -n "$hot_gpus" ]; then
            echo "🌡️ [$timestamp] GPU calde (>85°C): $hot_gpus"
        fi
        
        # 🚨 Controllo health Ollama periodico
        local unhealthy_ports=""
        for i in 0 1 2 3; do
            eval "port=\$OLLAMA_PORT$((i+1))"
            if ! timeout 10s curl -s "http://127.0.0.1:$port/api/tags" >/dev/null 2>&1; then
                unhealthy_ports="$unhealthy_ports $port"
                ((consecutive_errors++))
            fi
        done
        
        if [ -n "$unhealthy_ports" ]; then
            echo "🚨 [$timestamp] Porte Ollama non responsive:$unhealthy_ports"
            
            # Se troppe porte non rispondono, alert critico
            local unhealthy_count=$(echo $unhealthy_ports | wc -w)
            if [ $unhealthy_count -ge 3 ]; then
                echo "❌ [$timestamp] ALERT: $unhealthy_count/4 porte non responsive - possibile system failure"
            fi
        else
            consecutive_errors=0  # Reset error counter
        fi
        
        # 🚨 Alert per troppi errori consecutivi
        if [ $consecutive_errors -ge $error_threshold ]; then
            echo "🚨 [$timestamp] ALERT: $consecutive_errors errori consecutivi - sistema potenzialmente instabile"
            consecutive_errors=0  # Reset per evitare spam
        fi
    done
    
    echo "🔍 Monitor sistema terminato: $timestamp"
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
export OLLAMA_MODEL="mixtral:8x7b"
export PRODUCTION_MODE=1
export GPU_COUNT=$HEALTHY_COUNT              # Usa GPU effettivamente operative
export MAX_CONCURRENT_REQUESTS=$HEALTHY_COUNT  # Non più del numero GPU attive
export OLLAMA_TIMEOUT=600                    # Timeout per richieste Python
export BATCH_SIZE=100                        # Batch size ridotto per stabilità

# Salva configurazione con solo porte funzionanti
WORKING_PORTS=""
for i in 0 1 2 3; do
    eval "port=\$OLLAMA_PORT$((i+1))"
    if check_instance_health $port $i >/dev/null 2>&1; then
        if [ -z "$WORKING_PORTS" ]; then
            WORKING_PORTS="$port"
        else
            WORKING_PORTS="$WORKING_PORTS,$port"
        fi
    fi
done

echo "$WORKING_PORTS" > ollama_ports.txt
echo "✅ Porte operative salvate: $WORKING_PORTS"

echo "📊 Configurazione Python finale:"
echo "   🎯 GPU effettive: $HEALTHY_COUNT"
echo "   ⚡ Concurrent requests: $MAX_CONCURRENT_REQUESTS"
echo "   ⏱️ Timeout richiesta: $OLLAMA_TIMEOUT"
echo "   📦 Batch size: $BATCH_SIZE"
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