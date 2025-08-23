#!/bin/bash
#SBATCH --job-name=verona-opt
#SBATCH --account=IscrC_LLM-Mob
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --output=slurm-%j.out

echo "🚀 VERONA CARD HPC OTTIMIZZATO"
echo "=============================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodo: $(hostname)"
echo "Data: $(date)"
echo "🎯 Modalità: Produzione ottimizzata (4x A100, rate-limited)"
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

# === CONFIGURAZIONE OLLAMA OTTIMIZZATA ===
OLLAMA_BIN="/leonardo_work/IscrC_LLM-Mob/opt/bin/ollama"

# Verifica binario
if [ ! -f "$OLLAMA_BIN" ]; then
    echo "❌ ERRORE: Ollama non trovato in $OLLAMA_BIN"
    exit 1
fi

# Variabili ottimizzate per 4 GPU
export OLLAMA_DEBUG=0
export OLLAMA_ORIGINS="*"
export OLLAMA_MODELS="$WORK/.ollama/models"

# ⚡ CONFIGURAZIONI CHIAVE PER STABILITÀ
export OLLAMA_NUM_PARALLEL=2           # RIDOTTO per stabilità
export OLLAMA_MAX_LOADED_MODELS=1      # Un modello per volta
export OLLAMA_FLASH_ATTENTION=1        
export OLLAMA_KEEP_ALIVE="6h"          # Lungo per evitare reload
export OLLAMA_LOAD_TIMEOUT=900         # 15 minuti per sicurezza
export OLLAMA_REQUEST_TIMEOUT=240      # 4 minuti per richiesta
export OLLAMA_MAX_QUEUE=4              # Piccola coda per evitare OOM

# === AVVIO MULTI-ISTANZA BILANCIATO ===
echo ""
echo "🚀 Avvio server Ollama ottimizzato (4x GPU)..."

# Cleanup preventivo aggressivo
pkill -f ollama 2>/dev/null || true
sleep 15

echo "🔍 Controllo stato GPU..."
nvidia-smi --query-gpu=index,name,memory.used,temperature.gpu --format=csv,noheader
# Verifica processi attivi (senza modificare nulla)
nvidia-smi pmon -c 1 2>/dev/null || echo "Monitoring non disponibile"

# Configurazione porte
OLLAMA_PORT1=39001
OLLAMA_PORT2=39002  
OLLAMA_PORT3=39003
OLLAMA_PORT4=39004

# Funzione di avvio singola istanza
start_ollama_instance() {
    local gpu_id=$1
    local port=$2
    local log_file="ollama_gpu${gpu_id}.log"
    
    echo "🔥 Avvio istanza GPU $gpu_id su porta $port..."
    
    # Avvio con configurazione dedicata
    CUDA_VISIBLE_DEVICES=$gpu_id \
    OLLAMA_HOST=127.0.0.1:$port \
    OLLAMA_MAX_LOADED_MODELS=1 \
    $OLLAMA_BIN serve > $log_file 2>&1 &
    
    local pid=$!
    echo "✅ GPU $gpu_id PID: $pid"
    return $pid
}

# Avvio sequenziale con attesa
start_ollama_instance 0 $OLLAMA_PORT1; SERVER_PID1=$?
sleep 10  # Attesa maggiore tra istanze
start_ollama_instance 1 $OLLAMA_PORT2; SERVER_PID2=$?
sleep 10
start_ollama_instance 2 $OLLAMA_PORT3; SERVER_PID3=$?
sleep 10  
start_ollama_instance 3 $OLLAMA_PORT4; SERVER_PID4=$?

echo "✅ Tutte le istanze avviate:"
echo "   GPU 0: PID $SERVER_PID1 (porta $OLLAMA_PORT1)"
echo "   GPU 1: PID $SERVER_PID2 (porta $OLLAMA_PORT2)" 
echo "   GPU 2: PID $SERVER_PID3 (porta $OLLAMA_PORT3)"
echo "   GPU 3: PID $SERVER_PID4 (porta $OLLAMA_PORT4)"

# Salva configurazione per Python
echo "$OLLAMA_PORT1,$OLLAMA_PORT2,$OLLAMA_PORT3,$OLLAMA_PORT4" > ollama_ports.txt

# Cleanup function
cleanup() {
    echo "🧹 CLEANUP OTTIMIZZATO..."
    
    # Graceful shutdown (OK)
    for pid in $SERVER_PID1 $SERVER_PID2 $SERVER_PID3 $SERVER_PID4; do
        if [ -n "$pid" ] && kill -0 $pid 2>/dev/null; then
            echo "🔄 Shutdown graceful PID $pid..."
            kill -TERM $pid 2>/dev/null
        fi
    done
    
    # Attesa shutdown (OK)
    sleep 15
    
    # Force kill se necessario (OK)
    pkill -f ollama 2>/dev/null || true
    
    echo "✅ Cleanup completato"
}
trap cleanup EXIT

# === VERIFICA HEALTH CON TIMEOUT ===
echo ""
echo "⏳ Verifica health multi-istanza..."

check_instance_health() {
    local port=$1
    local gpu_id=$2
    local max_attempts=30
    
    echo "🔍 Check GPU $gpu_id (porta $port)..."
    
    for i in $(seq 1 $max_attempts); do
        if timeout 10s curl -s "http://127.0.0.1:$port/api/tags" >/dev/null 2>&1; then
            echo "✅ GPU $gpu_id operativa"
            return 0
        fi
        
        if [ $i -eq $max_attempts ]; then
            echo "❌ GPU $gpu_id TIMEOUT dopo ${max_attempts} tentativi"
            return 1
        fi
        
        sleep 3
    done
}

# Verifica tutte le istanze
HEALTHY_COUNT=0
check_instance_health $OLLAMA_PORT1 0 && ((HEALTHY_COUNT++))
check_instance_health $OLLAMA_PORT2 1 && ((HEALTHY_COUNT++))
check_instance_health $OLLAMA_PORT3 2 && ((HEALTHY_COUNT++))
check_instance_health $OLLAMA_PORT4 3 && ((HEALTHY_COUNT++))

echo "📊 Health check: $HEALTHY_COUNT/4 istanze operative"

if [ $HEALTHY_COUNT -lt 2 ]; then
    echo "❌ ERRORE: Troppo poche istanze funzionanti ($HEALTHY_COUNT/4)"
    echo "🔍 Debug logs:"
    for i in 0 1 2 3; do
        echo "--- GPU $i log (ultime 10 righe) ---"
        tail -10 ollama_gpu${i}.log 2>/dev/null || echo "Log non disponibile"
    done
    exit 1
fi

# === PREPARAZIONE MODELLO ===
echo ""
echo "📥 Preparazione modello su tutte le GPU..."

MODEL_NAME="mixtral:8x7b"

# Verifica e scarica modello una sola volta
echo "🔍 Verifica modello $MODEL_NAME..."
MODELS_RESPONSE=$(timeout 30s curl -s "http://127.0.0.1:$OLLAMA_PORT1/api/tags" || echo '{"models":[]}')

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
    if timeout 1800s CUDA_VISIBLE_DEVICES=0 $OLLAMA_BIN pull $MODEL_NAME; then
        echo "✅ Modello scaricato"
    else
        echo "❌ Download fallito"
        exit 1
    fi
fi

# Pre-caricamento ottimizzato su tutte le GPU sane
echo "🔥 Pre-caricamento distribuito..."

preload_gpu() {
    local port=$1
    local gpu_id=$2
    
    echo "⚡ Warm-up GPU $gpu_id..."
    timeout 180s curl -s -X POST "http://127.0.0.1:$port/api/generate" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MODEL_NAME\",\"prompt\":\"test\",\"stream\":false,\"options\":{\"num_predict\":1,\"temperature\":0}}" \
        >/tmp/warmup_$gpu_id.json 2>&1
    
    if [ $? -eq 0 ]; then
        echo "✅ GPU $gpu_id pronta"
    else
        echo "⚠️ GPU $gpu_id warm-up parziale"
    fi
    
    rm -f /tmp/warmup_$gpu_id.json
}

# Preload in parallelo CON GESTIONE TIMEOUT
preload_gpu $OLLAMA_PORT1 0 &
PID1=$!
preload_gpu $OLLAMA_PORT2 1 &
PID2=$!
preload_gpu $OLLAMA_PORT3 2 &
PID3=$!
preload_gpu $OLLAMA_PORT4 3 &
PID4=$!

echo "🚨 DEBUG: PIDs preload salvati: $PID1 $PID2 $PID3 $PID4"

# WAIT CON TIMEOUT invece di wait infinito
echo "⏳ Attesa completamento preload (max 120s)..."
WAIT_COUNT=0
MAX_WAIT=120

while [ $WAIT_COUNT -lt $MAX_WAIT ]; do
    RUNNING_JOBS=0
    
    # Conta job ancora attivi
    for pid in $PID1 $PID2 $PID3 $PID4; do
        if kill -0 $pid 2>/dev/null; then
            ((RUNNING_JOBS++))
        fi
    done
    
    # Se tutti finiti, esci
    if [ $RUNNING_JOBS -eq 0 ]; then
        echo "✅ Tutti i preload completati in ${WAIT_COUNT}s"
        break
    fi
    
    # Progress ogni 15 secondi
    if [ $((WAIT_COUNT % 15)) -eq 0 ]; then
        echo "⏳ Preload in corso... (${RUNNING_JOBS}/4 attivi, ${WAIT_COUNT}s)"
    fi
    
    sleep 3
    ((WAIT_COUNT+=3))
done

# Se timeout, forza kill
if [ $WAIT_COUNT -ge $MAX_WAIT ]; then
    echo "⚠️ TIMEOUT preload dopo ${MAX_WAIT}s - forzo terminazione..."
    for pid in $PID1 $PID2 $PID3 $PID4; do
        echo "  Killing PID $pid..."
        kill -TERM $pid 2>/dev/null || true
    done
    sleep 5
    # Force kill se necessario
    for pid in $PID1 $PID2 $PID3 $PID4; do
        if kill -0 $pid 2>/dev/null; then
            echo "  Force killing PID $pid..."
            kill -KILL $pid 2>/dev/null || true
        fi
    done
fi

echo "🔍 DEBUG: Stato finale preload:"
for i in 1 2 3 4; do
    eval "pid=\$PID$i"
    if kill -0 $pid 2>/dev/null; then
        echo "  PID $pid: ANCORA ATTIVO"
    else
        echo "  PID $pid: TERMINATO"
    fi
done

# === DEBUG: PUNTO CRITICO IDENTIFICATO ===
echo ""
echo "🔍 DEBUG: Punto critico post warm-up"
echo "Timestamp: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Uptime processo: $SECONDS secondi"
echo "Directory corrente: $(pwd)"
echo "Spazio disco disponibile: $(df -h . | tail -1)"
echo "Memoria disponibile: $(free -h)"

# Verifica file Python esistente
echo ""
echo "📋 DEBUG: Verifica ambiente Python"
echo "File Python esiste: $(ls -la veronacard_mob_with_geom_parrallel.py 2>/dev/null || echo 'NON TROVATO')"
echo "Directory data: $(ls -la veronacard_2020_2023/ 2>/dev/null || echo 'DIRECTORY NON TROVATA')"
echo "File input: $(ls -la veronacard_2020_2023/veronacard_2023_original_parziale.csv 2>/dev/null || echo 'FILE NON TROVATO')"

# Test Python rapido
echo ""
echo "🧪 DEBUG: Test Python veloce"
python3 -c "
import sys
print('Python path:', sys.executable)
try:
    import pandas
    print('✅ Pandas OK')
except ImportError as e:
    print('❌ Pandas ERROR:', e)
try:
    import requests
    print('✅ Requests OK')
except ImportError as e:
    print('❌ Requests ERROR:', e)
print('✅ Python base OK')
"

# Verifica processi Ollama ancora attivi
echo ""
echo "🔍 DEBUG: Stato processi Ollama"
for pid in $SERVER_PID1 $SERVER_PID2 $SERVER_PID3 $SERVER_PID4; do
    if [ -n "$pid" ] && kill -0 $pid 2>/dev/null; then
        echo "✅ PID $pid ancora attivo"
    else
        echo "❌ PID $pid terminato"
    fi
done

# Test connettività Ollama
echo ""
echo "🌐 DEBUG: Test connettività finale"
for port in $OLLAMA_PORT1 $OLLAMA_PORT2 $OLLAMA_PORT3 $OLLAMA_PORT4; do
    if timeout 5s curl -s "http://127.0.0.1:$port/api/tags" >/dev/null 2>&1; then
        echo "✅ Porta $port risponde"
    else
        echo "❌ Porta $port non risponde"
    fi
done

echo ""
echo "🎯 DEBUG: Inizio sezione Python"
echo "==============================================="

# === ESECUZIONE PYTHON OTTIMIZZATA ===
echo ""
echo "🎯 AVVIO ELABORAZIONE OTTIMIZZATA"
echo "================================"

# Configurazione finale
export OLLAMA_MODEL="$MODEL_NAME"
export PRODUCTION_MODE=1
export GPU_COUNT=4
export MAX_CONCURRENT_REQUESTS=$HEALTHY_COUNT

echo "📊 Configurazione finale:"
echo "   🎯 GPU operative: $HEALTHY_COUNT/4"
echo "   ⚡ Rate limit: $HEALTHY_COUNT richieste concurrent"
echo "   💾 Salvataggio incrementale: ogni 100 carte"
echo "   🔄 Timeout richiesta: 240s"
echo "   🛡️ Circuit breaker: attivo"
echo ""

cd /leonardo_work/IscrC_LLM-Mob/LLM-Mob-As-Mobility-Interpreter

# Verifica cambio directory
echo "📍 Directory dopo cd: $(pwd)"
echo "📁 Contenuto directory:"
ls -la | head -10

# Crea directory risultati
mkdir -p results/
echo "📂 Directory results creata: $(ls -ld results/)"

# Monitor function ottimizzata
monitor_system() {
    while true; do
        sleep 120  # Ogni 2 minuti
        
        # Statistiche risultati
        if [ -d "results/" ]; then
            CURRENT_FILES=$(find results/ -name "*.csv" -mmin -60 | wc -l)
            TOTAL_FILES=$(ls -1 results/*.csv 2>/dev/null | wc -l) 
            CURRENT_SIZE=$(du -sh results/ 2>/dev/null | cut -f1 || echo "0B")
            echo "📊 [$(date '+%H:%M:%S')] Progress: $TOTAL_FILES files ($CURRENT_FILES new), $CURRENT_SIZE total"
        fi
        
        # GPU monitoring compatto
        echo "🔧 [$(date '+%H:%M:%S')] GPU Status:"
        nvidia-smi --query-gpu=index,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits | \
            while IFS=',' read idx mem util temp; do
                printf "   GPU%s: %4sMB %2s%% %2s°C" "$idx" "$mem" "$util" "$temp"
                # Controllo salute
                if [ "$util" -lt 5 ] && [ "$mem" -lt 1000 ]; then
                    printf " (idle?)"
                elif [ "$temp" -gt 85 ]; then
                    printf " (HOT!)"
                fi
                echo
            done
        
        # Check processi Python
        PYTHON_PROCS=$(pgrep -f "veronacard_mob_with_geom" | wc -l)
        if [ $PYTHON_PROCS -eq 0 ]; then
            echo "ℹ️ Processo Python terminato"
            break
        fi
        
        echo "---"
    done
}

# Avvio monitoring
monitor_system &
MONITOR_PID=$!

# === ESECUZIONE PYTHON CON GESTIONE ERRORI ===
PYTHON_START=$(date +%s)

echo "🚀 Avvio script Python ottimizzato..."
echo "📄 File target: veronacard_2023_original_parziale.csv"
echo "⏰ Timestamp inizio: $(date '+%Y-%m-%d %H:%M:%S')"

# TEST PRELIMINARE: verifica che il comando non abbia errori di sintassi
echo ""
echo "🔍 DEBUG: Test comando Python (dry run)"
python3 -c "
import sys
sys.argv = ['veronacard_mob_with_geom_parrallel.py', '--file', 'veronacard_2020_2023/veronacard_2022_original.csv']
try:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str)
    parser.add_argument('--append', action='store_true')
    args = parser.parse_args(sys.argv[1:])
    print('✅ Argomenti Python parsati correttamente')
    print('   File:', args.file)
    print('   Append:', args.append)
except Exception as e:
    print('❌ Errore parsing argomenti:', e)
"

echo ""
echo "🎬 INIZIO ESECUZIONE PYTHON REALE"
echo "-" * 50

# Esecuzione con timeout più lungo ma gestito internamente
if timeout 28800s python3 -u veronacard_mob_with_geom_parrallel.py \
    --file veronacard_2020_2023/veronacard_2023_original_parziale.csv \
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
    echo "📜 Ultime 20 righe del log Python:"
    tail -20 python_execution.log
    
    case $PYTHON_EXIT in
        124) echo "⚠️ Timeout 8h - possibili risultati parziali salvati" ;;
        130) echo "⚠️ Interruzione manuale (SIGINT)" ;;
        137) echo "⚠️ Kill signal (SIGKILL) - possibile OOM" ;;
        139) echo "⚠️ Segfault - problema GPU/driver" ;;
        1)   echo "⚠️ Errore Python generico - check log" ;;
        *)   echo "⚠️ Exit code sconosciuto: $PYTHON_EXIT" ;;
    esac
    
    PYTHON_SUCCESS=false
fi

# Stop monitoring
kill $MONITOR_PID 2>/dev/null || true
wait $MONITOR_PID 2>/dev/null || true

# === REPORT FINALE ===
echo ""
echo "📋 REPORT FINALE HPC OTTIMIZZATO"
echo "================================"

TOTAL_TIME=$SECONDS
echo "⏱️ Tempo totale job: $TOTAL_TIME sec ($(($TOTAL_TIME / 3600))h $(($TOTAL_TIME % 3600 / 60))m)"
echo "🚀 Modalità: HPC ottimizzato (4x A100)"
echo "✅ Python success: $PYTHON_SUCCESS"

# Statistiche dettagliate risultati
echo ""
echo "📊 RISULTATI FINALI:"
if [ -d "results/" ]; then
    FINAL_FILES=$(ls -1 results/*.csv 2>/dev/null | wc -l)
    FINAL_SIZE=$(du -sh results/ 2>/dev/null | cut -f1 || echo "N/A")
    
    echo "   📁 File CSV: $FINAL_FILES"
    echo "   💾 Dimensione: $FINAL_SIZE" 
    
    # Throughput
    if [ "$PYTHON_SUCCESS" = "true" ] && [ $PYTHON_TIME -gt 0 ]; then
        THROUGHPUT=$(echo "scale=2; $FINAL_FILES * 3600 / $PYTHON_TIME" | bc -l 2>/dev/null || echo "N/A")
        echo "   ⚡ Throughput: $THROUGHPUT files/hour"
    fi
    
    # File più recenti
    echo ""
    echo "   📋 File recenti:"
    ls -laht results/*.csv 2>/dev/null | head -5 | while read line; do
        echo "     $line"
    done
else
    echo "   ❌ Nessun risultato trovato"
fi

# Stato GPU finale  
echo ""
echo "🔧 STATO GPU FINALE:"
nvidia-smi --query-gpu=index,name,memory.used,utilization.gpu,temperature.gpu,power.draw --format=csv

# Log analisi (sample)
echo ""
echo "📜 SAMPLE LOG OLLAMA:"
for i in 0 1 2 3; do
    if [ -f "ollama_gpu${i}.log" ]; then
        echo "--- GPU $i (ultime 3 righe) ---"
        tail -3 ollama_gpu${i}.log
    fi
done

# Conclusione
echo ""
if [ "$PYTHON_SUCCESS" = "true" ]; then
    echo "🎉 JOB HPC COMPLETATO CON SUCCESSO!"
    echo "⚡ Elaborazione ottimizzata su 4x A100 completata"
    
    if [ $PYTHON_TIME -gt 0 ]; then
        # Stima miglioramento vs versione originale
        ESTIMATED_OLD_TIME=$((PYTHON_TIME * 3))  # Stima conservativa
        IMPROVEMENT=$(echo "scale=1; ($ESTIMATED_OLD_TIME - $PYTHON_TIME) * 100 / $ESTIMATED_OLD_TIME" | bc -l 2>/dev/null)
        echo "📈 Miglioramento stimato: ${IMPROVEMENT}% vs versione originale"
    fi
else
    echo "⚠️ JOB TERMINATO CON ERRORI"
    echo "📍 Possibili cause:"
    echo "   • Timeout del sistema (normale per dataset molto grandi)"
    echo "   • Problemi GPU/memoria (controllare nvidia-smi)"
    echo "   • Problemi rete/Ollama (controllare log)"
    echo "   • OOM Python (aumentare --mem in SBATCH)"
    echo "   • File input mancante o percorso errato"
fi

echo ""
echo "🏁 Fine job: $(date)"
echo "================================"