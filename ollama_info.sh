#!/bin/bash
#SBATCH --job-name=ollama_install
#SBATCH --account=IscrC_LLM-Mob
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --output=slurm_ollama_install-%j.out

# ============= MODELLI DA INSTALLARE =============
# Aggiungi/rimuovi modelli da questa lista
MODELS_TO_INSTALL=(
    "qwen2.5:7b"
    "qwen2.5:14b"
    "deepseek-r1:32b"
    "deepseek-v3:8b" 
)

echo "🚀 OLLAMA MODELS INSTALLER"
echo "================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodo: $(hostname)"
echo "Data: $(date)"
echo "User: $(whoami)"
echo ""

echo "📋 MODELLI DA INSTALLARE:"
printf '%s\n' "${MODELS_TO_INSTALL[@]}" | sed 's/^/  - /'
echo ""

# ============= SETUP AMBIENTE =============
echo "📦 Setup ambiente HPC..."
module purge
module load python/3.11.6--gcc--8.5.0
module load cuda/12.3

echo "✅ Python: $(python3 --version)"
echo "✅ CUDA: $(nvcc --version | grep release)"

export CUDA_VISIBLE_DEVICES=0,1,2,3
export NVIDIA_VISIBLE_DEVICES=0,1,2,3

# Debug GPU iniziale
echo ""
echo "🔍 GPU DETECTION:"
nvidia-smi --query-gpu=index,name,memory.total,temperature.gpu --format=csv,noheader
echo ""

# ============= CONFIGURAZIONE OLLAMA =============
OLLAMA_BIN="/leonardo_work/IscrC_LLM-Mob/opt/bin/ollama"

if [ ! -f "$OLLAMA_BIN" ]; then
    echo "❌ ERRORE: Ollama non trovato in $OLLAMA_BIN"
    exit 1
fi

export OLLAMA_DEBUG=1
export OLLAMA_MODELS="$WORK/.ollama/models"
export OLLAMA_CACHE_DIR="$WORK/.ollama/cache"
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_KEEP_ALIVE="30m"
export OLLAMA_LLM_LIBRARY="cuda_v12"
export OLLAMA_FLASH_ATTENTION=1

# Directory temporanea
CUSTOM_TMP="$WORK/tmp_ollama_install_$SLURM_JOB_ID"
mkdir -p "$CUSTOM_TMP"
chmod 700 "$CUSTOM_TMP"
export OLLAMA_TMPDIR="$CUSTOM_TMP"

echo "🔧 CONFIGURAZIONE:"
echo "- Ollama binary: $OLLAMA_BIN"
echo "- Models directory: $OLLAMA_MODELS"
echo "- Cache directory: $OLLAMA_CACHE_DIR"
echo "- Temp directory: $CUSTOM_TMP"
echo ""

# ============= VERIFICA SPAZIO DISCO =============
echo "💾 VERIFICA SPAZIO DISCO"
echo "========================"

WORK_AVAILABLE=$(df "$WORK" | tail -1 | awk '{print $4}')
WORK_AVAILABLE_GB=$((WORK_AVAILABLE / 1024 / 1024))
echo "Spazio disponibile: ${WORK_AVAILABLE_GB}GB"

# Stima spazio necessario (approssimativo)
ESTIMATED_SPACE_GB=200  # Stima per tutti i modelli
if [ $WORK_AVAILABLE_GB -lt $ESTIMATED_SPACE_GB ]; then
    echo "⚠️ ATTENZIONE: Spazio potenzialmente insufficiente"
    echo "   Disponibile: ${WORK_AVAILABLE_GB}GB"
    echo "   Stimato necessario: ${ESTIMATED_SPACE_GB}GB"
    echo "   Continuando comunque..."
fi

# Spazio attualmente usato da Ollama
if [ -d "$OLLAMA_MODELS" ]; then
    CURRENT_OLLAMA_SIZE=$(du -sh "$OLLAMA_MODELS" 2>/dev/null | cut -f1 || echo "N/A")
    echo "Spazio attuale Ollama: $CURRENT_OLLAMA_SIZE"
fi
echo ""

# ============= CLEANUP PREVENTIVO =============
echo "🧹 Cleanup preventivo..."
pkill -f ollama 2>/dev/null || true
sleep 10

# ============= VARIABILI GLOBALI =============
SERVER_PID=""
OLLAMA_HOST="127.0.0.1:39000"
INSTALLATION_LOG="ollama_installation_detailed.log"

# ============= FUNZIONE CLEANUP =============
cleanup() {
    echo ""
    echo "🧹 Cleanup finale..."
    
    if [ -n "$SERVER_PID" ] && kill -0 $SERVER_PID 2>/dev/null; then
        echo "Stopping Ollama server (PID $SERVER_PID)..."
        kill -TERM $SERVER_PID 2>/dev/null
        sleep 10
        kill -KILL $SERVER_PID 2>/dev/null || true
    fi
    
    pkill -f ollama 2>/dev/null || true
    
    if [ -n "$CUSTOM_TMP" ] && [ -d "$CUSTOM_TMP" ]; then
        echo "Removing temporary directory..."
        rm -rf "$CUSTOM_TMP"
    fi
    
    echo "✅ Cleanup completato"
}
trap cleanup EXIT

# ============= FUNZIONE AVVIO SERVER =============
start_ollama_server() {
    echo "🚀 Avvio server Ollama..."
    
    # Crea directory cache se necessaria
    mkdir -p "$OLLAMA_CACHE_DIR"
    
    # Avvia server
    OLLAMA_HOST="$OLLAMA_HOST" \
    CUDA_VISIBLE_DEVICES=0 \
    $OLLAMA_BIN serve > ollama_server.log 2>&1 &
    SERVER_PID=$!
    
    echo "Server PID: $SERVER_PID"
    
    # Verifica che il processo sia vivo
    sleep 5
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "❌ Server fallito all'avvio!"
        echo "Log server:"
        cat ollama_server.log 2>/dev/null || echo "Log non disponibile"
        return 1
    fi
    
    # Attendi che l'API sia disponibile
    echo "⏳ Attesa disponibilità API..."
    for i in {1..60}; do
        if curl -s --connect-timeout 3 "http://$OLLAMA_HOST/api/version" >/dev/null 2>&1; then
            echo "✅ Server pronto dopo $i tentativi!"
            return 0
        fi
        echo "  Tentativo $i/60..."
        sleep 5
    done
    
    echo "❌ Server non risponde dopo 5 minuti"
    echo "Log server (ultime 30 righe):"
    tail -30 ollama_server.log 2>/dev/null || echo "Log non disponibile"
    return 1
}

# ============= FUNZIONE LISTA MODELLI =============
list_models() {
    local title="$1"
    echo ""
    echo "📋 $title"
    echo "$(echo "$title" | sed 's/./-/g')"
    
    # Usa ollama list tramite il server
    local list_output
    list_output=$(OLLAMA_HOST="$OLLAMA_HOST" timeout 30s $OLLAMA_BIN list 2>&1)
    local exit_code=$?
    
    if [ $exit_code -eq 0 ]; then
        echo "$list_output"
        
        # Conta i modelli
        local model_count=$(echo "$list_output" | grep -v "NAME" | grep -v "^$" | wc -l)
        echo ""
        echo "📊 Totale modelli: $model_count"
    else
        echo "❌ Errore nell'ottenere la lista:"
        echo "$list_output"
        
        echo ""
        echo "🔄 Tentativo alternativo via API..."
        local api_response
        api_response=$(curl -s --connect-timeout 10 --max-time 30 "http://$OLLAMA_HOST/api/tags" 2>&1)
        if [ $? -eq 0 ]; then
            echo "Risposta API:"
            echo "$api_response" | jq -r '.models[]? | "\(.name) \(.size // "N/A") \(.modified_at // "N/A")"' 2>/dev/null || echo "$api_response"
        else
            echo "❌ Anche API fallita: $api_response"
        fi
    fi
    echo ""
}

# ============= FUNZIONE INSTALLAZIONE MODELLO =============
install_model() {
    local model="$1"
    local start_time=$(date +%s)
    
    echo ""
    echo "🔽 INSTALLAZIONE: $model"
    echo "$(echo "🔽 INSTALLAZIONE: $model" | sed 's/./-/g')"
    echo "Inizio: $(date)"
    
    # Log dettagliato
    {
        echo ""
        echo "=== INSTALLAZIONE $model - $(date) ==="
    } >> "$INSTALLATION_LOG"
    
    # Verifica se il modello esiste già
    if OLLAMA_HOST="$OLLAMA_HOST" timeout 10s $OLLAMA_BIN list | grep -q "^$model"; then
        echo "⚠️ Modello $model già installato, salto..."
        return 0
    fi
    
    echo "📥 Download e installazione in corso..."
    echo "   (Questo può richiedere molto tempo per modelli grandi)"
    
    # Monitor progresso in background
    {
        while kill -0 $$ 2>/dev/null; do
            sleep 60
            echo "⏳ $(date): Download $model ancora in corso..."
            
            # Mostra utilizzo GPU
            echo "   GPU Status:"
            nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits | \
            while IFS=',' read -r idx mem_used mem_total; do
                mem_percent=$((mem_used * 100 / mem_total))
                printf "     GPU %s: %d%% (%d/%d MB)\n" "$idx" "$mem_percent" "$mem_used" "$mem_total"
            done
            
            # Verifica spazio disco
            local space_now=$(df "$WORK" | tail -1 | awk '{print $4}')
            local space_now_gb=$((space_now / 1024 / 1024))
            echo "   Spazio rimanente: ${space_now_gb}GB"
            
        done
    } &
    MONITOR_PID=$!
    
    # Installazione effettiva
    local install_output
    install_output=$(OLLAMA_HOST="$OLLAMA_HOST" timeout 3600s $OLLAMA_BIN pull "$model" 2>&1)
    local install_exit=$?
    
    # Stop monitor
    kill $MONITOR_PID 2>/dev/null || true
    
    # Log del risultato
    {
        echo "Exit code: $install_exit"
        echo "Output:"
        echo "$install_output"
        echo ""
    } >> "$INSTALLATION_LOG"
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    local duration_min=$((duration / 60))
    
    if [ $install_exit -eq 0 ]; then
        echo "✅ $model installato con successo!"
        echo "   Tempo impiegato: ${duration_min} minuti"
        
        # Verifica dimensione del modello installato
        if [ -d "$OLLAMA_MODELS" ]; then
            local model_size=$(find "$OLLAMA_MODELS" -name "*$(echo $model | tr ':' '_')*" -type f -exec du -sh {} + 2>/dev/null | head -1 | cut -f1 || echo "N/A")
            [ "$model_size" != "N/A" ] && echo "   Dimensione: $model_size"
        fi
        
        # Test rapido del modello
        echo "🧪 Test rapido modello..."
        local test_response
        test_response=$(OLLAMA_HOST="$OLLAMA_HOST" timeout 60s $OLLAMA_BIN run "$model" "Say 'Hello, I am working correctly!'" 2>&1)
        if [ $? -eq 0 ] && echo "$test_response" | grep -qi "hello"; then
            echo "✅ Test superato!"
        else
            echo "⚠️ Test fallito o incompleto:"
            echo "$test_response" | head -3
        fi
        
        return 0
    else
        echo "❌ $model installazione fallita!"
        echo "   Tempo impiegato: ${duration_min} minuti"
        echo "   Exit code: $install_exit"
        echo ""
        echo "Ultime righe output:"
        echo "$install_output" | tail -10 | sed 's/^/   /'
        
        return 1
    fi
}

# ============= AVVIO SERVER =============
if ! start_ollama_server; then
    echo "❌ ERRORE CRITICO: Server Ollama non avviabile"
    exit 1
fi

# ============= LISTA MODELLI INIZIALE =============
list_models "MODELLI INSTALLATI (PRIMA)"

# ============= SPAZIO DISCO PRE-INSTALLAZIONE =============
echo "💾 SPAZIO DISCO PRE-INSTALLAZIONE"
echo "================================="
df -h "$WORK"
echo ""

# ============= CICLO INSTALLAZIONE =============
SUCCESSFUL_INSTALLS=0
FAILED_INSTALLS=0
SKIPPED_INSTALLS=0

echo "🚀 INIZIO INSTALLAZIONI"
echo "======================="
echo "Totale modelli da processare: ${#MODELS_TO_INSTALL[@]}"
echo ""

for model in "${MODELS_TO_INSTALL[@]}"; do
    echo "📍 Progresso: $((SUCCESSFUL_INSTALLS + FAILED_INSTALLS + SKIPPED_INSTALLS + 1))/${#MODELS_TO_INSTALL[@]}"
    
    if install_model "$model"; then
        ((SUCCESSFUL_INSTALLS++))
    else
        ((FAILED_INSTALLS++))
        
        # Verifica se continuare dopo un fallimento
        echo ""
        echo "⚠️ Installazione $model fallita. Continuo con il prossimo..."
        
        # Verifica spazio disco dopo fallimento
        local space_check=$(df "$WORK" | tail -1 | awk '{print $4}')
        local space_check_gb=$((space_check / 1024 / 1024))
        if [ $space_check_gb -lt 10 ]; then
            echo "❌ ERRORE CRITICO: Spazio disco insufficiente (${space_check_gb}GB)"
            echo "   Interrompo le installazioni"
            break
        fi
    fi
    
    echo ""
    echo "📊 Stato attuale: ✅$SUCCESSFUL_INSTALLS ❌$FAILED_INSTALLS"
    echo ""
    
    # Pausa tra installazioni per stabilità
    if [ $((SUCCESSFUL_INSTALLS + FAILED_INSTALLS)) -lt ${#MODELS_TO_INSTALL[@]} ]; then
        echo "⏸️ Pausa 30s prima del prossimo modello..."
        sleep 30
    fi
done

# ============= LISTA MODELLI FINALE =============
echo ""
echo "🏁 INSTALLAZIONI COMPLETATE"
echo "============================"

list_models "MODELLI INSTALLATI (DOPO)"

# ============= SPAZIO DISCO POST-INSTALLAZIONE =============
echo "💾 SPAZIO DISCO POST-INSTALLAZIONE"
echo "=================================="
df -h "$WORK"
echo ""

if [ -d "$OLLAMA_MODELS" ]; then
    echo "Spazio totale utilizzato da Ollama:"
    du -sh "$OLLAMA_MODELS"
    echo ""
    
    echo "Top 10 file più grandi:"
    find "$OLLAMA_MODELS" -type f -exec du -sh {} + 2>/dev/null | sort -rh | head -10
fi

# ============= RIEPILOGO FINALE =============
echo ""
echo "📊 RIEPILOGO FINALE"
echo "==================="
echo "Modelli da installare: ${#MODELS_TO_INSTALL[@]}"
echo "✅ Installazioni riuscite: $SUCCESSFUL_INSTALLS"
echo "❌ Installazioni fallite: $FAILED_INSTALLS"
echo "⏭️ Installazioni saltate: $SKIPPED_INSTALLS"
echo "⏱️ Tempo totale: $((SECONDS / 60)) minuti"
echo ""

if [ -f "$INSTALLATION_LOG" ]; then
    echo "📜 Log dettagliato disponibile in: $INSTALLATION_LOG"
    echo "Dimensione log: $(du -sh "$INSTALLATION_LOG" 2>/dev/null | cut -f1 || echo "N/A")"
fi

echo ""
echo "🏁 JOB COMPLETATO"
echo "================="

# Exit code basato sui risultati
if [ $SUCCESSFUL_INSTALLS -eq ${#MODELS_TO_INSTALL[@]} ]; then
    echo "🎉 Tutti i modelli installati con successo!"
    exit 0
elif [ $SUCCESSFUL_INSTALLS -gt 0 ]; then
    echo "⚠️ Installazione parziale completata"
    exit 0
else
    echo "❌ Nessun modello installato"
    exit 1
fi