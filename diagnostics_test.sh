#!/bin/bash
#SBATCH --job-name=ollama-diagnostics
#SBATCH --account=IscrC_LLM-Mob
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=ollama-diag-%j.out

echo "🔧 DIAGNOSTICA OLLAMA COMPLETA"
echo "=============================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodo: $(hostname)"
echo "Data: $(date)"
echo "PWD: $(pwd)"

# Carica moduli
echo "📦 Caricamento moduli..."
module purge
module load python/3.11.6--gcc--8.5.0
module load cuda/12.3
source $SLURM_SUBMIT_DIR/LLM/bin/activate

# Verifica ambiente
echo ""
echo "🔍 VERIFICA AMBIENTE"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "SLURM_GPUS: $SLURM_GPUS"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
nvidia-smi --list-gpus
echo "GPU Memory:"
nvidia-smi --query-gpu=memory.total,memory.free,memory.used --format=csv,units=MiB

# Configurazioni
export CUDA_VISIBLE_DEVICES=0
export OLLAMA_GPU_OVERHEAD=0
export OLLAMA_HOST_GPU=1
export OLLAMA_DEBUG=1
export OLLAMA_FLASH_ATTENTION=0

MODEL_PATH="/leonardo/home/userexternal/smattiol/.ollama/models/blobs/sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29"
OLLAMA_PORT=39005
OLLAMA_BIN="$HOME/opt/ollama/bin/ollama"

export OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT
export OLLAMA_MODELS=$HOME/.ollama/models

echo "OLLAMA_BIN: $OLLAMA_BIN"
echo "OLLAMA_MODELS: $OLLAMA_MODELS"
echo "OLLAMA_HOST: $OLLAMA_HOST"

# Verifica file e directory
echo ""
echo "📁 VERIFICA FILE E DIRECTORY"
echo "Ollama binary exists: $(test -f "$OLLAMA_BIN" && echo "✅ SI" || echo "❌ NO")"
echo "Models directory exists: $(test -d "$OLLAMA_MODELS" && echo "✅ SI" || echo "❌ NO")"
echo "Model blob exists: $(test -f "$MODEL_PATH" && echo "✅ SI" || echo "❌ NO")"

if [ -d "$OLLAMA_MODELS" ]; then
    echo "Contenuto directory modelli:"
    find "$OLLAMA_MODELS" -type f -name "*.json" | head -10
    echo "Blobs disponibili:"
    ls -la "$OLLAMA_MODELS/blobs/" | head -10
fi

# Salva porta per riferimento
echo $OLLAMA_PORT > ollama_port_diag.txt

# Pulisci processi precedenti
echo ""
echo "🧹 PULIZIA PROCESSI PRECEDENTI"
pkill -f "ollama serve" 2>/dev/null || true
sleep 3

# Verifica porte occupate
echo "Porte in uso (39000-39010):"
for port in {39000..39010}; do
    if netstat -ln | grep ":$port " >/dev/null 2>&1; then
        echo "  Porta $port: OCCUPATA"
    fi
done

# Avvia server con logging dettagliato
echo ""
echo "🚀 AVVIO SERVER OLLAMA"
echo "Comando: $OLLAMA_BIN serve"
$OLLAMA_BIN serve > ollama_diagnostic.log 2>&1 &
SERVER_PID=$!
echo "PID Server: $SERVER_PID"

# Cleanup function
cleanup() {
    echo ""
    echo "🧹 CLEANUP FINALE..."
    if kill -0 $SERVER_PID 2>/dev/null; then
        echo "Terminando server PID $SERVER_PID"
        kill $SERVER_PID 2>/dev/null
        sleep 3
        kill -9 $SERVER_PID 2>/dev/null || true
    fi
    echo "Log finale:"
    tail -20 ollama_diagnostic.log 2>/dev/null || echo "Nessun log disponibile"
}
trap cleanup EXIT

# Attesa server con controlli più dettagliati
echo ""
echo "⏳ ATTESA E VERIFICA SERVER"
for i in {1..30}; do
    echo -n "Tentativo $i/30... "
    
    # Test connessione TCP
    if timeout 2 bash -c "echo >/dev/tcp/127.0.0.1/$OLLAMA_PORT" 2>/dev/null; then
        echo "TCP OK"
        
        # Test API health
        if curl -s --connect-timeout 3 --max-time 5 "http://127.0.0.1:$OLLAMA_PORT/api/tags" >/dev/null 2>&1; then
            echo "✅ Server API attivo dopo $((i * 2))s"
            break
        else
            echo "TCP OK ma API non risponde"
        fi
    else
        echo "TCP fallito"
    fi
    
    sleep 2
    
    # Mostra log ogni 10 tentativi
    if [ $((i % 10)) -eq 0 ]; then
        echo "--- Ultimi log del server ---"
        tail -5 ollama_diagnostic.log 2>/dev/null || echo "Nessun log"
        echo "--- Fine log ---"
    fi
done

# Verifica finale server
echo ""
echo "🔍 VERIFICA FINALE SERVER"
if curl -s --max-time 10 "http://127.0.0.1:$OLLAMA_PORT/api/tags" >/dev/null 2>&1; then
    echo "✅ Server risponde alle API"
else
    echo "❌ Server non risponde alle API"
    echo "--- LOG COMPLETO SERVER ---"
    cat ollama_diagnostic.log 2>/dev/null || echo "Nessun log disponibile"
    echo "--- FINE LOG ---"
    
    echo "--- PROCESSI OLLAMA ---"
    ps aux | grep ollama | grep -v grep
    echo "--- FINE PROCESSI ---"
    
    exit 1
fi

# LISTA MODELLI DISPONIBILI
echo ""
echo "📋 MODELLI DISPONIBILI"
echo "======================"

models_response=$(curl -s --max-time 10 "http://127.0.0.1:$OLLAMA_PORT/api/tags" 2>/dev/null)
if [ $? -eq 0 ] && echo "$models_response" | jq . >/dev/null 2>&1; then
    echo "Risposta API completa:"
    echo "$models_response" | jq . 2>/dev/null || echo "$models_response"
    
    echo ""
    echo "Modelli estratti:"
    model_names=$(echo "$models_response" | jq -r '.models[]?.name // empty' 2>/dev/null)
    if [ -n "$model_names" ]; then
        echo "$model_names" | while read -r model; do
            echo "  - $model"
        done
        
        # Conta modelli
        model_count=$(echo "$model_names" | wc -l)
        echo ""
        echo "Totale modelli: $model_count"
    else
        echo "  Nessun modello trovato nell'API"
    fi
else
    echo "❌ Errore nel recuperare lista modelli"
    echo "Risposta grezza: $models_response"
fi

# CREAZIONE MODELLO DI TEST (solo se non esistono modelli)
if [ -z "$model_names" ] || ! echo "$model_names" | grep -q "llama3.1"; then
    echo ""
    echo "🔨 CREAZIONE MODELLO DI TEST"
    echo "============================="
    
    MODEL_NAME="llama3.1:8b-test"
    
    # Crea Modelfile minimo
    cat > /tmp/Modelfile_test << EOF
FROM $MODEL_PATH
PARAMETER num_ctx 512
PARAMETER temperature 0.1
EOF

    echo "Modelfile creato:"
    cat /tmp/Modelfile_test
    
    echo ""
    echo "Creando modello '$MODEL_NAME'..."
    
    # Prepara JSON per creazione modello
    modelfile_content=$(cat /tmp/Modelfile_test | sed 's/"/\\"/g' | tr '\n' ' ')
    
    create_response=$(curl -X POST "http://127.0.0.1:$OLLAMA_PORT/api/create" \
           -H "Content-Type: application/json" \
           -d "{\"name\": \"$MODEL_NAME\", \"modelfile\": \"$modelfile_content\"}" \
           --max-time 300 -s 2>/dev/null)
    
    if [ $? -eq 0 ]; then
        echo "✅ Risposta creazione: $create_response"
        
        # Verifica modelli aggiornati
        sleep 2
        updated_models=$(curl -s "http://127.0.0.1:$OLLAMA_PORT/api/tags" | jq -r '.models[]?.name // empty' 2>/dev/null)
        echo "Modelli dopo creazione:"
        echo "$updated_models"
    else
        echo "❌ Errore nella creazione del modello"
    fi
fi

# TEST FUNZIONALITA
echo ""
echo "🧪 TEST FUNZIONALITÀ"
echo "===================="

# Ottieni primo modello disponibile
available_models=$(curl -s "http://127.0.0.1:$OLLAMA_PORT/api/tags" | jq -r '.models[]?.name // empty' 2>/dev/null)
if [ -n "$available_models" ]; then
    TEST_MODEL=$(echo "$available_models" | head -1)
    echo "Usando modello per test: $TEST_MODEL"
    
    # Test minimo
    echo ""
    echo "Test 1: Risposta minima"
    test_response=$(curl -X POST "http://127.0.0.1:$OLLAMA_PORT/api/generate" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$TEST_MODEL\",
            \"prompt\": \"Hi\",
            \"stream\": false,
            \"options\": {
                \"num_predict\": 1,
                \"temperature\": 0
            }
        }" \
        --max-time 60 -s 2>/dev/null)
    
    echo "Risposta grezza: $test_response"
    
    if echo "$test_response" | jq -e '.done' >/dev/null 2>&1; then
        response_text=$(echo "$test_response" | jq -r '.response // ""')
        echo "✅ Test OK - Risposta: '$response_text'"
    else
        echo "❌ Test fallito"
    fi
else
    echo "⚠️ Nessun modello disponibile per i test"
fi

# STATISTICHE FINALI
echo ""
echo "📊 STATISTICHE FINALI"
echo "====================="
echo "GPU Status:"
nvidia-smi --query-gpu=memory.used,utilization.gpu,temperature.gpu --format=csv,noheader

echo ""
echo "Processi Ollama attivi:"
ps aux | grep ollama | grep -v grep

echo ""
echo "Spazio disco modelli:"
du -sh "$OLLAMA_MODELS" 2>/dev/null || echo "Directory non accessibile"

echo ""
echo "🏁 DIAGNOSTICA COMPLETATA"
echo "Log salvato in: ollama_diagnostic.log"
echo "Porta utilizzata: $OLLAMA_PORT (salvata in ollama_port_diag.txt)"