#!/bin/bash
#SBATCH --job-name=ollama-test-fixed
#SBATCH --account=IscrC_LLM-Mob
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=test-fixed-%j.out

echo "🔧 TEST DIAGNOSTICO CORRETTO"
echo "============================"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodo: $(hostname)"
echo "Data: $(date)"

# Carica moduli
module purge
module load python/3.11.6--gcc--8.5.0
module load cuda/12.3
source $SLURM_SUBMIT_DIR/LLM/bin/activate

# Configurazioni base
export CUDA_VISIBLE_DEVICES=0
export OLLAMA_GPU_OVERHEAD=0
export OLLAMA_HOST_GPU=1
export OLLAMA_DEBUG=1

MODEL_PATH="/leonardo/home/userexternal/smattiol/.ollama/models/blobs/sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29"
OLLAMA_PORT=39005  # Porta diversa
OLLAMA_BIN="$HOME/opt/ollama/bin/ollama"

export OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT
export OLLAMA_MODELS=$HOME/.ollama/models

echo $OLLAMA_PORT > ollama_port_test.txt

# Pulisci processi precedenti
pkill -f "ollama serve" 2>/dev/null || true
sleep 3

# Avvia server
echo "🚀 Avvio server su porta $OLLAMA_PORT..."
$OLLAMA_BIN serve > ollama_test.log 2>&1 &
SERVER_PID=$!

cleanup() {
    echo "🧹 Cleanup..."
    if kill -0 $SERVER_PID 2>/dev/null; then
        kill $SERVER_PID 2>/dev/null
        sleep 2
        kill -9 $SERVER_PID 2>/dev/null
    fi
}
trap cleanup EXIT

# Attesa server
echo "⏳ Attesa server..."
for i in {1..20}; do
    if curl -s --connect-timeout 2 "http://127.0.0.1:$OLLAMA_PORT/api/tags" >/dev/null 2>&1; then
        echo "✅ Server attivo dopo $((i * 3))s"
        break
    fi
    sleep 3
done

# Verifica server
if ! curl -s "http://127.0.0.1:$OLLAMA_PORT/api/tags" >/dev/null 2>&1; then
    echo "❌ Server non risponde"
    exit 1
fi

# CORREZIONE: Crea Modelfile senza problemi di escape
echo "🔨 Creazione modello CORRETTA..."
MODEL_NAME="llama3.1:8b-test"

# Scrive il Modelfile direttamente senza escape problematici
cat > /tmp/Modelfile_test << 'EOF'
FROM /leonardo/home/userexternal/smattiol/.ollama/models/blobs/sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29

PARAMETER num_ctx 1024
PARAMETER num_batch 64
PARAMETER num_gpu 20
PARAMETER num_thread 4
PARAMETER temperature 0.1
PARAMETER top_k 10
PARAMETER top_p 0.9

TEMPLATE """<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
EOF

echo "📄 Contenuto Modelfile:"
cat /tmp/Modelfile_test
echo ""

# Metodo alternativo: usa file invece di inline JSON
echo "Creando modello tramite file..."
if curl -X POST "http://127.0.0.1:$OLLAMA_PORT/api/create" \
       -H "Content-Type: application/json" \
       -d @- <<EOF
{
  "name": "$MODEL_NAME",
  "modelfile": "$(cat /tmp/Modelfile_test | sed 's/"/\\"/g' | tr '\n' '\\n')"
}
EOF
then
    echo "✅ Creazione modello OK"
else
    echo "❌ Creazione modello FAIL"
    echo "--- LOG SERVER ---"
    tail -10 ollama_test.log
fi

# Verifica modelli disponibili
echo "📋 Modelli disponibili:"
curl -s "http://127.0.0.1:$OLLAMA_PORT/api/tags" | jq '.models[]?.name // empty' 2>/dev/null || echo "Nessun modello o errore JSON"

# Test con modello esistente (se c'è)
echo "🧪 Test con modelli esistenti..."

# Prima prova a vedere se esiste già llama3.1:8b
existing_models=$(curl -s "http://127.0.0.1:$OLLAMA_PORT/api/tags" | jq -r '.models[]?.name // empty' 2>/dev/null)
echo "Modelli trovati: $existing_models"

# Test con il primo modello disponibile
if echo "$existing_models" | grep -q "llama3.1:8b"; then
    TEST_MODEL="llama3.1:8b"
elif echo "$existing_models" | grep -q "$MODEL_NAME"; then
    TEST_MODEL="$MODEL_NAME"
else
    echo "⚠️ Nessun modello utilizzabile trovato"
    echo "Provo a creare modello base..."
    
    # Crea modello base senza parametri custom
    if curl -X POST "http://127.0.0.1:$OLLAMA_PORT/api/create" \
           -H "Content-Type: application/json" \
           -d "{\"name\": \"llama3.1:8b\", \"modelfile\": \"FROM $MODEL_PATH\"}" \
           --max-time 300; then
        TEST_MODEL="llama3.1:8b"
        echo "✅ Modello base creato"
    else
        echo "❌ Impossibile creare modello"
        exit 1
    fi
fi

echo "🎯 Test con modello: $TEST_MODEL"

# Test progressivi
echo "Test 1: Micro (1 token)"
start_time=$(date +%s)
response=$(curl -X POST "http://127.0.0.1:$OLLAMA_PORT/api/generate" \
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
    --max-time 60 --silent 2>/dev/null)
end_time=$(date +%s)
duration=$((end_time - start_time))

echo "Risposta (${duration}s): $response"
if echo "$response" | jq -e '.done' >/dev/null 2>&1; then
    response_text=$(echo "$response" | jq -r '.response // ""')
    echo "✅ Micro test OK: '$response_text'"
    
    # Test 2: Solo se micro OK
    echo "Test 2: Piccolo (5 token)"
    start_time=$(date +%s)
    response=$(curl -X POST "http://127.0.0.1:$OLLAMA_PORT/api/generate" \
        -H "Content-Type: application/json" \
        -d "{
            \"model\": \"$TEST_MODEL\",
            \"prompt\": \"Hello world\",
            \"stream\": false,
            \"options\": {
                \"num_predict\": 5,
                \"temperature\": 0.1
            }
        }" \
        --max-time 90 --silent 2>/dev/null)
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    
    echo "Risposta (${duration}s): $response"
    if echo "$response" | jq -e '.done' >/dev/null 2>&1; then
        response_text=$(echo "$response" | jq -r '.response // ""')
        echo "✅ Test piccolo OK: '$response_text'"
        
        # Test 3: Test JSON response
        echo "Test 3: JSON prompt"
        start_time=$(date +%s)
        response=$(curl -X POST "http://127.0.0.1:$OLLAMA_PORT/api/generate" \
            -H "Content-Type: application/json" \
            -d "{
                \"model\": \"$TEST_MODEL\",
                \"prompt\": \"Answer as JSON with prediction array: suggest 2 tourist attractions in Rome\",
                \"stream\": false,
                \"options\": {
                    \"num_predict\": 30,
                    \"temperature\": 0.1
                }
            }" \
            --max-time 120 --silent 2>/dev/null)
        end_time=$(date +%s)
        duration=$((end_time - start_time))
        
        echo "Risposta JSON (${duration}s): $response"
        if echo "$response" | jq -e '.done' >/dev/null 2>&1; then
            response_text=$(echo "$response" | jq -r '.response // ""')
            echo "✅ Test JSON OK: '$response_text'"
            echo "🎉 TUTTI I TEST SUPERATI!"
        else
            echo "⚠️ Test JSON parziale"
        fi
    else
        echo "❌ Test piccolo FAIL"
    fi
else
    echo "❌ Micro test FAIL"
fi

# Statistiche finali
echo ""
echo "📊 STATISTICHE FINALI"
echo "GPU:"
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader

echo "Processi Ollama:"
ps aux | grep ollama | grep -v grep

echo "Modelli finali:"
curl -s "http://127.0.0.1:$OLLAMA_PORT/api/tags" | jq '.models[]?.name // empty' 2>/dev/null

echo ""
echo "🏁 TEST COMPLETATO"