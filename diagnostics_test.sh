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
#SBATCH --output=diagnostics-%j.out

echo "🔍 DIAGNOSTICA OLLAMA E GPU"
echo "=========================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodo: $(hostname)"
echo "Data: $(date)"

# Carica moduli
module purge
module load python/3.11.6--gcc--8.5.0
module load cuda/12.3
source $SLURM_SUBMIT_DIR/LLM/bin/activate

# Info GPU dettagliata
echo "🔍 INFO GPU DETTAGLIATA:"
nvidia-smi
echo ""
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo ""

# Configura ambiente
export CUDA_VISIBLE_DEVICES=0
export OLLAMA_GPU_OVERHEAD=0
export OLLAMA_HOST_GPU=1
export OLLAMA_DEBUG=1
export OLLAMA_FLASH_ATTENTION=0  # Disabilita per test
export OLLAMA_KV_CACHE_TYPE=f16

# Variabili
MODEL_PATH="/leonardo/home/userexternal/smattiol/.ollama/models/blobs/sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29"
OLLAMA_PORT=39003  # Porta diversa
OLLAMA_BIN="$HOME/opt/ollama/bin/ollama"

export OLLAMA_HOST=127.0.0.1:$OLLAMA_PORT
export OLLAMA_MODELS=$HOME/.ollama/models
export OLLAMA_KEEP_ALIVE=15m
export OLLAMA_MAX_LOADED_MODELS=1
export OLLAMA_NUM_PARALLEL=1
export OLLAMA_LOAD_TIMEOUT=10m
export OLLAMA_REQUEST_TIMEOUT=120s

echo $OLLAMA_PORT > ollama_port_test.txt

# Test 1: Verifica che il modello esista
echo "📁 TEST 1: Verifica modello"
if [ -f "$MODEL_PATH" ]; then
    echo "✅ Modello trovato: $(du -h $MODEL_PATH)"
else
    echo "❌ Modello non trovato!"
    exit 1
fi

# Pulisci processi precedenti
pkill -f "ollama serve" 2>/dev/null || true
sleep 3

# Test 2: Avvio server con log dettagliato
echo "🚀 TEST 2: Avvio server"
$OLLAMA_BIN serve > ollama_diagnostic.log 2>&1 &
SERVER_PID=$!
echo "Server PID: $SERVER_PID"

# Cleanup function
cleanup() {
    echo "🧹 Cleanup..."
    if kill -0 $SERVER_PID 2>/dev/null; then
        kill $SERVER_PID 2>/dev/null
        sleep 2
        kill -9 $SERVER_PID 2>/dev/null
    fi
}
trap cleanup EXIT

# Test 3: Attesa server con monitoraggio
echo "⏳ TEST 3: Attesa server (max 60s)"
for i in {1..20}; do
    if curl -s --connect-timeout 2 "http://127.0.0.1:$OLLAMA_PORT/api/tags" >/dev/null 2>&1; then
        echo "✅ Server attivo dopo $((i * 3))s"
        break
    fi
    
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo "❌ Server morto!"
        echo "--- LOG ---"
        cat ollama_diagnostic.log
        exit 1
    fi
    
    if [ $((i % 5)) -eq 0 ]; then
        echo "   Attesa... ($((i * 3))s)"
        echo "   GPU Memory:"
        nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
    fi
    
    sleep 3
done

# Test 4: Crea modello minimal
echo "🔨 TEST 4: Creazione modello minimal"
cat > /tmp/Modelfile_minimal << EOF
FROM $MODEL_PATH

# Parametri minimi per test
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

MODEL_NAME="llama3.1:8b-minimal"

echo "Creando modello $MODEL_NAME..."
if timeout 300 curl -X POST "http://127.0.0.1:$OLLAMA_PORT/api/create" \
       -H "Content-Type: application/json" \
       -d "{\"name\": \"$MODEL_NAME\", \"modelfile\": \"$(cat /tmp/Modelfile_minimal | tr '\n' '\\n')\"}" \
       --show-error -v; then
    echo "✅ Modello creato"
else
    echo "❌ Errore creazione modello"
    echo "--- SERVER LOG ---"
    tail -20 ollama_diagnostic.log
fi

# Test 5: Test inferenza progressivi
echo "🧪 TEST 5: Test inferenza progressivi"

# Test 5a: Micro test (1 token)
echo "Test 5a: Micro (1 token)"
start_time=$(date +%s)
response=$(curl -X POST "http://127.0.0.1:$OLLAMA_PORT/api/generate" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'$MODEL_NAME'",
        "prompt": "Hi",
        "stream": false,
        "options": {
            "num_predict": 1,
            "temperature": 0,
            "num_ctx": 512
        }
    }' \
    --max-time 60 --silent)
end_time=$(date +%s)
duration=$((end_time - start_time))

if echo "$response" | jq -e '.done' >/dev/null 2>&1; then
    response_text=$(echo "$response" | jq -r '.response // ""')
    echo "✅ Micro test OK (${duration}s): '$response_text'"
else
    echo "❌ Micro test FAIL (${duration}s): $response"
fi

# Test 5b: Test piccolo (5 token)
echo "Test 5b: Piccolo (5 token)"
start_time=$(date +%s)
response=$(curl -X POST "http://127.0.0.1:$OLLAMA_PORT/api/generate" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'$MODEL_NAME'",
        "prompt": "Hello, how are",
        "stream": false,
        "options": {
            "num_predict": 5,
            "temperature": 0.1,
            "num_ctx": 1024
        }
    }' \
    --max-time 120 --silent)
end_time=$(date +%s)
duration=$((end_time - start_time))

if echo "$response" | jq -e '.done' >/dev/null 2>&1; then
    response_text=$(echo "$response" | jq -r '.response // ""')
    echo "✅ Test piccolo OK (${duration}s): '$response_text'"
else
    echo "❌ Test piccolo FAIL (${duration}s): $response"
fi

# Test 5c: Test normale (20 token)
echo "Test 5c: Normale (20 token)"
start_time=$(date +%s)
response=$(curl -X POST "http://127.0.0.1:$OLLAMA_PORT/api/generate" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "'$MODEL_NAME'",
        "prompt": "Write a short sentence about Rome",
        "stream": false,
        "options": {
            "num_predict": 20,
            "temperature": 0.1,
            "num_ctx": 2048
        }
    }' \
    --max-time 180 --silent)
end_time=$(date +%s)
duration=$((end_time - start_time))

if echo "$response" | jq -e '.done' >/dev/null 2>&1; then
    response_text=$(echo "$response" | jq -r '.response // ""')
    echo "✅ Test normale OK (${duration}s): '$response_text'"
else
    echo "❌ Test normale FAIL (${duration}s): $response"
fi

# Test 6: Monitoraggio risorse durante inferenza
echo "📊 TEST 6: Stato finale sistema"
echo "Processo Ollama:"
ps aux | grep ollama | grep -v grep || echo "Nessun processo ollama"

echo ""
echo "Memoria GPU finale:"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv

echo ""
echo "Spazio disco modelli:"
du -sh $HOME/.ollama/models/

echo ""
echo "Ultimi log server:"
tail -20 ollama_diagnostic.log

echo ""
echo "🏁 DIAGNOSTICA COMPLETATA"