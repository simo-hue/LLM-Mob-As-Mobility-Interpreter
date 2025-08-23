#!/bin/bash
#SBATCH --job-name=test
#SBATCH --account=IscrC_LLM-Mob
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=normal
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G

echo "🧪 TEST STABILITÀ OLLAMA MULTI-GPU"
echo "=================================="

# Leggi configurazione porte
if [ ! -f "ollama_ports.txt" ]; then
    echo "❌ File ollama_ports.txt non trovato"
    exit 1
fi

PORTS=$(cat ollama_ports.txt)
IFS=',' read -ra PORT_ARRAY <<< "$PORTS"

MODEL_NAME="mixtral:8x7b"

# Test di carico progressivo
test_gpu_load() {
    local port=$1
    local gpu_id=$2
    local num_requests=$3
    
    echo "🔥 Test carico GPU $gpu_id: $num_requests richieste simultanee"
    
    # Crea richieste in background
    for i in $(seq 1 $num_requests); do
        timeout 60s curl -s -X POST "http://127.0.0.1:$port/api/generate" \
            -H "Content-Type: application/json" \
            -d "{\"model\":\"$MODEL_NAME\",\"prompt\":\"Test $i\",\"stream\":false,\"options\":{\"num_predict\":3}}" \
            > "/tmp/test_${gpu_id}_${i}.json" 2>&1 &
    done
    
    # Aspetta completamento
    wait
    
    # Conta successi
    local successes=0
    for i in $(seq 1 $num_requests); do
        if grep -q '"done":true' "/tmp/test_${gpu_id}_${i}.json" 2>/dev/null; then
            ((successes++))
        fi
        rm -f "/tmp/test_${gpu_id}_${i}.json"
    done
    
    echo "   Risultato GPU $gpu_id: $successes/$num_requests successi"
    
    if [ $successes -eq $num_requests ]; then
        return 0
    else
        return 1
    fi
}

# Test progressivo di carico
WORKING_GPUS=0

for i in "${!PORT_ARRAY[@]}"; do
    port=${PORT_ARRAY[$i]}
    echo "🎯 Test GPU $i (porta $port)"
    
    # Test singola richiesta
    if test_gpu_load $port $i 1; then
        echo "✅ GPU $i: test base OK"
        
        # Test carico moderato
        if test_gpu_load $port $i 2; then
            echo "✅ GPU $i: test carico OK"
            ((WORKING_GPUS++))
        else
            echo "⚠️ GPU $i: fallisce sotto carico"
        fi
    else
        echo "❌ GPU $i: fallisce test base"
    fi
    
    sleep 5
done

echo ""
echo "📊 RISULTATO TEST: $WORKING_GPUS/${#PORT_ARRAY[@]} GPU stabili"

if [ $WORKING_GPUS -ge 2 ]; then
    echo "✅ Sistema pronto per elaborazione produzione"
    exit 0
else
    echo "❌ Sistema instabile - riavvio necessario"
    exit 1
fi