#!/bin/bash
#SBATCH --job-name=llm-mob
#SBATCH --account=IscrC_LLM-Mob
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg         # Debug - max 30 minuti
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=slurm-%j.out

# 1. Setup ambiente
module purge
module load python/3.11.6--gcc--8.5.0
module load cuda/12.3           
source $SLURM_SUBMIT_DIR/LLM/bin/activate

export CUDA_VISIBLE_DEVICES=0
export OLLAMA_MODEL=llama3.1:8b
export OLLAMA_GPU_OVERHEAD=0

# Trova porta libera dinamicamente
OLLAMA_PORT=$(comm -23 <(seq 30000 39999 | sort) <(ss -Htan | awk '{print $4}' | grep -o '[0-9]*$' | sort -u) | shuf | head -n 1)
export OLLAMA_PORT
export OLLAMA_HOST=http://127.0.0.1:$OLLAMA_PORT
echo $OLLAMA_PORT > $SLURM_SUBMIT_DIR/ollama_port.txt  # salvala per il Python

# 2. Check dipendenze
echo "=== VERIFICA DIPENDENZE ==="
python -c "import pandas; print('✓ pandas:', pandas.__version__)" || echo "✗ pandas mancante"
python -c "import numpy; print('✓ numpy:', numpy.__version__)" || echo "✗ numpy mancante" 
python -c "import requests; print('✓ requests:', requests.__version__)" || echo "✗ requests mancante"
python -c "import sklearn; print('✓ scikit-learn:', sklearn.__version__)" || echo "✗ scikit-learn mancante"
python -c "import tqdm; print('✓ tqdm:', tqdm.__version__)" || echo "✗ tqdm mancante"
echo "============================"

echo "=== CONFIGURAZIONE GPU ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "=========================="

# 3. Avvio Ollama
echo "Avvio Ollama server sulla porta $OLLAMA_PORT..."
ollama serve --port $OLLAMA_PORT &
OLLAMA_PID=$!

# Aspetta che il server sia attivo
for i in {1..10}; do
  if curl -s "http://127.0.0.1:$OLLAMA_PORT/api/tags" >/dev/null; then
    echo "✓ Server pronto sulla porta $OLLAMA_PORT"
    break
  else
    echo "⏳ Attesa che il server sia pronto... ($i)"
    sleep 3
  fi
done

# 4. Esegui script Python
cd $SLURM_SUBMIT_DIR
echo "Lancio script Python alle $(date)..."
python veronacard_mob_with_geom.py

# 5. Chiusura
echo "Chiusura Ollama alle $(date)..."
kill $OLLAMA_PID
wait $OLLAMA_PID 2>/dev/null
echo "✅ Job completato!"