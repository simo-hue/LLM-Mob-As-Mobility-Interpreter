#!/bin/bash
#SBATCH --job-name=llm-mob
#SBATCH --account=IscrC_LLM-Mob
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_bprod      # Produzione - fino a 24h
#SBATCH --time=02:00:00             # 2 ore iniziali
#SBATCH --nodes=1
#SBATCH --gpus=1                    # A100 64 GB
#SBATCH --cpus-per-task=32          # Minimo per bprod (invece di 16)
#SBATCH --mem=128G                  # Più memoria per bprod (invece di 64G)
#SBATCH --output=slurm-%j.out

########################
# 1. Ambiente software #
########################
module purge
module load python/3.11.6--gcc--8.5.0
module load cuda/12.3            
source $SLURM_SUBMIT_DIR/llm/bin/activate

# Verifica dipendenze critiche
echo "=== VERIFICA DIPENDENZE ==="
python -c "import pandas; print('✓ pandas:', pandas.__version__)" || echo "✗ pandas mancante"
python -c "import numpy; print('✓ numpy:', numpy.__version__)" || echo "✗ numpy mancante" 
python -c "import requests; print('✓ requests:', requests.__version__)" || echo "✗ requests mancante"
python -c "import sklearn; print('✓ scikit-learn:', sklearn.__version__)" || echo "✗ scikit-learn mancante"
python -c "import tqdm; print('✓ tqdm:', tqdm.__version__)" || echo "✗ tqdm mancante"
echo "============================"

# Variabili d'ambiente per GPU (CRITICHE!)
export CUDA_VISIBLE_DEVICES=0
export OLLAMA_HOST=127.0.0.1:11434  # Porta corretta
export OLLAMA_MODEL=llama3.1:8b
export OLLAMA_GPU_OVERHEAD=0

echo "=== CONFIGURAZIONE GPU ==="
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "=========================="

#######################
# 2. Avvio del server #
#######################
echo "Avvio Ollama server..."
ollama serve &
OLLAMA_PID=$!

# Attendi che il server sia pronto (più tempo per il caricamento modello)
sleep 8
echo "Test connessione server..."
curl -s http://127.0.0.1:11434/api/tags >/dev/null && echo "✓ Server pronto" || echo "⚠ Server non ancora pronto"

#############################
# 3. Lancio dello script    #
#############################
cd $SLURM_SUBMIT_DIR
echo "Lancio script Python alle $(date)..."
python veronacard_mob_with_geom.py --append

#############################
# 4. Pulizia e fine job     #
#############################
echo "Chiusura Ollama alle $(date)..."
kill $OLLAMA_PID
wait $OLLAMA_PID 2>/dev/null
echo "Job completato!"