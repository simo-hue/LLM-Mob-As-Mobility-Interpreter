#!/bin/bash
#SBATCH --job-name=llm-mob
#SBATCH --account=IscrC_LLM-Mob
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=slurm-%j.out

# === 1. AMBIENTE ===
module purge
module load python/3.11.6--gcc--8.5.0
module load cuda/12.3
source $SLURM_SUBMIT_DIR/LLM/bin/activate

export CUDA_VISIBLE_DEVICES=0
export OLLAMA_GPU_OVERHEAD=0

# === 2. CONFIGURAZIONE MODELLO ===
MODEL_PATH="/leonardo/home/userexternal/smattiol/.ollama/models/blobs/sha256-667b0c1932bc6ffc593ed1d03f895bf2dc8dc6df21db3042284a6f4416b06a29"
OLLAMA_PORT=39000
OLLAMA_BIN="$HOME/opt/ollama/bin/ollama"

echo $OLLAMA_PORT > $SLURM_SUBMIT_DIR/ollama_port.txt
echo "✅ Ho scritto la porta in ollama_port.txt: $(cat $SLURM_SUBMIT_DIR/ollama_port.txt)"


# === 3. AVVIO RUNNER ===
echo "▶ Avvio runner LLaMA sulla porta $OLLAMA_PORT..."
$OLLAMA_BIN runner \
  --model "$MODEL_PATH" \
  --ctx-size 8192 \
  --batch-size 512 \
  --n-gpu-layers 33 \
  --threads 32 \
  --parallel 2 \
  --port $OLLAMA_PORT &
RUNNER_PID=$!

# === 4. ATTESA SERVER ===
for i in {1..15}; do
  if curl -s "http://127.0.0.1:$OLLAMA_PORT/api/tags" >/dev/null; then
    echo "✓ Runner LLaMA attivo"
    break
  else
    echo "⏳ Attesa runner... ($i)"
    sleep 3
  fi
done

# === 5. SCRIPT PYTHON ===
cd $SLURM_SUBMIT_DIR
echo "▶ Lancio script Python alle $(date)..."
python veronacard_mob_with_geom.py

# === 6. CHIUSURA ===
echo "▶ Chiusura runner alle $(date)..."
kill $RUNNER_PID
wait $RUNNER_PID 2>/dev/null
echo "✅ Job completato!"