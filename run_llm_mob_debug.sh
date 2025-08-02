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
echo "▶ Nodo di esecuzione: $(hostname)"
echo "▶ Working directory: $(pwd)"

# Avvia il runner in background con logging migliorato
$OLLAMA_BIN runner \
  --model "$MODEL_PATH" \
  --ctx-size 8192 \
  --batch-size 512 \
  --n-gpu-layers 33 \
  --threads 32 \
  --parallel 2 \
  --port $OLLAMA_PORT > ollama_runner.log 2>&1 &
RUNNER_PID=$!

echo "▶ Runner PID: $RUNNER_PID"

# === 4. ATTESA SERVER MIGLIORATA ===
echo "▶ Attesa che il server sia pronto..."
MAX_WAIT=60  # Aumentato da 45 a 60 secondi
WAIT_INTERVAL=2

for i in $(seq 1 $MAX_WAIT); do
  # Controlla se il processo è ancora attivo
  if ! kill -0 $RUNNER_PID 2>/dev/null; then
    echo "❌ Il runner si è fermato inaspettatamente!"
    echo "▶ Log del runner:"
    cat ollama_runner.log
    exit 1
  fi
  
  # Prova a connettersi
  if curl -s --connect-timeout 3 --max-time 5 "http://127.0.0.1:$OLLAMA_PORT/api/tags" >/dev/null 2>&1; then
    echo "✓ Runner LLaMA attivo dopo $((i * WAIT_INTERVAL)) secondi"
    
    # Test aggiuntivo per verificare che sia davvero pronto
    if curl -s --connect-timeout 3 --max-time 5 "http://127.0.0.1:$OLLAMA_PORT/api/version" >/dev/null 2>&1; then
      echo "✓ Server completamente pronto"
      break
    fi
  fi
  
  if [ $((i % 5)) -eq 0 ]; then
    echo "⏳ Attesa runner... ($((i * WAIT_INTERVAL))s / $((MAX_WAIT * WAIT_INTERVAL))s)"
  fi
  
  sleep $WAIT_INTERVAL
done

# Verifica finale
if ! curl -s --connect-timeout 3 --max-time 5 "http://127.0.0.1:$OLLAMA_PORT/api/tags" >/dev/null 2>&1; then
  echo "❌ Server non risponde dopo $((MAX_WAIT * WAIT_INTERVAL)) secondi"
  echo "▶ Log del runner:"
  cat ollama_runner.log
  kill $RUNNER_PID 2>/dev/null
  exit 1
fi

# === 5. SCRIPT PYTHON ===
cd $SLURM_SUBMIT_DIR
echo "▶ Lancio script Python alle $(date)..."
echo "▶ Directory: $(pwd)"

# Esegui lo script Python con gestione errori
if python veronacard_mob_with_geom.py; then
  echo "✅ Script Python completato con successo"
else
  EXIT_CODE=$?
  echo "❌ Script Python fallito con codice $EXIT_CODE"
fi

# === 6. CHIUSURA ===
echo "▶ Chiusura runner alle $(date)..."
kill $RUNNER_PID 2>/dev/null
wait $RUNNER_PID 2>/dev/null

# Mostra le ultime righe del log del runner
echo "▶ Ultime righe del log runner:"
tail -10 ollama_runner.log 2>/dev/null || echo "Log non disponibile"

echo "✅ Job completato!"