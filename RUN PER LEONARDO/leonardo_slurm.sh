#!/bin/bash
#SBATCH --job-name=verona_poi
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --account=your_account_here
#SBATCH --output=verona_poi_%j.out
#SBATCH --error=verona_poi_%j.err

# Carica moduli necessari
module load python/3.11
module load cuda/11.8
module load gcc/11.3.0

# Attiva ambiente virtuale (crea se non esiste)
VENV_DIR="$HOME/venv_leonardo"
if [ ! -d "$VENV_DIR" ]; then
    echo "Creando ambiente virtuale..."
    python -m venv $VENV_DIR
    source $VENV_DIR/bin/activate
    
    # Installa dipendenze
    pip install --upgrade pip
    pip install pandas numpy scikit-learn tqdm
    
    # Installa cuML per GPU (se disponibile)
    pip install cuml-cu11 cupy-cuda11x
    
    # Installa transformers per LLM locale
    pip install torch transformers accelerate
else
    source $VENV_DIR/bin/activate
fi

# Variabili ambiente per ottimizzazioni
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=32
export NUMBA_CACHE_DIR=/tmp/numba_cache_$SLURM_JOB_ID

# Verifica GPU
nvidia-smi

# Parametri del job
INPUT_FILE="data/verona/dati_2014.csv"
POI_FILE="data/verona/vc_site.csv"
MAX_USERS=1000
WORKERS=32

echo "Avvio processing..."
echo "Input: $INPUT_FILE"
echo "POI: $POI_FILE"
echo "Max users: $MAX_USERS"
echo "Workers: $WORKERS"

# Esegui script ottimizzato
python leonardo_optimized.py \
    --input "$INPUT_FILE" \
    --poi "$POI_FILE" \
    --max-users $MAX_USERS \
    --workers $WORKERS \
    --gpu

echo "Job completato alle $(date)"

# Pulizia
rm -rf /tmp/numba_cache_$SLURM_JOB_ID
