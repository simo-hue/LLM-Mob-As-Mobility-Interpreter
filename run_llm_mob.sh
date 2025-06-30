#!/bin/bash
#SBATCH --job-name=llm-mob
#SBATCH --account=IscrC_LLM-Mob
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg        # 30′; cambia in bprod per run lunghe
#SBATCH --time=00:20:00
#SBATCH --nodes=1
#SBATCH --gpus=1                    # A100 64 GB
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --output=slurm-%j.out

########################
# 1. Ambiente software #
########################
module purge
module load python/3.11.6--gcc--8.5.0
source $SLURM_SUBMIT_DIR/llm/bin/activate

# variabili impostate da ~/.bashrc grazie allo script init
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_MODEL=llama3.1:8b     # forza 8B; cambia se vuoi 70B

#######################
# 2. Avvio del server #
#######################
ollama serve &
sleep 2                                   # garantisce che la porta sia up

#############################
# 3. Lancio dello   script  #
#############################
cd $SLURM_SUBMIT_DIR
python veronacard_mob.py --anchor middle --append

#############################
# 4. Pulizia e fine job     #
#############################
pkill ollama