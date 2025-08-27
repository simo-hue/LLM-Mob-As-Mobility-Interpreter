#!/bin/bash
#SBATCH --job-name=ollama_install
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB

export OLLAMA_MODELS=/leonardo_work/IscrC_LLM-Mob/.ollama/models

# Avvia Ollama
ollama serve &
sleep 10

echo "installazione modello"
# Installa i modelli
ollama pull qwen2.5:7b

echo "modello installato"

# Termina il servizio
pkill ollama