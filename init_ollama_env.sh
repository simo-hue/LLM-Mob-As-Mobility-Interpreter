#!/usr/bin/env bash
#
# Inizializza l’ambiente utente per Ollama + LLM-Mob.
# Esegui SOLO sul login node, SOLO la prima volta.

set -euo pipefail

# 1. Dove salvare i binari e i modelli  (cambia se vuoi)
BIN_DIR="$HOME/opt/ollama/bin"
MODEL_DIR="$HOME/.ollama/models"          # oppure $CINECA_SCRATCH/ollama/models

mkdir -p "$BIN_DIR" "$MODEL_DIR"

# 2. Scarica l’ultima release statica di Ollama
cd "$BIN_DIR"/..
curl -L https://ollama.com/download/ollama-linux-amd64.tgz -o ollama.tgz
tar -xzf ollama.tgz
rm ollama.tgz

# 3. Aggiorna PATH e variabili (idempotente)
grep -qxF "export PATH=$BIN_DIR:\$PATH" ~/.bashrc || \
  echo "export PATH=$BIN_DIR:\$PATH" >> ~/.bashrc
grep -qxF "export OLLAMA_MODELS=$MODEL_DIR" ~/.bashrc || \
  echo "export OLLAMA_MODELS=$MODEL_DIR" >> ~/.bashrc

# 4. Pull modelli di base
export PATH=$BIN_DIR:$PATH
export OLLAMA_MODELS=$MODEL_DIR
export OLLAMA_HOST=127.0.0.1:11434
ollama serve &                    # background
sleep 3
ollama pull llama3:8b:Q4_0        # debug
# ollama pull llama3:70b:Q4_0     # scommenta se ti serve la 70B
pkill ollama

echo "Setup completato ✔  (ricarica la shell o 'source ~/.bashrc')"