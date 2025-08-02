#!/bin/bash
#!/bin/bash
#SBATCH --job-name=llm-mob-fixed
#SBATCH --account=IscrC_LLM-Mob
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G

echo "🔍 SCRIPT DIAGNOSTICO OLLAMA SU LEONARDO"
echo "========================================"
echo "Data: $(date)"
echo "Nodo: $(hostname)"
echo "User: $(whoami)"
echo ""

# Informazioni sistema base
echo "📋 INFORMAZIONI SISTEMA"
echo "------------------------"
echo "OS: $(cat /etc/os-release | grep PRETTY_NAME | cut -d'"' -f2)"
echo "Kernel: $(uname -r)"
echo "Architettura: $(uname -m)"
echo "Uptime: $(uptime -p)"
echo ""

# Informazioni GPU
echo "🖥️  INFORMAZIONI GPU"
echo "-------------------"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-'non impostato'}"
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "=== nvidia-smi ==="
    nvidia-smi
    echo ""
    echo "=== GPU Memory Details ==="
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu --format=csv
    echo ""
    echo "=== Processi GPU attivi ==="
    nvidia-smi pmon -c 1 2>/dev/null || echo "Nessun processo GPU attivo"
else
    echo "❌ nvidia-smi non disponibile"
fi
echo ""

# Informazioni CUDA
echo "🔧 INFORMAZIONI CUDA"
echo "-------------------"
if command -v nvcc &> /dev/null; then
    echo "NVCC Version: $(nvcc --version | grep release | awk '{print $6}' | cut -d',' -f1)"
else
    echo "❌ nvcc non disponibile"
fi

if [ -f /usr/local/cuda/version.txt ]; then
    echo "CUDA Version (da file): $(cat /usr/local/cuda/version.txt)"
elif [ -f /usr/local/cuda/version.json ]; then
    echo "CUDA Version (da JSON): $(cat /usr/local/cuda/version.json | grep version | head -1)"
else
    echo "⚠️  File versione CUDA non trovato nelle posizioni standard"
fi

# Controlla variabili ambiente CUDA
echo "Variabili ambiente CUDA:"
env | grep -i cuda | sort
echo ""

# Informazioni Ollama
echo "🦙 INFORMAZIONI OLLAMA"
echo "---------------------"
OLLAMA_PATH="/leonardo/home/userexternal/smattiol/opt/ollama/bin/ollama"
if [ -f "$OLLAMA_PATH" ]; then
    echo "Path Ollama: $OLLAMA_PATH"
    echo "Versione Ollama: $($OLLAMA_PATH --version 2>/dev/null || echo 'Errore nel leggere versione')"
    echo "Dimensioni binario: $(ls -lh $OLLAMA_PATH | awk '{print $5}')"
    echo "Data modifica: $(ls -l $OLLAMA_PATH | awk '{print $6, $7, $8}')"
    echo ""
    
    # Controlla se il binario è eseguibile e funzionante
    if $OLLAMA_PATH --help &> /dev/null; then
        echo "✅ Binario Ollama funzionante"
    else
        echo "❌ Binario Ollama non funzionante"
    fi
else
    echo "❌ Ollama non trovato in $OLLAMA_PATH"
fi

# Controlla modelli Ollama
OLLAMA_MODELS_DIR="/leonardo/home/userexternal/smattiol/.ollama/models"
echo ""
echo "📁 MODELLI OLLAMA"
echo "----------------"
if [ -d "$OLLAMA_MODELS_DIR" ]; then
    echo "Directory modelli: $OLLAMA_MODELS_DIR"
    echo "Spazio occupato: $(du -sh $OLLAMA_MODELS_DIR 2>/dev/null | cut -f1)"
    echo ""
    echo "Modelli installati:"
    if [ -d "$OLLAMA_MODELS_DIR/manifests" ]; then
        find "$OLLAMA_MODELS_DIR/manifests" -name "*.json" 2>/dev/null | sed 's|.*/manifests/registry.ollama.ai/library/||' | sed 's|/.*||' | sort -u | head -10
    else
        echo "⚠️  Directory manifests non trovata"
    fi
    
    if [ -d "$OLLAMA_MODELS_DIR/blobs" ]; then
        echo ""
        echo "Blob più grandi (top 5):"
        find "$OLLAMA_MODELS_DIR/blobs" -type f -exec ls -lh {} \; 2>/dev/null | sort -k5 -hr | head -5 | awk '{print $5, $9}' || echo "Nessun blob trovato"
    fi
else
    echo "❌ Directory modelli non trovata"
fi
echo ""

# Informazioni di rete
echo "🌐 INFORMAZIONI RETE"
echo "-------------------"
echo "Hostname: $(hostname -f)"
echo "IP address: $(hostname -I | awk '{print $1}')"
echo ""

# Controlla porte disponibili
echo "🔌 CONTROLLO PORTE"
echo "-----------------"
for port in 39002 11434 8080; do
    if ss -tuln | grep -q ":$port "; then
        echo "❌ Porta $port già occupata:"
        ss -tuln | grep ":$port "
    else
        echo "✅ Porta $port libera"
    fi
done
echo ""

# Informazioni risorse sistema
echo "💾 RISORSE SISTEMA"
echo "-----------------"
echo "Memoria RAM:"
echo "  Totale: $(free -h | grep Mem | awk '{print $2}')"
echo "  Libera: $(free -h | grep Mem | awk '{print $7}')"
echo "  Usata: $(free -h | grep Mem | awk '{print $3}')"
echo ""
echo "CPU:"
echo "  Modello: $(grep 'model name' /proc/cpuinfo | head -1 | cut -d':' -f2 | xargs)"
echo "  Core: $(nproc)"
echo "  Load average: $(uptime | awk -F'load average:' '{print $2}' | xargs)"
echo ""

# Controlla processi Ollama esistenti
echo "🔍 PROCESSI OLLAMA ATTIVI"
echo "------------------------"
if pgrep -f ollama &> /dev/null; then
    echo "Processi Ollama trovati:"
    ps aux | grep -i ollama | grep -v grep
else
    echo "✅ Nessun processo Ollama attivo"
fi
echo ""

# Informazioni Python
echo "🐍 INFORMAZIONI PYTHON"
echo "---------------------"
if command -v python3 &> /dev/null; then
    echo "Python versione: $(python3 --version)"
    echo "Python path: $(which python3)"
    
    # Controlla librerie specifiche
    echo ""
    echo "Librerie Python rilevanti:"
    for lib in requests json subprocess; do
        python3 -c "import $lib; print('✅ $lib disponibile')" 2>/dev/null || echo "❌ $lib non disponibile"
    done
else
    echo "❌ Python3 non disponibile"
fi
echo ""

# Test connettività base
echo "🔗 TEST CONNETTIVITÀ"
echo "-------------------"
if ping -c 1 8.8.8.8 &> /dev/null; then
    echo "✅ Connessione internet OK"
else
    echo "❌ Nessuna connessione internet"
fi

if ping -c 1 127.0.0.1 &> /dev/null; then
    echo "✅ Loopback OK"
else
    echo "❌ Problema loopback"
fi
echo ""

# Informazioni SLURM (se disponibili)
echo "⚡ INFORMAZIONI SLURM"
echo "-------------------"
if [ ! -z "$SLURM_JOB_ID" ]; then
    echo "Job ID: $SLURM_JOB_ID"
    echo "Nodo: $SLURM_NODELIST"
    echo "Partizione: $SLURM_JOB_PARTITION"
    echo "Tempo limite: $SLURM_JOB_TIME_LIMIT"
    echo "CPU allocate: $SLURM_CPUS_ON_NODE"
    echo "GPU allocate: ${SLURM_GPUS:-'non specificato'}"
    
    if command -v squeue &> /dev/null; then
        echo ""
        echo "Info dettagliate job:"
        squeue -j $SLURM_JOB_ID -o "%.10i %.15P %.20j %.8u %.2t %.10M %.6D %R" 2>/dev/null || echo "Impossibile ottenere info dettagliate"
    fi
else
    echo "⚠️  Non in ambiente SLURM"
fi
echo ""

# Test rapido Ollama (se installato)
echo "🧪 TEST RAPIDO OLLAMA"
echo "--------------------"
if [ -f "$OLLAMA_PATH" ]; then
    echo "Tentativo di eseguire ollama list..."
    timeout 10 $OLLAMA_PATH list 2>&1 || echo "❌ Comando fallito o timeout"
else
    echo "⚠️  Ollama non trovato, test saltato"
fi
echo ""

echo "✅ DIAGNOSTICA COMPLETATA"
echo "========================"
echo ""
echo "💡 SUGGERIMENTI:"
echo "- Salva questo output in un file: ./diagnostic.sh > diagnostic_$(date +%Y%m%d_%H%M%S).log"
echo "- Condividi il file per l'analisi"
echo ""