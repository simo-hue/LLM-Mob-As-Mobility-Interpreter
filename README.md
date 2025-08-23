# 🚀 LLM-Mob: Predizione della Mobilità Turistica con Large Language Models

![Python](https://img.shields.io/badge/python-3.9--3.11-blue)
![Platform](https://img.shields.io/badge/platform-HPC%20Leonardo%20%7C%20Linux%20%7C%20macOS-green)
![GPUs](https://img.shields.io/badge/GPU-4x%20NVIDIA%20A100%2064GB-red)
![License](https://img.shields.io/badge/licenza-CC--BY--NC-lightgrey)

Sistema avanzato per la predizione dei comportamenti turistici utilizzando Large Language Models (LLM) su infrastruttura HPC. Basato sul paper [Where Would I Go Next? Large Language Models as Human Mobility Predictors](https://arxiv.org/abs/2308.15197).

## 📋 Indice

- [Caratteristiche Principali](#-caratteristiche-principali)
- [Architettura del Sistema](#-architettura-del-sistema)
- [Ottimizzazioni HPC](#-ottimizzazioni-hpc)
- [Installazione](#-installazione)
- [Utilizzo](#-utilizzo)
- [Dataset VeronaCard](#-dataset-veronacard)
- [Risultati e Metriche](#-risultati-e-metriche)
- [Troubleshooting](#-troubleshooting)

## 🌟 Caratteristiche Principali

### Innovazioni Implementate
1. **Sostituzione GPT con Llama/Mixtral** - Utilizzo di modelli open-source invece di API OpenAI a pagamento
2. **Parallelizzazione Multi-GPU** - Supporto nativo per 4x NVIDIA A100 su Leonardo HPC
3. **Sistema di Checkpoint Avanzato** - Ripresa automatica da interruzioni con gestione stato ottimizzata
4. **Circuit Breaker Pattern** - Protezione da cascading failures nel sistema distribuito
5. **Health Monitoring Intelligente** - Load balancing dinamico basato su performance reali
6. **Integrazione Dati Geografici** - Calcolo distanze tra POI per migliorare le predizioni

## 🏗️ Architettura del Sistema

### Componenti Principali

```
┌─────────────────────────────────────────────────────────────┐
│                     HPC Leonardo Node                        │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐ │
│  │   2x56 CPU  │  │   512 GB    │  │   4x NVIDIA A100    │ │
│  │   Cores     │  │    RAM      │  │   64GB VRAM each   │ │
│  └─────────────┘  └─────────────┘  └─────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Ollama Multi-Instance                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │ Port:    │  │ Port:    │  │ Port:    │  │ Port:    │   │
│  │ 11434    │  │ 11435    │  │ 11436    │  │ 11437    │   │
│  │ GPU: 0   │  │ GPU: 1   │  │ GPU: 2   │  │ GPU: 3   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Sistema di Orchestrazione                   │
│  ┌─────────────────┐  ┌──────────────┐  ┌───────────────┐ │
│  │ Health Monitor  │  │Circuit       │  │ Checkpoint    │ │
│  │ & Load Balancer│  │Breaker       │  │ Manager       │ │
│  └─────────────────┘  └──────────────┘  └───────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

### Classi e Moduli Ottimizzati

1. **`Config`** - Gestione centralizzata delle configurazioni
2. **`OllamaConnectionManager`** - Gestisce connessioni multiple Ollama con failover
3. **`HostHealthMonitor`** - Monitoraggio salute host e selezione intelligente
4. **`CircuitBreaker`** - Protezione da failure a cascata
5. **`CheckpointManager`** - Gestione stato per elaborazioni interrompibili
6. **`CardProcessor`** - Elaborazione parallela delle card turistiche
7. **`PromptBuilder`** - Generazione prompt ottimizzati con dati geografici

## ⚡ Ottimizzazioni HPC

### 1. Parallelizzazione Multi-GPU
```python
# Configurazione automatica per 4 GPU
OLLAMA_HOSTS = [
    "http://127.0.0.1:11434",  # GPU 0
    "http://127.0.0.1:11435",  # GPU 1
    "http://127.0.0.1:11436",  # GPU 2
    "http://127.0.0.1:11437",  # GPU 3
]

# Thread pool dinamico basato su GPU disponibili
optimal_workers = min(len(hosts) * 8, 64)
```

### 2. Rate Limiting Adattivo
```python
# Semaforo dinamico per evitare sovraccarico
MAX_CONCURRENT_REQUESTS = len(hosts) * 4
RATE_LIMIT_SEMAPHORE = Semaphore(MAX_CONCURRENT_REQUESTS)
```

### 3. Health Check e Load Balancing
- Monitoraggio continuo della salute degli host
- Selezione host basata su tempi di risposta
- Failover automatico in caso di errori
- Tracking del trend di performance

### 4. Gestione Memoria Ottimizzata
- Batch processing con salvataggio incrementale
- Buffer di risultati con flush periodico
- Checkpoint leggeri per ripresa veloce
- Cleanup automatico della memoria

### 5. Circuit Breaker Pattern
```
Stati del Circuit Breaker:
├── CLOSED (normale operazione)
├── OPEN (troppi errori, rifiuta richieste)
└── HALF_OPEN (test di recupero)
```

## 📦 Installazione

### Prerequisiti
- Python 3.9-3.11 (⚠️ Python 3.12+ non supportato da alcune dipendenze)
- CUDA 11.8+ per supporto GPU
- ~8 GB spazio disco per i modelli
- 32+ GB RAM consigliati

### Setup su Leonardo HPC

```bash
# 1. Carica modulo Python
module load python/3.11.6--gcc--8.5.0

# 2. Crea ambiente virtuale
python3.11 -m venv venv
source venv/bin/activate

# 3. Installa dipendenze
pip install --upgrade pip
pip install pandas numpy scikit-learn requests tqdm

# 4. Configura Ollama multi-GPU
# Crea file ollama_ports.txt con:
echo "11434,11435,11436,11437" > ollama_ports.txt

# 5. Lancia job SLURM
sbatch parallel_production_run_4_GPU.sh
```

### Setup Locale (Testing)

```bash
# 1. Clone repository
git clone https://github.com/simo-hue/LLM-Mob-As-Mobility-Interpreter.git
cd LLM-Mob-As-Mobility-Interpreter

# 2. Setup Python
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Installa e configura Ollama
# Linux
curl -fsSL https://ollama.com/install.sh | sh

# macOS
brew install ollama

# 4. Scarica modello
ollama pull mixtral:8x7b  # o llama3.1:8b

# 5. Configura porta singola per test
echo "11434" > ollama_ports.txt
```

## 🚀 Utilizzo

### Comandi Base

```bash
# Processa tutti i file
python veronacard_mob_with_geom_parrallel.py

# Processa con limite utenti
python veronacard_mob_with_geom_parrallel.py --max-users 1000

# Processa file specifico
python veronacard_mob_with_geom_parrallel.py --file dati_2014.csv

# Modalità append (riprendi da interruzione)
python veronacard_mob_with_geom_parrallel.py --append

# Forza rielaborazione completa
python veronacard_mob_with_geom_parrallel.py --force
```

### Flag e Opzioni

| Flag | Descrizione |
|------|-------------|
| `--file FILE` | Processa solo il file specificato |
| `--max-users N` | Limita elaborazione a N utenti per file |
| `--append` | Riprende elaborazione da checkpoint |
| `--force` | Ignora file esistenti e ricalcola tutto |
| `--anchor RULE` | Regola selezione POI ancora (`penultimate`, `first`, `middle`, indice) |

### Comandi Utili Leonardo HPC

```bash
# Sottometti job
sbatch parallel_production_run_4_GPU.sh

# Controlla stato job
squeue -u $USER

# Visualizza log in tempo reale
tail -f slurm-<JOBID>.out

# Cancella job
scancel <JOBID>

# Controlla budget computazionale
saldo -b <nome_progetto>
```

## 📊 Dataset VeronaCard

### Struttura File

```
data/verona/
├── vc_site.csv                 # 70 POI con coordinate GPS
├── dati_2014.csv               # ~370k timbrature
├── dati_2015.csv               # Log visite turistiche
├── ...                         # per anno
└── veronacard_2023_original.csv
```

### Formato Dati Visite
```csv
data,ora,name_short,card_id
15-08-14,10:30:45,Arena,0403E98ABF3181
15-08-14,14:15:30,Casa di Giulietta,0403E98ABF3181
```

### Formato POI (vc_site.csv)
```csv
name_short,latitude,longitude,category
Arena,45.4394,10.9947,Monument
Casa di Giulietta,45.4419,10.9988,Museum
```

## 📈 Risultati e Metriche

### Output Generato

```
results/
├── <nome_file>_pred_<timestamp>.csv    # Predizioni
└── <nome_file>_checkpoint.txt          # Stato elaborazione
```

### Formato Risultati
```csv
card_id,cluster,history,current_poi,prediction,ground_truth,reason,hit,processing_time,status
0403E98,3,"['Arena','Casa di Giulietta']","Torre Lamberti","['Ponte Pietra','Duomo']",Ponte Pietra,"vicino e panoramico",True,2.34,success
```

### Metriche di Performance

- **Hit Rate**: % predizioni corrette (target in top-k)
- **Processing Speed**: card/ora processate
- **GPU Utilization**: % utilizzo GPU
- **Success Rate**: % richieste completate con successo

### Analisi Risultati

```python
# Notebook per analisi
jupyter notebook notebook/analisi_risultati.ipynb
```

## 🛠️ Troubleshooting

### Problemi Comuni

| Problema | Soluzione |
|----------|-----------|
| `Ollama non risponde` | Verifica che `ollama serve` sia attivo su tutte le porte |
| `CUDA out of memory` | Riduci `num_ctx` in options o usa meno worker paralleli |
| `Timeout richieste` | Aumenta `REQUEST_TIMEOUT` in Config |
| `Circuit breaker aperto` | Sistema sotto stress, attendere reset automatico |
| `Checkpoint corrotto` | Elimina file checkpoint e usa `--force` |

### Debug Avanzato

```bash
# Verifica stato GPU
nvidia-smi

# Test connessione Ollama
curl http://localhost:11434/api/tags

# Verifica modelli disponibili
ollama list

# Log dettagliati
tail -f logs/run_*.log
```

## 📚 Documentazione Tecnica

### Architettura Pipeline

1. **Caricamento Dati** → Lettura CSV visite e POI
2. **Preprocessing** → Filtraggio visite valide, merge con POI
3. **Clustering** → K-means su matrice user-POI
4. **Generazione Prompt** → Include storia, posizione, POI vicini
5. **Inferenza LLM** → Predizione parallela su multi-GPU
6. **Post-processing** → Salvataggio risultati e checkpoint

### Ottimizzazioni Implementate

- **Thread-safe Operations**: Lock granulari per operazioni concorrenti
- **Memory Management**: Garbage collection esplicito dopo batch
- **Error Recovery**: Retry con backoff esponenziale
- **Resource Pooling**: Riuso connessioni HTTP
- **Async I/O**: Scrittura asincrona dei risultati

## 🤝 Contributi

Contributi benvenuti! Per modifiche importanti:

1. Apri una issue per discutere le modifiche
2. Fork il repository
3. Crea un branch per la feature (`git checkout -b feature/AmazingFeature`)
4. Commit delle modifiche (`git commit -m 'Add AmazingFeature'`)
5. Push al branch (`git push origin feature/AmazingFeature`)
6. Apri una Pull Request

## 📄 Licenza e Citazioni

### Licenza Dati
I dati VeronaCard sono forniti **esclusivamente per ricerca accademica** e non possono essere redistribuiti senza permesso di Ente Turismo Verona.

### Citazione
```bibtex
@article{mattioli2025llmmob,
  title={Large Language Models for Verona Card's Human Mobility Prediction},
  author={Mattioli, Simone},
  year={2025},
  institution={Università di Verona}
}
```

## 🙏 Ringraziamenti

- **CINECA** per l'accesso all'infrastruttura Leonardo HPC
- **Università di Verona** per il supporto alla ricerca e per i dati VeronaCard

---

📧 **Contatti**: Per domande o collaborazioni, contattare [mattioli.simone.10@gmail.com]