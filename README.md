# ***L***arge ***L***anguage ***M***odels for Human ***Mob***ility Prediction (LLM-Mob)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![Licence](https://img.shields.io/badge/licence-CC--BY--NC-lightgrey)
![Platform](https://img.shields.io/badge/platform-macOS%20%7C%20Linux-green)

Starting paper ***[Where Would I Go Next? Large Language Models as Human Mobility Predictors](https://arxiv.org/abs/2308.15197)***.

## My Changes
### 1. Implementing llama3.1 instead of GPT (openAI) Models ( They require payments )
## Where I could Find It?
In The file llm-Mob Directory there is the Fully Working original llm-mob without the OPENAI Dependency

## My Focus On The VeronaCard Dataset

### 🚀 Quickstart (copy‑paste) [How To Run It]

```bash
# 1. Clone the repo and move into the project root
git clone https://github.com/simo-hue/LLM-Mob-As-Mobility-Interpreter.git
cd LLM-Mob-As-Mobility-Interpreter

# 2. Install and launch a local Llama‑3 model with Ollama

# Ubuntu (NEW)
curl -fsSL https://ollama.com/install.sh | sh   # Linux – Ollama install script

# MacOS
brew install ollama                 # macOS – see <https://ollama.ai/> for Linux/Win

OLLAMA_HOST=127.0.0.1:11434 ollama serve &   # run Ollama in the background
ollama pull llama3

# 3. Create and activate a Python virtual env called “llm”
python3 -m venv llm
source llm/bin/activate
pip install -r requirements.txt

# 4. Follow The Next Chapter to understand how to run the script
```

### Run The Script

**nuovi flag CLI** introdotti nello script `veronacard_mob.py` per controllare la (ri)elaborazione dei file di log ed evitare calcoli ridondanti.

| Flag        | Behaviour                                                     |
|-------------|-------------------------------------------------------------------
| *(nothing)* | If exists **at least one** file `results/<stem>_pred_*.csv` skip the entire input |
| `--force`   | Ignore any pre-existing output and recalculate **everything**       | 
| `--append`  | Loads latest output, processes **only** missing cards, and queues new predictions |
| `--anchor <rule>` | Choose which POI to use as anchor for prediction (`penultimate` [default] \| `first` \| `middle` \| any integer index) |

### Naming 

Output saved as

```
results/<stem>_pred_<timestamp>.csv
```

where `<stem>` is the name of the input file ( NO extension ) (es. `dati_2014`).

### Examples of use

```bash
# 1) Only files not already processed ( skips the files in the results folder )
python veronacard_mob.py

# 2) Recalculate all files
python veronacard_mob.py --force

# 3) It starts from the last processed file where it left off
python veronacard_mob.py --append

# 4) Use the POI in the middle of the trajectory as anchor ( as an EXAMPLE )
python veronacard_mob.py --anchor middle
```

**Project layout reminder**

```
data/verona/                  raw VeronaCard logs (sub‑folders by year)
data/verona/vc_site.csv       POI catalogue (reference)
results/                      predictions_*.csv (auto‑generated)
veronacard_mob.py             loads every dati_*.csv, clusters,
                              queries LLaMA and writes user‑level results
```

The script scans every `*.csv` inside `data/verona/**` **except** `vc_site.csv`, performs a fresh K‑Means clustering for each log file, predicts the next POI for a sample of users (default = 50 per file) and stores detailed output in `results/`.

---

### ⚙️ Prerequisites

- macOS 12 / Ubuntu 22 or newer  
- **Python ≥ 3.9 and ≤ 3.11**  
  (⚠️ `kornia` fails to install with Python 3.12+)
- ~ 8 GB of free disk space to download **llama3**  
- (Optional) 32 GB RAM recommended for faster inference


---

### 🖥️ Windows notes
> **Windows 10/11** – download the official Ollama MSI, install it, then run  
> `setx OLLAMA_HOST 127.0.0.1:11434` in PowerShell **before** executing `ollama serve`.

---

### 🚀 GPU inference (optional)
If your build of Ollama supports GPU (CUDA on Linux, Metal on macOS), you can enable it with:

```bash
ollama run llama3:latest --gpu
```

*GPU support is not mandatory; the script works purely on CPU as well.*

---

### ✅ Quick model check

```bash
# After 'ollama serve' verify the model is registered
ollama list      # you should see a line like:
# NAME        ID            SIZE   MODIFIED
# llama3      …             3.8 GB  2 seconds ago
```

---

### 📄 Example output

```
card_id,cluster,history,current_poi,prediction,ground_truth,reason,hit
0403E98ABF3181,3,"['Arena','Casa di Giulietta']","Torre Lamberti","['Ponte Pietra','Duomo']",Ponte Pietra,"panoramic spot near the city center",True
```

At the end of a run you will see a summary such as:

```
Run completed: 32/50 hit (64.00%)
Full results saved to: /results/dati_2014_pred_20250507_153012.csv
```

---

### 🛠 Troubleshooting

| Issue | Quick fix |
|-------|-----------|
| `⚠️ Ollama is not running` | Make sure you launched `ollama serve` and port `11434` is free |
| `FutureWarning` from Pandas on `int(...)` | Already fixed with `.iloc[0]`; ignore if it still appears |
| `LLM timeout` | Increase `timeout=60` inside `get_chat_completion` |
| Missing log CSV files | Check the folder structure `data/verona/dataset_veronacard_YYYY_YYYY/` |

---

### 🗂 Dataset

- **`data/verona/vc_site.csv`**  →  70 official POIs with coordinates  
- **`data/verona/dataset_veronacard_2014_2020/`**  →  stamping logs (*dati_YYYY.csv*)  
  - e.g. *dati_2014.csv* ≈ 370 k records

Each file contains the columns `data, ora, name_short, card_id`.

---

### 📜 Data license

The VeronaCard logs are provided **exclusively for academic research** and may not be redistributed without permission from Ente Turismo Verona.  
The files in this repository are therefore sample data; replace them with your own if you have a different usage agreement.

---

### 🐳 Run everything in Docker (optional)

```dockerfile
# Dockerfile (minimal)
FROM python:3.11-slim
RUN pip install --no-cache-dir ollama==0.1.29 pandas scikit-learn tqdm matplotlib
WORKDIR /workspace
COPY . .
CMD ["python","veronacard_mob.py"]
```

Build & run:

```bash
docker build -t llm-mob .
docker run -it --rm -p 11434:11434 llm-mob
```

The container exposes Ollama on port `11434` and executes the full pipeline.

### Results and evaluation
We provide the actual prediction results obtained in our experiments in `/results`. 
To calculate the evaluation metrics, check the IPython notebook `metrics.ipynb` and run the scripts therein.

### Contributing
Pull requests are welcome! Please open an issue first to discuss major changes.

### HPC setup (Leonardo)

```bash
module load python/3.11.6--gcc--8.5.0
python -m venv llm
source llm/bin/activate
pip install -r requirements.txt
```

#### Comandi Utili Per Leonardo:
python3.11 -m venv llm          # Forzare la versione 3.11 nella creazione dell'ambiente
sbatch run_llm_mob.sh           # lanciare un Job
squeue -u $USER                 # stato job
tail -f slurm-<JOBID>.out       # log live
scancel <JOBID>                 # Rimozione del Job
saldo -b IscrC_LLM-Mob          # Controllare il SALDO
[Documentazione Tecnica di Leonardo](https://docs.hpc.cineca.it/hpc/leonardo.html#leonardo-card)

### Citation

```bibtex
@article{mattioli2025Thesis,
  title={Large Language Models for Verona Card's Human Mobility Prediction (LLM-Mob)},
  author={Mattioli, Simone},
  year={2025}
}
```