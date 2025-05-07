## 🚀 Quickstart (copy‑paste)

```bash
# 1.   move into the project root (adjust the path if different)
cd ~/Downloads/LLM-Mob-As-Mobility-Interpreter

# 2.   install and start a local Llama‑3 model with Ollama
brew install ollama                # macOS – see <https://ollama.ai/> for Linux/Win
ollama pull llama3
OLLAMA_HOST=127.0.0.1:11434 ollama serve &   # run Ollama in the background

# 3.   create & activate a Python virtual‑env called “llm”
python3 -m venv llm
source llm/bin/activate
pip install -r requirements.txt

# 4.   launch the VeronaCard prediction pipeline
python veronacard_mob.py           # outputs CSV files in ./results
```

**Project layout reminder**

```
data/verona/                  raw VeronaCard logs (per‑year subfolders)
data/verona/vc_site.csv       POI catalogue (reference)
results/                      predictions_*.csv (auto‑generated)
veronacard_mob.py             main script: loads every dati_*.csv, clusters,
                              queries LLaMA and writes user‑level results
```

The script scans every `*.csv` inside `data/verona/**` **except** `vc_site.csv`, runs
a fresh K‑Means clustering for each log file, predicts the next POI for a
sample of users (default 50 per file) and saves the detailed output in
`results/`.
---

## ⚙️ Prerequisiti

- macOS 12 / Ubuntu 22 o superiore  
- Python ≥ 3.9  
- ~ 8 GB di spazio libero per scaricare il modello **llama3**  
- (Opzionale) 32 GB RAM consigliati per velocizzare l’inferenza

---

## ✅ Verifica rapida del modello

```bash
# dopo 'ollama serve' puoi controllare che il modello sia registrato
ollama list      # l'output deve contenere una riga simile a:
# NAME        ID            SIZE   MODIFIED
# llama3      …             3.8 GB  2 seconds ago
```

---

## 🔧 Parametri principali dello script

| Variabile in `veronacard_mob.py` | Default | Significato |
|----------------------------------|---------|-------------|
| `TOP_K`      | `5`  | lunghezza della lista di POI predetti dall'LLM |
| `MAX_USERS`  | `50` | n. di card valutate per ciascun file di log |
| `N_TEST`     | `100`| n. di utenti usati per calcolare Hit@k globale |

Modificale a inizio script per adattare la durata dell'esecuzione o la granularità dei risultati.

---

## 📄 Esempio di output

```
card_id,cluster,history,current_poi,prediction,ground_truth,reason,hit
0403E98ABF3181,3,"['Arena','Casa di Giulietta']","Torre Lamberti","['Ponte Pietra','Duomo']",Ponte Pietra,"luogo panoramico vicino al centro",True
```

A fine run viene mostrato un riepilogo, ad es.:

```
Run completata: 32/50 hit (64.00%)
Risultati completi salvati in: /results/dati_2014_pred_20250507_153012.csv
```

---

## 🛠 Troubleshooting

| Problema | Soluzione rapida |
|----------|------------------|
| `⚠️ Ollama non è in esecuzione` | assicurati di aver lanciato `ollama serve` e che la porta `11434` sia libera |
| `FutureWarning` Pandas su `int(...)` | già risolto con `.iloc[0]`, puoi ignorare se compare ancora |
| `LLM timeout` | aumenta il valore `timeout=60` in `get_chat_completion` |
| CSV di log mancanti | verifica la struttura `data/verona/dataset_veronacard_YYYY_YYYY/` |

---

## 🗂 Dataset

- **`data/verona/vc_site.csv`**  &nbsp;→  70 POI ufficiali con coordinate  
- **`data/verona/dataset_veronacard_2014_2020/`**  &nbsp;→  log di timbrature (*dati_YYYY.csv*)  
  - ad es. *dati_2014.csv* ≈ 370 k record

Ogni file contiene le colonne `data, ora, name_short, card_id`.

---

## 📜 Licenza dei dati

I log VeronaCard sono forniti esclusivamente per scopi di ricerca accademica
e non possono essere ridistribuiti senza permesso dell’Ente Turismo Verona.
I file nel repository sono quindi d’esempio; sostituiscili con i tuoi se
hai un accordo di utilizzo differente.

---

## Results and evaluation
We provide the actual prediction results obtained in our experiments in `/results`. 
To calculate the evaluation metrics, check the IPython notebook `metrics.ipynb` and run the scripts therein.


## Citation

```bibtex
@article{mattioli2025Thesis,
  title={Large Language Models for Human Mobility Prediction (LLM-Mob)},
  author={Mattioli Simone},
  year={2025}
}
```
```bibtex
@article{wang2023would,
  title={Where would i go next? large language models as human mobility predictors},
  author={Wang, Xinglei and Fang, Meng and Zeng, Zichao and Cheng, Tao},
  journal={arXiv preprint arXiv:2308.15197},
  year={2023}
}
```