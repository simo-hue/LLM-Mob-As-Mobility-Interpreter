# Architettura GPU e Caricamento Modelli - Spiegazione Dettagliata

## 1. DOVE VIENE CARICATO IL MODELLO MIXTRAL:8x7b

### 1.1 Storage del Modello (Disco)
```bash
export OLLAMA_MODELS="$WORK/.ollama/models"
```

**Posizione**: Il modello viene **scaricato e salvato** su disco in:
```
/leonardo_work/IscrC_LLM-Mob/.ollama/models/
├── manifests/
│   └── registry.ollama.ai/
│       └── library/
│           └── mixtral/
│               └── 8x7b
└── blobs/
    ├── sha256-abc123... (layer 1 - ~4GB)
    ├── sha256-def456... (layer 2 - ~4GB) 
    ├── sha256-ghi789... (layer 3 - ~4GB)
    ├── ...
    └── sha256-xyz999... (layer N - ~4GB)
```

**Dimensione totale**: ~26GB su disco (filesystem Lustre Leonardo)

### 1.2 Caricamento in Memoria GPU (Runtime)
Quando Ollama avvia, il modello viene caricato **dalla storage su disco → VRAM GPU**:

```
DISCO ($WORK/.ollama/models) → VRAM GPU (64GB A100)
      26GB                         ~45GB utilizzati
```

**IMPORTANTE**: Il modello non viene duplicato 4 volte! Vediamo come...

## 2. RUOLO DELLE 4 GPU A100

### 2.1 Architettura Multi-GPU Distributiva

```
┌─────────────────────────────────────────────────────────────┐
│                     NODO LEONARDO BOOSTER                  │
├─────────────┬─────────────┬─────────────┬─────────────────┤
│   GPU 0     │    GPU 1    │    GPU 2    │     GPU 3       │
│  (MASTER)   │   (SLAVE)   │   (SLAVE)   │    (SLAVE)      │
├─────────────┼─────────────┼─────────────┼─────────────────┤
│ Mixtral     │  Mixtral    │  Mixtral    │   Mixtral       │
│ ~45GB VRAM  │ ~45GB VRAM  │ ~45GB VRAM  │  ~45GB VRAM     │
├─────────────┼─────────────┼─────────────┼─────────────────┤
│ Ollama      │  Ollama     │  Ollama     │   Ollama        │
│ Port 39001  │ Port 39002  │ Port 39003  │  Port 39004     │
├─────────────┼─────────────┼─────────────┼─────────────────┤
│ CUDA_       │ CUDA_       │ CUDA_       │  CUDA_          │
│ VISIBLE_    │ VISIBLE_    │ VISIBLE_    │  VISIBLE_       │
│ DEVICES=0   │ DEVICES=1   │ DEVICES=2   │  DEVICES=3      │
└─────────────┴─────────────┴─────────────┴─────────────────┘
```

### 2.2 **OGNI GPU HA UNA COPIA COMPLETA DEL MODELLO**

**Misconception comune**: "Il modello è distribuito tra le GPU"
**Realtà**: **Ogni GPU carica una copia completa** di Mixtral:8x7b nella propria VRAM

**Perché?**
1. **Indipendenza**: Ogni GPU può elaborare richieste in modo completamente autonomo
2. **Latenza**: Nessuna comunicazione inter-GPU necessaria durante inference
3. **Resilienza**: Se una GPU fallisce, le altre continuano a funzionare
4. **Semplicità**: Nessuna sincronizzazione complessa tra GPU

### 2.3 Strategia di Caricamento del Modello

#### **Fase 1: GPU 0 (MASTER) - Prima Copia**
```bash
# GPU 0 carica il modello da disco
CUDA_VISIBLE_DEVICES=0 ollama serve &
# Questo prende 26GB da disco → 45GB VRAM GPU 0
```

**Tempo**: ~3-5 minuti (lettura da disco + ottimizzazioni CUDA)

#### **Fase 2: GPU 1,2,3 (SLAVES) - Copia da GPU 0**
```bash
# GPU 1,2,3 copiano il modello già ottimizzato da GPU 0
CUDA_VISIBLE_DEVICES=1 ollama serve &  # Copia da GPU 0 → GPU 1
CUDA_VISIBLE_DEVICES=2 ollama serve &  # Copia da GPU 0 → GPU 2  
CUDA_VISIBLE_DEVICES=3 ollama serve &  # Copia da GPU 0 → GPU 3
```

**Tempo**: ~1-2 minuti ciascuna (copia ottimizzata inter-GPU via NVLink)

**Meccanismo**: Le GPU slave utilizzano **CUDA memory-to-memory copy** dalla GPU master, molto più veloce della rilettura da disco.

## 3. OLLAMA: UN'ISTANZA PER GPU

### 3.1 Architettura Multi-Processo

```
┌──────────────────────────────────────────────────────┐
│                    SISTEMA HOST                      │
├──────────────┬──────────────┬──────────────┬─────────┤
│  Processo 1  │  Processo 2  │  Processo 3  │Processo4│
│              │              │              │         │
│ ollama serve │ ollama serve │ ollama serve │ollama   │
│              │              │              │serve    │
│ PID: 12345   │ PID: 12346   │ PID: 12347   │PID:     │
│              │              │              │12348    │
│ GPU: 0       │ GPU: 1       │ GPU: 2       │GPU: 3   │
│ Port: 39001  │ Port: 39002  │ Port: 39003  │Port:    │
│              │              │              │39004    │
└──────────────┴──────────────┴──────────────┴─────────┘
```

### 3.2 **Ogni GPU ha un processo Ollama dedicato**

```bash
# 4 processi Ollama separati e indipendenti
CUDA_VISIBLE_DEVICES=0 OLLAMA_HOST=127.0.0.1:39001 ollama serve &  # PID A
CUDA_VISIBLE_DEVICES=1 OLLAMA_HOST=127.0.0.1:39002 ollama serve &  # PID B  
CUDA_VISIBLE_DEVICES=2 OLLAMA_HOST=127.0.0.1:39003 ollama serve &  # PID C
CUDA_VISIBLE_DEVICES=3 OLLAMA_HOST=127.0.0.1:39004 ollama serve &  # PID D
```

**Caratteristiche**:
- **Processi indipendenti**: Crash di uno non influenza gli altri
- **Porte diverse**: Ogni processo ascolta su una porta dedicata
- **GPU dedicata**: Ogni processo vede solo "la sua" GPU via `CUDA_VISIBLE_DEVICES`
- **Cache separata**: Ogni processo ha la sua directory cache

### 3.3 Isolamento e Sicurezza

```bash
# Processo GPU 0
CUDA_VISIBLE_DEVICES=0          # Vede solo GPU 0
OLLAMA_CACHE_DIR="$WORK/.ollama/cache/gpu0"  # Cache dedicata
OLLAMA_HOST=127.0.0.1:39001     # Porta dedicata

# Processo GPU 1  
CUDA_VISIBLE_DEVICES=1          # Vede solo GPU 1
OLLAMA_CACHE_DIR="$WORK/.ollama/cache/gpu1"  # Cache separata
OLLAMA_HOST=127.0.0.1:39002     # Porta diversa
```

## 4. FLUSSO DI UTILIZZO MEMORIA

### 4.1 Analisi Utilizzo VRAM per GPU

```
GPU A100-SXM4-64GB: 64GB VRAM totali
├── Sistema/Driver:     ~2GB   (riservato CUDA driver)
├── Ollama overhead:    ~1GB   (processo, cache, buffer)  
├── Mixtral modello:   ~45GB   (parametri + ottimizzazioni)
├── Context buffer:     ~4GB   (prompt/response in elaborazione)
├── Spazio libero:     ~12GB   (safety margin)
└── TOTAL:             64GB
```

**Utilizzo effettivo**: ~70-75% VRAM per GPU (come da configurazione `OLLAMA_MAX_VRAM_USAGE=0.75`)

### 4.2 Memoria RAM Sistema (256GB allocati)

```
RAM Sistema 256GB:
├── Sistema operativo:    ~8GB
├── Python script:       ~4GB
├── 4x Ollama processes: ~16GB  (4GB ciascuno)
├── Buffer I/O:          ~8GB
├── File temporanei:    ~20GB
├── Spazio libero:     ~200GB   (per cache, temp files, etc.)
└── TOTAL:             256GB
```

## 5. LOAD BALANCING E DISTRIBUZIONE RICHIESTE

### 5.1 Come il Python Script Distribuisce il Carico

```python
# File: ollama_ports.txt
39001,39002,39003,39004

# Python legge le porte e fa load balancing
class OllamaConnectionManager:
    def get_best_host(self) -> str:
        # Seleziona la GPU con minor carico
        return "http://127.0.0.1:39002"  # Esempio: GPU 1 meno occupata
```

### 5.2 Pattern di Distribuzione

```
Richiesta 1 → GPU 0 (Port 39001) → Mixtral inference → Response 1
Richiesta 2 → GPU 1 (Port 39002) → Mixtral inference → Response 2  
Richiesta 3 → GPU 2 (Port 39003) → Mixtral inference → Response 3
Richiesta 4 → GPU 3 (Port 39004) → Mixtral inference → Response 4
Richiesta 5 → GPU 0 (Port 39001) → ... (round-robin o load-based)
```

**Parallelismo**: 4 richieste possono essere elaborate **simultaneamente** (una per GPU)

## 6. VANTAGGI DI QUESTA ARCHITETTURA

### 6.1 **Scalabilità Lineare**
- 1 GPU = ~1 richiesta/secondo
- 4 GPU = ~4 richieste/secondo  
- Scaling quasi perfetto

### 6.2 **Fault Tolerance**
```
Se GPU 2 crasha:
├── GPU 0, 1, 3 continuano a funzionare
├── Throughput ridotto a 3/4 (75%)
├── Nessuna perdita di dati
└── Recovery automatico possibile
```

### 6.3 **Isolamento Completo**
- Nessuna condivisione di stato tra GPU
- Nessuna sincronizzazione necessaria
- Debug semplificato (log separati per GPU)

### 6.4 **Ottimizzazione Risorse**
- Ogni GPU utilizza la propria VRAM ottimalmente
- Nessun overhead di comunicazione inter-GPU
- Cache dedicata riduce I/O conflicts

## 7. CONFRONTO CON ALTERNATIVE

### 7.1 **Architettura Attuale vs. Distribuzione Tensor**

| Aspetto | **Multi-Istanza (Attuale)** | **Tensor Parallel** |
|---------|------------------------------|----------------------|
| Memoria modello | 4 × 45GB = 180GB | 1 × 45GB distribuito |
| Latenza | Bassa (no sync) | Media (sync inter-GPU) |
| Throughput | 4x parallelo | 1x (ma più veloce) |
| Fault tolerance | Alta | Bassa (single point) |
| Complessità | Bassa | Alta |

### 7.2 **Perché Multi-Istanza è Migliore per il Nostro Caso**

1. **Workload**: Molte richieste piccole/medie → parallelismo > velocità singola
2. **Reliability**: Sistema di produzione → fault tolerance critica  
3. **Sviluppo**: Meno complessità → manutenzione più semplice
4. **Risorse**: 4×64GB VRAM → spazio abbondante per copie multiple

## 8. MONITORING E DIAGNOSTICA

### 8.1 Come Verificare il Caricamento

```bash
# Verifica VRAM utilizzata per GPU
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv

# Output atteso:
# 0, 45234 MiB, 65536 MiB  # GPU 0: ~45GB utilizzati
# 1, 45156 MiB, 65536 MiB  # GPU 1: ~45GB utilizzati  
# 2, 45198 MiB, 65536 MiB  # GPU 2: ~45GB utilizzati
# 3, 45167 MiB, 65536 MiB  # GPU 3: ~45GB utilizzati
```

### 8.2 Verifica Processi Ollama

```bash
ps aux | grep ollama

# Output atteso:
# user 12345 ollama serve (GPU 0, port 39001)
# user 12346 ollama serve (GPU 1, port 39002)
# user 12347 ollama serve (GPU 2, port 39003)  
# user 12348 ollama serve (GPU 3, port 39004)
```

### 8.3 Test Connectivity

```bash
# Test tutti gli endpoint
curl http://127.0.0.1:39001/api/tags  # GPU 0
curl http://127.0.0.1:39002/api/tags  # GPU 1
curl http://127.0.0.1:39003/api/tags  # GPU 2
curl http://127.0.0.1:39004/api/tags  # GPU 3
```

## RIEPILOGO ARCHITETTURALE

🔹 **Modello**: Mixtral:8x7b caricato **4 volte** (una copia completa per GPU)
🔹 **Storage**: Disco condiviso ($WORK/.ollama/models) → 4x VRAM GPU
🔹 **Ollama**: **4 processi indipendenti**, uno per GPU, porte separate  
🔹 **Load Balancing**: Python script distribuisce richieste tra le 4 istanze
🔹 **Scalabilità**: 4x throughput parallelo con fault tolerance integrata

Questa architettura **sacrifica memoria per ottenere throughput, fault tolerance e semplicità**, perfetta per workload di produzione con molte richieste parallele come l'analisi turistica.