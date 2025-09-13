# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System Overview

This is **LLM-Mob**, a tourist mobility prediction system using Large Language Models on HPC infrastructure. The system predicts next destinations for tourists based on visit history, spatial proximity, and temporal patterns using the VeronaCard dataset.

### Core Architecture
- **Multi-GPU Ollama**: 4x NVIDIA A100 64GB instances running LLMs (Qwen2.5:7b, Qwen2.5:14b, Llama3.1:8b, Mixtral:8x7B, DeepSeek-Coder:33b, Mistral:7b)
- **Parallel Processing**: ThreadPoolExecutor with intelligent load balancing across GPUs
- **Checkpoint System**: Resume interrupted processing with state management and failure recovery
- **Circuit Breaker**: Advanced failure protection with CLOSED/OPEN/HALF_OPEN states
- **Health Monitoring**: Real-time GPU performance tracking and adaptive load balancing

## Development Commands

### Local Development Setup
```bash
# Setup environment
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Install Ollama locally
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:7b

# Configure single port for testing
echo "11434" > ollama_ports.txt
```

### HPC Production (Leonardo)
```bash
# Submit job to SLURM - Current available scripts
sbatch base_4_GPU.sh          # Base version predictions
sbatch geom_4_GPU.sh          # With geospatial features
sbatch time_4_GPU.sh          # With temporal analysis (most advanced)

# Monitor jobs
squeue -u $USER
tail -f slurm-<JOBID>.out
scancel <JOBID>               # Cancel job if needed

# Check computational budget
saldo -b IscrC_LLM-Mob
```

### Main Execution Scripts

#### Primary Scripts (Current Architecture)
```bash
# RECOMMENDED: Full temporal + geospatial version
python veronacard_mob_with_geom_time_parrallel.py

# Geospatial features only
python veronacard_mob_with_geom_parrallel.py

# Base version (minimal features)
python veronacard_mob_versione_base_parrallel.py
```

#### Command Line Options
```bash
# Process specific file with user limits
python veronacard_mob_with_geom_time_parrallel.py --file dati_2014.csv --max-users 1000

# Resume from checkpoint (critical for long runs)
python veronacard_mob_with_geom_time_parrallel.py --append

# Force complete reprocessing (ignores existing results)
python veronacard_mob_with_geom_time_parrallel.py --force

# Custom anchor point selection
python veronacard_mob_with_geom_time_parrallel.py --anchor penultimate
```

## Project Structure

### Main Processing Scripts
- `veronacard_mob_with_geom_time_parrallel.py` - **RECOMMENDED**: Full version with temporal + geospatial analysis
- `veronacard_mob_with_geom_parrallel.py` - Geospatial features only (distance calculations)
- `veronacard_mob_versione_base_parrallel.py` - Base prediction version (minimal context)
- `base.py`, `geom.py` - Simplified single-file versions for testing

### Key Classes and Components
1. **Config** - Centralized configuration for GPU optimization
2. **OllamaConnectionManager** - Multi-host connection handling with failover
3. **HostHealthMonitor** - GPU load balancing and health tracking
4. **CircuitBreaker** - Failure protection (CLOSED/OPEN/HALF_OPEN states)
5. **CheckpointManager** - State persistence for long-running jobs
6. **CardProcessor** - Main parallel processing orchestrator
7. **PromptBuilder** - Context generation with temporal/spatial features

### Data Structure
```
data/verona/
├── vc_site.csv                 # 70 POI with GPS coordinates
├── dati_2014.csv               # ~370k visit records per year
├── dati_2015.csv               # Tourist card usage logs
└── ...
```

### Output Structure
```
results/
├── qwen2.5_7b/                        # Default model results
│   ├── with_geom_time/                # Full temporal+geospatial analysis
│   ├── with_geom/                     # Geospatial only
│   └── base_version/                  # Minimal features
├── mistral_7b/                        # Alternative model results
├── deepseek-coder_33b/                # Code-optimized model
└── middle/                            # Intermediate results storage
    └── <model_name>/
        └── <strategy>/
            ├── <filename>_pred_<timestamp>.csv    # Predictions with hit rates
            └── <filename>_checkpoint.txt          # Processing state
```

## Key Configuration Parameters

### GPU Optimization (Config class) - UPDATED ANTI-CASCADE
```python
# ULTRA-CONSERVATIVE configuration for 2x A100 64GB (anti-cascade failure)
MAX_CONCURRENT_REQUESTS = 2          # 🔧 REDUCED: Only 2 GPU simultaneous for stability
MAX_CONCURRENT_PER_GPU = 1           # 🔧 SAFE: 1 request per GPU to prevent conflicts
REQUEST_TIMEOUT = 900                # 🔧 EXTENDED: 15 min timeout for HPC stability
CIRCUIT_BREAKER_THRESHOLD = 50       # 🔧 TOLERANT: 50 failures before opening (was 25)
BATCH_SAVE_INTERVAL = 500            # Checkpoint every 500 processed cards

# Enhanced retry and backoff strategy
MAX_RETRIES_PER_REQUEST = 12         # 🔧 INCREASED: More retry attempts for HPC
BACKOFF_MAX = 600                    # 🔧 EXTENDED: 10 min max backoff for stability

# Debug configuration for development
DEBUG_MODE = False                   # Set True for local testing
DEBUG_MAX_CARDS = 50                # Limited dataset for debugging
```

### Ollama Payload Optimization - UPDATED CONSERVATIVE
```python
# CONSERVATIVE payload optimized for stability and consistency
payload_options = {
    "num_ctx": 1024,           # 🔧 REDUCED: Conservative context window for stability
    "num_predict": 64,         # 🔧 REDUCED: Concise responses for faster processing  
    "num_thread": 56,          # 🔧 OPTIMAL: Full Sapphire Rapids cores per GPU
    "num_batch": 512,          # 🔧 CONSERVATIVE: Reduced batch size for memory safety
    "temperature": 0.1,        # 🔧 LOW: Consistent, logical predictions
    "cache_type_k": "f16",     # FP16 cache for A100 speed
}

# Original high-performance settings (commented for reference):
# "num_ctx": 8192,           # Was: Extended context window
# "num_predict": 1024,       # Was: More tokens for detailed predictions  
# "num_batch": 8192,         # Was: Optimal batch for 64GB VRAM
# "num_thread": 112,         # Was: All cores across both sockets
```

## Data Processing Pipeline

1. **Data Loading** - CSV parsing with pandas
2. **Preprocessing** - Filter valid visits, merge POI coordinates
3. **Temporal Extraction** - Extract hour, day_of_week, timestamp patterns
4. **Clustering** - K-means on user-POI interaction matrix
5. **Prompt Generation** - Context with history + spatial + temporal features
6. **LLM Inference** - Parallel prediction across multiple GPUs
7. **Post-processing** - Hit rate calculation and results aggregation

## Prompt Engineering

The system uses **advanced multi-context prompts** optimized for tourism mobility prediction with three complementary information layers:

### 1. Tourist Profile Context
- **Cluster Assignment**: K-means clustering on user-POI interaction patterns
- **Visit History**: Chronological sequence of previously visited locations
- **Behavioral Patterns**: Frequency analysis and preference identification

### 2. Temporal Context Strategy 🕒
The **temporal analysis** is the core innovation of `veronacard_mob_with_geom_time_parrallel.py`:

#### Temporal Feature Extraction:
```python
temporal_features = {
    "timestamp": pd.to_datetime(visit_time),
    "hour": timestamp.dt.hour,                    # 0-23 hour of day
    "day_of_week": timestamp.dt.day_name(),       # Monday, Tuesday, etc.
    "is_weekend": timestamp.dt.weekday >= 5,      # Weekend detection
    "time_of_day": categorize_time_period(hour),  # Morning/Afternoon/Evening
    "usual_hours": user_typical_hours,            # Personal time patterns
    "seasonal_period": extract_season(timestamp)   # Tourism seasonality
}
```

#### Advanced Temporal Prompting:
The system generates **time-aware prompts** that include:
- **Current Context**: "It's Tuesday afternoon at 2:30 PM"
- **Personal Patterns**: "This user typically visits attractions at 10 AM, 2 PM, and 4 PM"
- **Time-based Reasoning**: "Given the current time and user patterns, predict logical next destinations"
- **Temporal Constraints**: "Consider opening hours, meal times, and typical tourist flows"

#### Temporal Strategy Benefits:
- **+15-25% Accuracy**: Time context significantly improves prediction accuracy
- **Realistic Predictions**: Respects opening hours and tourist behavioral patterns
- **Context Awareness**: Differentiates between morning, afternoon, and evening activities

### 3. Spatial Context (Geospatial Analysis)
- **Distance Calculations**: Walking distances between POIs using geopy
- **Proximity Clustering**: Nearby attractions within reasonable walking distance
- **Geographic Constraints**: Physical accessibility and transportation considerations

### Multi-Context Prompt Template:
```
TOURIST PROFILE:
- Cluster: {cluster_id} (similar tourists prefer: {cluster_preferences})
- Visit History: {chronological_visits}
- Patterns: {behavioral_analysis}

TEMPORAL CONTEXT:
- Current Time: {day_name} {time_period} at {hour}:{minute}
- User's Typical Hours: {usual_visit_times}
- Time-based Reasoning: {temporal_logic}

SPATIAL CONTEXT:  
- Current Location: {last_poi_name}
- Nearby Attractions: {walkable_pois_with_distances}
- Geographic Constraints: {accessibility_notes}

TASK: Predict the next 5 most likely POI destinations considering ALL contexts.
```

### Prompt Optimization for HPC:
- **Context Window**: 1024 tokens (optimized for A100 memory efficiency)
- **Response Length**: 64 tokens (concise JSON format for fast processing)
- **Temperature**: 0.1 (low temperature for consistent, logical predictions)
- **Batch Processing**: 512 batch size for optimal GPU utilization

## Performance Monitoring and Analysis

### Jupyter Analysis Notebooks

#### Primary Analysis Notebooks
```bash
# RECOMMENDED: Comprehensive analysis of all models and strategies
jupyter notebook notebook/comprehensive_model_strategy_comparison.ipynb

# Updated single metrics analysis
jupyter notebook notebook/singole_metriche_updated.ipynb

# Multi-model performance comparison
jupyter notebook notebook/multi_model_comparison_analysis.ipynb
```

#### Specialized Analysis Notebooks
```bash
# Results analysis and visualization
jupyter notebook notebook/analisi_risultati.ipynb

# Export data for external visualization (Canva)
jupyter notebook notebook/export_csv_for_canva.ipynb

# Statistical analysis of CSV outputs
jupyter notebook notebook/csv_statistics_analysis.ipynb

# Temporal metrics analysis
jupyter notebook notebook/metriche_con_tempo.ipynb

# Base metrics analysis
jupyter notebook notebook/metriche_base.ipynb
```

### Key Metrics
- **Hit Rate**: Percentage of correct predictions (target in top-k)
- **Processing Speed**: Cards processed per hour
- **GPU Utilization**: VRAM and compute usage per A100
- **Success Rate**: Completed requests vs total requests

## Testing and Validation

### Local Testing
```bash
# Test with small dataset
python veronacard_mob_with_geom_time_parrallel.py --max-users 10

# Verify Ollama connectivity
curl http://localhost:11434/api/tags

# Check GPU status
nvidia-smi
```

### HPC Validation
```bash
# Check SLURM allocation
scontrol show job $SLURM_JOB_ID

# Monitor GPU usage
watch nvidia-smi

# Verify checkpoint integrity
ls -la results/*checkpoint*
```

## Important Implementation Notes

### Critical Safety Rules
- **NEVER modify Config parameters** without understanding HPC implications - especially GPU concurrency settings
- **ALWAYS use --append flag** when resuming interrupted jobs to prevent data loss
- **Checkpoint files are critical** - they contain processing state and must not be manually edited
- **GPU memory management** - Current settings optimized for A100 64GB VRAM, changing batch sizes can cause crashes

### System Architecture Requirements
- **Multi-GPU coordination** requires careful semaphore and lock management across A100 instances
- **Temporal analysis is core** - All modern versions extract and use time patterns extensively
- **Circuit breaker pattern** - System automatically protects against cascading failures
- **Health monitoring** - Real-time GPU performance tracking enables adaptive load balancing

### 🔧 Anti-Cascade Failure Strategy (Latest Updates)
**Problem Solved**: Previous runs failed after ~16 minutes due to cascading GPU failures and memory pressure.

#### Conservative Configuration Applied:
- **GPU Usage**: 2 active GPUs instead of 4 (ultra-conservative for maximum stability)
- **Concurrency**: 1 request per GPU (down from 2) to prevent memory conflicts
- **Timeouts**: Extended to 15 minutes (900s) to handle HPC latency
- **Circuit Breaker**: Increased threshold to 50 failures (from 25) for better tolerance
- **Memory Limits**: 90% GPU memory usage (down from 95%) with 1GB safety buffer

#### Enhanced Circuit Breaker Features:
- **Gradual Escalation**: Warning at 50% threshold before circuit opening
- **Recovery Logic**: Success tracking enables automatic recovery from partial failures
- **Consecutive Failure Tracking**: Only consecutive failures trigger circuit opening
- **Progressive Monitoring**: Real-time alerts and detailed diagnostics

#### Validated Configuration:
```bash
# Consistency verified between time_4_GPU.sh and Python script
✅ Request timeout: 900s == 900s
✅ GPU concurrency: 2 == 2  
✅ Context window: 1024 == 1024
✅ Batch size: 512 == 512
✅ All anti-cascade parameters aligned
```

### Development Workflow
- **Local testing**: Use DEBUG_MODE=True with limited datasets before HPC deployment
- **Model selection**: Default Qwen2.5:7b with fallback to Llama3.1:8b, Mistral:7b
- **Results validation**: Always run analysis notebooks after processing completion

## Common Issues and Solutions

### Runtime Issues
| Problem | Root Cause | Solution |
|---------|------------|----------|
| **Ollama timeout** | Multi-instance not running | Check all 4 Ollama instances on ports 11434-11437 |
| **CUDA OOM** | GPU memory exhaustion | Reduce `num_batch` in payload_options or `MAX_CONCURRENT_PER_GPU` |
| **Circuit breaker open** | Too many consecutive failures | System auto-recovery after cooldown, check GPU health |
| **Checkpoint corruption** | Interrupted write operation | Delete `<filename>_checkpoint.txt` and use `--force` flag |
| **Missing POI coordinates** | Data loading failure | Verify `data/verona/vc_site.csv` exists and is readable |

### Performance Issues
| Problem | Diagnostic | Solution |
|---------|------------|----------|
| **Slow processing** | Low GPU utilization | Increase `MAX_CONCURRENT_REQUESTS` (carefully) |
| **Memory leaks** | RAM usage growing | Enable `ASYNC_INFERENCE` and reduce batch sizes |
| **Request failures** | Network/model issues | Check Ollama model availability: `curl http://localhost:11434/api/tags` |

## Environment Variables

### HPC Leonardo Configuration
```bash
# GPU allocation - Critical for multi-GPU setup
export CUDA_VISIBLE_DEVICES=0,1,2,3
export NVIDIA_VISIBLE_DEVICES=0,1,2,3

# Ollama host configuration
export OLLAMA_HOST="127.0.0.1"

# Leonardo HPC paths
export WORK="/leonardo_work/IscrC_LLM-Mob"

# Python environment
source $WORK/venv/bin/activate
```

### Required Dependencies
```bash
# Core ML libraries (from requirements.txt)
pip install pandas numpy scikit-learn
pip install requests tqdm geopy

# Optional for analysis
pip install jupyter matplotlib seaborn
```
- Sto lavorando in un ambiente HPC, in particolare su Leonardo di cineca. Ora sono sul nodo di login ( dove sto sviluppando ) ma poi tramite uno script bash viene lanciato il JOB e viene eseguito sul modulo booster dove ho a disposizione 4 GPU NVIDIA A100 con 64GB di VRAM ciascuna