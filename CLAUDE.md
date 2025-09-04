# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System Overview

This is **LLM-Mob**, a tourist mobility prediction system using Large Language Models on HPC infrastructure. The system predicts next destinations for tourists based on visit history, spatial proximity, and temporal patterns using the VeronaCard dataset.

### Core Architecture
- **Multi-GPU Ollama**: 4x NVIDIA A100 64GB instances running LLMs (Qwen2.5:7b,Qwen2.5:14b, Llama3.1 8b, Mixtral 8x7B, deepseek coder)
- **Parallel Processing**: ThreadPoolExecutor with intelligent load balancing across GPUs
- **Checkpoint System**: Resume interrupted processing with state management
- **Circuit Breaker**: Failure protection for distributed system reliability

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
# Submit job to SLURM
sbatch base_4_GPU.sh          # Base version
sbatch geom_4_GPU.sh          # With geospatial features
sbatch time_4_GPU.sh          # With temporal analysis

# Monitor jobs
squeue -u $USER
tail -f slurm-<JOBID>.out
```

### Main Execution Scripts
```bash
# Process all data files
python veronacard_mob_with_geom_time_parrallel.py

# Process specific file with limits
python veronacard_mob_with_geom_time_parrallel.py --file dati_2014.csv --max-users 1000

# Resume from checkpoint
python veronacard_mob_with_geom_time_parrallel.py --append

# Force complete reprocessing
python veronacard_mob_with_geom_time_parrallel.py --force
```

## Project Structure

### Main Processing Scripts
- `veronacard_mob_with_geom_time_parrallel.py` - Full version with temporal + geospatial
- `veronacard_mob_with_geom_parrallel.py` - Geospatial features only
- `veronacard_mob_versione_base_parrallel.py` - Base prediction version

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
results/model/type_of_promt/
├── <filename>_pred_<timestamp>.csv    # Predictions with hit rates
└── <filename>_checkpoint.txt          # Processing state
```

## Key Configuration Parameters

### GPU Optimization (Config class)
- `MAX_CONCURRENT_REQUESTS = 12` - Total parallel requests (4 GPUs × 3)
- `MAX_CONCURRENT_PER_GPU = 3` - Optimized for A100 64GB VRAM
- `REQUEST_TIMEOUT = 180` - Reduced for A100 performance
- `BATCH_SAVE_INTERVAL = 1000` - Checkpoint frequency

### Ollama Payload Optimization
```python
payload_options = {
    "num_ctx": 8192,           # Extended context window
    "num_predict": 1024,       # More tokens for detailed predictions
    "num_thread": 112,         # All Sapphire Rapids cores (2x56)
    "num_batch": 8192,         # Optimal batch for 64GB VRAM
    "cache_type_k": "f16",     # FP16 cache for A100 speed
    "mirostat": 2,             # Output quality control
}
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

The system uses advanced prompts with three context types:
- **Tourist Profile**: Cluster assignment and visit history
- **Temporal Context**: Current time, usual hours, day patterns
- **Spatial Context**: Nearby POI with walking distances

Example temporal features:
```python
temporal_features = {
    "timestamp": pd.to_datetime(),
    "hour": timestamp.dt.hour,        # 0-23
    "day_of_week": timestamp.dt.day_name(),  # Monday, Tuesday, etc.
    "usual_hours": [10, 14, 16],      # User's typical visit hours
}
```

## Performance Monitoring

### Jupyter Analysis Notebooks
```bash
# Main analysis notebook
jupyter notebook notebook/singole_metriche_updated.ipynb

# Model comparison
jupyter notebook notebook/multi_model_comparison_analysis.ipynb

# CSV export for visualization
jupyter notebook notebook/export_csv_for_canva.ipynb
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

- **Never modify Config parameters** without understanding HPC implications
- **Always use --append flag** when resuming interrupted jobs to avoid data loss
- **Checkpoint files are critical** - they contain processing state for resume
- **GPU memory is precious** - batch sizes are optimized for 64GB A100 VRAM
- **Temporal analysis is core** - the system extracts and uses time patterns extensively
- **Multi-GPU coordination** requires careful semaphore and lock management

## Common Issues and Solutions

- **Ollama timeout**: Check that all 4 instances are running on ports 11434-11437
- **CUDA OOM**: Reduce `num_batch` or `MAX_CONCURRENT_PER_GPU` in Config
- **Circuit breaker open**: System overloaded, wait for automatic reset
- **Checkpoint corruption**: Remove checkpoint file and use `--force` flag
- **Missing POI coordinates**: Ensure `vc_site.csv` is properly loaded

## Environment Variables

```bash
# HPC Leonardo specific
CUDA_VISIBLE_DEVICES=0,1,2,3
OLLAMA_HOST="127.0.0.1"
WORK="/leonardo_work/IscrC_LLM-Mob"
```