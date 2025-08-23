import json
import random
import time
from typing import Optional, Dict, Any
import os, sys, argparse, logging
import pandas as pd
import requests
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import multiprocessing as mp
import itertools
from typing import List
from typing import Optional, Dict, Any, List, Tuple
import os, sys, argparse, logging
import pandas as pd
import requests
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import math
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from threading import Lock, Semaphore
import multiprocessing as mp
import itertools
import queue
import threading
from contextlib import contextmanager

# ============= CONFIGURAZIONI HPC OTTIMIZZATE =============

# Rate limiting per evitare sovraccarico
RATE_LIMIT_SEMAPHORE = None  # Inizializzato dinamicamente
N_GPUS = 4
MAX_CONCURRENT_REQUESTS = N_GPUS * 2  # 4 GPU × 2 richieste per GPU

PER_GPU_RATE_LIMIT = 4  # Richieste max per singola GPU
gpu_semaphores = {}  #  Semaforo per ogni GPU

# Configurazioni timeout ottimizzate
REQUEST_TIMEOUT = 240 # Per inferenze complesse su dataset grandi
BATCH_SAVE_INTERVAL = 500     
HEALTH_CHECK_INTERVAL = 300 
MAX_CONSECUTIVE_FAILURES = 20  # Max fallimenti consecutivi prima di pausa

# Configurazioni retry intelligenti
MAX_RETRIES_PER_REQUEST = 3
BACKOFF_BASE = 2
BACKOFF_MAX = 30
CIRCUIT_BREAKER_THRESHOLD = 50  # Fallimenti prima di circuit breaker

# Lock globali thread-safe
write_lock = Lock()
stats_lock = Lock()
health_lock = Lock()

# Statistiche globali per monitoraggio
global_stats = {
    'total_processed': 0,
    'total_errors': 0,
    'consecutive_failures': 0,
    'last_success_time': time.time(),
    'host_failures': {},
    'circuit_breaker_active': False
}
# -----------------------------------------------------------

# Funzione per inizializzare semafori per GPU
def setup_gpu_rate_limiting(hosts: List[str]):
    """Setup semafori individuali per ogni GPU"""
    global gpu_semaphores
    for host in hosts:
        gpu_semaphores[host] = Semaphore(PER_GPU_RATE_LIMIT)
    logger.info(f"Rate limiting: {PER_GPU_RATE_LIMIT} req/GPU, {len(hosts)} GPU")

# ============= CLASSI DI SUPPORTO =============

class CircuitBreaker:
    """Circuit Breaker pattern per gestire fallimenti a cascata"""
    
    def __init__(self, failure_threshold: int = 10, timeout: int = 300):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
    
    @contextmanager
    def call(self):
        if self.state == "OPEN":
            if time.time() - self.last_failure > self.timeout:
                self.state = "HALF_OPEN"
                logger.info("🔄 Circuit breaker: tentativo HALF_OPEN")
            else:
                raise Exception("Circuit breaker OPEN - sistema in pausa")
        
        try:
            yield
            if self.state == "HALF_OPEN":
                self.reset()
        except Exception as e:
            self.record_failure()
            raise e
    
    def record_failure(self):
        self.failures += 1
        self.last_failure = time.time()
        
        if self.failures >= self.failure_threshold:
            self.state = "OPEN"
            logger.error(f"⚠️ Circuit breaker APERTO dopo {self.failures} fallimenti")
            global_stats['circuit_breaker_active'] = True
    
    def reset(self):
        self.failures = 0
        self.state = "CLOSED"
        global_stats['circuit_breaker_active'] = False
        logger.info("✅ Circuit breaker RESET")

class HostHealthMonitor:
    """Monitora la salute degli host Ollama con ROUND-ROBIN"""
    
    def __init__(self, hosts: List[str]):
        self.hosts = hosts
        self.health_status = {host: True for host in hosts}
        self.last_check = {host: 0 for host in hosts}
        self.response_times = {host: [] for host in hosts}
        self.request_counts = {host: 0 for host in hosts}  # NUOVO: conta richieste per host
        self.current_host_index = 0  # NUOVO: per round-robin
        self._lock = Lock()
    
    def is_healthy(self, host: str) -> bool:
        with self._lock:
            return self.health_status.get(host, False)
    
    def get_healthy_hosts(self) -> List[str]:
        with self._lock:
            return [host for host, healthy in self.health_status.items() if healthy]
    
    def check_health(self, host: str) -> bool:
        """Verifica ottimizzata per HPC"""
        try:
            start_time = time.time()
            resp = requests.get(f"{host}/api/tags", timeout=3, stream=False)
            response_time = time.time() - start_time
            
            with self._lock:
                self.response_times[host].append(response_time)
                self.response_times[host] = self.response_times[host][-5:]
                
                is_healthy = resp.status_code == 200
                self.health_status[host] = is_healthy
                self.last_check[host] = time.time()
                
                return is_healthy
        except Exception:
            with self._lock:
                self.health_status[host] = False
            return False
    
    def get_next_host_round_robin(self) -> str:
        """NUOVO: Selezione round-robin per distribuzione equa"""
        healthy_hosts = self.get_healthy_hosts()
        if not healthy_hosts:
            return None
        
        with self._lock:
            # Prova a selezionare il prossimo host sano
            attempts = 0
            while attempts < len(healthy_hosts):
                host = self.hosts[self.current_host_index % len(self.hosts)]
                self.current_host_index += 1
                
                if host in healthy_hosts:
                    self.request_counts[host] += 1
                    logger.debug(f"🔄 Selezionato {host} (requests: {self.request_counts[host]})")
                    return host
                
                attempts += 1
            
            # Fallback al primo host sano se round-robin fallisce
            return healthy_hosts[0]
    
    def get_least_loaded_host(self) -> str:
        """Seleziona l'host con meno richieste elaborate"""
        healthy_hosts = self.get_healthy_hosts()
        if not healthy_hosts:
            return None
        
        with self._lock:
            # Trova l'host con meno richieste
            host_loads = [(host, self.request_counts.get(host, 0)) for host in healthy_hosts]
            return min(host_loads, key=lambda x: x[1])[0]

    def get_best_host(self) -> str:
        """MODIFICATO: Usa strategia mista per bilanciamento"""
        healthy_hosts = self.get_healthy_hosts()
        if not healthy_hosts:
            return None
        
        # STRATEGIA: 70% round-robin, 30% least-loaded per ottimizzazione
        import random
        if random.random() < 0.7:
            return self.get_next_host_round_robin()
        else:
            return self.get_least_loaded_host()
    
    def get_load_stats(self) -> dict:
        """NUOVO: Statistiche di carico per debugging"""
        with self._lock:
            return dict(self.request_counts)

# Istanze globali
circuit_breaker = CircuitBreaker()
health_monitor = None  # Inizializzato dopo setup hosts

# Lock globale per scrittura thread-safe sui file CSV
write_lock = Lock()

# -----------------------------------------------------------
TOP_K  = 5          # deve coincidere con top_k del prompt
MODEL_NAME = "mixtral:8x7b"
# -----------------------------------------------------------
# -----------------------------------------------------------
DEFAULT_ANCHOR_RULE = "penultimate"   # "penultimate" | "first" | "middle" | int

# ---------------- logging setup ----------------
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def should_skip_file(visits_path: Path, poi_path: Path, out_dir: Path, append: bool = False) -> bool:
    """
    Verifica se il file è già completamente processato e può essere saltato.
    
    Args:
        visits_path: Path del file delle visite
        poi_path: Path del file dei POI 
        out_dir: Directory di output
        append: Se siamo in modalità append
    
    Returns:
        True se il file può essere saltato, False altrimenti
    """
    if not append:
        return False
    
    try:
        # 1. Verifica se esiste almeno un file di output
        if not latest_output(visits_path, out_dir):
            logger.debug(f"📄 {visits_path.stem}: Nessun output esistente")
            return False
        
        # 2. Conta le card processate dal checkpoint (veloce)
        processed_cards = load_completed_cards_fast(visits_path, out_dir)
        if not processed_cards:
            logger.debug(f"📄 {visits_path.stem}: Nessuna card nel checkpoint")
            return False
        
        # 3. Conta le card eleggibili nel file (simula il preprocessing senza clustering)
        logger.debug(f"🔍 {visits_path.stem}: Controllo card eleggibili...")
        
        # Caricamento rapido (solo colonne necessarie)
        visits_df = pd.read_csv(
            visits_path,
            usecols=[0, 1, 2, 4],  # Le stesse colonne di load_visits
            names=["data", "ora", "name_short", "card_id"],
            header=0,
            dtype={"card_id": str}
        )
        
        # Carica POI (solo name_short per il merge)
        pois_df = pd.read_csv(poi_path, usecols=["name_short"])
        
        # Simula il preprocessing senza fare il clustering
        # Merge con POI validi
        valid_visits = visits_df[visits_df["name_short"].isin(pois_df["name_short"])]
        
        # Conta visite per card e filtra quelle con 2+ POI diversi
        card_stats = valid_visits.groupby("card_id").agg({
            "name_short": "nunique",  # POI diversi visitati
            "data": "count"           # Numero totale visite
        }).rename(columns={"data": "total_visits", "name_short": "unique_pois"})
        
        # Card valide: almeno 2 POI diversi (per multi-visit) E almeno 3 visite totali
        eligible_cards = card_stats[
            (card_stats["unique_pois"] > 1) & 
            (card_stats["total_visits"] >= 3)
        ].index.tolist()
        
        total_eligible = len(eligible_cards)
        total_processed = len(processed_cards)
        
        logger.info(f"📊 {visits_path.stem}: {total_processed}/{total_eligible} card processate")
        
        # 4. Verifica completamento
        if total_processed >= total_eligible:
            logger.info(f"✅ {visits_path.stem}: File completamente processato - SKIP")
            return True
        else:
            remaining = total_eligible - total_processed
            logger.info(f"🔄 {visits_path.stem}: {remaining} card rimanenti - PROCESSO")
            return False
            
    except Exception as e:
        logger.warning(f"⚠️ Errore controllo skip per {visits_path.stem}: {e}")
        logger.warning("🔄 Procedo comunque per sicurezza...")
        return False

def test_ollama_connection_multi(hosts: List[str], model: str = MODEL_NAME) -> bool:
    """Test di tutti gli host configurati"""
    working_hosts = []
    
    for host in hosts:
        logger.info(f"Test {host}...")
        if test_single_host(host, model):
            working_hosts.append(host)
            logger.info(f"{host} operativo")
        else:
            logger.error(f"{host} non funziona")
    
    if working_hosts:
        logger.info(f"{len(working_hosts)}/{len(hosts)} host operativi")
        return True
    else:
        logger.error("Nessun host funzionante")
        return False

def test_single_host(host: str, model: str) -> bool:
    """Test singolo host"""
    try:
        # Test tags
        resp = requests.get(f"{host}/api/tags", timeout=10)
        if resp.status_code != 200:
            return False
            
        models = [m.get('name', '') for m in resp.json().get('models', [])]
        if model not in models:
            return False
            
        # Test micro inference
        test_resp = requests.post(
            f"{host}/api/generate",
            json={
                "model": model,
                "prompt": "Hi",
                "stream": False,
                "options": {"num_predict": 1, "temperature": 0}
            },
            timeout=60
        )
        
        if test_resp.status_code == 200:
            data = test_resp.json()
            return data.get("done") and data.get("response")
                
    except Exception:
        pass
    return False

# ---------- chiamata LLaMA / Ollama ---------------------------------------
def get_chat_completion(prompt: str, model: str = MODEL_NAME, max_retries: int = 3) -> str | None:
    """Versione robusta con circuit breaker e backoff intelligente"""
    
    # Prova tutti gli host disponibili prima di fare retry
    available_hosts = [h for h in OLLAMA_HOSTS if is_host_healthy(h)]
    
    if not available_hosts:
        logger.error("❌ Nessun host disponibile")
        return None
    
    for attempt in range(1, max_retries + 1):
        # Seleziona host con round-robin tra quelli sani
        current_host = available_hosts[attempt % len(available_hosts)]
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}], 
            "stream": False,
            "options": {
                "num_ctx": 4096,
                "temperature": 0.1,
                "num_thread": 8,  # Riduci thread per GPU
                "num_predict": 256
            }
        }
        
        try:
            logger.debug(f"🔄 Richiesta a {current_host} (tentativo {attempt}/{max_retries})")
            
            # Timeout progressivo
            timeout = 60 + (attempt * 30)  # 60s, 90s, 120s
            
            resp = requests.post(
                f"{current_host}/api/chat", 
                json=payload, 
                timeout=timeout,
                headers={'Content-Type': 'application/json'}
            )
            
            # Gestione specifica per errore 503
            if resp.status_code == 503:
                logger.warning(f"⚠️ Servizio non disponibile su {host} (503)")
                mark_host_unhealthy(host, duration=30)
                continue
            
            if resp.status_code == 500:
                logger.warning(f"⚠️ Server error 500 su {current_host}, provo altro host...")
                mark_host_unhealthy(current_host, duration=60)  # Marca come non sano per 1min
                continue
            
            resp.raise_for_status()
            response_data = resp.json()
            
            if not response_data.get("done", False):
                logger.warning(f"⚠️ Risposta incompleta da {current_host}")
                continue
            
            content = response_data.get("message", {}).get("content", "")
            if content:
                mark_host_healthy(current_host)  # Ripristina salute
                return content
                
        except requests.exceptions.Timeout:
            logger.error(f"⏰ Timeout {timeout}s su {current_host}")
            mark_host_unhealthy(current_host, duration=30)
        except Exception as exc:
            logger.error(f"❌ Errore su {current_host}: {exc}")
            mark_host_unhealthy(current_host, duration=30)
        
        if attempt < max_retries:
            wait_time = min(2 ** attempt, 30)  # Exponential backoff: 2s, 4s, 8s...
            logger.info(f"⏳ Aspetto {wait_time}s prima del prossimo tentativo...")
            time.sleep(wait_time)
    
    logger.error("❌ Tutti i tentativi falliti su tutti gli host")
    return None



def is_host_healthy(host: str) -> bool:
    health_info = HOST_HEALTH.get(host, {"healthy": True, "unhealthy_until": 0})
    if not health_info["healthy"] and time.time() < health_info["unhealthy_until"]:
        return False
    health_info["healthy"] = True
    return True

def mark_host_unhealthy(host: str, duration: int = 60):
    HOST_HEALTH[host] = {
        "healthy": False, 
        "unhealthy_until": time.time() + duration
    }

def mark_host_healthy(host: str):
    HOST_HEALTH[host] = {"healthy": True, "unhealthy_until": 0}

def _make_request_with_retry(prompt: str, model: str, max_retries: int) -> str | None:
    """Logica retry con backoff esponenziale"""
    
    for attempt in range(1, max_retries + 1):
        # Seleziona host più performante
        host = health_monitor.get_next_host_round_robin()
        if not host:
            logger.error("❌ Nessun host sano disponibile")
            healthy_hosts = health_monitor.get_healthy_hosts()
            if not healthy_hosts:
                # Re-check tutti gli host
                for h in OLLAMA_HOSTS:
                    health_monitor.check_health(h)
                host = health_monitor.get_best_host()
                if not host:
                    raise Exception("Tutti gli host non funzionanti")
        
        try:
            # Health check rapido se necessario
            if time.time() - health_monitor.last_check.get(host, 0) > HEALTH_CHECK_INTERVAL:
                if not health_monitor.check_health(host):
                    logger.warning(f"⚠️ Host {host} non sano - retry")
                    continue
            
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}], 
                "stream": False,
                "options": {
                    "num_ctx": 8192,      # Sfrutta i 64GB di VRAM
                    "num_predict": 300,
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_thread": 16,     # Meglio per 56 cores per socket
                    "num_batch": 1024,    # Batch più grandi per A100
                    "repeat_penalty": 1.1 # Evita ripetizioni
                }
            }
            
            start_time = time.time()
            resp = requests.post(
                f"{host}/api/chat", 
                json=payload, 
                timeout=REQUEST_TIMEOUT,
                headers={'Content-Type': 'application/json'}
            )
            response_time = time.time() - start_time
            
            resp.raise_for_status()
            response_data = resp.json()
            
            if not response_data.get("done", False):
                logger.warning(f"⚠️ Risposta incompleta da {host}")
                continue
            
            content = response_data.get("message", {}).get("content", "")
            if content:
                # Aggiorna statistiche successo
                with stats_lock:
                    global_stats['total_processed'] += 1
                    global_stats['consecutive_failures'] = 0
                    global_stats['last_success_time'] = time.time()
                
                logger.debug(f"✅ Risposta da {host} in {response_time:.2f}s")
                return content
                
        except requests.exceptions.Timeout:
            logger.warning(f"⚠️ Timeout {REQUEST_TIMEOUT}s su {host} (tentativo {attempt})")
            health_monitor.health_status[host] = False
        except Exception as exc:
            logger.error(f"❌ Errore su {host}: {exc}")
            health_monitor.health_status[host] = False
            
            # Aggiorna statistiche errore
            with stats_lock:
                global_stats['total_errors'] += 1
                global_stats['consecutive_failures'] += 1
        
        # Backoff esponenziale tra retry
        if attempt < max_retries:
            backoff_time = min(BACKOFF_BASE ** attempt, BACKOFF_MAX)
            time.sleep(backoff_time)
    
    # Tutti i tentativi falliti
    logger.error(f"❌ Tutti i {max_retries} tentativi falliti")
    return None

def _make_request_with_retry_targeted(prompt: str, model: str, max_retries: int, target_host: str) -> str | None:
    """Versione che usa un host specifico invece di selezionarlo dinamicamente"""
    
    for attempt in range(1, max_retries + 1):
        # Verifica che l'host target sia ancora sano
        if not health_monitor.is_healthy(target_host):
            logger.warning(f"⚠️ Host target {target_host} non sano - verifico...")
            if not health_monitor.check_health(target_host):
                logger.error(f"❌ Host {target_host} confermato non funzionante")
                return None
        
        try:
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}], 
                "stream": False,
                "options": {
                    "num_ctx": 8192,
                    "num_predict": 300,
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_thread": 16,
                    "num_batch": 1024,
                    "repeat_penalty": 1.1
                }
            }
            
            start_time = time.time()
            resp = requests.post(
                f"{target_host}/api/chat", 
                json=payload, 
                timeout=REQUEST_TIMEOUT,
                headers={'Content-Type': 'application/json'}
            )
            response_time = time.time() - start_time
            
            resp.raise_for_status()
            response_data = resp.json()
            
            if not response_data.get("done", False):
                logger.warning(f"⚠️ Risposta incompleta da {target_host}")
                continue
            
            content = response_data.get("message", {}).get("content", "")
            if content:
                # Aggiorna statistiche successo
                with stats_lock:
                    global_stats['total_processed'] += 1
                    global_stats['consecutive_failures'] = 0
                    global_stats['last_success_time'] = time.time()
                
                logger.debug(f"✅ Risposta da {target_host} in {response_time:.2f}s")
                return content
                
        except requests.exceptions.Timeout:
            logger.warning(f"⚠️ Timeout {REQUEST_TIMEOUT}s su {target_host} (tentativo {attempt})")
            health_monitor.health_status[target_host] = False
        except Exception as exc:
            logger.error(f"❌ Errore su {target_host}: {exc}")
            health_monitor.health_status[target_host] = False
            
            # Aggiorna statistiche errore
            with stats_lock:
                global_stats['total_errors'] += 1
                global_stats['consecutive_failures'] += 1
        
        # Backoff esponenziale tra retry
        if attempt < max_retries:
            backoff_time = min(BACKOFF_BASE ** attempt, BACKOFF_MAX)
            time.sleep(backoff_time)
    
    logger.error(f"❌ Tutti i {max_retries} tentativi falliti per {target_host}")
    return None

def debug_gpu_status():
    """Debug dello stato GPU prima di iniziare"""
    logger.info("🔍 Debug stato GPU:")
    
    try:
        import subprocess
        
        # Stato GPU
        gpu_info = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total,temperature.gpu,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        
        if gpu_info.returncode == 0:
            logger.info(f"📊 GPU: {gpu_info.stdout.strip()}")
        else:
            logger.warning("⚠️  nvidia-smi non disponibile")
            
        # Processi GPU
        gpu_procs = subprocess.run(
            ["nvidia-smi", "pmon", "-c", "1"],
            capture_output=True, text=True, timeout=10
        )
        
        if gpu_procs.returncode == 0:
            lines = gpu_procs.stdout.strip().split('\n')
            if len(lines) > 2:  # Header + data
                logger.info("📋 Processi GPU attivi:")
                for line in lines[2:]:  # Skip headers
                    if line.strip():
                        logger.info(f"   {line}")
            else:
                logger.info("✓ Nessun processo GPU concorrente")
                
    except Exception as e:
        logger.warning(f"⚠️  Debug GPU fallito: {e}")

def monitor_gpu_utilization():
    """Monitora utilizzo GPU durante l'esecuzione"""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", 
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            avg_util = sum(int(line.split(',')[0]) for line in lines) / len(lines)
            return avg_util
    except:
        pass
    return 0

def warmup_all_hosts(model: str = MODEL_NAME) -> bool:
    """Warm-up parallelo di tutti gli host"""
    logger.info("🔥 Warm-up parallelo di tutti i host...")
    
    def warmup_single_host(host):
        payload = {
            "model": model,
            "prompt": "Test",
            "stream": False,
            "options": {
                "num_ctx": 1024,
                "num_predict": 5,
                "temperature": 0.1
            }
        }
        try:
            resp = requests.post(f"{host}/api/generate", json=payload, timeout=60)
            return resp.status_code == 200
        except:
            return False
    
    with ThreadPoolExecutor(max_workers=len(OLLAMA_HOSTS)) as executor:
        futures = [executor.submit(warmup_single_host, host) for host in OLLAMA_HOSTS]
        results = [f.result() for f in futures]
    
    success_count = sum(results)
    logger.info(f"✅ Warm-up completato: {success_count}/{len(OLLAMA_HOSTS)} host")
    return success_count > 0

def warmup_model(model: str = MODEL_NAME) -> bool:
    """
    Warm-up ottimizzato per problemi GPU
    """
    logger.info("🔥 Warm-up modello con parametri conservativi...")
    
    # Payload minimalista per warm-up
    payload = {
        "model": model,
        "prompt": "Hi",  # Prompt più corto possibile
        "stream": False,
        "options": {
            "num_ctx": 1024,      # Contesto minimo
            "num_predict": 3,     # Solo 3 token
            "temperature": 0.1,
            "num_thread": 4,      # Thread ridotti
            "num_batch": 64       # Batch piccolo
        }
    }
    
    try:
        logger.info("🔄 Tentativo warm-up...")
        resp = requests.post(
            f"{OLLAMA_HOSTS[0]}/api/generate",
            json=payload,
            timeout=120,  # 2 minuti per warm-up
            headers={'Content-Type': 'application/json'}
        )
        
        if resp.status_code == 200 and resp.content:
            try:
                result = resp.json()
                done = result.get("done", False)
                response_text = result.get("response", "")
                
                logger.info(f"📊 Warm-up result: done={done}, response_len={len(response_text)}")
                
                if done and response_text.strip():
                    logger.info("✓ Warm-up completato con successo")
                    return True
                elif result.get("done_reason") == "load":
                    logger.warning("⚠️  Modello non completamente caricato durante warm-up")
                    return False
                else:
                    logger.warning("⚠️  Warm-up parziale - potrebbe funzionare comunque")
                    return True  # Ritorna True per tentare comunque
                    
            except json.JSONDecodeError:
                logger.error(f"❌ Warm-up JSON malformato: {resp.text[:200]}")
                return False
        else:
            logger.warning(f"⚠️  Warm-up HTTP error: {resp.status_code}")
            return False
        
    except requests.exceptions.Timeout:
        logger.error("❌  Warm-up timeout - GPU molto lenta o bloccata")
        return False
    except Exception as exc:
        logger.error(f"❌  Warm-up errore: {exc}")
        return False

# --- CONFIGURAZIONE OLLAMA: leggi porta dal file ---
OLLAMA_PORT_FILE = "ollama_port.txt"

def setup_ollama_connections() -> List[str]:
    """Setup connessioni ottimizzato per HPC"""
    try:
        with open("ollama_ports.txt", "r") as f:
            ports_str = f.read().strip()
        
        if "," in ports_str:
            ports = ports_str.split(",")
            hosts = [f"http://127.0.0.1:{port.strip()}" for port in ports]
            logger.info(f"🔥 Configurazione multi-GPU: {len(hosts)} istanze")
            
            # Inizializza rate limiting basato sugli host reali
            global RATE_LIMIT_SEMAPHORE, MAX_CONCURRENT_REQUESTS
            MAX_CONCURRENT_REQUESTS = len(hosts) * 4  # 4 richieste per GPU/host
            RATE_LIMIT_SEMAPHORE = Semaphore(MAX_CONCURRENT_REQUESTS)
            
            setup_gpu_rate_limiting(hosts)  # Inizializza semafori per GPU
            return hosts
        else:
            host = f"http://127.0.0.1:{ports_str}"
            logger.info(f"⚠️ Fallback singola GPU: {host}")
            RATE_LIMIT_SEMAPHORE = Semaphore(1)
            MAX_CONCURRENT_REQUESTS = 1
            
            setup_gpu_rate_limiting(hosts)  # Inizializza semafori per GPU
            return [host]
            
    except FileNotFoundError:
        raise RuntimeError("❌ File ollama_ports.txt non trovato")

def wait_for_ollama(ollama_host, max_attempts=30, wait_interval=3):
    """Attende che Ollama sia pronto con retry più robusto"""
    print(f"🔄 Attesa Ollama su {ollama_host}...")
    
    for attempt in range(1, max_attempts + 1):
        try:
            # Prima prova un endpoint semplice
            response = requests.get(f"{ollama_host}/api/tags", 
                                  timeout=10,
                                  headers={'Accept': 'application/json'})
            
            if response.status_code == 200:
                print(f"✓ Ollama risponde con status {response.status_code}")
                
                # Test aggiuntivo: prova anche /api/version
                try:
                    version_resp = requests.get(f"{ollama_host}/api/version", timeout=5)
                    if version_resp.status_code == 200:
                        print("✓ Runner LLaMA completamente attivo")
                        return True
                except:
                    pass  # Non critico se version non risponde
                
                # Anche se version non risponde, tags OK è sufficiente
                print("✓ Runner LLaMA attivo (solo /api/tags)")
                return True
            else:
                print(f"🔄 Tentativo {attempt}/{max_attempts}: HTTP {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print(f"🔄 Tentativo {attempt}/{max_attempts}: Connessione rifiutata")
        except requests.exceptions.Timeout:
            print(f"🔄 Tentativo {attempt}/{max_attempts}: Timeout")
        except requests.exceptions.RequestException as e:
            print(f"🔄 Tentativo {attempt}/{max_attempts}: Errore {e}")
        
        if attempt < max_attempts:
            print(f"⏳ Attendo {wait_interval}s prima del prossimo tentativo...")
            time.sleep(wait_interval)
    
    return False

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calcola la distanza in chilometri tra due punti geografici
    usando la formula dell'haversine.
    """
    R = 6371  # Raggio della Terra in km
    
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c

def list_outputs(visits_path: Path, out_dir: Path) -> list[Path]:
    """Tutti i CSV già calcolati per quel file di visite."""
    pattern = f"{visits_path.stem}_pred_*.csv"
    return sorted(out_dir.glob(pattern))

def latest_output(visits_path: Path, out_dir: Path) -> Path | None:
    """L'output più recente (None se non esiste)."""
    outputs = list_outputs(visits_path, out_dir)
    return max(outputs, key=os.path.getmtime) if outputs else None

def anchor_index(seq_len: int, rule: str | int) -> int:
    """
    Restituisce l'indice (0‑based) del POI da usare come 'ancora'
    nella sequenza **senza** il target finale.

    Parameters
    ----------
    seq_len : int
        Lunghezza della sequenza considerata (escluso il target).
    rule : str | int
        Strategia di selezione:
        • "penultimate" → ultimo elemento del prefisso  
        • "first"       → indice 0  
        • "middle"      → seq_len // 2  
        • int           → indice esplicito (negativo ammesso)

    Raises
    ------
    ValueError se la regola è sconosciuta o l'indice è fuori range.
    """
    if rule == "penultimate":
        idx = seq_len - 1
    elif rule == "first":
        idx = 0
    elif rule == "middle":
        idx = seq_len // 2
    elif isinstance(rule, int):
        idx = rule if rule >= 0 else seq_len + rule
    else:
        raise ValueError(f"anchor_rule '{rule}' non valido")

    if not (0 <= idx < seq_len):
        raise ValueError("anchor index fuori range")
    return idx

# ---------- helper di caricamento -----------------------------------------
def already_processed(visits_path: Path, out_dir: Path) -> bool:
    """
    Ritorna True se esiste almeno un CSV di output per il file di visite.
    Esempio: dati_2014.csv  →  results/dati_2014_pred_*.csv
    """
    pattern = f"{visits_path.stem}_pred_*.csv"
    return any(out_dir.glob(pattern))

def load_pois(filepath: str | Path) -> DataFrame:
    df = pd.read_csv(filepath, usecols=["name_short", "latitude", "longitude"])
    
    logger.info(f"[load_pois] {len(df)} POI letti da {filepath}")
    return df

def list_visits_csv(base_dir="data/verona"):
    """
    Ritorna la lista dei CSV di visite (esclude vc_site.csv e qualsiasi file
    che contenga 'vc_site' nel nome).
    Cerca in tutte le sottocartelle di *base_dir*.
    """
    all_csv = Path(base_dir).rglob("*.csv")
    return [
        str(p) for p in all_csv
        if "vc_site.csv" not in p.name.lower()
        and "backup" not in str(p).lower()
    ]

def load_visits(filepath: str | Path) -> DataFrame:
    df = pd.read_csv(
        filepath,
        usecols=[0, 1, 2, 4],
        names=["data", "ora", "name_short", "card_id"],
        header=0,
        dtype={"card_id": "category", "name_short": "category"},
    )
    df["timestamp"] = pd.to_datetime(df["data"] + " " + df["ora"], format="%d-%m-%y %H:%M:%S")
    
    logger.info(f"[load_visits] {len(df)} timbrature da {filepath}")

    return df[["timestamp", "card_id", "name_short"]].sort_values("timestamp").reset_index(drop=True)

def merge_visits_pois(visits_df: DataFrame, pois_df: DataFrame) -> DataFrame:
    merged = visits_df.merge(pois_df[["name_short"]], on="name_short", how="inner")
    
    logger.info(f"[merge] visite valide dopo merge: {len(merged)}")

    return merged.sort_values("timestamp").reset_index(drop=True)

def filter_multi_visit_cards(df: DataFrame) -> DataFrame:
    valid_cards = df.groupby("card_id")["name_short"].nunique().loc[lambda s: s > 1].index
    logger.info(f"[filter] card multi-visita: {len(valid_cards)} / {df.card_id.nunique()}")
    return df[df["card_id"].isin(valid_cards)].reset_index(drop=True)

def create_user_poi_matrix(df: DataFrame) -> DataFrame:
    return pd.crosstab(df["card_id"], df["name_short"])

# ---------- prompt builder -------------------------------------------------
def create_prompt_with_cluster(
    df: pd.DataFrame,
    user_clusters: pd.DataFrame,
    pois_df: pd.DataFrame,
    card_id: str,
    *,
    top_k: int = 5,
    anchor_rule: str | int = DEFAULT_ANCHOR_RULE,
) -> str:
    visits = df[df["card_id"] == card_id].sort_values("timestamp")
    seq = visits["name_short"].tolist()

    if len(seq) < 3:
        raise ValueError("Sequenza troppo corta (minimo 3 tappe)")

    target = seq[-1]
    prefix = seq[:-1]
    idx = anchor_index(len(prefix), anchor_rule)
    current_poi = prefix[idx]
    history = [p for i, p in enumerate(prefix) if i != idx]

    cluster_id = user_clusters.loc[
        user_clusters["card_id"] == card_id, "cluster"
    ].values[0]

    # Ottieni coordinate del POI attuale
    current_poi_row = pois_df[pois_df["name_short"] == current_poi]
    if current_poi_row.empty:
        raise ValueError(f"POI {current_poi} non trovato nel database")
    
    current_lat = current_poi_row["latitude"].iloc[0]
    current_lon = current_poi_row["longitude"].iloc[0]

    # Calcola distanze e prendi solo i più vicini (max 10 per ridurre il prompt)
    nearby_pois = []
    for _, row in pois_df.iterrows():
        poi_name = row["name_short"]
        
        # Salta se già visitato o è il POI attuale
        if poi_name in history or poi_name == current_poi:
            continue
            
        distance = calculate_distance(
            current_lat, current_lon,
            row["latitude"], row["longitude"]
        )
        
        nearby_pois.append({
            "name": poi_name,
            "distance": distance
        })

    # Ordina per distanza e prendi solo i primi 10
    nearby_pois.sort(key=lambda x: x["distance"])
    nearby_pois = nearby_pois[:10]

    # Crea lista compatta
    pois_list = ", ".join([
        f"{poi['name']} ({poi['distance']:.1f}km)"
        for poi in nearby_pois
    ])

    # Prompt molto più conciso
    return f"""Turista cluster {cluster_id} a Verona.
Visitati: {', '.join(history) if history else 'nessuno'}
Attuale: {current_poi}
POI Più Vicini: {pois_list}

Suggerisci {top_k} POI più probabili come prossime visite considerando distanze e pattern turistici.
Rispondi SOLO JSON: {{"prediction": ["poi1", "poi2", ...], "reason": "breve spiegazione"}}"""
    
def get_nearby_pois(current_poi: str, pois_df: pd.DataFrame, 
                   visited_pois: list, max_distance: float = 2.0) -> list:
    """
    Restituisce i POI entro una certa distanza dal POI attuale,
    escludendo quelli già visitati.
    
    Args:
        current_poi: Nome del POI attuale
        pois_df: DataFrame con tutti i POI e le loro coordinate
        visited_pois: Lista dei POI già visitati
        max_distance: Distanza massima in km (default 2km per il centro di Verona)
    
    Returns:
        Lista di dizionari con informazioni sui POI vicini
    """
    current_poi_row = pois_df[pois_df["name_short"] == current_poi]
    if current_poi_row.empty:
        return []
    
    current_lat = current_poi_row["latitude"].iloc[0]
    current_lon = current_poi_row["longitude"].iloc[0]
    
    nearby_pois = []
    for _, row in pois_df.iterrows():
        poi_name = row["name_short"]
        
        # Salta se già visitato o è il POI attuale
        if poi_name in visited_pois or poi_name == current_poi:
            continue
            
        distance = calculate_distance(
            current_lat, current_lon,
            row["latitude"], row["longitude"]
        )
        
        # Include solo se entro la distanza massima
        if distance <= max_distance:
            nearby_pois.append({
                "name": poi_name,
                "distance": distance,
                "lat": row["latitude"],
                "lon": row["longitude"]
            })
    
    return sorted(nearby_pois, key=lambda x: x["distance"])    

def analyze_movement_patterns(df: pd.DataFrame, pois_df: pd.DataFrame) -> pd.DataFrame:
    """
    Analizza i pattern di movimento calcolando le distanze tra visite consecutive.
    Questo può aiutare a capire i comportamenti tipici dei turisti.
    """
    movement_data = []
    
    for card_id, group in df.groupby("card_id"):
        visits = group.sort_values("timestamp")
        poi_sequence = visits["name_short"].tolist()
        
        # Calcola distanze tra visite consecutive
        for i in range(len(poi_sequence) - 1):
            current_poi = poi_sequence[i]
            next_poi = poi_sequence[i + 1]
            
            # Trova coordinate dei due POI
            current_coords = pois_df[pois_df["name_short"] == current_poi]
            next_coords = pois_df[pois_df["name_short"] == next_poi]
            
            if not current_coords.empty and not next_coords.empty:
                distance = calculate_distance(
                    current_coords["latitude"].iloc[0],
                    current_coords["longitude"].iloc[0],
                    next_coords["latitude"].iloc[0],
                    next_coords["longitude"].iloc[0]
                )
                
                movement_data.append({
                    "card_id": card_id,
                    "from_poi": current_poi,
                    "to_poi": next_poi,
                    "distance": distance,
                    "visit_order": i + 1
                })
    
    return pd.DataFrame(movement_data)

# ---------- test-set builder ----------------------------------------------
def build_test_set(df: DataFrame) -> DataFrame:
    records = []
    for cid, grp in df.groupby("card_id"):
        seq = grp.sort_values("timestamp")["name_short"].tolist()
        if len(seq) >= 3:
            records.append(
                {"card_id": cid, "history": seq[:-2], "current": seq[-2], "target": seq[-1]}
            )
    return pd.DataFrame(records)

# ---------- Funzione WORKER ---------------------------------------
def process_single_card(args):
    """
    Worker function per processare una singola card.
    Ritorna un dizionario con i risultati o None se errore.
    """
    cid, filtered, user_clusters, pois, anchor_rule, top_k = args
    
    start_time = time.time()
    
    try:
        # Validazione preliminare
        seq = (
            filtered.loc[filtered.card_id == cid]
            .sort_values("timestamp")["name_short"].tolist()
        )
        
        if len(seq) < 3:
            return None
            
        target = seq[-1]
        idx_anchor = anchor_index(len(seq) - 1, anchor_rule)
        history_list = [p for i, p in enumerate(seq[:-1]) if i != idx_anchor]
        current_poi = seq[:-1][idx_anchor]

        # Creazione prompt ottimizzato
        prompt = create_prompt_with_cluster(
            filtered, user_clusters, pois, cid,  
            top_k=top_k, anchor_rule=anchor_rule
        )
        
        # Chiamata LLM con retry intelligente
        ans = get_chat_completion(prompt)

        # Preparazione record risultato
        rec = {
            "card_id": cid,
            "cluster": get_user_cluster(user_clusters, cid),
            "history": str(history_list),
            "current_poi": current_poi,
            "prediction": None,
            "ground_truth": target,
            "reason": None,
            "hit": False,
            "processing_time": time.time() - start_time,
            "status": "success" if ans else "failed"
        }

        if ans:
            try:
                obj = json.loads(ans)
                pred = obj["prediction"]
                pred_lst = pred if isinstance(pred, list) else [pred]
                rec["prediction"] = str(pred_lst)
                rec["reason"] = obj.get("reason", "")[:200]  # Limita lunghezza
                rec["hit"] = target in pred_lst
            except json.JSONDecodeError:
                logger.warning(f"⚠️ JSON malformato per card {cid}: {ans[:100]}...")
                rec["prediction"] = "PARSE_ERROR"
            except Exception as e:
                logger.warning(f"⚠️ Errore parsing per card {cid}: {e}")
                rec["prediction"] = "PROCESSING_ERROR"
                
        return rec
        
    except Exception as e:
        logger.error(f"❌ Errore grave processando card {cid}: {e}")
        return {
            "card_id": cid,
            "cluster": None,
            "history": None,
            "current_poi": None,
            "prediction": "FATAL_ERROR",
            "ground_truth": None,
            "reason": str(e)[:200],
            "hit": False,
            "processing_time": time.time() - start_time,
            "status": "fatal_error"
        }

def log_gpu_metrics():
    """Log periodico dello stato GPU"""
    try:
        import subprocess
        result = subprocess.run([
            "nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu",
            "--format=csv,noheader,nounits"
        ], capture_output=True, text=True, timeout=5)
        
        if result.returncode == 0:
            for i, line in enumerate(result.stdout.strip().split('\n')):
                mem_used, mem_total, util, temp = line.split(',')[1:]
                logger.info(f"GPU{i}: {mem_used}/{mem_total}MB ({util}% util, {temp}°C)")
    except:
        pass
    
# ---------- test su un singolo file ---------------------------------------
def run_on_visits_file(
    visits_path: Path, 
    poi_path: Path, 
    *, 
    max_users: int | None = None, 
    force: bool = False, 
    append: bool = False, 
    anchor_rule: str | int = "penultimate"
) -> None:
    """Versione ottimizzata per HPC con gestione avanzata degli errori"""
    
    global health_monitor
    
    # Inizializza monitor salute
    health_monitor = HostHealthMonitor(OLLAMA_HOSTS)
    
    def log_load_balancing_stats():
        """Log periodico delle statistiche di load balancing"""
        while True:
            time.sleep(300)  # Log ogni 5 minuti
            if health_monitor:
                stats = health_monitor.get_load_stats()
                healthy = health_monitor.get_healthy_hosts()
                total_requests = sum(stats.values())
                
                logger.info("📊 LOAD BALANCING STATS:")
                for host in OLLAMA_HOSTS:
                    requests = stats.get(host, 0)
                    percentage = (requests / max(total_requests, 1)) * 100
                    status = "✅" if host in healthy else "❌"
                    logger.info(f"   {host}: {requests} req ({percentage:.1f}%) {status}")
                
                # Check squilibrio
                if len(healthy) > 1 and total_requests > 50:
                    req_counts = [stats.get(host, 0) for host in healthy]
                    max_req = max(req_counts)
                    min_req = min(req_counts)
                    imbalance = (max_req - min_req) / max(max_req, 1) * 100
                    
                    if imbalance > 30:  # Soglia 30% di squilibrio
                        logger.warning(f"⚠️ Load imbalance: {imbalance:.1f}% tra GPU")


    load_monitor_thread = threading.Thread(target=log_load_balancing_stats, daemon=True)
    load_monitor_thread.start()
    
    # Setup output directory
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    
    # Check skip logica (come originale)
    if should_skip_file(visits_path, poi_path, out_dir, append):
        return
    
    # Load e preprocessing (invariato)
    logger.info("📊 Caricamento e preprocessing dati...")
    pois = load_pois(poi_path)
    visits = load_visits(visits_path)
    merged = merge_visits_pois(visits, pois)
    filtered = filter_multi_visit_cards(merged)
    
    # Clustering (invariato)
    matrix = create_user_poi_matrix(filtered)
    clusters = KMeans(n_clusters=7, random_state=42, n_init=10)\
              .fit_predict(StandardScaler().fit_transform(matrix))
    user_clusters = pd.DataFrame({"card_id": matrix.index, "cluster": clusters})
    
    # Selezione utenti
    eligible = (
        filtered.groupby("card_id").size()
        .loc[lambda s: s >= 3].index.tolist()
    )
    
    if max_users is None:
        demo_cards = eligible
    else:
        demo_cards = random.sample(eligible, k=min(max_users, len(eligible)))
    
    logger.info(f"🎯 Processamento {len(demo_cards)} carte")
    
    # Setup file output
    if append:
        output_file = latest_output(visits_path, out_dir)
        if output_file is None:
            ts = time.strftime("%Y%m%d_%H%M%S")
            output_file = out_dir / f"{visits_path.stem}_pred_{ts}.csv"
            write_header = True
        else:
            write_header = False
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_file = out_dir / f"{visits_path.stem}_pred_{ts}.csv"
        write_header = True
    
    # Calcola workers ottimali
    optimal_workers = min(MAX_CONCURRENT_REQUESTS * 2, 64)  # 2x per I/O overlap
    
    logger.info(f"🚀 Avvio elaborazione con {optimal_workers} worker ottimizzati")
    logger.info(f"⚡ Rate limit: {MAX_CONCURRENT_REQUESTS} richieste concurrent")
    logger.info(f"🔄 Timeout richiesta: {REQUEST_TIMEOUT}s")
    logger.info(f"💾 Salvataggio ogni: {BATCH_SAVE_INTERVAL} carte")
    
    # Prepara argomenti
    card_args = [
        (cid, filtered, user_clusters, pois, anchor_rule, 5) 
        for cid in demo_cards
    ]
    
    batch_buffer = []
    processed_count = 0
    start_time = time.time()
    last_health_check = time.time()
    
    # Elaborazione parallela ottimizzata
    with ThreadPoolExecutor(max_workers=optimal_workers, thread_name_prefix="Worker") as executor:
        
        # Sottometti tutti i job
        future_to_card = {
            executor.submit(process_single_card, args): args[0] 
            for args in card_args
        }
        
        with tqdm(total=len(demo_cards), desc="🔄 Processamento", unit="card") as pbar:
            for future in as_completed(future_to_card, timeout=None):
                cid = future_to_card[future]
                
                try:
                    # Timeout per singola carta (più permissivo)
                    rec = future.result(timeout=REQUEST_TIMEOUT + 60)
                    
                    if rec and rec.get('status') != 'fatal_error':
                        batch_buffer.append(rec)
                        processed_count += 1
                        
                        # Health check periodico
                        if time.time() - last_health_check > HEALTH_CHECK_INTERVAL:
                            for host in OLLAMA_HOSTS:
                                health_monitor.check_health(host)
                            last_health_check = time.time()
                        
                        # Salvataggio batch
                        if len(batch_buffer) >= BATCH_SAVE_INTERVAL:
                            save_batch(batch_buffer, output_file, write_header, append)
                            batch_buffer.clear()
                            write_header = False
                            
                            # Log progresso con statistiche
                            elapsed = time.time() - start_time
                            rate = processed_count / elapsed * 3600 if elapsed > 0 else 0
                            
                            with stats_lock:
                                success_rate = (global_stats['total_processed'] / 
                                              max(global_stats['total_processed'] + global_stats['total_errors'], 1) * 100)
                            
                            logger.info(f"📊 Progresso: {processed_count}/{len(demo_cards)} "
                                      f"({rate:.1f} cards/h, success: {success_rate:.1f}%)")
                    
                    elif rec:
                        logger.warning(f"⚠️ Card {cid} con errore: {rec.get('status')}")
                    
                except TimeoutError:
                    logger.error(f"⏰ Timeout processing card {cid}")
                except Exception as e:
                    logger.error(f"❌ Errore future per card {cid}: {e}")
                
                pbar.update(1)
                
                # Gestione circuit breaker
                with stats_lock:
                    if global_stats['consecutive_failures'] >= CIRCUIT_BREAKER_THRESHOLD:
                        logger.error("🚨 Troppi fallimenti consecutivi - pausa sistema")
                        time.sleep(60)  # Pausa di 1 minuto
                        global_stats['consecutive_failures'] = 0
    
    # Salvataggio finale
    if batch_buffer:
        save_batch(batch_buffer, output_file, write_header, append)
    
    # Report finale
    elapsed_total = time.time() - start_time
    final_rate = processed_count / elapsed_total * 3600 if elapsed_total > 0 else 0
    
    with stats_lock:
        total_requests = global_stats['total_processed'] + global_stats['total_errors']
        success_rate = (global_stats['total_processed'] / max(total_requests, 1) * 100)
    
    logger.info(f"✅ Completato: {processed_count} carte in {elapsed_total:.1f}s")
    logger.info(f"⚡ Rate finale: {final_rate:.1f} cards/h")
    logger.info(f"📊 Success rate: {success_rate:.1f}%")

def save_batch(batch_buffer: list, output_file: Path, write_header: bool, append: bool):
    """Salvataggio batch ottimizzato e thread-safe"""
    if not batch_buffer:
        return
    
    try:
        with write_lock:
            df_batch = pd.DataFrame(batch_buffer)
            
            # Ottimizzazioni per performance I/O
            if write_header and not append:
                df_batch.to_csv(output_file, mode="w", header=True, index=False, 
                               encoding='utf-8', compression=None)
            else:
                df_batch.to_csv(output_file, mode="a", header=False, index=False,
                               encoding='utf-8', compression=None)
            
            logger.debug(f"💾 Salvato batch di {len(batch_buffer)} risultati")
            
    except Exception as e:
        logger.error(f"❌ Errore salvataggio batch: {e}")
        # Backup dei dati in caso di errore
        backup_file = output_file.parent / f"backup_{output_file.stem}_{int(time.time())}.json"
        try:
            with open(backup_file, 'w') as f:
                json.dump(batch_buffer, f)
            logger.info(f"💾 Backup salvato in {backup_file}")
        except:
            logger.error("❌ Impossibile salvare anche il backup!")

# ---------- test su tutti i file ------------------------------------------
def run_all_verona_logs(max_users: int | None = None, force=False, append=False, anchor_rule: str | int = DEFAULT_ANCHOR_RULE) -> None:
    """
    Versione ottimizzata che salta file già completati
    """
    ROOT = Path(__file__).resolve().parent
    poi_csv = ROOT / "data" / "verona" / "vc_site.csv"

    visit_csvs = [
        p for p in (ROOT / "data" / "verona").rglob("*.csv")
        if p.name != "vc_site.csv"
    ]
    if not visit_csvs:
        raise RuntimeError("Nessun CSV di visite trovato sotto data/verona/")

    # Statistiche globali
    total_files = len(visit_csvs)
    skipped_files = 0
    processed_files = 0

    logger.info(f"🎯 Trovati {total_files} file da elaborare")
    
    for csv in sorted(visit_csvs):
        logger.info(f"\n🔍 Controllo {csv.name}...")
        
        # Il controllo skip è già integrato in run_on_visits_file
        # ma possiamo fare un pre-check per le statistiche
        out_dir = Path(__file__).resolve().parent / "results"
        if append and should_skip_file(csv, poi_csv, out_dir, append):
            skipped_files += 1
            continue
        
        try:
            run_on_visits_file(csv, poi_csv,
                             max_users=max_users,
                             force=force,
                             append=append,
                             anchor_rule=anchor_rule)
            processed_files += 1
        except Exception as e:
            logger.error(f"❌ Errore elaborando {csv.name}: {e}")
            continue
    
    # Statistiche finali
    logger.info("\n" + "=" * 70)
    logger.info(f"📈 STATISTICHE FINALI:")
    logger.info(f"   • File totali: {total_files}")
    logger.info(f"   • File saltati: {skipped_files}")
    logger.info(f"   • File elaborati: {processed_files}")
    logger.info(f"   • Efficienza: {skipped_files/total_files*100:.1f}% file evitati")
    logger.info("=" * 70)

def get_completed_cards(file_path: Path) -> set:
    """
    Restituisce il set delle card_id che hanno predizioni COMPLETE 
    (non None/vuote) nel file esistente.
    """
    if not file_path.exists():
        return set()
    
    try:
        df = pd.read_csv(file_path)
        # Considera complete solo le card con prediction non nulla/vuota
        completed = df[
            df['prediction'].notna() & 
            (df['prediction'] != '') & 
            (df['prediction'] != 'None')
        ]['card_id'].unique()
        
        logger.info(f"📊 Trovate {len(completed)} card già completate in {file_path.name}")
        return set(completed)
    except Exception as e:
        logger.error(f"❌ Errore lettura file esistente: {e}")
        return set()

def get_user_cluster(user_clusters: pd.DataFrame, card_id: str) -> int:
    """
    Restituisce il cluster ID per un determinato card_id.
    
    Args:
        user_clusters: DataFrame con le colonne 'card_id' e 'cluster'
        card_id: ID della card di cui cercare il cluster
        
    Returns:
        int: ID del cluster
        
    Raises:
        ValueError: Se la card_id non viene trovata
    """
    matching_rows = user_clusters[user_clusters.card_id == card_id]
    
    if matching_rows.empty:
        raise ValueError(f"Card ID {card_id} non trovata nei cluster")
    
    # Usa .iloc[0] su un DataFrame filtrato è più chiaro per il type checker
    return int(matching_rows["cluster"].iloc[0])

def get_checkpoint_file(visits_path: Path, out_dir: Path) -> Path:
    """Restituisce il path del file checkpoint per questo dataset"""
    return out_dir / f"{visits_path.stem}_checkpoint.txt"

def load_completed_cards_fast(visits_path: Path, out_dir: Path) -> set:
    """
    Carica velocemente le card completate dal file checkpoint.
    Fallback su CSV solo se checkpoint non esiste.
    """
    checkpoint_file = get_checkpoint_file(visits_path, out_dir)
    
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                completed_cards = {line.strip() for line in f if line.strip()}
            logger.info(f"⚡ Checkpoint: {len(completed_cards)} card già completate (lettura veloce)")
            return completed_cards
        except Exception as e:
            logger.warning(f"⚠️ Errore lettura checkpoint: {e}, fallback su CSV")
    
    # Fallback: scansione CSV con controllo robusto
    latest_csv = latest_output(visits_path, out_dir)
    if latest_csv:
        logger.info("🐌 Prima esecuzione append: scansiono CSV esistente...")
        try:
            # PRIMA leggi solo l'header per verificare le colonne
            df_sample = pd.read_csv(latest_csv, nrows=0)  # Solo header
            available_columns = df_sample.columns.tolist()
            
            # Verifica che le colonne necessarie esistano
            required_columns = ['card_id', 'prediction']
            missing_columns = [col for col in required_columns if col not in available_columns]
            
            if missing_columns:
                logger.warning(f"⚠️ Colonne mancanti nel CSV: {missing_columns}")
                logger.warning(f"📋 Colonne disponibili: {available_columns}")
                return set()
            
            # Se le colonne ci sono, procedi con la lettura completa
            df = pd.read_csv(latest_csv, usecols=required_columns)
            completed = df[
                df['prediction'].notna() & 
                (df['prediction'] != '') & 
                (df['prediction'] != 'None') &
                (df['prediction'] != 'NO_RESPONSE') &
                (~df['prediction'].str.startswith('ERROR', na=False)) &
                (~df['prediction'].str.startswith('PROCESSING_ERROR', na=False))
            ]['card_id'].unique()
            
            completed_set = set(completed)
            save_checkpoint(checkpoint_file, completed_set)
            logger.info(f"💾 Checkpoint salvato: {len(completed_set)} card")
            
            return completed_set
        except Exception as e:
            logger.error(f"❌ Errore lettura CSV: {e}")
            return set()
    
    return set()

def save_checkpoint(checkpoint_file: Path, completed_cards: set):
    """Salva il checkpoint delle card completate"""
    try:
        with open(checkpoint_file, 'w') as f:
            for card_id in sorted(completed_cards):
                f.write(f"{card_id}\n")
    except Exception as e:
        logger.warning(f"⚠️ Errore salvataggio checkpoint: {e}")

def update_checkpoint_incremental(checkpoint_file: Path, new_completed_cards: list):
    """Aggiorna il checkpoint con nuove card completate (append mode)"""
    if not new_completed_cards:
        return
    
    try:
        with open(checkpoint_file, 'a') as f:
            for card_id in new_completed_cards:
                f.write(f"{card_id}\n")
        logger.debug(f"📝 Checkpoint aggiornato con {len(new_completed_cards)} nuove card")
    except Exception as e:
        logger.warning(f"⚠️ Errore aggiornamento checkpoint: {e}")
        
def debug_file_processing(visits_path: Path, poi_path: Path):
    """Funzione di debug per verificare che i dati vengano processati correttamente"""
    print(f"🔍 Debug per {visits_path.name}")
    
    # Verifica i dati base
    pois = load_pois(poi_path)
    visits = load_visits(visits_path)
    merged = merge_visits_pois(visits, pois)
    filtered = filter_multi_visit_cards(merged)
    
    print(f"📊 POI: {len(pois)}, Visite: {len(visits)}, Filtrate: {len(filtered)}")
    
    # Verifica utenti idonei
    eligible = (
        filtered.groupby("card_id").size()
        .loc[lambda s: s >= 3].index.tolist()
    )
    print(f"👥 Utenti idonei (≥3 visite): {len(eligible)}")
    
    # Mostra esempio di sequenza
    if eligible:
        sample_card = eligible[0]
        seq = (
            filtered.loc[filtered.card_id == sample_card]
            .sort_values("timestamp")["name_short"].tolist()
        )
        print(f"📝 Esempio sequenza per {sample_card}: {seq}")
    
    return len(eligible) > 0

def run_single_file(file_path: str, max_users: int | None = None, 
                   force: bool = False, append: bool = False, 
                   anchor_rule: str | int = DEFAULT_ANCHOR_RULE) -> None:
    """
    Processa un singolo file specificato dall'utente.
    
    Args:
        file_path: Path del file da processare (relativo o assoluto)
        max_users: Numero massimo di utenti da processare
        force: Forza il ricalcolo anche se esistono output
        append: Riprende da dove si era interrotto
        anchor_rule: Regola per l'anchor POI
    """
    ROOT = Path(__file__).resolve().parent
    poi_csv = ROOT / "data" / "verona" / "vc_site.csv"
    
    # Converti il path in oggetto Path
    target_file = Path(file_path)
    
    # Se il path non è assoluto, prova a risolverlo relativamente alla directory base
    if not target_file.is_absolute():
        # Prima prova relativo alla directory corrente
        if not target_file.exists():
            # Poi prova relativo alla directory data/verona
            target_file = ROOT / "data" / "verona" / file_path
            if not target_file.exists():
                # Infine prova come nome file diretto nella directory verona
                target_file = ROOT / "data" / "verona" / target_file.name
    
    # Verifica che il file esista
    if not target_file.exists():
        logger.error(f"❌ File non trovato: {file_path}")
        logger.error(f"❌ Percorsi tentati:")
        logger.error(f"   • {Path(file_path)}")
        logger.error(f"   • {ROOT / 'data' / 'verona' / file_path}")
        logger.error(f"   • {ROOT / 'data' / 'verona' / Path(file_path).name}")
        return
    
    # Verifica che sia un CSV
    if not target_file.suffix.lower() == '.csv':
        logger.error(f"❌ Il file deve essere un CSV: {target_file}")
        return
    
    # Verifica che non sia il file POI
    if target_file.name.lower() == 'vc_site.csv':
        logger.error(f"❌ Non posso processare il file POI: {target_file}")
        return
    
    # Verifica che il file POI esista
    if not poi_csv.exists():
        logger.error(f"❌ File POI non trovato: {poi_csv}")
        return
    
    logger.info(f"🎯 Processamento file singolo: {target_file.name}")
    logger.info(f"📍 Path completo: {target_file}")
    
    try:
        run_on_visits_file(target_file, poi_csv,
                         max_users=max_users,
                         force=force,
                         append=append,
                         anchor_rule=anchor_rule)
        logger.info(f"✅ Processamento completato per {target_file.name}")
    except Exception as e:
        logger.error(f"❌ Errore processando {target_file.name}: {e}")
        raise

def wait_for_ollama_single(host: str, max_attempts=30, wait_interval=3):
    """Attende che un singolo host Ollama sia pronto"""
    print(f"Attesa Ollama su {host}...")
    
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(f"{host}/api/tags", 
                                  timeout=10,
                                  headers={'Accept': 'application/json'})
            
            if response.status_code == 200:
                print(f"✓ {host} risponde con status {response.status_code}")
                
                try:
                    version_resp = requests.get(f"{host}/api/version", timeout=5)
                    if version_resp.status_code == 200:
                        print(f"✓ {host} completamente attivo")
                        return True
                except:
                    pass
                
                print(f"✓ {host} attivo (solo /api/tags)")
                return True
            else:
                print(f"Tentativo {attempt}/{max_attempts}: HTTP {response.status_code} su {host}")
                
        except requests.exceptions.ConnectionError:
            print(f"Tentativo {attempt}/{max_attempts}: Connessione rifiutata su {host}")
        except requests.exceptions.Timeout:
            print(f"Tentativo {attempt}/{max_attempts}: Timeout su {host}")
        except requests.exceptions.RequestException as e:
            print(f"Tentativo {attempt}/{max_attempts}: Errore {e} su {host}")
        
        if attempt < max_attempts:
            print(f"Attendo {wait_interval}s prima del prossimo tentativo...")
            time.sleep(wait_interval)
    
    return False

# -----------------------------------------------------------
# Setup globale
try:
    OLLAMA_HOSTS = setup_ollama_connections()
    host_cycle = itertools.cycle(OLLAMA_HOSTS)
    HOST_HEALTH = {host: {"healthy": True, "unhealthy_until": 0} for host in OLLAMA_HOSTS}
    logger.info(f"🎯 Configurazione caricata: {len(OLLAMA_HOSTS)} host")
except Exception as e:
    logger.error(f"❌ Errore setup Ollama: {e}")
    sys.exit(1)

# ---------- MAIN -----------------------------------------------------------
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # Usa tutte e 4 le A100

if __name__ == "__main__":
        
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler("hpc_verona_optimized.log"),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    # Parser argomenti (semplificato)
    parser = argparse.ArgumentParser(description="VeronaCard HPC Ottimizzato")
    parser.add_argument("--file", type=str, help="File specifico da processare")
    parser.add_argument("--max-users", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--append", action="store_true")
    
    args = parser.parse_args()

    if args.force and args.append:
        parser.error("Non puoi usare insieme --force e --append.")

    # Attendi che tutti gli host Ollama siano pronti
    for i, host in enumerate(OLLAMA_HOSTS):
        print(f"Controllo host {i+1}/{len(OLLAMA_HOSTS)}: {host}")
        if not wait_for_ollama_single(host):
            raise RuntimeError(f"Host {host} non ha risposto dopo tutti i tentativi")

    print(f"Tutti gli {len(OLLAMA_HOSTS)} host Ollama sono pronti!")

    print("🎉 Connessione Ollama stabilita con successo!")

    # Controllo che OLLAMA risponda altrimenti esco 
    if not test_ollama_connection_multi(OLLAMA_HOSTS, MODEL_NAME):
        logger.error("Sistema multi-GPU non funziona, aborting")
        exit(1)

    logger.info("Sistema multi-GPU configurato correttamente!")

    try:
        # Decide se processare un singolo file o tutti i file
        if args.file:
            run_single_file(args.file,
                           max_users=args.max_users,
                           force=args.force,
                           append=args.append,
                           anchor_rule=DEFAULT_ANCHOR_RULE)
        else:
            run_all_verona_logs(max_users=args.max_users,
                              force=args.force,
                              append=args.append,
                              anchor_rule=DEFAULT_ANCHOR_RULE)
    except KeyboardInterrupt:
        logging.info("Interruzione manuale...")
        sys.exit(1)
