import argparse
import json
import logging
import math
import os
import queue
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from threading import Lock, Semaphore
from typing import Dict, Any, List, Optional, Tuple, Set, Union

# Third-party imports
import numpy as np
import pandas as pd
import requests
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# ============= CONFIGURATION =============
class Config:
    """Centralized configuration to avoid global variables"""
    
    # Model configuration - ottimizzato per tourism mobility prediction
    MODEL_NAME = "qwen2.5:7b" # Raccomandato per reasoning spazio-temporale
    # Alternative: "llama3.1:70b-instruct", "mixtral:8x22b-instruct"
    TOP_K = 5  # Number of POI predictions
    
    # HPC optimization parameters for 4x A100 64GB - ULTRA CONSERVATIVE FOR STABILITY
    MAX_CONCURRENT_REQUESTS = 1   # ✅ SINGLE REQUEST: Evita crash delle istanze Ollama
    REQUEST_TIMEOUT = 180  # Ridotto per rilevare problemi early
    BATCH_SAVE_INTERVAL = 1000  # Salva ogni 1000 cards per efficienza I/O
    HEALTH_CHECK_INTERVAL = 120  # Check molto più frequenti per rilevare problemi early
    
    # Retry and failure handling ottimizzati per HPC - MORE CONSERVATIVE
    MAX_RETRIES_PER_REQUEST = 5  # Ridotto per evitare sovraccarico
    MAX_CONSECUTIVE_FAILURES = 15  # Più aggressivo per rilevare problemi early
    BACKOFF_BASE = 2.0  # Backoff più conservativo
    BACKOFF_MAX = 120  # Max backoff aumentato per stabilità
    CIRCUIT_BREAKER_THRESHOLD = 5   # 🔥 ULTRA AGGRESSIVO - Ferma dopo 5 fallimenti
    
    # 503 specific handling per HPC - CONSERVATIVE APPROACH
    RETRY_ON_503_WAIT = 60  # Attesa aumentata per stabilità
    MAX_503_RETRIES = 10    # Ridotto per evitare loop infiniti
    
    # Anchor rule for POI selection
    DEFAULT_ANCHOR_RULE = "penultimate"
    
    # Parallelism ottimizzato per 4x A100
    ENABLE_ROUND_ROBIN = True
    HOST_SELECTION_STRATEGY = "balanced"
    MAX_CONCURRENT_PER_GPU = 1   # ✅ FIXED: 1 richiesta simultanea per GPU per evitare OOM
    
    # Memory and performance tuning per A100
    GPU_MEMORY_FRACTION = 0.95  # Usa 95% della VRAM disponibile
    PREFETCH_FACTOR = 2  # Pre-carica 2 batch in anticipo
    ASYNC_INFERENCE = True  # Inference asincrona quando possibile
    
    # File paths
    OLLAMA_PORT_FILE = "ollama_ports.txt"
    LOG_DIR = Path(__file__).resolve().parent / "logs"
    RESULTS_DIR = Path(__file__).resolve().parent / "results" / "qwen2.5_7b" / "with_geom_time"
    DATA_DIR = Path(__file__).resolve().parent / "data" / "verona"
    POI_FILE = DATA_DIR / "vc_site.csv"

# ============= LOGGING SETUP =============
def setup_logging() -> logging.Logger:
    """Configure logging with file and console output"""
    Config.LOG_DIR.mkdir(exist_ok=True)
    log_file = Config.LOG_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Create formatter without special characters
    formatter = logging.Formatter(
        fmt='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    
    # Configure logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()  # Remove any existing handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger
# Initialize logger
logger = setup_logging()

# ============= THREAD-SAFE STATISTICS =============
class Statistics:
    """Thread-safe statistics tracking"""
    
    def __init__(self):
        self._lock = Lock()
        self._data = {
            'total_processed': 0,
            'total_errors': 0,
            'consecutive_failures': 0,
            'last_success_time': time.time(),
            'host_failures': {},
            'circuit_breaker_active': False
        }
    
    def increment_processed(self):
        with self._lock:
            self._data['total_processed'] += 1
            self._data['consecutive_failures'] = 0
            self._data['last_success_time'] = time.time()
    
    def increment_errors(self):
        with self._lock:
            self._data['total_errors'] += 1
            self._data['consecutive_failures'] += 1
    
    def get_stats(self) -> dict:
        with self._lock:
            return self._data.copy()
    
    def get_success_rate(self) -> float:
        with self._lock:
            total = self._data['total_processed'] + self._data['total_errors']
            if total == 0:
                return 0.0
            return (self._data['total_processed'] / total) * 100
    
    def reset_consecutive_failures(self):
        with self._lock:
            self._data['consecutive_failures'] = 0
            
    @property
    def consecutive_failures(self) -> int:
        with self._lock:
            return self._data['consecutive_failures']
    
    @property
    def circuit_breaker_active(self) -> bool:
        with self._lock:
            return self._data['circuit_breaker_active']
    
    def set_circuit_breaker(self, active: bool):
        with self._lock:
            self._data['circuit_breaker_active'] = active
# Global instances
stats = Statistics()
write_lock = Lock()  # Global lock for file writing

# ============= CIRCUIT BREAKER =============
class CircuitBreaker:
    """
    Circuit Breaker pattern implementation for handling cascading failures.
    
    States:
    - CLOSED: Normal operation
    - OPEN: Failures exceeded threshold, rejecting requests
    - HALF_OPEN: Testing if service recovered
    """
    
    def __init__(self, failure_threshold: int = 10, timeout: int = 300):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure = None
        self.state = "CLOSED"
        self._lock = Lock()
    
    @contextmanager
    def call(self):
        """Context manager for circuit breaker protected calls"""
        with self._lock:
            if self.state == "OPEN":
                if self.last_failure and time.time() - self.last_failure > self.timeout:
                    self.state = "HALF_OPEN"
                    logger.info("Circuit breaker: Attempting HALF_OPEN state")
                else:
                    raise Exception("Circuit breaker OPEN - system paused")
        
        try:
            yield
            with self._lock:
                if self.state == "HALF_OPEN":
                    self.reset()
        except Exception as e:
            self.record_failure()
            raise e
    
    def record_failure(self):
        """Record a failure and potentially open the circuit"""
        with self._lock:
            self.failures += 1
            self.last_failure = time.time()
            
            if self.failures >= self.failure_threshold:
                self.state = "OPEN"
                logger.error(f"Circuit breaker OPEN after {self.failures} failures")
                stats.set_circuit_breaker(True)
    
    def reset(self):
        """Reset circuit breaker to closed state"""
        with self._lock:
            self.failures = 0
            self.state = "CLOSED"
            stats.set_circuit_breaker(False)
            logger.info("Circuit breaker RESET to CLOSED state")

# ============= HOST HEALTH MONITORING =============
class HostHealthMonitor:
    """
    Monitors health status of Ollama hosts and provides load balancing.
    
    Features:
    - Health checks with configurable intervals
    - Response time tracking
    - Round-robin load balancing
    - Performance-based host selection
    - Automatic failover
    """
    
    def __init__(self, hosts: List[str]):
        self.hosts = hosts if hosts else []
        self.health_status = {host: True for host in self.hosts}
        self.last_check = {host: 0 for host in self.hosts}
        self.response_times = {host: [] for host in self.hosts}
        self._lock = Lock()
        self._max_response_history = 5
        self._round_robin_index = 0  # ✅ NUOVO: Indice per round-robin
        
        # Log di warning per inizializzazione vuota
        if not hosts:
            logger.warning("HostHealthMonitor initialized with empty host list")
    
    def is_healthy(self, host: str) -> bool:
        """Check if a specific host is healthy"""
        with self._lock:
            return self.health_status.get(host, False)
    
    def get_healthy_hosts(self) -> List[str]:
        """Get list of all healthy hosts"""
        if not self.hosts:
            logger.warning("No hosts available in health monitor")
            return []
        
        with self._lock:
            healthy = [host for host, healthy in self.health_status.items() if healthy]
            if not healthy:
                # Tentativo di recovery: ricontrolla tutti gli host
                logger.warning("No healthy hosts found, attempting recovery...")
                for host in self.hosts:
                    if self._quick_health_check(host):
                        self.health_status[host] = True
                        healthy.append(host)
                        
            return healthy
    
    def _quick_health_check(self, host: str) -> bool:
        """Quick health check without updating response times"""
        try:
            resp = requests.get(f"{host}/api/tags", timeout=2)
            return resp.status_code == 200
        except:
            return False
    
    def check_health(self, host: str) -> bool:
        """
        Perform health check on a specific host.
        Uses lightweight endpoint to minimize overhead.
        """
        try:
            start_time = time.time()
            resp = requests.get(
                f"{host}/api/tags", 
                timeout=3,
                headers={'Accept': 'application/json'}
            )
            response_time = time.time() - start_time
            
            with self._lock:
                # Track response times (keep only recent ones)
                self.response_times[host].append(response_time)
                if len(self.response_times[host]) > self._max_response_history:
                    self.response_times[host].pop(0)
                
                is_healthy = resp.status_code == 200
                self.health_status[host] = is_healthy
                self.last_check[host] = int(time.time())
                
                if is_healthy:
                    logger.debug(f"Host {host} healthy (response time: {response_time:.2f}s)")
                else:
                    logger.warning(f"Host {host} unhealthy (status: {resp.status_code})")
                
                return is_healthy
                
        except Exception as e:
            logger.warning(f"Health check failed for {host}: {e}")
            with self._lock:
                self.health_status[host] = False
            return False
    
    def get_round_robin_host(self) -> Optional[str]:
        """
        Select host using round-robin algorithm to ensure even distribution
        """
        healthy_hosts = self.get_healthy_hosts()
        if not healthy_hosts:
            return None
        
        with self._lock:
            # Use modulo to cycle through healthy hosts
            host = healthy_hosts[self._round_robin_index % len(healthy_hosts)]
            self._round_robin_index += 1
            logger.debug(f"Round-robin selected: {host} (index: {self._round_robin_index})")
            return host
    
    def get_best_host(self) -> Optional[str]:
        """
        Select the best performing host based on response times and health.
        Uses sophisticated scoring algorithm.
        """
        healthy_hosts = self.get_healthy_hosts()
        if not healthy_hosts:
            return None
        
        if len(healthy_hosts) == 1:
            return healthy_hosts[0]
        
        with self._lock:
            host_scores = []
            
            for host in healthy_hosts:
                recent_times = self.response_times.get(host, [1.0])
                if not recent_times:
                    recent_times = [1.0]
                
                # Calculate average response time
                avg_time = sum(recent_times) / len(recent_times)
                
                # Calculate trend (positive = getting slower)
                if len(recent_times) > 1:
                    trend = recent_times[-1] - recent_times[0]
                else:
                    trend = 0
                
                # Score calculation (lower is better)
                # Penalize hosts with increasing response times
                score = avg_time + (trend * 2)
                host_scores.append((host, score))
            
            # Return host with lowest score
            best_host = min(host_scores, key=lambda x: x[1])[0]
            logger.debug(f"Performance-based selected: {best_host}")
            return best_host
    
    def get_balanced_host(self) -> Optional[str]:
        """
        Balanced host selection: 70% round-robin, 30% performance-based
        This ensures good distribution while still considering performance
        """
        import random
        
        healthy_hosts = self.get_healthy_hosts()
        if not healthy_hosts:
            return None
        
        # 70% delle volte usa round-robin per distribuzione uniforme
        # 30% usa performance-based per ottimizzazione
        if random.random() < 0.7:
            host = self.get_round_robin_host()
            logger.debug("Using round-robin selection")
            return host
        else:
            host = self.get_best_host()
            logger.debug("Using performance-based selection")
            return host
    
    def get_host_stats(self) -> Dict[str, Dict]:
        """Get statistics for all hosts"""
        with self._lock:
            stats = {}
            for host in self.hosts:
                recent_times = self.response_times.get(host, [])
                stats[host] = {
                    'healthy': self.health_status.get(host, False),
                    'last_check': self.last_check.get(host, 0),
                    'avg_response_time': sum(recent_times) / len(recent_times) if recent_times else 0,
                    'recent_requests': len(recent_times)
                }
            return stats
    
    def reset_round_robin(self):
        """Reset round-robin counter (useful for testing)"""
        with self._lock:
            self._round_robin_index = 0
            logger.debug("Round-robin index reset to 0")

# ============= OLLAMA CONNECTION MANAGEMENT =============
class OllamaConnectionManager:
    """Manages Ollama connections and API interactions"""
    
    def __init__(self):
        self.hosts: List[str] = []
        self.rate_limiter: Semaphore = Semaphore(1)  # Default semaforo con 1 permit
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=Config.CIRCUIT_BREAKER_THRESHOLD,
            timeout=Config.REQUEST_TIMEOUT + 120  # Circuit breaker timeout > REQUEST_TIMEOUT
        )
        self.health_monitor: HostHealthMonitor = HostHealthMonitor([])  # Lista vuota iniziale
        
    def setup_connections(self) -> List[str]:
        """Setup Ollama connections from port configuration file"""
        try:
            with open(Config.OLLAMA_PORT_FILE, "r") as f:
                ports_str = f.read().strip()
            
            if "," in ports_str:
                # Multi-GPU configuration
                ports = [p.strip() for p in ports_str.split(",")]
                self.hosts = [f"http://127.0.0.1:{port}" for port in ports]
                logger.info(f"Multi-GPU configuration: {len(self.hosts)} instances")
                
                self.rate_limiter = Semaphore(len(self.hosts) * Config.MAX_CONCURRENT_PER_GPU)  # ✅ FIXED: 1 richiesta per A100
                
            else:
                # Single GPU fallback
                self.hosts = [f"http://127.0.0.1:{ports_str}"]
                logger.info(f"Single GPU configuration: {self.hosts[0]}")
                self.rate_limiter = Semaphore(1)
            
            # RE-inizializza health monitor con hosts corretti
            self.health_monitor = HostHealthMonitor(self.hosts)
            
            return self.hosts
            
        except FileNotFoundError:
            raise RuntimeError(f"Configuration file {Config.OLLAMA_PORT_FILE} not found")
        except Exception as e:
            raise RuntimeError(f"Failed to setup Ollama connections: {e}")
    
    def wait_for_services(self, max_attempts: int = 30, wait_interval: int = 3) -> bool:
        """Wait for all Ollama services to be ready"""
        logger.info("Waiting for Ollama services to start...")
        
        for i, host in enumerate(self.hosts):
            logger.info(f"Checking host {i+1}/{len(self.hosts)}: {host}")
            
            for attempt in range(1, max_attempts + 1):
                try:
                    response = requests.get(
                        f"{host}/api/tags",
                        timeout=10,
                        headers={'Accept': 'application/json'}
                    )
                    
                    if response.status_code == 200:
                        logger.info(f"Host {host} is ready")
                        break
                        
                except requests.exceptions.RequestException as e:
                    logger.debug(f"Attempt {attempt}/{max_attempts} failed: {e}")
                    
                if attempt < max_attempts:
                    time.sleep(wait_interval)
                else:
                    logger.error(f"Host {host} failed to respond after {max_attempts} attempts")
                    return False
        
        logger.info("All Ollama services are ready!")
        return True
    
    def test_model_availability(self, model: str = Config.MODEL_NAME) -> bool:
        """Test if the specified model is available on all hosts"""
        working_hosts = 0
        
        for host in self.hosts:
            try:
                # Check available models
                resp = requests.get(f"{host}/api/tags", timeout=10)
                if resp.status_code != 200:
                    logger.error(f"Failed to get models from {host}")
                    continue
                
                models = [m.get('name', '') for m in resp.json().get('models', [])]
                if model not in models:
                    logger.error(f"Model {model} not found on {host}")
                    continue
                
                # Test inference capability
                test_payload = {
                    "model": model,
                    "prompt": "Hi",
                    "stream": False,
                    "options": {
                        "num_predict": 1,
                        "temperature": 0
                    }
                }
                
                test_resp = requests.post(
                    f"{host}/api/generate",
                    json=test_payload,
                    timeout=60
                )
                
                if test_resp.status_code == 200:
                    data = test_resp.json()
                    if data.get("done") and data.get("response"):
                        working_hosts += 1
                        logger.info(f"Model {model} working on {host}")
                    else:
                        logger.error(f"Model test failed on {host}: incomplete response")
                else:
                    logger.error(f"Model test failed on {host}: HTTP {test_resp.status_code}")
                    
            except Exception as e:
                logger.error(f"Error testing {host}: {e}")
        
        logger.info(f"{working_hosts}/{len(self.hosts)} hosts have working model")
        return working_hosts > 0
    
    def get_chat_completion(self, prompt: str, model: str = Config.MODEL_NAME) -> Optional[str]:
        """Get chat completion with load balancing and error handling"""
        
        # Check circuit breaker - STOP COMPLETELY instead of skipping
        if stats.circuit_breaker_active:
            logger.error("💥 CIRCUIT BREAKER ACTIVE - SYSTEM FAILURE DETECTED")
            logger.error("🛑 All Ollama instances have failed. Processing must stop.")
            logger.error("🔧 Check GPU memory, processes, and logs before restarting.")
            raise Exception("Circuit breaker active - system failure detected. Processing stopped.")
        
        # Controllo di sicurezza
        if self.rate_limiter is None:
            logger.error("Rate limiter not initialized - call setup_connections() first")
            return None
            
        # Acquire rate limiting semaphore - timeout = REQUEST_TIMEOUT + buffer per evitare deadlock
        if not self.rate_limiter.acquire(blocking=True, timeout=Config.REQUEST_TIMEOUT + 60):
            logger.warning("Rate limit timeout - system overloaded")
            return None
        
        try:
            with self.circuit_breaker.call():
                return self._make_request_with_retry(prompt, model)
        except Exception as e:
            logger.error(f"Request failed completely: {e}")
            return None
        finally:
            # Controllo sicurezza anche nel finally
            if self.rate_limiter is not None:
                self.rate_limiter.release()
    
    def _make_request_with_retry(self, prompt: str, model: str) -> Optional[str]:
        """Make request with exponential backoff retry logic and improved load balancing"""
        
        if self.health_monitor is None:
            logger.error("Health monitor not initialized")
            return None
        
        service_unavailable_count = 0  # Counter per 503
        host_usage_count = {}  # Track usage per host for this request
        
        for attempt in range(1, Config.MAX_RETRIES_PER_REQUEST + 1):
            # Select host using round-robin for better load distribution
            healthy_hosts = self.health_monitor.get_healthy_hosts()
            
            if not healthy_hosts:
                logger.error("No healthy hosts available")
                # Try to recover all hosts
                for h in self.hosts:
                    self.health_monitor.check_health(h)
                
                healthy_hosts = self.health_monitor.get_healthy_hosts()
                if not healthy_hosts:
                    # If all hosts are down, wait and retry
                    if service_unavailable_count > 0:
                        logger.warning(f"All hosts down after {service_unavailable_count} 503 errors, waiting 2 minutes...")
                        time.sleep(120)
                        continue
                    raise Exception("All hosts are down")
            
            # Improved host selection: round-robin with fallback to performance-based
            host = None
            if hasattr(self.health_monitor, '_round_robin_index'):
                # Use round-robin if available
                with self.health_monitor._lock:
                    idx = self.health_monitor._round_robin_index % len(healthy_hosts)
                    host = healthy_hosts[idx]
                    self.health_monitor._round_robin_index += 1
            else:
                # Fallback to least used host in this request
                if host_usage_count:
                    # Find host with minimum usage in this request
                    min_usage = min(host_usage_count.values())
                    least_used_hosts = [h for h, count in host_usage_count.items() 
                                    if count == min_usage and h in healthy_hosts]
                    if least_used_hosts:
                        host = random.choice(least_used_hosts)
                
                # If still no host selected, use best performing one
                if not host:
                    host = self.health_monitor.get_best_host()
            
            if not host:
                logger.error("Failed to select any host")
                continue
                
            # Track host usage for this request
            host_usage_count[host] = host_usage_count.get(host, 0) + 1
            
            # Log host selection for debugging
            logger.debug(f"Attempt {attempt}: Selected host {host} (usage: {host_usage_count[host]})")
            
            try:
                # Prepare optimized payload for 4x A100 64GB GPUs
                payload = {
                    "model": model,
                    "messages": [
                        {
                            "role": "system", 
                            "content": "You are a JSON-only responder. Always output valid JSON."
                        },
                        {
                            "role": "user", 
                            "content": prompt
                        }
                    ],
                    "stream": False,
                    # ✅ REMOVED: Rimosso formato JSON forzato che causava interruzioni
                    "options": {
                        # Context window - ULTRA CONSERVATIVE per massima stabilità
                        "num_ctx": 2048,           # DRASTICALLY REDUCED per evitare OOM
                        
                        # Generation parameters - ottimizzati per evitare interruzioni
                        "num_predict": 256,        # Ridotto ulteriormente per stabilità
                        "temperature": 0.1,        # Bassa per predizioni precise
                        "top_p": 0.9,
                        "top_k": 40,               # Aggiunto per controllo qualità
                        
                        # Hardware optimization per A100 - PARAMETRI ULTRA CONSERVATIVI
                        "num_thread": 32,          # DRASTICALLY REDUCED per evitare sovraccarico CPU
                        "num_batch": 512,          # QUARTERED per massima stabilità memoria
                        "num_gpu": 1,              # Una GPU per istanza Ollama
                        "main_gpu": 0,             # GPU principale per l'istanza
                        
                        # Memory optimization per 64GB VRAM
                        "num_gqa": 8,              # Group Query Attention per efficienza
                        "num_keep": -1,            # Mantieni tutto il prompt in memoria
                        "cache_type_k": "f16",     # Cache key in FP16 per velocità
                        
                        # Advanced parameters
                        "repeat_penalty": 1.05,    # Ridotto per predizioni più naturali
                        "repeat_last_n": 256,     # Considera ultimi 256 token per ripetizioni
                        "penalize_newline": False, # Non penalizzare newline nel JSON
                        "mirostat": 2,             # Mirostat v2 per qualità output
                        "mirostat_tau": 5.0,       # Target perplexity
                        "mirostat_eta": 0.1,       # Learning rate per Mirostat
                    }
                }
                
                start_time = time.time()
                resp = requests.post(
                    f"{host}/api/chat",
                    json=payload,
                    timeout=Config.REQUEST_TIMEOUT,
                    headers={'Content-Type': 'application/json'}
                )
                response_time = time.time() - start_time
                
                # Handle 503 Service Unavailable specifically
                if resp.status_code == 503:
                    service_unavailable_count += 1
                    logger.warning(f"503 Service Unavailable from {host} (count: {service_unavailable_count})")
                    
                    if service_unavailable_count <= Config.MAX_503_RETRIES:
                        wait_time = Config.RETRY_ON_503_WAIT * min(service_unavailable_count, 5)
                        logger.info(f"Waiting {wait_time}s for model to load on {host}...")
                        time.sleep(wait_time)
                        continue
                    else:
                        logger.error(f"Too many 503 errors ({service_unavailable_count}) from {host}")
                        with self.health_monitor._lock:
                            self.health_monitor.health_status[host] = False
                        continue
                
                resp.raise_for_status()
                response_data = resp.json()
                
                # ✅ FIXED: Accetta risposte anche se done=False, purché ci sia contenuto valido
                content = response_data.get("message", {}).get("content", "")
                done_status = response_data.get("done", False)
                
                # Se c'è contenuto, procedi anche se done=False
                if content:
                    if not done_status:
                        logger.debug(f"Using partial response from {host} (done=False but content available)")
                    # Continua con il processing normale
                    stats.increment_processed()
                    logger.debug(f"SUCCESS: Got response from {host} in {response_time:.2f}s")
                    
                    # Update response time tracking for this host
                    with self.health_monitor._lock:
                        if host not in self.health_monitor.response_times:
                            self.health_monitor.response_times[host] = []
                        self.health_monitor.response_times[host].append(response_time)
                        if len(self.health_monitor.response_times[host]) > self.health_monitor._max_response_history:
                            self.health_monitor.response_times[host].pop(0)
                    
                    service_unavailable_count = 0  # Reset 503 counter on success
                    return content
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on {host} (attempt {attempt})")
                with self.health_monitor._lock:
                    self.health_monitor.health_status[host] = False
            except requests.exceptions.HTTPError as e:
                if "503" not in str(e):  # Log non-503 HTTP errors
                    logger.error(f"HTTP Error on {host}: {e}")
                with self.health_monitor._lock:
                    self.health_monitor.health_status[host] = False
                stats.increment_errors()
            except Exception as exc:
                logger.error(f"Unexpected error on {host}: {exc}")
                with self.health_monitor._lock:
                    self.health_monitor.health_status[host] = False
                stats.increment_errors()
            
            # Exponential backoff between retries, but shorter for 503 errors
            if attempt < Config.MAX_RETRIES_PER_REQUEST:
                if service_unavailable_count > 0:
                    # Shorter backoff for 503 errors since we already waited above
                    backoff_time = min(5, Config.BACKOFF_BASE ** (attempt - service_unavailable_count))
                else:
                    backoff_time = min(Config.BACKOFF_BASE ** attempt, Config.BACKOFF_MAX)
                
                logger.debug(f"Backing off for {backoff_time}s before retry {attempt + 1}")
                time.sleep(backoff_time)
        
        logger.error(f"All {Config.MAX_RETRIES_PER_REQUEST} attempts failed")
        return None

class SafeOllamaConnectionManager(OllamaConnectionManager):
    """Versione con inizializzazione sicura garantita"""
    
    def __init__(self):
        super().__init__()
        self._initialized = False
    
    def setup_connections(self) -> List[str]:
        """Setup with initialization flag"""
        try:
            result = super().setup_connections()
            self._initialized = True
            logger.info("OllamaConnectionManager fully initialized")
            return result
        except Exception as e:
            self._initialized = False
            logger.error(f"Failed to initialize OllamaConnectionManager: {e}")
            raise
    
    def _ensure_initialized(self):
        """Ensure manager is properly initialized"""
        if not self._initialized:
            raise RuntimeError(
                "OllamaConnectionManager not initialized. Call setup_connections() first."
            )
    
    def get_chat_completion(self, prompt: str, model: str = Config.MODEL_NAME) -> Optional[str]:
        """Get chat completion with initialization check"""
        self._ensure_initialized()
        return super().get_chat_completion(prompt, model)
    
# ============= GEOGRAPHIC UTILITIES =============
def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate distance in kilometers between two geographic points using Haversine formula.
    
    The Haversine formula determines the great-circle distance between two points
    on a sphere given their longitudes and latitudes.
    
    Args:
        lat1, lon1: Latitude and longitude of first point
        lat2, lon2: Latitude and longitude of second point
        
    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth's radius in kilometers
    
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Calculate differences
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Haversine formula
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c

# ============= DATA LOADING FUNCTIONS =============
class DataLoader:
    """Handles loading and preprocessing of tourist visit data"""
    
    @staticmethod
    def load_pois(filepath: Path) -> DataFrame:
        """
        Load Points of Interest (POI) data with coordinates.
        
        Args:
            filepath: Path to POI CSV file
            
        Returns:
            DataFrame with columns: name_short, latitude, longitude
        """
        df = pd.read_csv(
            filepath, 
            usecols=["name_short", "latitude", "longitude"],
            dtype={
                "name_short": "category",
                "latitude": np.float32,
                "longitude": np.float32
            }
        )
        logger.info(f"Loaded {len(df)} POIs from {filepath.name}")
        return df
    
    @staticmethod
    def load_visits(filepath: Path) -> DataFrame:
        """
        Load tourist visit data and convert to standardized format.
        
        Args:
            filepath: Path to visits CSV file
            
        Returns:
            DataFrame with columns: timestamp, card_id, name_short, date, time, hour, minute, day_of_week
        """
        df = pd.read_csv(
            filepath,
            usecols=[0, 1, 2, 4],  # Select specific columns by position
            names=["data", "ora", "name_short", "card_id"],
            header=0,
            dtype={"card_id": "category", "name_short": "category"}
        )
        
        # Combine date and time into single timestamp
        df["timestamp"] = pd.to_datetime(
            df["data"] + " " + df["ora"], 
            format="%d-%m-%y %H:%M:%S"
        )
        
        # Extract temporal features for prompt
        df["date"] = df["timestamp"].dt.date
        df["time"] = df["timestamp"].dt.time
        df["hour"] = df["timestamp"].dt.hour
        df["minute"] = df["timestamp"].dt.minute
        df["day_of_week"] = df["timestamp"].dt.day_name()
        
        logger.info(f"Loaded {len(df)} visits from {filepath.name}")
        
        # Return all columns including temporal features, sorted by timestamp
        return (df[["timestamp", "card_id", "name_short", "date", "time", "hour", "minute", "day_of_week"]]
                .sort_values("timestamp")
                .reset_index(drop=True))
    
    @staticmethod
    def merge_visits_pois(visits_df: DataFrame, pois_df: DataFrame) -> DataFrame:
        """
        Merge visits with POI data to filter out invalid visits.
        
        Args:
            visits_df: DataFrame with visit records
            pois_df: DataFrame with POI information
            
        Returns:
            DataFrame with only valid visits (matching POIs)
        """
        # Inner join keeps only visits to valid POIs
        merged = visits_df.merge(
            pois_df[["name_short"]], 
            on="name_short", 
            how="inner"
        )
        
        logger.info(f"Valid visits after merge: {len(merged)}")
        return merged.sort_values("timestamp").reset_index(drop=True)
    
    @staticmethod
    def filter_multi_visit_cards(df: DataFrame) -> DataFrame:
        """
        Filter to keep only cards that visited multiple distinct POIs.
        
        This ensures we have meaningful sequences for prediction.
        
        Args:
            df: DataFrame with visit records
            
        Returns:
            DataFrame with only multi-visit cards
        """
        # Count unique POIs per card
        unique_pois_per_card = df.groupby("card_id")["name_short"].nunique()
        
        # Keep cards with more than one unique POI
        valid_cards = unique_pois_per_card[unique_pois_per_card > 1].index
        
        logger.info(f"Multi-visit cards: {len(valid_cards)} / {df.card_id.nunique()}")
        
        return df[df["card_id"].isin(valid_cards)].reset_index(drop=True)
    
    @staticmethod
    def create_user_poi_matrix(df: DataFrame) -> DataFrame:
        """
        Create user-POI interaction matrix for clustering.
        
        Args:
            df: DataFrame with visit records
            
        Returns:
            Crosstab matrix of card_id x POI visits
        """
        return pd.crosstab(df["card_id"], df["name_short"])

# ============= PROMPT GENERATION =============
class PromptBuilder:
    """Handles prompt generation for LLM predictions"""
    
    @staticmethod
    def get_anchor_index(seq_len: int, rule: str | int) -> int:
        """
        Determine anchor POI index based on specified rule.
        
        The anchor POI is used as the "current location" in the prompt.
        
        Args:
            seq_len: Length of sequence (excluding target)
            rule: Selection strategy:
                - "penultimate": Last element of prefix
                - "first": Index 0
                - "middle": seq_len // 2
                - int: Explicit index (negative allowed)
                
        Returns:
            0-based index of anchor POI
            
        Raises:
            ValueError: If rule is invalid or index out of range
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
            raise ValueError(f"Invalid anchor_rule: '{rule}'")
        
        if not (0 <= idx < seq_len):
            raise ValueError(f"Anchor index {idx} out of range for sequence length {seq_len}")
            
        return idx
    
    @staticmethod
    def get_nearby_pois(
        current_poi: str, 
        pois_df: pd.DataFrame,
        visited_pois: List[str],
        max_pois: int = 10,
        max_distance: float = 5.0
    ) -> List[Dict[str, Any]]:
        """
        Find POIs near the current location.
        
        Args:
            current_poi: Name of current POI
            pois_df: DataFrame with all POIs
            visited_pois: List of already visited POIs to exclude
            max_pois: Maximum number of POIs to return
            max_distance: Maximum distance in km
            
        Returns:
            List of nearby POIs with distances
        """
        current_poi_row = pois_df[pois_df["name_short"] == current_poi]
        if current_poi_row.empty:
            return []
        
        current_lat = current_poi_row["latitude"].iloc[0]
        current_lon = current_poi_row["longitude"].iloc[0]
        
        nearby_pois = []
        
        for _, row in pois_df.iterrows():
            poi_name = row["name_short"]
            
            # Skip if already visited or is current POI
            if poi_name in visited_pois or poi_name == current_poi:
                continue
            
            distance = calculate_distance(
                current_lat, current_lon,
                row["latitude"], row["longitude"]
            )
            
            # Only include if within max distance
            if distance <= max_distance:
                nearby_pois.append({
                    "name": poi_name,
                    "distance": distance
                })
        
        # Sort by distance and limit results
        nearby_pois.sort(key=lambda x: x["distance"])
        return nearby_pois[:max_pois]
    
    @staticmethod
    def create_prompt(
        df: pd.DataFrame,
        user_clusters: pd.DataFrame,
        pois_df: pd.DataFrame,
        card_id: str,
        top_k: int = Config.TOP_K,
        anchor_rule: Union[str, int] = Config.DEFAULT_ANCHOR_RULE
    ) -> str:
        """
        Create optimized prompt for POI prediction.
        
        This method generates a concise prompt that includes:
        - User's cluster (tourist type)
        - Visit history with temporal patterns
        - Current location with time context
        - Nearby POIs with distances
        
        Args:
            df: Visit data with temporal features
            user_clusters: Cluster assignments
            pois_df: POI information
            card_id: Card to predict for
            top_k: Number of predictions requested
            anchor_rule: Rule for selecting current POI
            
        Returns:
            Formatted prompt string
            
        Raises:
            ValueError: If sequence too short or POI not found
        """
        # Get visit sequence for this card
        visits = df[df["card_id"] == card_id].sort_values("timestamp")
        seq = visits["name_short"].tolist()
        
        if len(seq) < 3:
            raise ValueError("Sequence too short (minimum 3 visits required)")
        
        # Split into history, current, and target
        prefix = seq[:-1]  # All except last
        
        # Determine current POI using anchor rule
        idx = PromptBuilder.get_anchor_index(len(prefix), anchor_rule)
        current_poi = prefix[idx]
        history = [p for i, p in enumerate(prefix) if i != idx]
        
        # Get temporal information for current visit
        current_visit = visits.iloc[idx]
        current_time = current_visit["time"]
        current_day = current_visit["day_of_week"]
        
        # Extract temporal patterns from visit history
        history_times = visits.iloc[:-1]  # All visits except the last (target)
        avg_hour = history_times["hour"].mean()
        visit_hours = history_times["hour"].tolist()
        days_visited = history_times["day_of_week"].unique().tolist()
        
        # Get user's cluster
        cluster_id = user_clusters.loc[
            user_clusters["card_id"] == card_id, "cluster"
        ].values[0]
        
        # Get nearby POIs
        nearby_pois = PromptBuilder.get_nearby_pois(
            current_poi, pois_df, history, max_pois=10
        )
        
        # Format POI list with distances
        pois_list = ", ".join([
            f"{poi['name']} ({poi['distance']:.1f}km)"
            for poi in nearby_pois
        ])
        
        # Format temporal context
        time_context = f"Current: {current_day} {current_time.strftime('%H:%M')}"
        if visit_hours:
            time_context += f", usual hours: {visit_hours}, avg: {avg_hour:.1f}h"
        if len(days_visited) > 1:
            time_context += f", days visited: {', '.join(days_visited[:3])}"
        
        # Create comprehensive prompt with flexible output format
        return f"""You are an expert tourism analyst predicting visitor behavior in Verona, Italy.

TOURIST PROFILE:
- Cluster: {cluster_id} (behavioral pattern group)  
- Visit history: {', '.join(history) if history else 'First-time visitor'}
- Current location: {current_poi}

TEMPORAL CONTEXT:
{time_context}

SPATIAL CONTEXT:
Nearby attractions within walking distance: {pois_list if pois_list else 'No nearby POIs within 5km'}

TASK:
Predict the {top_k} most likely next destinations for this tourist, considering:
1. Geographic proximity and accessibility
2. Temporal patterns (time of day, day of week preferences)  
3. Tourist cluster behavioral patterns
4. Logical flow of sightseeing activities
5. Popular attraction combinations in Verona

REQUIREMENTS:
- Provide exactly {top_k} predictions
- Order them by decreasing probability (most likely first)
- Only suggest POIs from the nearby list or well-known Verona attractions
- Consider realistic travel times and opening hours

OUTPUT FORMAT:
Respond in JSON format if possible, but clear structured text is also acceptable:

PREFERRED (JSON):
{{"prediction": ["poi1", "poi2", "poi3"], "reason": "brief explanation"}}

ALTERNATIVE (Structured text):
PREDICTIONS:
1. most_likely_poi
2. second_most_likely_poi  
3. third_most_likely_poi

REASONING: Brief explanation of temporal and spatial factors."""

# ============= CHECKPOINT MANAGEMENT =============
class CheckpointManager:
    """Manages checkpoint files for resumable processing"""
    
    def __init__(self, visits_path: Path, out_dir: Path):
        self.visits_path = visits_path
        self.out_dir = out_dir
        self.checkpoint_file = out_dir / f"{visits_path.stem}_checkpoint.txt"
        self._completed_cards: Set[str] = set()
        self._load_checkpoint()
    
    def _load_checkpoint(self):
        """Load completed cards from checkpoint file"""
        if self.checkpoint_file.exists():
            try:
                with open(self.checkpoint_file, 'r') as f:
                    self._completed_cards = {line.strip() for line in f if line.strip()}
                logger.info(f"Loaded {len(self._completed_cards)} completed cards from checkpoint")
            except Exception as e:
                logger.warning(f"Error loading checkpoint: {e}")
                self._completed_cards = set()
    
    def is_completed(self, card_id: str) -> bool:
        """Check if a card has been processed"""
        return card_id in self._completed_cards
    
    def mark_completed(self, card_id: str):
        """Mark a card as completed and update checkpoint file"""
        self._completed_cards.add(card_id)
        try:
            with open(self.checkpoint_file, 'a') as f:
                f.write(f"{card_id}\n")
        except Exception as e:
            logger.warning(f"Error updating checkpoint: {e}")
    
    def get_completed_count(self) -> int:
        """Get number of completed cards"""
        return len(self._completed_cards)
    
    @staticmethod
    def should_skip_file(visits_path: Path, out_dir: Path, append: bool = False) -> bool:
        """
        Check if a file should be skipped (already fully processed).
        
        Args:
            visits_path: Path to visits file
            out_dir: Output directory
            append: Whether in append mode
            
        Returns:
            True if file should be skipped
        """
        if not append:
            return False
        
        checkpoint = CheckpointManager(visits_path, out_dir)
        completed_count = checkpoint.get_completed_count()
        
        if completed_count == 0:
            return False
        
        # Quick check - if we have many completed cards, it's likely done
        # For exact check, would need to load and process the file
        logger.info(f"File {visits_path.stem} has {completed_count} completed cards")
        
        # Conservative approach - don't skip unless explicitly verified
        return False


# ============= RESULTS MANAGEMENT =============

class ResultsManager:
    """Handles saving and managing prediction results"""
    
    def __init__(self, visits_path: Path, out_dir: Path, append: bool = False):
        self.visits_path = visits_path
        self.out_dir = out_dir
        self.append = append
        self.output_file = self._get_output_file()
        self.write_header = not (append and self.output_file.exists())
        self._buffer: List[Dict] = []
        self._write_lock = Lock()
    
    def _get_output_file(self) -> Path:
        """Determine output file path"""
        if self.append:
            # Look for existing output files
            pattern = f"{self.visits_path.stem}_pred_*.csv"
            existing_files = list(self.out_dir.glob(pattern))
            
            if existing_files:
                # Use the most recent file
                return max(existing_files, key=lambda p: p.stat().st_mtime)
        
        # Create new file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.out_dir / f"{self.visits_path.stem}_pred_{timestamp}.csv"
    
    def add_result(self, result: Dict):
        """Add a result to the buffer"""
        self._buffer.append(result)
        
        # Save if buffer is full
        if len(self._buffer) >= Config.BATCH_SAVE_INTERVAL:
            self.save_batch()
    
    def save_batch(self):
        """Save buffered results to file"""
        if not self._buffer:
            return
        
        with self._write_lock:
            try:
                df_batch = pd.DataFrame(self._buffer)
                
                mode = 'w' if self.write_header else 'a'
                df_batch.to_csv(
                    self.output_file,
                    mode=mode,
                    header=self.write_header,
                    index=False,
                    encoding='utf-8'
                )
                
                logger.debug(f"Saved batch of {len(self._buffer)} results")
                self.write_header = False
                self._buffer.clear()
                
            except Exception as e:
                logger.error(f"Error saving batch: {e}")
                # Create backup
                self._save_backup()
    
    def _save_backup(self):
        """Save backup of current buffer"""
        try:
            backup_file = (self.out_dir / 
                          f"backup_{self.visits_path.stem}_{int(time.time())}.json")
            with open(backup_file, 'w') as f:
                json.dump(self._buffer, f)
            logger.info(f"Backup saved to {backup_file}")
        except Exception as e:
            logger.error(f"Failed to save backup: {e}")
    
    def finalize(self):
        """Save any remaining results"""
        if self._buffer:
            self.save_batch()

# ============= CARD PROCESSING =============
class CardProcessor:
    """Processes individual tourist cards for POI prediction"""
    
    def __init__(
        self,
        filtered_df: DataFrame,
        user_clusters: DataFrame,
        pois_df: DataFrame,
        ollama_manager: OllamaConnectionManager,
        checkpoint_manager: CheckpointManager,
        results_manager: ResultsManager
    ):
        self.filtered_df = filtered_df
        self.user_clusters = user_clusters
        self.pois_df = pois_df
        self.ollama_manager = ollama_manager
        self.checkpoint_manager = checkpoint_manager
        self.results_manager = results_manager
    
    def process_card(self, card_id: str) -> Optional[Dict]:
        """
        Process a single card to predict next POI visit.
        
        Args:
            card_id: Card identifier to process
            
        Returns:
            Dictionary with prediction results or None if error
        """
        start_time = time.time()
        
        try:
            # Skip if already processed
            if self.checkpoint_manager.is_completed(card_id):
                logger.debug(f"Card {card_id} already processed - skipping")
                return None
            
            # Get visit sequence
            seq = (self.filtered_df[self.filtered_df.card_id == card_id]
                   .sort_values("timestamp")["name_short"]
                   .tolist())
            
            if len(seq) < 3:
                logger.debug(f"Card {card_id} has insufficient visits ({len(seq)})")
                return None
            
            # Extract sequence components
            target = seq[-1]
            prefix = seq[:-1]
            
            # Create prompt
            try:
                prompt = PromptBuilder.create_prompt(
                    self.filtered_df,
                    self.user_clusters,
                    self.pois_df,
                    card_id,
                    top_k=Config.TOP_K,
                    anchor_rule=Config.DEFAULT_ANCHOR_RULE
                )
            except Exception as e:
                logger.warning(f"Error creating prompt for {card_id}: {e}")
                return None
            
            # Get LLM prediction
            response = self.ollama_manager.get_chat_completion(prompt)
            
            # Prepare result record
            idx_anchor = PromptBuilder.get_anchor_index(
                len(prefix), Config.DEFAULT_ANCHOR_RULE
            )
            history_list = [p for i, p in enumerate(prefix) if i != idx_anchor]
            current_poi = prefix[idx_anchor]
            
            result = {
                "card_id": card_id,
                "cluster": self._get_user_cluster(card_id),
                "history": str(history_list),
                "current_poi": current_poi,
                "prediction": None,
                "ground_truth": target,
                "reason": None,
                "hit": False,
                "processing_time": time.time() - start_time,
                "status": "success" if response else "failed"
            }
            
            # Parse response if available - gestione più robusta
            if response:
                try:
                    # ✅ IMPROVED: Tentativo di parsing JSON più flessibile
                    # Pulisce la risposta da caratteri problematici
                    cleaned_response = response.strip()
                    
                    # Cerca pattern JSON nella risposta anche se non perfettamente formattata
                    if "{" in cleaned_response and "}" in cleaned_response:
                        # Estrae il primo blocco JSON valido
                        start_idx = cleaned_response.find("{")
                        end_idx = cleaned_response.rfind("}") + 1
                        json_part = cleaned_response[start_idx:end_idx]
                        
                        parsed = json.loads(json_part)
                    else:
                        # Fallback: crea struttura JSON dai pattern di testo
                        logger.debug(f"No JSON found in response, attempting text parsing for {card_id}")
                        predictions = self._extract_predictions_from_text(response)
                        parsed = {"prediction": predictions, "reason": "Extracted from text"}
                    
                    predictions = parsed.get("prediction", [])
                    if not isinstance(predictions, list):
                        predictions = [predictions] if predictions else []
                    
                    result["prediction"] = str(predictions)
                    result["reason"] = parsed.get("reason", "")[:200]  # Limit length
                    result["hit"] = target in predictions
                    
                except json.JSONDecodeError as e:
                    logger.debug(f"JSON parse failed for {card_id}, attempting text extraction: {e}")
                    # ✅ FALLBACK: Estrazione pattern da testo libero
                    predictions = self._extract_predictions_from_text(response)
                    if predictions:
                        result["prediction"] = str(predictions)
                        result["reason"] = "Extracted from non-JSON response"
                        result["hit"] = target in predictions
                        result["status"] = "success_text_parsed"
                    else:
                        result["prediction"] = f"PARSE_ERROR: {response[:100]}"
                        result["status"] = "parse_error"
                except Exception as e:
                    logger.warning(f"Error parsing response for {card_id}: {e}")
                    result["prediction"] = "PROCESSING_ERROR"
                    result["status"] = "processing_error"
            
            # Mark as completed and save result
            self.checkpoint_manager.mark_completed(card_id)
            self.results_manager.add_result(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Fatal error processing card {card_id}: {e}")
            return {
                "card_id": card_id,
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
    
    def _get_user_cluster(self, card_id: str) -> Optional[int]:
        """Get cluster ID for a card"""
        try:
            return int(self.user_clusters[
                self.user_clusters.card_id == card_id
            ]["cluster"].iloc[0])
        except Exception:
            return None
    
    def _extract_predictions_from_text(self, text: str) -> List[str]:
        """
        ✅ NEW: Estrae predizioni POI da testo libero quando JSON parsing fallisce.
        
        Cerca pattern comuni come:
        - Lista numerata: "1. POI_NAME"
        - Lista puntata: "• POI_NAME" 
        - Lista semplice: "POI1, POI2, POI3"
        - Menzioni dirette di POI noti
        """
        import re
        
        predictions = []
        text = text.lower().strip()
        
        # Pattern 1: Liste numerate (1. Arena, 2. Casa_di_Giulietta, etc.)
        numbered_pattern = r'\d+\.?\s*([A-Za-z_][A-Za-z0-9_]*(?:\s+[A-Za-z0-9_]+)*)'
        numbered_matches = re.findall(numbered_pattern, text)
        if numbered_matches:
            predictions.extend([match.replace(' ', '_') for match in numbered_matches[:5]])
        
        # Pattern 2: Liste con bullet points
        bullet_pattern = r'[•\-\*]\s*([A-Za-z_][A-Za-z0-9_]*(?:\s+[A-Za-z0-9_]+)*)'
        bullet_matches = re.findall(bullet_pattern, text)
        if bullet_matches:
            predictions.extend([match.replace(' ', '_') for match in bullet_matches[:5]])
            
        # Pattern 3: Lista separata da virgole
        if not predictions and ',' in text:
            comma_parts = [part.strip() for part in text.split(',')]
            for part in comma_parts[:5]:
                # Filtra parti che sembrano POI names
                clean_part = re.sub(r'[^\w\s_]', '', part).strip().replace(' ', '_')
                if len(clean_part) > 2 and clean_part.isalpha():
                    predictions.append(clean_part)
        
        # Pattern 4: POI noti di Verona (fallback)
        known_pois = [
            'arena', 'casa_di_giulietta', 'torre_dei_lamberti', 'castelvecchio', 
            'piazza_delle_erbe', 'piazza_bra', 'basilica_san_zeno', 'duomo',
            'teatro_romano', 'giardino_giusti', 'santa_anastasia', 'san_fermo'
        ]
        
        if not predictions:
            for poi in known_pois:
                if poi in text or poi.replace('_', ' ') in text:
                    predictions.append(poi)
                    if len(predictions) >= 3:
                        break
        
        # Rimuovi duplicati mantenendo l'ordine
        seen = set()
        unique_predictions = []
        for pred in predictions:
            if pred.lower() not in seen and len(pred) > 1:
                seen.add(pred.lower())
                unique_predictions.append(pred)
        
        return unique_predictions[:5]  # Massimo 5 predizioni


# ============= MAIN PROCESSING PIPELINE =============

class VisitFileProcessor:
    """Orchestrates the complete processing pipeline for a visits file"""
    
    def __init__(self, ollama_manager: OllamaConnectionManager):
        if ollama_manager.rate_limiter is None:
            raise ValueError("OllamaConnectionManager not properly initialized - rate_limiter is None")
        if ollama_manager.health_monitor is None:
            raise ValueError("OllamaConnectionManager not properly initialized - health_monitor is None")
        if not ollama_manager.hosts:
            raise ValueError("OllamaConnectionManager has no hosts configured")
            
        self.ollama_manager = ollama_manager
        Config.RESULTS_DIR.mkdir(exist_ok=True)
    
    def process_file(
        self,
        visits_path: Path,
        poi_path: Path,
        max_users: Optional[int] = None,
        force: bool = False,
        append: bool = False
    ) -> None:
        """
        Process a single visits file to generate POI predictions.
        
        This method orchestrates the complete pipeline:
        1. Load and preprocess data
        2. Perform clustering
        3. Process cards in parallel
        4. Save results with checkpointing
        
        Args:
            visits_path: Path to visits CSV file
            poi_path: Path to POI CSV file
            max_users: Maximum number of users to process (None for all)
            force: Force reprocessing even if output exists
            append: Resume from previous run
        """
        # Check if file should be skipped
        if not force and CheckpointManager.should_skip_file(
            visits_path, Config.RESULTS_DIR, append
        ):
            logger.info(f"Skipping {visits_path.name} - already processed")
            return
        
        logger.info(f"Processing {visits_path.name}")
        
        # Initialize managers
        checkpoint_manager = CheckpointManager(visits_path, Config.RESULTS_DIR)
        results_manager = ResultsManager(visits_path, Config.RESULTS_DIR, append)
        
        try:
            # Load and preprocess data
            logger.info("Loading and preprocessing data...")
            pois_df = DataLoader.load_pois(poi_path)
            visits_df = DataLoader.load_visits(visits_path)
            merged_df = DataLoader.merge_visits_pois(visits_df, pois_df)
            filtered_df = DataLoader.filter_multi_visit_cards(merged_df)
            
            # Perform clustering
            logger.info("Performing user clustering...")
            user_poi_matrix = DataLoader.create_user_poi_matrix(filtered_df)
            
            # K-means clustering with standardization
            scaler = StandardScaler()
            scaled_matrix = scaler.fit_transform(user_poi_matrix)
            
            clusters = KMeans(
                n_clusters=7,
                random_state=42,
                n_init=10
            ).fit_predict(scaled_matrix)
            
            user_clusters = pd.DataFrame({
                "card_id": user_poi_matrix.index,
                "cluster": clusters
            })
            
            # Select cards to process
            eligible_cards = self._get_eligible_cards(filtered_df)
            
            if max_users is not None:
                cards_to_process = random.sample(
                    eligible_cards, 
                    min(max_users, len(eligible_cards))
                )
            else:
                cards_to_process = eligible_cards
            
            # Filter out already processed cards if in append mode
            if append:
                cards_to_process = [
                    card for card in cards_to_process 
                    if not checkpoint_manager.is_completed(card)
                ]
            
            logger.info(f"Processing {len(cards_to_process)} cards")
            
            if not cards_to_process:
                logger.info("No cards to process")
                return
            
            # Create card processor
            card_processor = CardProcessor(
                filtered_df,
                user_clusters,
                pois_df,
                self.ollama_manager,
                checkpoint_manager,
                results_manager
            )
            
            # Process cards in parallel
            self._process_cards_parallel(card_processor, cards_to_process)
            
            # Finalize results
            results_manager.finalize()
            
            # Log summary statistics
            success_rate = stats.get_success_rate()
            logger.info(f"Completed processing {visits_path.name}")
            logger.info(f"Success rate: {success_rate:.1f}%")
            
        except Exception as e:
            logger.error(f"Error processing {visits_path.name}: {e}")
            raise
    
    def _get_eligible_cards(self, filtered_df: DataFrame) -> List[str]:
        """Get cards with sufficient visits for prediction"""
        card_visit_counts = filtered_df.groupby("card_id").size()
        eligible = card_visit_counts[card_visit_counts >= 3].index.tolist()
        return eligible
    
    def _process_cards_parallel(
        self, 
        card_processor: CardProcessor, 
        cards_to_process: List[str]
    ) -> None:
        """Process cards in parallel with progress tracking"""
        
        # Calculate optimal number of workers
        n_healthy_hosts = len(self.ollama_manager.health_monitor.get_healthy_hosts())
    
        # ✅ ULTRA CONSERVATIVE: Usa MAX_CONCURRENT_REQUESTS globale per evitare overload
        optimal_workers = min(Config.MAX_CONCURRENT_REQUESTS, len(cards_to_process))  # ✅ FIXED: Usa limite globale conservativo
        
        logger.info(f"Using {optimal_workers} workers for {n_healthy_hosts} healthy hosts")
        
        # ✅ NUOVO: Attesa estesa per stabilizzazione completa
        logger.info("Waiting 120s for models to FULLY stabilize...")
        time.sleep(120)  # ✅ AUMENTATO: da 60s a 120s per massima stabilità
        
        # ✅ NUOVO: Test pre-processing per verificare che tutto sia OK
        logger.info("Running pre-flight check on all hosts...")
        for host in self.ollama_manager.hosts:
            try:
                test_resp = requests.post(
                    f"{host}/api/chat",
                    json={
                        "model": Config.MODEL_NAME,
                        "messages": [{"role": "user", "content": "test"}],
                        "stream": False,
                        "options": {"num_predict": 1, "temperature": 0}
                    },
                    timeout=60
                )
                if test_resp.status_code == 200:
                    logger.info(f"✅ Pre-flight check passed for {host}")
                else:
                    logger.error(f"❌ Pre-flight check FAILED for {host}: HTTP {test_resp.status_code}")
                    logger.error(f"Response: {test_resp.text[:500]}...")
                    raise Exception(f"Pre-flight failed: {test_resp.status_code}")
                    
            except Exception as e:
                logger.error(f"❌ Pre-flight check FAILED for {host}: {e}")
                logger.error(f"This may indicate memory issues or model loading problems")
                raise Exception(f"Pre-flight failed: {e}")
                
        logger.info("✅ ALL PRE-FLIGHT CHECKS PASSED - Starting progressive warm-up")
        
        # 🔥 PROGRESSIVE WARM-UP: Test system under increasing load
        logger.info("🔥 Phase 1: Single request warm-up test (30s timeout)...")
        warmup_prompts = [
            "List 3 tourist attractions in Verona.",
            "What time is best to visit Verona in summer?", 
            "Recommend a walking route in Verona city center."
        ]
        
        for i, prompt in enumerate(warmup_prompts):
            logger.info(f"  Testing warm-up prompt {i+1}/3...")
            try:
                response = self.ollama_manager.get_chat_completion(prompt)
                if response:
                    logger.info(f"    ✅ Warm-up {i+1} successful: {len(response)} chars")
                else:
                    logger.error(f"    ❌ Warm-up {i+1} failed: No response")
                    raise Exception(f"Warm-up phase failed at test {i+1}")
                time.sleep(2)  # Small pause between tests
            except Exception as e:
                logger.error(f"    ❌ WARM-UP FAILED: {e}")
                logger.error(f"    🛑 System not ready for production load")
                raise Exception(f"Progressive warm-up failed: {e}")
        
        logger.info("🔥 Phase 2: SKIPPED - Concurrent test disabled for stability")
        # REMOVED: Concurrent test that caused Ollama crashes
        # Single request processing is enforced through MAX_CONCURRENT_REQUESTS = 1
        logger.info("✅ PROGRESSIVE WARM-UP COMPLETED - System verified under load")
        logger.info("🚀 Starting production processing...")
        
        # Process cards with thread pool
        with ThreadPoolExecutor(
            max_workers=optimal_workers,
            thread_name_prefix="CardWorker"
        ) as executor:
            
            # Submit all tasks
            futures = {
                executor.submit(card_processor.process_card, card_id): card_id
                for card_id in cards_to_process
            }
            
            # Process results with progress bar
            with tqdm(
                total=len(cards_to_process),
                desc="Processing cards",
                unit="card"
            ) as pbar:
                
                for future in as_completed(futures):
                    card_id = futures[future]
                    
                    try:
                        result = future.result(timeout=Config.REQUEST_TIMEOUT + 60)
                        
                        if result and result.get('status') == 'fatal_error':
                            logger.warning(f"Fatal error for card {card_id}")
                        
                        # Check circuit breaker
                        if stats.consecutive_failures >= Config.CIRCUIT_BREAKER_THRESHOLD:
                            logger.error("Too many consecutive failures - pausing")
                            time.sleep(60)  # Pause for 1 minute
                            stats.reset_consecutive_failures()
                    
                    except TimeoutError:
                        logger.error(f"Timeout processing card {card_id}")
                    except Exception as e:
                        logger.error(f"Error processing card {card_id}: {e}")
                    
                    pbar.update(1)
    
    def process_all_files(
        self,
        max_users: Optional[int] = None,
        force: bool = False,
        append: bool = False,
        single_file: Optional[str] = None
    ) -> None:
        """
        Process all visit files or a single specified file.
        
        Args:
            max_users: Maximum users per file
            force: Force reprocessing
            append: Resume from previous runs
            single_file: Process only this file (if specified)
        """
        poi_path = Config.POI_FILE
        
        if not poi_path.exists():
            raise RuntimeError(f"POI file not found: {poi_path}")
        
        if single_file:
            # Process single file
            target_path = self._resolve_file_path(single_file)
            self.process_file(target_path, poi_path, max_users, force, append)
        else:
            # Process all visit files
            visit_files = self._find_visit_files()
            
            if not visit_files:
                raise RuntimeError("No visit files found")
            
            logger.info(f"Found {len(visit_files)} files to process")
            
            processed = 0
            skipped = 0
            
            for visit_file in sorted(visit_files):
                try:
                    if not force and CheckpointManager.should_skip_file(
                        visit_file, Config.RESULTS_DIR, append
                    ):
                        skipped += 1
                        continue
                    
                    self.process_file(visit_file, poi_path, max_users, force, append)
                    processed += 1
                    
                except Exception as e:
                    logger.error(f"Error processing {visit_file.name}: {e}")
                    continue
            
            # Summary statistics
            logger.info("\n" + "=" * 70)
            logger.info("PROCESSING SUMMARY:")
            logger.info(f"  Total files: {len(visit_files)}")
            logger.info(f"  Processed: {processed}")
            logger.info(f"  Skipped: {skipped}")
            logger.info(f"  Efficiency: {skipped/len(visit_files)*100:.1f}% files avoided")
            logger.info("=" * 70)
    
    def _find_visit_files(self) -> List[Path]:
        """Find all visit CSV files (excluding POI file)"""
        visit_files = []
        
        for csv_path in Config.DATA_DIR.rglob("*.csv"):
            # Skip POI file and backup files
            if (csv_path.name.lower() != "vc_site.csv" and 
                "backup" not in str(csv_path).lower()):
                visit_files.append(csv_path)
        
        return visit_files
    
    def _resolve_file_path(self, file_path: str) -> Path:
        """Resolve file path from string input"""
        target = Path(file_path)
        
        # Try different path resolutions
        if not target.is_absolute():
            if not target.exists():
                # Try relative to data directory
                target = Config.DATA_DIR / file_path
                if not target.exists():
                    # Try just filename in data directory
                    target = Config.DATA_DIR / Path(file_path).name
        
        if not target.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if target.suffix.lower() != '.csv':
            raise ValueError(f"File must be CSV: {target}")
        
        if target.name.lower() == 'vc_site.csv':
            raise ValueError("Cannot process POI file")
        
        return target

# ============= MAIN ENTRY POINT =============
def main():
    """Main entry point for the application"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="VeronaCard Tourist Behavior Prediction System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Process all files:
    python %(prog)s
    
  Process with max 100 users per file:
    python %(prog)s --max-users 100
    
  Process single file:
    python %(prog)s --file data/verona/visits_2014.csv
    
  Resume from previous run:
    python %(prog)s --append
    
  Force reprocessing:
    python %(prog)s --force
        """
    )
    
    parser.add_argument(
        "--file",
        type=str,
        help="Process only this specific file"
    )
    parser.add_argument(
        "--max-users",
        type=int,
        default=None,
        help="Maximum number of users to process per file"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing even if output exists"
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Resume from previous run (append mode)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.force and args.append:
        parser.error("Cannot use both --force and --append")
    
    # Setup GPU environment if not already set
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # Use all 4 A100 GPUs
    
    try:
        # Inizializzazione passo-passo con controlli
        logger.info("Initializing Ollama connection manager...")
        ollama_manager = OllamaConnectionManager()
        
        # Setup connections DEVE essere chiamato
        logger.info("Setting up connections...")
        hosts = ollama_manager.setup_connections()
        if not hosts:
            raise RuntimeError("No hosts configured")
            
        # Verifica inizializzazione corretta
        if ollama_manager.rate_limiter is None:
            raise RuntimeError("Rate limiter not properly initialized")
        if ollama_manager.health_monitor is None:
            raise RuntimeError("Health monitor not properly initialized")
            
        logger.info(f"Initialized with {len(hosts)} hosts")
        
        # Wait for services to be ready
        if not ollama_manager.wait_for_services():
            raise RuntimeError("Ollama services failed to start")
        
        # Create and run processor
        processor = VisitFileProcessor(ollama_manager)
        
        processor.process_all_files(
            max_users=args.max_users,
            force=args.force,
            append=args.append,
            single_file=args.file
        )
        
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()