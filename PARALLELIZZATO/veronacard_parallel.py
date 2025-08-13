#!/usr/bin/env python3
"""
Versione parallelizzata per HPC Leonardo con multiple A100
Ottimizzazioni principali:
1. Multiprocessing per i file
2. Threading per le richieste LLM concurrent
3. Batch processing ottimizzato
4. Load balancing intelligente
"""

import json
import random
import time
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
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from functools import partial
import threading
from queue import Queue, Empty
import signal
import psutil

# ---------------- Configurazione HPC ----------------
# Rileva automaticamente le risorse disponibili
AVAILABLE_CORES = psutil.cpu_count(logical=False)  # Core fisici
AVAILABLE_MEMORY_GB = psutil.virtual_memory().total / (1024**3)

# Configurazione ottimale per Leonardo
MAX_WORKERS_FILES = min(4, AVAILABLE_CORES // 4)  # Processi per file paralleli
MAX_WORKERS_LLM = 8   # Thread per richieste LLM concurrent per processo
BATCH_SIZE = 50       # Utenti per batch
QUEUE_TIMEOUT = 300   # Timeout code per evitare deadlock

print(f"🖥️  HPC Info: {AVAILABLE_CORES} cores, {AVAILABLE_MEMORY_GB:.1f}GB RAM")
print(f"⚙️  Config: {MAX_WORKERS_FILES} file workers, {MAX_WORKERS_LLM} LLM threads")

# ---------------- logging setup ottimizzato ----------------
LOG_DIR = Path(__file__).resolve().parent / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"run_parallel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# Logger thread-safe per multiprocessing
def setup_logger(name: str, log_file: Path = None) -> logging.Logger:
    """Setup logger thread-safe per multiprocessing"""
    logger = logging.getLogger(name)
    if logger.handlers:  # Evita duplicati
        return logger
        
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(processName)s-%(threadName)s] %(levelname)s: %(message)s"
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler se specificato
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

logger = setup_logger(__name__, LOG_FILE)

# ---------------- Connection Pool per Ollama ----------------
class OllamaConnectionPool:
    """Pool di connessioni riutilizzabili per Ollama con load balancing"""
    
    def __init__(self, host: str, pool_size: int = 20):
        self.host = host
        self.pool_size = pool_size
        self._session_queue = Queue(maxsize=pool_size)
        self._create_sessions()
        self._stats = {
            'requests': 0,
            'successes': 0,
            'failures': 0,
            'avg_response_time': 0
        }
        self._stats_lock = threading.Lock()
    
    def _create_sessions(self):
        """Crea pool di sessioni HTTP riutilizzabili"""
        for _ in range(self.pool_size):
            session = requests.Session()
            session.headers.update({
                'Content-Type': 'application/json',
                'Connection': 'keep-alive',
                'User-Agent': 'LLM-Mob-HPC-Parallel/2.0'
            })
            # Configurazione ottimizzata per HPC
            adapter = requests.adapters.HTTPAdapter(
                pool_connections=10,
                pool_maxsize=10,
                max_retries=0  # Gestiamo retry manualmente
            )
            session.mount('http://', adapter)
            session.mount('https://', adapter)
            self._session_queue.put(session)
    
    def get_session(self, timeout: float = 5.0) -> requests.Session:
        """Ottieni una sessione dal pool"""
        try:
            return self._session_queue.get(timeout=timeout)
        except Empty:
            # Se pool esaurito, crea sessione temporanea
            logger.warning("⚠️  Pool sessioni esaurito, creo sessione temporanea")
            session = requests.Session()
            session.headers.update({'Content-Type': 'application/json'})
            return session
    
    def return_session(self, session: requests.Session):
        """Restituisci sessione al pool"""
        try:
            self._session_queue.put_nowait(session)
        except:
            pass  # Se queue piena, ignora
    
    def update_stats(self, success: bool, response_time: float):
        """Aggiorna statistiche thread-safe"""
        with self._stats_lock:
            self._stats['requests'] += 1
            if success:
                self._stats['successes'] += 1
            else:
                self._stats['failures'] += 1
            
            # Media mobile del tempo di risposta
            alpha = 0.1
            self._stats['avg_response_time'] = (
                alpha * response_time + 
                (1 - alpha) * self._stats['avg_response_time']
            )
    
    def get_stats(self) -> Dict[str, Any]:
        """Ottieni statistiche correnti"""
        with self._stats_lock:
            success_rate = (
                self._stats['successes'] / max(1, self._stats['requests'])
            )
            return {
                **self._stats.copy(),
                'success_rate': success_rate
            }

# Pool globale di connessioni
connection_pool = None

def init_connection_pool(host: str):
    """Inizializza il pool globale di connessioni"""
    global connection_pool
    connection_pool = OllamaConnectionPool(host, pool_size=MAX_WORKERS_LLM * 2)

# ---------------- LLM Request ottimizzato ----------------
def get_chat_completion_parallel(
    prompt: str, 
    model: str = "llama3.1:8b", 
    max_retries: int = 2,
    timeout: float = 180
) -> Optional[str]:
    """
    Versione ottimizzata per requests paralleli con connection pooling
    """
    global connection_pool
    if not connection_pool:
        logger.error("❌ Connection pool non inizializzato")
        return None
    
    session = None
    start_time = time.time()
    
    for attempt in range(1, max_retries + 1):
        try:
            # Ottieni sessione dal pool
            session = connection_pool.get_session(timeout=5)
            
            # Payload ottimizzato per A100
            payload = {
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "top_p": 0.9,
                    "num_ctx": 1024,
                    "num_predict": 100,
                    "num_thread": 2,  # Ridotto per parallelismo
                    "num_batch": 32,
                    "num_gpu": 33,
                    "low_vram": False,
                    "f16_kv": True,
                    "stop": ["\n\n", "```", "---"]
                }
            }
            
            # Timeout adattivo
            request_timeout = min(timeout / attempt, 180)
            
            resp = session.post(
                f"{connection_pool.host}/api/chat",
                json=payload,
                timeout=request_timeout
            )
            
            resp.raise_for_status()
            response_data = resp.json()
            
            # Verifica completezza
            if response_data.get("done", False):
                content = response_data.get("message", {}).get("content", "")
                if content.strip():
                    elapsed = time.time() - start_time
                    connection_pool.update_stats(True, elapsed)
                    return content
            
            logger.warning(f"⚠️  Risposta incompleta (tentativo {attempt})")
            
        except requests.exceptions.Timeout:
            logger.warning(f"⚠️  Timeout {request_timeout}s (tentativo {attempt})")
        except requests.exceptions.HTTPError as e:
            logger.warning(f"⚠️  HTTP {e.response.status_code} (tentativo {attempt})")
        except Exception as e:
            logger.warning(f"⚠️  Errore: {e} (tentativo {attempt})")
        
        finally:
            # Restituisci sempre la sessione al pool
            if session:
                connection_pool.return_session(session)
                session = None
        
        # Backoff breve tra tentativi
        if attempt < max_retries:
            time.sleep(min(attempt * 2, 10))
    
    elapsed = time.time() - start_time
    connection_pool.update_stats(False, elapsed)
    return None

# ---------------- Processamento batch utenti ----------------
def process_user_batch(
    batch_data: Tuple[List[str], DataFrame, DataFrame, DataFrame, str, int]
) -> List[Dict[str, Any]]:
    """
    Processa un batch di utenti in parallelo con thread pool
    """
    card_ids, filtered_df, user_clusters, pois_df, anchor_rule, top_k = batch_data
    
    # Setup logger per questo processo
    proc_logger = setup_logger(f"batch_worker_{os.getpid()}")
    proc_logger.info(f"🔄 Processando batch di {len(card_ids)} utenti")
    
    def process_single_user(cid: str) -> Dict[str, Any]:
        """Processa singolo utente"""
        try:
            # Ottieni sequenza utente
            user_visits = filtered_df[filtered_df.card_id == cid].sort_values("timestamp")
            seq = user_visits["name_short"].tolist()
            
            if len(seq) < 3:
                return None
            
            target = seq[-1]
            idx_anchor = anchor_index(len(seq) - 1, anchor_rule)
            history_list = [p for i, p in enumerate(seq[:-1]) if i != idx_anchor]
            current_poi = seq[:-1][idx_anchor]
            
            # Crea prompt
            prompt = create_prompt_with_cluster_optimized(
                cid, user_clusters, pois_df, history_list, current_poi, top_k
            )
            
            # Richiesta LLM
            response = get_chat_completion_parallel(prompt, timeout=120)
            
            # Costruisci risultato
            result = {
                "card_id": cid,
                "cluster": get_user_cluster(user_clusters, cid),
                "history": str(history_list),
                "current_poi": current_poi,
                "prediction": None,
                "ground_truth": target,
                "reason": None,
                "hit": False
            }
            
            # Parsing risposta
            if response:
                try:
                    obj = json.loads(response)
                    pred = obj.get("prediction", [])
                    pred_lst = pred if isinstance(pred, list) else [pred]
                    result["prediction"] = str(pred_lst)
                    result["reason"] = obj.get("reason", "")
                    result["hit"] = target in pred_lst
                except json.JSONDecodeError:
                    proc_logger.warning(f"⚠️  JSON invalido per utente {cid}")
            
            return result
            
        except Exception as e:
            proc_logger.error(f"❌ Errore processando utente {cid}: {e}")
            return None
    
    # Processo utenti in parallelo con ThreadPool
    results = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS_LLM) as executor:
        future_to_user = {
            executor.submit(process_single_user, cid): cid 
            for cid in card_ids
        }
        
        for future in as_completed(future_to_user, timeout=600):  # 10 min timeout
            try:
                result = future.result(timeout=30)
                if result:
                    results.append(result)
            except Exception as e:
                user_id = future_to_user[future]
                proc_logger.error(f"❌ Fallito utente {user_id}: {e}")
    
    proc_logger.info(f"✓ Batch completato: {len(results)}/{len(card_ids)} utenti")
    return results

# ---------------- Funzioni helper ottimizzate ----------------
def create_prompt_with_cluster_optimized(
    card_id: str,
    user_clusters: pd.DataFrame,
    pois_df: pd.DataFrame,
    history: List[str],
    current_poi: str,
    top_k: int = 5
) -> str:
    """Versione ottimizzata del prompt builder"""
    try:
        cluster_id = user_clusters.loc[
            user_clusters["card_id"] == card_id, "cluster"
        ].iloc[0]
    except (IndexError, KeyError):
        cluster_id = 0
    
    # Cache delle coordinate POI per performance
    poi_coords = pois_df.set_index("name_short")[["latitude", "longitude"]].to_dict("index")
    
    if current_poi not in poi_coords:
        return f"Errore: POI {current_poi} non trovato"
    
    current_lat = poi_coords[current_poi]["latitude"]
    current_lon = poi_coords[current_poi]["longitude"]
    
    # Calcolo distanze vettorizzato
    visited_set = set(history + [current_poi])
    nearby_pois = []
    
    for poi_name, coords in poi_coords.items():
        if poi_name in visited_set:
            continue
        
        distance = calculate_distance(
            current_lat, current_lon,
            coords["latitude"], coords["longitude"]
        )
        
        if distance <= 2.5:  # Max 2.5km nel centro di Verona
            nearby_pois.append((poi_name, distance))
    
    # Top 8 più vicini per ridurre lunghezza prompt
    nearby_pois.sort(key=lambda x: x[1])
    pois_list = ", ".join([
        f"{name} ({dist:.1f}km)" for name, dist in nearby_pois[:8]
    ])
    
    return f"""Turista cluster {cluster_id} a Verona.
Visitati: {', '.join(history) if history else 'nessuno'}
Attuale: {current_poi}
Disponibili: {pois_list}

Suggerisci {top_k} POI più probabili come prossime visite.
Rispondi SOLO JSON: {{"prediction": ["poi1", "poi2", ...], "reason": "breve"}}"""

def split_users_into_batches(card_ids: List[str], batch_size: int = BATCH_SIZE) -> List[List[str]]:
    """Divide utenti in batch per parallelizzazione"""
    return [card_ids[i:i + batch_size] for i in range(0, len(card_ids), batch_size)]

# ---------------- File Processing parallelizzato ----------------
def process_file_parallel(
    file_data: Tuple[Path, Path, Optional[int], bool, bool, str]
) -> Optional[str]:
    """
    Processa un singolo file in modo parallelizzato
    """
    visits_path, poi_path, max_users, force, append, anchor_rule = file_data
    
    # Setup logger per questo processo
    proc_logger = setup_logger(f"file_worker_{os.getpid()}")
    
    try:
        proc_logger.info(f"🔄 Inizio processamento: {visits_path.name}")
        
        # Inizializza connection pool per questo processo
        ollama_host, _ = setup_ollama_connection()
        init_connection_pool(ollama_host)
        
        # Caricamento e preprocessing dati
        pois = load_pois(poi_path)
        visits = load_visits(visits_path)
        merged = merge_visits_pois(visits, pois)
        filtered = filter_multi_visit_cards(merged)
        
        if filtered.empty:
            proc_logger.warning(f"⚠️  Nessun dato valido in {visits_path.name}")
            return None
        
        # Gestione modalità append
        processed_cards = set()
        out_dir = Path(__file__).resolve().parent / "results"
        out_dir.mkdir(exist_ok=True)
        
        if append:
            prev_path = latest_output(visits_path, out_dir)
            if prev_path and prev_path.exists():
                try:
                    prev_df = pd.read_csv(prev_path, usecols=['card_id'])
                    processed_cards = set(prev_df['card_id'])
                    proc_logger.info(f"📋 Trovate {len(processed_cards)} card già processate")
                except Exception as e:
                    proc_logger.warning(f"⚠️  Errore leggendo file precedente: {e}")
        
        # Filtra card già processate
        if processed_cards:
            filtered = filtered[~filtered['card_id'].isin(processed_cards)]
            if filtered.empty:
                proc_logger.info(f"✓ {visits_path.name}: tutte le card già processate")
                return visits_path.name
        
        # Clustering
        matrix = create_user_poi_matrix(filtered)
        if matrix.empty:
            proc_logger.warning(f"⚠️  Matrice vuota per {visits_path.name}")
            return None
            
        clusters = KMeans(n_clusters=min(7, len(matrix)), random_state=42, n_init=10)\
                  .fit_predict(StandardScaler().fit_transform(matrix))
        user_clusters = pd.DataFrame({"card_id": matrix.index, "cluster": clusters})
        
        # Utenti idonei
        eligible = (
            filtered.groupby("card_id").size()
            .loc[lambda s: s >= 3].index.tolist()
        )
        
        if not eligible:
            proc_logger.warning(f"⚠️  Nessun utente idoneo in {visits_path.name}")
            return None
        
        # Campionamento se richiesto (se max_users=None, processa tutti)
        if max_users and len(eligible) > max_users:
            eligible = random.sample(eligible, max_users)
            proc_logger.info(f"🎲 Campionamento: {max_users} utenti su {len(eligible)} totali")
        else:
            proc_logger.info(f"💯 Processando TUTTI i {len(eligible)} utenti idonei")
        
        proc_logger.info(f"👥 Processando {len(eligible)} utenti idonei")
        
        # Split in batch e processamento parallelo
        batches = split_users_into_batches(eligible, BATCH_SIZE)
        all_results = []
        
        # Processa batch sequenzialmente ma utenti in parallelo
        for i, batch_cards in enumerate(batches):
            proc_logger.info(f"📦 Batch {i+1}/{len(batches)}: {len(batch_cards)} utenti")
            
            batch_data = (
                batch_cards, filtered, user_clusters, pois, anchor_rule, 5  # TOP_K
            )
            
            batch_results = process_user_batch(batch_data)
            all_results.extend(batch_results)
            
            # Log progresso
            if i % 3 == 0 and connection_pool:
                stats = connection_pool.get_stats()
                proc_logger.info(
                    f"📊 LLM Stats: {stats['success_rate']:.1%} success, "
                    f"{stats['avg_response_time']:.1f}s avg"
                )
        
        # Salvataggio risultati
        if all_results:
            df_out = pd.DataFrame(all_results)
            
            # Determina file output
            if append:
                output_file = latest_output(visits_path, out_dir)
                if not output_file:
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    output_file = out_dir / f"{visits_path.stem}_pred_{ts}.csv"
                    write_header = True
                else:
                    write_header = False
            else:
                ts = time.strftime("%Y%m%d_%H%M%S")
                output_file = out_dir / f"{visits_path.stem}_pred_{ts}.csv"
                write_header = True
            
            # Salva con lock per evitare race condition
            if append and not write_header:
                df_out.to_csv(output_file, mode="a", header=False, index=False)
            else:
                df_out.to_csv(output_file, index=False)
            
            hit_rate = df_out["hit"].mean()
            proc_logger.info(
                f"✅ {visits_path.name}: {len(all_results)} risultati, "
                f"Hit@5: {hit_rate:.2%}, salvato: {output_file.name}"
            )
            
            return visits_path.name
        else:
            proc_logger.warning(f"⚠️  Nessun risultato per {visits_path.name}")
            return None
            
    except Exception as e:
        proc_logger.error(f"❌ Errore processando {visits_path.name}: {e}")
        import traceback
        proc_logger.error(traceback.format_exc())
        return None

# ---------------- Main parallelizzato ----------------
def run_all_verona_logs_parallel(
    max_users: Optional[int] = None,
    force: bool = False,
    append: bool = False,
    anchor_rule: str = "penultimate"
) -> None:
    """
    Versione parallelizzata del processamento di tutti i log
    """
    logger.info("🚀 Avvio processamento parallelizzato")
    
    ROOT = Path(__file__).resolve().parent
    poi_csv = ROOT / "data" / "verona" / "vc_site.csv"
    
    # Trova tutti i CSV di visite
    visit_csvs = [
        p for p in (ROOT / "data" / "verona").rglob("*.csv")
        if p.name != "vc_site.csv" and "backup" not in str(p).lower()
    ]
    
    if not visit_csvs:
        raise RuntimeError("Nessun CSV di visite trovato")
    
    logger.info(f"📂 Trovati {len(visit_csvs)} file da processare")
    
    # Prepara dati per processi paralleli
    file_data_list = [
        (csv_path, poi_csv, max_users, force, append, anchor_rule)
        for csv_path in sorted(visit_csvs)
    ]
    
    # Processamento parallelo con ProcessPoolExecutor
    completed_files = []
    failed_files = []
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=MAX_WORKERS_FILES) as executor:
        # Submit tutti i job
        future_to_file = {
            executor.submit(process_file_parallel, file_data): file_data[0].name
            for file_data in file_data_list
        }
        
        # Processa risultati man mano che completano
        for future in as_completed(future_to_file, timeout=3600):  # 1h timeout
            file_name = future_to_file[future]
            try:
                result = future.result(timeout=60)
                if result:
                    completed_files.append(result)
                    logger.info(f"✅ Completato: {result}")
                else:
                    failed_files.append(file_name)
                    logger.warning(f"⚠️  Fallito: {file_name}")
                    
            except Exception as e:
                failed_files.append(file_name)
                logger.error(f"❌ Errore {file_name}: {e}")
    
    # Statistiche finali
    total_time = time.time() - start_time
    logger.info("=" * 70)
    logger.info("📊 STATISTICHE FINALI")
    logger.info("=" * 70)
    logger.info(f"⏱️  Tempo totale: {total_time/60:.1f} minuti")
    logger.info(f"✅ File completati: {len(completed_files)}")
    logger.info(f"❌ File falliti: {len(failed_files)}")
    
    if completed_files:
        logger.info("✅ File completati:")
        for fname in completed_files:
            logger.info(f"   • {fname}")
    
    if failed_files:
        logger.info("❌ File falliti:")
        for fname in failed_files:
            logger.info(f"   • {fname}")

# ---------------- Import delle funzioni originali ----------------
# Importa le funzioni helper dal codice originale
def anchor_index(seq_len: int, rule: str | int) -> int:
    """Funzione anchor_index dal codice originale"""
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

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Formula haversine dal codice originale"""
    R = 6371  # Raggio della Terra in km
    
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c

# Altre funzioni helper importate...
def load_pois(filepath: str | Path) -> DataFrame:
    df = pd.read_csv(filepath, usecols=["name_short", "latitude", "longitude"])
    logger.info(f"[load_pois] {len(df)} POI letti da {filepath}")
    return df

def load_visits(filepath: str | Path) -> DataFrame:
    df = pd.read_csv(
        filepath,
        usecols=[0, 1, 2, 4],
        names=["data", "ora", "name_short", "card_id"],
        header=0,
        dtype={"card_id": str},
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

def get_user_cluster(user_clusters: pd.DataFrame, card_id: str) -> int:
    matching_rows = user_clusters[user_clusters.card_id == card_id]
    if matching_rows.empty:
        return 0  # Default cluster
    return int(matching_rows["cluster"].iloc[0])

def latest_output(visits_path: Path, out_dir: Path) -> Path | None:
    """L'output più recente (None se non esiste)."""
    pattern = f"{visits_path.stem}_pred_*.csv"
    outputs = list(out_dir.glob(pattern))
    return max(outputs, key=os.path.getmtime) if outputs else None

def setup_ollama_connection():
    """Setup connessione Ollama dal file di porta"""
    OLLAMA_PORT_FILE = "ollama_port.txt"
    try:
        with open(OLLAMA_PORT_FILE, "r") as f:
            port = f.read().strip()
        ollama_host = f"http://127.0.0.1:{port}"
        return ollama_host, port
    except FileNotFoundError:
        raise RuntimeError(f"❌ File {OLLAMA_PORT_FILE} non trovato")

# ---------------- MAIN ----------------
if __name__ == "__main__":
    # Setup gestione segnali per cleanup
    def signal_handler(signum, frame):
        logger.info("🛑 Interruzione ricevuta, cleanup in corso...")
        sys.exit(1)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    parser = argparse.ArgumentParser(
        description="Versione parallelizzata per HPC Leonardo - Calcola raccomandazioni su tutti i log VeronaCard"
    )
    parser.add_argument("--force", action="store_true",
                        help="ricalcola e sovrascrive anche se gli output esistono")
    parser.add_argument("--append", action="store_true", 
                        help="riprende da dove si era interrotto")
    parser.add_argument("--max-users", type=int, default=None,
                        help="numero massimo di utenti da processare per file (None = tutti)")
    parser.add_argument("--anchor", type=str, default="penultimate",
                        dest="anchor_rule",
                        help="Regola per scegliere il POI ancora (penultimate|first|middle|int)")
    parser.add_argument("--parallel-files", type=int, default=MAX_WORKERS_FILES,
                        help=f"numero di file da processare in parallelo (default: {MAX_WORKERS_FILES})")
    parser.add_argument("--parallel-llm", type=int, default=MAX_WORKERS_LLM,
                        help=f"numero di thread LLM per processo (default: {MAX_WORKERS_LLM})")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help=f"dimensione batch utenti (default: {BATCH_SIZE})")
    parser.add_argument("--dry-run", action="store_true",
                        help="esegue solo il test delle connessioni senza processare")
    
    args = parser.parse_args()
    
    if args.force and args.append:
        parser.error("Non puoi usare insieme --force e --append")
    
    # Aggiorna configurazioni globali se specificate
    if args.parallel_files != MAX_WORKERS_FILES:
        MAX_WORKERS_FILES = args.parallel_files
    if args.parallel_llm != MAX_WORKERS_LLM:
        MAX_WORKERS_LLM = args.parallel_llm
    if args.batch_size != BATCH_SIZE:
        BATCH_SIZE = args.batch_size
    
    logger.info(f"🔧 Config finale: {MAX_WORKERS_FILES} file workers, {MAX_WORKERS_LLM} LLM threads, batch size {BATCH_SIZE}")
    
    # Test connessione Ollama
    try:
        ollama_host, ollama_port = setup_ollama_connection()
        logger.info(f"🔗 Ollama host: {ollama_host}")
        
        # Test di connettività
        if not wait_for_ollama(ollama_host, max_attempts=10):
            raise RuntimeError("❌ Ollama non risponde")
        
        logger.info("✅ Connessione Ollama verificata")
        
        # Test preliminare del modello
        test_response = get_chat_completion_parallel(
            "Test", model="llama3.1:8b", timeout=60
        )
        if test_response:
            logger.info("✅ Test modello LLM superato")
        else:
            logger.warning("⚠️  Test modello fallito, ma continuo...")
        
    except Exception as e:
        logger.error(f"❌ Setup Ollama fallito: {e}")
        sys.exit(1)
    
    # Modalità dry-run per test
    if args.dry_run:
        logger.info("🧪 Modalità dry-run: solo test di connettività")
        sys.exit(0)
    
    # Inizializza connection pool globale
    init_connection_pool(ollama_host)
    
    try:
        # Avvia processamento parallelizzato
        run_all_verona_logs_parallel(
            max_users=args.max_users,
            force=args.force,
            append=args.append,
            anchor_rule=args.anchor_rule
        )
        
        logger.info("🎉 Processamento completato con successo!")
        
    except KeyboardInterrupt:
        logger.info("⏹️  Interruzione manuale ricevuta")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Errore fatale: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)

# ---------------- Funzioni helper aggiuntive ----------------

def wait_for_ollama(ollama_host: str, max_attempts: int = 30, wait_interval: int = 3) -> bool:
    """Attende che Ollama sia pronto"""
    logger.info(f"🔄 Attesa Ollama su {ollama_host}...")
    
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(f"{ollama_host}/api/tags", timeout=10)
            if response.status_code == 200:
                logger.info("✅ Ollama completamente attivo")
                return True
            else:
                logger.debug(f"🔄 Tentativo {attempt}/{max_attempts}: HTTP {response.status_code}")
                
        except requests.exceptions.RequestException as e:
            logger.debug(f"🔄 Tentativo {attempt}/{max_attempts}: {type(e).__name__}")
        
        if attempt < max_attempts:
            time.sleep(wait_interval)
    
    return False

def monitor_system_resources():
    """Monitora risorse sistema durante l'esecuzione"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        logger.info(
            f"📊 Sistema: CPU {cpu_percent:.1f}%, "
            f"RAM {memory.percent:.1f}% ({memory.used/1024**3:.1f}GB used)"
        )
        
        # GPU info se disponibile
        try:
            import subprocess
            gpu_info = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", 
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            if gpu_info.returncode == 0:
                gpu_util, gpu_mem_used, gpu_mem_total = gpu_info.stdout.strip().split(', ')
                gpu_mem_percent = (int(gpu_mem_used) / int(gpu_mem_total)) * 100
                logger.info(f"🎮 GPU: {gpu_util}% util, {gpu_mem_percent:.1f}% memoria")
        except:
            pass  # GPU monitoring opzionale
            
    except Exception as e:
        logger.debug(f"Monitoring error: {e}")

# Avvia monitoring in background thread
def start_background_monitoring():
    """Avvia monitoring risorse in background"""
    def monitor_loop():
        while True:
            try:
                monitor_system_resources()
                time.sleep(60)  # Ogni minuto
            except Exception:
                break
    
    monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
    monitor_thread.start()
    return monitor_thread