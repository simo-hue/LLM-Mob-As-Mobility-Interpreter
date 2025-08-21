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


# Lock globale per scrittura thread-safe sui file CSV
write_lock = Lock()

# -----------------------------------------------------------
TOP_K  = 5          # deve coincidere con top_k del prompt
MODEL_NAME = "llama3.1:8b"
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
def get_chat_completion(prompt: str, model: str = MODEL_NAME, max_retries: int = 2) -> str | None:
    """Versione load-balanced della chiamata"""
    
    for attempt in range(1, max_retries + 1):
        # QUESTA È LA PARTE NUOVA: round-robin tra gli host
        current_host = next(host_cycle)
        
        try:
            # Health check veloce (sostituisci OLLAMA_HOST con current_host)
            health_check = requests.get(f"{current_host}/api/tags", timeout=3)
            if health_check.status_code != 200:
                logger.warning(f"Host {current_host} non risponde (tentativo {attempt}/{max_retries})")
                continue

        except requests.exceptions.RequestException:
            logger.warning(f"Health check fallito per {current_host}")
            continue

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}], 
            "stream": False
        }
        
        try:
            logger.debug(f"Richiesta a {current_host} (tentativo {attempt}/{max_retries})")
            
            timeout = 300 if attempt == 1 else 180
            
            # SOSTITUISCI OLLAMA_HOST con current_host
            resp = requests.post(
                f"{current_host}/api/chat", 
                json=payload, 
                timeout=timeout,
                headers={'Content-Type': 'application/json'}
            )
            
            resp.raise_for_status()
            response_data = resp.json()
            
            if not response_data.get("done", False):
                logger.warning(f"Risposta incompleta da {current_host}")
                continue
            
            content = response_data.get("message", {}).get("content", "")
            if content:
                logger.debug(f"Risposta da {current_host} ({len(content)} caratteri)")
                return content
                
        except requests.exceptions.Timeout:
            logger.error(f"Timeout {timeout}s su {current_host}")
        except Exception as exc:
            logger.error(f"Errore su {current_host}: {exc}")
        
        if attempt < max_retries:
            wait_time = min(attempt * 5, 15)
            time.sleep(wait_time)
    
    logger.error("Tutti i tentativi falliti su tutti gli host")
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
            f"{OLLAMA_HOST}/api/generate",
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
    """Setup connessioni multiple Ollama con fallback"""
    try:
        with open("ollama_ports.txt", "r") as f:
            ports_str = f.read().strip()
        
        if "," in ports_str:
            # Multi-istanza
            ports = ports_str.split(",")
            hosts = [f"http://127.0.0.1:{port.strip()}" for port in ports]
            logger.info(f"🔥 Configurazione multi-GPU: {len(hosts)} istanze")
            for i, host in enumerate(hosts):
                logger.info(f"   GPU {i}: {host}")
            return hosts
        else:
            # Singola istanza (fallback)
            host = f"http://127.0.0.1:{ports_str}"
            logger.info(f"⚠️  Fallback singola GPU: {host}")
            return [host]
            
    except FileNotFoundError:
        raise RuntimeError("❌ File ollama_ports.txt non trovato")
    except Exception as e:
        logger.error(f"❌ Errore setup connessioni: {e}")
        raise

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
    
    try:
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

        prompt = create_prompt_with_cluster(
            filtered, user_clusters, pois, cid,  
            top_k=top_k, anchor_rule=anchor_rule
        )
        
        ans = get_chat_completion(prompt)

        rec = {
            "card_id": cid,
            "cluster": get_user_cluster(user_clusters, cid),
            "history": str(history_list),
            "current_poi": current_poi,
            "prediction": None,
            "ground_truth": target,
            "reason": None,
            "hit": False
        }

        if ans:
            try:
                obj = json.loads(ans)
                pred = obj["prediction"]
                pred_lst = pred if isinstance(pred, list) else [pred]
                rec["prediction"] = str(pred_lst)
                rec["reason"] = obj.get("reason")
                rec["hit"] = target in pred_lst
            except Exception:
                pass
                
        return rec
        
    except Exception as e:
        logger.error(f"Errore processando card {cid}: {e}")
        return None

# ---------- test su un singolo file ---------------------------------------
def run_on_visits_file(
    visits_path: Path, 
    poi_path: Path, 
    *, 
    max_users: int | None = None, 
    force: bool = False, 
    append: bool = False, 
    anchor_rule: str | int = DEFAULT_ANCHOR_RULE,
    save_every: int = 500 # per non perdere troppo
) -> None:
    """
    Esegue l'intera pipeline (carica, clusterizza, predice, salva) su un
    singolo file di log VeronaCard.

    Parameters
    ----------
    visits_path : Path
        CSV con le timbrature (es. dati_2014.csv).
    poi_path : Path
        CSV vc_site.csv (reference POI).
    max_users : int
        Quante card valutare (per ridurre tempi).
    """
    
    # ---------- 0. CHECK SKIP ANTICIPATO ----------
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    
    # Controllo skip PRIMA di qualsiasi elaborazione pesante
    if should_skip_file(visits_path, poi_path, out_dir, append):
        return  # Esce immediatamente

    # ---------- Modalità force / append / skip ----------
    if force and append:
        raise ValueError("Non puoi usare --force e --append insieme.")

    if append:
        processed_cards = load_completed_cards_fast(visits_path, out_dir)
    else:
        processed_cards = set()
        # Se non siamo in append, rimuovi eventuali checkpoint vecchi
        checkpoint_file = get_checkpoint_file(visits_path, out_dir)
        if checkpoint_file.exists():
            checkpoint_file.unlink()


    if not force and not append and latest_output(visits_path, out_dir):
        logger.info(f" Output esistente per {visits_path.stem}. Usa --force o --append.")
        return
    
    # ---------- 0.5 Scrivo su LOG ----------
    logger.info("\n" + "=" * 70)
    logger.info(f"▶  PROCESSO FILE: {visits_path.name}")
    logger.info("=" * 70)

    logger.info(f"▶  ho caricato e pulito i dati")
    # ---------- 1. load & clean ----------
    pois     = load_pois(poi_path)
    visits   = load_visits(visits_path)
    merged   = merge_visits_pois(visits, pois)
    filtered = filter_multi_visit_cards(merged)
    
    # Se siamo in modalità --append, togliamo le card già processate
    if processed_cards:
        filtered = filtered[~filtered['card_id'].isin(processed_cards)]
        if filtered.empty:
            logger.info(f"Tutte le card di {visits_path.stem} erano già elaborate. Skip.")
            return

    # ---------- 2. clustering ----------
    matrix   = create_user_poi_matrix(filtered)
    clusters = KMeans(n_clusters=7, random_state=42, n_init=10)\
              .fit_predict(StandardScaler().fit_transform(matrix))
    user_clusters = pd.DataFrame({"card_id": matrix.index, "cluster": clusters})
    
    logger.info(f"▶  ho fatto il clustering")

    # ---------- 3. utenti idonei ----------
    eligible = (
        filtered.groupby("card_id").size()
        .loc[lambda s: s >= 3].index.tolist()
    )
    if max_users is None:          # process *all* eligible users
        demo_cards = eligible
    else:                          # random sub‑sample for quick runs
        demo_cards = random.sample(eligible, k=min(max_users, len(eligible)))

    # ---------- 4. Gestione file di output ----------
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    
    # Determina il file di output in base alla modalità
    if append:
        # In modalità append, usa il file più recente esistente
        output_file = latest_output(visits_path, out_dir)
        if output_file is None:
            # Se non esiste un file precedente, crea un nuovo file
            ts = time.strftime("%Y%m%d_%H%M%S")
            output_file = out_dir / f"{visits_path.stem}_pred_{ts}.csv"
            write_header = True
        else:
            write_header = False
    else:
        # In modalità normale, crea sempre un nuovo file
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_file = out_dir / f"{visits_path.stem}_pred_{ts}.csv"
        write_header = True
    
    #logger.info("▶  Analisi pattern di movimento geografico")
    #movement_patterns = analyze_movement_patterns(filtered, pois)
    #avg_distance = movement_patterns["distance"].mean()
    #logger.info(f"▶  Distanza media tra visite consecutive: {avg_distance:.2f} km")

    results_list = []
    processed_count = 0
    first_save = True
    
    # ---------- 5. ciclo parallelo su utenti ----------
    n_gpus = 4  # A100 disponibili su Leonardo
    n_workers = min(n_gpus * 2, 8)  # Max 2 richieste concurrent per GPU ( per non saturare OLLAMA )
    
    logger.info(f"🚀 Avvio elaborazione parallela con {n_workers} worker")
    
    results_list = []
    processed_count = 0
    first_save = True
    
    # Prepara argomenti per i worker
    card_args = [
        (cid, filtered, user_clusters, pois, anchor_rule, TOP_K) 
        for cid in demo_cards
    ]
    
    # Buffer per batch di risultati
    batch_buffer = []
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Sottometti tutti i job
        future_to_card = {
            executor.submit(process_single_card, args): args[0] 
            for args in card_args
        }
        
        # Processa risultati man mano che completano
        with tqdm(total=len(demo_cards), desc="Card", unit="card") as pbar:
            for future in as_completed(future_to_card):
                cid = future_to_card[future]
                
                try:
                    rec = future.result(timeout=300)  # Timeout per card
                    
                    if rec:
                        batch_buffer.append(rec)
                        processed_count += 1
                        
                        # Salvataggio intermedio thread-safe
                        if len(batch_buffer) >= save_every:
                            with write_lock:
                                logger.info(f"💾 Salvataggio batch: {processed_count}/{len(demo_cards)}")
                                
                                df_batch = pd.DataFrame(batch_buffer)
                                
                                if first_save and not append:
                                    df_batch.to_csv(output_file, mode="w", header=True, index=False)
                                    first_save = False
                                else:
                                    df_batch.to_csv(output_file, mode="a", header=False, index=False)
                                
                                # Aggiorna checkpoint
                                if append:
                                    checkpoint_file = get_checkpoint_file(visits_path, out_dir)
                                    completed_in_batch = [
                                        r['card_id'] for r in batch_buffer 
                                        if r.get('prediction') and 
                                        r['prediction'] not in ['None', '', 'NO_RESPONSE']
                                    ]
                                    update_checkpoint_incremental(checkpoint_file, completed_in_batch)
                                
                                batch_buffer.clear()
                                
                except Exception as e:
                    logger.error(f"Errore futuro per card {cid}: {e}")
                
                pbar.update(1)
    
    # Salva eventuali risultati rimanenti nel buffer
    if batch_buffer:
        results_list.extend(batch_buffer)
            
    # ---------- 7. Salvataggio finale ----------
    if results_list:  # Solo se abbiamo dei risultati
        df_out = pd.DataFrame(results_list)
    
        # Salva il file in base alla modalità
        if append and not write_header:
            # Append ai dati esistenti senza header
            df_out.to_csv(output_file, mode="a", header=False, index=False)
        else:
            # Scrivi normalmente (nuovo file o primo append)
            df_out.to_csv(output_file, index=False)
    
        # Calcola e mostra statistiche
        hit_rate = df_out["hit"].mean()
        logger.info(f"✔  Salvato {output_file.name} – Hit@{TOP_K}: {hit_rate:.2%}")
    else:
        logger.warning("⚠️  Nessun risultato da salvare!")
        
    # Aggiornamento finale del checkpoint
        if append and results_list:
            checkpoint_file = get_checkpoint_file(visits_path, out_dir)
            completed_final = [
                rec['card_id'] for rec in results_list 
                if rec.get('prediction') and 
                rec['prediction'] not in ['None', '', 'NO_RESPONSE'] and
                not str(rec['prediction']).startswith(('ERROR', 'PROCESSING_ERROR'))
            ]
            update_checkpoint_incremental(checkpoint_file, completed_final)

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
    
    # Prima prova: checkpoint file (velocissimo)
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                completed_cards = {line.strip() for line in f if line.strip()}
            logger.info(f"⚡ Checkpoint: {len(completed_cards)} card già completate (lettura veloce)")
            return completed_cards
        except Exception as e:
            logger.warning(f"⚠️ Errore lettura checkpoint: {e}, fallback su CSV")
    
    # Fallback: scansione CSV (lento ma necessario la prima volta)
    latest_csv = latest_output(visits_path, out_dir)
    if latest_csv:
        logger.info("🐌 Prima esecuzione append: scansiono CSV esistente...")
        try:
            # Leggi solo le colonne necessarie per velocizzare
            df = pd.read_csv(latest_csv, usecols=['card_id', 'prediction'])
            completed = df[
                df['prediction'].notna() & 
                (df['prediction'] != '') & 
                (df['prediction'] != 'None') &
                (df['prediction'] != 'NO_RESPONSE') &
                (~df['prediction'].str.startswith('ERROR', na=False)) &
                (~df['prediction'].str.startswith('PROCESSING_ERROR', na=False))
            ]['card_id'].unique()
            
            completed_set = set(completed)
            
            # Salva il checkpoint per le prossime volte
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
OLLAMA_HOSTS = setup_ollama_connections()
host_cycle = itertools.cycle(OLLAMA_HOSTS)

# ---------- MAIN -----------------------------------------------------------
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"  # Usa tutte e 4 le A100

if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(
        description="Calcola raccomandazioni su tutti i log VeronaCard."
    )
    parser.add_argument("--force", action="store_true",
                        help="ricalcola e sovrascrive anche se gli output esistono")
    parser.add_argument("--append", action="store_true",
                        help="riprende da dove si era interrotto (non ricalcola card già presenti)")
    parser.add_argument("--max-users", type=int, default=None,
                        help="numero massimo di utenti da processare per file (default 50)")
    parser.add_argument("--anchor", type=str, default=DEFAULT_ANCHOR_RULE,
                        dest="anchor_rule",
                        help="Regola per scegliere il POI ancora (penultimate|first|middle|int)")
    parser.add_argument("--file", type=str, default=None,
                        help="Processa solo un file specifico (path relativo o assoluto)")
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
                           anchor_rule=args.anchor_rule)
        else:
            run_all_verona_logs(max_users=args.max_users,
                              force=args.force,
                              append=args.append,
                              anchor_rule=args.anchor_rule)
    except KeyboardInterrupt:
        logging.info("Interruzione manuale...")
        sys.exit(1)
