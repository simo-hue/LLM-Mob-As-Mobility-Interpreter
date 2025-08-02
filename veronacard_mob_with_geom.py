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

# -----------------------------------------------------------
TOP_K  = 5          # deve coincidere con top_k del prompt
# -----------------------------------------------------------
# -----------------------------------------------------------
DEFAULT_ANCHOR_RULE = "penultimate"   # "penultimate" | "first" | "middle" | int
# -----------------------------------------------------------

# FUNZIONE HELPER per test rapido di Ollama
def test_ollama_connection(host: str, model: str = "llama3.1:8b") -> bool:
    """
    Test rapido di connettività e funzionalità Ollama
    """
    try:
        # Test 1: Health check
        resp = requests.get(f"{host}/api/tags", timeout=10)
        if resp.status_code != 200:
            logger.error(f"❌ Health check failed: {resp.status_code}")
            return False
            
        # Test 2: Verifica modello
        models = [m.get('name', '') for m in resp.json().get('models', [])]
        if model not in models:
            logger.error(f"❌ Model '{model}' not found. Available: {models}")
            return False
            
        # Test 3: Micro inference
        test_resp = requests.post(
            f"{host}/api/generate",
            json={
                "model": model,
                "prompt": "Hi",
                "stream": False,
                "options": {"num_predict": 1, "temperature": 0}
            },
            timeout=30
        )
        
        if test_resp.status_code == 200:
            data = test_resp.json()
            if data.get("done") and data.get("response"):
                logger.info(f"✅ Ollama test passed - Response: '{data.get('response')}'")
                return True
                
        logger.error(f"❌ Micro inference failed")
        return False
        
    except Exception as e:
        logger.error(f"❌ Ollama connection test failed: {e}")
        return False

def get_chat_completion(prompt: str, model: str = "llama3.1:8b", max_retries: int = 3) -> Optional[str]:
    """
    Versione ottimizzata per HPC Leonardo con gestione timeout GPU avanzata
    """
    
    for attempt in range(1, max_retries + 1):
        try:
            # 1. HEALTH CHECK MIGLIORATO con info sul modello
            logger.debug(f"🔍 Health check (tentativo {attempt}/{max_retries})")
            
            health_resp = requests.get(
                f"{OLLAMA_HOST}/api/tags", 
                timeout=10,  # Aumentato da 5 a 10s
                headers={'Connection': 'close'}
            )
            
            if health_resp.status_code != 200:
                logger.warning(f"⚠️  Health check HTTP {health_resp.status_code}")
                if attempt < max_retries:
                    time.sleep(5 + attempt * 5)  # Backoff progressivo
                    continue
                else:
                    return None
            
            # Verifica che il modello sia disponibile
            try:
                models_data = health_resp.json()
                available_models = [m.get('name', '') for m in models_data.get('models', [])]
                if model not in available_models:
                    logger.error(f"❌ Modello '{model}' non trovato. Disponibili: {available_models}")
                    return None
                else:
                    logger.debug(f"✓ Modello '{model}' disponibile")
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning(f"⚠️  Errore parsing models: {e}")

        except requests.exceptions.RequestException as exc:
            logger.error(f"❌ Health check errore: {exc}")
            if attempt < max_retries:
                time.sleep(10 + attempt * 5)
                continue
            else:
                return None

        # 2. PREPARAZIONE PAYLOAD OTTIMIZZATA
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,  # Non streaming per stabilità
            "options": {
                # Parametri base ottimizzati per A100
                "temperature": 0.1,
                "top_p": 0.9,
                "top_k": 40,
                
                # CORREZIONI CRITICHE per timeout GPU
                "num_ctx": 1024,        # RIDOTTO da 2048 - meno memoria GPU
                "num_predict": 100,     # RIDOTTO da 150 - risposte più brevi
                "num_thread": 4,        # RIDOTTO da 8 - meno carico CPU
                "num_batch": 64,        # RIDOTTO da 128 - batch più piccoli
                
                # Parametri GPU specifici per A100
                "num_gpu": 33,          # Tutti i layer su GPU
                "num_gqa": 8,           # Group Query Attention ottimizzato
                "num_head": 32,         # Attention heads
                "num_head_kv": 8,       # Key-Value heads
                
                # Ottimizzazioni memoria
                "low_vram": False,      # Usa VRAM completa su A100
                "f16_kv": True,         # FP16 per cache K-V
                "use_mmap": True,       # Usa memory mapping
                "use_mlock": False,     # Non bloccare memoria
                
                # NUOVO: Parametri anti-hang
                "tfs_z": 1.0,          # Tail free sampling
                "typical_p": 1.0,       # Typical sampling
                "repeat_penalty": 1.1,
                "repeat_last_n": 64,
                "penalize_nl": True,
                
                # Stop tokens per evitare generazioni infinite
                "stop": ["\n\n", "```", "---", "===", "<|", "|>", "[END]"]
            }
        }
        
        try:
            logger.debug(f"🔄 Richiesta Ollama (tentativo {attempt}/{max_retries})")
            logger.debug(f"📝 Prompt length: {len(prompt)} chars")
            
            # 3. TIMEOUT DINAMICO basato su tentativo e lunghezza prompt
            base_timeout = 60  # Base: 1 minuto
            prompt_factor = min(len(prompt) // 100, 5)  # +1s ogni 100 caratteri, max +5s
            attempt_factor = (4 - attempt) * 30  # Primo tentativo più lungo
            
            timeout = base_timeout + prompt_factor + attempt_factor
            timeout = max(60, min(timeout, 300))  # Tra 1 e 5 minuti
            
            logger.debug(f"⏰ Timeout calcolato: {timeout}s")
            
            # 4. RICHIESTA CON GESTIONE ERRORI AVANZATA
            start_time = time.time()
            
            resp = requests.post(
                f"{OLLAMA_HOST}/api/generate",
                json=payload,
                timeout=timeout,
                headers={
                    'Content-Type': 'application/json',
                    'Connection': 'close',
                    'User-Agent': 'LLM-Mob-HPC/1.0'
                },
                # Disabilita retry automatico di requests
                allow_redirects=False
            )
            
            elapsed = time.time() - start_time
            logger.debug(f"📈 Risposta in {elapsed:.1f}s - Status: {resp.status_code}")
            
            # 5. VALIDAZIONE RISPOSTA HTTP
            if resp.status_code != 200:
                logger.error(f"❌ HTTP {resp.status_code}")
                if resp.status_code == 500:
                    logger.error(f"❌ Server error: {resp.text[:300]}")
                elif resp.status_code == 404:
                    logger.error(f"❌ Endpoint non trovato - verificare OLLAMA_HOST")
                    return None  # Errore definitivo
                continue
                
            # Verifica contenuto non vuoto
            if not resp.content:
                logger.warning(f"⚠️  Risposta HTTP vuota")
                continue
                
            # 6. PARSING JSON ROBUSTO
            try:
                response_data = resp.json()
            except json.JSONDecodeError as e:
                logger.error(f"❌ JSON malformato: {e}")
                logger.error(f"❌ Raw response (primi 500 char): {resp.text[:500]}")
                continue
            
            # Debug dettagliato struttura
            logger.debug(f"🔍 Response keys: {list(response_data.keys())}")
            if 'error' in response_data:
                logger.error(f"❌ Ollama error: {response_data['error']}")
                continue
            
            # 7. VERIFICA COMPLETAMENTO E ESTRAZIONE CONTENUTO
            done = response_data.get("done", False)
            response_content = response_data.get("response", "").strip()
            done_reason = response_data.get("done_reason", "unknown")
            
            logger.debug(f"🏁 Done: {done}, Reason: {done_reason}, Content length: {len(response_content)}")
            
            # Gestione stati specifici
            if done_reason == "load":
                logger.warning(f"⚠️  Modello in caricamento, attendo 20s...")
                time.sleep(20)
                continue
            elif done_reason == "stop":
                logger.debug(f"✓ Generazione fermata da stop token")
            elif done_reason == "length":
                logger.debug(f"✓ Generazione fermata per lunghezza massima")
            elif not done and not response_content:
                logger.warning(f"⚠️  Risposta incompleta senza contenuto")
                continue
                
            # Accetta anche risposte parziali se hanno contenuto utile
            if response_content and len(response_content) > 10:  # Almeno 10 caratteri
                logger.debug(f"✅ Risposta valida ricevuta")
                logger.debug(f"📄 Preview: {response_content[:150]}...")
                
                # Statistiche finali
                total_tokens = response_data.get("eval_count", 0)
                prompt_tokens = response_data.get("prompt_eval_count", 0)
                eval_duration = response_data.get("eval_duration", 0) / 1e9  # ns to seconds
                
                if total_tokens > 0 and eval_duration > 0:
                    tokens_per_sec = total_tokens / eval_duration
                    logger.debug(f"📊 {total_tokens} tokens in {eval_duration:.1f}s ({tokens_per_sec:.1f} tok/s)")
                
                return response_content
            else:
                logger.warning(f"⚠️  Contenuto insufficiente: '{response_content[:50]}'")
                
        except requests.exceptions.Timeout:
            logger.error(f"❌ Timeout dopo {timeout}s (tentativo {attempt}/{max_retries})")
            
            # Timeout al primo tentativo = problema serio
            if attempt == 1:
                logger.error(f"🚨 TIMEOUT CRITICO - Possibili cause:")
                logger.error(f"   • GPU sovraccarica o memoria insufficiente")
                logger.error(f"   • Modello non completamente caricato")
                logger.error(f"   • Configurazione CUDA problematica")
                
        except requests.exceptions.ConnectionError as exc:
            logger.error(f"❌ Connessione fallita: {exc}")
            logger.error(f"💡 Verificare che Ollama sia attivo su {OLLAMA_HOST}")
            
        except requests.exceptions.RequestException as exc:
            logger.error(f"❌ Errore richiesta: {exc}")
            
        except Exception as exc:
            logger.error(f"❌ Errore inaspettato: {exc}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
        
        # 8. BACKOFF INTELLIGENTE tra tentativi
        if attempt < max_retries:
            # Backoff progressivo: 10s, 20s, 40s
            wait_time = min(10 * (2 ** (attempt - 1)), 60)
            logger.info(f"⏳ Attesa {wait_time}s prima del prossimo tentativo...")
            time.sleep(wait_time)
    
    logger.error(f"❌ FALLIMENTO DEFINITIVO dopo {max_retries} tentativi")
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

def warmup_model(model: str = "llama3.1:8b") -> bool:
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

def setup_ollama_connection():
    """Setup della connessione Ollama con retry robusto"""
    try:
        with open(OLLAMA_PORT_FILE, "r") as f:
            port = f.read().strip()
        print(f"👉 Porta letta da ollama_port.txt: '{port}'")
        
        ollama_host = f"http://127.0.0.1:{port}"
        print(f"👉 Provo a contattare {ollama_host}/api/tags")
        
        # PRINT DI DEBUG
        print(f"📂 Working dir: {os.getcwd()}")
        print(f"📄 Contenuto di ollama_port.txt: '{port}'")
        
        return ollama_host, port
    except FileNotFoundError:
        raise RuntimeError(f"❌ File {OLLAMA_PORT_FILE} non trovato. Il job SLURM deve generarlo.")

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

# Setup della connessione
OLLAMA_HOST, OLLAMA_PORT = setup_ollama_connection()

# --------------------------------------------------------------

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

# --------------------------------------------------------------
def list_outputs(visits_path: Path, out_dir: Path) -> list[Path]:
    """Tutti i CSV già calcolati per quel file di visite."""
    pattern = f"{visits_path.stem}_pred_*.csv"
    return sorted(out_dir.glob(pattern))

def latest_output(visits_path: Path, out_dir: Path) -> Path | None:
    """L'output più recente (None se non esiste)."""
    outputs = list_outputs(visits_path, out_dir)
    return max(outputs, key=os.path.getmtime) if outputs else None
# --------------------------------------------------------------

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
Disponibili: {pois_list}

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

# ---------- chiamata LLaMA / Ollama ---------------------------------------
def get_chat_completion(prompt: str, model: str = "llama3.1:8b", max_retries: int = 2) -> str | None:
    """
    Ottiene una completion dal modello LLaMA tramite Ollama con timeout ottimizzati.
    """
    
    for attempt in range(1, max_retries + 1):
        try:
            # Health check veloce
            health_check = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=3)
            if health_check.status_code != 200:
                logger.warning(f"⚠️  Ollama non risponde (tentativo {attempt}/{max_retries})")
                if attempt < max_retries:
                    time.sleep(5)
                    continue
                else:
                    return None

        except requests.exceptions.RequestException as exc:
            logger.error(f"❌  Health check fallito (tentativo {attempt}/{max_retries}): {exc}")
            if attempt < max_retries:
                time.sleep(5)
                continue
            else:
                return None

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}], 
            "stream": False
        }
        
        try:
            logger.debug(f"🔄 Richiesta Ollama (tentativo {attempt}/{max_retries})")
            
            # Timeout progressivo: primo tentativo più lungo per warm-up
            timeout = 300 if attempt == 1 else 180
            
            resp = requests.post(
                f"{OLLAMA_HOST}/api/chat", 
                json=payload, 
                timeout=timeout,
                headers={'Content-Type': 'application/json'}
            )
            
            resp.raise_for_status()
            response_data = resp.json()
            
            # Verifica che la risposta sia completa
            if not response_data.get("done", False):
                logger.warning(f"⚠️  Risposta incompleta (tentativo {attempt}/{max_retries})")
                continue
            
            message = response_data.get("message", {})
            content = message.get("content", "")
            if content:
                logger.debug(f"✓ Risposta ricevuta (lunghezza: {len(content)} caratteri)")
                return content
            else:
                logger.warning(f"⚠️  Risposta vuota (tentativo {attempt}/{max_retries})")
                
        except requests.exceptions.Timeout:
            logger.error(f"❌  Timeout {timeout}s (tentativo {attempt}/{max_retries})")
        except requests.exceptions.HTTPError as exc:
            logger.error(f"❌  HTTP {resp.status_code}: {exc} (tentativo {attempt}/{max_retries})")
            try:
                error_detail = resp.json()
                logger.error(f"❌  Dettaglio: {error_detail}")
            except:
                logger.error(f"❌  Risposta raw: {resp.text[:500]}")
        except requests.exceptions.RequestException as exc:
            logger.error(f"❌  Errore richiesta: {exc} (tentativo {attempt}/{max_retries})")
        except Exception as exc:
            logger.error(f"❌  Errore inaspettato: {exc} (tentativo {attempt}/{max_retries})")
        
        # Backoff progressivo tra tentativi
        if attempt < max_retries:
            wait_time = min(attempt * 10, 30)  # Max 30s di attesa
            logger.info(f"⏳ Attendo {wait_time}s prima del prossimo tentativo...")
            time.sleep(wait_time)
    
    logger.error(f"❌  Tutti i {max_retries} tentativi falliti")
    return None

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

# ---------- test su un singolo file ---------------------------------------
def run_on_visits_file(visits_path: Path, poi_path: Path, *, max_users: int | None = None, force: bool = False, append: bool = False, anchor_rule: str | int = DEFAULT_ANCHOR_RULE) -> None:
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
    
    # ---------- 0. check risultati già esistenti ----------
    out_dir = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)

    # ---------- Modalità force / append / skip ----------
    if force and append:
        raise ValueError("Non puoi usare --force e --append insieme.")

    if append:
        prev_path = latest_output(visits_path, out_dir)
        if prev_path:
            prev_df = pd.read_csv(prev_path, usecols=['card_id'])
            processed_cards = set(prev_df['card_id'])
        else:
            processed_cards = set()
    else:
        processed_cards = set()


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
    
    logger.info("▶  Analisi pattern di movimento geografico")
    movement_patterns = analyze_movement_patterns(filtered, pois)
    avg_distance = movement_patterns["distance"].mean()
    logger.info(f"▶  Distanza media tra visite consecutive: {avg_distance:.2f} km")

    results_list = []
    
    # ---------- 5. ciclo su utenti ----------
    for cid in tqdm(demo_cards, desc="Card", unit="card"):
        seq = (
            filtered.loc[filtered.card_id == cid]
            .sort_values("timestamp")["name_short"].tolist()
        )
        target       = seq[-1]
        idx_anchor   = anchor_index(len(seq) - 1, anchor_rule)
        history_list = [p for i, p in enumerate(seq[:-1]) if i != idx_anchor]
        current_poi  = seq[:-1][idx_anchor]

        prompt = create_prompt_with_cluster(
            filtered, user_clusters, pois, cid,  
            top_k=TOP_K, anchor_rule=anchor_rule
        )
        ans = get_chat_completion(prompt)

        # ---------- 6. Analisi risultato ----------
        rec = {
            "card_id":   cid,
            "cluster":   get_user_cluster(user_clusters, cid),
            "history":   str(history_list),
            "current_poi": current_poi,
            "prediction": None,
            "ground_truth": target,
            "reason":    None,
            "hit":       False
        }

        if ans:
            try:
                obj  = json.loads(ans)
                pred = obj["prediction"]
                pred_lst = pred if isinstance(pred, list) else [pred]
                rec["prediction"] = str(pred_lst)
                rec["reason"]     = obj.get("reason")
                rec["hit"]        = target in pred_lst
            except Exception:
                pass

        results_list.append(rec)

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

# ---------- test su tutti i file ------------------------------------------
def run_all_verona_logs(max_users: int | None = None, force=False, append=False, anchor_rule: str | int = DEFAULT_ANCHOR_RULE) -> None:
    ROOT   = Path(__file__).resolve().parent
    poi_csv = ROOT / "data" / "verona" / "vc_site.csv"

    # trova tutti i CSV eccetto vc_site
    visit_csvs = [
        p for p in (ROOT / "data" / "verona").rglob("*.csv")
        if p.name != "vc_site.csv"
    ]
    if not visit_csvs:
        raise RuntimeError("Nessun CSV di visite trovato sotto data/verona/")

    for csv in sorted(visit_csvs):
        run_on_visits_file(csv, poi_csv,
                           max_users=max_users,
                           force=force,
                           append=append,
                           anchor_rule=anchor_rule)

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

# ---------- MAIN -----------------------------------------------------------

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
    args = parser.parse_args()

    if args.force and args.append:
        parser.error("Non puoi usare insieme --force e --append.")

    # Attendi che Ollama sia pronto
    if not wait_for_ollama(OLLAMA_HOST):
        raise RuntimeError("❌ Ollama non ha risposto dopo tutti i tentativi")

    print("🎉 Connessione Ollama stabilita con successo!")

    # Warm-up del modello prima di iniziare
    if not warmup_model():
        logger.warning("⚠️  Warm-up fallito, ma continuo comunque...")
    
    debug_gpu_status()
    
    # Warm-up con retry
    warmup_success = False
    for i in range(3):
        if warmup_model():
            warmup_success = True
            break
        else:
            logger.warning(f"⚠️  Warm-up tentativo {i+1}/3 fallito")
            if i < 2:
                time.sleep(20) 
    
    if not warmup_success:
        logger.error("❌  Warm-up completamente fallito - possibili problemi GPU gravi")
       
    # Controllo che OLLAMA risponda altrimenti esco 
    if not test_ollama_connection(OLLAMA_HOST, "llama3.1:8b"):
        logger.error("❌ Ollama non funziona, aborting")
        exit(1)

    try:
        run_all_verona_logs(max_users=args.max_users,
                            force=args.force,
                            append=args.append,
                            anchor_rule=args.anchor_rule)
    except KeyboardInterrupt:
        logging.info("Interruzione manuale...")
        sys.exit(1)
