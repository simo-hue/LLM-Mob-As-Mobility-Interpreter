import json
import random
import time
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

# --- CONFIGURAZIONE OLLAMA: leggi porta dal file ---
OLLAMA_PORT_FILE = "ollama_port.txt"
try:
    with open(OLLAMA_PORT_FILE, "r") as f:
        port = f.read().strip()
    print(f"👉 Porta letta da ollama_port.txt: '{port}'")
    print(f"👉 Provo a contattare http://127.0.0.1:{port}/api/tags")
    OLLAMA_HOST = f"http://127.0.0.1:{port}"
        
    # PRINT DI DEBUG
    print(f"📂 Working dir: {os.getcwd()}")
    print(f"📄 Contenuto di ollama_port.txt: '{port}'")
except FileNotFoundError:
    raise RuntimeError(f"❌ File {OLLAMA_PORT_FILE} non trovato. Il job SLURM deve generarlo.")

# --- Attendi che il runner sia attivo ---
for _ in range(10):
    try:
        r = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        if r.status_code == 200:
            print("✓ Runner LLaMA attivo")
            break
    except requests.exceptions.RequestException:
        print("⏳ Attendo LLaMA...")
        time.sleep(3)
else:
    raise RuntimeError("❌ LLaMA non ha risposto dopo 30 secondi")

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
    top_k: int = 1,
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
        # Fallback se non troviamo le coordinate
        return create_basic_prompt(...)  # versione senza coordinate
    
    current_lat = current_poi_row["latitude"].iloc[0]
    current_lon = current_poi_row["longitude"].iloc[0]

    # Calcola distanze da tutti gli altri POI disponibili
    available_pois_with_distance = []
    for _, row in pois_df.iterrows():
        poi_name = row["name_short"]
        
        # Salta se è già stato visitato o è il POI attuale
        if poi_name in history or poi_name == current_poi:
            continue
            
        # Calcola distanza dal POI attuale
        distance = calculate_distance(
            current_lat, current_lon,
            row["latitude"], row["longitude"]
        )
        
        available_pois_with_distance.append({
            "name": poi_name,
            "distance": distance,
            "lat": row["latitude"],
            "lon": row["longitude"]
        })

    # Ordina per distanza (i più vicini prima)
    available_pois_with_distance.sort(key=lambda x: x["distance"])

    # Crea la lista formattata per il prompt
    pois_with_distance_text = "\n".join([
        f"- {poi['name']} (distanza: {poi['distance']:.2f} km)"
        for poi in available_pois_with_distance
    ])

    return f"""
        Sei un assistente turistico esperto di Verona con conoscenza dettagliata della geografia della città.
        
        <cluster_id>: {cluster_id}
        <history>: {history}
        <current_poi>: {current_poi}
        
        <pois_disponibili_con_distanze>:
        {pois_with_distance_text}
        
        Obiettivo: suggerisci i {top_k} POI più probabili che l'utente visiterà dopo, considerando:
        - La distanza dal POI attuale (i turisti tendono a visitare luoghi vicini)
        - La logica dei percorsi turistici a Verona
        - I pattern tipici di movimento in base al cluster {cluster_id}
        
        • Escludi i POI già in <history> e <current_poi>.
        • Considera che distanze minori indicano maggiore probabilità di visita.
        • Rispondi con **una sola riga** JSON:
        {{ "prediction": [...], "reason": "..." }}
        
        Nella tua ragione, menziona le distanze e perché questi POI sono logici geograficamente.
        Rispondi in italiano.
        """.strip() 
    
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
def get_chat_completion(prompt: str, model: str = "llama3.1:8b") -> str | None:
    try:
        if requests.get(f"{OLLAMA_HOST}/api/tags", timeout=2).status_code != 200:
            logger.warning("⚠️  Ollama non è in esecuzione.")
            return None
    except requests.exceptions.RequestException as exc:
        logger.error(f"❌  Connessione Ollama fallita: {exc}")
        return None

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"raw": True},
    }
    try:
        resp = requests.post(f"{OLLAMA_HOST}/api/chat", json=payload, timeout=60)
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content")
    except requests.exceptions.RequestException as exc:
        logger.error(f"❌  Errore HTTP: {exc}")
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

    try:
        run_all_verona_logs(max_users=args.max_users,
                            force=args.force,
                            append=args.append,
                            anchor_rule=args.anchor_rule)
    except KeyboardInterrupt:
        logging.info("Interruzione manuale...")
        sys.exit(1)
