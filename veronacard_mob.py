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
    card_id: str,
    *,
    top_k: int = 1,
    anchor_rule: str | int = DEFAULT_ANCHOR_RULE,
) -> str:
    """
    Genera un prompt (in italiano) per prevedere il prossimo POI
    usando come 'ancora' la posizione scelta da *anchor_rule*.

    La sequenza deve avere almeno 3 visite:
    [history ... | current_poi | target]
    """
    visits = df[df["card_id"] == card_id].sort_values("timestamp")
    seq = visits["name_short"].tolist()

    if len(seq) < 3:
        raise ValueError("Sequenza troppo corta (minimo 3 tappe)")

    target      = seq[-1]      # non usato nel prompt
    prefix      = seq[:-1]     # tutto tranne il target
    idx         = anchor_index(len(prefix), anchor_rule)
    current_poi = prefix[idx]
    history     = [p for i, p in enumerate(prefix) if i != idx]

    cluster_id = user_clusters.loc[
        user_clusters["card_id"] == card_id, "cluster"
    ].values[0]

    return f"""
        Sei un assistente turistico esperto di Verona.
        <cluster_id>: {cluster_id}
        <history>: {history}
        <current_poi>: {current_poi}

        Obiettivo: suggerisci i {top_k} POI più probabili che l'utente visiterà dopo.
        • Escludi i POI già in <history> e <current_poi>.
        • Rispondi con **una sola riga** JSON:
        {{ "prediction": [...], "reason": "..." }}
        Rispondi in italiano.
        """.strip()

# ---------- chiamata LLaMA / Ollama ---------------------------------------
def get_chat_completion(prompt: str, model: str = "llama3:latest") -> str | None:
    base_url = "http://localhost:11434"
    try:
        if requests.get(f"{base_url}/api/tags", timeout=2).status_code != 200:
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
        resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=60)
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

    # ---------- 4. CSV di output ----------
    out_dir  = Path(__file__).resolve().parent / "results"
    out_dir.mkdir(exist_ok=True)
    ts       = time.strftime("%Y%m%d_%H%M%S")
    out_file = out_dir / f"{visits_path.stem}_pred_{ts}.csv"

    df_out = pd.DataFrame(columns=[
        "card_id","cluster","history","current_poi",
        "prediction","ground_truth","reason","hit"
    ])

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
            filtered, user_clusters, cid,
            top_k=TOP_K, anchor_rule=anchor_rule
        )
        ans = get_chat_completion(prompt)

        rec = {
            "card_id":   cid,
            "cluster":   int(user_clusters.loc[user_clusters.card_id == cid, "cluster"].iloc[0]),
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

        df_out.loc[len(df_out)] = rec

        if append and latest_output(visits_path, out_dir):
            # Append senza header
            df_out.to_csv(prev_path, mode="a", header=False, index=False)
        else:
            df_out.to_csv(out_file, index=False)

        hit_rate = df_out.hit.mean()
        logger.info(f"✔  Salvato {out_file.name} – Hit@{TOP_K}: {hit_rate:.2%}")

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
