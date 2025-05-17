import json
import random
import time
import pandas as pd
import requests
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

import logging
from datetime import datetime

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
MAX_USERS = None  # None → use *all* users; set an int to sample
TOP_K  = 5          # deve coincidere con top_k del prompt
N_TEST = None        # quanti utenti valutare (None = tutti)
# -----------------------------------------------------------

# ---------- helper di caricamento -----------------------------------------
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
    exclude_last_n: int = 1,
) -> str:
    """
    Genera un prompt (in italiano) per prevedere il prossimo POI.
    - exclude_last_n = 1 → scenario live
    - exclude_last_n = 2 → scenario test offline
    """
    visits = df[df["card_id"] == card_id].sort_values("timestamp")
    seq = visits["name_short"].tolist()

    if len(seq) <= exclude_last_n:
        raise ValueError(f"Sequenza troppo corta per exclude_last_n={exclude_last_n}")

    history = seq[:-exclude_last_n]
    current_poi = seq[-exclude_last_n]
    cluster_id = user_clusters.loc[user_clusters["card_id"] == card_id, "cluster"].values[0]

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
def run_on_visits_file(visits_path: Path, poi_path: Path, *, max_users: int = 50) -> None:
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
    logger.info("\n" + "=" * 70)
    logger.info(f"▶  PROCESSO FILE: {visits_path.name}")
    logger.info("=" * 70)

    logger.info(f"▶  ho caricato e pulito i dati")
    # ---------- 1. load & clean ----------
    pois     = load_pois(poi_path)
    visits   = load_visits(visits_path)
    merged   = merge_visits_pois(visits, pois)
    filtered = filter_multi_visit_cards(merged)

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
        ground = seq[-1]
        prompt = create_prompt_with_cluster(
            filtered, user_clusters, cid,
            top_k=TOP_K, exclude_last_n=2
        )
        ans = get_chat_completion(prompt)

        rec = {
            "card_id":   cid,
            "cluster":   int(user_clusters.loc[user_clusters.card_id == cid, "cluster"].iloc[0]),
            "history":   str(seq[:-2]),
            "current_poi": seq[-2],
            "prediction": None,
            "ground_truth": ground,
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
                rec["hit"]        = ground in pred_lst
            except Exception:
                pass

        df_out.loc[len(df_out)] = rec

    df_out.to_csv(out_file, index=False)
    hit_rate = df_out.hit.mean()
    logger.info(f"✔  Salvato {out_file.name} – Hit@{TOP_K}: {hit_rate:.2%}")

# ---------- test su tutti i file ------------------------------------------
def run_all_verona_logs(max_users: int = 50) -> None:
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
        run_on_visits_file(csv, poi_csv, max_users=max_users)

# ---------- MAIN -----------------------------------------------------------
# ---------------------------------------------------------------------------
def main() -> None:
    """
    Avvia l'elaborazione su tutti i CSV di timbrature presenti in
    data/verona/**, escluso vc_site.csv e eventuali cartelle di backup.
    """
    run_all_verona_logs(max_users=MAX_USERS)

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Interruzione manuale. Uscita…")
