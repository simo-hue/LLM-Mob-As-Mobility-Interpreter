import json
import random
import time
from pathlib import Path
import pandas as pd
import requests
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------
TOP_K  = 5          # deve coincidere con top_k del prompt
N_TEST = 5        # quanti utenti valutare (None = tutti)
# -----------------------------------------------------------


# ---------- helper di caricamento -----------------------------------------
def load_pois(filepath: str | Path) -> DataFrame:
    df = pd.read_csv(filepath, usecols=["name_short", "latitude", "longitude"])
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
    return df[["timestamp", "card_id", "name_short"]].sort_values("timestamp").reset_index(drop=True)

def merge_visits_pois(visits_df: DataFrame, pois_df: DataFrame) -> DataFrame:
    merged = visits_df.merge(pois_df[["name_short"]], on="name_short", how="inner")
    return merged.sort_values("timestamp").reset_index(drop=True)

def filter_multi_visit_cards(df: DataFrame) -> DataFrame:
    valid_cards = df.groupby("card_id")["name_short"].nunique().loc[lambda s: s > 1].index
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
            print("⚠️  Ollama non è in esecuzione.")
            return None
    except requests.exceptions.RequestException as exc:
        print(f"❌  Connessione Ollama fallita: {exc}")
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
        print(f"❌  Errore HTTP: {exc}")
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

# ---------- MAIN -----------------------------------------------------------
def main() -> None:
    # -------------------------------------------------------------------- #
    # 1) Path ai CSV                                                       #
    # -------------------------------------------------------------------- #
    ROOT = Path(__file__).resolve().parent
    visits_file = ROOT / "data" / "verona" / "dataset_veronacard_2014_2020" / "dati_2014.csv"
    poi_file    = ROOT / "data" / "verona" / "vc_site.csv"

    # -------------------------------------------------------------------- #
    # 2) Load & clean                                                      #
    # -------------------------------------------------------------------- #
    pois     = load_pois(poi_file)
    visits   = load_visits(visits_file)
    merged   = merge_visits_pois(visits, pois)
    filtered = filter_multi_visit_cards(merged)

    print(f"\nVisite totali (merge): {len(merged):,}")
    print(f"Card multi-visita    : {filtered['card_id'].nunique():,}")
    print(f"Record finali        : {len(filtered):,}\n")

    # -------------------------------------------------------------------- #
    # 3) Clustering K-Means                                                #
    # -------------------------------------------------------------------- #
    matrix   = create_user_poi_matrix(filtered)
    scaler   = StandardScaler()
    kmeans   = KMeans(n_clusters=7, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaler.fit_transform(matrix))

    user_clusters = pd.DataFrame({"card_id": matrix.index, "cluster": clusters})
    print(user_clusters["cluster"].value_counts(), "\n")

    # -------------------------------------------------------------------- #
    # 4) Selezione utenti idonei (≥3 visite)                               #
    # -------------------------------------------------------------------- #
    eligible_cards = (
        filtered.groupby("card_id")
        .size()
        .loc[lambda s: s >= 3]       # 3 = exclude_last_n (2) + 1
        .index
        .tolist()
    )
    if not eligible_cards:
        raise RuntimeError("Nessun utente con almeno 3 visite nel dataset!")

    # facoltativo: limita il numero di utenti da processare
    MAX_USERS = 20
    demo_cards = random.sample(eligible_cards, k=min(MAX_USERS, len(eligible_cards)))

    # -------------------------------------------------------------------- #
    # 5) Loop di predizione & confronto                                    #
    # -------------------------------------------------------------------- #
    hits = 0
    for idx, card_id in enumerate(demo_cards, 1):
        seq = (
            filtered[filtered["card_id"] == card_id]
            .sort_values("timestamp")["name_short"]
            .tolist()
        )
        true_next = seq[-1]                      # ground-truth
        prompt = create_prompt_with_cluster(
            filtered, user_clusters, card_id,
            top_k=TOP_K,
            exclude_last_n=2                    # penultimo = current_poi
        )
        answer = get_chat_completion(prompt)
        if not answer:
            print(f"[{idx:02}] {card_id} – nessuna risposta dal modello")
            continue

        try:
            pred_obj = json.loads(answer)
            pred     = pred_obj["prediction"]
            pred_lst = pred if isinstance(pred, list) else [pred]
        except Exception:
            print(f"[{idx:02}] {card_id} – parsing JSON fallito")
            continue

        hit = true_next in pred_lst
        hits += hit

        print(f"[{idx:02}] card_id={card_id}")
        print(f"     history … {seq[:-2]} → current={seq[-2]}")
        print(f"     ✅ vero next : {true_next}")
        print(f"     🔮 predizione: {pred_lst}")
        print(f"     🎯 hit@{TOP_K}: {'YES' if hit else 'NO'}\n")

    print("-" * 60)
    print(f"Hit@{TOP_K} complessivo: {hits}/{len(demo_cards)}  "
          f"({hits/len(demo_cards):.2%})")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterruzione manuale. Uscita…")
