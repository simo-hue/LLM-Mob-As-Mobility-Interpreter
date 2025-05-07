import json
import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm         # barra di avanzamento
import random

def load_pois(filepath: str) -> DataFrame:
    """
    Carica il file dei POI (vc_sites) e restituisce un DataFrame con colonne name_short, latitude, longitude.

    Args:
        filepath (str): percorso relativo del CSV dei POI

    Returns:
        DataFrame: POI con colonne ['name_short', 'latitude', 'longitude']
    """
    df_pois = pd.read_csv(filepath)
    df_pois = df_pois[['name_short', 'latitude', 'longitude']]
    return df_pois

def load_visits(filepath: str) -> DataFrame:
    """
    Carica il file delle visite VeronaCard, costruisce un timestamp
    e restituisce un DataFrame con ['timestamp', 'card_id', 'name_short'] ordinato.

    Args:
        filepath (str): percorso del CSV visite, con colonne ['data','ora','name_short','card_id']

    Returns:
        DataFrame: visite con colonne ['timestamp', 'card_id', 'name_short'], ordinate per timestamp
    """
    df = pd.read_csv(
        filepath,
        usecols=[0, 1, 2, 4],
        names=['data', 'ora', 'name_short', 'card_id'],
        header=0,
        dtype={'card_id': str}
    )
    df['timestamp'] = pd.to_datetime(
        df['data'] + ' ' + df['ora'],
        format='%d-%m-%y %H:%M:%S'
    )
    df = (
        df[['timestamp', 'card_id', 'name_short']]
        .sort_values('timestamp')
        .reset_index(drop=True)
    )
    return df

def merge_visits_pois(
    visits_df: DataFrame,
    pois_df: DataFrame
) -> DataFrame:
    """
    Unisce il DataFrame delle visite con quello dei POI su 'name_short',
    restituendo timestamp, card_id e name_short.

    Args:
        visits_df (DataFrame): visite con ['timestamp','card_id','name_short']
        pois_df (DataFrame): POI con ['name_short']

    Returns:
        DataFrame: colonne ['timestamp','card_id','name_short'], ordinate per timestamp
    """
    pois_sel = pois_df[['name_short']]
    merged = pd.merge(
        visits_df,
        pois_sel,
        on='name_short',
        how='inner'
    )
    merged = (
        merged[['timestamp', 'card_id', 'name_short']]
        .sort_values('timestamp')
        .reset_index(drop=True)
    )
    return merged

def filter_multi_visit_cards(df: DataFrame) -> DataFrame:
    """
    Filtra le visite mantenendo solo le card_id che hanno visitato
    più di un POI distinto.

    Args:
        df (DataFrame): DataFrame con ['timestamp','card_id','name_short']

    Returns:
        DataFrame: sottoinsieme di df con visite di card_id multi-visita
    """
    poi_counts = df.groupby('card_id')['name_short'].nunique()
    valid_cards = poi_counts[poi_counts > 1].index
    return df[df['card_id'].isin(valid_cards)].reset_index(drop=True)

def create_user_poi_matrix(df):
    user_poi_matrix = pd.crosstab(df['card_id'], df['name_short'])
    return user_poi_matrix

def create_prompt_with_cluster(df, user_clusters, card_id,top_k: int = 1) -> str:
    """
    Costruisce un prompt per LLaMA/Ollama che chieda la previsione
    del prossimo POI, con risposta in JSON.

    Parameters
    ----------
    df : DataFrame
        Tutte le visite (timestamp, card_id, name_short).
    user_clusters : DataFrame
        Tabella con colonne ['card_id', 'cluster'].
    card_id : str
        ID della card da predire.
    top_k : int, default 1
        Quante raccomandazioni chiedere (1 per singola, 3–5–10 per lista).
    language : {'it','en'}, default 'it'
        Lingua della risposta desiderata.

    Returns
    -------
    str
        Prompt pronto da passare a get_chat_completion().
    """
    # --- recupera cronologia e cluster ---
    visits = df[df['card_id'] == card_id].sort_values('timestamp')
    history = visits['name_short'].tolist()[:-1]          # senza l'ultimo
    last_poi = visits['name_short'].tolist()[-1]          # punto di partenza
    cluster_id = user_clusters.loc[
        user_clusters['card_id'] == card_id, 'cluster'
    ].values[0]

    # --- costruzione prompt ---
    template = f"""
    Sei un assistente turistico che conosce la città di Verona.
    <cluster_id>: {cluster_id}
    <history>: {history}
    <current_poi>: {last_poi}

    Obiettivo: suggerisci i {{
        1: "più probabile",
        2: "2 POI più probabili",
        # qualsiasi top_k > 1
    }}[top_k] che l'utente visiterà dopo.

    • Escludi i POI già in <history> e <current_poi>.
    • Restituisci **una sola riga** JSON:
    {{ "prediction": [...], "reason": "..." }}

    Se top_k = 1, "prediction" sarà una stringa; altrimenti
    una lista ordinata. La chiave "reason" deve essere breve (≤ 25 parole).
    Rispondi in italiano.
    """.strip()

    return template

def plot_number_of_cluster(user_poi_matrix_scaled, max):
    sse = []
    k_range = range(1, max)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(user_poi_matrix_scaled)
        sse.append(km.inertia_)  # inertia è la somma degli errori quadratici

    # grafico Elbow Method
    plt.plot(k_range, sse, marker='o')
    plt.xlabel('Numero di cluster (k)')
    plt.ylabel('SSE (Somma errori quadratici)')
    plt.title('Metodo del gomito (Elbow method)')
    plt.grid(True)
    plt.show()

def get_chat_completion(prompt: str, model: str = "llama3:latest") -> str | None:
    """
    Interroga un modello LLaMA esposto dal server *Ollama* e restituisce la
    risposta testuale.

    Parameters
    ----------
    prompt : str
        Testo da inviare al modello.
    model : str, default "llama3:latest"
        Nome o tag del modello registrato in Ollama.

    Returns
    -------
    str | None
        Contenuto generato dal modello, oppure `None` se si verifica un errore.
    """
    base_url = "http://localhost:11434"
    api_tags  = f"{base_url}/api/tags"   # endpoint “health check”
    api_chat  = f"{base_url}/api/chat"   # endpoint di chat

    # 1) Verifica che Ollama sia in esecuzione
    try:
        if requests.get(api_tags, timeout=2).status_code != 200:
            print("⚠️  Server Ollama non raggiungibile. Avvialo con `ollama serve`.")
            return None
    except requests.exceptions.RequestException as exc:
        print(f"❌  Errore di connessione a Ollama: {exc}")
        return None

    # 2) Prepara il payload
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"raw": True}
    }

    # 3) Invia la richiesta
    try:
        resp = requests.post(api_chat, json=payload, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        # 4) Estrai e restituisci il contenuto
        return data.get("message", {}).get("content")
    except requests.exceptions.HTTPError as http_err:
        print(f"❌  HTTP error: {http_err} – payload: {resp.text}")
    except requests.exceptions.RequestException as exc:
        print(f"❌  Errore HTTP: {exc}")
    return None

def main():
    # 1) Caricamento e pulizia dati -----------------------------------------
    visits_file = "data/verona/dataset_veronacard_2014_2020/dati_2014.csv"
    poi_file    = "data/verona/vc_sites.csv"
    
    pois    = load_pois(poi_file)
    visits  = load_visits(visits_file)

    merged_visits  = merge_visits_pois(visits, pois)
    print(f"\nTotale righe dopo merge_visits_pois: {len(merged_visits)}")

    filtered_visits = filter_multi_visit_cards(merged_visits)
    unique_cards    = filtered_visits['card_id'].nunique()
    print(f"Card con >1 POI distinto: {unique_cards}")
    print(f"Totale righe finali: {len(filtered_visits)}\n")

    # 2) Matrice utente-POI + clusterizzazione ------------------------------
    user_poi_matrix = create_user_poi_matrix(filtered_visits)

    scaler = StandardScaler()
    user_poi_matrix_scaled = scaler.fit_transform(user_poi_matrix)

    kmeans   = KMeans(n_clusters=7, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(user_poi_matrix_scaled)

    user_clusters = pd.DataFrame({
        "card_id": user_poi_matrix.index,
        "cluster": clusters
    })

    print(user_clusters["cluster"].value_counts(), "\n")

    # 3) SCELTA DI UN UTENTE DA TESTARE -------------------------------------
    card_id = random.choice(user_clusters["card_id"].tolist())
    print(f"Testo la predizione sul card_id = {card_id}")

    # 4) COSTRUZIONE DEL PROMPT ---------------------------------------------
    prompt = create_prompt_with_cluster(
        filtered_visits,
        user_clusters,
        card_id,
        top_k=5)

    # 5) CHIAMATA AL MODELLO LLaMA / OLLAMA ---------------------------------
    llm_answer = get_chat_completion(prompt)

    if llm_answer is None:
        print("⚠️  Nessuna risposta dal modello LLaMA.")
        return

    print("\nRisposta grezza del modello:\n", llm_answer)

    # 6) PARSING  ------------------------------------------------------------
    try:
        obj = json.loads(llm_answer)
        prediction = obj.get("prediction")
        reason     = obj.get("reason")
        print("\n🎯 Prossimo POI previsto:", prediction)
        print("💡 Motivo/distanza sintetica:", reason)
    except json.JSONDecodeError:
        print("ℹ️  La risposta non è in JSON – la lascio così com'è.")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n\nInterruzione manuale rilevata. Uscita in modo sicuro...")
