import pandas as pd
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np

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

# DA REVISIONARE
def create_prompt_with_cluster(df, user_clusters, card_id):
    user_visits = df[df['card_id'] == card_id].sort_values('timestamp')
    poi_sequence = user_visits['name_short'].tolist()

    user_cluster = user_clusters.loc[user_clusters['card_id'] == card_id, 'cluster'].values[0]

    prompt = f"Cluster utente: {user_cluster}\n"
    prompt += f"Storico visite POI: {poi_sequence[:-1]}\n"
    prompt += f"Prevedi il prossimo POI che visiterà:"

    return prompt

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
def main():
    poi_file = "data/verona/vc_site.csv"
    visits_file = "data/verona/dataset_veronacard_2014_2020/dati_2014.csv"

    pois = load_pois(poi_file)
    visits = load_visits(visits_file)

    merged_visits = merge_visits_pois(visits, pois)
    print(f"Totale righe dopo merge_visits_pois: {len(merged_visits)}")

    filtered_visits = filter_multi_visit_cards(merged_visits)
    unique_cards = filtered_visits['card_id'].nunique()
    print(f"Numero di card_id con >1 visita distinta: {unique_cards}")
    print(f"Totale righe dopo filtro multi-visita: {len(filtered_visits)}")

    user_poi_matrix = create_user_poi_matrix(filtered_visits)
    # print(user_poi_matrix.head())
    
    # user_poi_matrix_scaled come matrice NumPy standardizzata pronta per KMeans.
    scaler = StandardScaler()
    user_poi_matrix_scaled = scaler.fit_transform(user_poi_matrix)

    # Grafico per capire quanti cluster
    plot_number_of_cluster(user_poi_matrix_scaled, 10)
    
    # Inizio la KMeans clusterizzazione
    kmeans = KMeans(n_clusters=7, random_state=42, n_init=10)

    clusters = kmeans.fit_predict(user_poi_matrix_scaled)

    # Salviamo i risultati in DataFrame per chiarezza
    user_clusters = pd.DataFrame({
        'card_id': user_poi_matrix.index,
        'cluster': clusters
    })

    print(user_clusters.head())
    
    user_poi_matrix['cluster'] = clusters
    cluster_analysis = user_poi_matrix.groupby('cluster').mean()

    print(cluster_analysis)
    print(user_poi_matrix['cluster'].value_counts())  # Quanti utenti per ciascun cluster



if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n\nInterruzione manuale rilevata. Uscita in modo sicuro...")
