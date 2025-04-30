import pandas as pd
from pandas import DataFrame

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

    print(filtered_visits.head())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n\nInterruzione manuale rilevata. Uscita in modo sicuro...")
