import pandas as pd
import sys
import numpy as np
from pandas import DataFrame

def load_pois(filepath):
    """Carica il file vc_sites

    Args:
        filepath (String): percorso relativo

    Returns:
        _type_: df
    """
    df_pois = pd.read_csv(filepath)
    df_pois = df_pois[['id', 'name_short', 'latitude', 'longitude']]
    df_pois['id'] = df_pois['id'].astype(int)
    return df_pois

def merge_visits_pois(
    visits_df: DataFrame,
    pois_df: DataFrame
) -> DataFrame:
    """
    Unisce il DataFrame delle visite con quello dei POI sulla colonna 'name_short',
    restituendo solo timestamp, id e name_short, ordinati per timestamp.

    Parameters
    ----------
    visits_df : DataFrame
        Deve contenere almeno le colonne:
         - 'timestamp'
         - 'name_short'
    pois_df : DataFrame
        Deve contenere almeno le colonne:
         - 'id'
         - 'name_short'

    Returns
    -------
    DataFrame
        Colonne ['timestamp', 'id', 'name_short'], ordinate per 'timestamp'.
    """
    # Seleziono solo le colonne necessarie dai POI
    pois_sel = pois_df[['id', 'name_short']]

    # Inner join sulle visite
    merged = pd.merge(
        visits_df,
        pois_sel,
        on='name_short',
        how='inner'
    )

    # Filtra e ordina
    merged = (
        merged[['timestamp', 'id', 'name_short']]
        .sort_values('timestamp')
        .reset_index(drop=True)
    )

    return merged

def load_visits(filepath: str) -> DataFrame:
    """
    Carica il file delle visite in formato CSV, costruisce un timestamp
    e restituisce un DataFrame con ['timestamp', 'name_short'], ordinato.

    Args:
        filepath (str): percorso del CSV, con colonne almeno 'data','ora','name_short'

    Returns:
        DataFrame: colonne ['timestamp','name_short'], indicizzate da 0, ordinate per timestamp
    """
    # Leggo solo le colonne necessarie
    df = pd.read_csv(
        filepath,
        usecols=['data', 'ora', 'name_short']
    )

    # Concateno data e ora in una stringa uniforme
    ts_str = df['data'] + ' ' + df['ora']

    # Parsing rapido e senza warning: giorno-mese-anno(2 cifre) ore:min:sec
    df['timestamp'] = pd.to_datetime(
        ts_str,
        format='%d-%m-%y %H:%M:%S'
    )

    # Keep only what matters, ordino e resetto l'indice
    df = (
        df[['timestamp', 'name_short']]
        .sort_values('timestamp')
        .reset_index(drop=True)
    )
    return df

def filter_single_visit_users(df):
    poi_counts = df.groupby('user_id')['id'].nunique()
    valid_users = poi_counts[poi_counts > 1].index
    filtered_df = df[df['user_id'].isin(valid_users)]
    return filtered_df

def main():
    
    # Carico vc_site
    pois = load_pois("data/verona/vc_site.csv")
    print(pois.head())

    # Carico .csv con i dati della verona card
    visits = load_visits("data/verona/dataset_veronacard_2014_2020/dati_2014.csv")
    print(visits.head())
    
    # Faccio il JOIN su name_short
    merged_visits = merge_visits_pois(visits, pois)
    print(merged_visits.head())
    
    
    filtered_visits = filter_single_visit_users(visits_with_users)
    print(filtered_visits.head())

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n\nInterruzione manuale rilevata. Uscita in modo sicuro...")
