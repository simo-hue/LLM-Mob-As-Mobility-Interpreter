import json
import random
import time
import os, sys, argparse, logging
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
import pickle
from typing import Optional, List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# GPU libraries (optional)
try:
    import cupy as cp
    from cuml.cluster import KMeans as cuKMeans
    from cuml.preprocessing import StandardScaler as cuStandardScaler
    CUDA_AVAILABLE = True
    print("✅ CUDA disponibile - usando cuML")
except ImportError:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    CUDA_AVAILABLE = False
    print("⚠️  CUDA non disponibile - usando sklearn")

# Transformers per LLM locale (alternativa a Ollama)
try:
    from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
    import torch
    TRANSFORMERS_AVAILABLE = True
    print("✅ Transformers disponibile")
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Transformers non disponibile")

from tqdm import tqdm
from pandas import DataFrame

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calcola la distanza in chilometri tra due punti geografici
    usando la formula dell'haversine con ottimizzazioni numpy.
    """
    R = 6371  # Raggio della Terra in km
    
    # Vectorized haversine formula
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c

def calculate_distance_matrix_gpu(coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
    """
    Calcola matrice delle distanze usando GPU (se disponibile)
    """
    if CUDA_AVAILABLE:
        coords1_gpu = cp.asarray(coords1)
        coords2_gpu = cp.asarray(coords2)
        
        # Implementazione haversine su GPU
        R = 6371
        lat1 = cp.radians(coords1_gpu[:, 0])[:, None]
        lon1 = cp.radians(coords1_gpu[:, 1])[:, None]
        lat2 = cp.radians(coords2_gpu[:, 0])[None, :]
        lon2 = cp.radians(coords2_gpu[:, 1])[None, :]
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = cp.sin(dlat/2)**2 + cp.cos(lat1) * cp.cos(lat2) * cp.sin(dlon/2)**2
        c = 2 * cp.arcsin(cp.sqrt(a))
        
        return cp.asnumpy(R * c)
    else:
        # Fallback CPU
        distances = np.zeros((len(coords1), len(coords2)))
        for i, (lat1, lon1) in enumerate(coords1):
            for j, (lat2, lon2) in enumerate(coords2):
                distances[i, j] = calculate_distance(lat1, lon1, lat2, lon2)
        return distances

class GPUOptimizedClusterer:
    """Clustering ottimizzato per GPU usando cuML"""
    
    def __init__(self, n_clusters: int = 7, random_state: int = 42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        
        if CUDA_AVAILABLE:
            self.scaler = cuStandardScaler()
            self.kmeans = cuKMeans(n_clusters=n_clusters, random_state=random_state)
        else:
            self.scaler = StandardScaler()
            self.kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit e predict in una sola operazione"""
        if CUDA_AVAILABLE:
            X_scaled = self.scaler.fit_transform(X)
            return self.kmeans.fit_predict(X_scaled).to_numpy()
        else:
            X_scaled = self.scaler.fit_transform(X)
            return self.kmeans.fit_predict(X_scaled)

class LocalLLMPredictor:
    """Predictor usando modelli locali invece di Ollama"""
    
    def __init__(self, model_name: str = "microsoft/DialoGPT-medium"):
        self.model = None
        self.tokenizer = None
        self.model_name = model_name
        
        if TRANSFORMERS_AVAILABLE:
            self._load_model()
    
    def _load_model(self):
        """Carica il modello locale"""
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"✅ Modello {self.model_name} caricato su {device}")
        except Exception as e:
            print(f"❌ Errore nel caricamento del modello: {e}")
            self.model = None
    
    def predict(self, prompt: str) -> Optional[str]:
        """Genera predizione usando il modello locale"""
        if not self.model:
            return None
            
        try:
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response[len(prompt):].strip()
        except Exception as e:
            print(f"❌ Errore nella predizione: {e}")
            return None

# Funzioni di utilità ottimizzate
def load_data_optimized(visits_path: Path, poi_path: Path) -> tuple:
    """Caricamento dati ottimizzato con caching"""
    cache_file = visits_path.parent / f"{visits_path.stem}_cached.pkl"
    
    if cache_file.exists():
        print(f"📦 Caricamento cache: {cache_file}")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    
    # Caricamento normale
    pois = pd.read_csv(poi_path, usecols=["name_short", "latitude", "longitude"])
    
    visits = pd.read_csv(
        visits_path,
        usecols=[0, 1, 2, 4],
        names=["data", "ora", "name_short", "card_id"],
        header=0,
        dtype={"card_id": str},
    )
    visits["timestamp"] = pd.to_datetime(
        visits["data"] + " " + visits["ora"], 
        format="%d-%m-%y %H:%M:%S"
    )
    visits = visits[["timestamp", "card_id", "name_short"]].sort_values("timestamp")
    
    # Merge e filtro
    merged = visits.merge(pois[["name_short"]], on="name_short", how="inner")
    valid_cards = merged.groupby("card_id")["name_short"].nunique()
    valid_cards = valid_cards[valid_cards > 1].index
    filtered = merged[merged["card_id"].isin(valid_cards)]
    
    # Salva cache
    data = (pois, filtered)
    with open(cache_file, 'wb') as f:
        pickle.dump(data, f)
    
    return data

def process_single_card(args_tuple) -> Dict[str, Any]:
    """
    Processa una singola card - funzione separata per multiprocessing
    """
    (card_id, filtered_data, pois_data, user_clusters, anchor_rule, top_k) = args_tuple
    
    # Ricostruisci i dati necessari
    seq = (
        filtered_data.loc[filtered_data.card_id == card_id]
        .sort_values("timestamp")["name_short"].tolist()
    )
    
    if len(seq) < 3:
        return None
    
    target = seq[-1]
    prefix = seq[:-1]
    
    # Calcola anchor index
    if anchor_rule == "penultimate":
        idx = len(prefix) - 1
    elif anchor_rule == "first":
        idx = 0
    elif anchor_rule == "middle":
        idx = len(prefix) // 2
    else:
        idx = int(anchor_rule) if isinstance(anchor_rule, int) else len(prefix) - 1
    
    current_poi = prefix[idx]
    history_list = [p for i, p in enumerate(prefix) if i != idx]
    
    # Ottieni cluster
    cluster_id = user_clusters.loc[
        user_clusters["card_id"] == card_id, "cluster"
    ].values[0]
    
    # Calcola distanze (versione semplificata)
    current_poi_row = pois_data[pois_data["name_short"] == current_poi]
    if current_poi_row.empty:
        prediction = random.sample(
            [p for p in pois_data["name_short"] if p not in history_list and p != current_poi],
            min(top_k, len(pois_data) - len(history_list) - 1)
        )
    else:
        current_lat = current_poi_row["latitude"].iloc[0]
        current_lon = current_poi_row["longitude"].iloc[0]
        
        # Calcola distanze da tutti i POI
        available_pois = []
        for _, row in pois_data.iterrows():
            poi_name = row["name_short"]
            if poi_name in history_list or poi_name == current_poi:
                continue
            
            distance = calculate_distance(
                current_lat, current_lon,
                row["latitude"], row["longitude"]
            )
            available_pois.append((poi_name, distance))
        
        # Ordina per distanza e prendi i top_k
        available_pois.sort(key=lambda x: x[1])
        prediction = [poi[0] for poi in available_pois[:top_k]]
    
    return {
        "card_id": card_id,
        "cluster": cluster_id,
        "history": str(history_list),
        "current_poi": current_poi,
        "prediction": str(prediction),
        "ground_truth": target,
        "reason": f"Distanza geografica - cluster {cluster_id}",
        "hit": target in prediction
    }

def run_parallel_processing(
    visits_path: Path, 
    poi_path: Path, 
    max_users: Optional[int] = None,
    n_workers: Optional[int] = None
) -> None:
    """
    Versione ottimizzata per processing parallelo
    """
    if n_workers is None:
        n_workers = min(cpu_count(), 32)  # Limita per evitare overhead
    
    print(f"🚀 Avvio processing parallelo con {n_workers} worker")
    
    # Carica dati
    pois, filtered = load_data_optimized(visits_path, poi_path)
    
    # Clustering ottimizzato
    matrix = pd.crosstab(filtered["card_id"], filtered["name_short"])
    clusterer = GPUOptimizedClusterer(n_clusters=7, random_state=42)
    clusters = clusterer.fit_predict(matrix.values)
    user_clusters = pd.DataFrame({"card_id": matrix.index, "cluster": clusters})
    
    # Seleziona utenti
    eligible = (
        filtered.groupby("card_id").size()
        .loc[lambda s: s >= 3].index.tolist()
    )
    
    if max_users:
        eligible = random.sample(eligible, min(max_users, len(eligible)))
    
    # Prepara argomenti per multiprocessing
    args_list = [
        (card_id, filtered, pois, user_clusters, "penultimate", 5)
        for card_id in eligible
    ]
    
    results = []
    
    # Processing parallelo
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_card = {
            executor.submit(process_single_card, args): args[0] 
            for args in args_list
        }
        
        for future in tqdm(as_completed(future_to_card), total=len(args_list), desc="Cards"):
            try:
                result = future.result()
                if result:
                    results.append(result)
            except Exception as e:
                card_id = future_to_card[future]
                print(f"❌ Errore per card {card_id}: {e}")
    
    # Salva risultati
    if results:
        df_out = pd.DataFrame(results)
        output_dir = Path("results")
        output_dir.mkdir(exist_ok=True)
        
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_file = output_dir / f"{visits_path.stem}_pred_{ts}.csv"
        df_out.to_csv(output_file, index=False)
        
        hit_rate = df_out["hit"].mean()
        print(f"✅ Salvato {output_file} - Hit@5: {hit_rate:.2%}")

def main():
    parser = argparse.ArgumentParser(description="Versione ottimizzata per Leonardo")
    parser.add_argument("--input", type=str, required=True, help="File CSV di input")
    parser.add_argument("--poi", type=str, required=True, help="File POI CSV")
    parser.add_argument("--max-users", type=int, help="Numero massimo di utenti")
    parser.add_argument("--workers", type=int, help="Numero di worker paralleli")
    parser.add_argument("--gpu", action="store_true", help="Forza uso GPU")
    
    args = parser.parse_args()
    
    # Verifica GPU
    if args.gpu and not CUDA_AVAILABLE:
        print("⚠️  GPU richiesta ma CUDA non disponibile")
        sys.exit(1)
    
    # Esegui processing
    run_parallel_processing(
        Path(args.input),
        Path(args.poi),
        max_users=args.max_users,
        n_workers=args.workers
    )

if __name__ == "__main__":
    main()
