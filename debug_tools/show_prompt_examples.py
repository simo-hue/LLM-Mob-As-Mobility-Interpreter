#!/usr/bin/env python3
"""
Script per generare esempi di prompt reali dal dataset VeronaCard
Mostra la complessità dei prompt temporali vs base vs geom
"""

import os
import sys
import pandas as pd
import numpy as np
import random
from pathlib import Path
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import math

# Aggiungi il path dello script principale per importare le classi
script_dir = Path(__file__).resolve().parent
sys.path.append(str(script_dir))

# Import delle classi necessarie
from veronacard_mob_with_geom_time_parrallel import DataLoader, PromptBuilder

def calculate_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calcola distanza in km tra due punti geografici"""
    R = 6371  # Raggio Terra in km
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c

def create_base_prompt(visits, card_id, target_poi):
    """Prompt base - solo sequenza POI"""
    seq = visits[visits.card_id == card_id].sort_values("timestamp")["name_short"].tolist()
    prefix = seq[:-1]
    current_poi = prefix[-1]  # penultimate
    history = prefix[:-1]
    
    return f"""Predict the next POI for this tourist in Verona:

VISIT HISTORY: {', '.join(history) if history else 'First-time visitor'}
CURRENT LOCATION: {current_poi}

Task: Predict the next most likely destination.
Output: List the top 3 POI names.
"""

def create_geom_prompt(visits, pois_df, card_id, target_poi):
    """Prompt geom - base + contesto spaziale"""
    seq = visits[visits.card_id == card_id].sort_values("timestamp")["name_short"].tolist()
    prefix = seq[:-1]
    current_poi = prefix[-1]
    history = prefix[:-1]
    
    # Trova POI vicini
    current_poi_row = pois_df[pois_df["name_short"] == current_poi]
    if current_poi_row.empty:
        nearby_text = "No nearby POIs found"
    else:
        current_lat = current_poi_row["latitude"].iloc[0]
        current_lon = current_poi_row["longitude"].iloc[0]
        
        nearby_pois = []
        for _, row in pois_df.iterrows():
            poi_name = row["name_short"]
            if poi_name not in prefix and poi_name != current_poi:
                distance = calculate_distance(current_lat, current_lon, row["latitude"], row["longitude"])
                if distance <= 2.0:  # Entro 2km
                    nearby_pois.append(f"{poi_name} ({distance:.1f}km)")
        
        nearby_text = ", ".join(nearby_pois[:5]) if nearby_pois else "No nearby POIs within 2km"
    
    return f"""Predict the next POI for this tourist in Verona:

VISIT HISTORY: {', '.join(history) if history else 'First-time visitor'}
CURRENT LOCATION: {current_poi}
NEARBY ATTRACTIONS: {nearby_text}

Task: Predict the next most likely destination considering spatial proximity.
Output: List the top 3 POI names.
"""

def create_time_prompt(visits, pois_df, user_clusters, card_id, target_poi):
    """Prompt temporale completo - base + geom + tempo"""
    card_visits = visits[visits.card_id == card_id].sort_values("timestamp")
    seq = card_visits["name_short"].tolist()
    prefix = seq[:-1]
    current_poi = prefix[-1]
    history = prefix[:-1]
    
    # Info temporali
    current_visit = card_visits.iloc[-2]  # penultimate visit
    current_time = current_visit["time"]
    current_day = current_visit["day_of_week"]
    
    # Pattern temporali dall'history
    history_visits = card_visits.iloc[:-1]
    avg_hour = history_visits["hour"].mean()
    visit_hours = history_visits["hour"].tolist()
    days_visited = history_visits["day_of_week"].unique().tolist()
    
    # Cluster del turista
    cluster_id = user_clusters.loc[user_clusters["card_id"] == card_id, "cluster"].values[0]
    
    # POI vicini (come geom)
    current_poi_row = pois_df[pois_df["name_short"] == current_poi]
    if current_poi_row.empty:
        nearby_text = "No nearby POIs found"
    else:
        current_lat = current_poi_row["latitude"].iloc[0]
        current_lon = current_poi_row["longitude"].iloc[0]
        
        nearby_pois = []
        for _, row in pois_df.iterrows():
            poi_name = row["name_short"]
            if poi_name not in prefix and poi_name != current_poi:
                distance = calculate_distance(current_lat, current_lon, row["latitude"], row["longitude"])
                if distance <= 2.0:
                    nearby_pois.append(f"{poi_name} ({distance:.1f}km)")
        
        nearby_text = ", ".join(nearby_pois[:5]) if nearby_pois else "No nearby POIs within 2km"
    
    # Contesto temporale
    time_context = f"Current: {current_day} {current_time.strftime('%H:%M')}"
    if visit_hours:
        time_context += f", usual hours: {visit_hours}, avg: {avg_hour:.1f}h"
    if len(days_visited) > 1:
        time_context += f", days visited: {', '.join(days_visited[:3])}"
    
    return f"""You are an expert tourism analyst predicting visitor behavior in Verona, Italy.

TOURIST PROFILE:
- Cluster: {cluster_id} (behavioral pattern group)  
- Visit history: {', '.join(history) if history else 'First-time visitor'}
- Current location: {current_poi}

TEMPORAL CONTEXT:
{time_context}

SPATIAL CONTEXT:
Nearby attractions within walking distance: {nearby_text}

TASK:
Predict exactly 5 most likely next destinations for this tourist. Order them by decreasing probability (most likely first)

OUTPUT FORMAT:
Respond in JSON format like this: {{"prediction": ["poi1", "poi2", "poi3", "poi4", "poi5"], "reason": "brief explanation"}}.
"""

def main():
    """Genera esempi di prompt per confronto"""
    print("🔍 VERONA CARD - PROMPT EXAMPLES GENERATOR")
    print("=" * 60)
    
    # Percorsi file
    data_dir = Path("data/verona")
    poi_file = data_dir / "vc_site.csv" 
    visits_file = data_dir / "dati_2014.csv"
    
    if not poi_file.exists():
        print(f"❌ POI file not found: {poi_file}")
        return
    
    if not visits_file.exists():
        print(f"❌ Visits file not found: {visits_file}")
        return
    
    print(f"📁 Loading data from {data_dir}")
    
    # Carica e preprocessa dati
    try:
        pois_df = DataLoader.load_pois(poi_file)
        visits_df = DataLoader.load_visits(visits_file)
        merged_df = DataLoader.merge_visits_pois(visits_df, pois_df)
        filtered_df = DataLoader.filter_multi_visit_cards(merged_df)
        
        print(f"✅ Loaded {len(filtered_df)} valid visits from {filtered_df.card_id.nunique()} cards")
        
        # Clustering semplificato
        user_poi_matrix = DataLoader.create_user_poi_matrix(filtered_df)
        scaler = StandardScaler()
        scaled_matrix = scaler.fit_transform(user_poi_matrix)
        clusters = KMeans(n_clusters=7, random_state=42, n_init=10).fit_predict(scaled_matrix)
        user_clusters = pd.DataFrame({
            "card_id": user_poi_matrix.index,
            "cluster": clusters
        })
        
        print(f"✅ Clustering completed - {len(user_clusters)} users in 7 clusters")
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Seleziona 3 carte casuali con abbastanza visite
    eligible_cards = filtered_df.groupby("card_id").size()
    eligible_cards = eligible_cards[eligible_cards >= 5].index.tolist()
    
    if len(eligible_cards) < 3:
        print("❌ Not enough eligible cards found")
        return
    
    sample_cards = random.sample(eligible_cards, 3)
    print(f"📋 Selected sample cards: {sample_cards}")
    
    # Genera esempi per ogni tipo di prompt
    output_dir = Path("prompt_examples")
    output_dir.mkdir(exist_ok=True)
    
    for i, card_id in enumerate(sample_cards, 1):
        card_visits = filtered_df[filtered_df.card_id == card_id].sort_values("timestamp")
        seq = card_visits["name_short"].tolist()
        target = seq[-1]
        
        print(f"\n🎯 EXAMPLE {i} - Card: {card_id}")
        print(f"   Sequence: {' → '.join(seq)}")
        print(f"   Target: {target}")
        
        # Genera i 3 tipi di prompt
        try:
            base_prompt = create_base_prompt(filtered_df, card_id, target)
            geom_prompt = create_geom_prompt(filtered_df, pois_df, card_id, target) 
            time_prompt = create_time_prompt(filtered_df, pois_df, user_clusters, card_id, target)
            
            # Salva esempi
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            with open(output_dir / f"example_{i}_base_{timestamp}.txt", 'w') as f:
                f.write(f"CARD: {card_id}\n")
                f.write(f"SEQUENCE: {' → '.join(seq)}\n")
                f.write(f"TARGET: {target}\n")
                f.write(f"LENGTH: {len(base_prompt)} chars\n")
                f.write("=" * 50 + "\n\n")
                f.write(base_prompt)
            
            with open(output_dir / f"example_{i}_geom_{timestamp}.txt", 'w') as f:
                f.write(f"CARD: {card_id}\n")
                f.write(f"SEQUENCE: {' → '.join(seq)}\n")
                f.write(f"TARGET: {target}\n")
                f.write(f"LENGTH: {len(geom_prompt)} chars\n")
                f.write("=" * 50 + "\n\n")
                f.write(geom_prompt)
            
            with open(output_dir / f"example_{i}_time_{timestamp}.txt", 'w') as f:
                f.write(f"CARD: {card_id}\n")
                f.write(f"SEQUENCE: {' → '.join(seq)}\n")
                f.write(f"TARGET: {target}\n")
                f.write(f"LENGTH: {len(time_prompt)} chars\n")
                f.write("=" * 50 + "\n\n")
                f.write(time_prompt)
            
            # Mostra statistiche
            print(f"   📏 BASE:  {len(base_prompt):4d} chars, {len(base_prompt.splitlines()):2d} lines")
            print(f"   📏 GEOM:  {len(geom_prompt):4d} chars, {len(geom_prompt.splitlines()):2d} lines")
            print(f"   📏 TIME:  {len(time_prompt):4d} chars, {len(time_prompt.splitlines()):2d} lines")
            print(f"   💾 Saved to: prompt_examples/example_{i}_*.txt")
            
        except Exception as e:
            print(f"   ❌ Error generating prompts for {card_id}: {e}")
    
    print(f"\n✅ COMPLETED - Examples saved in {output_dir}/")
    print("\n📊 SUMMARY:")
    print("   BASE  = Only POI sequence")
    print("   GEOM  = POI sequence + spatial context (nearby POIs)")  
    print("   TIME  = POI sequence + spatial + temporal context (time patterns, clusters)")
    print("\nCheck the generated files to see prompt complexity differences!")

if __name__ == "__main__":
    main()