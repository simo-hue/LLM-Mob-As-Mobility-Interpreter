#!/usr/bin/env python3
"""
Test dello script per generare prompt TIME semplificati
Confronta la complessità originale vs semplificata
"""

import sys
from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Import delle classi
sys.path.append(str(Path(__file__).resolve().parent))
from veronacard_mob_with_geom_time_parrallel import DataLoader, PromptBuilder, Config

def main():
    """Testa il prompt semplificato"""
    print("🔧 TESTING SIMPLIFIED TIME PROMPT")
    print("=" * 50)
    
    # Carica dati mini-sample
    data_dir = Path("data/verona")
    pois_df = DataLoader.load_pois(data_dir / "vc_site.csv")
    visits_df = DataLoader.load_visits(data_dir / "dati_2014.csv")
    merged_df = DataLoader.merge_visits_pois(visits_df, pois_df)
    filtered_df = DataLoader.filter_multi_visit_cards(merged_df)
    
    # Clustering veloce
    user_poi_matrix = DataLoader.create_user_poi_matrix(filtered_df)
    clusters = KMeans(n_clusters=7, random_state=42, n_init=5).fit_predict(
        StandardScaler().fit_transform(user_poi_matrix)
    )
    user_clusters = pd.DataFrame({
        "card_id": user_poi_matrix.index,
        "cluster": clusters
    })
    
    # Prende una carta di esempio
    sample_cards = filtered_df.groupby("card_id").size()
    sample_cards = sample_cards[sample_cards >= 5].index.tolist()
    card_id = sample_cards[0]
    
    print(f"📋 Testing with card: {card_id}")
    
    # Testa con DEBUG_MODE = False (originale)
    Config.DEBUG_MODE = False
    original_prompt = PromptBuilder.create_prompt(
        filtered_df, user_clusters, pois_df, card_id
    )
    
    # Testa con DEBUG_MODE = True (semplificato)
    Config.DEBUG_MODE = True
    simplified_prompt = PromptBuilder.create_prompt(
        filtered_df, user_clusters, pois_df, card_id
    )
    
    print(f"\n📏 ORIGINAL PROMPT:")
    print(f"   Length: {len(original_prompt)} chars")
    print(f"   Lines:  {len(original_prompt.splitlines())} lines")
    
    print(f"\n📏 SIMPLIFIED PROMPT:")
    print(f"   Length: {len(simplified_prompt)} chars")
    print(f"   Lines:  {len(simplified_prompt.splitlines())} lines")
    
    reduction = (1 - len(simplified_prompt) / len(original_prompt)) * 100
    print(f"\n🎯 REDUCTION: {reduction:.1f}%")
    
    print(f"\n" + "="*30 + " ORIGINAL " + "="*30)
    print(original_prompt)
    
    print(f"\n" + "="*30 + " SIMPLIFIED " + "="*28)
    print(simplified_prompt)
    
    print(f"\n✅ Simplified prompt is {reduction:.1f}% smaller!")
    print("   → Should enable parallel processing like GEOM version")
    print("   → Maintains temporal info but reduces LLM complexity")
    print("   → Compatible with CSV output format requirement")
    
    # Test del parser
    test_responses = [
        "Arena, Casa Giulietta, Torre Lamberti, San Zeno, Duomo",
        "1. Arena\n2. Casa Giulietta\n3. Torre Lamberti",
        "Arena"
    ]
    
    print(f"\n🧪 TESTING PARSER:")
    from veronacard_mob_with_geom_time_parrallel import CardProcessor
    
    # Create a dummy processor for testing
    processor = type('obj', (object,), {
        '_parse_simple_response': CardProcessor._parse_simple_response
    })()
    
    for i, test_resp in enumerate(test_responses, 1):
        parsed = processor._parse_simple_response(test_resp)
        print(f"   Test {i}: {test_resp[:30]}...")
        print(f"   Result: {parsed}")
        print(f"   CSV format: {str(parsed)}")

if __name__ == "__main__":
    main()