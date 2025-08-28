# 🦙 LLaMA 3.1 8B - Extended Results  

## 📁 Contenuto della cartella
Risultati dettagliati per LLaMA 3.1 8B (modello baseline) con tutte e tre le strategie di prompting.

## 📋 Grafici richiesti:

### 🎯 **Confronto strategie di prompting**

1. **`strategy_comparison_top1.png`**
   - **Contenuto**: Top-1 Accuracy per le 3 strategie
   - **Strategie**: Base → Geospaziale → Spazio-temporale
   - **Tipo**: Barre verticali con miglioramento percentuale

2. **`strategy_comparison_top5.png`**
   - **Contenuto**: Top-5 Hit Rate per le 3 strategie
   - **Stesso formato del precedente**

3. **`strategy_comparison_mrr.png`**
   - **Contenuto**: MRR per le 3 strategie
   - **Stesso formato del precedente**

### 📊 **Analisi dettagliate per strategia geospaziale**

4. **`confusion_matrix_geospatial.png`**
   - **Contenuto**: Matrice confusione per top-20 POI più frequenti
   - **Strategia**: Geospaziale (nome + coordinate)

5. **`worst_performing_pairs_geospatial.png`**
   - **Contenuto**: Top-10 coppie POI con più errori
   - **Formato**: Barre orizzontali con conteggi errori

6. **`temporal_performance_geospatial.png`**
   - **Contenuto**: Performance per anno (2014-2023)
   - **Metriche**: Top-1, Top-5, MRR in un unico grafico
   - **Evidenziare**: Anomalia COVID-19 (2020-2021)

7. **`cluster_performance_breakdown.png`**  
   - **Contenuto**: Performance per cluster turistico (K=7)
   - **Tipo**: Barre raggruppate (Top-1, Top-5, MRR per cluster)

### 🔍 **Analisi di interpretabilità**

8. **`geographic_bias_analysis.png`**
   - **Contenuto**: Frequenza predizioni vs posizione geografica POI
   - **Tipo**: Scatter plot con overlay mappa Verona

9. **`prediction_distance_distribution.png`**
   - **Contenuto**: Distribuzione distanze POI predetti vs attuali
   - **Tipo**: Istogramma con curva di densità

## 🎨 Specifiche tecniche
- **Colori**: Palette blu per coerenza con identità LLaMA
- **Dimensioni**: 1200x800 pixel (300 DPI)
- **Evidenziare**: Risultati baseline in tutti i confronti