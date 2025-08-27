# 🔥 Mixtral 8x7B - Mixture of Experts Results

## 📁 Contenuto della cartella
Risultati per Mixtral 8x7B (Mixture-of-Experts, 47B parametri attivi).

## 📋 Grafici richiesti:

### 🎯 **Performance principali**

1. **`mixtral_performance_overview.png`**
   - **Contenuto**: Top-1, Top-5, MRR, Coverage in barre raggruppate
   - **Confronto**: Con baseline LLaMA 3.1 8B
   - **Colore**: Arancione principale (#ff7f0e)

2. **`mixtral_vs_llama_comparison.png`**
   - **Contenuto**: Confronto diretto Mixtral vs LLaMA
   - **Tipo**: Barre affiancate per ogni metrica
   - **Indicare**: Miglioramento/peggioramento percentuale

### 📊 **Analisi specifiche Mixtral**

3. **`mixtral_confusion_matrix.png`**
   - **Contenuto**: Matrice confusione per strategia geospaziale
   - **Formato**: Heat map 20x20 POI più frequenti

4. **`mixtral_temporal_analysis.png`**
   - **Contenuto**: Performance per anno (2014-2023)
   - **Confrontare**: Stabilità vs LLaMA nel tempo

5. **`mixtral_error_patterns.png`**
   - **Contenuto**: Pattern di errore specifici di Mixtral
   - **Focus**: Errori diversi da LLaMA (analisi qualitativa)

6. **`mixtral_efficiency_analysis.png`**
   - **Contenuto**: Tempo processing vs accuracy
   - **Evidenziare**: Trade-off MoE (più lento ma più accurato?)

### 🔍 **Analisi architetturale**

7. **`mixtral_experts_utilization.png`** *(se disponibile)*
   - **Contenuto**: Utilizzo dei diversi expert durante inferenza
   - **Tipo**: Distribuzione percentuale expert attivati

8. **`mixtral_geographic_preferences.png`**
   - **Contenuto**: Bias geografici specifici di Mixtral
   - **Confronto**: Con pattern geografici di LLaMA

## 🎨 Specifiche
- **Colore primario**: `#ff7f0e` (arancione)
- **Colore confronto**: `#1f77b4` (blu LLaMA)
- **Evidenziare**: Caratteristiche unique di MoE architecture