# 🤔 DeepSeek-V3 8B - Reasoning-Focused Results

## 📁 Contenuto della cartella
Risultati per DeepSeek-V3 8B (architettura ibrida con focus su ragionamento).

## 📋 Grafici richiesti:

### 🎯 **Performance e ragionamento**

1. **`deepseek_performance_overview.png`**
   - **Contenuto**: Metriche principali con focus su reasoning quality
   - **Confronto**: Con baseline LLaMA 3.1 8B
   - **Colore**: Rosso principale (#d62728)

2. **`deepseek_reasoning_quality.png`**
   - **Contenuto**: Analisi qualità delle spiegazioni fornite
   - **Metrica**: Lunghezza e coerenza dei "reason" fields
   - **Tipo**: Box plot distribuzione lunghezza reasoning

3. **`deepseek_vs_baseline_detailed.png`**
   - **Contenuto**: Confronto approfondito con LLaMA
   - **Focus**: Differenze in complex reasoning tasks

### 📊 **Analisi architetturale**

4. **`deepseek_confusion_matrix.png`**
   - **Contenuto**: Matrice confusione strategia geospaziale
   - **Evidenziare**: Pattern errori diversi da baseline

5. **`deepseek_complex_scenarios.png`**
   - **Contenuto**: Performance su scenari complessi
   - **Definizione**: Turisti con >5 POI visitati, pattern non-standard

6. **`deepseek_temporal_reasoning.png`**
   - **Contenuto**: Capacità reasoning su constraints temporali
   - **Analisi**: Accuracy quando inclusi vincoli orari

### 🔍 **Capacità di ragionamento**

7. **`deepseek_logical_consistency.png`**
   - **Contenuto**: Coerenza logica nelle predizioni
   - **Metrica**: Predizioni che violano constraints geografici/temporali

8. **`deepseek_explanation_analysis.png`**
   - **Contenuto**: Qualità spiegazioni vs accuracy
   - **Tipo**: Scatter plot explanation_score vs hit_rate

9. **`deepseek_edge_cases.png`**
   - **Contenuto**: Performance su casi limite
   - **Focus**: Turisti con behavior anomali

## 🎨 Specifiche
- **Colore primario**: `#d62728` (rosso)
- **Focus**: Reasoning capabilities e logical consistency
- **Evidenziare**: Qualità explanations oltre pure performance