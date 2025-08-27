# 📊 Multi-Model Comparison Graphics

## 📁 Contenuto della cartella
Questa cartella contiene i grafici comparativi tra tutti i modelli testati.

## 📋 Grafici richiesti e naming convention:

### 🎯 **Grafici principali comparativi**

1. **`models_performance_comparison.png`**
   - **Tipo**: Grafico a barre raggruppate
   - **Contenuto**: Confronto Top-1 Accuracy, Top-5 Hit Rate, MRR per tutti i 6 modelli
   - **Uso in LaTeX**: `\label{fig:multi_model_comparison}` (linea 299)

2. **`performance_efficiency_scatter.png`** 
   - **Tipo**: Scatter plot
   - **Contenuto**: Performance (MRR) vs Efficienza (Cards/Second)
   - **Note**: Dimensione punti = numero parametri
   - **Uso in LaTeX**: Figura trade-off analysis

3. **`models_performance_heatmap.png`**
   - **Tipo**: Heat map
   - **Contenuto**: Matrice modelli × metriche (normalizzata 0-100)
   - **Righe**: 6 modelli, **Colonne**: Top-1, Top-5, MRR, Coverage, Efficiency

4. **`models_radar_comparison.png`**
   - **Tipo**: Grafico radar multi-dimensionale  
   - **Contenuto**: 5 metriche per tutti i modelli sovrapposti
   - **Scale**: Normalizzata 0-100

### 📈 **Grafici per singola metrica**

5. **`top1_accuracy_by_model.png`**
   - **Tipo**: Barre verticali con error bars
   - **Ordine modelli**: LLaMA → Mixtral → Qwen → DeepSeek → Gemma3-8B → Gemma3-27B

6. **`top5_hitrate_by_model.png`**
   - **Tipo**: Barre verticali con error bars
   - **Stesso ordine dei modelli**

7. **`mrr_by_model.png`**
   - **Tipo**: Barre verticali con error bars
   - **Stesso ordine dei modelli**

8. **`coverage_by_model.png`**
   - **Tipo**: Barre verticali 
   - **Contenuto**: Catalogue Coverage per modello

9. **`processing_time_by_model.png`**
   - **Tipo**: Barre verticali (logaritmiche se necessario)
   - **Contenuto**: Tempo di elaborazione medio per card

### 🎨 **Specifiche di stile**
- **Colori consistenti** per ogni modello attraverso tutti i grafici
- **Palette suggerita**:
  - LLaMA 3.1: `#1f77b4` (blu)
  - Mixtral: `#ff7f0e` (arancione) 
  - Qwen 2.5: `#2ca02c` (verde)
  - DeepSeek-V3: `#d62728` (rosso)
  - Gemma3-8B: `#9467bd` (viola)
  - Gemma3-27B: `#8c564b` (marrone)

- **Dimensioni**: 1200x800 pixel (300 DPI)
- **Font**: Arial/Helvetica, size 12-14
- **Background**: Bianco
- **Grid**: Sottile, grigio chiaro

## ✅ Checklist creazione grafici
- [ ] Tutti i 9 grafici creati
- [ ] Colori consistenti tra grafici
- [ ] Legende chiare e leggibili
- [ ] Dimensioni e qualità appropriate
- [ ] Nomi file corrispondenti esattamente a quelli specificati