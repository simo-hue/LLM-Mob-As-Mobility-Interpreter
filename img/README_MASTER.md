# 🎯 Master Guide - Multi-Model Graphics Structure

## 📁 Directory Structure Created

```
img/
├── multi_model_comparison/     # Grafici comparativi tra tutti i modelli
├── llama3.1_8b_extended/      # LLaMA 3.1 8B (baseline esteso)
├── mixtral_8x7b/              # Mixtral 8x7B (Mixture of Experts)
├── qwen2.5_7b/                # Qwen 2.5 7B (multilingue efficiente)
├── deepseek_v3_8b/            # DeepSeek-V3 8B (reasoning-focused)
├── gemma3_8b/                 # Gemma 3 8B (Google baseline)
└── gemma3_27b/                # Gemma 3 27B (Google large-scale)
```

## 🎨 Color Palette Standardizzata

| Modello | Colore Primario | Hex Code | Uso |
|---------|----------------|----------|-----|
| LLaMA 3.1 8B | Blu | `#1f77b4` | Baseline reference |
| Mixtral 8x7B | Arancione | `#ff7f0e` | MoE architecture |
| Qwen 2.5 7B | Verde | `#2ca02c` | Efficiency focus |
| DeepSeek-V3 8B | Rosso | `#d62728` | Reasoning capabilities |
| Gemma 3 8B | Viola | `#9467bd` | Google baseline |
| Gemma 3 27B | Marrone | `#8c564b` | Large-scale model |

## 📊 Grafici Prioritari per LaTeX

### 🥇 **PRIORITÀ ALTA** (necessari per compilazione)
1. `multi_model_comparison/models_performance_comparison.png`
2. Tabella comparativa (dati numerici da inserire in LaTeX)

### 🥈 **PRIORITÀ MEDIA** (migliorano significativamente la tesi)
3. `multi_model_comparison/performance_efficiency_scatter.png`
4. `multi_model_comparison/models_performance_heatmap.png`
5. Per ogni modello: `*_performance_overview.png`

### 🥉 **PRIORITÀ BASSA** (analisi approfondite)
6. Tutti gli altri grafici dettagliati per singolo modello

## 📋 LaTeX Integration Checklist

- [ ] **Aggiornare tabella** linea 272-288 in `5-Strategie di prompt e risultati sperimentali.tex`
- [ ] **Sostituire placeholder** linea 299 con `\includegraphics{../../img/multi_model_comparison/models_performance_comparison.png}`
- [ ] **Aggiungere figure** per ogni sezione modello-specifica
- [ ] **Verificare path** `../../img/` corretto da `tex/` directory

## 🚀 Workflow Suggerito

1. **Prima fase**: Crea i 6 grafici `*_performance_overview.png` + tabella comparativa
2. **Seconda fase**: Grafici comparativi in `multi_model_comparison/`
3. **Terza fase**: Analisi dettagliate per modelli specifici
4. **Quarta fase**: Grafici specializzati (confusion matrices, temporal analysis, etc.)

## 🔧 Specifiche Tecniche Universali

- **Dimensioni**: 1200x800 pixel
- **DPI**: 300 
- **Formato**: PNG con transparent background dove appropriato
- **Font**: Arial/Helvetica, size 12-14
- **Grid**: Sottile, grigio chiaro (#e0e0e0)
- **Background**: Bianco (#ffffff)

## ⚠️ Note Importanti

- Ogni cartella ha un `README.md` specifico con istruzioni dettagliate
- I nomi file sono **esatti** - rispettare esattamente per LaTeX integration
- Colori devono essere **consistenti** attraverso tutti i grafici
- Priorità: Prima funzionalità LaTeX, poi estetica avanzata