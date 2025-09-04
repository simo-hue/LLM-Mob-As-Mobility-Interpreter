# 📊 CSV per Presentazione Canva

Questa directory contiene i file CSV ottimizzati per l'importazione in Canva, generati dal notebook `export_csv_for_canva.ipynb`.

## 📁 File CSV Generati

### 1. `metriche_per_anno.csv`
**Scopo**: Grafici temporali delle performance per anno
- **Colonne principali**: `year`, `Top-1_Accuracy_Percent`, `Top-5_Hit_Rate_Percent`, `MRR_Percent`
- **Utilizzo in Canva**: Grafici a linee o a barre per mostrare trend temporali
- **Contenuto**: Performance annuali del modello principale

### 2. `confronto_modelli.csv`
**Scopo**: Confronto delle performance tra diversi modelli
- **Colonne principali**: `Model`, `Top-1_Accuracy_Percent`, `Top-5_Hit_Rate_Percent`, `Rank_Top1`
- **Utilizzo in Canva**: Grafici a barre per confronti, tabelle comparative
- **Contenuto**: Performance aggregate di tutti i modelli testati

### 3. `metriche_globali.csv`
**Scopo**: KPI principali per dashboard/overview
- **Colonne principali**: `Metric`, `Percentage`, `Scope`
- **Utilizzo in Canva**: KPI cards, metriche principali
- **Contenuto**: Metriche aggregate globali

### 4. `coverage_analysis.csv`
**Scopo**: Analisi della copertura del catalogo POI
- **Colonne principali**: `Model`, `Catalogue_Coverage`, `Coverage_Recall`, `Coverage_Precision`
- **Utilizzo in Canva**: Grafici di copertura, analisi qualitative
- **Contenuto**: Metriche di copertura per ogni modello

### 5. `yearly_summary.csv`
**Scopo**: Tabelle dettagliate con statistiche complete
- **Colonne principali**: `year`, `Total_Predictions`, `Top1_Mean_Percent`, `Error_Rate_Percent`
- **Utilizzo in Canva**: Tabelle dettagliate, appendici
- **Contenuto**: Statistiche complete per anno con deviazioni standard

## 🎨 Suggerimenti per Canva

### Grafici Temporali
- Usa `metriche_per_anno.csv` con colonne `year` (X) e `*_Percent` (Y)
- Perfetto per line chart o bar chart temporali

### Confronti Modelli
- Usa `confronto_modelli.csv` con `Model` (X) e `Top-1_Accuracy_Percent` (Y)
- Ideale per horizontal bar charts

### KPI Dashboard
- Usa `metriche_globali.csv` per creare cards con valori principali
- Filtra per `Metric` specifiche

### Tabelle Dettagliate
- Usa `yearly_summary.csv` per tabelle complete
- Include deviazioni standard e conteggi

## 🔄 Rigenerazione dei File

Per aggiornare i CSV:
1. Apri `export_csv_for_canva.ipynb`
2. Esegui tutte le celle
3. I nuovi file CSV verranno sovrascritti in questa directory

## 📋 Formato Dati

- **Percentuali**: Già convertite (es. 4.32 = 4.32%)
- **Nomi modelli**: Formattati per presentazione
- **Anni**: In formato numerico (2014, 2015, etc.)
- **Encoding**: UTF-8 compatibile con Canva

## 🎯 Tipologie di Visualizzazioni Consigliate

### Per `metriche_per_anno.csv`:
- Line chart: Trend temporali
- Bar chart: Confronto anno per anno
- Area chart: Evoluzione nel tempo

### Per `confronto_modelli.csv`:
- Horizontal bar chart: Ranking modelli  
- Radar chart: Confronto multi-dimensionale
- Table: Dettagli numerici

### Per `metriche_globali.csv`:
- KPI cards: Metriche principali
- Gauge charts: Performance indicators
- Icon + number: Dashboard style

Tutti i file sono ottimizzati per mantenere uno stile coerente nella tua presentazione Canva! 🚀