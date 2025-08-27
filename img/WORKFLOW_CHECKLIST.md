# ✅ Workflow Checklist - Multi-Model Graphics

## 📋 Phase 1: Essential Data Collection
- [ ] **Run experiments** for all 6 models with geospatial strategy
- [ ] **Collect metrics**: Top-1 Acc, Top-5 HR, MRR, Coverage, Processing Time
- [ ] **Calculate statistics**: Mean, std deviation, confidence intervals
- [ ] **Document configurations**: Hyperparameters, hardware specs per model

## 🎯 Phase 2: Priority Graphics Creation

### Core Comparison Graphics (MUST HAVE)
- [ ] `multi_model_comparison/models_performance_comparison.png` 
- [ ] Update LaTeX table (lines 272-288) with real data
- [ ] `multi_model_comparison/performance_efficiency_scatter.png`

### Per-Model Overview Graphics
- [ ] `llama3.1_8b_extended/strategy_comparison_top1.png` (baseline)
- [ ] `mixtral_8x7b/mixtral_performance_overview.png`
- [ ] `qwen2.5_7b/qwen_performance_overview.png` 
- [ ] `deepseek_v3_8b/deepseek_performance_overview.png`
- [ ] `gemma3_8b/gemma3_8b_performance_overview.png`
- [ ] `gemma3_27b/gemma3_27b_performance_overview.png`

## 🔄 Phase 3: LaTeX Integration
- [ ] Uncomment `\includegraphics` lines in tex file (lines 297, 304)
- [ ] Replace `--\%` placeholders with actual data in table
- [ ] Test LaTeX compilation
- [ ] Verify all image paths are correct (`../../img/...`)

## 📊 Phase 4: Advanced Analytics (NICE TO HAVE)
- [ ] `multi_model_comparison/models_performance_heatmap.png`
- [ ] Confusion matrices for each model
- [ ] Temporal analysis graphics  
- [ ] Scaling laws analysis (Gemma 8B vs 27B)

## 🎨 Phase 5: Quality Assurance
- [ ] **Color consistency** across all graphics
- [ ] **Font sizes** readable and consistent
- [ ] **Legends** clear and positioned correctly
- [ ] **File sizes** optimized but high-quality
- [ ] **Naming convention** exactly as specified

## 🚀 Phase 6: Thesis Integration
- [ ] All priority graphics created and placed
- [ ] LaTeX compiles successfully
- [ ] Captions are informative and accurate
- [ ] Cross-references work correctly
- [ ] Graphics support the narrative flow

## 📝 Notes Section
```
Write your progress notes here:
- Date started: ___________
- Models completed: _____ / 6
- Critical graphics ready: _____ / 8
- LaTeX integration status: _______
- Estimated completion: ___________
```

## 🆘 Troubleshooting
- **LaTeX won't compile**: Check image paths and ensure files exist
- **Graphics too large**: Resize to ~1MB each, maintain quality
- **Colors inconsistent**: Reference the master color palette in README_MASTER.md
- **Table formatting issues**: Check booktabs package is loaded