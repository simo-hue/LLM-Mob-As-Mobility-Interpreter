# 🛠️ DEBUG TOOLS

This directory contains utilities for analyzing and debugging the TIME prompt version of the mobility prediction system.

## 📁 Files

### **Analysis Scripts**
- `show_prompt_examples.py` - Generates real prompt examples from Verona data
- `test_simplified_time_prompt.py` - Tests simplified vs complex prompt comparison  
- `production_test.py` - Validates production readiness

### **Example Data**
- `prompt_examples/` - Generated examples of BASE, GEOM, and TIME prompts
- `simplified_time_prompt_example.txt` - Comparison of original vs simplified prompts

### **Documentation**
- `SIMPLIFIED_TIME_SUMMARY.md` - Complete implementation summary

## 🚀 Usage

### Generate Prompt Examples
```bash
cd debug_tools/
python3 show_prompt_examples.py
```

### Test Prompt Simplification
```bash
python3 test_simplified_time_prompt.py
```

### Check Production Readiness
```bash
python3 production_test.py
```

## 📊 What These Tools Do

1. **`show_prompt_examples.py`** - Analyzes prompt complexity differences:
   - BASE: ~230 chars (simple sequence)
   - GEOM: ~400 chars (+ spatial context)
   - TIME: ~800 chars (+ temporal context)

2. **`test_simplified_time_prompt.py`** - Shows 56% size reduction achieved in simplified prompts

3. **`production_test.py`** - Validates all configuration settings for production deployment

These tools were used during development to optimize the TIME version for parallel processing while maintaining temporal intelligence.