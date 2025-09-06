#!/usr/bin/env python3
"""
🚀 PRODUCTION TEST SCRIPT
Verifica rapida che il sistema sia pronto per produzione
"""

import sys
from pathlib import Path

# Import del sistema
sys.path.append(str(Path(__file__).resolve().parent.parent))
from veronacard_mob_with_geom_time_parrallel import Config

def main():
    print("🚀 PRODUCTION READINESS CHECK")
    print("=" * 50)
    
    # 1. Check configurazione produzione
    print("1. 📊 CONFIGURATION CHECK:")
    print(f"   DEBUG_MODE: {Config.DEBUG_MODE} ({'❌ Should be False' if Config.DEBUG_MODE else '✅ OK'})")
    print(f"   MAX_CONCURRENT_REQUESTS: {Config.MAX_CONCURRENT_REQUESTS} ({'✅ OK' if Config.MAX_CONCURRENT_REQUESTS == 4 else '⚠️ Limited parallelism'})")
    print(f"   REQUEST_TIMEOUT: {Config.REQUEST_TIMEOUT}s ({'✅ OK' if Config.REQUEST_TIMEOUT == 300 else '⚠️ Non-standard timeout'})")
    print(f"   MODEL_NAME: {Config.MODEL_NAME}")
    
    # 2. Check file necessari
    print("\n2. 📁 FILES CHECK:")
    # Change to parent directory for relative paths  
    import os
    original_dir = os.getcwd()
    os.chdir(Path(__file__).parent.parent)
    
    required_files = [
        "data/verona/vc_site.csv",
        "data/verona/dati_2014.csv", 
        "ollama_ports.txt"
    ]
    
    for file_path in required_files:
        exists = Path(file_path).exists()
        print(f"   {file_path}: {'✅ Found' if exists else '❌ Missing'}")
    
    # 3. Check script SLURM
    print("\n3. 🖥️ SLURM SCRIPT CHECK:")
    slurm_file = Path("time_4_GPU.sh")
    if slurm_file.exists():
        content = slurm_file.read_text()
        gpu_count = content.count("--gres=gpu:4") > 0
        job_name = "time_prod" in content
        production_mode = "PRODUCTION MODE" in content
        
        print(f"   GPU allocation: {'✅ 4 GPU' if gpu_count else '❌ Wrong GPU count'}")
        print(f"   Job name: {'✅ time_prod' if job_name else '❌ Debug job name'}")
        print(f"   Production mode: {'✅ Yes' if production_mode else '❌ Debug mode'}")
    else:
        print("   time_4_GPU.sh: ❌ Missing")
    
    # 4. Preparazione finale
    print("\n4. 🎯 PRODUCTION READINESS:")
    
    issues = []
    if Config.DEBUG_MODE:
        issues.append("DEBUG_MODE is enabled")
    if not Path("data/verona/vc_site.csv").exists():
        issues.append("Missing POI data")
    if not Path("data/verona/dati_2014.csv").exists():
        issues.append("Missing visits data")
    
    if not issues:
        print("   ✅ READY FOR PRODUCTION!")
        print("\n🚀 LAUNCH COMMAND:")
        print("   sbatch time_4_GPU.sh")
        print("\n📊 MONITORING COMMANDS:")
        print("   squeue -u $USER")
        print("   tail -f mobility-qwen_time_prod-<JOBID>.out")
        print("   tail -f qwen_time_production_execution.log")
    else:
        print("   ❌ ISSUES FOUND:")
        for issue in issues:
            print(f"      - {issue}")
        print("\n🔧 FIX NEEDED BEFORE PRODUCTION")
    
    print("\n" + "=" * 50)
    
    # Restore original directory
    os.chdir(original_dir)

if __name__ == "__main__":
    main()