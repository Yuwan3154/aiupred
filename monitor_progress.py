#!/usr/bin/env python3
"""
Monitor the progress of parallel AIUPred processing
"""

import os
import sys
import time
import glob
import pandas as pd
from pathlib import Path

DATA_DIR = "/home/jupyter-chenxi/data/afdb"

def check_gpu_usage():
    """Check current GPU usage"""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total', 
                               '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            print("🖥️  GPU Status:")
            print("   GPU | Util% | Memory (MB)")
            print("   ----|-------|------------")
            for line in lines:
                parts = line.split(', ')
                if len(parts) >= 4:
                    gpu_id, util, mem_used, mem_total = parts[:4]
                    print(f"   {gpu_id:3s} | {util:4s}% | {mem_used:5s}/{mem_total}")
    except Exception as e:
        print(f"❌ Could not check GPU status: {e}")

def check_processing_progress():
    """Check progress of dataset processing"""
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    if not csv_files:
        print("📊 No output files found yet")
        return
    
    print(f"\n📊 Processing Progress ({len(csv_files)} datasets):")
    print("   Dataset                    | Proteins | Size (MB) | Status")
    print("   ---------------------------|----------|-----------|--------")
    
    for csv_file in sorted(csv_files):
        try:
            df = pd.read_csv(csv_file)
            size_mb = os.path.getsize(csv_file) / (1024*1024)
            name = os.path.basename(csv_file).replace('.csv', '')
            
            # Check if file was recently modified (within last 5 minutes)
            mod_time = os.path.getmtime(csv_file)
            is_recent = (time.time() - mod_time) < 300
            status = "🔄 Active" if is_recent else "✅ Done"
            
            print(f"   {name:26s} | {len(df):8d} | {size_mb:8.1f} | {status}")
            
        except Exception as e:
            name = os.path.basename(csv_file).replace('.csv', '')
            print(f"   {name:26s} | {'?':8s} | {'?':8s} | ❌ Error")

def check_running_processes():
    """Check for running Python processes"""
    try:
        import subprocess
        result = subprocess.run(['pgrep', '-f', 'process_single_dataset.py'], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            pids = [pid for pid in pids if pid]
            print(f"\n🔄 Running Processes: {len(pids)} active")
            
            # Get more details about running processes
            for pid in pids:
                try:
                    cmd_result = subprocess.run(['ps', '-p', pid, '-o', 'pid,etime,cmd'], 
                                              capture_output=True, text=True)
                    if cmd_result.returncode == 0:
                        lines = cmd_result.stdout.strip().split('\n')
                        if len(lines) > 1:
                            print(f"   {lines[1]}")
                except:
                    pass
        else:
            print("\n🔄 No active processing jobs found")
    except Exception as e:
        print(f"❌ Could not check running processes: {e}")

def main():
    """Main monitoring loop"""
    print("📊 AIUPred Processing Monitor")
    print("============================")
    print("Press Ctrl+C to exit")
    
    try:
        while True:
            os.system('clear' if os.name == 'posix' else 'cls')
            
            print("📊 AIUPred Processing Monitor")
            print("============================")
            print(f"⏰ {time.strftime('%Y-%m-%d %H:%M:%S')}")
            
            check_gpu_usage()
            check_running_processes()
            check_processing_progress()
            
            print(f"\n🔄 Refreshing in 60 seconds... (Ctrl+C to exit)")
            time.sleep(60)
            
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")

if __name__ == "__main__":
    main()