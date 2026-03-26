#!/bin/bash
LOGFILE="/mnt/raid/rl_gaming/RL4VLM/training_monitor.log"
echo "=== Training Monitor Started at $(date) ===" >> "$LOGFILE"
echo "Checking every 5 minutes..." >> "$LOGFILE"
echo "" >> "$LOGFILE"

while true; do
    echo "===============================================" >> "$LOGFILE"
    echo "Timestamp: $(date)" >> "$LOGFILE"
    
    # Check if process is still alive
    if ps -p 3488088 > /dev/null 2>&1; then
        ELAPSED=$(ps -o etime= -p 3488088 2>/dev/null | tr -d ' ')
        CPU=$(ps -o pcpu= -p 3488088 2>/dev/null | tr -d ' ')
        MEM_KB=$(grep VmRSS /proc/3488088/status 2>/dev/null | awk '{print $2}')
        MEM_GB=$(echo "scale=1; ${MEM_KB:-0}/1048576" | bc)
        echo "Status: RUNNING | Elapsed: $ELAPSED | CPU: ${CPU}% | RAM: ${MEM_GB}GB" >> "$LOGFILE"
        
        # GPU stats
        echo "GPU Status:" >> "$LOGFILE"
        nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader -i 0,1 >> "$LOGFILE" 2>&1
        
        # Check threads
        THREADS=$(grep Threads /proc/3488088/status 2>/dev/null | awk '{print $2}')
        echo "Worker threads: $THREADS" >> "$LOGFILE"
        
        # Check context switches (increasing = actively doing work)
        VCTX=$(grep voluntary_ctxt_switches /proc/3488088/status 2>/dev/null | awk '{print $2}')
        NVCTX=$(grep nonvoluntary_ctxt_switches /proc/3488088/status 2>/dev/null | awk '{print $2}')
        echo "Context switches: voluntary=$VCTX nonvoluntary=$NVCTX" >> "$LOGFILE"
    else
        echo "Status: PROCESS ENDED (PID 3488088 not found)" >> "$LOGFILE"
        echo "Checking exit: last few lines of dmesg for OOM:" >> "$LOGFILE"
        dmesg -T 2>/dev/null | tail -5 >> "$LOGFILE"
        echo "" >> "$LOGFILE"
        echo "GPU Memory (should be freed):" >> "$LOGFILE"
        nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader -i 0,1 >> "$LOGFILE" 2>&1
        echo "=== TRAINING FINISHED/CRASHED at $(date) ===" >> "$LOGFILE"
        break
    fi
    
    echo "" >> "$LOGFILE"
    sleep 300  # 5 minutes
done
