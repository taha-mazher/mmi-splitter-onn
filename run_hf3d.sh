#!/bin/bash
# 3D resolution-30 set: the first 36 designs of the 2D list (23 buildable), 3 at a time on cores 9-11.
# 24-175 min (mean 58) and 0.7-1.9 GB per design, so a guard stops everything if Windows
# free memory drops below 1.5 GB; the CSVs are resumable, so rerunning this script continues where it stopped.
cd "$(dirname "$0")"
PS=/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe
free_mb() { $PS -NoProfile -Command "[int]((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1024)" | tr -d '\r'; }
echo "start $(date +%T)" >> data/hf3d_done.log
for k in 0 1 2; do
    FIRST=36 OMP_NUM_THREADS=1 taskset -c $((9 + k)) nice -n 5 $HOME/meep-env/bin/python dataset.py hf3d 3 30 2000 $k 3 > data/hf3d_$k.log 2>&1 &
done
while pgrep -f "dataset.py hf3d" > /dev/null; do
    if [ "$(free_mb)" -lt 1536 ]; then pkill -f "dataset.py hf3d"; echo "GUARD: stopped, low memory, $(date +%T)" >> data/hf3d_done.log; exit; fi
    sleep 20
done
echo "HF3D_DONE $(date +%T)" >> data/hf3d_done.log
