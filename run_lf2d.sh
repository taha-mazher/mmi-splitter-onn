#!/bin/bash
# 2,000-design 2D dataset (resolution 20). Resumable.
cd "$(dirname "$0")"
for k in 0 1; do
    OMP_NUM_THREADS=1 taskset -c $((6 + k)) nice -n 5 $HOME/meep-env/bin/python dataset.py lf2d 2 20 2000 $k 2 > data/lf2d_$k.log 2>&1 &
done
wait
echo "LF2D_DONE $(date +%T)" >> data/lf2d_done.log
