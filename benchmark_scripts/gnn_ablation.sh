#!/bin/bash

python3 benchmark.py --mode pretrain --model_type GNN --dataset /mnt/d/MIMIC-IV/MIMIC-IV-Data-Pipeline/data_default_3days_12h_1h/ --lookback-window 12hours --total-epochs 10 --fairness

python3 benchmark.py --mode test --model_type GNN --batch-size 16 --restore /mnt/d/MIMIC-IV/MIMIC-IV-Data-Pipeline/run/3days/12hours/1h/GNN_12hours_1h.pth --dataset /mnt/d/MIMIC-IV/MIMIC-IV-Data-Pipeline/data_default_3days_12h_1h/ --lookback-window 12hours --fairness

python3 benchmark.py --mode test --model_type GNN --batch-size 32 --restore /mnt/d/MIMIC-IV/MIMIC-IV-Data-Pipeline/run/3days/12hours/1h/GNN_12hours_1h.pth --dataset /mnt/d/MIMIC-IV/MIMIC-IV-Data-Pipeline/data_default_3days_12h_1h/ --lookback-window 12hours --fairness

python3 benchmark.py --mode test --model_type GNN --batch-size 64 --restore /mnt/d/MIMIC-IV/MIMIC-IV-Data-Pipeline/run/3days/12hours/1h/GNN_12hours_1h.pth --dataset /mnt/d/MIMIC-IV/MIMIC-IV-Data-Pipeline/data_default_3days_12h_1h/ --lookback-window 12hours --fairness

python3 benchmark.py --mode test --model_type GNN --batch-size 128 --restore /mnt/d/MIMIC-IV/MIMIC-IV-Data-Pipeline/run/3days/12hours/1h/GNN_12hours_1h.pth --dataset /mnt/d/MIMIC-IV/MIMIC-IV-Data-Pipeline/data_default_3days_12h_1h/ --lookback-window 12hours --fairness

python3 benchmark.py --mode test --model_type GNN --batch-size 256 --restore /mnt/d/MIMIC-IV/MIMIC-IV-Data-Pipeline/run/3days/12hours/1h/GNN_12hours_1h.pth --dataset /mnt/d/MIMIC-IV/MIMIC-IV-Data-Pipeline/data_default_3days_12h_1h/ --lookback-window 12hours --fairness

python3 benchmark.py --mode test --model_type GNN --batch-size 512 --restore /mnt/d/MIMIC-IV/MIMIC-IV-Data-Pipeline/run/3days/12hours/1h/GNN_12hours_1h.pth --dataset /mnt/d/MIMIC-IV/MIMIC-IV-Data-Pipeline/data_default_3days_12h_1h/ --lookback-window 12hours --fairness
