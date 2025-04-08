#!/bin/bash

# Train 
python3 benchmark.py --mode pretrain --model_type Time-series-LSTM --dataset /mnt/d/MIMIC-IV/MIMIC-IV-Data-Pipeline/data_default_3days_12h_1h/ --lookback-window 12hours #--fairness

python3 benchmark.py --mode pretrain --model_type Time-series-CNN --dataset /mnt/d/MIMIC-IV/MIMIC-IV-Data-Pipeline/data_default_3days_12h_1h/ --lookback-window 12hours #--fairness

python3 benchmark.py --mode pretrain --model_type Hybrid-LSTM --dataset /mnt/d/MIMIC-IV/MIMIC-IV-Data-Pipeline/data_default_3days_12h_1h/ --lookback-window 12hours #--fairness

python3 benchmark.py --mode pretrain --model_type Hybrid-CNN --dataset /mnt/d/MIMIC-IV/MIMIC-IV-Data-Pipeline/data_default_3days_12h_1h/ --lookback-window 12hours #--fairness

python3 benchmark.py --mode pretrain --model_type GNN --dataset /mnt/d/MIMIC-IV/MIMIC-IV-Data-Pipeline/data_default_3days_12h_1h/ --lookback-window 12hours #--fairness

# Test
python3 benchmark.py --mode test --model_type Time-series-LSTM --restore /mnt/d/MIMIC-IV/MIMIC-IV-Data-Pipeline/run/3days/12hours/1h/Time-series-LSTM_12hours_1h.pth --dataset /mnt/d/MIMIC-IV/MIMIC-IV-Data-Pipeline/data_default_3days_12h_1h/ --lookback-window 12hours #--fairness

python3 benchmark.py --mode test --model_type Time-series-CNN --restore /mnt/d/MIMIC-IV/MIMIC-IV-Data-Pipeline/run/3days/12hours/1h/Time-series-CNN_12hours_1h.pth --dataset /mnt/d/MIMIC-IV/MIMIC-IV-Data-Pipeline/data_default_3days_12h_1h/ --lookback-window 12hours #--fairness

python3 benchmark.py --mode test --model_type Hybrid-LSTM --restore /mnt/d/MIMIC-IV/MIMIC-IV-Data-Pipeline/run/3days/12hours/1h/Hybrid-LSTM_12hours_1h.pth --dataset /mnt/d/MIMIC-IV/MIMIC-IV-Data-Pipeline/data_default_3days_12h_1h/ --lookback-window 12hours #--fairness

python3 benchmark.py --mode test --model_type Hybrid-CNN --restore /mnt/d/MIMIC-IV/MIMIC-IV-Data-Pipeline/run/3days/12hours/1h/Hybrid-CNN_12hours_1h.pth --dataset /mnt/d/MIMIC-IV/MIMIC-IV-Data-Pipeline/data_default_3days_12h_1h/ --lookback-window 12hours #--fairness

python3 benchmark.py --mode test --model_type GNN --restore /mnt/d/MIMIC-IV/MIMIC-IV-Data-Pipeline/run/3days/12hours/1h/GNN_12hours_1h.pth --dataset /mnt/d/MIMIC-IV/MIMIC-IV-Data-Pipeline/data_default_3days_12h_1h/ --lookback-window 12hours #--fairness

