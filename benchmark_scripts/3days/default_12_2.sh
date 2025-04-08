#!/bin/bash

# Train 
python3 benchmark.py --mode pretrain --model_type Time-series-LSTM --dataset /mnt/d/MIMIC-IV/MIMIC-IV-Data-Pipeline/data_default_3days_12h_2h/ --lookback-window 12hours --timebucket-size 2h #--fairness

python3 benchmark.py --mode pretrain --model_type Time-series-CNN --dataset /mnt/d/MIMIC-IV/MIMIC-IV-Data-Pipeline/data_default_3days_12h_2h/ --lookback-window 12hours --timebucket-size 2h --kernel-size 6 #--fairness

python3 benchmark.py --mode pretrain --model_type Hybrid-LSTM --dataset /mnt/d/MIMIC-IV/MIMIC-IV-Data-Pipeline/data_default_3days_12h_2h/ --lookback-window 12hours --timebucket-size 2h #--fairness

python3 benchmark.py --mode pretrain --model_type Hybrid-CNN --dataset /mnt/d/MIMIC-IV/MIMIC-IV-Data-Pipeline/data_default_3days_12h_2h/ --lookback-window 12hours --timebucket-size 2h --kernel-size 6 #--fairness

python3 benchmark.py --mode pretrain --model_type GNN --dataset /mnt/d/MIMIC-IV/MIMIC-IV-Data-Pipeline/data_default_3days_12h_2h/ --lookback-window 12hours --timebucket-size 2h #--fairness

# Test
python3 benchmark.py --mode test --model_type Time-series-LSTM --restore /mnt/d/MIMIC-IV/MIMIC-IV-Data-Pipeline/run/3days/12hours/2h/Time-series-LSTM_12hours_2h.pth --dataset /mnt/d/MIMIC-IV/MIMIC-IV-Data-Pipeline/data_default_3days_12h_2h/ --lookback-window 12hours --timebucket-size 2h #--fairness

python3 benchmark.py --mode test --model_type Time-series-CNN --restore /mnt/d/MIMIC-IV/MIMIC-IV-Data-Pipeline/run/3days/12hours/2h/Time-series-CNN_12hours_2h.pth --dataset /mnt/d/MIMIC-IV/MIMIC-IV-Data-Pipeline/data_default_3days_12h_2h/ --lookback-window 12hours --timebucket-size 2h --kernel-size 6 #--fairness

python3 benchmark.py --mode test --model_type Hybrid-LSTM --restore /mnt/d/MIMIC-IV/MIMIC-IV-Data-Pipeline/run/3days/12hours/2h/Hybrid-LSTM_12hours_2h.pth --dataset /mnt/d/MIMIC-IV/MIMIC-IV-Data-Pipeline/data_default_3days_12h_2h/ --lookback-window 12hours --timebucket-size 2h #--fairness

python3 benchmark.py --mode test --model_type Hybrid-CNN --restore /mnt/d/MIMIC-IV/MIMIC-IV-Data-Pipeline/run/3days/12hours/2h/Hybrid-CNN_12hours_2h.pth --dataset /mnt/d/MIMIC-IV/MIMIC-IV-Data-Pipeline/data_default_3days_12h_2h/ --lookback-window 12hours --timebucket-size 2h --kernel-size 6 #--fairness

python3 benchmark.py --mode test --model_type GNN --restore /mnt/d/MIMIC-IV/MIMIC-IV-Data-Pipeline/run/3days/12hours/2h/GNN_12hours_2h.pth --dataset /mnt/d/MIMIC-IV/MIMIC-IV-Data-Pipeline/data_default_3days_12h_2h/ --lookback-window 12hours --timebucket-size 2h #--fairness

