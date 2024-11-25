import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmark_utils import mimic_models_constructor, vocab_loader, logger_utils, mimic_loader, comp_utils

# Set the recommended option to avoid the downcasting warning in replace
pd.set_option('future.no_silent_downcasting', True)

parser = argparse.ArgumentParser()

# Basic options
parser.add_argument("--mode", type=str, required=True, choices=["pretrain", "test"])
parser.add_argument("--model_type", type=str, required=True, choices=["Time-series-LSTM", "Time-series-CNN", "Hybrid-LSTM", "Hybrid-CNN", "GNN"])
parser.add_argument("--verbose", action="store_true", default=False)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--total-epochs", type=int, default=20)
parser.add_argument("--lr-decay-milestones", default="60,80", type=str, help="milestones for learning rate decay")
parser.add_argument("--lr-decay-gamma", default=0.1, type=float)
parser.add_argument("--lr", default=0.01, type=float, help="learning rate")
parser.add_argument("--oversampling", action="store_true", default=False)
parser.add_argument("--restore", type=str, default=None)
parser.add_argument('--output-dir', default='run', help='path where to save')

# Architectural options
parser.add_argument("--embed_size", type=int, default=52)
parser.add_argument("--rnn_size", type=int, default=64)
parser.add_argument("--latent-size", type=int, default=152)
parser.add_argument("--rnnLayers", type=int, default=2)

# General
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--global-pruning", action="store_true", default=False)
parser.add_argument("--sl-total-epochs", type=int, default=100, help="epochs for sparsity learning")
parser.add_argument("--sl-lr", default=0.01, type=float, help="learning rate for sparsity learning")
parser.add_argument("--sl-lr-decay-milestones", default="60,80", type=str, help="milestones for sparsity learning")
parser.add_argument("--sl-reg-warmup", type=int, default=0, help="epochs for sparsity learning")
parser.add_argument("--sl-restore", type=str, default=None)
parser.add_argument("--iterative-steps", default=400, type=int)
parser.add_argument("--scheduler", type=str, default="MultiStep")

# Record Keeping
parser.add_argument("--lookback-window", type=str,  default="24hours", choices=["12hours", "24hours"])
parser.add_argument("--timebucket-size", type=str, default="1h", choices=["1h", "2h", "3h", "4h", "5h"])

args = parser.parse_args()

def eval_extended(
    model,
    test_loader,
    device=None,
    acc=True,
    ppv=True,
    sensi=True,
    tnr=True,
    npv=True,
    auroc=True,
    auroc_plot=False,
    auprc=True,
    auprc_plot=False,
    calibration=False,
    calibration_plot=False
):
    correct = 0
    total = 0
    loss = 0
    model.to(device)
    model.eval()
    
    # Accumulators for outputs and labels to calculate metrics at the end
    all_probs = []
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        num_batches = len(test_loader)
        for i, batch in enumerate(tqdm(test_loader, desc="Evaluating", unit="batch")):
            # Skip the last batch
            if i == num_batches - 1:
                continue
            
            # Parse data and move to device
            data = {key: value.to(device) for key, value in batch.items() if key != "label"}
            target = batch["label"].to(device).float().view(-1, 1)
            
            # Forward pass to get outputs
            out, logits = model(**data)
            
            # Accumulate predictions and logits for final metric calculation
            all_probs.append(out)
            all_labels.append(target)
            all_logits.append(logits)
            
            # Compute binary cross-entropy loss
            loss += F.binary_cross_entropy(out, target, reduction="sum")
            
            # Get predictions by thresholding the probabilities at 0.5
            pred = (out > 0.5).float()  # Apply threshold at 0.5 to get binary predictions
            
            # Count correct predictions
            correct += (pred == target).sum()
            total += target.size(0)

    # Compute accuracy and average loss per sample
    accuracy = (correct / total).item()
    avg_loss = (loss / total).item()
    
    # Concatenate all accumulated tensors for metric calculation
    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    all_logits = torch.cat(all_logits, dim=0)
    
    # Calculate detailed metrics using the helper function
    metrics_results = comp_utils.evaluation_utils.calculate_metrics_and_loss(
        prob=all_probs,
        labels=all_labels,
        logits=all_logits,
        device=device,
        acc=acc,
        ppv=ppv,
        sensi=sensi,
        tnr=tnr,
        npv=npv,
        auroc=auroc,
        auroc_plot=auroc_plot,
        auprc=auprc,
        auprc_plot=auprc_plot,
        calibration=calibration,
        calibration_plot=calibration_plot
    )
    
    return accuracy, avg_loss, metrics_results

def eval(model, test_loader, device=None):
    correct = 0
    total = 0
    loss = 0
    model.to(device)
    model.eval()
    with torch.no_grad():
        num_batches = len(test_loader)
        for i, batch in enumerate(tqdm(test_loader, desc="Evaluating", unit="batch")):
            # Skip the last batch
            if i == num_batches - 1:
                continue
            # Parse data and move to device
            data = {key: value.to(device) for key, value in batch.items() if key != "label"}
            #target = batch["label"].view(-1, 1).to(device)
            target = batch["label"].to(device).float().view(-1, 1)
            out, logits = model(**data)
            # Compute binary cross-entropy loss
            loss += F.binary_cross_entropy(out, target, reduction="sum")
            
            # Get predictions by thresholding the probabilities at 0.5
            pred = (out > 0.5).float()  # Apply threshold at 0.5 to get binary predictions
            
            # Count correct predictions
            correct += (pred == target).sum()
            total += target.size(0)
            
    #print("[INFO] Evaluation loop done!")
    accuracy = (correct / total).item()  # Accuracy
    avg_loss = (loss / total).item()  # Average loss per sample
    #print("[INFO] ACCURACY: ", accuracy)
    #print("[INFO] Average LOSS: ", avg_loss)
    return accuracy, avg_loss

def inference_latency_eval(model, example_inputs, device=None):
    model.to(device)
    #dummy_input = torch.randn(input_size, dtype=torch.float).to(device)

    # INIT LOGGERS
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    repetitions = 1000
    timings=np.zeros((repetitions,1))
    #GPU-WARM-UP
    for _ in range(200):
        _ = model(**example_inputs)
    # MEASURE PERFORMANCE
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(**example_inputs)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)

    return mean_syn, std_syn

def inference_throughput_eval(model, input_size, device=None):
    # https://deci.ai/blog/measure-inference-time-deep-neural-networks/
    # optimal_batch_sizes_per_model: 
    # 1. resnet56-cifar=2048*2
    # 2. mobilenetv2=2048*4
    optimal_batch_size = 2048*4 # CIFAR10 RUNS OUT OF MEMORY AT 2048*32
    model.to(device)
    dummy_input = torch.randn(optimal_batch_size, input_size[1], input_size[2], input_size[3], dtype=torch.float).to(device)

    repetitions=100
    total_time = 0
    with torch.no_grad():
        for rep in range(repetitions):
            starter, ender = torch.cuda.Event(enable_timing=True),   torch.cuda.Event(enable_timing=True)
            starter.record()
            _ = model(dummy_input)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)/1000
            total_time += curr_time
    throughput = (repetitions*optimal_batch_size)/total_time
    return throughput

def train_model(
    model,
    train_loader,
    test_loader,
    epochs, # 20
    lr,
    lr_decay_milestones, # [10, 15, 18]
    lr_decay_gamma=args.lr_decay_gamma,
    save_as=None,
    scheduler="MultiStep",
    
    # For pruning
    weight_decay=5e-4,
    save_state_dict_only=True,
    pruner=None,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #print(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=weight_decay if pruner is None else 0,
    )
    milestones = [int(ms) for ms in lr_decay_milestones.split(",")]
    if scheduler == "MultiStep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=lr_decay_gamma
        )
    elif scheduler == "Cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=0.0001 
        )

    model.to(device)
    best_acc = -1
    for epoch in range(epochs):
        model.train()
        
        #for i, batch in tqdm(enumerate(train_loader)):
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}', unit='batch') as pbar:
            for i, batch in enumerate(pbar):
                # Parse data and move to device
                data = {key: value.to(device) for key, value in batch.items() if key != "label"}
                #target = batch["label"].view(-1, 1).to(device)
                target = batch["label"].to(device).float().view(-1, 1)
                # Batch training step 
                optimizer.zero_grad()
                out, logits = model(**data)
                loss = F.binary_cross_entropy(out, target)
                loss.backward()
                optimizer.step()
                # Progress Bar (add as arguments any info's progress you want to showcase per batch iteration)
                pbar.set_postfix(batch=i, loss=loss.item(), lr=optimizer.param_groups[0]["lr"], device=device)

        # Epoch Evaluation
        model.eval()
        acc, val_loss, eval_metrics = eval_extended(model, test_loader, device=device)
        # Extract each metric from eval_metrics for easy formatting

        # Use dictionary access to retrieve each metric
        classify_loss = eval_metrics.get("BCE_Loss", "N/A")
        auc = eval_metrics.get("AUC", "N/A")
        apr = eval_metrics.get("APR", "N/A")
        accuracy = eval_metrics.get("Accuracy", "N/A")
        precision = eval_metrics.get("Precision", "N/A")
        recall = eval_metrics.get("Recall", "N/A")
        specificity = eval_metrics.get("Specificity", "N/A")
        npv = eval_metrics.get("NPV", "N/A")
        ece = eval_metrics.get("ECE", "N/A")
        mce = eval_metrics.get("MCE", "N/A")

        # Logging the extended metrics with conditional formatting
        args.logger.info(
            "Epoch {:d}/{:d}, Acc={:.4f}, Val Loss={:.4f}, lr={:.4f}\n"
            "    - BCE Loss: {}\n"
            "    - AUC: {}\n"
            "    - APR: {}\n"
            "    - Accuracy: {}\n"
            "    - Precision: {}\n"
            "    - Recall: {}\n"
            "    - Specificity: {}\n"
            "    - NPV: {}\n"
            "    - ECE: {}\n"
            "    - MCE: {}".format(
                epoch+1, epochs, acc, val_loss, optimizer.param_groups[0]["lr"],
                f"{classify_loss:.4f}" if isinstance(classify_loss, (float, int)) else classify_loss,
                f"{auc:.4f}" if isinstance(auc, (float, int)) else auc,
                f"{apr:.4f}" if isinstance(apr, (float, int)) else apr,
                f"{accuracy:.4f}" if isinstance(accuracy, (float, int)) else accuracy,
                f"{precision:.4f}" if isinstance(precision, (float, int)) else precision,
                f"{recall:.4f}" if isinstance(recall, (float, int)) else recall,
                f"{specificity:.4f}" if isinstance(specificity, (float, int)) else specificity,
                f"{npv:.4f}" if isinstance(npv, (float, int)) else npv,
                f"{ece:.4f}" if isinstance(ece, (float, int)) else ece,
                f"{mce:.4f}" if isinstance(mce, (float, int)) else mce
            )
        )

        # Save best model 
        if best_acc < acc:
            os.makedirs(args.output_dir, exist_ok=True)
            if args.mode == "pretrain":
                if save_as is None:
                    save_as = os.path.join(args.output_dir, "{}_{}_{}.pth".format(args.model_type, args.lookback_window, args.timebucket_size))
                torch.save(model.state_dict(), save_as)
            best_acc = acc
        scheduler.step()
    args.logger.info("Best Acc=%.4f" % (best_acc))

def get_model(model_type, embed_size, rnn_size):
    if model_type=='Time-series-LSTM':
        net = mimic_models_constructor.LSTMBase(args.device,
                            args.cond_vocab_size,
                            args.proc_vocab_size,
                            args.med_vocab_size,
                            args.out_vocab_size,
                            args.chart_vocab_size,
                            args.lab_vocab_size,
                            args.eth_vocab_size,args.gender_vocab_size,args.age_vocab_size,args.ins_vocab_size,
                            args.modalities,
                            embed_size=embed_size,rnn_size=rnn_size,latent_size=args.latent_size, rnnLayers=args.rnnLayers,
                            batch_size=args.batch_size) 
    elif model_type=='Time-series-CNN':
        net = mimic_models_constructor.CNNBase(args.device,
                            args.cond_vocab_size,
                            args.proc_vocab_size,
                            args.med_vocab_size,
                            args.out_vocab_size,
                            args.chart_vocab_size,
                            args.lab_vocab_size,
                            args.eth_vocab_size,args.gender_vocab_size,args.age_vocab_size,args.ins_vocab_size,
                            args.modalities,
                            embed_size=embed_size,rnn_size=rnn_size,latent_size=args.latent_size,
                            batch_size=args.batch_size) 
    elif model_type=='Hybrid-LSTM':
        net = mimic_models_constructor.LSTMBaseH(args.device,
                            args.cond_vocab_size,
                            args.proc_vocab_size,
                            args.med_vocab_size,
                            args.out_vocab_size,
                            args.chart_vocab_size,
                            args.lab_vocab_size,
                            args.eth_vocab_size,args.gender_vocab_size,args.age_vocab_size,args.ins_vocab_size,
                            args.modalities,
                            embed_size=embed_size,rnn_size=rnn_size,latent_size=args.latent_size, rnnLayers=args.rnnLayers,
                            batch_size=args.batch_size) 
    elif model_type=='Hybrid-CNN':
        net = mimic_models_constructor.CNNBaseH(args.device,
                            args.cond_vocab_size,
                            args.proc_vocab_size,
                            args.med_vocab_size,
                            args.out_vocab_size,
                            args.chart_vocab_size,
                            args.lab_vocab_size,
                            args.eth_vocab_size,args.gender_vocab_size,args.age_vocab_size,args.ins_vocab_size,
                            args.modalities,
                            embed_size=embed_size,rnn_size=rnn_size,latent_size=args.latent_size,
                            batch_size=args.batch_size) 
    elif model_type=='GNN':
        net = mimic_models_constructor.GNNBase(args.device,
                            args.cond_vocab_size,
                            args.proc_vocab_size,
                            args.med_vocab_size,
                            args.out_vocab_size,
                            args.chart_vocab_size,
                            args.lab_vocab_size,
                            args.eth_vocab_size,args.gender_vocab_size,args.age_vocab_size,args.ins_vocab_size,
                            args.modalities,
                            embed_size=embed_size,gnn_size=rnn_size,
                            batch_size=args.batch_size) 

    return net

def load_mimic_iv(labels_path, data_dir, gender_vocab, eth_vocab, ins_vocab, age_vocab, data_icu=False):
    """
    Loads the entire dataset without k-fold and returns it in PyTorch format.

    Args:
        labels_path (str): Path to the labels CSV file.
        data_dir (str): Directory where individual sample folders are stored.
        gender_vocab (dict): Vocabulary mapping for gender.
        eth_vocab (dict): Vocabulary mapping for ethnicity.
        ins_vocab (dict): Vocabulary mapping for insurance.
        age_vocab (dict): Vocabulary mapping for age.
        data_icu (bool): Whether to use ICU stay_id instead of hadm_id for labels.

    Returns:
        meds, chart, out, proc, lab, stat_df, demo_df, y_df: Tensors for each feature set and labels.
    """
    labels = pd.read_csv(labels_path)
    
    dyn_df = []
    meds = torch.zeros(size=(0,0))
    chart = torch.zeros(size=(0,0))
    proc = torch.zeros(size=(0,0))
    out = torch.zeros(size=(0,0))
    lab = torch.zeros(size=(0,0))
    stat_df = torch.zeros(size=(1,0))
    demo_df = torch.zeros(size=(1,0))
    y_df = []
    
    # Load keys for dynamic data structure
    sample_id = labels.iloc[0, 0]  # Use the first sample to get column keys
    dyn = pd.read_csv(f'{data_dir}/{sample_id}/dynamic.csv', header=[0,1])
    keys = dyn.columns.levels[0]
    
    # Prepare dyn_df placeholders for each key
    for _ in range(len(keys)):
        dyn_df.append(torch.zeros(size=(1,0)))

    # Loop through each sample to load and process data
    for _, row in labels.iterrows():
        sample = row['stay_id'] if data_icu else row['hadm_id']
        label = row['label']
        y_df.append(int(label))
        
        # Load and process dynamic data for each key
        dyn = pd.read_csv(f'{data_dir}/{sample}/dynamic.csv', header=[0,1])
        for key_idx, key_name in enumerate(keys):
            dyn_temp = torch.tensor(dyn[key_name].to_numpy()).unsqueeze(0).type(torch.LongTensor)
            dyn_df[key_idx] = torch.cat((dyn_df[key_idx], dyn_temp), 0) if dyn_df[key_idx].nelement() else dyn_temp
        
        # Load and process static data
        stat = pd.read_csv(f'{data_dir}/{sample}/static.csv', header=[0,1])['COND']
        stat = torch.tensor(stat.to_numpy())
        stat_df = torch.cat((stat_df, stat.unsqueeze(0)), 0) if stat_df[0].nelement() else stat.unsqueeze(0)
        
        # Load and process demographic data
        demo = pd.read_csv(f'{data_dir}/{sample}/demo.csv', header=0).copy()  # Explicitly copy to avoid warnings
        demo["gender"] = demo["gender"].replace(gender_vocab)
        demo["ethnicity"] = demo["ethnicity"].replace(eth_vocab)
        demo["insurance"] = demo["insurance"].replace(ins_vocab)
        demo["Age"] = demo["Age"].replace(age_vocab)
        
        demo_tensor = torch.tensor(demo[["gender", "ethnicity", "insurance", "Age"]].values)
        demo_df = torch.cat((demo_df, demo_tensor.unsqueeze(0)), 0) if demo_df[0].nelement() else demo_tensor.unsqueeze(0)

    # Map keys to respective feature variables
    for k, key_name in enumerate(keys):
        if key_name == 'MEDS':
            meds = dyn_df[k]
        elif key_name == 'CHART':
            chart = dyn_df[k]
        elif key_name == 'OUT':
            out = dyn_df[k]
        elif key_name == 'PROC':
            proc = dyn_df[k]
        elif key_name == 'LAB':
            lab = dyn_df[k]

    # Convert y_df to a tensor
    y_df = torch.tensor(y_df).type(torch.LongTensor)
    
    # Return all feature tensors and labels
    return meds, chart, out, proc, lab, stat_df, demo_df, y_df

def main():
    if args.seed is not None:
        torch.manual_seed(args.seed)

    # Logger
    if args.mode == "pretrain":
        args.output_dir = os.path.join(args.output_dir, args.lookback_window, args.timebucket_size)
        logger_name = "{}-{}-{}".format(args.model_type, args.embed_size, args.rnn_size)
        log_file = "{}/{}.txt".format(args.output_dir, logger_name)
    elif args.mode == "test":
        log_file = None
        logger_name = "{}-{}-{}".format(args.lookback_window, args.timebucket_size, args.model_type) # ADDED LINE
    args.logger = logger_utils.get_logger(logger_name, output=log_file)

    # Model & Dataset
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ### Feature Filtering ###
    diag_flag = True
    proc_flag = True
    out_flag = True
    chart_flag = True
    med_flag = True
    lab_flag = False
    if args.mode == "pretrain": 
        args.cond_vocab_size,args.proc_vocab_size,args.med_vocab_size,args.out_vocab_size,args.chart_vocab_size,args.lab_vocab_size,args.eth_vocab,args.gender_vocab,args.age_vocab,args.ins_vocab=vocab_loader.init(diag_flag,proc_flag,out_flag,chart_flag,med_flag,lab_flag)
    else:
        args.cond_vocab_size,args.proc_vocab_size,args.med_vocab_size,args.out_vocab_size,args.chart_vocab_size,args.lab_vocab_size,args.eth_vocab,args.gender_vocab,args.age_vocab,args.ins_vocab=vocab_loader.init_read(diag_flag,proc_flag,out_flag,chart_flag,med_flag,lab_flag)
    
    args.eth_vocab_size,args.gender_vocab_size,args.age_vocab_size,args.ins_vocab_size=len(args.eth_vocab),len(args.gender_vocab),len(args.age_vocab),len(args.ins_vocab)
    args.modalities = diag_flag+proc_flag+out_flag+chart_flag+med_flag+lab_flag

    ### Model Loader ###
    model = get_model(args.model_type, args.embed_size, args.rnn_size)
    args.logger.info("Model Summary: {model}".format(model=model))

    ### Data Loader ###
    train_loader, test_loader = mimic_loader.mimic_dataloaders('./data/csv/labels.csv', './data/csv/', 
                                                                args.gender_vocab, args.eth_vocab, args.ins_vocab, args.age_vocab, 
                                                                batch_size=args.batch_size, data_icu=True, train_ratio=0.8)

    for k, v in logger_utils.flatten_dict(vars(args)).items():  # print args
        args.logger.info("%s: %s" % (k, v))
    
    if args.restore is not None:
        loaded = torch.load(args.restore, map_location="cpu")
        if isinstance(loaded, nn.Module):
            model = loaded
        else:
            model.load_state_dict(loaded)
        args.logger.info("Loading model from {restore}".format(restore=args.restore))
    model = model.to(args.device)

    ######################################################
    # Training / Pruning / Testing
    # Assuming train_loader has already been created
    batch = next(iter(train_loader))  # Get a single batch from the train loader

    # Move each tensor in the batch to the specified device (e.g., GPU)
    # Here I'm assuming `batch` is a dictionary with keys "meds", "chart", etc.
    example_inputs = {key: value.to(args.device) for key, value in batch.items() if key != "label"}

    # Now, `example_inputs` contains the entire batch on the desired device
    print("Batch shapes:")
    for key, tensor in example_inputs.items():
        print(f"{key}: {tensor.shape}")
    if args.mode == "pretrain":
        ops, params = comp_utils.count_ops_and_params(
            model, example_inputs=example_inputs,
        )
        args.logger.info("Params: {:.2f} M".format(params / 1e6))
        args.logger.info("ops: {:.2f} M".format(ops / 1e6))
        train_model(
            model=model,
            epochs=args.total_epochs,
            lr=args.lr,
            lr_decay_milestones=args.lr_decay_milestones,
            train_loader=train_loader,
            test_loader=test_loader, 
            scheduler=args.scheduler
        )
    elif args.mode == "test":
        model.eval()
        ops, params = comp_utils.count_ops_and_params(
            model, example_inputs=example_inputs,
        )
        args.logger.info("Params: {:.2f} M".format(params / 1e6))
        args.logger.info("ops: {:.2f} M".format(ops / 1e6))
        """
        acc, val_loss, eval_metrics = eval_extended(model, test_loader, device=args.device)
        # Extract each metric from eval_metrics for easy formatting

        # Use dictionary access to retrieve each metric
        classify_loss = eval_metrics.get("BCE_Loss", "N/A")
        auc = eval_metrics.get("AUC", "N/A")
        apr = eval_metrics.get("APR", "N/A")
        accuracy = eval_metrics.get("Accuracy", "N/A")
        precision = eval_metrics.get("Precision", "N/A")
        recall = eval_metrics.get("Recall", "N/A")
        specificity = eval_metrics.get("Specificity", "N/A")
        npv = eval_metrics.get("NPV", "N/A")
        ece = eval_metrics.get("ECE", "N/A")
        mce = eval_metrics.get("MCE", "N/A")

        # Logging the extended metrics with conditional formatting
        args.logger.info(
            "    - Val Loss={}\n"
            "    - BCE Loss: {}\n"
            "    - AUC: {}\n"
            "    - APR: {}\n"
            "    - Accuracy: {}\n"
            "    - Precision: {}\n"
            "    - Recall: {}\n"
            "    - Specificity: {}\n"
            "    - NPV: {}\n"
            "    - ECE: {}\n"
            "    - MCE: {}".format(
                f"{val_loss:.4f}" if isinstance(val_loss, (float, int)) else val_loss,
                f"{classify_loss:.4f}" if isinstance(classify_loss, (float, int)) else classify_loss,
                f"{auc:.4f}" if isinstance(auc, (float, int)) else auc,
                f"{apr:.4f}" if isinstance(apr, (float, int)) else apr,
                f"{accuracy:.4f}" if isinstance(accuracy, (float, int)) else accuracy,
                f"{precision:.4f}" if isinstance(precision, (float, int)) else precision,
                f"{recall:.4f}" if isinstance(recall, (float, int)) else recall,
                f"{specificity:.4f}" if isinstance(specificity, (float, int)) else specificity,
                f"{npv:.4f}" if isinstance(npv, (float, int)) else npv,
                f"{ece:.4f}" if isinstance(ece, (float, int)) else ece,
                f"{mce:.4f}" if isinstance(mce, (float, int)) else mce
            )
        )
        """
        mean_sys, std_sys = inference_latency_eval(model, example_inputs, device=args.device)
        args.logger.info("Latency Mean: {:.4f} Latency STD: {:.4f}".format(mean_sys, std_sys))
        throughput = inference_throughput_eval(model, example_inputs, device=args.device)
        args.logger.info("Throughput: {:.4f} (Samples per Second)\n".format(throughput))

if __name__ == "__main__":
    main()
