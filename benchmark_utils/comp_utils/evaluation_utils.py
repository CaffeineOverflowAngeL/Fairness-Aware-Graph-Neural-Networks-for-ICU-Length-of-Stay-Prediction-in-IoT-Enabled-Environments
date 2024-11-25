import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn import metrics

def calculate_metrics_and_loss(
    prob,
    labels,
    logits,
    device,
    standalone=False,
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
    calibration_plot=False,
    verbose=False
):
    """
    Calculates various metrics and losses for binary classification.
    
    Parameters:
    - prob: Tensor of probabilities from the model.
    - labels: Ground truth labels as a Tensor.
    - logits: Raw model output (logits).
    - device: Device to move tensors to.
    - train: Flag to indicate if in training mode.
    - standalone: If True, converts inputs to tensors.
    - acc, ppv, sensi, tnr, npv, auroc, auroc_plot, auprc, auprc_plot, calibration, calibration_plot:
        Flags to control the calculation of specific metrics.
    
    Returns:
    - A list of calculated metric values and losses, or only the training loss if `train=True`.
    """
    # Initialize result variables with default 'NA' to indicate skipped metrics
    classify_loss, auc, apr, base, accur = 'NA', 'NA', 'NA', 'NA', 'NA'
    prec, recall, spec, npv_val, ECE, MCE = 'NA', 'NA', 'NA', 'NA', 'NA', 'NA'
    
    if standalone:
        prob = torch.tensor(prob)
        labels = torch.tensor(labels)
        logits = torch.tensor(logits)

    # Ensure tensors are of the correct type and device
    prob = prob.to(device).float()
    labels = labels.to(device).float()
    logits = logits.to(device).float()

    # Calculate Binary Cross-Entropy Loss (BCE) using positive and negative samples
    pos_ind = labels >= 0.5
    neg_ind = labels < 0.5
    pos_loss = F.binary_cross_entropy(prob[pos_ind], labels[pos_ind]) if pos_ind.any() else 0
    neg_loss = F.binary_cross_entropy(prob[neg_ind], labels[neg_ind]) if neg_ind.any() else 0
    classify_loss = pos_loss + neg_loss
    classify_loss2 = F.binary_cross_entropy_with_logits(logits, labels)

    # Detach and move to CPU for metric calculation
    labels_np = labels.cpu().numpy()
    prob_np = prob.cpu().numpy()

    # Calculate AUROC
    if auroc:
        fpr, tpr, _ = metrics.roc_curve(labels_np, prob_np)
        auc = metrics.auc(fpr, tpr)
    if auroc_plot:
        auroc_plot(labels, prob)
    
    # Calculate AUPRC
    if auprc:
        precision_curve, recall_curve, _ = metrics.precision_recall_curve(labels_np, prob_np)
        apr = metrics.auc(recall_curve, precision_curve)
        base = (labels_np == 1).mean()

    # Calculate Accuracy
    if acc:
        accur = metrics.accuracy_score(labels_np, prob_np >= 0.5)

    # Calculate Precision (PPV)
    if ppv:
        prec = metrics.precision_score(labels_np, prob_np >= 0.5)

    # Calculate Recall (Sensitivity/TPR)
    if sensi:
        tp = (prob_np >= 0.5) & (labels_np == 1)
        fn = (prob_np < 0.5) & (labels_np == 1)
        recall = tp.sum() / (tp.sum() + fn.sum()) if tp.sum() + fn.sum() > 0 else 'NA'

    # Calculate Specificity (TNR)
    if tnr:
        tn = (prob_np < 0.5) & (labels_np == 0)
        fp = (prob_np >= 0.5) & (labels_np == 0)
        spec = tn.sum() / (tn.sum() + fp.sum()) if tn.sum() + fp.sum() > 0 else 'NA'

    # Calculate Negative Predictive Value (NPV)
    if npv:
        npv_val = tn.sum() / (tn.sum() + fn.sum()) if tn.sum() + fn.sum() > 0 else 'NA'

    # Calibration Metrics (ECE, MCE) if required
    if calibration:
        if(calibration_plot):
            ECE, MCE = calb_metrics(prob,labels,True)
        else:
            ECE, MCE = calb_metrics(prob,labels,False)

    if verbose:
        # Print metrics
        print("BCE Loss: {}".format(type(classify_loss)))
        print("AU-ROC: {}".format(type(auc)))
        print("AU-PRC: {}".format(type(apr)))
        print("AU-PRC Baseline: {}".format(type(base)))
        print("Accuracy: {}".format(type(accur)))
        print("Precision: {}".format(type(prec)))
        print("Recall: {}".format(type(recall)))
        print("Specificity: {}".format(type(spec)))
        print("NPV: {}".format(type(npv_val)))
        print("ECE: {}".format(type(ECE)))
        print("MCE: {}".format(type(MCE)))
    
    return {
            "BCE_Loss": classify_loss,
            "AUC": auc,
            "APR": apr,
            "Base": base,
            "Accuracy": accur,
            "Precision": prec,
            "Recall": recall,
            "Specificity": spec,
            "NPV": npv_val,
            "ECE": ECE,
            "MCE": MCE
            }

def auroc_plot(self,label, pred):
    plt.figure(figsize=(8,6))
    plt.plot([0, 1], [0, 1],'r--')

    
    fpr, tpr, thresh = metrics.roc_curve(label, pred)
    auc = metrics.roc_auc_score(label, pred)
    plt.plot(fpr, tpr, label=f'Deep Learning Model, auc = {str(round(auc,3))}')


    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.title("AUC-ROC")
    plt.legend()
    plt.savefig('./data/output/'+"auroc_plot.png")
    #plt.show()
        
def calb_curve(self,bins,bin_accs,ECE, MCE):
    import matplotlib.patches as mpatches

    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca()

    # x/y limits
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1)

    # x/y labels
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')

    # Create grid
    ax.set_axisbelow(True) 
    ax.grid(color='gray', linestyle='dashed')

    # Error bars
    plt.bar(bins, bins,  width=0.1, alpha=0.3, edgecolor='black', color='r', hatch='\\')

    # Draw bars and identity line
    plt.bar(bins, bin_accs, width=0.1, alpha=1, edgecolor='black', color='b')
    plt.plot([0,1],[0,1], '--', color='gray', linewidth=2)

    # Equally spaced axes
    plt.gca().set_aspect('equal', adjustable='box')

    # ECE and MCE legend
    ECE_patch = mpatches.Patch(color='green', label='ECE = {:.2f}%'.format(ECE*100))
    MCE_patch = mpatches.Patch(color='red', label='MCE = {:.2f}%'.format(MCE*100))
    plt.legend(handles=[ECE_patch, MCE_patch])
    plt.savefig('./data/output/'+"callibration_plot.png")
    #plt.show()
    
def calb_bins(self,preds,labels):
    # Assign each prediction to a bin
    num_bins = 10
    bins = np.linspace(0.1, 1, num_bins)
    binned = np.digitize(preds, bins)

    # Save the accuracy, confidence and size of each bin
    bin_accs = np.zeros(num_bins)
    bin_confs = np.zeros(num_bins)
    bin_sizes = np.zeros(num_bins)

    for bin in range(num_bins):
        bin_sizes[bin] = len(preds[binned == bin])
        if bin_sizes[bin] > 0:
            bin_accs[bin] = (labels[binned==bin]).sum() / bin_sizes[bin]
            bin_confs[bin] = (preds[binned==bin]).sum() / bin_sizes[bin]

    return bins, binned, bin_accs, bin_confs, bin_sizes


def calb_metrics(self,preds,labels,curve):
    ECE = 0
    MCE = 0
    bins, _, bin_accs, bin_confs, bin_sizes = self.calb_bins(preds,labels)
    
    for i in range(len(bins)):
        abs_conf_dif = abs(bin_accs[i] - bin_confs[i])
        ECE += (bin_sizes[i] / sum(bin_sizes)) * abs_conf_dif
        MCE = max(MCE, abs_conf_dif)
    if curve:
        self.calb_curve(bins,bin_accs,ECE, MCE)
    return ECE, MCE
