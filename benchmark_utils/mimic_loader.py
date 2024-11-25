import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
import numpy as np

class MIMICDataset(Dataset):
    def __init__(self, labels_path, data_dir, gender_vocab, eth_vocab, ins_vocab, age_vocab, data_icu=False):
        self.labels = pd.read_csv(labels_path)
        self.data_dir = data_dir
        self.gender_vocab = gender_vocab
        self.eth_vocab = eth_vocab
        self.ins_vocab = ins_vocab
        self.age_vocab = age_vocab
        self.data_icu = data_icu

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        sample_id = row['stay_id'] if self.data_icu else row['hadm_id']
        label = int(row['label'])
        
        # Load dynamic data
        dyn = pd.read_csv(f'{self.data_dir}/{sample_id}/dynamic.csv', header=[0,1])
        keys = dyn.columns.levels[0]

        dyn_data = {}
        for key in keys:
            dyn_data[key] = torch.tensor(dyn[key].to_numpy(), dtype=torch.long)

        
        # Load static data
        stat = pd.read_csv(f'{self.data_dir}/{sample_id}/static.csv', header=[0,1])['COND']
        stat_tensor = torch.tensor(stat.to_numpy(), dtype=torch.long).squeeze()
        
        # Load demographic data
        demo = pd.read_csv(f'{self.data_dir}/{sample_id}/demo.csv', header=0).copy()
        demo["gender"] = demo["gender"].replace(self.gender_vocab).astype(int)
        demo["ethnicity"] = demo["ethnicity"].replace(self.eth_vocab).astype(int)
        demo["insurance"] = demo["insurance"].replace(self.ins_vocab).astype(int)
        demo["Age"] = demo["Age"].replace(self.age_vocab).astype(int)
        demo_tensor = torch.tensor(demo[["gender", "ethnicity", "insurance", "Age"]].values, dtype=torch.long).squeeze()
        
        # Organize dynamic data by category
        meds = dyn_data.get('MEDS', torch.tensor([]))
        chart = dyn_data.get('CHART', torch.tensor([]))
        out = dyn_data.get('OUT', torch.tensor([]))
        proc = dyn_data.get('PROC', torch.tensor([]))
        lab = dyn_data.get('LAB', torch.tensor([]))

        # Return all tensors in a dictionary, along with the label
        return {
            "meds": meds,
            "chart": chart,
            "out": out,
            "proc": proc,
            "lab": lab,
            "conds": stat_tensor,
            "demo": demo_tensor,
            "label": torch.tensor(label, dtype=torch.long)
        }

# Example usage with DataLoader
def mimic_dataloaders(labels_path, data_dir, gender_vocab, eth_vocab, ins_vocab, age_vocab, batch_size=32, data_icu=False, train_ratio=0.8):
    # Create the full dataset
    full_dataset = MIMICDataset(labels_path, data_dir, gender_vocab, eth_vocab, ins_vocab, age_vocab, data_icu=data_icu)
    
    # Calculate split sizes
    train_size = int(train_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    # Split the dataset into train and test sets
    train_dataset, test_dataset = random_split(full_dataset, [train_size, test_size])
    
    # Create DataLoader for training and testing
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=4,
        drop_last=True,
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=4,
        drop_last=False,  # Keep all batches in test loader
        shuffle=False
    )
    
    return train_loader, test_loader

def load_labels(oversampling):
    labels=pd.read_csv('./data/csv/labels.csv', header=0)
    
    hids=labels.iloc[:,0]
    y=labels.iloc[:,1]
    args.logger.info("Total Samples/Positive Samples: {:d}/{:d}.".format(len(hids), y.sum()))
    #print(len(hids))
    if oversampling:
        print("=============OVERSAMPLING===============")
        oversample = RandomOverSampler(sampling_strategy='minority')
        hids=np.asarray(hids).reshape(-1,1)
        hids, y = oversample.fit_resample(hids, y)
        #print(hids.shape)
        hids=hids[:,0]
        args.logger.info("Oversampled Total Samples/Oversampled Positive Samples: {:d}/{:d}.".format(len(hids), y.sum()))
    
    return hids, y