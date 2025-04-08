import numpy as np
from collections import Counter
from tqdm import tqdm

def calculate_entropy(feature_values):
    """
    Calculate the entropy of a feature's distribution.
    
    Args:
        feature_values (list or tensor): Values of the feature (e.g., age, ethnicity, or gender).
    
    Returns:
        float: Entropy of the feature distribution.
    """
    total_count = len(feature_values)
    value_counts = Counter(feature_values)
    probabilities = [count / total_count for count in value_counts.values()]
    entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)
    return entropy

def extract_sensitive_features(train_loader):
    """
    Extracts sensitive feature columns (gender, ethnicity, age) from train_loader.

    Args:
        train_loader (DataLoader): PyTorch DataLoader containing the training data.

    Returns:
        gender_list, ethnicity_list, age_list: Lists of extracted sensitive features.
    """
    gender_list = []
    ethnicity_list = []
    age_list = []

    # Iterate through the train_loader to extract sensitive features
    for batch in tqdm(train_loader, desc="Evaluating", unit="batch"):
        data = {key: value for key, value in batch.items() if key != "label"}
        gender_list.extend(data["demo"][:, 0].tolist())  # Assuming gender is column 0
        ethnicity_list.extend(data["demo"][:, 1].tolist())  # Assuming ethnicity is column 1
        age_list.extend(data["demo"][:, 3].tolist())  # Assuming age is column 3

    return gender_list, ethnicity_list, age_list

def calculate_feature_entropies(train_loader):
    """
    Calculates entropy for sensitive features in train_loader.

    Args:
        train_loader (DataLoader): PyTorch DataLoader containing the training data.

    Returns:
        gender_entropy, ethnicity_entropy, age_entropy: Entropy values for each feature.
    """
    # Extract sensitive features from the train_loader
    gender_list, ethnicity_list, age_list = extract_sensitive_features(train_loader)

    # Calculate entropy for each sensitive feature
    gender_entropy = calculate_entropy(gender_list)
    ethnicity_entropy = calculate_entropy(ethnicity_list)
    age_entropy = calculate_entropy(age_list)

    return gender_entropy, ethnicity_entropy, age_entropy