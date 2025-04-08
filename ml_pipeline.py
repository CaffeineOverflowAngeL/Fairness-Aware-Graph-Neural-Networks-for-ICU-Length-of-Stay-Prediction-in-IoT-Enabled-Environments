import pandas as pd
import numpy as np
import os
from model.ml_models import ML_models

# Define available models
models = [ 'Logistic Regression', 'Gradient Boosting', 'Random Forest', 'XGBoost'] 

# Parameters
DATA_ICU = True  # Adjust based on your dataset
CONCAT = False  # Whether to concatenate time-series features
OVERSAMPLING = False  # Whether to apply oversampling

# Initialize results dictionary
results = []

for model in models:
    print(f"Running model: {model}")
    ml_model = ML_models(DATA_ICU, model, CONCAT, OVERSAMPLING)
    
    model_performance = ml_model.ml_train()
    results.append({"Model": model, "Performance": model_performance})

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Save to Excel
output_path = os.path.join("./data_default_3days_12h_1h/output/", "model_performance.xlsx")
results_df.to_excel(output_path, index=False)

print(f"Results saved to {output_path}")