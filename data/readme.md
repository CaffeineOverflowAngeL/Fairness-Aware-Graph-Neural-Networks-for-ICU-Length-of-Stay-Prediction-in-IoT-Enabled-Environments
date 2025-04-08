### Files in the folder

This folder will contain following folders -
    
- **/cohort**,
  consist of cohort extracted in csv files which is output of **Block 1** in **mainPipeline.ipynb**

- **/features**,
  consist of csv files with dump of features selected in **Block 2-6** in **mainPipeline.ipynb**

- **/csv**,
  consist of csv files consisting of time-series data which is output of **Block 7** in **mainPipeline.ipynb**

- **/dict**,
  consist of json files consisting of time-series data is output of **Block 7** in **mainPipeline.ipynb**

- **/summary**,
  consist of summary of features extracted in csv files which is output of **Block 4** in **mainPipeline.ipynb**.
  It also consists of _features file which consists of list of feature codes for feature selection in **Block 5** in **mainPipeline.ipynb**.

- **/output**,
  consist of output of model training and testing in **Block 9** in **mainPipeline.ipynb**


Additionally in the context of this paper (*Fairness-Aware Graph Neural Networks for ICU Length of Stay Prediction in IoT-Enabled Environments*):
  
1. for ml_pipeline.py which is used to generate the outputs of 'Logistic Regression', 'Gradient Boosting', 'Random Forest', and 'XGBoost' please specify the folder path within model/ml_models.py. 
  
2. for benchmark.py which is used to generate all DL models outputs, specify the dataset folder path using the cli flag --dataset (Examples on this subject are presented within the **benchmark_scripts folder).