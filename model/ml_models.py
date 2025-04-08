import pandas as pd
import numpy as np
import pickle
import torch
import random
import os
import importlib
import sys
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report

sys.path.append(os.path.dirname(os.path.abspath(__file__)))  # Add current folder to path
import evaluation
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + './../..')

class ML_models():
    def __init__(self, data_icu, model_type, concat, oversampling):
        self.data_icu = data_icu
        self.model_type = model_type
        self.concat = concat
        self.oversampling = oversampling
        self.loss = evaluation.Loss('cpu', True, True, True, True, True, True, True, True, True, True, True)
        self.ml_train()

    def create_kfolds(self):
        labels = pd.read_csv('./data_default_3days_12h_1h/csv/labels.csv', header=0)
        hids = labels.iloc[:, 0]
        y = labels.iloc[:, 1]

        if self.oversampling:
            oversample = RandomOverSampler(sampling_strategy='minority')
            hids, y = oversample.fit_resample(np.asarray(hids).reshape(-1, 1), y)
            hids = hids[:, 0]

        ids = list(range(len(hids)))
        batch_size = len(ids) // 5
        random.shuffle(ids)
        
        test_hids = np.array(hids[ids[:batch_size]])  # First fold as test set
        train_hids = np.array(hids[ids[batch_size:]])  # Remaining as training set
        
        return train_hids, test_hids

    def ml_train(self):
        train_hids, test_hids = self.create_kfolds()
        labels = pd.read_csv('./data_default_3days_12h_1h/csv/labels.csv', header=0)
        
        print(f"Training on {len(train_hids)} samples, Testing on {len(test_hids)} samples")

        X_train, Y_train = self.getXY(train_hids, labels)
        X_test, Y_test = self.getXY(test_hids, labels)

        # Encode categorical variables
        encoders = {col: LabelEncoder().fit(X_train[col]) for col in ['gender', 'insurance', 'ethnicity']}
        for col, encoder in encoders.items():
            X_train[col] = encoder.transform(X_train[col])
            X_test[col] = encoder.transform(X_test[col])
        
        # Ensure age is a float
        X_train['Age'] = X_train['Age'].astype(float)
        X_test['Age'] = X_test['Age'].astype(float)
        
        X_train.columns = X_train.columns.astype(str)
        X_test.columns = X_test.columns.astype(str)

        test_scores = self.train_model(X_train, Y_train, X_test, Y_test)
        print("Test Score:", test_scores)

    def train_model(self, X_train, Y_train, X_test, Y_test):
        print("===============MODEL TRAINING===============")
        model = None
        if self.model_type == 'Gradient Boosting':
            model = HistGradientBoostingClassifier().fit(X_train, Y_train)
        elif self.model_type == 'Logistic Regression':
            X_train = pd.get_dummies(X_train, columns=['gender', 'insurance', 'ethnicity'])
            X_test = pd.get_dummies(X_test, columns=['gender', 'insurance', 'ethnicity'])
            model = LogisticRegression().fit(X_train, Y_train)
        elif self.model_type == 'Random Forest':
            X_train = pd.get_dummies(X_train, columns=['gender', 'insurance', 'ethnicity'])
            X_test = pd.get_dummies(X_test, columns=['gender', 'insurance', 'ethnicity'])
            model = RandomForestClassifier().fit(X_train, Y_train)
        elif self.model_type == 'XGBoost':
            X_train = pd.get_dummies(X_train, columns=['gender', 'insurance', 'ethnicity'])
            X_test = pd.get_dummies(X_test, columns=['gender', 'insurance', 'ethnicity'])
            model = xgb.XGBClassifier(objective="binary:logistic").fit(X_train, Y_train)
        
        prob = model.predict_proba(X_test)[:, 1]
        logits = np.log2(prob / (1 - prob))
        test_scores = self.loss(prob, np.asarray(Y_test), logits, False, True)
        return test_scores

    def getXY(self, ids, labels):
        X_list, y_list = [], []
        for sample in tqdm(ids, desc="Processing samples"):
            y = labels.loc[labels.iloc[:, 0] == sample, 'label'].values[0]
            y_list.append(y)
            dyn = pd.read_csv(f'./data_default_3days_12h_1h/csv/{sample}/dynamic.csv', header=[0, 1])
            demo = pd.read_csv(f'./data_default_3days_12h_1h/csv/{sample}/demo.csv', header=0)
            X_list.append(pd.concat([dyn.mean().to_frame().T, demo], axis=1))
        return pd.concat(X_list, ignore_index=True), pd.Series(y_list)

    def save_output(self,labels,prob,logits):
        
        output_df=pd.DataFrame()
        output_df['Labels']=labels.values
        output_df['Prob']=prob
        output_df['Logits']=np.asarray(logits)
        output_df['ethnicity']=list(self.test_data['ethnicity'])
        output_df['gender']=list(self.test_data['gender'])
        output_df['age']=list(self.test_data['Age'])
        output_df['insurance']=list(self.test_data['insurance'])
        
        with open('./data_default_3days_12h_1h/output/'+'outputDict', 'wb') as fp:
               pickle.dump(output_df, fp)
        
    
    def save_outputImp(self,labels,prob,logits,importance,features):
        
        output_df=pd.DataFrame()
        output_df['Labels']=labels.values
        output_df['Prob']=prob
        output_df['Logits']=np.asarray(logits)
        output_df['ethnicity']=list(self.test_data['ethnicity'])
        output_df['gender']=list(self.test_data['gender'])
        output_df['age']=list(self.test_data['Age'])
        output_df['insurance']=list(self.test_data['insurance'])
        
        with open('./data_default_3days_12h_1h/output/'+'outputDict', 'wb') as fp:
               pickle.dump(output_df, fp)
        
        imp_df=pd.DataFrame()
        imp_df['imp']=importance
        imp_df['feature']=features
        imp_df.to_csv('./data_default_3days_12h_1h/output/'+'feature_importance.csv', index=False)
                
                

