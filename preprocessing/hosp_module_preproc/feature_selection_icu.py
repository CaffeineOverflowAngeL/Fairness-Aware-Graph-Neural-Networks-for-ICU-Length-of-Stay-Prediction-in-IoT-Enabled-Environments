import os
import pickle
import glob
import importlib
#print(os.getcwd())
#os.chdir('../../')
#print(os.getcwd())
import pandas as pd
from scipy.stats import entropy
import numpy as np

import utils.icu_preprocess_util
from utils.icu_preprocess_util import * 
importlib.reload(utils.icu_preprocess_util)
import utils.icu_preprocess_util
from utils.icu_preprocess_util import *# module of preprocessing functions

import utils.outlier_removal
from utils.outlier_removal import *  
importlib.reload(utils.outlier_removal)
import utils.outlier_removal
from utils.outlier_removal import *

import utils.uom_conversion
from utils.uom_conversion import *  
importlib.reload(utils.uom_conversion)
import utils.uom_conversion
from utils.uom_conversion import *


if not os.path.exists("./data/features"):
    os.makedirs("./data/features")
if not os.path.exists("./data/features/chartevents"):
    os.makedirs("./data/features/chartevents")

def feature_icu(cohort_output, version_path, diag_flag=True,out_flag=True,chart_flag=True,proc_flag=True,med_flag=True):
    if diag_flag:
        print("[EXTRACTING DIAGNOSIS DATA]")
        diag = preproc_icd_module("./"+version_path+"/hosp/diagnoses_icd.csv.gz", './data/cohort/'+cohort_output+'.csv.gz', './utils/mappings/ICD9_to_ICD10_mapping.txt', map_code_colname='diagnosis_code')
        diag[['subject_id', 'hadm_id', 'stay_id', 'icd_code','root_icd10_convert','root']].to_csv("./data/features/preproc_diag_icu.csv.gz", compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED DIAGNOSIS DATA]")
    
    if out_flag:  
        print("[EXTRACTING OUPTPUT EVENTS DATA]")
        out = preproc_out("./"+version_path+"/icu/outputevents.csv.gz", './data/cohort/'+cohort_output+'.csv.gz', 'charttime', dtypes=None, usecols=None)
        out[['subject_id', 'hadm_id', 'stay_id', 'itemid', 'charttime', 'intime', 'event_time_from_admit']].to_csv("./data/features/preproc_out_icu.csv.gz", compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED OUPTPUT EVENTS DATA]")
    
    if chart_flag:
        print("[EXTRACTING CHART EVENTS DATA]")
        chart=preproc_chart("./"+version_path+"/icu/chartevents.csv.gz", './data/cohort/'+cohort_output+'.csv.gz', 'charttime', dtypes=None, usecols=['stay_id','charttime','itemid','valuenum','valueuom'])
        chart = drop_wrong_uom(chart, 0.95)
        chart[['stay_id', 'itemid','event_time_from_admit','valuenum']].to_csv("./data/features/preproc_chart_icu.csv.gz", compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED CHART EVENTS DATA]")
    
    if proc_flag:
        print("[EXTRACTING PROCEDURES DATA]")
        proc = preproc_proc("./"+version_path+"/icu/procedureevents.csv.gz", './data/cohort/'+cohort_output+'.csv.gz', 'starttime', dtypes=None, usecols=['stay_id','starttime','itemid'])
        proc[['subject_id', 'hadm_id', 'stay_id', 'itemid', 'starttime', 'intime', 'event_time_from_admit']].to_csv("./data/features/preproc_proc_icu.csv.gz", compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED PROCEDURES DATA]")
    
    if med_flag:
        print("[EXTRACTING MEDICATIONS DATA]")
        med = preproc_meds("./"+version_path+"/icu/inputevents.csv.gz", './data/cohort/'+cohort_output+'.csv.gz')
        med[['subject_id', 'hadm_id', 'stay_id', 'itemid' ,'starttime','endtime', 'start_hours_from_admit', 'stop_hours_from_admit','rate','amount','orderid']].to_csv('./data/features/preproc_med_icu.csv.gz', compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED MEDICATIONS DATA]")

def preprocess_features_icu(cohort_output, diag_flag, group_diag,chart_flag,clean_chart,impute_outlier_chart,thresh,left_thresh):
    if diag_flag:
        print("[PROCESSING DIAGNOSIS DATA]")
        diag = pd.read_csv("./data/features/preproc_diag_icu.csv.gz", compression='gzip',header=0)
        if(group_diag=='Keep both ICD-9 and ICD-10 codes'):
            diag['new_icd_code']=diag['icd_code']
        if(group_diag=='Convert ICD-9 to ICD-10 codes'):
            diag['new_icd_code']=diag['root_icd10_convert']
        if(group_diag=='Convert ICD-9 to ICD-10 and group ICD-10 codes'):
            diag['new_icd_code']=diag['root']

        diag=diag[['subject_id', 'hadm_id', 'stay_id', 'new_icd_code']].dropna()
        print("Total number of rows",diag.shape[0])
        diag.to_csv("./data/features/preproc_diag_icu.csv.gz", compression='gzip', index=False)
        print("[SUCCESSFULLY SAVED DIAGNOSIS DATA]")
        
    if chart_flag:
        if clean_chart:   
            print("[PROCESSING CHART EVENTS DATA]")
            chart = pd.read_csv("./data/features/preproc_chart_icu.csv.gz", compression='gzip',header=0)
            chart = outlier_imputation(chart, 'itemid', 'valuenum', thresh,left_thresh,impute_outlier_chart)
            
#             for i in [227441, 229357, 229358, 229360]:
#                 try:
#                     maj = chart.loc[chart.itemid == i].valueuom.value_counts().index[0]
#                     chart = chart.loc[~((chart.itemid == i) & (chart.valueuom == maj))]
#                 except IndexError:
#                     print(f"{idx} not found")
            print("Total number of rows",chart.shape[0])
            chart.to_csv("./data/features/preproc_chart_icu.csv.gz", compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED CHART EVENTS DATA]")
            
        
        
def generate_summary_icu(diag_flag,proc_flag,med_flag,out_flag,chart_flag):
    print("[GENERATING FEATURE SUMMARY]")
    if diag_flag:
        diag = pd.read_csv("./data/features/preproc_diag_icu.csv.gz", compression='gzip',header=0)
        print(diag.head(5))
        freq=diag.groupby(['stay_id','new_icd_code']).size().reset_index(name="mean_frequency")
        freq=freq.groupby(['new_icd_code'])['mean_frequency'].mean().reset_index()
        total=diag.groupby('new_icd_code').size().reset_index(name="total_count")
        summary=pd.merge(freq,total,on='new_icd_code',how='right')
        summary=summary.fillna(0)
        summary.to_csv('./data/summary/diag_summary.csv',index=False)
        summary['new_icd_code'].to_csv('./data/summary/diag_features.csv',index=False)


    if med_flag:
        med = pd.read_csv("./data/features/preproc_med_icu.csv.gz", compression='gzip',header=0)
        freq=med.groupby(['stay_id','itemid']).size().reset_index(name="mean_frequency")
        freq=freq.groupby(['itemid'])['mean_frequency'].mean().reset_index()
        
        missing=med[med['amount']==0].groupby('itemid').size().reset_index(name="missing_count")
        total=med.groupby('itemid').size().reset_index(name="total_count")
        summary=pd.merge(missing,total,on='itemid',how='right')
        summary=pd.merge(freq,summary,on='itemid',how='right')
        #summary['missing%']=100*(summary['missing_count']/summary['total_count'])
        summary=summary.fillna(0)
        summary.to_csv('./data/summary/med_summary.csv',index=False)
        summary['itemid'].to_csv('./data/summary/med_features.csv',index=False)

    
    
    if proc_flag:
        proc = pd.read_csv("./data/features/preproc_proc_icu.csv.gz", compression='gzip',header=0)
        freq=proc.groupby(['stay_id','itemid']).size().reset_index(name="mean_frequency")
        freq=freq.groupby(['itemid'])['mean_frequency'].mean().reset_index()
        total=proc.groupby('itemid').size().reset_index(name="total_count")
        summary=pd.merge(freq,total,on='itemid',how='right')
        summary=summary.fillna(0)
        summary.to_csv('./data/summary/proc_summary.csv',index=False)
        summary['itemid'].to_csv('./data/summary/proc_features.csv',index=False)

        
    if out_flag:
        out = pd.read_csv("./data/features/preproc_out_icu.csv.gz", compression='gzip',header=0)
        freq=out.groupby(['stay_id','itemid']).size().reset_index(name="mean_frequency")
        freq=freq.groupby(['itemid'])['mean_frequency'].mean().reset_index()
        total=out.groupby('itemid').size().reset_index(name="total_count")
        summary=pd.merge(freq,total,on='itemid',how='right')
        summary=summary.fillna(0)
        summary.to_csv('./data/summary/out_summary.csv',index=False)
        summary['itemid'].to_csv('./data/summary/out_features.csv',index=False)
        
    if chart_flag:
        chart=pd.read_csv("./data/features/preproc_chart_icu.csv.gz", compression='gzip',header=0)
        freq=chart.groupby(['stay_id','itemid']).size().reset_index(name="mean_frequency")
        freq=freq.groupby(['itemid'])['mean_frequency'].mean().reset_index()

        missing=chart[chart['valuenum']==0].groupby('itemid').size().reset_index(name="missing_count")
        total=chart.groupby('itemid').size().reset_index(name="total_count")
        summary=pd.merge(missing,total,on='itemid',how='right')
        summary=pd.merge(freq,summary,on='itemid',how='right')
        #summary['missing_perc']=100*(summary['missing_count']/summary['total_count'])
        #summary=summary.fillna(0)

#         final.groupby('itemid')['missing_count'].sum().reset_index()
#         final.groupby('itemid')['total_count'].sum().reset_index()
#         final.groupby('itemid')['missing%'].mean().reset_index()
        summary=summary.fillna(0)
        summary.to_csv('./data/summary/chart_summary.csv',index=False)
        summary['itemid'].to_csv('./data/summary/chart_features.csv',index=False)

    print("[SUCCESSFULLY SAVED FEATURE SUMMARY]")


def generate_expressive_summary_icu(diag_flag, proc_flag, med_flag, out_flag, chart_flag):
    print("[GENERATING EXPRESSIVE FEATURE SUMMARY]")

    def calculate_entropy(df, id_column):
        """
        Helper function to calculate entropy for a given feature column (e.g., new_icd_code or itemid).
        """
        # Total occurrences for each unique ID
        total_counts = df.groupby(id_column)['count_per_stay'].sum().reset_index()
        total_counts.columns = [id_column, "total_count"]
        
        # Calculate probability per stay
        df = df.merge(total_counts, on=id_column)
        df['probability'] = df['count_per_stay'] / df['total_count']
        
        # Entropy formula: -sum(p * log(p)) for each unique ID
        entropy_scores = df.groupby(id_column).apply(lambda x: -np.sum(x['probability'] * np.log(x['probability'])))
        #print(entropy_scores.head())

        # Reset the index and rename columns after
        entropy_scores = entropy_scores.reset_index()
        entropy_scores.columns = [id_column, 'entropy']
        
        return entropy_scores

    if diag_flag:
        print("\tAnalysing diag data...")
        diag = pd.read_csv("./data/features/preproc_diag_icu.csv.gz", compression='gzip', header=0)
        
        # Calculate mean frequency
        freq = diag.groupby(['stay_id', 'new_icd_code']).size().reset_index(name="count_per_stay")
        mean_freq = freq.groupby('new_icd_code')['count_per_stay'].mean().reset_index(name="mean_frequency")
        
        # Total counts and entropy for expressiveness
        total = diag.groupby('new_icd_code').size().reset_index(name="total_count")
        entropy_scores = calculate_entropy(freq, 'new_icd_code')
        
        # Merge all into summary
        summary = pd.merge(mean_freq, total, on='new_icd_code', how='right')
        summary = pd.merge(summary, entropy_scores, on='new_icd_code', how='right')
        summary = summary.fillna(0)
        
        # Save results
        summary.to_csv('./data/summary/exp/diag_summary.csv', index=False)
        summary['new_icd_code'].to_csv('./data/summary/exp/diag_features.csv', index=False)

    if med_flag:
        print("\tAnalysing med data...")
        med = pd.read_csv("./data/features/preproc_med_icu.csv.gz", compression='gzip', header=0)
        
        # Mean frequency and missing data percentage
        freq = med.groupby(['stay_id', 'itemid']).size().reset_index(name="count_per_stay")
        mean_freq = freq.groupby('itemid')['count_per_stay'].mean().reset_index(name="mean_frequency")
        
        missing = med[med['amount'] == 0].groupby('itemid').size().reset_index(name="missing_count")
        total = med.groupby('itemid').size().reset_index(name="total_count")
        
        # Entropy for expressiveness
        entropy_scores = calculate_entropy(freq, 'itemid')
        
        # Merge all into summary
        summary = pd.merge(mean_freq, missing, on='itemid', how='right')
        summary = pd.merge(summary, total, on='itemid', how='right')
        summary = pd.merge(summary, entropy_scores, on='itemid', how='right')
        summary = summary.fillna(0)
        
        # Save results
        summary.to_csv('./data/summary/exp/med_summary.csv', index=False)
        summary['itemid'].to_csv('./data/summary/exp/med_features.csv', index=False)

    if proc_flag:
        print("\tAnalysing proc data...")
        proc = pd.read_csv("./data/features/preproc_proc_icu.csv.gz", compression='gzip', header=0)
        
        # Mean frequency
        freq = proc.groupby(['stay_id', 'itemid']).size().reset_index(name="count_per_stay")
        mean_freq = freq.groupby('itemid')['count_per_stay'].mean().reset_index(name="mean_frequency")
        
        # Total counts and entropy for expressiveness
        total = proc.groupby('itemid').size().reset_index(name="total_count")
        entropy_scores = calculate_entropy(freq, 'itemid')
        
        # Merge all into summary
        summary = pd.merge(mean_freq, total, on='itemid', how='right')
        summary = pd.merge(summary, entropy_scores, on='itemid', how='right')
        summary = summary.fillna(0)
        
        # Save results
        summary.to_csv('./data/summary/exp/proc_summary.csv', index=False)
        summary['itemid'].to_csv('./data/summary/exp/proc_features.csv', index=False)

    if out_flag:
        print("\tAnalysing out data...")
        out = pd.read_csv("./data/features/preproc_out_icu.csv.gz", compression='gzip', header=0)
        
        # Mean frequency
        freq = out.groupby(['stay_id', 'itemid']).size().reset_index(name="count_per_stay")
        mean_freq = freq.groupby('itemid')['count_per_stay'].mean().reset_index(name="mean_frequency")
        
        # Total counts and entropy for expressiveness
        total = out.groupby('itemid').size().reset_index(name="total_count")
        entropy_scores = calculate_entropy(freq, 'itemid')
        
        # Merge all into summary
        summary = pd.merge(mean_freq, total, on='itemid', how='right')
        summary = pd.merge(summary, entropy_scores, on='itemid', how='right')
        summary = summary.fillna(0)
        
        # Save results
        summary.to_csv('./data/summary/exp/out_summary.csv', index=False)
        summary['itemid'].to_csv('./data/summary/exp/out_features.csv', index=False)

    if chart_flag:
        print("\tAnalysing chart data...")
        chart = pd.read_csv("./data/features/preproc_chart_icu.csv.gz", compression='gzip', header=0)
        
        # Mean frequency and missing data percentage
        freq = chart.groupby(['stay_id', 'itemid']).size().reset_index(name="count_per_stay")
        mean_freq = freq.groupby('itemid')['count_per_stay'].mean().reset_index(name="mean_frequency")
        
        missing = chart[chart['valuenum'] == 0].groupby('itemid').size().reset_index(name="missing_count")
        total = chart.groupby('itemid').size().reset_index(name="total_count")
        
        # Entropy for expressiveness
        entropy_scores = calculate_entropy(freq, 'itemid')
        
        # Merge all into summary
        summary = pd.merge(mean_freq, missing, on='itemid', how='right')
        summary = pd.merge(summary, total, on='itemid', how='right')
        summary = pd.merge(summary, entropy_scores, on='itemid', how='right')
        summary = summary.fillna(0)
        
        # Save results
        summary.to_csv('./data/summary/exp/chart_summary.csv', index=False)
        summary['itemid'].to_csv('./data/summary/exp/chart_features.csv', index=False)

    print("[SUCCESSFULLY SAVED FEATURE SUMMARY]")

    
def features_selection_icu(cohort_output, diag_flag,proc_flag,med_flag,out_flag,chart_flag,group_diag,group_med,group_proc,group_out,group_chart):
    if diag_flag:
        if group_diag:
            print("[FEATURE SELECTION DIAGNOSIS DATA]")
            diag = pd.read_csv("./data/features/preproc_diag_icu.csv.gz", compression='gzip',header=0)
            features=pd.read_csv("./data/summary/diag_features.csv",header=0)
            diag=diag[diag['new_icd_code'].isin(features['new_icd_code'].unique())]
        
            print("Total number of rows",diag.shape[0])
            diag.to_csv("./data/features/preproc_diag_icu.csv.gz", compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED DIAGNOSIS DATA]")
    
    if med_flag:       
        if group_med:   
            print("[FEATURE SELECTION MEDICATIONS DATA]")
            med = pd.read_csv("./data/features/preproc_med_icu.csv.gz", compression='gzip',header=0)
            features=pd.read_csv("./data/summary/med_features.csv",header=0)
            med=med[med['itemid'].isin(features['itemid'].unique())]
            print("Total number of rows",med.shape[0])
            med.to_csv('./data/features/preproc_med_icu.csv.gz', compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED MEDICATIONS DATA]")
    
    
    if proc_flag:
        if group_proc:
            print("[FEATURE SELECTION PROCEDURES DATA]")
            proc = pd.read_csv("./data/features/preproc_proc_icu.csv.gz", compression='gzip',header=0)
            features=pd.read_csv("./data/summary/proc_features.csv",header=0)
            proc=proc[proc['itemid'].isin(features['itemid'].unique())]
            print("Total number of rows",proc.shape[0])
            proc.to_csv("./data/features/preproc_proc_icu.csv.gz", compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED PROCEDURES DATA]")
        
        
    if out_flag:
        if group_out:            
            print("[FEATURE SELECTION OUTPUT EVENTS DATA]")
            out = pd.read_csv("./data/features/preproc_out_icu.csv.gz", compression='gzip',header=0)
            features=pd.read_csv("./data/summary/out_features.csv",header=0)
            out=out[out['itemid'].isin(features['itemid'].unique())]
            print("Total number of rows",out.shape[0])
            out.to_csv("./data/features/preproc_out_icu.csv.gz", compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED OUTPUT EVENTS DATA]")
            
    if chart_flag:
        if group_chart:            
            print("[FEATURE SELECTION CHART EVENTS DATA]")
            
            chart=pd.read_csv("./data/features/preproc_chart_icu.csv.gz", compression='gzip',header=0, index_col=None)
            
            features=pd.read_csv("./data/summary/chart_features.csv",header=0)
            chart=chart[chart['itemid'].isin(features['itemid'].unique())]
            print("Total number of rows",chart.shape[0])
            chart.to_csv("./data/features/preproc_chart_icu.csv.gz", compression='gzip', index=False)
            print("[SUCCESSFULLY SAVED CHART EVENTS DATA]")

def entropy_feature_values_filtering(diag_flag_exp, proc_flag_exp, med_flag_exp, out_flag_exp, chart_flag_exp, pruning_percentage=10):

    if diag_flag_exp:
        # Load the diagnosis summary and features files
        diag_sum = pd.read_csv("./data/summary/exp/diag_summary.csv")
        diag_feat = pd.read_csv("./data/summary/exp/diag_features.csv")

        # Define the percentage of most expressive features to retain
        n = pruning_percentage  # Replace with your desired percentage, e.g., 10 for the top 10%

        # Calculate the initial number of feature values
        initial_feature_count = len(diag_feat)

        # Calculate the number of top features based on the specified percentage
        top_n_count = int(len(diag_sum) * n / 100)

        # Sort diag_summary by 'entropy' in descending order to get the most expressive features
        diag_sum_sorted = diag_sum.sort_values(by='entropy', ascending=False)

        # Select the top n% of features by entropy
        top_expressive_features = diag_sum_sorted.head(top_n_count)

        # Filter diag_feat to include only the top expressive features by matching 'new_icd_code'
        filtered_diag_feat = diag_feat[diag_feat['new_icd_code'].isin(top_expressive_features['new_icd_code'])]

        # Calculate the filtered number of feature values
        filtered_feature_count = len(filtered_diag_feat)

        # Print the initial and filtered feature counts
        print(f"Initial feature count: {initial_feature_count}")
        print(f"Filtered feature count (top {n}% most expressive): {filtered_feature_count}")

        # Save the filtered features list back to CSV
        filtered_diag_feat.to_csv("./data/summary/diag_features.csv", index=False)

        print("[SUCCESS] Filtered features list saved to diag_features.csv")

    if proc_flag_exp: 
        # Load the diagnosis summary and features files
        proc_sum = pd.read_csv("./data/summary/exp/proc_summary.csv")
        proc_feat = pd.read_csv("./data/summary/exp/proc_features.csv")

        # Define the percentage of most expressive features to retain
        n = pruning_percentage  # Replace with your desired percentage, e.g., 10 for the top 10%

        # Calculate the initial number of feature values
        initial_feature_count = len(proc_feat)

        # Calculate the number of top features based on the specified percentage
        top_n_count = int(len(proc_sum) * n / 100)

        # Sort proc_summary by 'entropy' in descending order to get the most expressive features
        proc_sum_sorted = proc_sum.sort_values(by='entropy', ascending=False)

        # Select the top n% of features by entropy
        top_expressive_features = proc_sum_sorted.head(top_n_count)

        # Filter proc_feat to include only the top expressive features by matching 'itemid'
        filtered_proc_feat = proc_feat[proc_feat['itemid'].isin(top_expressive_features['itemid'])]

        # Calculate the filtered number of feature values
        filtered_feature_count = len(filtered_proc_feat)

        # Print the initial and filtered feature counts
        print(f"Initial feature count: {initial_feature_count}")
        print(f"Filtered feature count (top {n}% most expressive): {filtered_feature_count}")

        # Save the filtered features list back to CSV
        filtered_proc_feat.to_csv("./data/summary/proc_features.csv", index=False)

        print("[SUCCESS] Filtered features list saved to proc_features.csv")

    if med_flag_exp: 
        # Load the diagnosis summary and features files
        med_sum = pd.read_csv("./data/summary/exp/med_summary.csv")
        med_feat = pd.read_csv("./data/summary/exp/med_features.csv")

        # Define the percentage of most expressive features to retain
        n = pruning_percentage  # Replace with your desired percentage, e.g., 10 for the top 10%

        # Calculate the initial number of feature values
        initial_feature_count = len(med_feat)

        # Calculate the number of top features based on the specified percentage
        top_n_count = int(len(med_sum) * n / 100)

        # Sort med_summary by 'entropy' in descending order to get the most expressive features
        med_sum_sorted = med_sum.sort_values(by='entropy', ascending=False)

        # Select the top n% of features by entropy
        top_expressive_features = med_sum_sorted.head(top_n_count)

        # Filter med_feat to include only the top expressive features by matching 'itemid'
        filtered_med_feat = med_feat[med_feat['itemid'].isin(top_expressive_features['itemid'])]

        # Calculate the filtered number of feature values
        filtered_feature_count = len(filtered_med_feat)

        # Print the initial and filtered feature counts
        print(f"Initial feature count: {initial_feature_count}")
        print(f"Filtered feature count (top {n}% most expressive): {filtered_feature_count}")

        # Save the filtered features list back to CSV
        filtered_med_feat.to_csv("./data/summary/med_features.csv", index=False)

        print("[SUCCESS] Filtered features list saved to med_features.csv")

    if out_flag_exp: 
        # Load the diagnosis summary and features files
        out_sum = pd.read_csv("./data/summary/exp/out_summary.csv")
        out_feat = pd.read_csv("./data/summary/exp/out_features.csv")

        # Define the percentage of most expressive features to retain
        n = pruning_percentage  # Replace with your desired percentage, e.g., 10 for the top 10%

        # Calculate the initial number of feature values
        initial_feature_count = len(out_feat)

        # Calculate the number of top features based on the specified percentage
        top_n_count = int(len(out_sum) * n / 100)

        # Sort out_summary by 'entropy' in descending order to get the most expressive features
        out_sum_sorted = out_sum.sort_values(by='entropy', ascending=False)

        # Select the top n% of features by entropy
        top_expressive_features = out_sum_sorted.head(top_n_count)

        # Filter out_feat to include only the top expressive features by matching 'itemid'
        filtered_out_feat = out_feat[out_feat['itemid'].isin(top_expressive_features['itemid'])]

        # Calculate the filtered number of feature values
        filtered_feature_count = len(filtered_out_feat)

        # Print the initial and filtered feature counts
        print(f"Initial feature count: {initial_feature_count}")
        print(f"Filtered feature count (top {n}% most expressive): {filtered_feature_count}")

        # Save the filtered features list back to CSV
        filtered_out_feat.to_csv("./data/summary/out_features.csv", index=False)

        print("[SUCCESS] Filtered features list saved to out_features.csv")

    if chart_flag_exp: 
        # Load the diagnosis summary and features files
        chart_sum = pd.read_csv("./data/summary/exp/chart_summary.csv")
        chart_feat = pd.read_csv("./data/summary/exp/chart_features.csv")

        # Define the percentage of most expressive features to retain
        n = pruning_percentage  # Replace with your desired percentage, e.g., 10 for the top 10%

        # Calculate the initial number of feature values
        initial_feature_count = len(chart_feat)

        # Calculate the number of top features based on the specified percentage
        top_n_count = int(len(chart_sum) * n / 100)

        # Sort chart_summary by 'entropy' in descending order to get the most expressive features
        chart_sum_sorted = chart_sum.sort_values(by='entropy', ascending=False)

        # Select the top n% of features by entropy
        top_expressive_features = chart_sum_sorted.head(top_n_count)

        # Filter chart_feat to include only the top expressive features by matching 'itemid'
        filtered_chart_feat = chart_feat[chart_feat['itemid'].isin(top_expressive_features['itemid'])]

        # Calculate the filtered number of feature values
        filtered_feature_count = len(filtered_chart_feat)

        # Print the initial and filtered feature counts
        print(f"Initial feature count: {initial_feature_count}")
        print(f"Filtered feature count (top {n}% most expressive): {filtered_feature_count}")

        # Save the filtered features list back to CSV
        filtered_chart_feat.to_csv("./data/summary/chart_features.csv", index=False)

        print("[SUCCESS] Filtered features list saved to chart_features.csv")

