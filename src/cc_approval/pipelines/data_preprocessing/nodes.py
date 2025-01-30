import os
import numpy as np
import scorecardpy as sc
import pandas as pd
import kagglehub

def get_dataset():
    path = kagglehub.dataset_download("rikdifos/credit-card-approval-prediction")
    application_record_path = os.path.join(path, 'application_record.csv')
    credit_record_path = os.path.join(path, 'credit_record.csv')

    raw_application_record = pd.read_csv(application_record_path)
    raw_credit_record = pd.read_csv(credit_record_path)
    return raw_application_record, raw_credit_record

def preprocess_application_record(data:pd.DataFrame) -> pd.DataFrame:
    data['OCCUPATION_TYPE'] = np.where(data['DAYS_EMPLOYED'] > 0, 'Unemployed', data['OCCUPATION_TYPE'])
    data.fillna({'OCCUPATION_TYPE':'Unknown'}, inplace=True)
    return data

def preprocess_credit_record(data:pd.DataFrame) -> pd.DataFrame:
    data['IS_BAD'] = np.where(data['STATUS'].isin(['2','3','4','5']), 1, 0)
    data = data.groupby(['ID'])['IS_BAD'].sum().reset_index()

    data['IS_APPROVED'] = np.where(data['IS_BAD'] > 1, 0, 1)
    data.drop(columns=['IS_BAD'], inplace=True)
    return data

def merge_dataset(predictors:pd.DataFrame, target:pd.DataFrame) -> pd.DataFrame:
    merged_dataset = pd.merge(left=predictors, right=target, on='ID', how='inner')
    return merged_dataset

def feature_selection(data:pd.DataFrame) -> pd.DataFrame:
    reduced_data = data.drop(columns=['ID'])
    reduced_data = sc.var_filter(data, y='IS_APPROVED')
    return reduced_data

def feature_engineering(data:pd.DataFrame) -> pd.DataFrame:
    data['AGE'] = -data['DAYS_BIRTH'] / 365
    data['YEARS_EMPLOYED'] = data['DAYS_EMPLOYED'].apply(lambda x: x / -365 if x < 0 else 0)
    data['IS_RETIREE'] = (data['DAYS_EMPLOYED'] == 365243).astype(int)

    data['FAMILY_SIZE'] = data['CNT_FAM_MEMBERS'] + data['CNT_CHILDREN']
    data['HAS_CHILDREN'] = (data['CNT_CHILDREN'] > 0).astype(int)

    data['OWNS_PROPERTY'] = ((data['FLAG_OWN_CAR'] == 'Y') | (data['FLAG_OWN_REALTY'] == 'Y')).astype(int)
    return data

def make_bins(data:pd.DataFrame) -> dict:
    bins = sc.woebin(data, y='IS_APPROVED')
    return bins

def woe_transformer(data:pd.DataFrame, bins:dict) -> pd.DataFrame:
    merged_dataset_woe = sc.woebin_ply(data, bins)
    return merged_dataset_woe