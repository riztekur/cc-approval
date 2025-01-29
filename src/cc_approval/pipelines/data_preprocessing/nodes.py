import numpy as np
import scorecardpy as sc
import pandas as pd

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
    reduced_data = sc.var_filter(data, y='IS_APPROVED')
    return reduced_data

def woe_transformer(data:pd.DataFrame) -> pd.DataFrame:
    bins = sc.woebin(data, y='IS_APPROVED')
    merged_dataset_woe = sc.woebin_ply(data, bins)
    return merged_dataset_woe