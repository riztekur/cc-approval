import pandas as pd
import matplotlib
import scorecardpy as sc
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")

def split_data(data: pd.DataFrame, parameters: dict) -> tuple:
    X = data.drop(columns=parameters["target_feature"])
    y = data[parameters["target_feature"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"], stratify=y
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series, parameters: dict) -> LogisticRegression:
    regressor = LogisticRegression(**parameters)
    regressor.fit(X_train, y_train)
    return regressor


def evaluate_model(
    regressor: LogisticRegression, 
    X_train: pd.DataFrame,
    X_test: pd.DataFrame, 
    y_train: pd.Series,
    y_test: pd.Series
):
    train_pred = regressor.predict_proba(X_train)[:,1]
    test_pred = regressor.predict_proba(X_test)[:,1]

    train_perf = sc.perf_eva(y_train.squeeze(), train_pred, title = "train", plot_type=["ks", "roc"])
    test_perf = sc.perf_eva(y_test.squeeze(), test_pred, title = "test", plot_type=["ks", "roc"])

    performance_summary = {
        'KS': {
            'train': train_perf['KS'],
            'test': test_perf['KS']
        },
        'AUC': {
            'train': train_perf['AUC'],
            'test': test_perf['AUC']
        },
        'Gini': {
            'train': train_perf['Gini'],
            'test': test_perf['Gini']
        }
    }

    return performance_summary, train_perf['pic'], test_perf['pic'], 