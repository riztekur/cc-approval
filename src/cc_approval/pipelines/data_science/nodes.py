import pandas as pd
import matplotlib
import scorecardpy as sc
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")

def split_data(data: pd.DataFrame, parameters: dict) -> tuple:
    X = data.drop(columns=parameters["target_feature"])
    y = data[parameters["target_feature"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=parameters["test_size"], random_state=parameters["random_state"], stratify=y
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series, parameters: dict) -> XGBClassifier:
    regressor = XGBClassifier(**parameters)
    regressor.fit(X_train, y_train)
    return regressor


def evaluate_model(regressor: XGBClassifier, bins, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series):
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

    # card = sc.scorecard(bins, regressor, X_train.columns)
    
    # train = X_train
    # train['IS_APPROVED'] = y_train
    # train.columns = train.columns.str.replace("_woe", "", regex=True)

    # test = X_test
    # test['IS_APPROVED'] = y_test
    # test.columns = test.columns.str.replace("_woe", "", regex=True)

    # train_score = sc.scorecard_ply(train, card, print_step=0)
    # test_score = sc.scorecard_ply(test, card, print_step=0)

    # psi = sc.perf_psi(
    #     score = {'train':train_score, 'test':test_score},
    #     label = {'train':y_train, 'test':y_test}
    # )

    return performance_summary, train_perf['pic'], test_perf['pic']