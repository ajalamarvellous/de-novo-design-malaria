import time
import argparse

import mlflow
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from utils import prepare_data, get_metrics

# Import necessary libraries
from sklearn.model_selection import (
    GridSearchCV, KFold, cross_val_score)

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier

from models import NNClassifier

def mlflow_logging(model, params: dict, metrics: dict) -> None:
    """
    Function to log to mlflow
    
    Argument(s)
    ------------
    model: str \n
        the model whose parameters are to be logged

    class_weighted: bool \n
        information whether the model was passed with class_weigh

    metrics: dict \n
        the metrics to be logged   
    """
    Precision_recall_curve = metrics.pop("Precision_recall_curve")
    fig1, ax1 = plt.subplots()
    ax1.plot(Precision_recall_curve["precision"],
            Precision_recall_curve["recall"])
    plt.title("Precison recall curve")

    roc_curve = metrics.pop("roc_curve")
    fig2, ax2 = plt.subplots()
    ax2.plot(roc_curve["fpr"],
            roc_curve["tpr"])
    plt.title("ROC curve")

    with mlflow.start_run():
        mlflow.log_params(params)
        mlflow.log_params(model.get_params())
        mlflow.log_metrics(metrics)
        mlflow.log_figure(fig1, "precision_recall_graph.png")
        mlflow.log_figure(fig2, "ROC_Curve.png")

def train_models(train_data: str, test_data: str, SEED: int=2024) -> None:

    print("Getting train data...")
    X_train, y_train = prepare_data(train_data)
    class_weight = dict(
        zip([0,1], np.bincount(y_train)/ len(y_train))
    )
    print(X_train.shape, y_train.shape)

    print("Getting test data...")
    X_test, y_test = prepare_data(test_data)

    params_list = {
        "LogisticRegression": {
            "class_weight": [None, class_weight],
            "C": np.linspace(0.001, 1, 11),
        },
        "DecisionTreeClassifier": {
            "class_weight": [None, class_weight],
            "max_depth": np.arange(1, 10),
            "max_features": list(map(round, np.linspace(10, 200, 10))),
        },
        "RandomForestClassifier": {
            "class_weight": [None, class_weight],
            "max_depth": np.arange(1, 10),
            "n_estimators": np.arange(20, 200, 20),
            "max_features": list(map(round, np.linspace(10, 200, 10))),
        },
        "AdaBoostClassifier": {
            "class_weight": [None, class_weight],
            "max_depth": np.arange(1, 10),
            "n_estimators": np.arange(20, 200, 20),
            "learning_rate": np.linspace(0.001, 1, 11),
        },
        "XGBClassifier": {
            "class_weight": [None, class_weight],
            "max_depth": np.arange(1, 10),
            "n_estimators": np.arange(100, 1000, 100),
            "learning_rate": np.linspace(0.001, 1, 11),
        },
        "MLPClassifier": {
            "class_weight": [None, class_weight],
            "alpha": np.linspace(0.0001, 1, 10), 
            "batch_size": [16, 32, 64, 128], 
            "learning_rate_init": np.linspace(0.001, 0.1, 5),
            "max_iter": np.arange(100, 500, 100),
        },
        "GaussianNB": {
            "class_weight": [None, class_weight],
            "var_smoothing": np.linspace(1e-10, 1, 10)   
        }
        }

    models = [
       # NNClassifier(
        # input_size=X_train.shape[1], hidden_size=[512],
        # batch_size=10, epochs=10) ]
        LogisticRegression(random_state=SEED),
        DecisionTreeClassifier(random_state=SEED),
        RandomForestClassifier(random_state=SEED),
        MLPClassifier(random_state=SEED),
        AdaBoostClassifier(random_state=SEED),
        GaussianNB(),
        XGBClassifier(objective="binary:logistic", random_state=SEED)
    ]
   
    n_folds = 5 
   
    print("Beginning training...")
    # Evaluate each model in turn
    for model in models:
        name = model.__str__().split("(")[0]
        params = params_list[name]
        print(f"Training {name} now...")
        start_time = time.time()

        kfold = KFold(n_splits=n_folds)
        grid_search = GridSearchCV(model, params, scoring="accuracy", cv=kfold, n_jobs=-1)
        grid_search.fit(X_train, y_train)
        print(f"Best score: {grid_search.best_score_}")

        predictions = grid_search.predict_proba(X_test)[:, 1]
        metrics = get_metrics(y_test, predictions, baseline=0.5)
        params = {"name": name,
                  "Data type": "Fingerprints",
                  "Run_time": time.time() - start_time}
        
        mlflow_logging(grid_search.best_estimator_, params=params, metrics=metrics)
        print(f"Accuracy: {metrics['Accuracy']} \n \
                Precision: {metrics['Precision']} \n \
                Recall: {metrics['Recall']} \n \
                True Positives: {metrics['True_positives']} \n \
                False Positives: {metrics['False_positives']} \n \
                Total positives: {sum(y_test)} \n \
                Total positive predicted values: {sum(predictions)} \n \
                Runtime: {params['Run_time']}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=str, default=None, help="Train dataset to train the model")
    parser.add_argument("--test", type=str, default=None, help="Test dataset to evaluate the model")
    parser.add_argument("--mlflow_tracking_uri", type=str, default="http://localhost:8080", help="URI to dump mlflow tracking data")
    args = parser.parse_args()

    # setting mlflow parameters
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment("Malaria project")
    train_models(args.train, args.test)

if __name__ == "__main__":
    main()
    