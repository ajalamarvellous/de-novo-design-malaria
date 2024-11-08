import time
import argparse
import logging

import mlflow
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from utils import (prepare_data, 
                   get_metrics,
                   GridSearch)

# Import necessary libraries
from sklearn.model_selection import (
    GridSearchCV, KFold, cross_val_score)

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from xgboost import XGBClassifier

from models import NNClassifier

# basic logging config
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] [%(funcName)s] [%(levelname)s]: %(message)s ",
)
logger = logging.getLogger()


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
    logger.info("All metrics logged to mlflow")


def train_models(train_data: str, test_data: str, datatype: str, SEED: int=2024) -> None:

    scaler = StandardScaler()
    logger.info(f"Getting train data using {datatype} representation...")
    X_train, y_train = prepare_data(train_data, datatype=datatype, dataset="train")
    class_weight = dict(
        zip([0,1], np.bincount(y_train)/ len(y_train))
    )
    X_train_scaled = scaler.fit_transform(X_train)
    logger.info(f"Trainset shape: {X_train.shape, y_train.shape}")

    logger.info("Getting test data...")
    X_test, y_test = prepare_data(test_data, datatype=datatype, dataset="test")
    logger.info(f"Testset shape {X_test.shape, y_test.shape}")
    X_test_scaled = scaler.transform(X_test)

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
            "max_depth": np.arange(2, 11, 2),
            "n_estimators": np.arange(50, 210, 50),
            "max_features": list(map(round, np.linspace(10, 200, 10))),
        },
        "AdaBoostClassifier": {
            "n_estimators": np.arange(50, 201, 50),
            "learning_rate": np.linspace(0.001, 1, 11),
        },
        "XGBClassifier": {
            "max_depth": np.arange(2, 11, 2),
            "n_estimators": np.arange(200, 1001, 200),
            "learning_rate": np.linspace(0.001, 1, 11),
        },
        "MLPClassifier": {
            "alpha": [0.001, 0.01, 0.1, 1, 10], 
            "batch_size": [16, 32, 64, 128], 
            "learning_rate_init": [0.001, 0.01, 0.1 ],
            "max_iter": [100, 250, 500, 1000],
        },
        "GaussianNB": {
            "var_smoothing": np.linspace(1e-10, 1, 10)   
        }
        }

    n_folds = 5 
    print("Beginning training...")
    models = [
            (LogisticRegression, {"random_state": SEED}),  
            (DecisionTreeClassifier, {"random_state": SEED}),
            (RandomForestClassifier, {"random_state":SEED}),
            (MLPClassifier, {"random_state":SEED, 
                             "early_stopping": True}),
            (AdaBoostClassifier, {"random_state":SEED,
                                  "algorithm": "SAMME"}),
            (GaussianNB, {}),
            (XGBClassifier, {"objective":"binary:logistic",
                             "random_state":SEED, "n_jobs":-1})
                             ]
  
    for model in models:
        name = str(model[0]).split(".")[-1].split("'")[0]
        logger.info(f"Training {name} now...")
        start_time = time.time()

        if name == "LogisticRegression" or name == "MLPClassifier":
            train_loop(model, params_list, X_train_scaled, y_train, 
                       X_test_scaled, y_test, name, datatype, 
                       start_time, n_folds)
        else:
            train_loop(model, params_list, X_train, y_train, 
                       X_test, y_test, name, datatype, 
                       start_time, n_folds)


def train_loop(model, params_list, X_train, y_train, X_test, y_test, name, datatype, start_time, n_folds ):
    kfold = KFold(n_splits=n_folds)
    grid_search = GridSearch(model[0], params_list[name], "accuracy", other_params=model[1])
    grid_search.fit(X_train, y_train)
    logger.info(f"Best score: {grid_search.best_score_}, ")
    for _, param in grid_search._all_params.items():
        cross_val = cross_val_score(grid_search._set_params(param),
                                    X_train,
                                    y_train,
                                    cv=kfold,
                                    n_jobs=-1)
        logger.info(f"Cross validation score {np.mean(cross_val)} +/- {np.std(cross_val)}")

        model = grid_search._set_params(param)
        model.fit(X_train, y_train)
        predictions = model.predict_proba(X_test)[:, 1]
        logger.debug(f"Shape of y_test: {y_test.shape}, shape of y_pred: {predictions.shape}")
        metrics = get_metrics(y_test, predictions, baseline=0.5)
        params = {"name": name,
                "Data type": datatype,
                "Run_time": time.time() - start_time}
        
        mlflow_logging(grid_search.best_estimator_, params=params, metrics=metrics)
        logger.info(f"Accuracy: {metrics['Accuracy']} \n \
                Precision: {metrics['Precision']} \n \
                Recall: {metrics['Recall']} \n \
                True Positives: {metrics['True_positives']} \n \
                False Positives: {metrics['False_positives']} \n \
                Total positives: {sum(y_test)} \n \
                Total positive predicted values: {sum(np.where(predictions > 0.5, 1, 0))} \n \
                Runtime: {params['Run_time']}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=str, default=None, help="Train dataset to train the model")
    parser.add_argument("--test", type=str, default=None, help="Test dataset to evaluate the model")
    parser.add_argument("--datatype", type=str, default="Descriptors", help="Type of data format (fingerprint, descriptor etc) to use to train the model")
    parser.add_argument("--mlflow_tracking_uri", type=str, default="http://localhost:8080", help="URI to dump mlflow tracking data")
    args = parser.parse_args()

    # setting mlflow parameters
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment("Malaria project")
    train_models(args.train, args.test, args.datatype)

if __name__ == "__main__":
    main()
    