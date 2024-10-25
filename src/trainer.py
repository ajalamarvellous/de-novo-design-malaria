import os
import argparse

import mlflow
from sklearn.linear_model import LogisticRegression
from pathlib import Path

from utils import prepare_data, get_metrics
# Import necessary libraries
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC


def mlflow_logging(name: str, class_weighted: bool, metrics: dict) -> None:
    """
    Function to log to mlflow
    
    Argument(s)
    ------------
    name: str \n
        Name of the model to log 

    class_weighted: bool \n
        information whether the model was passed with class_weigh

    metrics: dict \n
        the metrics to be logged   
    """
    with mlflow.start_run():
        mlflow.log_param("model_name", name)
        mlflow.log_param("class_weighted", class_weighted)

        mlflow.log_metric("Accuracy", metrics["Accuracy"])
        mlflow.log_metric("Precision", metrics["Precision"])
        mlflow.log_metric("Recall", metrics["Recall"])
        mlflow.log_metric("True Positives", metrics["True_positives"])
        mlflow.log_metric("False Positives", metrics["False_positives"])


def train_models(train_data: str, test_data: str) -> None:

    print("Getting train data...")
    X_train, y_train = prepare_data(train_data)

    print("Getting test data...")
    X_test, y_test = prepare_data(test_data)

    models = [
        LogisticRegression(),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1),
        AdaBoostClassifier(),
        GaussianNB(),
        SVC(kernel='linear'),
        SVC(kernel='rbf'),
        SVC(kernel='sigmoid')
    ]
    n_folds = 5 

    print("Beginning training...")
    # Evaluate each model in turn
    results = []
    for model in models:
        name = model.__str__().split("(")[0]
        print(f"Training {name} now...")
        kfold = KFold(n_splits=n_folds)
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring="accuracy")
        results.append(cv_results)
        print(f"Cross validation ({n_folds}) done \n  \
            Accuracy: {cv_results.mean()} +/- {cv_results.std()}")

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        metrics = get_metrics(y_test, predictions)
        
        mlflow_logging(name=name, class_weighted=False, metrics=metrics)
        print(f"Accuracy: {metrics['Accuracy']} \n \
                Precision: {metrics['Precision']} \n \
                Recall: {metrics['Recall']} \n \
                True Positives: {metrics['True_positives']} \n \
                False Positives: {metrics['False_positives']} \n \
                Total positives: {sum(y_test)}")
        

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=str, default=None, help="Train dataset to train the model")
    parser.add_argument("--test", type=str, default=None, help="Test dataset to evaluate the model")
    parser.add_argument("--mlflow_tracking_uri", type=str, default="http://localhost:8080", help="URI to dump mlflow tracking data")
    args = parser.parse_args()

    # setting mlflow parameters
    print("Path exists: ", os.path.exists(args.mlflow_tracking_uri))
    mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    mlflow.set_experiment("Malaria project")
    train_models(args.train, args.test)

if __name__ == "__main__":
    main()
    