import argparse
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.linear_model import LogisticRegression
from pathlib import Path

from utils import read_file, get_mols, get_fingerprints, get_metrics
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


def train_models(train_data: str, test_data: str) -> None:

    print("Getting train data...")
    train_df = read_file(train_data)
    train_mols = get_mols(train_df)
    X_train = get_fingerprints(train_mols)
    y_train = np.where(train_df["ACTIVITY"].values == True, 1, 0)

    print("Getting test data...")
    test_df = read_file(test_data)
    test_mols = get_mols(test_df)
    X_test = get_fingerprints(test_mols)
    y_test = np.where(test_df["ACTIVITY"].values == True, 1, 0)

    models = [
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
    args = parser.parse_args()

    train_models(args.train, args.test)

if __name__ == "__main__":
    main()
    