
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.linear_model import LogisticRegression
from pathlib import Path

from utils import read_file, get_mols, get_fingerprints
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
from sklearn.metrics import classification_report, confusion_matrix






def train_models(train_data, test_data):

    train_df = read_file(train_data)
    train_mols = get_mols(train_df, "SMILES")
    X_train = get_fingerprints(train_mols)
    y_train = np.where(train_df["ACTIVITY"].values == True, 1, 0)

    test_df = read_file(test_data)
    test_mols = get_mols(test_df, "SMILES")
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
        print(classification_report(y_test, predictions))
        print(f"Number of positive {
            confusion_matrix(y_test, predictions)[1,1]
            } out of {sum(y_test)}")