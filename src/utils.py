import os
from rdkit import Chem
from rdkit.DataStructs import cDataStructs
from rdkit.Chem import rdFingerprintGenerator
from molfeat.trans.fp import FPVecTransformer

import tqdm
import pandas as pd
import numpy as np
from itertools import product

from pathlib import Path
from typing import TypeVar, List, Tuple, Union, NewType


from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve
)


# defining new variable for our artifact locations
addressType = TypeVar("addressType", str, Path)
molType = NewType("molType", Chem.rdchem.Mol)
fngrprntType = NewType("fngrprntType", cDataStructs.ExplicitBitVect)


def read_file(file_address: addressType, dtype="csv") -> pd.DataFrame:
    """
    Read the data file in pandas and return a pandas dataframe

    Argument(s)
    ------------
    file_address: str \n
        address of the file to be opened
    
    dtype: str \n
        data type, whether csv or table

    Return(s)
    -----------
    file_dataframe: pd.DataFrame \n
        pandas dataframe of the opened file
    """
    if dtype != "table":
        df = pd.read_csv(file_address, low_memory=False)
    else:
        df = pd.read_table(file_address, low_memory=False)
    return df


def get_mols(df: pd.DataFrame, column: str="SMILES") -> List[molType]:
    """
    Return the MOL for the smiles given

    Argument(s)
    ------------
    df: pd.DataFrame \n
        DataFrame containing the dataset

    column: string \n
        The name of the column containing the smiles

    Return(s)
    -----------
    smiles: list \n
        list of the rdkit MOLs for the smiles supplied
    """
    mols = [Chem.MolFromSmiles(x) for x in df[column]]
    return mols


def get_fingerprints(
        mols: List[molType], radius: int=2, fp_size: int= 1028, as_numpy: bool=True
        ) -> Union[np.array, fngrprntType]:
    """
    Return morgan Fingerprints of the MOL supplied

    Argument(s)
    ------------
    mols: List[molType] \n
        List of mols supplied
    
    radius: int \n
        parameter for the fingerprint generator to specify the radius for the 
        fingerprint to be returned
    
    fp_size: int \n
        len of the fingerprint to be returned

    as_numpy: bool \n
        whether to return the fingerprints as numpy

    Return(s)
    -----------
    fingerprints: np.array \n
        array of the fingerprints for the rdkit MOLs supplied
    """
    fingerprnt_gen = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=fp_size
        )
    if as_numpy:
        fingerprints = np.array([fingerprnt_gen.GetFingerprintAsNumPy(x) for x in mols])
    else:
        fingerprints = fingerprnt_gen.GetFingerprints(mols)
    return fingerprints
    

def prepare_data(
        file_name: Union[str, Path], 
        datatype: str,
        dataset: str,
        smiles_col: str="SMILES", 
        target_col: str="ACTIVITY",
        ) -> Tuple[np.array]:
    """
    Combination of read_file, get_mols and get_fingerprint plus y_data preprocssing
    
    Argument(s)
    ------------
    file_name: Union[str, Path] \n
        file address of the dataset to be returned    

    datatype: str \n
        type of output desired, options are
        [Fingerprints, Descriptors]

    dataset: str \n
        the dataset passed in whether test or train

    smiles_col: str \n
        name of the smile column    

    target_col: str \n
        name of the target column
    
    Return(s)
    -----------
    dataset: Tuple[np.array] \n
        Tuple of the processed dataset in the order (X, y)

    Sequence 
    ---------
    read_file -> get_mols -> get_fingerprints -> preprocess_y -> final_dataset
    """

    df = read_file(file_name)
    if datatype == "Fingerprints":
        mols = get_mols(df, smiles_col)
        X = get_fingerprints(mols)
    elif datatype == "Descriptors":
        X = get_descriptors(df, dataset, smiles_col)
    y = np.where(df[target_col].values == True, 1, 0)
    return (X, y)


def get_metrics(y_test: np.array, y_pred: np.array, baseline: float) -> dict:
    """
    Get and return basic metrics to evaluate the model performance
    Specifically return: Accuracy, precision, recall, n_true_pos, n_false_pos
    
    Argument(s)
    ------------
    y_tests: np.array \n
        value of ground truths 

    y_pred: np.array \n
        value of the predicted values 

    Return(s)
    -----------
    metric: dict \n
        dictionary of the metrics
    """
    predicted_class = np.where(y_pred > baseline, 1, 0)
    metrics = {}

    precision, recall, thresshold = precision_recall_curve(y_test, y_pred)
    fpr, tpr, threshold = roc_curve(y_test, y_pred)

    metrics["Accuracy"] = accuracy_score(y_test, predicted_class)
    metrics["Precision"] = precision_score(y_test, predicted_class)
    metrics["Recall"] = recall_score(y_test, predicted_class)
    metrics["True_positives"] = confusion_matrix(y_test, predicted_class)[1, 1]
    metrics["False_positives"] = confusion_matrix(y_test, predicted_class)[0, 1]
    metrics["ROC_AUC_Score"] = roc_auc_score(y_test, y_pred)
    metrics["Precision_recall_curve"] = {
        "precision": precision,
        "recall": recall,
    }
    metrics["roc_curve"]= {
        "fpr": fpr,
        "tpr": tpr,
    }
    return metrics


class GridSearch:
    def __init__(self, model, params_grid, scoring, other_params, top_n=5):
        self.model = model
        self._params_grid = self._get_param_combination(params_grid)
        self._scoring = scoring
        self._top_n = top_n
        self._other_params = other_params
        self._all_params = {}
        self._best_params = {}
        self.best_score_ = 0
        self.best_estimator_ = None

    def fit(self, X, y):
        print(f"Training {len(self._params_grid)} cobinations")
        for param in tqdm.tqdm(self._params_grid):
            if self._other_params != {}:
                param.update(self._other_params)
            model = self._set_params(param)
            model.fit(X, y)
            preds = model.predict(X)
            metric = self._get_metric(y, preds)
            self._all_params[metric] = param
            if metric > self.best_score_:
                self.best_score_ = metric
                self.best_estimator_ = model
                self._best_params = param
        self._sort_top_params()

    def _get_param_combination(self,params):
        keys = list(params.keys())
        combinations = list(product(*params.values()))
        c = [dict((k,v) for k,v in zip(keys,values)) for values in combinations]
        return c

    def _set_params(self, params):
        return self.model(**params)
    
    def _get_metric(self, y_true, y_pred):
        if self._scoring == "accuracy":
            return accuracy_score(y_true, y_pred)
        elif self._scoring == "precision":
            return precision_score(y_true, y_pred)
        elif self._scoring == "recall":
            return accuracy_score(y_true, y_pred)
        elif self._scoring == "roc_auc":
            return roc_auc_score(y_true, y_pred)
        
    def _sort_top_params(self):
        all_keys = list(self._all_params.keys())
        top_n = np.sort(all_keys)[-self._top_n:]
        for key in all_keys:
            if key not in top_n:
                del self._all_params[key]


def get_descriptors(df: pd.DataFrame, dataset: str, column: str="SMILES") -> np.array:
    """
    Return the 2D descriptors for the smiles given

    Argument(s)
    ------------
    df: pd.DataFrame \n
        DataFrame containing the dataset

    dataset: str \n
        description of the data set, whether train or test

    column: string \n
        The name of the column containing the smiles

    Return(s)
    -----------
    smiles: list \n
        list of the rdkit MOLs for the smiles supplied
    """
    smi_list = df[column].values
    file_name = Path(__file__).parents[1]/ f"data/{dataset}.npy"
    if file_name.exists():
        descriptors = np.load(file_name)
    else:
        transformer = FPVecTransformer(kind="desc2D", dtype=float)
        descriptors = transformer(smi_list)
        descriptors = np.nan_to_num(descriptors)
        np.save(file_name, descriptors)
        os.system("clear")
    return descriptors
