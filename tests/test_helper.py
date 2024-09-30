import sys
import random
from pathlib import Path
import pytest

import pandas as np
import numpy as np

from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.insert(1, "/Users/madeofajala/Projects/Malaria/src/")
from helper import split_dataset

FILE_LOCATION = Path("data/Virtual-Screening.csv")
@pytest.fixture
def df(file_location=FILE_LOCATION):
    return pd.read_csv(file_location)


def test_split_data(df):
    idx = random.choices(list(range(len(df))), k=500)
    df = df.iloc[idx]
    test_frac = 0.3
    fpg = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    test, train, _ = split_dataset(df, 
                                      similarity_threshold=0.4,
                                      test_size=test_frac)
    
    train_df = df.iloc[train]
    test_df = df.iloc[test]
    
    
    assert round(len(test_df)/len(df), 1) >= test_frac, "train set smaller than specified"

    train_df["FP"] = train_df.loc[:,"SMILES"].apply(
        lambda x: fpg.GetFingerprint(Chem.MolFromSmiles(x))
        )
    test_df["FP"] = test_df.loc[:,"SMILES"].apply(
        lambda x: fpg.GetFingerprint(Chem.MolFromSmiles(x))
        )
    
    data_leakage = False

    sims = np.array([
        [DataStructs.BulkTanimotoSimilarity(x, train_df.loc[:,"FP"].to_list())]
        for x in test_df["FP"]])
    for s in sims:
        if (np.array(s) == 1).any():
            data_leakage = True
        sns.histplot(s, binwidth=0.1)
    plt.savefig("visualizations/test_sim.png")
    assert data_leakage is False, "There is leakage of the test into the training"
    n_over = sum(np.array(sims).flatten() > 0.4)
    assert n_over/len(test) < 0.1, f"Not properly separated, {n_over/len(test)} \
        with similarity value greater than threshold"