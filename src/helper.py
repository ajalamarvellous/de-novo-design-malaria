from pathlib import Path
from typing import Tuple

import argparse
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

# Fingerprint generators
fpgs = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def split_dataset(df: pd.DataFrame, 
                  similarity_threshold: float,
                  test_size: float,
                  smiles_column: str="SMILES",
                  fpgs: object=fpgs) -> Tuple[np.array]:
    """
    Splits the data into train and test set using Lo-Hi splitter
    ref: https://arxiv.org/abs/2310.06399

    Argument(s)
    ------------
    df: pd.DataFrame \n
        dataframe containing our smiles and activity information
    similarity_threshhold: float \n
        max similarity based on Tanimoto similarity that should exist \
        between training and test set
    test_size: float \n
        min fraction of the dataset that should be in the test set
    smiles_column: str \n
        column that contains the smiles value
    fpgs: object \n
        rdkit fingerprint generator

    Return(s)
    -----------
    splits: Tuple[pd.DataFrame]
        the splitted dataset in the order (train, test)
    """

    test_no = int(test_size * len(df))
    test_dataset = []
    fpgs = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)

    smiles = df[smiles_column].to_list()
    mols = [Chem.MolFromSmiles(x) for x in smiles]
    fps = [fpgs.GetFingerprint(x) for x in mols]
    fps_idx = list(np.arange(len(fps)))

    n = 0
    new_points = [np.random.choice(fps_idx)]
    while len(test_dataset) < test_no:
        # print(new_points)
        idx = new_points[n]
        start = fps[idx]
        fps_idx.remove(idx)
        sim = np.array(DataStructs.BulkTanimotoSimilarity(start, [fps[i] for i in fps_idx]))
        sim_comp = sim > similarity_threshold

        test_dataset.append(idx)
        # print(sum(sim_comp))
        if sum(sim_comp) > 0:
            for i, j in enumerate(sim_comp):
                if j is True:
                    new_points.append(i)
                    fps_idx.pop(i)
        if len(new_points) == n+1:
            again = True
            while again:
                new_value = np.random.choice(fps_idx)
                if new_value not in new_points:
                    new_points.append(np.random.choice(fps_idx))
                    again = False        
        n += 1
    return np.array(test_dataset), np.array(fps_idx), new_points
