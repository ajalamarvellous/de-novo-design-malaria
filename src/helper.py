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


    smiles = df.loc[:, "SMILES"].to_list()
    mols = [Chem.MolFromSmiles(x) for x in smiles]
    fps = [fpgs.GetFingerprint(x) for x in mols]
    fps_idx = list(np.arange(len(fps)))

    n = 0
    new_points = [np.random.choice(fps_idx)]
    while len(test_dataset) < test_no:
        idx = new_points[n]
        mol_fgp = fps[idx]
        if idx in fps_idx:
            fps_idx.remove(idx)
        sim = np.array(
            DataStructs.BulkTanimotoSimilarity(mol_fgp, [fps[i] for i in fps_idx])
            )
        sim_comp = sim > similarity_threshold

        test_dataset.append(idx)
        if sum(sim_comp) > 0:
            print(f"\n Sum of values above threshhold: {sum(sim_comp)}", end=" ")
            for i, j in enumerate(sim):
                if j > similarity_threshold:
                    idx_value = fps_idx[i]
                    fps_idx.remove(idx_value)
                    new_points.append(idx_value)
        if len(new_points) == n+1:
            again = True
            while again:
                new_idx = np.random.choice(fps_idx)
                if new_idx not in new_points:
                    fps_idx.remove(new_idx)
                    new_points.append(new_idx)
                    again = False        
        n += 1
    return np.array(test_dataset), np.array(fps_idx), np.array(new_points)


def main(file_address):
    file_folder = Path(file_address).parent
    df = pd.read_csv(file_address)
    train, test, _ = split_dataset(df,
                              similarity_threshold=0.4, 
                              test_set=0.3)
    print(f"Len of splitted data: train {len(train)}, test {len(test)}")
    print(f"Fraction of splitted data: \
          train {len(train)/ len(df)}, test {len(test)/ len(df)}")
    df.iloc[train].to_csv(file_folder/ "train.csv", index=False)
    df.iloc[test].to_csv(file_folder/ "test.csv", index=False)