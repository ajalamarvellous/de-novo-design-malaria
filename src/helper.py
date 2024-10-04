from pathlib import Path
from typing import Tuple
import logging

import argparse
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

# Fingerprint generators
fpgs = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

# basic logging config
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(funcName)s[%(levelname)s]: %(message)s ",
)
logger = logging.getLogger()


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
        Max. fraction of the dataset that should be in the test set
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
    parsed = []
    fpgs = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048)

    smiles = df.loc[:, smiles_column].to_list()
    mols = [Chem.MolFromSmiles(x) for x in smiles]
    logger.info("Mol retrieved from all mols")

    fps = [fpgs.GetFingerprint(x) for x in mols]
    fps_idx = list(np.arange(len(fps)))
    logger.info("All fingerprints retrieved succesfully")

    n = 0
    others = [np.random.choice(fps_idx)]
    p = len(fps_idx) / len(df)
    # At the end of the splitting we will have 3 lists, test_dataset, train_dataset
    # and others where fps_idx will be our test_set and parsed our train dataset
    # thus, stop when the fps_idx fraction is less than specified test_set fraction
    while p > test_size:
        # select the first value in the others list
        idx = others[n]
        # get the fingerprint of the selected mol
        mol_fgp = fps[idx]
        # if the mol is still in the original list of all, remove them
        if idx in fps_idx:
            fps_idx.remove(idx)
        # get the Tanimoto similarity between selected mol and every other mol
        sim = np.array(
            DataStructs.BulkTanimotoSimilarity(mol_fgp, [fps[i] for i in fps_idx])
            )
        # compare the similarities to the set threshold
        sim_comp = sim > similarity_threshold

        # add compared mol to list of parsed mols
        parsed.append(idx)
        # check if there's any similarity greater than threshhold and count no if any
        n_greater = sum(sim_comp)  
        sim_comp = list(sim_comp)   
        if n_greater > 0:
            logger.debug(f"{n_greater} mols found similar to selected mol idc{idx}")
            for i in range(n_greater):
                # if the similarity value is greater than threshold, add the index of mol 
                # to others that will be compared while removing the idx from the original list of mols
                # list().index returns the index of first value that match, why we remove each after it 
                # has been found
                idx_ = sim_comp.index(True)
                idx_value = fps_idx[idx_]
                fps_idx.remove(idx_value)
                others.append(idx_value)
                sim_comp.pop(idx)
            # set the new list of indices to the one with mols greater than threshhold removed
            # fps_idx = fps_idx_copy
        # if no mol has similarity greater than threshhold and no new mol has been added to others
        if len(others) == n+1:
            # while mol has not been previously added to others, randomly select one new compound
            new_idx = np.random.choice(fps_idx)
            fps_idx.remove(new_idx)
            others.append(new_idx)
        n += 1
        p = len(fps_idx) / len(df)
        # print result at every 500 mols checkpoints
        if n % 500 == 0:
            logger.info(f"{n}:{p} reached...")
    return np.array(parsed), np.array(fps_idx), np.array(others)


def get_datasplits(file_address):
    file_folder = Path(file_address).parent
    df = pd.read_csv(file_address)
    logger.info("file loaded successfully...")
    train, test, _ = split_dataset(df,
                              similarity_threshold=0.4, 
                              test_size=0.3)
    logger.info("Splitting sucessfully completed")
    logger.info(f"Len of splitted data: train {len(train)}, test {len(test)} others {len(_)}")
    logger.info(f"Fraction of splitted data: \
          train {len(train)/ len(df)}, test {len(test)/ len(df)}, val test {len(_)/ len(df)}")
    df.iloc[train].to_csv(file_folder/ "train.csv", index=False)
    df.iloc[test].to_csv(file_folder/ "test.csv", index=False)
    df.iloc[_].to_csv(file_folder/ "other.csv", index=False)
    logger.info("Files sucessfully saved, closing application...")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Splitting data into train and test set using lohi splitter")
    parser.add_argument("--file_address", type=str)
    args = parser.parse_args()
    file = args.file_address
    get_datasplits(file)