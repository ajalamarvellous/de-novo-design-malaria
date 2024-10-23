from rdkit import Chem
from rdkit.DataStructs import cDataStructs
from rdkit.Chem import rdFingerprintGenerator

import pandas as pd
import numpy as np

from typing import TypeVar, List, Union, NewType
from pathlib import Path


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
    