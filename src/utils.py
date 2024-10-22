from rdkit import Chem

import pandas as pd
import numpy as np

from typing import TypeVar
from pathlib import Path


# defining new variable for our artifact locations
addressType = TypeVar("addressType", str, Path)

def read_file(file_address: addressType) -> pd.DataFrame:
    """
    Read the data file in pandas and return a pandas dataframe

    Argument(s)
    ------------
    file_address: str \n
        address of the file to be opened

    Return(s)
    -----------
    file_dataframe: pd.DataFrame \n
        pandas dataframe of the opened file
    """
    df = pd.read_table(file_address, low_memory=False)
    return df

def get_mols(df: pd.DataFrame, column: str) -> list:
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
