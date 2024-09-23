import os
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from rdkit.Chem import PandasTools


def read_file(file_address: str) -> pd.DataFrame:
    """
    Read the data file in pandas and return a pandas dataframe

    Argument(s)
    ------------
    file_address: str
        address of the file to be opened

    Return(s)
    -----------
    file_dataframe: pd.DataFrame
        pandas dataframe of the opened file
    """
    return pd.read_table(file_address)

