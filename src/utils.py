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