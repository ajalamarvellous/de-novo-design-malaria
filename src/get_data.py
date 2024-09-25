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


def describe_data(df: pd.DataFrame) -> None:
    """
    Basic description of the dataset
    """
    print(f"No of columns: {df.shape[1]} and rows: {df.shape[0]}")
    print(f"The Columns of the data \n {list(df.columns)}")
    print(f"The top 5 rows of the data \n {df.head().T}")
    print(f"The number of duplicate values: {df.duplicated().sum()}")
    print(f"The number of unique smile values: {df.CANONICAL_SMILES.nunique()}")
    print(f"The number of rows with no target values: {df.STANDARD_VALUE.isna().sum()}")
    print(f"The major types of activities in the data: {df.STANDARD_TYPE.unique()}")


def subselect_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Subselect a portion of the dataset relevant for the dataset, 
    specifically, we want to subselect the dataset for potency
    
    Argument(s)
    -------------
    df: pd.DataFrame
        original complete dataset

    Return(s)
    ------------
    new_df: pd.DataFrame
        new subselected dataset
    """
    # subselect only potency and IC50 standard test type
    df = df[(df["STANDARD_TYPE"] == "Potency") | (df["STANDARD_TYPE"] == "IC50")]
    # subselect only dataset with activity comment
    df = df[df.ACTIVITY_COMMENT.notnull()]
    return df


def lowercase_column(df: pd.DataFrame, column: str="ACTIVITY_COMMENT") -> pd.DataFrame:
    """
    This fn helps to normalise a column, specifically the activity column to
    ensure that all values are in lowercase
    
    Argument(s)
    -------------
    df: pd.DataFrame
        the dataset
    column: column we want to select
    """
    df[column] = df[column].apply(lambda x: x.lower())
    return df


if __name__ == "__main__":
    df = read_file("data/MalariaData_bioactivity.txt")
    describe_data(df)