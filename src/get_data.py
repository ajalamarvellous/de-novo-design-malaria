import os
from pathlib import Path
from typing import List

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

def group_by(df: pd.DataFrame, key: str, columns=List[str]) -> pd.DataFrame:
    """
    Group the dataframe by a key (smiles) and returns specified columns

    Argument(s)
    -------------
    df: pd.DataFrame
        dataframe to make edit on
    key: str
        column that is to serve as the key/primary identification
    column: List[str]
        List of the other columns to return

    Return(s)
    -----------
    new_df: pd.DataFrame
        new dataset that has been subselected
    """
    return pd.DataFrame(df.groupby(key)[columns])


def create_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given the dataset grouped by the SMILES with the standard value and activity comment, return a 
    dataframe returning with the smile and true if any standard value returns positive.
    NB: This is based on the subselected potency and IC50 dataset

    Argument(s)
    ------------
    df: pd.DataFrame
        dataframe created based on groupby smiles containing just standard_value and activity comment

    Return(s)
    ----------
    new_df: pd.DataFrame
        new dataframe returning smiles and true is any standard experiment showed it's active
    """

    def get_activity(df: pd.DataFrame) -> bool:
        for i in list(df["ACTIVITY_COMMENT"]):
            if i == "active" :
                return True
        return False
    new_ds = []
    for i in range(len(df)):
        row = df.iloc[i]
        smiles = row[0]
        activity = get_activity(row[1])
        new_ds.append([smiles, activity])
    return pd.DataFrame(new_ds, columns=["SMILES", "ACTIVITY"])


if __name__ == "__main__":
    df = read_file("data/MalariaData_bioactivity.txt")
    describe_data(df)