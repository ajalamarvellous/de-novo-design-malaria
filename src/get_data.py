import os
import logging
from pathlib import Path
from typing import List, TypeVar

import pandas as pd

# defining new variable for our artifact locations
addressType = TypeVar("addressType", str, Path)
# basic logging config
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(funcName)s[%(levelname)s]: %(message)s ",
)
logger = logging.getLogger()

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
    logger.info("File read successfully...")
    return df


def describe_data(df: pd.DataFrame) -> None:
    """
    Basic description of the dataset
    """
    logger.debug(f"No of columns: {df.shape[1]} and rows: {df.shape[0]}")
    logger.debug(f"The Columns of the data \n {list(df.columns)}")
    logger.debug(f"The top 5 rows of the data \n {df.head().T}")
    logger.debug(f"The number of duplicate values: {df.duplicated().sum()}")
    logger.debug(f"The number of unique smile values: {df.CANONICAL_SMILES.nunique()}")
    logger.debug(f"The number of rows with no target values: {df.STANDARD_VALUE.isna().sum()}")
    logger.debug(f"The major types of activities in the data: {df.STANDARD_TYPE.unique()}")


def subselect_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Subselect a portion of the dataset relevant for the dataset, 
    specifically, we want to subselect the dataset for potency
    
    Argument(s)
    -------------
    df: pd.DataFrame \n
        original complete dataset

    Return(s)
    ------------
    new_df: pd.DataFrame \n
        new subselected dataset
    """
    # subselect only potency and IC50 standard test type
    df = df[(df["STANDARD_TYPE"] == "Potency") | (df["STANDARD_TYPE"] == "IC50")]
    # subselect only dataset with activity comment
    df = df[df.ACTIVITY_COMMENT.notnull()]
    logger.info("Subset of the data successfully selected...")
    return df


def lowercase_column(df: pd.DataFrame, column: str="ACTIVITY_COMMENT") -> pd.DataFrame:
    """
    This fn helps to normalise a column, specifically the activity column to
    ensure that all values are in lowercase
    
    Argument(s)
    -------------
    df: pd.DataFrame \n
        the dataset
    column: str \n 
        column we want to select

    Return(s)
    -----------
    new_df: pd.DataFrame \n
        new dataset with selected item all in lowercase
    """
    df[column] = df[column].apply(lambda x: x.lower())
    logger.info(f"All items of column {column} converted to lowercase...")
    return df


def group_by(df: pd.DataFrame, key: str, columns=List[str]) -> pd.DataFrame:
    """
    Group the dataframe by a key (smiles) and returns specified columns

    Argument(s)
    -------------
    df: pd.DataFrame \n
        dataframe to make edit on
    key: str \n
        column that is to serve as the key/primary identification
    column: List[str] \n
        List of the other columns to return

    Return(s)
    -----------
    new_df: pd.DataFrame
        new dataset that has been subselected
    """
    df = pd.DataFrame(df.groupby(key)[columns])
    logger.info(f"Data sucessfully grouped by {key}...")
    return df


def create_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given the dataset grouped by the SMILES with the standard value and activity comment, return a 
    dataframe returning with the smile and true if any standard value returns positive. \n
    NB: This is based on the subselected potency and IC50 dataset

    Argument(s)
    ------------
    df: pd.DataFrame \n
        dataframe created based on groupby smiles containing just standard_value and activity comment

    Return(s)
    ----------
    new_df: pd.DataFrame \n
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
    df = pd.DataFrame(new_ds, columns=["SMILES", "ACTIVITY"])
    logger.info("Final dataset created")
    return df


def main(
        FILE_ADDRESS: addressType, 
        key: str, 
        COLUMNS: List, 
         VIRTUAL_SCREENING_DS: addressType, 
         DENOVO_DS: addressType
    ) -> None:
    df = read_file(FILE_ADDRESS)
    describe_data(df)
    df = subselect_data(df)
    df = lowercase_column(df)
    df = group_by(df, key=key, columns=COLUMNS)
    final_df = create_dataset(df)
    # save the final dataset to be used for virtual screening
    final_df.to_csv(VIRTUAL_SCREENING_DS, index=False)
    # query for just the positive molecules and save just their smiles for denovo design
    final_df.query("ACTIVITY == True")["SMILES"].to_csv(DENOVO_DS, index=False)
    logger.info(f"Files saved to {VIRTUAL_SCREENING_DS} and {DENOVO_DS}")


if __name__ == "__main__":
    FILE_ADDRESS = Path("data/MalariaData_bioactivity.txt")
    VIRTUAL_SCREENING_DS = Path("data/Virtual-Screening.csv")
    DENOVO_DS = Path("data/de-novo.csv")

    key = "CANONICAL_SMILES"
    COLUMNS = ["STANDARD_VALUE", "ACTIVITY_COMMENT"]
    
    logger.info("Task starting...")
    main(FILE_ADDRESS, key, COLUMNS, VIRTUAL_SCREENING_DS, DENOVO_DS)
    logger.info("Task completed...")
