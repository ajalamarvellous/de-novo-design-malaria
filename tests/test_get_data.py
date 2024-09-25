import sys
import os
from pathlib import Path
import pytest

import pandas as pd

sys.path.insert(1, "/Users/madeofajala/Projects/Malaria/src/")
from get_data import (read_file, 
                      describe_data,
                      subselect_data, 
                      lowercase_column)

FILE_ADDRESS = "data/MalariaData_bioactivity.txt"

@pytest.fixture()
def df(FILE_ADDRESS=FILE_ADDRESS):
    """
    Returns a pandas dataframe of the data
    """
    return pd.read_table(FILE_ADDRESS, low_memory=False)


def test_read_file(FILE_ADDRESS=FILE_ADDRESS):
    df = read_file(FILE_ADDRESS)
    assert type(df) == pd.DataFrame, "Returned object not pandas DataFrame"

def test_describe_data(df):
    describe_data(df)

def test_subselect_data(df):
    new_df = subselect_data(df)
    assert len(new_df) < len(df), "New dataset size not smaller than original dataset size"
    assert new_df.STANDARD_TYPE.nunique() == 2, f"Standard_type does not contain \
          only potency and IC50, contains {new_df.STANDARD_TYPE.nunique()}"
    assert list(set(list(new_df.STANDARD_TYPE.unique())).difference(
        ["Potency", "IC50"])) == [], f"presence of other stardard types, all \
            available types: {new_df.STANDARD_TYPE.unique()}"
    assert new_df.ACTIVITY_COMMENT.isna().sum() == 0, "Presence of null values in the activity_comment"

def test_lowercase_column(df):
    column = "ACTIVITY_COMMENT"
    new_df = lowercase_column(subselect_data(df), column)
    assert new_df[column].apply(
      lambda x: x.islower()
    ).sum() == new_df.shape[0], "Not all values in lowercase"