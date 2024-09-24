import sys
import os
from pathlib import Path

import pandas as pd

sys.path.insert(1, "/Users/madeofajala/Projects/Malaria/src/")
from get_data import (read_file, 
                      describe_data,
                        subselect_data, )


def test_read_file():
    file_address = "data/MalariaData_bioactivity.txt"
    df = read_file(file_address)
    assert type(df) == pd.DataFrame, "Returned object not pandas DataFrame"

def test_describe_data():
    file_address = "data/MalariaData_bioactivity.txt"
    df = read_file(file_address)
    describe_data(df)

def test_subselect_data():
    file_address = "data/MalariaData_bioactivity.txt"
    df = read_file(file_address)
    new_df = subselect_data(df)
    assert len(new_df) < len(df), "New dataset size not smaller than original dataset size"
    assert new_df.STANDARD_TYPE.nunique() == 2, f"Standard_type does not contain \
          only potency and IC50, contains {new_df.STANDARD_TYPE.nunique()}"
    assert list(set(list(new_df.STANDARD_TYPE.unique())).difference(
        ["Potency", "IC50"])) == [], f"presence of other stardard types, all \
            available types: {new_df.STANDARD_TYPE.unique()}"
    assert new_df.ACTIVITY_COMMENT.isna().sum() == 0, "Presence of null values in the activity_comment"