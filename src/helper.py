from typing import Tuple
import pandas as pd
import lohi_splitter as lohi


def split_dataset(df: pd.DataFrame, 
                  similarity_threshold: float, 
                  train_frac: float,
                  test_frac: float,
                  coarsening_frac: float=0.4,
                  max_mip_gap: float=0.01) -> Tuple[pd.DataFrame]:
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
    train_frac: float \n
        min fraction of the dataset that should be in the train set
    test_frac: float \n
        min fraction of the dataset that should be in the test set
    coarsening_frac: float \n
        threshold for graph clustering, used to tune the fraction \
        of data to drop and how fast the computation should be
    max_mip_gap: float (range 0-1)
        how close to theoretical optimum to terminate operation, used \
        to tune the fraction of data to drop and how fast the computation \
        should be. Higher value to run faster and discard more molecules

    Return(s)
    -----------
    splits: Tuple[pd.DataFrame]
        the splitted dataset in the order (train, test)
    """

    smiles = df["SMILES"].to_dict
    train_test_partition = lohi.hi_train_test_splitter(
                                smiles=smiles, 
                                similarity_threshhold=similarity_threshold, 
                                train_min_frac=train_frac, 
                                test_min_frac=test_frac, 
                                coarsening_fthreshold=coarsening_frac,
                                max_mip_gap=max_mip_gap
                                )
    train_df = df.iloc[train_test_partition[0]]
    test_df = df.iloc[train_test_partition[1]]
    return (train_df, test_df)