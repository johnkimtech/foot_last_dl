import pandas as pd
import numpy as np


def find_nearest_entry(input_vector, database):
    """
    Find the entry in the database with the minimum Euclidean distance to the input vector.

    Parameters:
    - input_vector (numpy array): The input vector for which to find the nearest entry.
    - database (numpy array): The database of vectors to search through.

    Returns:
    - int: The index of the nearest entry in the database.
    """
    distances = np.linalg.norm(database - input_vector, axis=1)
    nearest_entry_index = np.argmin(distances)
    return nearest_entry_index


def find_last(
    last_param, last_info_csv, headers=["발 길이", "발폭", "발볼 높이", "앞코 높이", "힐 높이"]
):
    last_df = pd.read_csv(last_info_csv).reset_index(drop=True)
    last_matrix = last_df.loc[:, headers].to_numpy()
    match_idx = find_nearest_entry(last_param, last_matrix)
    return last_df.iloc[match_idx, :]