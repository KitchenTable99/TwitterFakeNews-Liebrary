import pandas as pd
import random
from typing import List
import numpy as np


def convert_int_code(int_codes: List[int], num_categories: int) -> np.ndarray:
    num_obs = len(int_codes)
    one_hot_array = np.zeros((num_obs, num_categories))

    for obs_num, int_code in enumerate(int_codes):
        one_hot_array[obs_num, int_code] = 1

    return one_hot_array


def make_one_hot_df(one_hot_array: np.ndarray, category_name: str) -> pd.DataFrame:
    if category_name.endswith('_'):
        category_name = category_name[:-1]

    num_categories = one_hot_array.shape[1]
    headers = [f'{category_name}_{i}' for i in range(num_categories)]

    return pd.DataFrame(one_hot_array, columns=headers)


def main():
    test = [i for i in range(10)]
    category = 'dick_size'
    oh_matrix = convert_int_code(test, 10)
    print(make_one_hot_df(oh_matrix, category))


if __name__ == "__main__":
    main()
