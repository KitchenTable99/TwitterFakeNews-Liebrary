import sqlite3
import numpy as np
import random
from numpy.random import normal
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
from tqdm import tqdm
import logging


def normal_generator():
    while True:
        val = random.gauss(mu=0, sigma=1)
        yield (val,)


def main():
    conn = sqlite3.connect('./unzip.db')
    # cols = [res[1] for res in conn.execute('PRAGMA table_info(normal_tweet_features)')]
    # with open('./column_order.pickle', 'rb') as fp:
    #     need = pickle.load(fp)
    #
    # tweet_need = set(item for item in need if 'tweet__' in item)
    # print('missing columns:', tweet_need - set(cols))
    #
    # col_info = {}
    # for col in tqdm(cols[1:]):
    #     cur = conn.execute(f'SELECT (100.0 * count({col}) / count(1)) FROM normal_tweet_features')
    #     col_info[col] = cur.fetchall()
    #
    # with open('col_info.pickle', 'wb') as fp:
    #     pickle.dump(col_info, fp)
    # logging.info('Getting training data')
    # train_df = pd.read_sql('SELECT * FROM normal_tweet_features WHERE tweet__has_place IS NOT NULL', conn)
    # train_df.to_sql('infer_tweet_features', conn)

    # infer_cols = ['tweet__contains_media', 'tweet__has_place', 'tweet__nr_of_medias']
    normal_cols = ['tweet__ratio_adjectives', 'tweet__ratio_nouns', 'tweet__ratio_verbs']
    # all_nan_cols = list(set(infer_cols).union(set(normal_cols)))

    # knn_models = {}
    # train_x = train_df.drop(columns=all_nan_cols)
    # for infer_col in tqdm(infer_cols, desc='Creating KNN models'):
    #     train_y = train_df[infer_col]
    #     knn = KNeighborsRegressor(n_neighbors=2) if infer_col != 'tweet__nr_of_medias' else KNeighborsRegressor(n_neighbors=5)
    #     knn_models[infer_col] = knn.fit(train_x, train_y)

    norm_gen = normal_generator()
    for col in tqdm(normal_cols):
        logging.info('Counting column')
        count = conn.execute(f'SELECT COUNT(*) FROM infer_tweet_features WHERE {col} IS NULL').fetchone()[0]
        logging.info('Creating values')
        replace_values = [next(norm_gen) for _ in range(count)]
        logging.info('Replacing')
        with conn:
            conn.executemany(f'UPDATE infer_tweet_features SET {col}=COALESCE({col}, ?)', replace_values)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
