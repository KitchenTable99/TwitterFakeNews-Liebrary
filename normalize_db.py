import sqlite3
from pprint import pprint
from sklearn.neighbors import KNeighborsRegressor
from one_hot import make_one_hot_df, convert_int_code
from tqdm import tqdm
import logging
from typing import Tuple, Dict
import pandas as pd
import pickle


class STDDEV:
    """An iterative implementation of standard deviation.
    For the formula see https://en.wikipedia.org/wiki/Standard_deviation#Identities_and_mathematical_properties"""

    def __init__(self):
        self.count = 0
        self.sum_x = 0
        self.sum_x2 = 0

    def step(self, value):
        if value is None:
            return

        self.count += 1
        self.sum_x += value
        self.sum_x2 += value**2

    def finalize(self):
        try:
            left = self.sum_x2 / self.count
            right = (self.sum_x / self.count) ** 2
            inner = left - right

            return inner ** .5
        except Exception as e:
            print(e)
            raise e


def get_conn(testing=True):
    return sqlite3.connect('./test_network_features.db') if testing else sqlite3.connect('./network_features.db')


def build_global_norm_dict(conn, table_name, cols=None) -> Tuple[Dict[str, float], Dict[str, float]]:
    with open('normalize.pickle', 'rb') as fp:
        normalize_columns = pickle.load(fp)

    avgs = {}
    stds = {}
    iter_cols = cols if cols else normalize_columns
    for col in tqdm(iter_cols, desc='Finding avg and std'):
        logging.debug(f'Finding std and avg for col: {col}')
        col_present = conn.execute(f"SELECT COUNT(*) FROM pragma_table_info('{table_name}') WHERE name='{col}';").fetchone()
        if col_present[0] == 0:  # the column is not present
            logging.warning(f'column not present: {col}')
            continue
        avg = conn.execute(f"SELECT AVG({col}) from {table_name};").fetchone()[0]
        avgs[col] = avg

        std = conn.execute(f"SELECT STDDEV({col}) from {table_name};").fetchone()[0]
        stds[col] = std

    return (avgs, stds)


def get_drop_cols(oh_columns, df_columns):
    to_return = []

    to_return.extend(oh_columns)
    to_return.extend(['tweet__is_withheld_copyright', 'tweet__url_only',
                      'user__has_url', 'user__is_english',
                      'user__more_than_50_tweets', 'tweet__possibly_sensitive_news',
                      'tweet__day_of_month_0', 'tweet__day_of_week_7',
                      'user__tweets_per_week', 'user__has_country',
                      'tweet__is_quote_status', 'tweet__user_id', 'tweet__hour_of_day'])
    with open('./column_order.pickle', 'rb') as fp:
        keep_cols = pickle.load(fp)
    to_return.extend([col for col in df_columns if col not in keep_cols])

    to_return.remove('id')
    to_return.remove('author_id')

    return to_return


def normalize_chunk(df, avgs, stds, cols=None):
    pprint(df.columns.to_list())
    with open('normalize.pickle', 'rb') as fp:
        normalize_columns = pickle.load(fp)

    itercols = cols if cols else normalize_columns
    for col in tqdm(itercols, desc='Normalizing columns'):
        avg = avgs.get(col, None)
        std = stds.get(col, None)
        if avg is None:
            continue
        df[col] = (df[col] - avg) / std

    with open('one_hot.pickle', 'rb') as fp:
        one_hot_columns = pickle.load(fp)
        one_hot_columns.remove('tweet__hour_of_day')
    oh_dfs = []
    for column in one_hot_columns:
        if column not in df.columns:
            continue
        num_categories = len(df[column].unique()) + 1 if column != 'user__created_hour_of_day' else 24
        num_categories = 32 if column == 'tweet__day_of_month' else num_categories
        # TODO: map the column to the proper value
        oh_matrix = convert_int_code(df[column].to_list(), num_categories)
        oh_dfs.append(make_one_hot_df(oh_matrix, column))

    oh_columns = pd.concat(oh_dfs, axis=1)
    df = pd.concat([df, oh_columns], axis=1)

    drop_cols = get_drop_cols(one_hot_columns, df.columns)
    df = df.drop(columns=drop_cols, errors='ignore')
    df = df.rename(columns={'user__tweets_in_different_lang': 'user__nr_languages_tweeted'})

    # TODO: add inference once all columns are dropped and user columns are added
    # TODO: infer medias as well
    # needs_inference = df['possibly_sensitive'].isna()
    # all_but_inference = df.loc[:, ~df.columns.isin(['possibly_sensitive'])]
    # knn_train_x = all_but_inference[~needs_inference]
    # knn_train_y = df[~needs_inference]['possibly_sensitive']
    # knn_test_x = all_but_inference[needs_inference]

    # knn = KNeighborsRegressor(n_neighbors=2)
    # if df.isna().any().sum() != 0:
    #     knn.fit(knn_train_x, knn_train_y)
    #     knn_test_x['possibly_sensitive'] = knn.predict(knn_test_x)
    #
    #     no_inference = df[~needs_inference]
    #     finished_inference = pd.concat([knn_test_x, no_inference])
    #     df = finished_inference

    return df


def main():
    testing = False
    logging.basicConfig(level=logging.WARN)

    conn = get_conn(testing)
    conn.create_aggregate('STDDEV', 1, STDDEV)

    if conn.execute("SELECT COUNT(*) FROM pragma_table_info('like_features') WHERE name='tweet__favorite_count'").fetchone()[0] == 0:
        conn.execute("ALTER TABLE like_features RENAME like_count TO tweet__favorite_count")
    if conn.execute("SELECT COUNT(*) FROM pragma_table_info('like_features') WHERE name='tweet__retweet_count'").fetchone()[0] == 0:
        conn.execute("ALTER TABLE like_features RENAME retweet_count TO tweet__retweet_count")
    if conn.execute("SELECT COUNT(*) FROM pragma_table_info('like_features') WHERE name='tweet__user_id'").fetchone()[0] == 0:
        conn.execute("ALTER TABLE like_features ADD tweet__user_id")
        conn.execute("UPDATE like_features SET tweet__user_id = author_id")
        conn.commit()

    avgs, stds = build_global_norm_dict(conn, 'like_features')
    if not testing:
        with open('avg_std.pickle', 'wb') as fp:
            to_write = (avgs, stds)
            pickle.dump(to_write, fp)

    df = pd.read_sql('SELECT * FROM like_features', conn)
    df = normalize_chunk(df, avgs, stds)
    print(df)
    conn.close()


if __name__ == "__main__":
    main()
