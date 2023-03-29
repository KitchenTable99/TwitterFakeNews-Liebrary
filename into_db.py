from tqdm import tqdm as tqdm
import os
import logging
import pickle
import pandas as pd
import sqlite3


with open('./column_order.pickle', 'rb') as fp:
    COLUMN_ORDER = pickle.load(fp)


def get_conn(testing=True):
    return sqlite3.connect(':memory:') if testing else sqlite3.connect('network_features.db')


def insert_into_db(file, conn):
    logging.debug(f'{file = }')
    df = pd.read_parquet(file)
    table_name = 'like_features' if 'likes' in file else 'retweet_features'
    df.to_sql(table_name, conn, if_exists='append')


def main():
    conn = get_conn(testing=False)
    to_read = os.listdir('./processed_tweets/')
    for file in tqdm(to_read):
        if file == 'full.zip':
            continue
        insert_into_db(f'processed_tweets/{file}', conn)
    conn.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
