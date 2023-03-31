import sqlite3
import pandas as pd


def get_conn(testing=True):
    return sqlite3.connect(':memory:') if testing else sqlite3.connect('./network_features.db')


def main():
    conn = get_conn(testing=False)
    df = pd.read_parquet('./final_user_features.parquet.gzip')
    df = df.rename(columns={col: f'user__{col}' for col in df.columns if col != 'id' and not col.startswith('user__')})

    df.to_sql('user_features', conn, if_exists='append')


if __name__ == "__main__":
    main()
