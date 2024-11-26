import pandas as pd
from one_hot import convert_int_code, make_one_hot_df
import pickle
from sklearn.neighbors import KNeighborsRegressor


def normalize_tweets(df):
    with open('normalize.pickle', 'rb') as fp:
        normalize_columns = pickle.load(fp)

    tweet_normalize = [col for col in normalize_columns if 'tweet__' in col]
    df[tweet_normalize] = (df[tweet_normalize] - df[tweet_normalize].mean()) / df[tweet_normalize].std()

    with open('one_hot.pickle', 'rb') as fp:
        one_hot_columns = pickle.load(fp)
        one_hot_columns.remove('tweet__hour_of_day')
    oh_dfs = []
    for column in one_hot_columns:
        num_categories = len(df[column].unique()) + 1 if column != 'user__created_hour_of_day' else 24
        num_categories = 32 if column == 'tweet__day_of_month' else num_categories
        # TODO: map the column to the proper value
        oh_matrix = convert_int_code(df[column].to_list(), num_categories)
        oh_dfs.append(make_one_hot_df(oh_matrix, column))

    oh_columns = pd.concat(oh_dfs, axis=1)
    df = pd.concat([df, oh_columns], axis=1)
    df = df.drop(columns=one_hot_columns)
    df = df.drop(columns=['tweet__is_withheld_copyright', 'tweet__url_only',
                          'user__has_url', 'user__is_english',
                          'user__more_than_50_tweets', 'tweet__possibly_sensitive_news',
                          'tweet__day_of_month_0', 'tweet__day_of_week_7',
                           'user__tweets_per_week', 'user__has_country',
                          'tweet__is_quote_status', 'tweet__user_id', 'tweet__hour_of_day'], errors='ignore')
    df = df.rename(columns={'user__tweets_in_different_lang': 'user__nr_languages_tweeted'})

    needs_inference = df['tweet__possibly_sensitive'].isna()
    all_but_inference = df.loc[:, ~df.columns.isin(['tweet__possibly_sensitive'])]
    knn_train_x = all_but_inference[~needs_inference]
    knn_train_y = df[~needs_inference]['tweet__possibly_sensitive']
    knn_test_x = all_but_inference[needs_inference]

    knn = KNeighborsRegressor(n_neighbors=2)
    if df.isna().any().sum() != 0:
        knn.fit(knn_train_x, knn_train_y)
        knn_test_x['tweet__possibly_sensitive'] = knn.predict(knn_test_x)

        no_inference = df[~needs_inference]
        finished_inference = pd.concat([knn_test_x, no_inference])
        df = finished_inference

    return df
