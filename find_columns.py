import pandas as pd
from normalizer import normalize_tweets
import pickle


def main():
    with open('./column_order.pickle', 'rb') as fp:
        col_order = pickle.load(fp)
    tweet_features = [col for col in col_order if 'tweet__' in col]
    tweet_features = set(tweet_features)


    df = pd.read_parquet('./processed_tweets/processed_likes_0.parquet.gzip')
    df = normalize_tweets(df)
    existing_tweet_features = [col for col in df.columns if 'tweet__' in col]
    existing_tweet_features = set(existing_tweet_features)

    print(f'Missing: {len(tweet_features - existing_tweet_features)}')
    print(tweet_features - existing_tweet_features)




if __name__ == "__main__":
    main()
