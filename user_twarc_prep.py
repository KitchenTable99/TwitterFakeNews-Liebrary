import pandas as pd


def main():
    twarc_df = pd.read_csv('dev_users.csv')
    twarc_df = twarc_df.rename(columns={
        'public_metrics.following_count': 'friends_count',
        'public_metrics.followers_count': 'followers_count',
        'public_metrics.listed_count': 'listed_count',
        'public_metrics.tweet_count': 'statuses_count',
        })
    twarc_df.to_csv('prepped.csv', index=False)

if __name__ == "__main__":
    main()
