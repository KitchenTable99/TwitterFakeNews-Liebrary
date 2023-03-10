import pandas as pd
from typing import Optional, Dict
import logging

from FeatureCreator import contains_hashtag, contains_user_mention, contains_url, string_length, \
    get_avg_user_mentions_per_tweet, get_avg_hashtags_per_tweet, get_avg_urls_per_tweet, \
    get_avg_post_time, get_tweets_per_day, get_minimum_time_between_tweets, \
    get_maximum_time_between_tweets, get_median_time_between_tweets, get_avg_time_between_tweets, \
    get_nr_of_retweets_per_tweet, get_nr_of_quotes_per_tweet, get_nr_of_replies_per_tweet, \
    get_user_lang_counts, get_percent_with_url, get_percent_with_hashtag, get_percent_with_user_mention, \
    get_nr_of_retweets_by_user, get_nr_of_quotes_by_user, get_nr_of_replies_by_user


def create_user_features(user_df: pd.DataFrame, feature_df: Optional[pd.DataFrame] = None):
    """
    creates the user features and returns them
    :return:
    """
    logging.info(f'{user_df = }\n{feature_df = }')

    if not feature_df:
        logging.info('Creating new feature dataframe')
        feature_df = pd.DataFrame()

    feature_df['id'] = user_df['id']
    # location
    feature_df['has_location'] = user_df['location'].apply(user__has_location)

    # user description
    feature_df['has_desc'] = user_df['description'] is not None and user_df['description'] != ''
    feature_df['desc_contains_hashtags'] = user_df['description'].apply(contains_hashtag)
    feature_df['desc_contains_user_mention'] = user_df['description'].apply(contains_user_mention)
    feature_df['desc_contains_url'] = user_df['description'].apply(contains_url)
    feature_df['desc_length'] = user_df['description'].apply(lambda x: len(x) if not pd.isna(x) else 0)
    feature_df['more_than_50_tweets'] = user_df['statuses_count'] > 50

    # user url
    feature_df['url_length'] = user_df['url'].apply(string_length)

    # followers, favourites, friends, lists
    feature_df['has_list'] = user_df['listed_count'] > 0
    feature_df['has_friends'] = user_df['friends_count'] > 0
    feature_df['friends_per_follower'] = user_df.apply(lambda x: friends_per_follower(x), axis=1)
    feature_df['is_following_more_than_100'] = user_df['friends_count'] >= 100
    feature_df['at_least_30_follower'] = user_df['followers_count'] >= 30

    return feature_df


def get_depth2_tweet_features(d2_tweets: pd.DataFrame) -> Dict:
    """Takes in ALL depth=2 tweets for a given user in the dataset and
    returns a dictionary with values for the appropriate features"""
    logging.info('DESCRIPTION:' + str(d2_tweets.describe()))
    logging.info('HEAD:' + str(d2_tweets.head()))
    logging.info('LEN:' + str(len(d2_tweets)))

    feature_dict = dict()

    feature_dict['avg_user_mention_per_tweet'] = get_avg_user_mentions_per_tweet(d2_tweets)
    feature_dict['avg_hashtags_per_tweet'] = get_avg_hashtags_per_tweet(d2_tweets)
    feature_dict['avg_urls_per_tweet'] = get_avg_urls_per_tweet(d2_tweets)

    feature_dict['percent_with_url'] = get_percent_with_url(d2_tweets)
    feature_dict['percent_with_hashtag'] = get_percent_with_hashtag(d2_tweets)
    feature_dict['percent_with_user_mention'] = get_percent_with_user_mention(d2_tweets)

    # tweet times
    feature_dict['avg_post_time'] = get_avg_post_time(d2_tweets)
    feature_dict['tweets_per_day'] = get_tweets_per_day(d2_tweets)
    feature_dict['min_time_between_tweets'] = get_minimum_time_between_tweets(d2_tweets)
    feature_dict['max_time_between_tweets'] = get_maximum_time_between_tweets(d2_tweets)
    feature_dict['median_time_between_tweets'] = get_median_time_between_tweets(d2_tweets)
    feature_dict['avg_time_between_tweets'] = get_avg_time_between_tweets(d2_tweets)

    # nr of tweets/retweets/quotes/replies
    feature_dict['nr_of_retweets'] = get_nr_of_retweets_by_user(d2_tweets)
    feature_dict['nr_of_retweets_per_tweet'] = get_nr_of_retweets_per_tweet(d2_tweets)
    feature_dict['nr_of_quotes'] = get_nr_of_quotes_by_user(d2_tweets)
    feature_dict['nr_of_quotes_per_tweet'] = get_nr_of_quotes_per_tweet(d2_tweets)
    feature_dict['nr_of_replies'] = get_nr_of_replies_by_user(d2_tweets)
    feature_dict['nr_of_replies_per_tweet'] = get_nr_of_replies_per_tweet(d2_tweets)

    # additional user features
    feature_dict['has_tweets_in_different_lang'] = get_user_lang_counts(d2_tweets) > 1
    feature_dict['tweets_in_different_lang'] = get_user_lang_counts(d2_tweets)

    return feature_dict


def favourites_per_follower(x):
    """return #favourties/#followers"""
    return x['Favourites_count'] / x['followers_count']


def friends_per_follower(x):
    """return #favourties/#followers"""
    if x['followers_count'] == 0:
        return 0
    return x['friends_count'] / x['followers_count']


def friends_per_favourite(x):
    """return #friends/#favourties, if favourites = 0 returns 0"""
    if x['favourites_count'] == 0:
        return 0
    else:
        return x['friends_count'] / x['favourites_count']


def user__has_location(loc):
    logging.info(f'{loc = }')
    if loc is None:
        return False
    elif loc == '':
        return False
    elif loc == ' ':
        return False
    else:
        return True


def main():
    logging.basicConfig(filename='user_features.log', encoding='utf-8', level=logging.INFO, format='%(asctime)s - %(message)s')
    user_df = pd.read_csv('prepped.csv')
    
    user_features = create_user_features(user_df)
    user_features.to_parquet('user_features2.parquet.gzip', compression='gzip', index=False)
    depth_df = pd.read_parquet('fifthf.parquet.gzip')
    get_depth2_tweet_features(depth_df)
    


if __name__ == "__main__":
    main()
