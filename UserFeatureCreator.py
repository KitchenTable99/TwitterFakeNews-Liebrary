import json
import pandas as pd
from typing import Optional
import logging

from FeatureEngineering.FeatureCreator import contains_hashtag, contains_user_mention, contains_url, string_length, \
    get_avg_user_mentions_per_tweet, get_avg_hashtags_per_tweet, get_avg_urls_per_tweet, \
    get_avg_post_time, get_tweets_per_week, get_tweets_per_month, get_tweets_per_day, get_minimum_time_between_tweets, \
    get_maximum_time_between_tweets, get_median_time_between_tweets, get_avg_time_between_tweets, \
    get_nr_of_retweets_per_tweet, get_nr_of_quotes_per_tweet, get_nr_of_replies_per_tweet, is_translator_type, \
    get_user_lang_counts, get_percent_with_url, get_percent_with_hashtag, get_percent_with_user_mention, \
    get_top_level_domain_of_expanded_url, get_top_level_domain_type
from Utility.TimeUtils import TimeUtils


def create_user_features(user_df: str | pd.DataFrame, feature_df: Optional[pd.DataFrame] = None):
    """
    creates the user features and stores them in the DB
    :return:
    """
    logging.debug(f'{user_df = }\n{feature_df = }')

    print("---insert-physical-locations----")
    # TODO: replace location finder

    if not feature_df:
        logging.info('Creating new feature dataframe')
        df = pd.DataFrame()
    else:
        logging.info('Reading feature dataframe')
        df = feature_df

    if isinstance(user_df, str):
        logging.info('Reading user dataframe')
        user_df = pd.read_csv(user_df)

    df['id'] = user_df['id']
    # location
    df['has_location'] = user_df['location'].apply(user__has_location)
    df['has_country'] = user_df['country'] is not None

    # user description
    df['has_desc'] = user_df['description'] is not None and user_df['description'] != ''
    df['desc_contains_hashtags'] = user_df['description'].apply(contains_hashtag)
    df['desc_contains_user_mention'] = user_df['description'].apply(contains_user_mention)
    df['desc_contains_url'] = user_df['description'].apply(contains_url)
    df['desc_length'] = user_df['description'].apply(len)

    # user url
    df['url_length'] = user_df['url'].apply(string_length)

    # followers, favourites, friends, lists
    df['has_list'] = user_df['listed_count'] > 0
    df['has_friends'] = user_df['friends_count'] > 0
    df['friends_per_follower'] = user_df['friends_count'] / user_df['followers_count']
    df['is_following_more_than_100'] = user_df['friends_count'] >= 100
    df['at_least_30_follower'] = user_df['followers_count'] >= 30

    print(df)
    df.to_csv('current.csv', index=False)


# def get_depth2_tweet_features(feature_df: Optional[pd.DataFrame] = None):
#     if not feature_df:
#         df = pd.DataFrame()
#     else:
#         df = feature_df
#
#     df['avg_user_mention_per_tweet'] = get_avg_user_mentions_per_tweet(id)
#     df['avg_hashtags_per_tweet'] = get_avg_hashtags_per_tweet(id)
#     df['avg_urls_per_tweet'] = get_avg_urls_per_tweet(id)
#
#     df['percent_with_url'] = get_percent_with_url(id)
#     df['percent_with_hashtag'] = get_percent_with_hashtag(id)
#     df['percent_with_user_mention'] = get_percent_with_user_mention(id)
#
#     # tweet times
#     df['avg_post_time'] = get_avg_post_time(id)
#     df['tweets_per_day'] = get_tweets_per_day(id)
#     df['tweets_per_week'] = get_tweets_per_week(id)
#     df['min_time_between_tweets'] = get_minimum_time_between_tweets(id)
#     df['max_time_between_tweets'] = get_maximum_time_between_tweets(id)
#     df['median_time_between_tweets'] = get_median_time_between_tweets(id)
#     df['avg_time_between_tweets'] = get_avg_time_between_tweets(id)
#
#     # nr of tweets/retweets/quotes/replies
#     df['more_than_50_tweets'] = user_df['statuses_count'] > 50
#     df['nr_of_retweets'] = db.get_nr_of_retweets_by_user(id)
#     df['nr_of_retweets_per_tweet'] = get_nr_of_retweets_per_tweet(id)
#     df['nr_of_quotes'] = db.get_nr_of_quotes_by_user(id)
#     df['nr_of_quotes_per_tweet'] = get_nr_of_quotes_per_tweet(id)
#     df['nr_of_replies'] = db.get_nr_of_replies_by_user(id)
#     df['nr_of_replies_per_tweet'] = get_nr_of_replies_per_tweet(id)
#
#     # additional user features
#     df['has_tweets_in_different_lang'] = get_user_lang_counts(id) > 1
#     df['tweets_in_different_lang'] = get_user_lang_counts(id)


def favourites_per_follower(x):
    """return #favourties/#followers"""
    return x['Favourites_count'] / x['followers_count']


def friends_per_follower(x):
    """return #favourties/#followers"""
    return x['friends_count'] / x['followers_count']


def friends_per_favourite(x):
    """return #friends/#favourties, if favourites = 0 returns 0"""
    if x['favourites_count'] == 0:
        return 0
    else:
        return x['friends_count'] / x['favourites_count']


def user__has_default_profile_after_two_month(u):
    return TimeUtils.month_ago(u['created_at']) >= 2 and u['default_profile_image']


def user__has_location(loc):
    logging.debug(f'{loc = }')
    loc = loc.replace(' ', '')
    return (loc != '') and (loc is not None)


def main():
    logging.basicConfig(filename='user_features.log', encoding='utf-8', level=logging.DEBUG, format='%(asctime)s - %(message)s')
    blank_user_df = pd.DataFrame()
    user_df = pd.read_csv('prepped.csv')
    create_user_features(user_df)


if __name__ == "__main__":
    main()
