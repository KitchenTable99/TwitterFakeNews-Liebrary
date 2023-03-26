import json
import logging
import ast
from collections import Counter
import numpy as np
from collections import Counter

import pandas as pd
import re

from nltk import ngrams

from Emoji import Emojis
from NLPUtils import NLPUtils
from TextParser import TextParser
from TextPreprocessor import TextPreprocessor

# from TextModel import TextModel

from UrlUtils import UrlUtils

punctuation = NLPUtils.get_punctuation()

def json_string_to_counter(x):
    return Counter(json.loads(x))


SECONDS_IN_DAY = 86400

def create_features(data):
    # emojis
    data['tweet__nr_of_unicode_emojis'] = data['tweet__unicode_emojis'].map(lambda x: len(NLPUtils.str_list_to_list(x)))
    data['tweet__contains_unicode_emojis'] = data['tweet__nr_of_unicode_emojis'].map(lambda x: x > 0)
    data['tweet__contains_face_positive_emojis'] = data['tweet__unicode_emojis'].map(
        lambda x: Emojis.unicode_emoji_in_category(NLPUtils.str_list_to_list(x), 'face-positive') > 0)
    data['tweet__contains_face_negative_emojis'] = data['tweet__unicode_emojis'].map(
        lambda x: Emojis.unicode_emoji_in_category(NLPUtils.str_list_to_list(x), 'face-negative') > 0)
    data['tweet__contains_face_neutral_emojis'] = data['tweet__unicode_emojis'].map(
        lambda x: Emojis.unicode_emoji_in_category(NLPUtils.str_list_to_list(x), 'face-neutral') > 0)

    # print("drop column tweet__unicode_emojis")
    # data = data.drop('tweet__unicode_emojis', 1)

    data['tweet__nr_of_ascii_emojis'] = data['text'].map(
            lambda x: Emojis.count_ascii_emojis(TextPreprocessor.remove_urls(x)))
    data['tweet__contains_ascii_emojis'] = data['tweet__nr_of_ascii_emojis'].map(lambda x: x > 0)

    # print("drop column tweet__ascii_emojis")
    # data = data.drop('tweet__ascii_emojis', 1)

    data['tweet__tokenized_um_url_removed'] = data['tweet__tokenized_text'].map(
        lambda x: remove_um_url(NLPUtils.str_list_to_list(x)))

    data['tweet__possibly_sensitive_news'] = data['possibly_sensitive'].map(lambda x: get_possibly_sensitive(x))
    data['tweet__no_text'] = data['tweet__tokenized_um_url_removed'].map(
        lambda x: len(NLPUtils.str_list_to_list(x)) == 0)

    data['tweet__nr_of_words'] = data['tweet__tokenized_um_url_removed'].map(
        lambda x: get_nr_of_words(NLPUtils.str_list_to_list(x)))

    # pos tag related features
    data['tweet__nr_tokens'] = data['tweet__tokenized_um_url_removed'].map(
        lambda x: get_nr_of_words(x))
    data['tweet__ratio_adjectives'] = data.apply(
        lambda x: get_tag_ratio(x['pos_tags'], 'A', x['tweet__nr_tokens']),
        axis=1)
    data['tweet__ratio_nouns'] = data.apply(lambda x: get_tag_ratio(x['pos_tags'], 'N', x['tweet__nr_tokens']),
                                            axis=1)
    data['tweet__ratio_verbs'] = data.apply(lambda x: get_tag_ratio(x['pos_tags'], 'V', x['tweet__nr_tokens']),
                                            axis=1)
    data['tweet__contains_named_entities'] = data['pos_tags'].map(
        lambda x: tweet_contains_named_entities(x))
    data['tweet__contains_pronouns'] = data['pos_tags'].map(lambda x: "U" in [t['tag'] for t in x])

    # POS trigrams
    # trigram_vectors = find_frequent_pos_trigrams(data, min_doc_frequency=1000, no_above=0.4, keep_n=100)
    # for key, vector in trigram_vectors.items():
    #     data['tweet__contains_pos_trigram_{}'.format(re.sub(" ", "_", str(key)))] = vector

    # text/word length
    data['tweet__avg_word_length'] = data['tweet__tokenized_um_url_removed'].map(
        lambda x: tweet_avg_word_length(NLPUtils.str_list_to_list(x)))
    data['tweet__nr_of_slang_words'] = data['tweet__tokenized_um_url_removed'].map(
        lambda x: len(slang_words_in_tweet(NLPUtils.str_list_to_list(x))))
    data['tweet__ratio_uppercase_letters'] = data['tweet__tokenized_um_url_removed'].map(
        lambda x: tweet_ratio_uppercase_letters(x))
    data['tweet__ratio_capitalized_words'] = data.apply(
        lambda x: tweet_ratio_capitalized_words(x), axis=1)
    data['tweet__ratio_all_capitalized_words'] = data.apply(
        lambda x: tweet_ratio_all_capitalized_words(x), axis=1)

    # data = data.drop('tweet__tokenized_um_url_removed', 1)

    data['tweet__nr_of_tokens'] = data['tweet__tokenized_text'].map(lambda x: len(NLPUtils.str_list_to_list(x)))

    data['tweet__text_length'] = data.apply(
        lambda row: get_tweet_text_length(row['text']), axis=1)
    data['tweet__percent_of_text_used'] = data['tweet__text_length'] / 140
    data['tweet__ratio_words_tokens'] = data['tweet__nr_of_words'] / data['tweet__nr_of_tokens']

    # url
    data['tweet__nr_of_urls'] = data.apply(lambda row: tweet_nr_of_urls(row), axis=1)
    data['tweet__contains_urls'] = data['tweet__nr_of_urls'].map(lambda x: x > 0)
    data['tweet__avg_url_length'] = data.apply(lambda row: tweet_avg_url_length(row), axis=1)

    # stock symbol
    data['tweet__contains_stock_symbol'] = data['text'].map(lambda x: bool(tweet_find_stock_mention(x)))

    # punctuation
    data['tweet__nr_of_punctuations'] = data['text'].map(
        lambda x: sum(get_nr_of_punctuation(x).values()))
    data['tweet__contains_punctuation'] = data['tweet__nr_of_punctuations'].map(lambda x: x > 0)
    data['tweet__ratio_punctuation_tokens'] = data.apply(
        lambda x: ratio_punctuation_tokens(x['tweet__tokenized_text'], x['tweet__nr_of_tokens']), axis=1)
    data['tweet__nr_of_exclamation_marks'] = data['text'].map(
        lambda x: get_nr_of_punctuation(x)['!'])
    data['tweet__contains_exclamation_mark'] = data['tweet__nr_of_exclamation_marks'].map(lambda x: x > 0)
    data['tweet__multiple_exclamation_marks'] = data['tweet__nr_of_exclamation_marks'].map(lambda x: x > 1)
    data['tweet__nr_of_question_marks'] = data['text'].map(
        lambda x: get_nr_of_punctuation(x)['?'])
    data['tweet__contains_question_mark'] = data['tweet__nr_of_question_marks'].map(lambda x: x > 0)
    data['tweet__multiple_question_marks'] = data['tweet__nr_of_question_marks'].map(lambda x: x > 1)

    # further NLP

    data['tweet__contains_character_repetitions'] = data['text'].map(
        lambda x: tweet_contains_character_repetitions(x))
    data['tweet__contains_slang'] = data['tweet__nr_of_slang_words'].map(
        lambda x: 0 < x)

    data['tweet__is_all_uppercase'] = data['text'].map(lambda x: is_upper(x))
    data['tweet__contains_uppercase_text'] = data['text'].map(lambda x: contains_all_uppercase(x))

    data['tweet__contains_number'] = data['tweet__additional_preprocessed_text'].map(
        lambda x: contains_number(NLPUtils.str_list_to_list(x)))
    data['tweet__contains_quote'] = data['text'].map(lambda x: contains_quote(x))
    # data = data.drop('tweet__text', 1)

    # media
    # these columns are taken care of while cleaning

    # user mentions
    data['tweet__nr_of_user_mentions'] = data.apply(lambda x: tweet_nr_of_user_mentions(x), axis=1)
    data['tweet__contains_user_mention'] = data['tweet__nr_of_user_mentions'].map(lambda x: x > 0)


    data['tweet__nr_of_hashtags'] = data.apply(lambda row: tweet_nr_of_hashtags(row), axis=1)
    data['tweet__contains_hashtags'] = data['tweet__nr_of_hashtags'].map(lambda x: x > 0)


    # TODO:
    # data['tweet__additional_preprocessed_is_empty'] = data['tweet__additional_preprocessed_wo_stopwords'].map(
    #     lambda x: len(NLPUtils.str_list_to_list(x)) == 0)

    # sentiment related
    # BUG: changed to 0. instead 0.5
    data['tweet__contains_sentiment'] = data['tweet__sentiment_score'].map(lambda x: x != 0.)
    data['tweet__ratio_pos_sentiment_words'] = data.apply(
        lambda x: tweet_ratio_sentiment_words(x['tweet__nr_pos_sentiment_words'], x['tweet__nr_of_sentiment_words']), axis=1)
    data['tweet__ratio_neg_sentiment_words'] = data.apply(
        lambda x: tweet_ratio_sentiment_words(x['tweet__nr_neg_sentiment_words'], x['tweet__nr_of_sentiment_words']), axis=1)
    data['tweet__ratio_stopwords'] = data.apply(lambda x: tweet_ratio_tokens_before_after_prepro(
        NLPUtils.str_list_to_list(x['tweet__additional_preprocessed_text']),
        NLPUtils.str_list_to_list(x['tweet__additional_preprocessed_wo_stopwords'])), axis=1)

    # time features
    data['tweet__day_of_week'] = data['created_at'].apply(lambda x: int(x.weekday()))
    data['tweet__day_of_month'] = data['created_at'].apply(lambda x: int(x.day))
    # TODO: get local time
    # data['tweet__hour_of_day'] = data.apply(
    #     lambda x: TimeUtils.hour_of_day(TimeUtils.mysql_to_python_datetime(x['tweet__created_at']),
    #                                     x['user__utc_offset']),
    #     axis=1)

    return data


def tweet_ratio_all_capitalized_words(x):
    nr_of_words = x['tweet__nr_of_words']
    if nr_of_words == 0:
        return 0
    else:
        return nr_of_all_capitalized_words(
            NLPUtils.str_list_to_list(x['tweet__tokenized_um_url_removed'])) / nr_of_words


def tweet_ratio_capitalized_words(x):
    nr_of_words = x['tweet__nr_of_words']

    if nr_of_words == 0:
        return 0
    else:
        return nr_of_capitalized_words(NLPUtils.str_list_to_list(x['tweet__tokenized_um_url_removed'])) / nr_of_words


def tweet_ratio_uppercase_letters(x):
    upper_count = nr_of_cased_characters(NLPUtils.str_list_to_list(x))
    if upper_count == 0:
        return 0
    else:
        return tweet_count_upper_letters(NLPUtils.str_list_to_list(x)) / upper_count


def bigram_in_tweet_pos(pos_tags, bigram):
    """counts the number of POS bigrams in a tweet and normalizes it by the length of the tweet"""
    pos_tags = [token['tag'] for token in pos_tags]
    bigrams = list(ngrams(pos_tags, 2))
    count = 0
    if bigram in bigrams:
        count += 1

    if len(bigrams) == 0:
        return None
    return count / len(bigrams)


def find_frequent_pos_trigrams(data, min_doc_frequency=10000, no_above=0.5, keep_n=100):
    """
    finds trigrams that meat the frequency threshold for POS tags
    :param data: pandas dataframe
    :param threshhold: to minimum number of accurances in the data
    :return: 
    """

    tweets_tags = data['pos_tags'].tolist()

    trigram_pos = []
    for tweet in tweets_tags:
        tweet = json.loads(tweet)
        pos_tags = [token['tag'] for token in tweet]
        trigram_pos.append(NLPUtils.generate_n_grams(pos_tags, 3))

    pos_tri_model = TextModel()
    pos_tri_model.init_corpus(trigram_pos)
    return pos_tri_model.build_bag_of_words(variant='pos_trigram',tf_idf=True, min_doc_frequency=min_doc_frequency, no_above=no_above,
                                            keep_n=keep_n)


def get_user_lang_counts(tweet_df):
    """counts the languages used by a user. Does not count lang 'und' since language
    is automatically detected by twitter and almost every account contains a tweet with undefined language"""
    return len(tweet_df.groupby(['lang'])) - 1


def favourites_per_follower(x):
    """return #favourties/#followers"""
    return x['user__favourites_count'] / x['user__followers_count']


def friends_per_follower(x):
    """return #favourties/#followers"""
    return x['user__friends_count'] / x['user__followers_count']


def friends_per_favourite(x):
    """return #friends/#favourties, if favourites = 0 returns 0"""
    if x['user__favourites_count'] == 0:
        return 0
    else:
        return x['user__friends_count'] / x['user__favourites_count']


def contains_hashtag(x):
    """manually detect hashtags"""
    result = TextParser.find_all_hashtags(str(x))
    if result:
        return True
    else:
        return False


def contains_user_mention(x):
    result = TextParser.find_all_user_mentions(str(x))
    if result:
        return True
    else:
        return False


def contains_url(x):
    # SearchStr = '(http(s)?:\/\/.)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)'
    search_str = '(?P<url>https?://[^\s]+)'
    result = re.search(search_str, str(x))
    if result:
        return True
    else:
        return False


def string_length(x):
    if x is not None:
        return len(str(x))
    else:
        return 0


def is_translator_type(x):
    # potentially more classes
    if x == 'regular':
        return True
    else:
        return False


def tweet_contains_hashtags(entity_id):
    if pd.isnull(entity_id):
        return False
    else:
        hashtags = db.get_hashtags_of_tweet(entity_id)
        if len(hashtags) > 0:
            return True
        else:
            return False


def tweet_nr_of_hashtags(row):
    hashtags = row['hashtags']
    if pd.isna(hashtags):
        return 0
    return len(ast.literal_eval(hashtags) if hashtags else [])


def tweet_nr_of_urls(row):
    length_1 = len(TextParser.find_all_urls(row['text']))
    urls = row['urls']
    length_2 = len(ast.literal_eval(urls) if urls else [])

    if length_1 > length_2:
        return length_1
    else:
        return length_2


def tweet_contains_urls(entity_id, text):
    if pd.isnull(entity_id):
        return False
    else:
        length_1 = len(TextParser.find_all_urls(str(text)))
        length_2 = len(db.get_urls_of_tweet(entity_id))

        return (length_1 > 0) or (length_2 > 0)


def tweet_contains_media(entity_id):
    if pd.isnull(entity_id):
        return False
    else:
        media = db.get_media(entity_id)
        if len(media) > 0:
            return True
    return False


def tweet_nr_of_medias(entity_id):
    if pd.isnull(entity_id):
        return 0
    else:
        media = db.get_media(entity_id)
        if media is not None:
            return len(media)
    return 0


def tweet_contains_user_mention(entity_id):
    if pd.isnull(entity_id):
        return False
    else:
        media = db.get_user_mentions(entity_id)
        if len(media) > 0:
            return True
    return False


def tweet_nr_of_user_mentions(row):
    mentions = row['mentions']
    logging.debug(f'{mentions = }')
    if pd.isna(mentions):
        return 0
    return len(json.loads(mentions) if mentions else [])


def tweet_avg_url_length(row):
    """returns the length of an url in a tweet.
    If tweet contains more than one url, average length is returned"""
    parsed_urls = TextParser.find_all_urls(row['text'])
    urls = row['urls']
    urls_from_tweepy = ast.literal_eval(urls) if urls else []

    length_1 = len(parsed_urls) # this gives the number of urls
    length_2 = len(urls_from_tweepy)

    if length_1 == 0 and length_2 == 0:
        return 0
    elif length_1 > length_2:
        sum = 0
        for i in parsed_urls:
            sum += len(i)
        return sum / len(parsed_urls)
    else:
        sum = 0
        for i in urls_from_tweepy:
            sum += len(i)
        return sum / len(urls_from_tweepy)


def tweet_contains_link_to_users_website(user_url, entities_id):
    """returns true if one of the urls in the tweet link to the users website"""
    tweet_urls = db.get_manual_expanded_urls_of_tweet(entities_id)
    user_urls = db.get_manual_expanded_url_by_url(user_url)

    if len(user_urls) > 0:
        user_domain = UrlUtils.extract_domain(user_urls[0])
        for i in tweet_urls:
            domain = UrlUtils.extract_domain(i)
            if domain == user_domain:
                return True
    return False


def tweet_avg_expanded_url_length(entity_id):
    """returns the avg length of the expanded url"""
    urls = db.get_manual_expanded_urls_of_tweet(entity_id)
    sum = 0
    for i in urls:
        sum += len(i)
    if len(urls) == 0:
        return 0
    return sum / len(urls)


def one_of_tweet_urls_is_expandable(entity_id):
    """returns true if a tweet contains an shortened link"""
    urls = db.get_urls(entity_id)
    for u in urls:
        url = u['url']
        expanded_url = u['expanded_url']
        manual_expanded_url = u['manual_expanded_url']

        if re.match("www.*", url):
            url = "http://" + url

        url = re.sub("www\.", '', url)
        expanded_url = re.sub("www\.", '', expanded_url)
        manual_expanded_url = re.sub("www\.", '', manual_expanded_url)

        if url == expanded_url and expanded_url == manual_expanded_url:
            return False
        elif expanded_url is None:
            return False
        elif url != expanded_url:
            return True
        elif manual_expanded_url is None:
            return False
        elif url != manual_expanded_url:
            return True
        elif expanded_url != manual_expanded_url:
            return True


def get_top_level_domain_type(tld):
    """returns the type of the top level domain
    0: generic
    1: country-code"""
    type = UrlUtils.get_top_level_domain_type(tld)

    if type == "country-code":
        return 1
    else:
        return 0


def get_tweet_text_length(text):
    t = str(text)
    t = TextPreprocessor.unescape_html(t)
    norm_text = TextParser.normalize(t)
    length = len(norm_text)

    return length


def get_avg_post_time(tweet_df):
    """returns the average post time in hours of day"""
    return tweet_df['created_at'].dt.hour.mean()


def get_maximum_time_between_tweets(tweet_df):
    """calculates the maximum time between two tweets of a user"""
    if len(tweet_df) == 1:
        return np.nan
        # since all passed DataFrames represent either every tweet an account has made or the most recent 100 tweets, 
        # this edge case only occurs when an account tweets exactly once. That is basically never tweeting so I return np.nan
    sorted_tweets = tweet_df.sort_values(by='created_at')

    return sorted_tweets['created_at'].diff().max()


def get_minimum_time_between_tweets(tweet_df):
    """calculates the minimum time between two tweets of a user"""
    if len(tweet_df) == 1:
        return np.nan
        # since all passed DataFrames represent either every tweet an account has made or the most recent 100 tweets, 
        # this edge case only occurs when an account tweets exactly once. That is basically never tweeting so I return np.nan
    sorted_tweets = tweet_df.sort_values(by='created_at')

    return sorted_tweets['created_at'].diff().min()


def get_median_time_between_tweets(tweet_df):
    """calculates the median time between two tweets of a user"""
    if len(tweet_df) == 1:
        return np.nan
        # since all passed DataFrames represent either every tweet an account has made or the most recent 100 tweets, 
        # this edge case only occurs when an account tweets exactly once. That is basically never tweeting so I return np.nan
    sorted_tweets = tweet_df.sort_values(by='created_at')

    return sorted_tweets['created_at'].diff().median()


def get_avg_time_between_tweets(tweet_df):
    """calculates the avg time between two tweets of a user"""
    if len(tweet_df) == 1:
        return np.nan
        # since all passed DataFrames represent either every tweet an account has made or the most recent 100 tweets, 
        # this edge case only occurs when an account tweets exactly once. That is basically never tweeting so I return np.nan
    sorted_tweets = tweet_df.sort_values(by='created_at')

    return sorted_tweets['created_at'].diff().mean()


def get_tweets_per_day(tweet_df):
    """returns the average number of tweets per day. Imputed based on frequency of tweets per second in the sample"""
    duration = tweet_df['created_at'].max() - tweet_df['created_at'].min()
    duration_seconds = duration.total_seconds()
    if duration_seconds == 0:
        return 0
        # this would only occur if the earliest and latest tweet in tweet_df was the same tweet
        # since all passed DataFrames represent either every tweet an account has made or the most recent 100 tweets, 
        # this edge case only occurs when an account tweets exactly once. That is basically never tweeting so I return 0

    tweets_per_second = len(tweet_df) / duration_seconds

    return tweets_per_second / SECONDS_IN_DAY


def get_avg_user_mentions_per_tweet(tweet_df):
    """returns the average number of user mentions per tweet of user user_id"""
    num_tweets = len(tweet_df)
    total_mentions = tweet_df['mentions'].apply(lambda x: x.count('username') if x else 0).sum()

    return total_mentions // num_tweets


def get_avg_hashtags_per_tweet(tweet_df):
    """returns the average number of hashtags per tweet of user user_id"""
    num_tweets = len(tweet_df)
    total_hashtags = tweet_df['hashtags'].apply(lambda x: x.count('tag') if x else 0).sum()

    return total_hashtags // num_tweets


def get_avg_urls_per_tweet(tweet_df):
    """returns the average number of urls per tweet of user user_id"""
    num_tweets = len(tweet_df)
    total_urls = tweet_df['urls'].apply(lambda x: x.count('url') if x else 0).sum()

    return total_urls // num_tweets


def get_percent_with_url(tweet_df):
    """returns the percentage of tweets of user user_id that contains at least one url"""
    num_tweets = len(tweet_df)
    url_counts = tweet_df['urls'].apply(lambda x: x.count('url') if x else 0)
    at_least_one = url_counts.apply(lambda x: x >= 1).sum()

    return at_least_one / num_tweets


def get_percent_with_hashtag(tweet_df):
    """returns the percentage of tweets of user user_id that contains at least one hashtag"""
    num_tweets = len(tweet_df)
    tag_counts = tweet_df['hashtags'].apply(lambda x: x.count('tag') if x else 0)
    at_least_one = tag_counts.apply(lambda x: x >= 1).sum()

    return at_least_one / num_tweets


def get_percent_with_user_mention(tweet_df):
    """returns the percentage of tweets of user user_id that contains at least one user_mention"""
    num_tweets = len(tweet_df)
    mention_counts = tweet_df['mentions'].apply(lambda x: x.count('username') if x else 0)
    at_least_one = mention_counts.apply(lambda x: x >= 1).sum()

    return at_least_one / num_tweets


def get_nr_of_retweets_by_user(tweet_df):
    return tweet_df.groupby(['tweet_type']).size()['retweet']

def get_nr_of_quotes_by_user(tweet_df):
    return tweet_df.groupby(['tweet_type']).size()['qrt']

def get_nr_of_replies_by_user(tweet_df):
    return tweet_df.groupby(['tweet_type']).size()['reply']


def get_nr_of_retweets_per_tweet(tweet_df):
    nr_of_retweets = tweet_df.groupby(['tweet_type']).size()['retweet']

    return nr_of_retweets / len(tweet_df)


def get_nr_of_replies_per_tweet(tweet_df):
    nr_of_replies = tweet_df.groupby(['tweet_type']).size()['reply']

    return nr_of_replies / len(tweet_df)


def get_nr_of_quotes_per_tweet(tweet_df):
    nr_of_quotes = tweet_df.groupby(['tweet_type']).size()['qrt']

    return nr_of_quotes / len(tweet_df)


def get_possibly_sensitive(x):
    """returns the original value if it is not null, if there is a missing value, returns 2"""
    if pd.isnull(x):
        return 2
    return x


def tweet_nr_of_hashtags_in_popular_hashtags(entity_id, popular_hashtags):
    """returns the number of hashtags of a tweet that are in the most popular hashtags"""
    if pd.isnull(entity_id):
        return 0
    else:
        hashtags = db.get_hashtags_of_tweet(entity_id)
        count = 0
        for h in popular_hashtags:
            if h in hashtags:
                count += 1
        return count


def slang_words_in_tweet(tokens):
    """returns a list with the slang words found in the tweet"""
    slang = NLPUtils.get_slang_words()
    return [t for t in tokens if t in slang]


def tweet_avg_word_length(tokens):
    """returns the avg word length of a tokenized sentence"""
    sum = 0
    for t in tokens:
        t = TextPreprocessor.replace_all_punctuation(t)
        if t != "":
            if t[0] != '#' and t[0] != '@' and t[0] != '$' and t != '' and t.lower() != 'rt':
                sum += len(t)
    if len(tokens) == 0:
        return 0
    else:
        return sum / len(tokens)


def nr_of_capitalized_words(tokens):
    """returns the number of capitalized words. All capitalized words do not count"""
    count = 0
    for t in tokens:
        # do not count words like 'I'
        if len(t) > 1:
            if t[0].isupper():
                for i in range(1, len(t)):
                    if t[i].isupper():
                        break
                else:
                    count += 1
    return count


def nr_of_all_capitalized_words(tokens):
    """returns all words with only uppercase letters"""
    count = 0
    for t in tokens:
        if t != '':
            t = TextPreprocessor.replace_all_punctuation(t)
            if t != "":
                if t[0] != '#' and t[0] != '@' and t[0] != '$' and t != '' and t.lower() != 'rt':
                    for i in range(len(t)):
                        if t[i].islower():
                            break
                    else:
                        count += 1
    return count


def tweet_find_stock_mention(text):
    """finds stock mentions in the text (all words that start with $)"""
    stocks = list()
    tokens = text.split()
    for t in tokens:
        if len(t) > 0:
            if re.match("\$\w+", t):
                stocks.append(t)
    return stocks


def nr_of_cased_characters(tokens):
    """returns the number of alphabet characters in a tokenized tweet"""
    count = 0
    for token in tokens:
        for t in token:
            if t.isupper() or t.islower():
                count += 1
    return count


def tweet_count_upper_letters(tokens):
    """returns the number of upper case letters"""
    count = 0
    for token in tokens:
        for t in token:
            if t.isupper():
                count += 1
    return count


def contains_all_uppercase(text):
    """finds sequences of at least 5 uppercase characters."""
    text = TextPreprocessor.remove_user_mentions(text)
    text = TextPreprocessor.remove_hashtags(text)
    if re.findall(r'([A-Z]+[!.,]?(.)?){5,}', text):
        return True
    else:
        return False


def is_upper(text):
    """removes urls, hashtags and user mentions, then checks for isupper()"""
    text = TextPreprocessor.remove_urls(text)
    text = TextPreprocessor.remove_hashtags(text)
    text = TextPreprocessor.remove_user_mentions(text)
    return text.isupper()


def get_tag_ratio(tagged_text, tag, nr_of_words):
    """returns the ratio of tokens with a specific tag.
    tag: N: Noun, A: Adjective, V: Verb"""

    # tagged = json.loads(tagged_text)
    tagged = tagged_text
    a_count = 0
    for t in tagged:
        if t['tag'] == tag:
            token = TextPreprocessor.replace_all_punctuation(t['token'])
            if token != "":
                if token[0] != '#' and token[0] != '@' and token[0] != '$' and token != '' and token.lower() != 'rt':
                    a_count += 1
    if nr_of_words == 0:
        return 0
    else:
        return a_count / nr_of_words


def tweet_contains_character_repetitions(text):
    # look for a character followed by at least two repetition of itself.
    pattern = re.compile(r"(.)\1{2,}")
    if re.findall(pattern, text):
        return True
    else:
        return False


def get_nr_of_words(tokens):
    "counts all tokens except punctuation, user mentions, stock or hashtags"
    count = 0
    for t in tokens:
        if t != '':
            t = TextPreprocessor.replace_all_punctuation(t)
            if t != "":
                if t[0] != '#' and t[0] != '@' and t[0] != '$' and t != '' and t.lower() != 'rt':
                    count += 1
    return count


def ratio_punctuation_tokens(tokenized_text, nr_of_tokens):
    if nr_of_tokens == 0:
        return 0
    else:
        tokens = NLPUtils.str_list_to_list(tokenized_text)

        punctuation = NLPUtils.get_punctuation()
        count = 0
        for token in tokens:
            if token in punctuation:
                count += 1

        return count / nr_of_tokens


def get_top_level_domain_of_expanded_url(url):
    """looks up the expanded versions of the url and returns the top level domain of the most expanded"""
    if url is not None:
        url = db.get_url(url)

        manual_expanded_url = url['manual_expanded_url']
        expanded_url = url['expanded_url']

        if manual_expanded_url is not None:
            return UrlUtils.get_top_level_domain(manual_expanded_url)
        elif expanded_url is not None:
            return UrlUtils.get_top_level_domain(expanded_url)
        else:
            return UrlUtils.get_top_level_domain(url)
    else:
        return ""


def tweet_contains_named_entities(pos_tags):
    """returns True if the tweets contain at 
    least one token that is tagged as an named entity"""
    for token in pos_tags:
        if token["tag"] == '^':
            return True
    return False


def get_nr_of_punctuation(text):
    """counts punctuation"""
    cnt = Counter()
    punctuation = NLPUtils.get_punctuation()
    for t in text:
        if t in punctuation:
            cnt[t] += 1
    return cnt


def tweet_ratio_tokens_before_after_prepro(tokens_before, tokens_after):
    """calculates the ratio of the number of tokens before 
    to the number of tokens after additional preprocessing"""
    if not tokens_before:
        return 0
    else:
        return len(tokens_after) / len(tokens_before)


def tweet_quarter(month):
    """returns the quarter of the year"""
    if month <= 3:
        return 0
    elif 3 < month <= 6:
        return 1
    elif 6 < month <= 9:
        return 2
    elif 9 < month:
        return 3


def remove_um_url(tokens):
    """
    replaces urls and user mentions
    :param tokens: 
    :return: 
    """
    new = list()
    for token in tokens:
        token = TextPreprocessor.remove_urls(token)
        token = TextPreprocessor.remove_user_mentions(token)
        token = TextPreprocessor.remove_hashtags(token)
        if token != "":
            new.append(token)
    return str(new)


def contains_number(tokens):
    """
    True, if tweet contains a token that is a number
    :param tokens: 
    :return: 
    """
    for token in tokens:
        if re.match('^\d+$', token):
            return True
    return False


def contains_quote(text):
    """
    finds quotes in a text
    :param text: 
    :return: 
    """
    res = re.findall('"([^"]*)"', text)
    if res:
        return True
    else:
        return False


def tweet_ratio_sentiment_words(pos_neg, nr_sent_words):
    if nr_sent_words == 0:
        return 0
    else:
        return pos_neg / nr_sent_words


if __name__ == "__main__":
    almost = pd.read_parquet('almost.parquet.gzip')
    done = create_features(almost)
    done.to_parquet('done.parquet.gzip', compression='gzip', index=False)


