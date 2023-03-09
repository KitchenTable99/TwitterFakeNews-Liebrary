import json
import pandas as pd
import sys
from collections import Counter
from nltk import TweetTokenizer, sent_tokenize
from Emoji import Emojis
from NLPUtils import NLPUtils
from SentimentAnalysis import SentimentAnalysis
from TextPreprocessor import TextPreprocessor

off_abbrevs = NLPUtils.get_official_abbreviations()
sl_abbrevs = NLPUtils.get_slang_abbreviations()
curr_thread_nr = Counter()

# TODO: add pos tags


def insert_tokenized_tweets(data: pd.DataFrame):
    """tokenizes tweets and inserts them into the db"""
    tt = TweetTokenizer()
    data['tweet__tokenized_text'] = data['text'].apply(lambda x: TextPreprocessor.tokenize_tweet(tt, x))


def insert_sent_tokenized_tweets(data: pd.DataFrame):
    """sentence tokenizes a tweets text"""
    data['tweet__sent_tokenized_text'] = data.apply(lambda row: sent_tokenize(TextPreprocessor.preprocess_for_sent_tokenize(row['text'], row['unicode_emojis'], row['ascii_emojis'])), axis=1)


def insert_additional_preprocessed_text(data: pd.DataFrame):
    additional, spelling = data['pos_tags'].apply(lambda x: TextPreprocessor.additional_text_preprocessing_with_pos(json.loads(x)))
    data['tweet__additional_preprocessed_text'] = additional
    data['tweet__contains_spelling_mistake'] = spelling


def parse_ascii_emojis_into_db(data: pd.DataFrame):
    """
    parses the ascii emojis in the tweets' text into a database column
    :return: 
    """
    emojis = Emojis.read_ascii_emojis()
    data['tweet__ascii_emojis'] = data['text'].apply(lambda x: Emojis.find_ascii_emojis(TextPreprocessor.remove_urls(x), emojis))


def parse_unicode_emojis_into_db(data: pd.DataFrame):
    """parses the unicode emojis into a database column"""
    data['tweet__unicode_emojis'] = data['text'].apply(lambda x: Emojis.find_unicode_emojis(x))


def insert_additional_preprocessed_text_wo_stopwords(data: pd.DataFrame):
    """
    removes stopwords from additional_preprocessed_text and inserts result into database
    :return: 
    """
    stopwords = NLPUtils.get_stopwords()
    data['tweet__additional_preprocessed_text_wo_stopwords'] = data['tweet__additional_preprocessed_text'].apply(lambda x: str([token for token in x if token not in stopwords]))


if __name__ == "__main__":
    # run from command line to use various threads
    bin = None
    if sys.argv[1:]:
        bin = sys.argv[1]

    testset=True

    # Shows which of the preprocessing columns have not yet been completed
    # db.preprocessing_values_missing(testset)
    # db.clear_column(tablename="tweet", columnname="tokenized_text")

    # parse_ascii_emojis_into_db(testset, bin)
    # parse_unicode_emojis_into_db(testset, bin)
    #
    # insert_tokenized_tweets(testset, bin)
    # insert_sent_tokenized_tweets(testset, bin)

    # based on POS-tags:
    # insert_contains_spelling_mistake(testset, bin)
    # insert_additional_preprocessed_text(testset, bin)
    # insert_additional_preprocessed_text_wo_stopwords(testset, bin)
    #
    # SentimentAnalysis.insert_sentiment_scores(testset)
    # SentimentAnalysis.insert_polarity_score(testset)
    SentimentAnalysis.insert_nr_pos_neg_words(testset=False)
    #
    # insert_is_trending_topic(testset)
    # insert_is_local_trending_topic()

    # replace_emoji_in_ascii_emojis()
    #
