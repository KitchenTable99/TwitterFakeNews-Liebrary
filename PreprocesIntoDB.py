import pandas as pd
from tqdm import tqdm
from collections import Counter
from nltk import TweetTokenizer, sent_tokenize, pos_tag
from Emoji import Emojis
from NLPUtils import NLPUtils
from SentimentAnalysis import SentimentAnalysis
from TextPreprocessor import TextPreprocessor

off_abbrevs = NLPUtils.get_official_abbreviations()
sl_abbrevs = NLPUtils.get_slang_abbreviations()
curr_thread_nr = Counter()
tqdm.pandas()


def insert_pos_tags(data: pd.DataFrame):
    pos_list = data['tweet__tokenized_text'].apply(pos_tag)
    pos_dicts = [[{'token': tup[0], 'tag': tup[1]} for tup in tup_list] for tup_list in pos_list]
    df['pos_tags'] = pos_dicts


def insert_tokenized_tweets(data: pd.DataFrame):
    """tokenizes tweets and inserts them into the db"""
    tt = TweetTokenizer()
    data['tweet__tokenized_text'] = data['text'].apply(lambda x: TextPreprocessor.tokenize_tweet(tt, x))


def insert_sent_tokenized_tweets(data: pd.DataFrame):
    """sentence tokenizes a tweets text"""
    data['tweet__sent_tokenized_text'] = data.apply(lambda row: sent_tokenize(TextPreprocessor.preprocess_for_sent_tokenize(row['text'], row['tweet__unicode_emojis'], row['tweet__ascii_emojis'])), axis=1)


def insert_additional_preprocessed_text(data: pd.DataFrame):
    results = data['pos_tags'].apply(lambda x: TextPreprocessor.additional_text_preprocessing_with_pos(x))
    additional, spelling = zip(*results)
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
    data['tweet__additional_preprocessed_wo_stopwords'] = data['tweet__additional_preprocessed_text'].apply(lambda x: str([token for token in x if token not in stopwords]))


if __name__ == "__main__":
    df = pd.read_parquet('likes.parquet.gzip')
    insert_tokenized_tweets(df)
    insert_pos_tags(df)

    parse_ascii_emojis_into_db(df)
    parse_unicode_emojis_into_db(df)

    insert_sent_tokenized_tweets(df)

    SentimentAnalysis.insert_sentiment_scores(df)
    SentimentAnalysis.insert_nr_pos_neg_words(df)
    SentimentAnalysis.insert_subjectivity_score(df)

    # df = pd.read_parquet('almost.parquet.gzip')
    # based on POS-tags:
    insert_additional_preprocessed_text(df)
    insert_additional_preprocessed_text_wo_stopwords(df)
    df.to_parquet('almost.parquet.gzip', compression='gzip', index=False)



    # replace_emoji_in_ascii_emojis()
