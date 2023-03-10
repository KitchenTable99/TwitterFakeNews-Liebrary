import pandas as pd
import re

from nltk import TabTokenizer
from nltk.corpus import sentiwordnet as swn
from nltk.wsd import lesk
from textblob import TextBlob

from Emoji import Emojis, NLPUtils


class SentimentAnalysis:

    EMOJI_SENTS = Emojis.read_unicode_emoji_sents_map()
    EMOJI_SENTS_ASCII = Emojis.read_ascii_emoji_sents_map()

    @staticmethod
    def score_word_sentiment(word, pos_tag, tweet_pos):
        """
        :param word: word to score
        :param pos_tag: n - NOUN, v - VERB, a - ADJECTIVE, r - ADVERB
        :param tweet_pos: POS tagged tokens
        :return: sentiment score (pos_score - neg_score)
        """
        from TextPreprocessor import TextPreprocessor
        pos_tag = pos_tag.lower()
        is_adjective = 'jj' in pos_tag
        is_verb = 'vb' in pos_tag
        is_noun = 'nn' in pos_tag
        is_adverb = 'rb' in pos_tag
        allowed = is_adjective or is_verb or is_noun or is_adverb
        if allowed:
            word = TextPreprocessor.lemmatize(word, pos_tag)
            synset = SentimentAnalysis.disambiguate_word(word, pos_tag, tweet_pos)
            if synset is None:
                to_return = 0.0
                # print('synset was None')
            else:
                sent_word = swn.senti_synset(synset.name())
                to_return = sent_word.pos_score()-sent_word.neg_score()
                # print('found_score in synset')
        elif re.match("U\+.{4,5}", word):
            # scores the unicode emojis
            # print('scoring unicode emoji')
            to_return = SentimentAnalysis.EMOJI_SENTS[word]['score']
        elif word in SentimentAnalysis.EMOJI_SENTS_ASCII:
            # print('scoring ascii emoji')
            to_return = SentimentAnalysis.EMOJI_SENTS_ASCII[word]['score']
        else:
            # print('other token passed')
            to_return = 0.

        return to_return

    @staticmethod
    def disambiguate_word(word, tag, tweet_pos):
        """
        disambiguates a word in a tweet
        :param word: word to disambiguate
        :param tag: pos tag of the word
        :param tweet_pos: list with POS tagged tokens
        :return: best matching Synset
        """
        sent = [key["token"] for key in tweet_pos]
        wordnet_tag = NLPUtils.get_wordnet_pos(tag)
        return lesk(sent, word, wordnet_tag)

    @staticmethod
    def score_tweet_sentiment(tweet_pos):
        """
        Scores a tweet according to its sentiment
        :param tweet_pos: POS tagged tweet
        :return: summed up sentiment score, number of sentiment words
        """
        score = 0
        nr_sent_words = 0
        for t in tweet_pos:
            score_t = SentimentAnalysis.score_word_sentiment(t["token"], t["tag"],tweet_pos)
            # print("{} (score: {})".format(t["token"], score_t))
            score += score_t
            if score_t != 0:
                nr_sent_words += 1
        if score != 0:
            score = SentimentAnalysis.normalize_score(score/nr_sent_words)

        return score, nr_sent_words

    @staticmethod
    def count_pos_neg_sentiment_words(tweet_pos):
        """
        counts positive and negative sentiment words
        :param tweet_pos: POS tagged tweet
        :return: summed up sentiment score, number of sentiment words
        """
        nr_pos_words = 0
        nr_neg_words = 0
        for t in tweet_pos:
            score_t = SentimentAnalysis.score_word_sentiment(t["token"], t["tag"],tweet_pos)
            # print("{} (score: {})".format(t["token"], score_t))
            if score_t > 0:
                nr_pos_words += 1
            if score_t < 0:
                nr_neg_words += 1
        return nr_pos_words, nr_neg_words


    @staticmethod
    def insert_nr_pos_neg_words(data: pd.DataFrame):
        returned = data['pos_tags'].apply(SentimentAnalysis.count_pos_neg_sentiment_words)
        n_pos, n_neg = zip(*returned.values)
        data['tweet__nr_pos_sentiment_words'] = n_pos
        data['tweet__nr_neg_sentiment_words'] = n_neg


    @staticmethod
    def insert_sentiment_scores(data: pd.DataFrame):
        """
        inserts a column with the sentiment score as well as the number of sentiment words in the text
        :return: -
        """
        returned = data['pos_tags'].apply(SentimentAnalysis.score_tweet_sentiment)
        score, nr_sent_words = zip(*returned.values)
        data['tweet__sentiment_score'] = score
        data['tweet__nr_of_sentiment_words'] = nr_sent_words


    @staticmethod
    def assess_subjectivity(pos_tags):
        """
        determines the polarity of sentence with the TextBlob library
        :param pos_tags: 
        :return: 
        """
        from textblob.en.sentiments import PatternAnalyzer
        from NLPUtils import NLPUtils
        words = list()
        for token in pos_tags:
            word = token['token']
            pos_tag = token['tag']
            # allowed_tags = ['a','n','v','r']
            # if pos_tag.lower() in allowed_tags:
            #     word = TextPreprocessor.lemmatize(word, pos_tag.lower())
            #     pass
            word = Emojis.remove_unicode_emojis(word)
            # if pos_tag != '#' and pos_tag != '@' and pos_tag != 'U' and pos_tag != 'E' and word not in NLPUtils.get_punctuation():
            if pos_tag != '#' and pos_tag != '@' and pos_tag != 'U' and word not in NLPUtils.get_punctuation():
                words.append(word)
        text = "\t".join(words)
        tokenizer = TabTokenizer()
        testimonial3 = TextBlob(text, analyzer=PatternAnalyzer(), tokenizer=tokenizer)
        subjectivity = testimonial3.sentiment.subjectivity
        return subjectivity

    @staticmethod
    def insert_subjectivity_score(data: pd.DataFrame):
        """insert polarity"""
        data['tweet__subjectivity_score'] = data['pos_tags'].apply(SentimentAnalysis.assess_subjectivity)

    @staticmethod
    def normalize_score(score):
        """
        normalizes the score to a range from 0 to 1
        :param score: 
        :return: 
        """
        return (score + 1)/2

