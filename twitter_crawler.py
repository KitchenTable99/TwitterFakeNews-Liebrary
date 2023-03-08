import glob
import logging
import pickle
import pandas as pd
from tqdm import tqdm as progress
from typing import Callable, Generator, List, Literal, Set, Tuple, TypeVar

from UserFeatureCreator import get_depth2_tweet_features as extract_depth_features


CrawlerType = Literal['test', 'harvard', 'likes', 'left', 'depth_2']
T = TypeVar('T')


DATA_PATH = '/home/ec2-user/congressional_tweets'


class TweetCrawler:

    def __init__(self, crawler_type: CrawlerType):
        self.paths = self.read_dfs(crawler_type)
        if crawler_type == 'test':
            self.parquet_count = 1
        elif crawler_type == 'full':
            self.parquet_count = 11
        else:
            self.parquet_count = -1

    def read_dfs(self, crawler_type: CrawlerType):
        if crawler_type == 'harvard':
            paths = [file for file in glob.glob(f'{DATA_PATH}/full_harvard/*parquet*', recursive=True) if 'likes' not in file]
        elif crawler_type == 'test':
            paths = [f'{DATA_PATH}/senators_115.parquet.gzip']
        elif crawler_type =='left':
            paths = [file for file in glob.glob(f'{DATA_PATH}/**/*parquet*', recursive=True) if 'likes' in file or 'retweets' in file]
        elif crawler_type == 'depth_2':
            paths = [file for file in glob.glob(f'{DATA_PATH}/depth_2/*parquet*', recursive=True)]
        else:
            paths = [file for file in glob.glob(f'{DATA_PATH}/**/*parquet*', recursive=True) if 'likes' in file]

        return paths

    def apply_function(self, callable: Callable[[pd.DataFrame], T]) -> Generator[T, None, None]:
        for df_path in progress(self.paths):
            df = pd.read_parquet(df_path)
            yield callable(df)


def get_ids(df: pd.DataFrame) -> Set[int]:
    return set(df['id'].to_list())


def get_count(df: pd.DataFrame) -> int:
    return len(df['id'])


def get_count_by_document(df: pd.DataFrame) -> pd.DataFrame:
    df['day'] = pd.DatetimeIndex(df.created_at).normalize()
    return df.groupby(['day']).count()


def get_user_ids(df: pd.DataFrame) -> Set[int]:
    return set(df['author_id'].to_list())


def get_retweets(df: pd.DataFrame) -> pd.DataFrame:
    return df[df['tweet_type'] == 'retweet']


def test():
    left = TweetCrawler('left')
    retweets = left.apply_function(get_user_ids)

    for i in retweets:
        print(i)


def main():
    left_crawler = TweetCrawler('harvard')
    retweets = left_crawler.apply_function(get_retweets)
    all_op_tweets = set()

    for retweet_df in progress(retweets):
        original_tweets = set(retweet_df['original_tweet_id'].to_list())
        all_op_tweets.update(original_tweets)

    with open('originals.txt', 'w') as fp:
        for user_id in all_op_tweets:
            fp.write(f'{user_id:.20f}')
            fp.write('\n')


def get_user_tweets(df):
    return df.groupby(['author_id'])


def extract_depth_2():
    pd.options.mode.chained_assignment = None
    likes_crawler = TweetCrawler('depth_2')
    grouped_dfs = likes_crawler.apply_function(get_user_tweets)

    with open('fuckups.pickle', 'rb') as fp:
        fuckups = pickle.load(fp)
    fuckups_dict = {f: [] for f in fuckups}

    user_depth_features = []

    for grouped_df in grouped_dfs:
        for group in progress(grouped_df.groups, leave=False):
            if group in fuckups:
                fuckups_dict[group].append(grouped_df.get_group(group))
            else:
                depth_features = extract_depth_features(grouped_df.get_group(group))
                user_depth_features.append(depth_features)

    for fuckup_list in fuckups_dict.values():
        single_df = pd.concat(fuckup_list)
        depth_features = extract_depth_features(single_df)
        user_depth_features.append(depth_features)

    all_depth_features = pd.DataFrame(user_depth_features)
    print(all_depth_features)

    all_depth_features.to_parquet('user_depth_features.parquet.gzip', compression='gzip', index=False)

def likes():
    likes_crawler = TweetCrawler('depth_2')
    author_ids = likes_crawler.apply_function(get_user_ids)
    fuckups = set()
    seen = set()
    for id_set in author_ids:
        for id in id_set:
            if id in seen:
                fuckups.add(id)
            else:
                seen.add(id)

    print(fuckups)
    with open('fuckups.pickle', 'wb') as fp:
        pickle.dump(fuckups, fp)

    # with open('liked_user_ids.txt', 'w') as fp:
    #     for user_id in total_ids:
    #         fp.write(str(user_id))
    #         fp.write('\n')

    # likes_gb = likes_crawler.apply_function(get_count)
    # count = 0
    # for a in likes_gb:
    #     count += a
    # print(count)
    # counts = pd.concat(list(likes_gb), axis=1).sum(axis=1)
    # with open('counts.pickle', 'wb') as fp:
    #     pickle.dump(counts, fp)


if __name__ == "__main__":
    logging.basicConfig(filename='twetter_crawler.log', encoding='utf-8', level=logging.INFO, format='%(asctime)s - %(message)s')
    extract_depth_2()
