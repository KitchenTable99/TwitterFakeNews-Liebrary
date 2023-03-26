import glob
import os
from FeatureCreator import create_features
import logging
import pickle
import pandas as pd
from tqdm import tqdm as progress
from typing import Callable, Generator, Literal, Set, TypeVar
from PreprocesIntoDB import preprocess_df as preprocess_function

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


def get_user_ids(df: pd.DataFrame) -> Set[int]:
    return set(df['author_id'].to_list())


def get_user_tweets(df):
    return df.groupby(['author_id'])


def preprocess_likes():
    left_crawler = TweetCrawler('likes')
    tweets = left_crawler.apply_function(preprocess_function)

    count = 0
    for processed_df in tweets:
        if processed_df is None:
            continue
        write_name = f'processed_likes_{count}'
        count += 1

        processed_df.to_parquet(write_name,
                                index=False,
                                compression='gzip')


def create_tweet_features():
    files = os.listdir('./processed/')
    for file in progress(files):
        if os.path.exists(f'./real_processed/{file}'):
            continue
        file_path = f'./processed/{file}'
        df = pd.read_parquet(file_path)
        tweets = create_features(df)
        if tweets is None:
            return

        tweets.to_parquet(f'./real_processed/{file}',
                           index=False,
                           compression='gzip')



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

    with open('depth_features.pickle', 'wb') as fp:
        pickle.dump(all_depth_features, fp)
    all_depth_features.to_parquet('user_depth_features.parquet.gzip', compression='gzip', index=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    create_tweet_features()
