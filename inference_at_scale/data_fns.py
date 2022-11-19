import pickle
import gzip
from glob import glob
from tqdm import tqdm
from datetime import datetime

from datasets import load_dataset


def open_realnews():

    start = datetime.now()
    file_list = [f'/c4/realnewslike/c4-train.{str(i).zfill(5)}-of-00512.json.gz' for i in range(512)]
    file_list.extend(glob('/c4/realnewslike/c4-validation**'))

    corpus = []
    print("Loading data ...")
    for file in tqdm(file_list):

        with gzip.open(file, 'r') as fin:
            json_bytes = fin.read()
            json_str = json_bytes.decode('utf-8')
            str_split = json_str.split("\n")
            for string in str_split:
                if len(string) != 0:
                    text = string.split('"text":"')[1].split('","timestamp"')[0]
                    corpus.append(text)

    print(len(corpus), "files in corpus")
    print("Time taken:", datetime.now() - start)

    return corpus


def open_c4_by_url(pattern="patents.google.com", name="patents"):

    start = datetime.now()

    full_corpus = []
    for set in ['train', 'validation']:
        file_list = glob(f'/c4/en/c4-{set}**.json.gz')

        corpus = []
        print("Loading data ...")
        for file in tqdm(file_list):

            with gzip.open(file, 'r') as fin:
                json_bytes = fin.read()
                json_str = json_bytes.decode('utf-8')
                str_split = json_str.split("\n")
                for string in str_split:
                    if len(string) != 0:

                        url = string.split('"url":"')[1].split('"')[-2]
                        if pattern in url:

                            text = string.split('"text":"')[1].split('","timestamp"')[0]
                            corpus.append(text)

        print(len(corpus), f"files in {set} set")
        print("Time taken:", datetime.now() - start)

        with open(f"/c4/{name}_{set}.pkl", "wb") as f:
            pickle.dump(corpus, f, protocol=4)

        full_corpus.extend(corpus)

    return corpus


def get_super_glue():
    ''' Retrieves data from SuperGLUE. '''

    # Which sections of data to load for SuperGLUE
    split_dict = {
        'wsc': [['text']],
        'boolq': [['passage']],
        'cb': [['premise', 'hypothesis']],
        'copa': [['premise', 'choice1'], ['premise', 'choice2']],
        'multirc': [['paragraph']],
        'record': [['passage']],
        'rte': [['premise', 'hypothesis']],
        'wic': [['sentence1'], ['sentence2']],
        }

    for ds in split_dict.keys():
        print(f"**** {ds} ****")

        # Select text data
        dataset = load_dataset("super_glue", ds)

        dev_set = dataset['validation']
        texts = []
        for feat_list in split_dict[ds]:
            for i in range(len(dev_set)):
                text_list = [dev_set[feat][i] for feat in feat_list]
                texts.append(" ".join(text_list))

        print(len(texts), "texts in corpus")
        corpus = list(set(texts))
        print(len(corpus), "texts in corpus after deduplication")

    return corpus