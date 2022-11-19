"""
Python 3.8
Compares SuperGlue to RealNews and pulls out duplicates between the sets.
"""

import pickle
import json
from tqdm import tqdm
from datetime import datetime
import math
import numpy as np
import os

from datasets import load_dataset
import faiss

import data_fns
from biencoder_at_scale import embed


def find_real_news_neighbours(query_embeddings, corpus_embeddings, k=900, d=768, corpus_batch_size=10000000):

    """
    Pull k nearest neighbours to every text in query embeddings from each batch of corpus embeddings
    """

    start_time = datetime.now()

    # Initialise faiss
    res = faiss.StandardGpuResources()

    n_corpus_batches = math.ceil(corpus_embeddings.shape[0] / corpus_batch_size)
    print("Total batches:", n_corpus_batches)

    # Batch over corpus
    dist_list = []
    nn_list = []

    for j in range(n_corpus_batches):

        print(f"***** Corpus batch {j} *****")

        gpu_index_flat = faiss.GpuIndexFlatIP(res, d)
        gpu_index_flat.add(corpus_embeddings[(corpus_batch_size * j):corpus_batch_size * (j + 1)])

        D, I = gpu_index_flat.search(query_embeddings, k)
        dist_list.append(D)

        # Adjust IDs for not starting at 0
        I_adj = I + j * corpus_batch_size
        nn_list.append(I_adj)

        gpu_index_flat.reset()

    end_time = datetime.now()
    print("total time elapsed: ", (end_time - start_time))

    dist_list = np.concatenate(dist_list, axis=1)
    nn_list = np.concatenate(nn_list, axis=1)

    print(dist_list.shape)
    print(nn_list.shape)

    return dist_list, nn_list


def compare_to_superglue(rn_embedding_file_list, biencoder_model, biencoder_threshold, working_directory):

    """
    Find duplicates between realnews and all the test sets in superglue.
    Biencoder embeddings are loaded from saved file (can be computed in biencoder_at_scale.py)
    SuperGLUE texts downloaded from Datasets.
    Texts of duplicates are saved in 'pairs'
    """

    os.makedirs(working_directory, exist_ok=True)
    os.makedirs(f'{working_directory}/embeddings', exist_ok=True)
    os.makedirs(f'{working_directory}/pairs', exist_ok=True)

    # Load real news embeddings
    print("Loading real news embeddings from...", rn_embedding_file_list)
    rn_embeddings = []
    for file in tqdm(rn_embedding_file_list):
        with open(file, 'rb') as f:
            rn_embeddings.append(pickle.load(f))
    real_news_embeddings = np.concatenate(rn_embeddings, axis=0)

    # Load real news texts
    real_news_corpus = data_fns.open_realnews()

    # Load SuperGlue data
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
        print(len(corpus), "texts in corpus after removing exact duplicates")

        # Embed
        embeddings = embed(
            corpus,
            trained_model=biencoder_model,
            save_stem=f'{working_directory}/embeddings/{ds}',
            batch_size=1024
        )

        # Compare to real news embeddings
        dist, nn = find_real_news_neighbours(embeddings, real_news_embeddings, k=900, d=768, corpus_batch_size=5000000)

        # Subset to neighbours under threshold
        under_th = [nn[i][(dist[i] >= biencoder_threshold)] for i in range(len(nn))]

        # Convert to edges
        nn_edge_list = [(i, j) for i in range(len(under_th)) for j in under_th[i] if j != i]
        print("Pairs found:", len(nn_edge_list))

        print(f"Number of {ds} texts with at least one duplicate:", len(set([edge[0] for edge in nn_edge_list])))

        # Match back to texts
        out_list = []
        for edge in nn_edge_list:
            out_list.append({
                str(edge[0]): str(corpus[edge[0]]),
                str(edge[1]): str(real_news_corpus[edge[1]]),
            })

        with open(f'{working_directory}/pairs/{ds}.json', 'w') as f:
            json.dump(out_list, f, indent=4)


if __name__ == '__main__':

    rn_embedding_file_list = [f"/embeddings/real_news_like_embeddings_{i}.pkl" for i in range(0, 21)]

    compare_to_superglue(
        rn_embedding_file_list=rn_embedding_file_list,
        biencoder_model='',
        biencoder_threshold=0.94,
        working_directory=''
    )


