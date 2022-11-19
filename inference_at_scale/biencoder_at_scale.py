"""
Python 3.8
Runs biencoder inference over C4 at scale.
"""

import pickle
from tqdm import tqdm
from glob import glob
import os
import sys

from datetime import datetime
import numpy as np
import math

import faiss
from sentence_transformers import SentenceTransformer

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)
sys.path.append(grandparentdir)

import utils
import data_fns

def embed(
        corpus,
        trained_model,
        save_stem,
        batch_size=64,
):

    """
    Embeds a list of text (corpus) using supplied biencoder model. Saves embeddings in chunks of 1M .
    """

    # Initialise model
    embedder = SentenceTransformer(trained_model)

    # Embed data
    print("Embedding corpus ...")
    all_embeddings = []

    chunk_size = 1000000
    nchunks = math.ceil(len(corpus)/chunk_size)

    for i in range(nchunks):

        print(f"Chunk {i}/{nchunks}")

        corpus_embeddings = embedder.encode(corpus[(chunk_size*i):chunk_size*(i+1)], show_progress_bar=True, batch_size=batch_size)

        # Normalize the embeddings to unit length
        corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

        with open(f'{save_stem}_{i}.pkl', 'wb') as f:
            pickle.dump(corpus_embeddings, f)

        all_embeddings.append(corpus_embeddings)

    corpus_embeddings = np.concatenate(all_embeddings, axis=0)
    print(len(corpus_embeddings), "embeddings in corpus")

    return corpus_embeddings


def find_nearest_neighbours(embedding_list, save_stem, k=5, d=768, query_batch_size=1000, corpus_batch_size=10000000, normalize=False):

    """
    Takes list of embeddings and compares each embedding to each other, retuning k nearest neighbours, per corpus batch
    Nearest neighbours and distances are saved.
    Batches approach https://davidefiocco.github.io/nearest-neighbor-search-with-faiss/
    """

    if normalize:
        faiss.normalize_L2(embedding_list)
    
    start_time = datetime.now()

    # Initialise faiss
    res = faiss.StandardGpuResources()

    n_query_batches = math.ceil(embedding_list.shape[0] / query_batch_size)
    n_corpus_batches = math.ceil(embedding_list.shape[0] / corpus_batch_size)
    print("Total batches:", n_query_batches * n_corpus_batches)

    # Batch over corpus
    for j in range(n_corpus_batches):

        print(f"***** Corpus batch {j} *****")

        gpu_index_flat = faiss.GpuIndexFlatIP(res, d)
        gpu_index_flat.add(embedding_list[(corpus_batch_size*j):corpus_batch_size*(j+1)])

        for i in range(n_query_batches):
            print(f"\n Corpus batch {j}, query batch {i}: {(i*j) +i}/{n_query_batches * n_corpus_batches}")
            start_batch = datetime.now()
            D, I = gpu_index_flat.search(embedding_list[(query_batch_size*i):query_batch_size*(i+1)], k)
            end_batch = datetime.now()

            print("Time taken by batch was ", end_batch-start_batch)
            print("Saving intermediate results...")

            # Adjust IDs for not starting at 0
            I_adj = I + j*corpus_batch_size

            # Save intermediate results for this batch
            with open(f"{save_stem}/nn_list_batch_{i}_{j}.pkl", "wb") as f:
                pickle.dump(I_adj, f, protocol=4)
            with open(f"{save_stem}/dist_list_batch_{i}_{j}.pkl", "wb") as f:
                pickle.dump(D, f, protocol=4)

        gpu_index_flat.reset()

    end_time = datetime.now()
    print("total time elapsed: ", (end_time-start_time))


def subset_nn_data(saved_stem, threshold):

    """
    Reloads nearest neighbours, subsets to those below a distance threshold and compiles across multiple batches.
    """

    print("Reloading nearest neighbours ...")

    nn_files = glob(f"{saved_stem}/dist_list_batch*.pkl")

    i_list = set([int(file.split("_")[-2]) for file in nn_files])
    j_list = set([int(file.split("_")[-1].split(".")[0]) for file in nn_files])
    i_list = sorted(i_list)
    j_list = sorted(j_list)

    under_th = []

    for i in tqdm(i_list):  # For all batches of queries

        dist_list = []
        nn_list = []

        for j in j_list:   # Grab results for all batches of the corpus
            with open(f"{saved_stem}/dist_list_batch_{i}_{j}.pkl", 'rb') as f:
                dist_list.append(pickle.load(f))
            with open(f"{saved_stem}/nn_list_batch_{i}_{j}.pkl", 'rb') as f:
                nn_list.append(pickle.load(f))

        dist_list = np.concatenate(dist_list, axis=1)
        nn_list = np.concatenate(nn_list, axis=1)

        under_th.extend([nn_list[i][(dist_list[i] >= threshold)] for i in range(len(nn_list))])

    print(len(under_th))

    return under_th


def nearest_neighbours_to_pairs(final_under_thresh, save_dir):

    """
    Reformat nearest neighbour list as pairs
    """

    print("Sanity check ...")
    for i in tqdm(range(len(final_under_thresh))):
        assert i in final_under_thresh[i]

    print("Creating pairs ...")
    edge_list = [(i, j) for i in range(len(final_under_thresh)) for j in final_under_thresh[i] if j != i]

    # Remove edges that are in twice
    print("Removing duplicate edges ...")
    edge_list = list({*map(tuple, map(sorted, edge_list))})

    print("Total edges:", len(edge_list))

    # Save
    with open(f'{save_dir}/nn_edge_list.pkl', 'wb') as f:
        pickle.dump(edge_list, f, protocol=4)

    return edge_list


def run_inference(corpus, working_directory, biencoder_model, biencoder_threshold):

    """
    From a corpus, finds all pairs that are under a certain distance apart. Forms clusters, either suing connected
    components or community detection, and then imposes transitivity over these clusters.
    Save cudf of edge ID pairs.
    """

    os.makedirs(working_directory, exist_ok=True)
    os.makedirs(f'{working_directory}/embeddings', exist_ok=True)
    os.makedirs(f'{working_directory}/all_nearest_neighbours', exist_ok=True)
    os.makedirs(f'{working_directory}/edges/', exist_ok=True)

    # Create embeddings
    corpus_embeddings = embed(
        corpus,
        trained_model=biencoder_model,
        save_stem=f'{working_directory}/embeddings/embeddings',
        batch_size=1024
    )

    # Return list of nearest neighbours and their distances
    find_nearest_neighbours(
        corpus_embeddings,
        save_stem=f'{working_directory}/all_nearest_neighbours/',
        k=900,
        d=768,
        query_batch_size=150000,
        corpus_batch_size=10000000
    )

    # Reload and subset nearest neighbour data, to pairs under given threshold
    under_th = subset_nn_data(f'{working_directory}/all_nearest_neighbours/', threshold=biencoder_threshold)

    nn_edges = nearest_neighbours_to_pairs(under_th, save_dir=f'{working_directory}/edges/')

    # Put on graph
    G = utils.cnx_make_graph_from_edges(nn_edges)

    # Pull all edges from connected components
    # utils.gpu_connected_components(G, save_file='f'{working_directory}/edges/connected_comps_edges.pkl')

    # Pull all edges after running community detection
    utils.gpu_connected_components(G, save_file=f'{working_directory}/edges/community_edges.pkl', detect_communities=True)


if __name__ == '__main__':

    # C4 data
    # corpus = data_fns.open_realnews()
    corpus = data_fns.open_c4_by_url(pattern="patents.google.com", name="patents")

    run_inference(
        corpus,
        working_directory="",
        biencoder_model='',
        biencoder_threshold=0.94
    )
