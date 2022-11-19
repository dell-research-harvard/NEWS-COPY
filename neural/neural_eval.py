"""
Python 3.8
Evaluation for all neural models. Biencoder + cross encoder (optional) + community detection
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from itertools import combinations

from sentence_transformers import LoggingHandler, util, logging, SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import logging as lg

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)
sys.path.append(grandparentdir)

import utils
from rule_based import rule_based_utils, ngrams

# Config logging
lg.set_verbosity_error()
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)


def filter_data(
        corpus_dict,
        filter_type='ngrams',
        parameters={'n_gram_size': 5, 'overlap': 0.01}
):

    """
    Return all pairs from some filtering step
    - If biencoder: return all pairs closer than biencoder threshold
    - If ngrams return all pairs with ngrams in common
    - If none, return all pairs
    """

    if filter_type == 'ngrams':

        cleaned_id_list, cleaned_text_list = rule_based_utils.clean_text(
            corpus_dict,
            first_n_tok=None,
            min_tok=None,
            spell_check="symspell"
        )

        data_dict = rule_based_utils.get_ngrams(cleaned_text_list, cleaned_id_list, n_gram_size=parameters['n_gram_size'])

        # Calculate n-gram overlaps and return pairs that meet overlap threshold
        pairs_to_compare = ngrams.ngram_overlap(data_dict, overlap=parameters['overlap'])

    elif filter_type == 'biencoder':

        # Initialise model
        embedder = SentenceTransformer(parameters['model'])

        corpus = []
        for art_id in list(corpus_dict.keys()):
            corpus.append(corpus_dict[art_id]['article'])

        # Embed data
        print("Embedding corpus ...")
        corpus_embeddings = embedder.encode(corpus, show_progress_bar=True, batch_size=512)
        corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

        corpus_ids = list(corpus_dict.keys())

        # Compute distances
        cosine_scores = util.cos_sim(corpus_embeddings, corpus_embeddings)

        # Compare to threshold
        above_threshold = cosine_scores > parameters['threshold']
        upper_only = np.triu(np.ones((len(corpus_ids), len(corpus_ids))) - np.identity(len(corpus_ids)))
        result = above_threshold * upper_only
        indices = [index for index, value in np.ndenumerate(result) if value]
        pairs_to_compare = [[corpus_ids[pair[0]], corpus_ids[pair[1]]] for pair in indices]

    else:
        pairs_to_compare = [list(comb) for comb in combinations(list(corpus_dict.keys()), 2)]

    print("Number of pairs to compare:", len(pairs_to_compare))

    inf_samples = []
    for pair in pairs_to_compare:
        inf_samples.append([str(corpus_dict[pair[0]]['article']), str(corpus_dict[pair[1]]['article'])])

    return inf_samples, pairs_to_compare


def crossencoder_inference(
        inf_samples,
        pairs_to_compare,
        trained_model_path,
        batch_size=128,
        save_dir=None
):

    """
    Run cross-encoder inference over set of pairs
    """

    # Run evaluation
    trained_model = CrossEncoder(trained_model_path, num_labels=1)

    dev_results = pd.read_csv(f"{trained_model_path}/CEBinaryClassificationEvaluator_dev_results.csv")
    threshold = list(dev_results['F1_Threshold'])[-1]

    inference = trained_model.predict(inf_samples, batch_size=batch_size, apply_softmax=True, show_progress_bar=True)

    outputs = []
    for i in range(len(inference)):
        if inference[i] > threshold:
            outputs.append(pairs_to_compare[i])

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        with open(f'{save_dir}/predicted_edges.json', 'w') as of:
            json.dump(outputs, of, indent=4)

    return outputs


def evaluate(gt_path, inf_data_path, biencoder_model, cross_encoder=False, cross_encoder_model=''):

    """
    Run filtering step (see filter_data), then cross-encoder (optional). Evaluate results
    """

    with open(inf_data_path) as f:
        inf_data = json.load(f)

    # Filter edges
    filtered_data, pred_ids = filter_data(
        inf_data,
        filter_type='biencoder',
        # parameters={'n_gram_size': 3, 'overlap': 0.05},
        parameters={'threshold': 0.94, 'model': biencoder_model}  # 0.92 is best with cross-encoder, 0.94 for biencoder alone
    )

    # Run cross-encoder
    if cross_encoder:
        pred_ids = crossencoder_inference(
            filtered_data,
            pred_ids,
            trained_model_path=cross_encoder_model,
            batch_size=1024,
        )

    # Community detection
    cd_edges = utils.detect_communities_nx(pred_ids, resolution=1)

    # Evaluate
    utils.evaluate(cd_edges, gt_edge_path=gt_path)
    utils.cluster_eval(cd_edges, gt_edges=gt_path, all_ids=list(inf_data.keys()))


if __name__ == '__main__':

    evaluate(
        gt_path='',
        inf_data_path='',
        biencoder_model='',
        cross_encoder_model='',
    )
