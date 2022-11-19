from tqdm import tqdm
from itertools import combinations
from datasketch import MinHash
from collections import defaultdict
import json
import copy
import sys
import os
from os.path import dirname as up


sys.path.append(up(up(up(os.path.realpath(__file__)))))

import rule_based_utils


def get_counts(data_dict, num_hashes):
    ''' Gets parwise counts of shared hashes.'''

    pairwise_counts = defaultdict(int)
    hash_dict = [defaultdict(list) for _ in range(num_hashes)]
    m1 = MinHash(num_perm=num_hashes)

    # iterate through all the articles
    for date in data_dict:
        for index, ngrams in enumerate(data_dict[date]['ngram_list']):
            m1.clear()
            article_id = data_dict[date]['id_list'][index]

            # hash ngrams
            for ngram in ngrams:
                m1.update(ngram.encode('utf8'))

            # add hashes to hash table
            for index, hash_val in enumerate(list(m1.digest())):
                hash_dict[index][hash_val].append(article_id)

    # iterate through hash table and form edges
    for table in hash_dict:
        for val in table:
            if len(table[val]) >= 2:
                for pair in combinations(table[val], 2):
                    pairwise_counts[pair] += 1
            
    return pairwise_counts


def get_edges(counts, threshold):
    ''' Get edges given pairwise counts and threshold. '''
    edges = set()

    for pair in counts:
        if counts[pair] >= threshold:
            edges.add(pair)

    return list(edges)


if __name__ == '__main__':

    cleaned_text, cleaned_ids = rule_based_utils.gather_data(data_file_path='')
    ground_truth_path = ''

    # Set hyperparameters
    num_hashes = 10

    for (n_gram_size, thresholds) in tqdm([(10, [3, 4]), (15, [3, 4]), (20, [2, 3]), (25, [2, 3])]):

        # Get n-grams data for articles
        data_dict = rule_based_utils.get_ngrams(copy.deepcopy(cleaned_text), copy.deepcopy(cleaned_ids), n_gram_size=n_gram_size, concat=True, char=True)

        # Calculate n-gram overlaps and return pairs that meet overlap threshold
        counts = get_counts(copy.deepcopy(data_dict), num_hashes=num_hashes)

        for threshold in tqdm(thresholds):

            # Get edges from pairwise counts, threshold
            edges_list = get_edges(counts, threshold)

            # Get evaluation metrics
            edge_results, cluster_results, full_results = rule_based_utils.get_eval_metrics(edges_list, ground_truth_path, cleaned_ids, community_detection=True)

            # Add to dictionary of grid-search results
            edge_metrics[str((num_hashes, threshold, n_gram_size))] = edge_results
            cluster_metrics[str((num_hashes, threshold, n_gram_size))] = cluster_results
            full_metrics[str((num_hashes, threshold, n_gram_size))] = full_results

    # Create dictionary to export results
    results = {"edge_metrics": edge_metrics, "cluster_metrics": cluster_metrics, "full_metrics": full_metrics}

    # Save results
    j = json.dumps(results, indent=4)
    f = open('', 'w')
    print(j, file=f)
    f.close()
