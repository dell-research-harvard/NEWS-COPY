from tqdm import tqdm
from datetime import datetime
from itertools import product
import multiprocessing

import json
import copy
import sys
import os
from os.path import dirname as up

sys.path.append(up(up(up(os.path.realpath(__file__)))))

from rule_based import rule_based_utils


def ngram_overlap(data_dict, overlap):
    '''Returns edges between articles given overlap threshold.'''

    global data

    data = data_dict

    # Compare pairs of passages, calculate overlap percentage and keep pair if meets overlap threshold
    print("\n Calculating overlaps ...")
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    list_of_edges_lists = pool.starmap(compare_passage_pairs, [(date, overlap) for date in list(data.keys())])
    pool.close()

    # Collapse into single list
    edges_list = [item for sublist in list_of_edges_lists for item in sublist]
    return edges_list


def compare_passage_pairs(date_str, overlap):
    """Module for calculating N-Gram overlap on multiple cores"""

    same_day_dat = data[date_str]
    n_same_day = len(same_day_dat['art_list'])
    edges = []

    # iterate through all other dates
    for alt_date in list(data.keys()):
        date_dt = datetime.strptime(date_str, "%b-%d-%Y")
        new_dt = datetime.strptime(alt_date, "%b-%d-%Y")

        # only compare to later dates (to prevent repetitions)
        if date_dt >= new_dt:
            new_day_dat = data[alt_date]
            n_new_day = len(new_day_dat['art_list'])

            print(f"\n Computing N-gram overlap between {date_str} and {alt_date} passages...")
            for i, j in product(range(n_same_day), range(n_new_day)):
                compare_overlap_to_threshold(i, j, data_i=same_day_dat, data_j=new_day_dat, overlap=overlap, outfile=edges)

    return edges


def compare_overlap_to_threshold(i, j, data_i, data_j, overlap, outfile):

    # count ngram overlaps
    passage_i_set = data_i['ngram_list'][i]
    passage_j_set = data_j['ngram_list'][j]
    intersect = passage_i_set.intersection(passage_j_set)
    overlap_count = len(intersect)

    # compute percentage of possible ngrams that overlapped
    if len(passage_i_set) != 0 and len(passage_j_set) != 0:
        # overlap_pct = overlap_count / min(len(passage_i_set), len(passage_j_set))
        overlap_pct = overlap_count / len(passage_i_set.union(passage_j_set))
    else:
        overlap_pct = 0

    # compare to overlap threshold and add edge if meets threshold
    if overlap_pct >= overlap:
        id_i = data_i['id_list'][i]
        id_j = data_j['id_list'][j]
        text_i = data_i['art_list'][i]
        text_j = data_j['art_list'][j]
        outfile.append((
            id_i,
            id_j,
            {
                'text_1': text_i,
                'text_2': text_j,
                'overlap': overlap_pct
            }
        ))


if __name__ == '__main__':

    cleaned_text, cleaned_ids = rule_based_utils.gather_data(data_file_path='')

    edge_metrics = {}
    cluster_metrics = {}
    full_metrics = {}

    ground_truth_path = ''

    for (n_gram_size, overlaps) in tqdm([(15, [0.4])]):
        for overlap in overlaps:
            edge_metrics[str((overlap, n_gram_size))] = {}
            cluster_metrics[str((overlap, n_gram_size))] = {}
            full_metrics[str((overlap, n_gram_size))] = {}

            data_dict = rule_based_utils.get_ngrams(copy.deepcopy(cleaned_text), copy.deepcopy(cleaned_ids), n_gram_size=n_gram_size, concat=False, char=True)

            # Calculate n-gram overlaps and return pairs that meet overlap threshold
            edges_list = ngram_overlap(copy.deepcopy(data_dict), overlap=overlap)
            
            # Get evaluation metrics
            edge_results, cluster_results, full_results = rule_based_utils.get_eval_metrics(edges_list, ground_truth_path, cleaned_ids, community_detection=True)

            # Add to dictionary of grid-search results
            edge_metrics[str((overlap, n_gram_size))] = edge_results
            cluster_metrics[str((overlap, n_gram_size))] = cluster_results
            full_metrics[str((overlap, n_gram_size))] = full_results

    # Create dictionary to export results
    results = {"edge_metrics": edge_metrics, "cluster_metrics": cluster_metrics, "full_metrics": full_metrics}

    # Save results
    j = json.dumps(results, indent=4)
    f = open('', 'w')
    print(j, file=f)
    f.close()
