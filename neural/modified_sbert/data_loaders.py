import os
import sys
import random

from sentence_transformers.readers import InputExample

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import utils


def load_data_as_individuals(data, type):

    sentence_1_list = data['sentence_1']
    sentence_2_list = data['sentence_2']
    labels = data['labels']

    # Organise by cluster
    edges_list = []
    for i in range(len(sentence_1_list)):
        if labels[i] == "same":
            edges_list.append([sentence_1_list[i], sentence_2_list[i]])

    cluster_dict = utils.clusters_from_edges(edges_list)

    # Pull out texts and cluster IDs
    indv_data = []
    guid = 1
    for cluster_id in list(cluster_dict.keys()):

        for text in cluster_dict[cluster_id]:
            indv_data.append(InputExample(guid=guid, texts=[text], label=cluster_id))

            guid += 1

    print(f'{len(indv_data)} {type} examples')

    return indv_data


def load_data_as_pairs(data, type):

    sentence_1_list = data['sentence_1']
    sentence_2_list = data['sentence_2']
    labels = data['labels']

    label2int = {"same": 1, "different": 0, 1: 1, 0: 0}

    paired_data = []
    for i in range(len(sentence_1_list)):
        label_id = label2int[labels[i]]
        paired_data.append(InputExample(texts=[sentence_1_list[i], sentence_2_list[i]], label=float(label_id)))

    print(f'{len(paired_data)} {type} pairs')

    return paired_data


def load_data_as_triplets(data, type):

    sentence_1_list = data['sentence_1']
    sentence_2_list = data['sentence_2']
    labels = data['labels']

    # Create dict of examples where you have labels, at the anchor level
    def add_to_samples(sent1, sent2, label):
        if sent1 not in anchor_dict:
            anchor_dict[sent1] = {'same': set(), 'different': set()}
        anchor_dict[sent1][label].add(sent2)

    anchor_dict = {}
    for i in range(len(sentence_1_list)):
        add_to_samples(sentence_1_list[i], sentence_2_list[i], labels[i])
        add_to_samples(sentence_2_list[i], sentence_1_list[i], labels[i])  #Also add the opposite

    # Create triplets
    triplet_data = []
    for anchor, others in anchor_dict.items():
        while len(others['same']) > 0 and len(others['different']) > 0:

            same_sent = random.choice(list(others['same']))
            dif_sent = random.choice(list(others['different']))

            triplet_data.append(InputExample(texts=[anchor, same_sent, dif_sent]))

            others['same'].remove(same_sent)
            others['different'].remove(dif_sent)

    print(f'{len(triplet_data)} {type} triplets')

    return triplet_data
