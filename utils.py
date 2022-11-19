import json
import pickle
from itertools import combinations
from datetime import datetime

from tqdm import tqdm
import random

from sklearn.metrics import average_precision_score, adjusted_mutual_info_score, rand_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.cluster import AgglomerativeClustering, DBSCAN
import hdbscan

import networkx as nx
import networkx.algorithms.community as nx_comm
import cugraph as cnx
import cudf as gd


def cluster(cluster_type, cluster_params, corpus_embeddings, corpus_ids=None):

    """
    Perform specified clustering method
    """

    if cluster_type not in ["agglomerative", "HDBScan", "SLINK"]:
        raise ValueError('cluster_type must be "agglomerative", "HDBScan", "community" or "SLINK"')
    if cluster_type == "agglomerative":
        if "threshold" not in cluster_params:
            raise ValueError('cluster_params must contain "threshold"')
        if "clustering linkage" not in cluster_params:
            raise ValueError('cluster_params must contain "clustering linkage"')
        if "metric" not in cluster_params:
            raise ValueError('cluster_params must contain "metric"')
    if cluster_type == "HDBScan":
        if "min cluster size" not in cluster_params:
            raise ValueError('cluster_params must contain "min cluster size"')
        if "min samples" not in cluster_params:
            raise ValueError('cluster_params must contain "min cluster size"')
    if cluster_type == "SLINK":
        if "min cluster size" not in cluster_params:
            raise ValueError('cluster_params must contain "min cluster size"')
        if "threshold" not in cluster_params:
            raise ValueError('cluster_params must contain "threshold"')
        if "clustering affinity" not in cluster_params:
            raise ValueError('cluster_params must contain "clustering affinity"')

    if cluster_type == "agglomerative":
        clustering_model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=cluster_params["threshold"],
            linkage=cluster_params["clustering linkage"],
            affinity=cluster_params["metric"]
        )

    if cluster_type == "SLINK":
        clustering_model = DBSCAN(
            eps=cluster_params["threshold"],
            min_samples=cluster_params["min cluster size"],
            metric=cluster_params["metric"]
        )

    if cluster_type == "HDBScan":
        clustering_model = hdbscan.HDBSCAN(
            min_cluster_size=cluster_params["min cluster size"],
            min_samples=cluster_params["min samples"],
            gen_min_span_tree=True
        )

    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_

    clustered_ids = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        if int(cluster_id) not in clustered_ids:
            clustered_ids[int(cluster_id)] = []

        if corpus_ids:
            clustered_ids[int(cluster_id)].append(corpus_ids[sentence_id])
        else:
            clustered_ids[int(cluster_id)].append(sentence_id)

    # HDBScan has a cluster where it puts all the unassigned nodes
    if cluster_type == "HDBScan" or cluster_type == "SLINK" and -1 in clustered_ids:
        del clustered_ids[-1]

    return clustered_ids


def clusters_from_edges(edges_list):
    """Identify clusters of passages given a dictionary of edges"""

    # clusters via NetworkX
    G = nx.Graph()
    G.add_edges_from(edges_list)
    sub_graphs = [G.subgraph(c).copy() for c in nx.connected_components(G)]

    sub_graph_dict = {}
    for i in range(len(sub_graphs)):
        sub_graph_dict[i] = list(sub_graphs[i].nodes())

    return sub_graph_dict


def edges_from_clusters(cluster_dict):
    """
    Convert every pair in a cluster into an edge
    """
    cluster_edges = []
    for cluster_id in list(cluster_dict.keys()):
        art_ids_list = cluster_dict[cluster_id]
        edge_list = [list(comb) for comb in combinations(art_ids_list, 2)]
        cluster_edges.extend(edge_list)

    return cluster_edges


def evaluate(pred_edges, gt_edge_path=None, gt_edges=None, print_metrics=True, print_incorrects=False, two_way=True, save_incorrects=False):

    """
    Return F1, recall, precision, from set of predicted edges and gt set
    """

    if not gt_edges and not gt_edge_path:
        raise ValueError("either gt_edge_path or gt_edges must be specified")

    # Prep ground truth
    if not gt_edges:
        with open(gt_edge_path) as f:
            gt_edges = json.load(f)

    set_gt = set(map(tuple, gt_edges))

    # Prep preds
    pred_edges_list = [[edge[0], edge[1]] for edge in pred_edges]
    set_preds = set(map(tuple, pred_edges_list))

    # Metrics
    if two_way:
        tps = len([i for i in set_gt if i in set_preds or (i[1], i[0]) in set_preds])
        fps = len([i for i in set_preds if i not in set_gt and (i[1], i[0]) not in set_gt])
        fns = len([i for i in set_gt if i not in set_preds and (i[1], i[0]) not in set_preds])
    else:
        tps = len([i for i in set_gt if i in set_preds])
        fps = len([i for i in set_preds if i not in set_gt])
        fns = len([i for i in set_gt if i not in set_preds])

    if tps + fps > 0:
        precision = tps / (tps + fps)
    else:
        precision = 0
    if tps + fns > 0:
        recall = tps / (tps + fns)
    else:
        recall = 0
    if precision + recall > 0:
        f_score = 2 * (precision * recall) / (precision + recall)
    else:
        f_score = 0

    metrics = {"precision": precision, "recall": recall, "f_score": f_score, "tps": tps, "fps": fps, "fns": fns}

    # Look at wrong ones
    if print_incorrects:
        fp_list = [i for i in set_preds if i not in set_gt]
        fn_list = [i for i in set_gt if i not in set_preds]

        print(fn_list)
        print(len(fn_list))
        print(tps, fps, fns)

    if print_metrics:
        print(metrics)

    if save_incorrects:
        fp_list = [i for i in set_preds if i not in set_gt and (i[1], i[0]) not in set_gt]
        fn_list = [i for i in set_gt if i not in set_preds and (i[1], i[0]) not in set_preds]

        print(tps, fps, fns)

        fp_list = random.sample(fp_list, 50)
        fn_list = random.sample(fn_list, 50)

        return fp_list, fn_list

    else:

        return metrics


def cluster_eval(pred_edges, gt_edges, all_ids):

    """
    Return RI, ARI, NMI, AMI, from set of predicted edges and gt set
    """

    pred_clusters = clusters_from_edges(pred_edges)

    with open(gt_edges) as f:
        gt_edges = json.load(f)

    set_gt = set(map(tuple, gt_edges))
    gt_clusters = clusters_from_edges(set_gt)

    # get dictionary mapping article to cluster number
    pred_dict = {}
    pred_count = 0
    for cluster in pred_clusters:
        for article in pred_clusters[cluster]:
            pred_dict[article] = pred_count
        pred_count += 1

    gt_dict = {}
    gt_count = 0
    for cluster in gt_clusters:
        for article in gt_clusters[cluster]:
            gt_dict[article] = gt_count
        gt_count += 1

    # fill in clusters with unclustered articles
    full_pred_clusters = []
    full_gt_clusters = []
    for article in all_ids:
        if article in pred_dict:
            full_pred_clusters.append(pred_dict[article])
        else:
            full_pred_clusters.append(pred_count)
            pred_count += 1

        if article in gt_dict:
            full_gt_clusters.append(gt_dict[article])
        else:
            full_gt_clusters.append(gt_count)
            gt_count += 1

    assert len(full_pred_clusters) == len(full_gt_clusters)

    RI = rand_score(full_pred_clusters, full_gt_clusters)
    ARI = adjusted_rand_score(full_pred_clusters, full_gt_clusters)
    NMI = normalized_mutual_info_score(full_pred_clusters, full_gt_clusters)
    AMI = adjusted_mutual_info_score(full_pred_clusters, full_gt_clusters)

    print({"RI": RI, "ARI": ARI, "NMI": NMI, "AMI": AMI})

    return {"RI": RI, "ARI": ARI, "NMI": NMI, "AMI": AMI}


def detect_communities_nx(edges, resolution=1):

    """Louvain community detection using nx"""

    G = nx.Graph()
    G.add_edges_from(edges)

    communities = nx_comm.louvain_communities(G, resolution=resolution)

    sub_graph_dict = {}
    for i in range(len(communities)):
        sub_graph_dict[i] = list(communities[i])

    return edges_from_clusters(sub_graph_dict)


def cnx_make_graph_from_edges(edge_list):

    """Make a graph from list of lists of neighbors"""

    time_graph_start = datetime.now()

    # Build edges into a gpu dataframe
    edge_df = gd.DataFrame({'src': gd.Series([i[0] for i in edge_list]), 'dst': gd.Series([i[1] for i in edge_list])})

    # Make graph
    G = cnx.Graph()
    G.from_cudf_edgelist(edge_df, source='src', destination='dst')

    print("Number of nodes:", cnx.structure.graph_implementation.simpleGraphImpl.number_of_vertices(G))
    print("Number of edges before imposing transistivty:", cnx.structure.graph_implementation.simpleGraphImpl.number_of_edges(G))

    time_graph_end = datetime.now()
    print("Time taken to make graph: ", time_graph_end-time_graph_start)

    return G


def gpu_connected_components(G, save_file, detect_communities=False):

    """
    Impose transitivity and return edges, either with or without community detection
    """

    time_cc_start = datetime.now()

    print("Imposing transitivity ...")

    if detect_communities:
        ccs, _ = cnx.louvain(G, resolution=1)
        ccs = ccs.rename(columns={"partition": "labels"})
    else:
        ccs = cnx.connected_components(G)

    print("Distinct connected components: ", ccs.labels.nunique())

    total_perms = 0
    total_reduced = 0

    def get_edges(df_1, df_2, total_perms, total_reduced):

        df_1 = df_1.merge(df_2, on='labels', how='inner')
        df_1 = df_1.drop(['labels'], axis=1)
        total_perms += len(df_1)

        df_1 = df_1[df_1['vertex_x'] < df_1['vertex_y']]  # remove both directions and loops
        total_reduced += len(df_1)

        return df_1, total_perms, total_reduced

    lengths = []
    all_edges = []

    ccs_pd = ccs.to_pandas()

    for label in tqdm(ccs_pd.labels.unique()):
        sub_frame = ccs[ccs['labels'] == label]

        lengths.append(len(sub_frame))

        if len(sub_frame) < 50000:  # Larger subframes don't fit on GPU, so run on CPU (though slower!)
            edge_df, total_perms, total_reduced = get_edges(sub_frame, sub_frame, total_perms, total_reduced)
            all_edges.append(edge_df)

        else:
            sub_frame_A = sub_frame[:30000]
            sub_frame_B = sub_frame[30000:]

            edge_df, total_perms, total_reduced = get_edges(sub_frame_A, sub_frame_A, total_perms, total_reduced)
            all_edges.append(edge_df)
            edge_df, total_perms, total_reduced = get_edges(sub_frame_A, sub_frame_B, total_perms, total_reduced)
            all_edges.append(edge_df)
            edge_df, total_perms, total_reduced = get_edges(sub_frame_B, sub_frame_A, total_perms, total_reduced)
            all_edges.append(edge_df)
            edge_df, total_perms, total_reduced = get_edges(sub_frame_B, sub_frame_B, total_perms, total_reduced)
            all_edges.append(edge_df)

    squares = [i*i for i in lengths]

    assert total_perms == sum(squares)
    assert total_reduced == (sum(squares) - len(ccs))/2

    time_cc_end = datetime.now()

    edges = gd.concat(all_edges)
    assert len(edges) == total_reduced

    print("Time taken to find connected components: ", time_cc_end-time_cc_start)
    print("Number of edges after imposing transitivity:", len(edges))

    with open(save_file, 'wb') as f:
        pickle.dump(edges, f)
