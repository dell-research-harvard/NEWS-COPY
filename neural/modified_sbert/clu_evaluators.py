import os
import sys
import numpy as np
import csv
from itertools import combinations
from typing import List

from sentence_transformers import evaluation, LoggingHandler
from sentence_transformers.readers import InputExample

import logging
from transformers import logging as lg

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

import utils

lg.set_verbosity_error()
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
logger = logging.getLogger(__name__)


class ClusterEvaluator(evaluation.SentenceEvaluator):

    """
    Evaluate a model based on allocation of texts into correct clusters.
    Embeddings are clustered with the specified clustering algorithm using cosine distance. Best clustering parameters
    (distance threshold) are found using an approximate search method to speed to evaluation time.

    All possible combination of articles are split into pairs, with positives being in the same cluster and negatives
    being in different clusters.
    Metrics are precision, recall and F1.

    Returned metrics are F1 along with the optimal clustering threshold.

    The results are written in a CSV. If a CSV already exists, then values are appended.

    :param sentences1: The first column of sentences
    :param sentences2: The second column of sentences
    :param labels: labels[i] is the label for the pair (sentences1[i], sentences2[i]). Must be 0 or 1
    :param name: Name for the output
    :param batch_size: Batch size used to compute embeddings
    :param show_progress_bar: If true, prints a progress bar
    :param write_csv: Write results to a CSV file
    :param cluster_type: Clustering algoritm to use. Supports "agglomerative" (hierarchical), "SLINK", "HDBScan"

    Modelled on: https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/evaluation/BinaryClassificationEvaluator.py
    """

    def __init__(
            self,
            sentences1: List[str],
            sentences2: List[str],
            labels: List[int],
            name: str = '',
            batch_size: int = 512,
            show_progress_bar: bool = False,
            write_csv: bool = True,
            cluster_type: str = "agglomerative"
    ):

        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.labels = labels
        self.cluster_type = cluster_type

        assert len(self.sentences1) == len(self.sentences2)
        assert len(self.sentences1) == len(self.labels)
        for label in labels:
            assert (label == 0 or label == 1)

        self.write_csv = write_csv
        self.name = name
        self.batch_size = batch_size
        if show_progress_bar is None:
            show_progress_bar = (logger.getEffectiveLevel() == logging.INFO or logger.getEffectiveLevel() == logging.DEBUG)
        self.show_progress_bar = show_progress_bar

        self.csv_file = "clustering_evaluation" + ("_"+name if name else '') + "_results.csv"
        self.csv_headers = ["epoch", "steps", "accuracy", "accuracy_threshold", "f1", "precision", "recall", "f1_threshold"]

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        sentences1 = []
        sentences2 = []
        scores = []

        for example in examples:
            sentences1.append(example.texts[0])
            sentences2.append(example.texts[1])
            scores.append(example.label)
        return cls(sentences1, sentences2, scores, **kwargs)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:

        if epoch != -1:
            if steps == -1:
                out_txt = f" after epoch {epoch}:"
            else:
                out_txt = f" in epoch {epoch} after {steps} steps:"
        else:
            out_txt = ":"

        logger.info("Cluster Evaluation of the model on " + self.name + " dataset" + out_txt)

        scores = self.compute_metrices(model)

        #Main score is F1
        main_score = scores['f1']

        file_output_data = [epoch, steps]

        for score in self.csv_headers:
            if score in scores:
                file_output_data.append(scores[score])

        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            if not os.path.isfile(csv_path):
                with open(csv_path, newline='', mode="w", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.csv_headers)
                    writer.writerow(file_output_data)
            else:
                with open(csv_path, newline='', mode="a", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(file_output_data)

        return main_score

    def compute_metrices(self, model):

        sentences = []
        labels = []
        for i in range(len(self.sentences1)):

            if self.sentences1[i] not in sentences:
                sentences.append(self.sentences1[i])
            s1_id = sentences.index(self.sentences1[i])
            if self.sentences2[i] not in sentences:
                sentences.append(self.sentences2[i])
            s2_id = sentences.index(self.sentences2[i])

            if self.labels[i] == 1:
                labels.append([s1_id, s2_id])

        embeddings = model.encode(sentences, batch_size=self.batch_size, show_progress_bar=self.show_progress_bar)

        # Normalize the embeddings to unit length
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

        def cluster_eval(threshold, embeddings, labels, cluster_type='agglomerative'):

            clustered_ids = utils.cluster(
                cluster_type,
                cluster_params={"threshold": threshold, "clustering linkage": 'average', "metric": 'cosine', "min cluster size": 2},
                corpus_embeddings=embeddings
            )

            # Convert every pair in a cluster into an edge
            cluster_edges = utils.edges_from_clusters(clustered_ids)

            metrics = utils.evaluate(pred_edges=cluster_edges, gt_edges=labels, print_metrics=False)

            total = len(list(combinations(range(len(embeddings)), 2)))
            cluster_tn = total - metrics["tps"] - metrics["fps"] - metrics["fns"]

            metrics["accuracy"] = (metrics["tps"] + cluster_tn)/total

            return metrics

        tenths = {}
        for threshold in [0.01] + [round(x, 2) for x in (np.linspace(0.1, 0.9, 9))] + [0.99]:
            tenths[threshold] = cluster_eval(threshold, embeddings, labels, cluster_type=self.cluster_type)

        def best_threshold(dictionary, metric):

            ths = list(dictionary.keys())
            scores = []

            for th in ths:
                scores.append(dictionary[th][metric])

            sorted_scores = sorted(scores)

            best_score = sorted_scores[-1]
            second_best_score = sorted_scores[-2]
            third_best_score = sorted_scores[-3]

            if best_score == second_best_score:
                best_indices = [i for i, x in enumerate(scores) if x == best_score]
                best_thresholds = []
                for idx in best_indices:
                    best_thresholds.append(ths[idx])

                best_th = max(best_thresholds)
                second_best_th = min(best_thresholds)

            elif second_best_score == third_best_score:
                second_indices = [i for i, x in enumerate(scores) if x == second_best_score]
                second_thresholds = []
                for idx in second_indices:
                    second_thresholds.append(ths[idx])

                best_th = max(second_thresholds)
                second_best_th = min(second_thresholds)

            else:
                best_idx = scores.index(best_score)
                best_th = ths[best_idx]

                second_best_idx = scores.index(second_best_score)
                second_best_th = ths[second_best_idx]

            return best_th, second_best_th

        max_f1_th, second_f1_th = best_threshold(tenths, metric='f_score')
        max_acc_th, second_acc_th = best_threshold(tenths, metric='accuracy')

        min_th = min(max_f1_th, second_f1_th, max_acc_th, second_acc_th)
        max_th = max(max_f1_th, second_f1_th, max_acc_th, second_acc_th)

        hundreths = {}
        for threshold in np.arange(min_th, max_th, 0.01):
            hundreths[threshold] = cluster_eval(threshold, embeddings, labels)

        hd_max_f1_th, _ = best_threshold(hundreths, 'f_score')
        hd_max_acc_th, _ = best_threshold(hundreths, 'accuracy')

        acc = hundreths[hd_max_acc_th]['accuracy']
        acc_threshold = hd_max_acc_th

        f1 = hundreths[hd_max_f1_th]['f_score']
        precision = hundreths[hd_max_f1_th]['precision']
        recall = hundreths[hd_max_f1_th]['recall']
        f1_threshold = hd_max_f1_th

        logger.info("Cluster Accuracy:           {:.2f}\t(Threshold: {:.2f})".format(acc * 100, acc_threshold))
        logger.info("Cluster F1:                 {:.2f}\t(Threshold: {:.2f})".format(f1 * 100, f1_threshold))
        logger.info("Cluster Precision:          {:.2f}".format(precision * 100))
        logger.info("Cluster Recall:             {:.2f}\n".format(recall * 100))

        output_scores = {
            'accuracy': acc,
            'accuracy_threshold': acc_threshold,
            'f1': f1,
            'f1_threshold': f1_threshold,
            'precision': precision,
            'recall': recall,
        }

        return output_scores


class CEClusterEvaluator():

    def __init__(self, sentence_pairs: List[List[str]], labels: List[int], name: str='', write_csv: bool = True):
        assert len(sentence_pairs) == len(labels)
        for label in labels:
            assert (label == 0 or label == 1)

        self.sentence_pairs = sentence_pairs
        self.labels = np.asarray(labels)
        self.name = name

        self.csv_file = "CEClusterEvaluator" + ("_" + name if name else '') + "_results.csv"
        self.csv_headers = ["epoch", "steps", "accuracy", "accuracy_threshold", "f1", "f1_threshold", "precision", "recall"]
        self.write_csv = write_csv

    @classmethod
    def from_input_examples(cls, examples: List[InputExample], **kwargs):
        sentence_pairs = []
        labels = []

        for example in examples:
            sentence_pairs.append(example.texts)
            labels.append(example.label)
        return cls(sentence_pairs, labels, **kwargs)

    def __call__(self, model, output_path: str = None, epoch: int = -1, steps: int = -1) -> float:
        if epoch != -1:
            if steps == -1:
                out_txt = " after epoch {}:".format(epoch)
            else:
                out_txt = " in epoch {} after {} steps:".format(epoch, steps)
        else:
            out_txt = ":"

        logger.info("CEClusterEvaluator: Evaluating the model on " + self.name + " dataset" + out_txt)

        pred_scores = model.predict(self.sentence_pairs, convert_to_numpy=True, show_progress_bar=False)

        acc, acc_threshold, f1, precision, recall, f1_threshold = self.find_best_acc_and_f1(pred_scores, self.sentence_pairs, self.labels)

        logger.info("Cluster Accuracy:           {:.2f}\t(Threshold: {:.2f})".format(acc * 100, acc_threshold))
        logger.info("Cluster F1:                 {:.2f}\t(Threshold: {:.2f})".format(f1 * 100, f1_threshold))
        logger.info("Cluster Precision:          {:.2f}".format(precision * 100))
        logger.info("Cluster Recall:             {:.2f}\n".format(recall * 100))


        if output_path is not None and self.write_csv:
            csv_path = os.path.join(output_path, self.csv_file)
            output_file_exists = os.path.isfile(csv_path)
            with open(csv_path, mode="a" if output_file_exists else 'w', encoding="utf-8") as f:
                writer = csv.writer(f)
                if not output_file_exists:
                    writer.writerow(self.csv_headers)

                writer.writerow([epoch, steps, acc, acc_threshold, f1, f1_threshold, precision, recall])

        return f1

    @staticmethod
    def find_best_acc_and_f1(scores, sentence_pairs, labels):

        assert len(scores) == len(labels)

        sentences = []
        pair_ids = []
        for pair in sentence_pairs:
            if pair[0] not in sentences:
                sentences.append(pair[0])
            s1_id = sentences.index(pair[0])
            if pair[1] not in sentences:
                sentences.append(pair[1])
            s2_id = sentences.index(pair[1])
            pair_ids.append([s1_id, s2_id])

        gt_edges = [pair_ids[i] for i in range(len(pair_ids)) if labels[i] == 1]
        gt_edges = utils.edges_from_clusters(utils.clusters_from_edges(gt_edges))   # Impose transitivity

        total_possible_edges = len(labels)

        thds = list(set([round(score, 2) for score in scores]))
        accuracies = []
        precisions = []
        recalls = []
        f1s = []

        for th in thds:

            preds = [pair_ids[i] for i in range(len(pair_ids)) if scores[i] > th]

            # Impose transitivity
            pred_edges = utils.edges_from_clusters(utils.clusters_from_edges(preds))

            metrics = utils.evaluate(pred_edges=pred_edges, gt_edges=gt_edges, print_metrics=False)
            precisions.append(metrics['precision'])
            recalls.append(metrics['recall'])
            f1s.append(metrics['f_score'])

            cluster_tn = total_possible_edges - metrics["tps"] - metrics["fps"] - metrics["fns"]
            accuracies.append((metrics["tps"] + cluster_tn) / total_possible_edges)

        # Find max values
        max_acc = max(accuracies)
        acc_idx = accuracies.index(max_acc)
        acc_threshold = thds[acc_idx]

        max_f1 = max(f1s)
        f1_idx = f1s.index(max_f1)
        precision = precisions[f1_idx]
        recall = recalls[f1_idx]
        f1_threshold = thds[f1_idx]

        return max_acc, acc_threshold, max_f1, precision, recall, f1_threshold



