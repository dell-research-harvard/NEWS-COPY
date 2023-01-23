import json

from tqdm import tqdm
import re
import os
from os.path import dirname as up
import sys
from transformers import BertTokenizerFast
import logging

from symspellpy import SymSpell, Verbosity

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
grandparentdir = os.path.dirname(parentdir)
sys.path.append(parentdir)
sys.path.append(grandparentdir)

import utils


def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)


def spellcheck(list_of_texts, spell_check_type):
    ''' Runs spell-checker over the list of texts. '''

    if spell_check_type == "symspell":
        spell_checked_texts = symspell_check_ocr(list_of_texts)
    if spell_check_type == "fixed":
        spell_checked_texts = fixed_dict(list_of_texts)
    if spell_check_type is None:
        return list_of_texts

    return spell_checked_texts


def symspell_setup(resource_dir="", edit_distance=2):

    sym_spell = SymSpell(max_dictionary_edit_distance=edit_distance, prefix_length=7)

    dictionary_path = os.path.join(resource_dir, "frequency_dictionary_en_82_765.txt")
    bigram_path = os.path.join(resource_dir, "frequency_bigramdictionary_en_243_342.txt")

    print("Dictionary Path:", dictionary_path)
    sym_spell.load_dictionary(dictionary_path, term_index=0, count_index=1)
    sym_spell.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)

    return sym_spell


def fixed_dict(ocr_article_clean_texts):
    '''Very flexible spell checker.'''

    sym_spell = symspell_setup(edit_distance=5)

    ocr_spell_texts = []

    print("\n Spell checking ...")
    for text in tqdm(ocr_article_clean_texts):
        spell_corr = []
        for input_term in text.split():
            suggestions = sym_spell.lookup(input_term, Verbosity.TOP, max_edit_distance=None, include_unknown=True,
                                           transfer_casing=True)
            spell_corr.append(suggestions[0].term)
        ocr_spell_texts.append(" ".join(spell_corr))

    return ocr_spell_texts


def symspell_check_ocr(ocr_article_clean_texts):
    """Corrects spelling of OCR article texts"""

    sym_spell = symspell_setup()

    ocr_spell_texts = []

    print("\n Spell checking ...")
    for text in tqdm(ocr_article_clean_texts):
        spell_corr = []
        for input_term in text.split():
            suggestions = sym_spell.lookup(input_term, Verbosity.CLOSEST, max_edit_distance=2, include_unknown=True,
                                           transfer_casing=True)
            spell_corr.append(suggestions[0].term)
        ocr_spell_texts.append(" ".join(spell_corr))

    return ocr_spell_texts


def remove_odd_characters(list_of_texts):
    ''' Removes punctuation, unknown characters. '''
    chars_to_remove = r'"#$%&\()*+/:;<=>@[\\]^_`{|}~.?,!\''
    ocr_article_clean_texts = []

    for text in list_of_texts:
        text = text.replace("-\n", "").replace("\n", " ")
        text = text.translate(str.maketrans('', '', chars_to_remove))
        text = text.encode('ascii', 'ignore').decode()
        ocr_article_clean_texts.append(text)

    return ocr_article_clean_texts


def clean_text(corpus_dict, first_n_tok=None, min_tok=None, spell_check="symspell"):
    ''' Cleans texts by removing punctuation, optionally spell-checking. '''

    cleaned_ids = []
    org_texts = []

    # instantiate tokenizer
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    set_global_logging_level(logging.ERROR, ["transformers", "BertTokenizerFast"])

    for key in list(corpus_dict.keys()):
        text = corpus_dict[key]['byline'] + " " + corpus_dict[key]['article']

        if first_n_tok is not None:
            tokens = tokenizer.encode(text, truncation=False)
            text = tokenizer.decode(tokens[1:first_n_tok])
        if min_tok is not None:
            if len(tokens) > min_tok:
                cleaned_ids.append(key)
                org_texts.append(text)
        else:
            cleaned_ids.append(key)
            org_texts.append(text)

    cleaned_texts = remove_odd_characters(org_texts)

    if spell_check:
        cleaned_texts = spellcheck(cleaned_texts, spell_check_type=spell_check)

    return cleaned_ids, cleaned_texts


def gather_data(data_file_path, min_tok=None, n_tok=None, spell_check=None):
    ''' Gathers article data. '''

    with open(data_file_path) as f:
        corpus_dict = json.load(f)

    cleaned_id_list, cleaned_text_list = clean_text(
        corpus_dict,
        first_n_tok=n_tok,
        min_tok=min_tok,
        spell_check=spell_check
    )

    return cleaned_text_list, cleaned_id_list


def list_ngrams(list_of_texts, n_gram_size=5, concat=False, char=False):
    ''' Returns list of n-grams given list of texts. '''

    # Create list of all n-grams in all passages
    ngram_sets = []
    for passage in list_of_texts:
        # Creates character-based n-grams
        if char:
            words = passage.split()
            passage = " ".join(words)
            n_grams = list(zip(*[passage[i:] for i in range(n_gram_size)]))
        # Creates word-based n-grams
        else:
            words = passage.split()
            n_grams = list(zip(*[words[i:] for i in range(n_gram_size)]))
        
        # concatenates n-grams instead of leaving them as tuples
        if concat:
            n_grams = [" ".join(x) for x in n_grams]

        ngram_sets.append(set(n_grams))

    return ngram_sets


def get_ngrams(cleaned_text_list, cleaned_id_list, n_gram_size, concat=False, char=False):
    ''' Returns formatted dictionary of n-grams. '''

    # Create list of n-grams for each article
    n_gram_list = list_ngrams(cleaned_text_list, n_gram_size, concat=concat, char=char)

    # Split into dictionary item per day
    date_list = []
    for art_id in cleaned_id_list:
        date_list.append("-".join(art_id.split("-")[-5:-2]))
    unique_date_list = list(set(date_list))

    data_dict = {}
    for date in unique_date_list:
        indices = [i for i, x in enumerate(date_list) if x == date]
        data_dict[date] = {
            "id_list": [cleaned_id_list[i] for i in indices],
            "art_list": [cleaned_text_list[i] for i in indices],
            "ngram_list": [n_gram_list[i] for i in indices]
        }

    return data_dict


def get_eval_metrics(edges, gt_path, ids, community_detection=False):
    ''' Gets evaluation metrics after clustering. '''

    # Store different metrics for clustering
    edge_metrics = {}
    cluster_metrics = {}
    full_metrics = {}

    # Perform edge-level evaluation
    metrics = utils.evaluate(edges, gt_edge_path=gt_path)
    for value in metrics:
        edge_metrics[value] = metrics[value]

    # Optionally perform community detection
    if community_detection:
        edges = utils.detect_communities_nx(edges)

    # Impose transitivty for edges
    cluster_dict = utils.clusters_from_edges(edges)
    edges = utils.edges_from_clusters(cluster_dict)

    # Perform cluster-level evaluation (Recall, Precision, F1)
    metrics = utils.evaluate(edges, gt_edge_path=gt_path)
    for value in metrics:
        cluster_metrics[value] = metrics[value]
    
    # Perform cluster-level evaluation (RI, ARI, NMI, AMI)
    metrics = utils.cluster_eval(pred_edges=edges, gt_edges=gt_path, all_ids=ids)
    for value in metrics:
        full_metrics[value] = metrics[value]

    return edge_metrics, cluster_metrics, full_metrics
