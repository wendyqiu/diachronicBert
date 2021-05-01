"""
part two: similarity & correlation
step 1: get one list of word_emb for each version of Bert, word_emb in ascending id order
step 2: compute similarity (inverse of Euclidean distance?)
step 3: mantel test of spearman's rank correlation
"""

import pickle
from os import path
import json
from operator import add
import numpy as np
from skbio.stats.distance import mantel

KEYWORD = 'net'

DIR = 'C:/Users/Mizuk/Documents/'
save_DIR = path.join(DIR, 'BERT/after_model/full_coha/pickle/human_sim/fixed/')
human_sim_path = path.join(save_DIR, 'human_sim.matrix')
segments_dict_path = path.join(save_DIR, 'DUPS_segments.dict')
index_dict_path = path.join(save_DIR, 'index_list_of.dict')
judgements_path = path.join(save_DIR, 'judgements.dict')

json_DIR = path.join(DIR, 'BERT/after_model/full_coha/new/', KEYWORD)
original_path = path.join(json_DIR, 'original_output.jsonl')
prototype_path = path.join(json_DIR, 'prototype_output.jsonl')
five_histbert_path = path.join(json_DIR, '5_histbert_output.jsonl')
ten_histbert_path = path.join(json_DIR, '10_histbert_output.jsonl')
full_histbert_path = path.join(json_DIR, 'full_output.jsonl')


"""helper funcs"""
def load_pickle(pickle_path):
    result_embedding = []
    with open(pickle_path) as f:
        for line in f:
            result_embedding.append(json.loads(line))
    return result_embedding


def get_word_vec(features, word_idx, lemma):
    feature = features[word_idx]
    token_text = feature["token"]
    print(token_text)
    if lemma != token_text:
        raise KeyError("lemma and token_text mismatch: {0} vs. {1}".format(lemma, token_text))
    token_embedding = feature["layers"][0]["values"]
    for i in range(len(feature["layers"])):
        token_embedding = list(map(add, token_embedding, feature["layers"][i]["values"]))
    print(f"token: {token_text}")
    print(f"embedding: {token_embedding[:10]}")
    return token_embedding


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def compute_sim_for_each_bert(word_emb_list):
    usage_vector_1 = word_emb_list[id_a]
    usage_vector_2 = word_emb_list[id_b]
    sim_score_ = cosine_similarity(usage_vector_1, usage_vector_2)
    return sim_score_


""" STEP 1: create word_emb lists """

# loading from pickle/files
with open(index_dict_path, 'rb') as i_handle:
    index_dict = pickle.load(i_handle)
print("done loading <index_dict> from {}".format(index_dict_path))

original_embedding = load_pickle(original_path)
proto_embedding = load_pickle(prototype_path)
five_embedding = load_pickle(five_histbert_path)
ten_embedding = load_pickle(ten_histbert_path)
full_embedding = load_pickle(full_histbert_path)
print("done loading pickles")

bert_embedding_list = []
proto_embedding_list = []
five_embedding_list = []
ten_embedding_list = []
full_embedding_list = []

if any(len(lst) != len(original_embedding) for lst in [proto_embedding, five_embedding,
                                                        ten_embedding, full_embedding]):
    print("length of original_embedding: {}".format(len(original_embedding)))
    print("length of proto_embedding: {}".format(len(proto_embedding)))
    print("length of five_embedding: {}".format(len(five_embedding)))
    print("length of ten_embedding: {}".format(len(ten_embedding)))
    print("length of full_embedding: {}".format(len(full_embedding)))
    raise KeyError("length mismatch in embedding lists")


word_idx_list = index_dict[KEYWORD]
print("lemma: {}".format(KEYWORD))
print("word_idx: {}".format(word_idx_list))

if len(original_embedding) != len(word_idx_list):
    print("length of embeddings: {}".format(len(original_embedding)))
    print("length of word_idx_list: {}".format(len(word_idx_list)))
    raise KeyError("length mismatch in length of embeddings vs. word_idx_list")

orig_word_emb_list = []
proto_word_emb_list = []
five_word_emb_list = []
ten_word_emb_list = []
full_word_emb_list = []
for sentence_idx in range(len(original_embedding)):
    print("sentence_idx: {}".format(sentence_idx))
    original_features = original_embedding[sentence_idx]["features"]
    proto_features = proto_embedding[sentence_idx]["features"]
    five_features = five_embedding[sentence_idx]["features"]
    ten_features = ten_embedding[sentence_idx]["features"]
    full_features = full_embedding[sentence_idx]["features"]
    word_idx = word_idx_list[sentence_idx] + 1           # Important: [CLS] at start, so need to increment index by 1
    print("word_idx {}".format(word_idx))

    orig_word_emb_list.append(get_word_vec(original_features, word_idx, lemma=KEYWORD))
    proto_word_emb_list.append(get_word_vec(proto_features, word_idx, lemma=KEYWORD))
    five_word_emb_list.append(get_word_vec(five_features, word_idx, lemma=KEYWORD))
    ten_word_emb_list.append(get_word_vec(ten_features, word_idx, lemma=KEYWORD))
    full_word_emb_list.append(get_word_vec(full_features, word_idx, lemma=KEYWORD))

    if any(len(lst) != len(orig_word_emb_list) for lst in [proto_word_emb_list, five_word_emb_list,
                                                           ten_word_emb_list, full_word_emb_list]):
        print("length of orig_word_emb_list: {}".format(len(orig_word_emb_list)))
        print("length of proto_word_emb_list: {}".format(len(proto_word_emb_list)))
        print("length of five_word_emb_list: {}".format(len(five_word_emb_list)))
        print("length of ten_word_emb_list: {}".format(len(ten_word_emb_list)))
        print("length of full_word_emb_list: {}".format(len(full_word_emb_list)))
        raise KeyError("length mismatch in WORD embedding lists")

print("step 1 finished")


""" STEP 2: get pair of word_emb from id_a, id_b, put into matrix """

with open(human_sim_path, 'rb') as m_handle:
    human_sim_matrices = pickle.load(m_handle)
print("done loading <human_sim_matrices> from {}".format(human_sim_matrices))

with open(judgements_path, 'rb') as j_handle:
    judgements = pickle.load(j_handle)
print("done loading <judgements> from {}".format(judgements_path))

bert_sim_matrices = {}
proto_sim_matrices = {}
five_sim_matrices = {}
ten_sim_matrices = {}
full_sim_matrices = {}
# todo: switch to process all keywords
# for w in judgements:
bert_sim_matrices[KEYWORD] = np.zeros_like(human_sim_matrices[KEYWORD])
proto_sim_matrices[KEYWORD] = np.zeros_like(human_sim_matrices[KEYWORD])
five_sim_matrices[KEYWORD] = np.zeros_like(human_sim_matrices[KEYWORD])
ten_sim_matrices[KEYWORD] = np.zeros_like(human_sim_matrices[KEYWORD])
full_sim_matrices[KEYWORD] = np.zeros_like(human_sim_matrices[KEYWORD])
for (id_a, id_b) in judgements[KEYWORD]:
    b_sim_score = compute_sim_for_each_bert(orig_word_emb_list)
    bert_sim_matrices[KEYWORD][id_a, id_b] = b_sim_score
    bert_sim_matrices[KEYWORD][id_b, id_a] = b_sim_score

    p_sim_score = compute_sim_for_each_bert(proto_word_emb_list)
    proto_sim_matrices[KEYWORD][id_a, id_b] = p_sim_score
    proto_sim_matrices[KEYWORD][id_b, id_a] = p_sim_score

    f_sim_score = compute_sim_for_each_bert(five_word_emb_list)
    five_sim_matrices[KEYWORD][id_a, id_b] = f_sim_score
    five_sim_matrices[KEYWORD][id_b, id_a] = f_sim_score

    t_sim_score = compute_sim_for_each_bert(ten_word_emb_list)
    ten_sim_matrices[KEYWORD][id_a, id_b] = t_sim_score
    ten_sim_matrices[KEYWORD][id_b, id_a] = t_sim_score

    full_sim_score = compute_sim_for_each_bert(full_word_emb_list)
    full_sim_matrices[KEYWORD][id_a, id_b] = full_sim_score
    full_sim_matrices[KEYWORD][id_b, id_a] = full_sim_score


""" STEP 3: compute correlation """

def compute_correlations(sim_matrices):
    coeffs_ = {}
    for w in sim_matrices:
        # todo: delete these 2 lines!
        if w != KEYWORD:
            continue
        coeff, p_value, n = mantel(
            human_sim_matrices[w],
            sim_matrices[w],
            method='spearman',  # pearson
            permutations=999,
            alternative='two-sided'  # greater, less
        )
        print(w)
        print('spearman: {:.4f}    p: {:.4f}'.format(coeff, p_value))
        coeffs_[w] = coeff, p_value
    return coeffs_


bert_sig_coeffs = {}
bert_coeffs = compute_correlations(bert_sim_matrices)
for w in bert_coeffs:
    coeff_, p_value_ = bert_coeffs[w]
    if p_value_ < 0.05:
        bert_sig_coeffs[w] = coeff_, p_value_
print("bert_sim_matrices:")
print('{}/{} significant correlations'.format(len(bert_sig_coeffs), len(bert_coeffs)))
for w, (c, p) in bert_sig_coeffs.items():
    print('{}  spearman: {:.4f}    p: {:.4f}'.format(w, c, p))


proto_sig_coeffs = {}
proto_coeffs = compute_correlations(proto_sim_matrices)
for w in proto_coeffs:
    coeff_, p_value_ = proto_coeffs[w]
    if p_value_ < 0.05:
        proto_sig_coeffs[w] = coeff_, p_value_
print("prototype:")
print('{}/{} significant correlations'.format(len(proto_sig_coeffs), len(proto_coeffs)))
for w, (c, p) in proto_sig_coeffs.items():
    print('{}  spearman: {:.4f}    p: {:.4f}'.format(w, c, p))


five_sig_coeffs = {}
five_coeffs = compute_correlations(five_sim_matrices)
for w in five_coeffs:
    coeff_, p_value_ = five_coeffs[w]
    if p_value_ < 0.05:
        five_sig_coeffs[w] = coeff_, p_value_
print("five_bert:")
print('{}/{} significant correlations'.format(len(five_sig_coeffs), len(five_coeffs)))
for w, (c, p) in five_sig_coeffs.items():
    print('{}  spearman: {:.4f}    p: {:.4f}'.format(w, c, p))


ten_sig_coeffs = {}
ten_coeffs = compute_correlations(ten_sim_matrices)
for w in ten_coeffs:
    coeff_, p_value_ = ten_coeffs[w]
    if p_value_ < 0.05:
        ten_sig_coeffs[w] = coeff_, p_value_
print("ten_bert:")
print('{}/{} significant correlations'.format(len(ten_sig_coeffs), len(ten_coeffs)))
for w, (c, p) in ten_sig_coeffs.items():
    print('{}  spearman: {:.4f}    p: {:.4f}'.format(w, c, p))

full_sig_coeffs = {}
full_coeffs = compute_correlations(full_sim_matrices)
for w in full_coeffs:
    coeff_, p_value_ = full_coeffs[w]
    if p_value_ < 0.05:
        full_sig_coeffs[w] = coeff_, p_value_
print("full_bert:")
print('{}/{} significant correlations'.format(len(full_sig_coeffs), len(full_coeffs)))
for w, (c, p) in full_sig_coeffs.items():
    print('{}  spearman: {:.4f}    p: {:.4f}'.format(w, c, p))

# http://scikit-bio.org/docs/0.1.3/generated/skbio.math.stats.distance.mantel.html