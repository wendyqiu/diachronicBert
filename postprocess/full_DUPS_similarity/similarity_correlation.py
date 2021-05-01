"""
step 2: compute similarity (inverse of Euclidean distance
step 3: mantel test of spearman's rank correlation
"""

import pickle
from os import path
import json
from operator import add

KEYWORD = 'virus'

DIR = 'C:/Users/Mizuk/Documents/'
save_DIR = path.join(DIR, 'BERT/after_model/full_coha/pickle/human_sim/fixed/')
human_sim_path = path.join(save_DIR, 'human_sim.matrix')
segments_dict_path = path.join(save_DIR, 'DUPS_segments.dict')
index_dict_path = path.join(save_DIR, 'index_list_of.dict')

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

for sentence_idx in range(len(original_embedding)):
    print("sentence_idx: {}".format(sentence_idx))
    original_features = original_embedding[sentence_idx]["features"]
    proto_features = proto_embedding[sentence_idx]["features"]
    five_features = five_embedding[sentence_idx]["features"]
    ten_features = ten_embedding[sentence_idx]["features"]
    full_features = full_embedding[sentence_idx]["features"]
    word_idx = word_idx_list[sentence_idx] + 1           # Important: [CLS] at start, so need to increment index by 1
    print("word_idx {}".format(word_idx))

    orig_word_emb_list = []
    proto_word_emb_list = []
    five_word_emb_list = []
    ten_word_emb_list = []
    full_word_emb_list = []
    orig_word_emb_list.append(get_word_vec(original_features, word_idx, lemma=KEYWORD))
    proto_word_emb_list.append(get_word_vec(proto_features, word_idx, lemma=KEYWORD))
    five_word_emb_list.append(get_word_vec(five_features, word_idx, lemma=KEYWORD))
    ten_word_emb_list.append(get_word_vec(ten_features, word_idx, lemma=KEYWORD))
    full_word_emb_list.append(get_word_vec(full_features, word_idx, lemma=KEYWORD))

    



# http://scikit-bio.org/docs/0.1.3/generated/skbio.math.stats.distance.mantel.html