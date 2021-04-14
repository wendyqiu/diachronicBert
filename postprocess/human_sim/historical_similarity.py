
"""
compute historical similarity, using extracted features of language models based on the DUPS snippets
[step 3]
"""
import json
from os import path
import pickle
from operator import add
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.stats import spearmanr

DIR = 'D:/HistBERT/evaluation/human_sim/'
embed_save_path = path.join(DIR, 'temp', '12_full_embedding_dict.p')
index_save_path = path.join(DIR, 'temp', '12_full_index_dict.p')
sentence_text_path = path.join(DIR, 'temp', '12_full_sent_text_dict.p')
keywords_path = path.join(DIR, 'DUPS_form.list')
dups_sim_path = path.join(DIR, 'sim_scores.list')

PREV_PATH = path.join(DIR, 'extract_feature', 'dups_1_1910_3500.jsonl')
AFTER_PATH = path.join(DIR, 'extract_feature', 'dups_2_1910_3500.jsonl')

with open(keywords_path, 'rb') as handle:
    form_list = pickle.load(handle)
print(form_list)

prev_embedding = []
with open(PREV_PATH) as f:
    for line in f:
        prev_embedding.append(json.loads(line))
print("done 1st embedding...")

after_embedding = []
with open(AFTER_PATH) as f:
    for line in f:
        after_embedding.append(json.loads(line))
print("done 2nd embedding...")

form_list = form_list[-len(prev_embedding):]
print("length problem: {0}, {1}, {2}".format(len(form_list), len(prev_embedding), len(after_embedding)))
# error checking
if len(form_list) != len(prev_embedding) or len(form_list) != len(after_embedding) or len(after_embedding) != len(
        prev_embedding):
    raise ValueError("length problem: {0}, {1}, {2}".format(len(form_list), len(prev_embedding), len(after_embedding)))


old_embedding_list = []
index_list_first = []
full_text_first = []
print("\nprocessing first pair: \n")
for sentence_idx, prev_sentence in enumerate(prev_embedding):
    prev_features = prev_embedding[sentence_idx]["features"]
    print("sentence {}".format(sentence_idx))

    KEYWORD = form_list[sentence_idx]
    print("KEYWORD: {}".format(KEYWORD))

    word_idx_in_current_sent_first = []
    curr_sent_text_first = []
    curr_token_emb_old = []
    for word_idx, feature in enumerate(prev_features):
        token_text = feature["token"]
        curr_sent_text_first.append(token_text)
        token_embedding = feature["layers"][0]["values"]
        for i in range(len(feature["layers"])):
            token_embedding = list(map(add, token_embedding, feature["layers"][i]["values"]))
        if KEYWORD == token_text:
            curr_token_emb_old.append(token_embedding)
            print("original BERT: ")
            print(f"token: {token_text}")
            print(f"embedding: {token_embedding[:10]}")
            # add index to list
            word_idx_in_current_sent_first.append(word_idx)

    print("word_idx_in_current_sent_first: {}".format(word_idx_in_current_sent_first))
    index_list_first.append(word_idx_in_current_sent_first)
    full_text_first.append(curr_sent_text_first)
    if len(word_idx_in_current_sent_first) != 1:
        if curr_sent_text_first[0] == 'recent' or curr_sent_text_first[0] == 'public':
            old_embedding_list.append(curr_token_emb_old[0])
        elif curr_sent_text_first[0] == 'if' and curr_sent_text_first[1] == 'i':
            old_embedding_list.append(curr_token_emb_old[1])
        else:
            old_embedding_list.append(curr_token_emb_old[-1])
    else:
        old_embedding_list.append(curr_token_emb_old[-1])
    print("old_embedding_list: {}".format(old_embedding_list))


new_embedding_list = []
index_list_second = []
full_text_second = []
print("\nprocessing second pair: \n")
for a_sentence_idx, after_sentence in enumerate(after_embedding):
    after_features = after_embedding[a_sentence_idx]["features"]
    print("sentence {}".format(a_sentence_idx))

    KEYWORD_ = form_list[a_sentence_idx]
    print("KEYWORD: {}".format(KEYWORD_))

    curr_token_emb = []
    word_idx_in_current_sent_second = []
    curr_sent_text_second = []
    for a_word_idx, a_feature in enumerate(after_features):
        a_token_text = a_feature["token"]
        curr_sent_text_second.append((a_token_text))
        a_token_embedding = a_feature["layers"][0]["values"]
        for j in range(len(a_feature["layers"])):
            a_token_embedding = list(map(add, a_token_embedding, a_feature["layers"][j]["values"]))
        if KEYWORD_ == a_token_text:
            curr_token_emb.append(a_token_embedding)
            print("new BERT: ")
            print(f"token: {a_token_text}")
            print(f"embedding: {a_token_embedding[:10]}")
            # add index to list
            word_idx_in_current_sent_second.append(a_word_idx)
    new_embedding_list.append(curr_token_emb[-1])
    print("new_embedding_list: {}".format(new_embedding_list))
    print("word_idx_in_current_sent_second: {}".format(word_idx_in_current_sent_second))
    index_list_second.append(word_idx_in_current_sent_second)
    full_text_second.append(curr_sent_text_second)

print("\nFinishing up all embeddings...")
print("len(full_text_second): {}".format(len(full_text_second)))
print("old_embedding_list:")
print(len(old_embedding_list))
print("new_embedding_list:")
print(len(new_embedding_list))

# save embedding_dict to pickle to later compute cosine similarity
print("savinging to {}".format(embed_save_path))
embedding_dict = {'first': old_embedding_list, 'second': new_embedding_list}
print("embedding_dict:")
print(embedding_dict)
# input("check?")
with open(embed_save_path, 'wb') as handle:
    pickle.dump(embedding_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

index_dict = {'first': index_list_second, 'second': index_list_second}
with open(index_save_path, 'wb') as i_handle:
    pickle.dump(index_dict, i_handle, protocol=pickle.HIGHEST_PROTOCOL)

full_sent_text_dict = {'first': full_text_first, 'second': full_text_second}
print("full_sent_text_dict: {}".format(full_sent_text_dict))
with open(sentence_text_path, 'wb') as s_handle:
    pickle.dump(full_sent_text_dict, s_handle, protocol=pickle.HIGHEST_PROTOCOL)

# for feature in prev_features:
#     token_text = feature["token"]
#     # if token_text == KEYWORD:
#     token_embedding = feature["layers"][0]["values"]
#     print("original BERT: ")
#     print(f"token: {token_text}")
#     print(f"embedding: {token_embedding[:10]}")
#     print("\n")
#
# for a_feature in after_features:
#     a_token_text = a_feature["token"]
#     # if a_token_text == KEYWORD:
#     a_token_embedding = a_feature["layers"][0]["values"]
#     print("new BERT: ")
#     print(f"token: {a_token_text}")
#     print(f"embedding: {a_token_embedding[:10]}")
#     print("\n")

# automatic loading from pickle
with open(embed_save_path, 'rb') as handle:
    embedding_dict = pickle.load(handle)
old_emb_list = embedding_dict['first']
new_emb_list = embedding_dict['second']
print("old_emb_list:")
print(len(old_emb_list))
print("new_emb_list:")
print(len(new_emb_list))

with open(index_save_path, 'rb') as i_handle:
    index_list = pickle.load(i_handle)
print("index_list: {}".format(index_list))
first_index_list = index_list['first']
second_index_list = index_list['second']

with open(sentence_text_path, 'rb') as s_handle:
    sent_list = pickle.load(s_handle)
print("sent_list: {}".format(sent_list))

# verify index and embeddings match the keyword location in input texts
if len(sent_list) != len(index_list):
    raise Exception("ERROR: mismatch in length of sent_list and index_list: {0} vs. {1}"
                    .format(len(sent_list), len(index_list)))


# compute cosine similarity
similarity_list = []
for i, form in enumerate(form_list):
    curr_similarity = cosine_similarity(np.array(old_emb_list[i]).reshape(1, -1), np.array(new_emb_list[i]).reshape(1, -1))[0][0]
    similarity_list.append(curr_similarity)
    print("pair # {0}: \nsentence 1: {1} \nsentence 2: {2} \nsimilarity score: {3} \n\n".
          format(i+1, old_emb_list[i], new_emb_list[i], curr_similarity))

# compare old and new
print("similarity_list: ")
print(similarity_list)
print("average change in similarity: {}".format(sum(similarity_list)/len(similarity_list)))

print("max: {0} at {1}".format(max(similarity_list), similarity_list.index(max(similarity_list))))
print("min: {0} at {1}".format(min(similarity_list), similarity_list.index(min(similarity_list))))

# print(new_similarity_list[10])
# print(new_similarity_list[62])


# compute spearman's correlation b/w BERT and dups
with open(dups_sim_path, 'rb') as handle:
    dups_sim_scores = pickle.load(handle)

print("len(similarity_list): {}".format(len(similarity_list)))
dups_sim_scores = dups_sim_scores[-len(prev_embedding):]


print("computing spearman's correlation...")
if len(dups_sim_scores) != len(similarity_list):
    raise ValueError("similarity_list and dups_sim_scores have different lengths! {0} vs. {1}"
                     .format(len(similarity_list), len(dups_sim_scores)))
# dups_sim_scores.sort()
# similarity_list.sort()
dups_sim_scores = [i/4.0 for i in dups_sim_scores]
print("dups_sim_scores: {}".format(dups_sim_scores))
print("similarity_list: {}".format(similarity_list))

coef, _ = spearmanr(dups_sim_scores, similarity_list)
print("spearman's correlation of <{0}>: {1}, {2}".format('?', coef, _))



from collections import defaultdict

print("form_list: {}".format(form_list))
grouped_dict = defaultdict(list)
for i, x in enumerate(form_list):
    grouped_dict[x].append(i)
print("d: {}".format(grouped_dict))

for lemma, idx_list in grouped_dict.items():
    dups = [dups_sim_scores[x] for x in idx_list]
    man = [similarity_list[x] for x in idx_list]
    # dups.sort()
    # man.sort()
    coef, _ = spearmanr(dups, man)
    print("spearman's correlation of <{0}>: {1}, {2}".format(lemma, coef, _))
