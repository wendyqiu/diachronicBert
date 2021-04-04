"""
view the output of single sample for $KEYWORD$
prev_embedding: the original BERT
after_embedding: additional pretraining on COHA texts that cover the word $KEYWORD$

index_dict: {'sentence_idx': [list of idx for the selected keyword], '': [], '': [], ...}
eg. {'0': [2, 14], '1': [8], '2': [16], '3': [2, 13], '4': [10], '5': [18]}

embedding_dict:
{'old': [list of embedding vectors], 'new': [list of embedding vectors]}

"""
import json
from os import path
import pickle

KEYWORD = 'coach'

DIR = 'C:/Users/Mizuk/Documents/BERT/after_model/coach'
embed_save_path = path.join(DIR, 'pickle', 'embedding_dict.p')
index_save_path = path.join(DIR, 'pickle', 'index_dict.p')

PREV_PATH = path.join(DIR, '0_output.jsonl')
AFTER_PATH = path.join(DIR, '1000_output.jsonl')

prev_embedding = []
with open(PREV_PATH) as f:
    for line in f:
        prev_embedding.append(json.loads(line))

after_embedding = []
with open(AFTER_PATH) as f:
    for line in f:
        after_embedding.append(json.loads(line))

old_embedding_list = []
new_embedding_list = []
index_dict = {}
for sentence_idx in range(len(prev_embedding)):
    prev_features = prev_embedding[sentence_idx]["features"]
    after_features = after_embedding[sentence_idx]["features"]
    print("sentence {}".format(sentence_idx))

    word_idx_in_current_sent = []
    for word_idx in range(len(prev_features)):
        feature = prev_features[word_idx]
        token_text = feature["token"]
        token_embedding = feature["layers"][0]["values"]
        if token_text == KEYWORD:
            old_embedding_list.append(token_embedding)
            print("original BERT: ")
            print(f"token: {token_text}")
            print(f"embedding: {token_embedding[:10]}")
            # add index to list
            word_idx_in_current_sent.append(word_idx)

        a_feature = after_features[word_idx]
        a_token_text = a_feature["token"]
        a_token_embedding = a_feature["layers"][0]["values"]
        if token_text == KEYWORD:
            new_embedding_list.append(a_token_embedding)
            print("new BERT: ")
            print(f"token: {a_token_text}")
            print(f"embedding: {a_token_embedding[:10]}")
            print("\n")

    # add all keyword idx for this sentence to the index_dict
    index_dict[str(sentence_idx)] = word_idx_in_current_sent
    print("index_dict: {}".format(index_dict))

    print("old_embedding_list:")
    print(old_embedding_list)
    print("new_embedding_list:")
    print(new_embedding_list)

# save embedding_dict to pickle to later compute cosine similarity
embedding_dict = {'old': old_embedding_list, 'new': new_embedding_list}
print("embedding_dict:")
print(embedding_dict)
with open(embed_save_path, 'wb') as handle:
    pickle.dump(embedding_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
# save index_dict
with open(index_save_path, 'wb') as i_handle:
    pickle.dump(index_dict, i_handle, protocol=pickle.HIGHEST_PROTOCOL)

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