"""
view the output of single sample for $KEYWORD$
prev_embedding: the original BERT
after_embedding: additional pretraining on COHA texts that cover the word $KEYWORD$

index_list: {[list of idx for the selected keyword], [], []...}
eg. [[2, 14], [8], [16], [2, 13], [10], [18]]

embedding_dict:
{'old': [list of embedding vectors], 'new': [list of embedding vectors]}

"""
import json
from os import path
import pickle

KEYWORD = 'coach'

DIR = 'C:/Users/Mizuk/Documents/BERT/after_model/coach'
embed_save_path = path.join(DIR, 'pickle', 'embedding_dict.p')
index_save_path = path.join(DIR, 'pickle', 'index_list.p')
sentence_text_path = path.join(DIR, 'pickle', 'full_sent_text.p')

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
index_list = []
full_sent_text = []
for sentence_idx in range(len(prev_embedding)):
    prev_features = prev_embedding[sentence_idx]["features"]
    after_features = after_embedding[sentence_idx]["features"]
    print("sentence {}".format(sentence_idx))

    word_idx_in_current_sent = []
    curr_sent_text = []
    for word_idx in range(len(prev_features)):
        feature = prev_features[word_idx]
        token_text = feature["token"]
        curr_sent_text.append(token_text)
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

    full_sent_text.append(curr_sent_text)

    # add all keyword idx for this sentence to the index_list
    index_list.append(word_idx_in_current_sent)
    print("index_list: {}".format(index_list))

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

with open(index_save_path, 'wb') as i_handle:
    pickle.dump(index_list, i_handle, protocol=pickle.HIGHEST_PROTOCOL)

print("full_sent_text: {}".format(full_sent_text))
with open(sentence_text_path, 'wb') as s_handle:
    pickle.dump(full_sent_text, s_handle, protocol=pickle.HIGHEST_PROTOCOL)

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