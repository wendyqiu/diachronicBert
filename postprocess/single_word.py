# view the output of single sample for $KEYWORD$
# prev_embedding: the original BERT
# after_embedding: additional pretraining on COHA texts that cover the word $KEYWORD$
import json
from os import path
import pickle

KEYWORD = 'coach'

DIR = 'C:/Users/Mizuk/Documents/BERT/after_model/coach'
save_path = path.join(DIR, 'pickle', 'embedding_list.p')
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

embedding_list = []
for idx in range(len(prev_embedding)):
    prev_features = prev_embedding[idx]["features"]
    after_features = after_embedding[idx]["features"]
    print("sentence {}".format(idx))

    current_pair = {}
    for i in range(len(prev_features)):
        feature = prev_features[i]
        token_text = feature["token"]
        token_embedding = feature["layers"][0]["values"]
        if token_text == KEYWORD:
            current_pair['old'] = token_embedding
            print("original BERT: ")
            print(f"token: {token_text}")
            print(f"embedding: {token_embedding[:10]}")

        a_feature = after_features[i]
        a_token_text = a_feature["token"]
        a_token_embedding = a_feature["layers"][0]["values"]
        if token_text == KEYWORD:
            current_pair['new'] = a_token_embedding
            print("new BERT: ")
            print(f"token: {a_token_text}")
            print(f"embedding: {a_token_embedding[:10]}")
            print("\n")

    embedding_list.append(current_pair)

# save embedding_list to pickle to later compute cosine similarity
with open(save_path, 'wb') as handle:
    pickle.dump(embedding_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

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