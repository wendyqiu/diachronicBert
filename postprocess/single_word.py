# view the output of single sample for $KEYWORD$
# prev_embedding: the original BERT
# after_embedding: additional pretraining on COHA texts that cover the word $KEYWORD$
import json
from os import path

KEYWORD = 'coach'

DIR = 'C:/Users/Mizuk/Documents/BERT/after_model/tmp'
PREV_PATH = path.join(DIR, 'before_output_disk.jsonl')
AFTER_PATH = path.join(DIR, 'after_output_disk.jsonl')

prev_embedding = []
with open(PREV_PATH) as f:
    for line in f:
        prev_embedding.append(json.loads(line))
prev_features = prev_embedding[0]["features"]

after_embedding = []
with open(AFTER_PATH) as f:
    for line in f:
        after_embedding.append(json.loads(line))
after_features = after_embedding[0]["features"]

for feature in prev_features:
    token_text = feature["token"]
    if token_text == 'disk':
        token_embedding = feature["layers"][0]["values"]
        print("original BERT: ")
        print(f"token: {token_text}")
        print(f"embedding: {token_embedding[:10]}")
        print("\n")

for feature in after_features:
    a_token_text = feature["token"]
    if a_token_text == 'disk':
        a_token_embedding = feature["layers"][0]["values"]
        print("original BERT: ")
        print(f"token: {a_token_text}")
        print(f"embedding: {a_token_embedding[:10]}")
        print("\n")