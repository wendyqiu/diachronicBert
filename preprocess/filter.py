"""
pre-training on BERT takes a lot of resources and time
for efficiency, select relevant data from COHA
and use these samples in the domain-specific pre-training stage
"""
import os
from nltk import tokenize

# 16 words in the human similarity dataset (PWHS: pairwise human similarity)
PWHS = ["federal", "spine", "optical", "compact", "signal", "leaf", "net", "coach", "sphere", "mirror",
        "card", "virus", "disk", "brick", "virtual", "energy", "environmental", "users", "virtual", "disk"]

# words from GEMS dataset (incomplete)
GEMS = ["environmental", "users", "virtual", "disk", "tenure", "coach", "address"]

#

filter_keywords = PWHS + GEMS

CORPUS_DIR = 'C:/Users/Mizuk/Documents/phD/csc2611/COHA/raw/'

# read in files
sub_dir_list = []
for dirpath, dirnames, files in os.walk(CORPUS_DIR):
    sub_dir_list.append(dirpath)

# todo: modify this
current_dir = sub_dir_list[1]
print("current directory: {}".format(current_dir))
for file_name in os.listdir(current_dir):
    file_path = os.path.join(current_dir, file_name)
    if file_path.endswith(".txt"):
        print("processing {}".format(file_path))
        with open(file_path, 'r', encoding='utf-8') as fr:
            text = fr.read()
            # check filter keywords
            for sentence in tokenize.sent_tokenize(text):
                # write to new file
                pass
