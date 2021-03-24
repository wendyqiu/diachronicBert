"""
pre-training on BERT takes a lot of resources and time
for efficiency, select relevant data from COHA
and use these samples in the domain-specific pre-training stage
"""
import os
import re
from nltk import tokenize

# decade/directory number (from 1 to 10)
NUM = 10
current_decade = '19' + str(NUM) + '0'
CHAR_SET = 'utf-8'

# 16 words in the human similarity dataset (PWHS: pairwise human similarity)
PWHS = ["federal", "spine", "optical", "compact", "signal", "leaf", "net", "coach", "sphere", "mirror",
        "card", "virus", "disk", "brick", "virtual", "energy", "environmental", "users", "virtual", "disk"]

# words from GEMS dataset (incomplete)
GEMS = ["environmental", "users", "virtual", "disk", "tenure", "coach", "address"]

#

filter_keywords = PWHS + GEMS

CORPUS_DIR = 'C:/Users/Mizuk/Documents/phD/csc2611/COHA/raw/'
OUT_DIR = 'C:/Users/Mizuk/Documents/phD/csc2611/COHA/filtered/'

# read in files
sub_dir_list = []
for dirpath, dirnames, files in os.walk(CORPUS_DIR):
    sub_dir_list.append(dirpath)

current_dir = sub_dir_list[NUM]
current_out_dir = os.path.join(OUT_DIR, current_decade)
print("current directory: {}".format(current_dir))
print("output directory: {}".format(current_out_dir))

all_counter = 0
use_counter = 0
for file_name in os.listdir(current_dir):
    file_path = os.path.join(current_dir, file_name)
    if file_path.endswith(".txt"):
        all_counter += 1
        print("processing {}".format(file_path))
        # input("correct?")
        with open(file_path, 'r', encoding=CHAR_SET) as fr:
            text = fr.read()
            # check filter keywords
            save = False
            for search_word in filter_keywords:
                if save:
                    break
                elif search_word in text:
                    print("search_word is {}".format(search_word))
                    save = True
            if save:
                use_counter += 1
                # write to a new file
                output_path = os.path.join(current_out_dir, file_name)
                print("saving to : {}".format(output_path))
                with open(output_path, 'w', encoding=CHAR_SET) as fw:
                    # write sentence in lines
                    for sent in tokenize.sent_tokenize(text):
                        sent += "\n"
                        fw.write(sent)

print("total number of files processed in {0}s: {1}".format(current_decade, all_counter))
print("number of usable files after filtering: {}".format(use_counter))