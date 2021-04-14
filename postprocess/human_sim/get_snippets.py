"""
get snippets from DUPS: aggregate.csv
sanitize and organize into 2 txt files, one for each group (pair) of the same lemma
for example, form_list = ['virus', 'sphere', ...]
text_1: ['there is no vaccine for this type of virus ...', 'the sphere shaped object ...', ]
text_2: ['scan your system to detect virus ...', 'we love our sphere ...', ]
[step 1]
"""

import numpy as np
import csv
from collections import defaultdict
from transformers import BertTokenizer
import re
import pickle
from os import path
from tqdm import tqdm

SPLIT = True        # output file too large, need to split and process
THRESHOLD = 500     # number of lines per file

DIR = 'C:/Users/Mizuk/Documents/'
form_list_path = path.join(DIR, 'BERT/after_model/full_coha/pickle/human_sim/DUPS_form.list')
score_list_path = path.join(DIR, 'BERT/after_model/full_coha/pickle/human_sim/sim_scores.list')
DUPS_path = path.join(DIR, 'phD/csc2611/small_testset/DUPS/aggregate.csv')
text_file_1 = path.join(DIR, 'BERT/after_model/full_coha/pickle/human_sim/DUPS_sentences_1.txt')
text_file_2 = path.join(DIR, 'BERT/after_model/full_coha/pickle/human_sim/DUPS_sentences_2.txt')
split_dir = 'D:/HistBERT/evaluation/human_sim/new_split'

# Load results aggregated by usage pair - using averaging
all_data = []
with open(DUPS_path, newline='\n', mode='r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in reader:
        all_data.append(row)

# field name -> field idx
f2i = {f: i for i, f in enumerate(all_data[0])}

# remove fields row
all_data = all_data[1:]

# Collect usages and judgements
score_list = []
snippets = defaultdict(dict)
judgements = defaultdict(dict)
for datum in all_data:
    lemma = datum[f2i['lemma']]
    id_a = int(datum[f2i['id_a']])
    id_b = int(datum[f2i['id_b']])
    a = datum[f2i['a']]
    b = datum[f2i['b']]

    # store sentence in word-specific snippet list
    if a not in snippets[lemma]:
        snippets[lemma][id_a] = a.lower()
    if b not in snippets[lemma]:
        snippets[lemma][id_b] = b.lower()

    # get all data and average the score
    sim_score_before = datum[f2i['sim_score']]
    sim_scores = [int(s) for s in sim_score_before.split('\n')]
    score_avg = sum(sim_scores) / len(sim_scores)
    score_list.append(score_avg)

    # store judgement in word-specific score list
    judgements[lemma][(id_a, id_b)] = float(score_avg)

# Reformat snippets so that it's clear what is the form of the target word (and its position in the sentence)
for w in snippets:
    for id_, sent in snippets[w].items():
        tokens = list(map(str.lower, sent.split()))
        form = None
        for t in tokens:
            if t.startswith('[[') and t.endswith(']]'):
                form = t[2:-2]
        snippets[w][id_] = (form, sent)

sim_matrices = {}
for w in judgements:
    n_sent = len(snippets[w])
    m = np.zeros((n_sent, n_sent))
    for (id_a, id_b), score in judgements[w].items():
        m[id_a, id_b] = float(score)
        m[id_b, id_a] = float(score)
    sim_matrices[w] = m
# print("sim_matrices")
# print(sim_matrices)

def remove_contractions(sent):
    print("raw: ")
    print(sent)
    sent = re.sub(r'\s([?,\'.:!/)%"](?:\s|$))', r'\1', sent)
    print("re: ")
    print(sent)
    sent = sent.replace("' ", "'").replace("( ", "(").replace(") .", ").").replace(" - - ", "--").\
        replace(" - ", "-").replace(" n't", "n't")
    print("final: ")
    print(sent)
    return sent

# get bert similarity scores:
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')

text_list_1 = []
text_list_2 = []
form_list_1 = []
form_list_2 = []
for lemma in tqdm(judgements):
# for lemma in ['virus', 'sphere']:

    for (id_a, id_b) in judgements[lemma]:

        form1, s1 = snippets[lemma][id_a]
        form2, s2 = snippets[lemma][id_b]

        if form1 != form2:
            print("form1: {}".format(form1))
            print("form2: {}".format(form2))
            raise ValueError('form1 and form2 not the same')

        s1 = s1.replace("[[", "")
        s1 = s1.replace("]]", "")
        s2 = s2.replace("[[", "")
        s2 = s2.replace("]]", "")

        f_s1 = remove_contractions(s1)
        f_s2 = remove_contractions(s2)
        text_list_1.append(f_s1)
        text_list_2.append(f_s2)

        form_list_1.append(form1)
        form_list_2.append(form2)

print("len(text_list_1): {}".format(len(text_list_1)))
print("len(text_list_2): {}".format(len(text_list_2)))
if len(text_list_1) != len(text_list_2):
    raise ValueError('length of list1 and 2 not the same')

print("form_list: {}".format(form_list_1))
print("form_list's length: {}".format(len(form_list_2)))
if form_list_1 != form_list_2:
    raise ValueError('form list 1 and 2 not the same')

# write to file/pickles
with open(score_list_path, 'wb') as handle:
    pickle.dump(score_list, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(form_list_path, 'wb') as handle:
    pickle.dump(form_list_1, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open(text_file_1, 'w') as wf:
    for sent_1 in text_list_1:
        wf.write(sent_1)
        wf.write('\n')
with open(text_file_2, 'w') as wf:
    for sent_2 in text_list_2:
        wf.write(sent_2)
        wf.write('\n')

if SPLIT:
    lines_per_file = THRESHOLD
    smallfile = None
    with open(text_file_1) as bigfile:
        for lineno, line in enumerate(bigfile):
            if lineno % lines_per_file == 0:
                if smallfile:
                    smallfile.close()
                small_filename = path.join(split_dir, 'pair_1_small_{}.txt'.format(lineno + lines_per_file))
                smallfile = open(small_filename, "w")
            smallfile.write(line)
        if smallfile:
            smallfile.close()

    smallfile2 = None
    with open(text_file_2) as bigfile2:
        for lineno2, line2 in enumerate(bigfile2):
            if lineno2 % lines_per_file == 0:
                if smallfile2:
                    smallfile2.close()
                small_filename2 = path.join(split_dir, 'pair_2_small_{}.txt'.format(lineno2 + lines_per_file))
                smallfile2 = open(small_filename2, "w")
            smallfile2.write(line2)
        if smallfile2:
            smallfile2.close()