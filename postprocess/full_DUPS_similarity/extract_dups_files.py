"""
extract segments from DUPS, one text file per lemma
also store a matrix for human similarity
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
# form_list_path = path.join(DIR, 'BERT/after_model/full_coha/pickle/human_sim/DUPS_form.list')
# score_list_path = path.join(DIR, 'BERT/after_model/full_coha/pickle/human_sim/sim_scores.list')
DUPS_path = path.join(DIR, 'phD/csc2611/small_testset/DUPS/aggregate.csv')
human_sim_path = path.join(DIR, 'BERT/after_model/full_coha/pickle/human_sim/fixed/human_sim.matrix')
segments_dict_path = path.join(DIR, 'BERT/after_model/full_coha/pickle/human_sim/fixed/DUPS_segments.dict')
index_dict_path = path.join(DIR, 'BERT/after_model/full_coha/pickle/human_sim/fixed/index_list_of.dict')

# text_file_1 = path.join(DIR, 'BERT/after_model/full_coha/pickle/human_sim/DUPS_sentences_1.txt')
# text_file_2 = path.join(DIR, 'BERT/after_model/full_coha/pickle/human_sim/DUPS_sentences_2.txt')
# split_dir = 'D:/HistBERT/evaluation/human_sim/new_split'

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

id_a_set_dict = {}
id_b_set_dict = {}
for datum in all_data:
    lemma = datum[f2i['lemma']]
    id_a = int(datum[f2i['id_a']])
    try:
        id_a_set_dict[lemma].add(id_a)
    except KeyError:
        id_a_set_dict[lemma] = {id_a}
    id_b = int(datum[f2i['id_b']])
    try:
        id_b_set_dict[lemma].add(id_b)
    except KeyError:
        id_b_set_dict[lemma] = {id_b}
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

print("id_a_set_dict: {}".format(id_a_set_dict))
print("id_b_set_dict: {}".format(id_b_set_dict))


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
print("sim_matrices")
print(sim_matrices)
# write to file/pickles
with open(human_sim_path, 'wb') as m_handle:
    pickle.dump(sim_matrices, m_handle, protocol=pickle.HIGHEST_PROTOCOL)
print("saving sim_matrices to {}".format(human_sim_path))


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

bert_sim_matrices = {}
segment_dict = {}
index_dict_of_list = {}
# for lemma in tqdm(judgements):
for lemma in ['virus', 'sphere']:
    bert_sim_matrices[lemma] = np.zeros_like(sim_matrices[lemma])
    id_max = max(id_a_set_dict[lemma] | id_b_set_dict[lemma])
    raw_sent_list = []
    index_list = []
    for cur_id in range(id_max):
        form, raw_sent = snippets[lemma][cur_id]
        processed_sent = raw_sent.replace("[[", "")
        processed_sent = processed_sent.replace("]]", "")

        # get token index
        tokens_sent = tokenizer.tokenize(raw_sent)
        new_tokens_sent = []
        skip_till = -1
        target_pos = None
        form_index = None
        for i, tok in enumerate(tokens_sent):
            if i <= skip_till:
                continue
            if tok == '[' and tokens_sent[i + 1] == '[' and tokens_sent[i + 2] == form:
                skip_till = i + 4
                target2_pos = len(new_tokens_sent)
                new_tokens_sent.append(form)
                form_index = i
            elif tok == '[' and tokens_sent[i + 1] == '[' and tokens_sent[i + 2] == lemma \
                    and tokens_sent[i + 3].startswith('##'):
                skip_till = i + 5
                target2_pos = len(new_tokens_sent)
                new_tokens_sent.append(lemma)
                new_tokens_sent.append(tokens_sent[i + 3])
                form_index = i
            else:
                new_tokens_sent.append(tok)

        tokens_processed_sent = tokenizer.tokenize(processed_sent)
        if tokens_processed_sent[form_index] != lemma:
            print("lemma vs. tokens_processed_sent[form_index]: {0} vs. {1}".format(lemma, tokens_processed_sent[form_index]))
            raise ValueError('process-tokenize and tokenize-process mismatch!')
        raw_sent_list.append(processed_sent)

        print("tokens_sent: {}".format(tokens_sent))
        print("new_tokens_sent: {}".format(new_tokens_sent))
        print("form_index: {}".format(form_index))
        if form_index == None:
            raise ValueError('form_index is None???')
        index_list.append(form_index)
        print("check lemma vs. form from from_index: {0} vs. {1}".format(lemma, new_tokens_sent[form_index]))

        # token_ids = tokenizer.encode(new_tokens_sent)
        # print("token_ids: {}".format(token_ids))
        # input("check?")

    if len(raw_sent_list) != len(index_list):
        print("length check: raw_sent_list vs. index_list: {0} vs. {1}".format(len(raw_sent_list), len(index_list)))
        raise ValueError('form_index is None???')
    segment_dict[lemma] = raw_sent_list
    index_dict_of_list[lemma] = index_list

    # todo: write to dump...
    # write to file/pickles
    with open(segments_dict_path, 'wb') as d_handle:
        pickle.dump(segment_dict, d_handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("done writing segment_dict at {}".format(segments_dict_path))

    with open(index_dict_path, 'wb') as i_handle:
        pickle.dump(index_dict_of_list, i_handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("done writing index_dict_of_list at {}".format(index_dict_path))

    for lemma, sent_list in segment_dict.items():
        # write sentences to one text file for each lemma
        sent_list

    exit()

    for (id_a, id_b) in judgements[lemma]:

        form1, s1 = snippets[lemma][id_a]
        form2, s2 = snippets[lemma][id_b]

        if form1 != form2:
            print("form1: {}".format(form1))
            print("form2: {}".format(form2))
            raise ValueError('form1 and form2 not the same')

        tokens_s1 = tokenizer.tokenize(s1)
        tokens_s2 = tokenizer.tokenize(s2)

        new_tokens_s1 = []
        skip_till = -1
        target1_pos = None
        for i, tok in enumerate(tokens_s1):
            if i <= skip_till:
                continue
            if tok == '[' and tokens_s1[i + 1] == '[' and tokens_s1[i + 2] == form1:
                skip_till = i + 4
                target1_pos = len(new_tokens_s1)
                new_tokens_s1.append(form1)
            elif tok == '[' and tokens_s1[i + 1] == '[' and tokens_s1[i + 2] == lemma and tokens_s1[i + 3].startswith(
                    '##'):
                skip_till = i + 5
                target1_pos = len(new_tokens_s1)
                new_tokens_s1.append(lemma)
                new_tokens_s1.append(tokens_s1[i + 3])
            else:
                new_tokens_s1.append(tok)

        new_tokens_s2 = []
        skip_till = -1
        target2_pos = None
        for i, tok in enumerate(tokens_s2):
            if i <= skip_till:
                continue
            if tok == '[' and tokens_s2[i + 1] == '[' and tokens_s2[i + 2] == form2:
                skip_till = i + 4
                target2_pos = len(new_tokens_s2)
                new_tokens_s2.append(form2)
            elif tok == '[' and tokens_s2[i + 1] == '[' and tokens_s2[i + 2] == lemma and tokens_s2[i + 3].startswith(
                    '##'):
                skip_till = i + 5
                target2_pos = len(new_tokens_s2)
                new_tokens_s2.append(lemma)
                new_tokens_s2.append(tokens_s2[i + 3])
            else:
                new_tokens_s2.append(tok)

        token_ids_1 = tokenizer.encode(new_tokens_s1)
        token_ids_2 = tokenizer.encode(new_tokens_s2)

        with torch.no_grad():
            input_ids_tensor_1 = torch.tensor([token_ids_1])
            input_ids_tensor_2 = torch.tensor([token_ids_2])

            outputs_1 = lm(input_ids_tensor_1)
            outputs_2 = lm(input_ids_tensor_2)

            #             print(outputs_1[0].shape, outputs_2[1].shape)

            hidden_states_1 = np.stack([l.clone().numpy() for l in outputs_1[2]])
            hidden_states_2 = np.stack([l.clone().numpy() for l in outputs_2[2]])

            hidden_states_1 = hidden_states_1.squeeze(1)
            hidden_states_2 = hidden_states_2.squeeze(1)

            usage_vector_1 = hidden_states_1[LAYER_SEQ, target1_pos, :]
            usage_vector_2 = hidden_states_2[LAYER_SEQ, target2_pos, :]

            if MODE == 'sum':
                usage_vector_1 = np.sum(usage_vector_1, axis=0)
                usage_vector_2 = np.sum(usage_vector_2, axis=0)
            else:
                usage_vector_1 = usage_vector_1.reshape((usage_vector_1.shape[0] * usage_vector_1.shape[1]))
                usage_vector_2 = usage_vector_2.reshape((usage_vector_2.shape[0] * usage_vector_2.shape[1]))
            #
            #             print(usage_vector_1.shape, usage_vector_2.shape)

            sim_score = cosine_similarity(usage_vector_1, usage_vector_2)
            bert_sim_matrices[lemma][id_a, id_b] = sim_score
            bert_sim_matrices[lemma][id_b, id_a] = sim_score

layer_seq_str = ','.join(list(map(str, LAYER_SEQ)))
with open('{}-{}-{}.dict'.format(MODEL, MODE, layer_seq_str), 'wb') as f:
    pickle.dump(obj=bert_sim_matrices, file=f)

coeffs = {}
sig_coeffs = {}
for w in bert_sim_matrices:
    coeff, p_value, n = mantel(
        sim_matrices[w],
        bert_sim_matrices[w],
        method='spearman',  # pearson
        permutations=999,
        alternative='two-sided'  # greater, less
    )
    print(w)
    print('spearman: {:.2f}    p: {:.2f}'.format(coeff, p_value))

    coeffs[w] = coeff, p_value
    if p_value < 0.05:
        sig_coeffs[w] = coeff, p_value

print('{}/{} significant correlations'.format(len(sig_coeffs), len(coeffs)))
for w, (c, p) in sig_coeffs.items():
    print('{}  spearman: {:.2f}    p: {:.2f}'.format(w, c, p))

with open('{}-{}-{}.corrs.dict'.format(MODEL, MODE, layer_seq_str), 'wb') as f:
    pickle.dump(obj=bert_sim_matrices, file=f)




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