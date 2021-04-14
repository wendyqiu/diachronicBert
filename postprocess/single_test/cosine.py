"""
calculate cosine similarity b/w old and new BERT

index_list: {[list of idx for the selected keyword], [], []...}
eg. [[2, 14], [8], [16], [2, 13], [10], [18]]

embedding_dict:
{'old': [list of embedding vectors], 'new': [list of embedding vectors]}

"""

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from os import path
import pickle
from transformers import BertTokenizer

MANUAL = False
KEYWORD = 'full_coha'

DIR = path.join('C:/Users/Mizuk/Documents/BERT/after_model/', KEYWORD)
embed_save_path = path.join(DIR, 'pickle', '12_full_embedding_dict.p')
index_save_path = path.join(DIR, 'pickle', '12_full_index_list.p')
sentence_text_path = path.join(DIR, 'pickle', '12_full_sent_text.p')

# find the index of WORD (not CHAR) of keyword given the sentence
# note that this tokenizer is the BASIC one, without [CLS] and [SEP] so the indexing is different from the input texts
def word_search(KEYWORD, sentence):
    idx_list = []
    tz = BertTokenizer.from_pretrained("bert-base-uncased")
    words = tz.tokenize(sentence.lower())
    for i, word in enumerate(words):
        if KEYWORD.lower() in word:
            idx_list.append(i)
    return idx_list


if MANUAL:
    sent_list = ["Here comes Cinderella's coach to take a princess to her fairy tale.",
                 "If you can't catch the bus, you can go for an express coach.",
                 "The coachman and footman are standing in front of the coach.",
                 "The coachman and footman are standing in front of the stage coach.",
                 ]

    old_emb_list = [
        [0.585397, -0.40014, -0.514952, 0.230011, -0.498719, 0.664502, 0.545653, 0.340335, -0.024614, -0.274674],
        [0.718635, -0.937265, -0.041139, 0.081166, -0.411698, -0.193864, 0.506655, 0.940227, 0.124328, 0.067172],
        [0.588245, -0.143918, -0.097894, -0.278353, 0.199236, -0.0422, 0.643288, 0.589765, 0.064859, -0.65601],
        [0.471816, -0.248529, 0.34731, -0.296595, 0.283033, -0.03519, 0.79194, 0.383818, 0.077225, -0.047714],
    ]

    new_emb_list = [
        [0.340819, -0.316007, -0.388633, 0.225926, -0.352641, 0.729501, 0.338422, 0.279181, -0.131765, -0.312808],
        [0.60515, -0.91075, 0.03075, 0.243405, -0.301251, -0.10162, 0.267794, 0.865666, 0.125156, -0.109787],
        [0.449564, -0.114714, -0.055027, -0.151782, 0.208818, 0.154318, 0.460523, 0.400324, 0.079055, -0.770398],
        [0.345147, -0.265823, 0.385962, -0.154863, 0.211088, 0.0707, 0.591785, 0.076552, 0.162814, -0.251823],
    ]
else:
    # automatic loading from pickle
    with open(embed_save_path, 'rb') as handle:
        embedding_dict = pickle.load(handle)
    old_emb_list = embedding_dict['old']
    new_emb_list = embedding_dict['new']
    with open(index_save_path, 'rb') as i_handle:
        index_list = pickle.load(i_handle)
    print("index_list: {}".format(index_list))
    with open(sentence_text_path, 'rb') as s_handle:
        sent_list = pickle.load(s_handle)
    print("sent_list: {}".format(sent_list))

    # verify index and embeddings match the keyword location in input texts
    if len(sent_list) != len(index_list):
        raise Exception("ERROR: mismatch in length of sent_list and index_list: {0} vs. {1}"
                        .format(len(sent_list), len(index_list)))
    # else:
    #     # search for all occurrences of the keyword in input texts
    #     for sent_idx, sentence in enumerate(sent_list):
    #         print("sentence: {}".format(sentence))
    #         keyword_idx = word_search(KEYWORD, sentence)
    #         if len(keyword_idx) != len(index_list[sent_idx]):
    #             raise Exception("ERROR: mismatch in keyword index! "
    #                             "keyword_idx from input texts: {0}, index_list[sent_idx] from pickle: {1}"
    #                             .format(keyword_idx, index_list[sent_idx]))

# compute cosine similarity
old_similarity_list = []
for first_old in old_emb_list:
    for second_old in old_emb_list:
        first = old_emb_list.index(first_old)
        second = old_emb_list.index(second_old)
        if first < second:
            old_similarity = \
                cosine_similarity(np.array(first_old).reshape(1, -1), np.array(second_old).reshape(1, -1))[0][0]
            old_similarity_list.append(old_similarity)
            print("old: sentence {0} vs. {1}: {2}".format(first + 1, second + 1, old_similarity))

new_similarity_list = []
for first_new in new_emb_list:
    for second_new in new_emb_list:
        n_first = new_emb_list.index(first_new)
        n_second = new_emb_list.index(second_new)
        if n_first < n_second:
            new_similarity = \
                cosine_similarity(np.array(first_new).reshape(1, -1), np.array(second_new).reshape(1, -1))[0][0]
            new_similarity_list.append(new_similarity)
            print("new: sentence {0} vs. {1}: {2}".format(n_first + 1, n_second + 1, new_similarity))


# compare old and new
difference = [new_similarity_list[i] - old_similarity_list[i] for i in range(len(old_similarity_list))]
print("difference in similarity: ")
print(difference)
print("average change in similarity: {}".format(sum(difference)/len(difference)))

print("max: {0} at {1}".format(max(difference), difference.index(max(difference))))
print("min: {0} at {1}".format(min(difference), difference.index(min(difference))))

print(new_similarity_list[10])
print(new_similarity_list[62])



# todo: print pairwise (triangle) results
# for sent_idx, token_idx_list in enumerate(index_list):
#     print("sentence {0} has {1} occurences of keyword {2}:".format(sent_idx+1, len(token_idx_list), KEYWORD))
#     print(sent_list[sent_idx])
#     print("")


