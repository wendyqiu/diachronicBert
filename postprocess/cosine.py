"""calculate cosine similarity b/w old and new BERT"""

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

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

# TODO: load pickled embedding_list
# with open(save_path, 'rb') as handle:
#     embedding_list = pickle.load(handle)

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
