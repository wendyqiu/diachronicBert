"""

"""

import plotly
import pickle
import plotly.graph_objs as go
from sklearn.decomposition import PCA
from os import path

KEYWORD = 'coach'

DIR = path.join('C:/Users/Mizuk/Documents/BERT/after_model/', KEYWORD)
embed_save_path = path.join(DIR, 'pickle', '12_embedding_dict.p')
index_save_path = path.join(DIR, 'pickle', '12_index_list.p')
sentence_text_path = path.join(DIR, 'pickle', '12_full_sent_text.p')

def display_pca_scatterplot_2D(old_list, new_list, sentences, topn=1):
    print("len of old_emb_list: {}".format(len(old_list)))

    full_list = old_list + new_list
    two_dim = PCA(random_state=0).fit_transform(full_list)[:, :2]

    print("len of two_dim: {}".format(len(two_dim)))

    data = []
    count = 0

    for i in range(len(full_list)):
        if i >= len(old_list):
            trace = go.Scatter(
                x=two_dim[count:count + topn, 0],
                y=two_dim[count:count + topn, 1],
                # # z=three_dim[count:count + topn, 2],
                # text=sentences[count:count + topn],
                text='new_' + str(i % (len(old_list))+1),
                textposition="top center",
                textfont_size=20,
                mode='markers+text',
                marker={
                    'size': 10,
                    'opacity': 0.8,
                    'color': 2
                },
                showlegend=False
            )
        else:
            name_ = str(i+1) + '_' + sentences[i%(len(old_list))]
            trace = go.Scatter(
                x=two_dim[count:count + topn, 0],
                y=two_dim[count:count + topn, 1],
                # # z=three_dim[count:count + topn, 2],
                # text=sentences[count:count + topn],
                text=str(i%(len(old_list))+1),
                name=name_,
                textposition="top center",
                textfont_size=20,
                mode='markers+text',
                marker={
                    'size': 10,
                    'opacity': 0.8,
                    'color': 2
                }

            )

        data.append(trace)
        count += topn

    # for j in range(len(new_list)):
    #     print("j: {}".format(j))
    #     text_j = str(j) + '_new'
    #     trace_input = go.Scatter(
    #         x=two_dim[count:count + topn, 0],
    #         y=two_dim[count:count + topn, 1],
    #         # # z=three_dim[count:count + topn, 2],
    #         # text=sentences[count:count + topn],
    #         text=text_j,
    #         name=sentence_i,
    #         textposition="top center",
    #         textfont_size=20,
    #         mode='markers+text',
    #         marker={
    #             'size': 10,
    #             'opacity': 0.8,
    #             'color': 2
    #         }
    #     )
    #     data.append(trace_input)

    # Configure the layout

    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
        showlegend=True,
        legend=dict(
            x=1,
            y=0.5,
            font=dict(
                family="Courier New",
                size=25,
                color="black"
            )),
        font=dict(
            family=" Courier New ",
            size=15),
        autosize=False,
        width=1000,
        height=1000
    )

    plot_figure = go.Figure(data=data, layout=layout)
    plot_figure.show()

# def display_pca_scatterplot_nD(full, user_input=None, words=None, topn=5, sample=10):
#
#     # word_vectors = np.array([model[w] for w in words])
#
#     word_vectors = full
#     # three_dim = PCA(random_state=0).fit_transform(word_vectors)[:, :3]
#     # For 2D, change the three_dim variable into something like two_dim like the following:
#     two_dim = PCA(random_state=0).fit_transform(word_vectors)[:,:2]
#
#     data = []
#     count = 0
#
#     for i in range(len(user_input)):
#         trace = go.Scatter(
#             x=two_dim[count:count + topn, 0],
#             y=two_dim[count:count + topn, 1],
#             # z=three_dim[count:count + topn, 2],
#             text=words[count:count + topn],
#             name=user_input[i],
#             textposition="top center",
#             textfont_size=20,
#             mode='markers+text',
#             marker={
#                 'size': 10,
#                 'opacity': 0.8,
#                 'color': 2
#             }
#
#         )
#
#         # For 2D, instead of using go.Scatter3d, we need to use go.Scatter and delete the z variable. Also, instead of using
#         # variable three_dim, use the variable that we have declared earlier (e.g two_dim)
#
#         data.append(trace)
#         count = count + topn
#
#     trace_input = go.Scatter(
#         x=two_dim[count:, 0],
#         y=two_dim[count:, 1],
#         # z=three_dim[count:, 2],
#         text=words[count:],
#         name='input words',
#         textposition="top center",
#         textfont_size=20,
#         mode='markers+text',
#         marker={
#             'size': 10,
#             'opacity': 1,
#             'color': 'black'
#         }
#     )
#
#     # For 2D, instead of using go.Scatter3d, we need to use go.Scatter and delete the z variable.  Also, instead of using
#     # variable three_dim, use the variable that we have declared earlier (e.g two_dim)
#
#     data.append(trace_input)
#
#     # Configure the layout
#
#     layout = go.Layout(
#         margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
#         showlegend=True,
#         legend=dict(
#             x=1,
#             y=0.5,
#             font=dict(
#                 family="Courier New",
#                 size=25,
#                 color="black"
#             )),
#         font=dict(
#             family=" Courier New ",
#             size=15),
#         autosize=False,
#         width=1000,
#         height=1000
#     )
#
#     plot_figure = go.Figure(data=data, layout=layout)
#     plot_figure.show()


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

full = old_emb_list + new_emb_list
word_len = len(old_emb_list)
old_labels = [str(x+1) for x in range(word_len)]
new_labels = [str(x+1) + '_new' for x in range(word_len)]

labels = ["Cinderella's coach", "stagecoach", "coachman", "stagecoach", "(express) coach", "express coach",
              "coachman", "stage-coach", "head coach", "(stage) coach", "go coach", "stagecoach", "mail coaches",
              "Coach IP", "Coach New York", "brand Coach", "Shop Coach new arrivals"]

display_pca_scatterplot_2D(old_emb_list, new_emb_list, labels, topn=1)
# display_pca_scatterplot_2D(full, user_input=['a'], words=old_labels+new_labels)
# display_pca_scatterplot_3D(model, user_input, similar_word, labels, color_map)