"""Turn each text into a vector and perform clustering"""

from pprint import pprint

import pandas as pd
import numpy as np

import gensim
from gensim import corpora
from gensim.models import CoherenceModel, KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pyLDAvis
import pyLDAvis.gensim_models


def visualize(glove_file: str = 'vectors.txt') -> None:
    """Visualize the GloVe embedding generated using TSNE"""

    word_glove_vectors = pd.read_csv(glove_file, sep=' ', header=None, index_col=0)
    print(word_glove_vectors)
    print(word_glove_vectors.index)

    tsne = TSNE(n_components=2, random_state=0, n_iter=5000, perplexity=3)
    np.set_printoptions(suppress=True)
    T = tsne.fit_transform(word_glove_vectors)
    labels = word_glove_vectors.index

    plt.figure(figsize=(12, 6))
    plt.scatter(T[:, 0], T[:, 1], c='orange', edgecolors='r')
    for label, x, y in zip(labels, T[:, 0], T[:, 1]):
        plt.annotate(label, xy=(x + 1, y + 1), xytext=(0, 0), textcoords='offset points')


def convert_glove_w2v() -> None:
    """Generates word2vec vectors' file from GloVe file using Gensim library"""

    glove_file = 'vectors.txt'
    w2v_glove_file = 'word2vec.txt'
    glove2word2vec(glove_file, w2v_glove_file)


def tokenize_data() -> list:
    """Output sample.txt with words tokenized and each doc as a element in a list"""

    data_file = 'sample.txt'
    store_lst = []
    with open(data_file) as f:
        for line in f:
            lst = line.split()
            store_lst.append(lst)
        # content = f.readlines()
        # content = [x.strip() for x in content]
    print(store_lst[:1])
    return store_lst


def lda(w2v_file: str = 'word2vec.txt') -> None:
    """Creates LDA model and outputs topics and related scores"""

    # model = KeyedVectors.load_word2vec_format(w2v_file, binary=False)
    tokenized_corpora = tokenize_data()
    id2word = corpora.Dictionary(tokenized_corpora)
    corpus = [id2word.doc2bow(text) for text in tokenized_corpora]
    # print(corpus[:1])
    # print([[(id2word[id], freq) for id, freq in cp] for cp in corpus[:1]])

    lda_model = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                                id2word=id2word,
                                                num_topics=10,  # optimal 15
                                                random_state=42,
                                                update_every=1,
                                                chunksize=10,
                                                per_word_topics=True)
    pprint(lda_model.print_topics())

    # ------------------------------------------------------------
    # values that measures the model
    # a measure of how good the model is. lower the better.
    print('\nPerplexity: ', lda_model.log_perplexity(corpus))

    # topic coherence score
    coherence_model_lda = CoherenceModel(model=lda_model, texts=tokenized_corpora,
                                         dictionary=id2word, coherence='c_v')
    coherence_lda = coherence_model_lda.get_coherence()
    print('\nCoherence Score: ', coherence_lda)
    # ------------------------------------------------------------

    # Use pyLDAvis package to visualize the topics
    pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word)
    pyLDAvis.save_html(vis, 'lda_vis_v4.html')

    # ------------------------------------------------------------
    # obtain relevant tables
    # topics in docs
    display_topic_of_doc(ldamodel=lda_model, corpus=corpus, texts=tokenized_corpora)


def format_topics_sentences(ldamodel, corpus, texts):
    """Finds dominant topic in each sentence"""

    # Init output
    sent_topics_df = pd.DataFrame()

    # Get main topic in each document
    for i, row in enumerate(ldamodel[corpus]):
        print(row)
        # row = sorted(row, key=lambda x: (x[1]), reverse=True)
        row = sorted(row[0], key=lambda x: (x[1]), reverse=True)
        # Get the Dominant topic, Perc Contribution and Keywords for each document
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # => dominant topic
                wp = ldamodel.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                sent_topics_df = sent_topics_df.append(
                    pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]),
                    ignore_index=True)
            else:
                break
    sent_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']

    # Add original text to the end of the output
    contents = pd.Series(texts)
    sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
    return sent_topics_df


def display_topic_of_doc(ldamodel, corpus, texts) -> None:
    """Display a table that shows texts and its topics"""

    # format
    df_topic_sents_keywords = format_topics_sentences(ldamodel=ldamodel, corpus=corpus, texts=texts)

    # Format
    df_dominant_topic = df_topic_sents_keywords.reset_index()
    df_dominant_topic.columns = ['Document_No', 'Dominant_Topic',
                                 'Topic_Perc_Contrib', 'Keywords', 'Text']

    # Show
    print(df_dominant_topic)
    # Save
    # df_dominant_topic.to_csv('doc_topic.csv')
    # ------------------------------------------------------------

    # table that shows representative texts for topics
    sent_topics_sorteddf_mallet = pd.DataFrame()
    sent_topics_outdf_grpd = df_topic_sents_keywords.groupby('Dominant_Topic')
    for i, grp in sent_topics_outdf_grpd:
        sent_topics_sorteddf_mallet = pd.concat([sent_topics_sorteddf_mallet,
                                                 grp.sort_values(['Perc_Contribution'],
                                                                 ascending=[0]).head(1)],
                                                axis=0)

    # Reset Index
    sent_topics_sorteddf_mallet.reset_index(drop=True, inplace=True)

    # Format
    sent_topics_sorteddf_mallet.columns = ['Topic_Num', "Topic_Perc_Contrib", "Keywords", "Text"]

    # Show
    sent_topics_sorteddf_mallet.head()

    # Save
    # sent_topics_sorteddf_mallet.to_csv('topic_doc.csv')

    # ------------------------------------------------------------
    # topic distribution across documents

    # Number of Documents for Each Topic
    topic_counts = df_topic_sents_keywords['Dominant_Topic'].value_counts()
    # Percentage of Documents for Each Topic
    topic_contribution = round(topic_counts / topic_counts.sum(), 4)
    # Topic Number and Keywords
    topic_num_keywords = df_topic_sents_keywords[['Dominant_Topic', 'Topic_Keywords']]
    # Concatenate Column wise
    df_dominant_topics = pd.concat([topic_num_keywords, topic_counts, topic_contribution], axis=1)
    # Change Column names
    df_dominant_topics.columns = ['Dominant_Topic', 'Topic_Keywords', 'Num_Documents',
                                  'Perc_Documents']

    # Show
    # df_dominant_topics.to_csv('topic_distribution.csv')


if __name__ == '__main__':
    lda()
