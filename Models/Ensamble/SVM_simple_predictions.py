'''
This script is to be used for the ensemble system to get SVM predictions.
This SVM needs 2 datasets, train and test, training itself on the trainset and outputting predictions for each X in testset
Predictions stored in pickle
'''
import argparse
import re
import statistics as stats
import stop_words
import json
import pickle
import gensim.models as gm
import numpy as np
from scipy.sparse import hstack, csr_matrix

import features
from sklearn.model_selection import KFold, cross_validate, cross_val_predict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, FeatureUnion


def read_corpus(corpus_file, binary=True):
    '''Reading in data from corpus file'''

    tweets = []
    labels = []
    with open(corpus_file, 'r', encoding='utf-8') as fi:
        for line in fi:
            data = line.strip().split('\t')
            # making sure no missing labels
            if len(data) != 3:
                raise IndexError('Missing data for tweet "%s"' % data[0])

            tweets.append(data[1])

            if binary:
                # 2-class problem: OTHER vs. OFFENSE
                labels.append(data[2])
            else:
#                # 4-class problem: OTHER, PROFANITY, INSULT, ABUSE
#                labels.append(data[2])
                print("This is another task!")

    return tweets, labels


def read_test(path_to_file):
    '''Reading in the real test data, with no labels'''

    Xtest = []
    with open(path_to_file, 'r', encoding='utf-8') as fi:
        for line in fi:
            line_stripped = line.strip()
            line_splitted = line_stripped.split("\t")
            input_data = line_splitted[1]
            if input_data != '':
                Xtest.append(input_data)
    return Xtest


def load_embeddings(embedding_file):
    '''
    loading embeddings from file
    input: embeddings stored as json (json), pickle (pickle or p) or gensim model (bin)
    output: embeddings in a dict-like structure available for look-up, vocab covered by the embeddings as a set
    '''
    if embedding_file.endswith('json'):
        f = open(embedding_file, 'r', encoding='utf-8')
        embeds = json.load(f)
        f.close
        vocab = {k for k,v in embeds.items()}
    elif embedding_file.endswith('bin'):
        embeds = gm.KeyedVectors.load(embedding_file).wv
        vocab = {word for word in embeds.index2word}
    elif embedding_file.endswith('p') or embedding_file.endswith('pickle'):
        f = open(embedding_file,'rb')
        embeds = pickle.load(f)
        f.close
        vocab = {k for k,v in embeds.items()}

    return embeds, vocab


if __name__ == '__main__':

    print('Reading in train and test data...')

    #Xtrain, Ytrain = read_corpus('../../Data/haspeede_FB-train.tsv') # FB TRAIN data
    Xtrain, Ytrain = read_corpus('../../Data/haspeede_TW-train.tsv') # TW TRAIN data
    assert len(Xtrain) == len(Ytrain), 'Unequal length for Xtrain and Ytrain!'
    print('{} train samples'.format(len(Xtrain)))


    Xtest = read_test('../../Data/Test/haspeede_FB-test.tsv') # FB TEST data
    #Xtest = read_test('../../Data/Test/haspeede_TW-test.tsv') # TW TEST data
#    assert len(Xtest) == len(Ytest), 'Unequal length for Xtest and Ytest!'
    print('{} test samples'.format(len(Xtest)))


    # Vectorizing data / Extracting feature
    # unweighted word uni and bigrams
    count_word = CountVectorizer(ngram_range=(1,2), stop_words=stop_words.get_stop_words('it'))
    count_char = CountVectorizer(analyzer='char', ngram_range=(3,7))

    # Getting hate embeddings
    path_to_embs = '../../Data/embeddings/model_hate_300.bin'
    print('Getting pretrained word embeddings from {}...'.format(path_to_embs))
    embeddings, _ = load_embeddings(path_to_embs)
    print('Done')

    vectorizer = FeatureUnion([('word', count_word),
                                ('char', count_char),
                                ('word_embeds', features.Embeddings(embeddings, pool='max'))])

    classifier = Pipeline([
                            ('vectorize', vectorizer),
                            ('classify', SVC(kernel='linear', probability=True))
    ])

    print('Fitting model...')
    classifier.fit(Xtrain, Ytrain)

    print('Predicting...')
    Yguess = classifier.predict_proba(Xtest)

    print('Turning to scipy:')
    Ysvm = csr_matrix(Yguess)
    print(type(Ysvm))
    print(Ysvm.shape)

    # Pickling the predictions
    #save_to = open('TEST-FB-FB-svm-ensamble.p', 'wb')
    #save_to = open('TEST-TW-TW-svm-ensamble.p', 'wb')
    #save_to = open('TEST-FB-TW-svm-ensamble.p', 'wb')
    save_to = open('TEST-TW-FB-svm-ensamble.p', 'wb')
    pickle.dump(Ysvm, save_to)
    save_to.close()
