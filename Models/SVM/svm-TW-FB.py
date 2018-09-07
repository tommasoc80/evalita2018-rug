'''
SVM systems for EVALITA 2018
'''
import argparse
import re
import statistics as stats
import json
import pickle
import gensim.models as gm

import features
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_validate, cross_val_predict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC, SVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.utils import shuffle

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import cross_val_score

from nltk.corpus import stopwords

from scipy.sparse import hstack, csr_matrix



def read_corpus(TW_train):
    '''Reading in data from corpus file'''


    # Load train data from the FB file
    samples, labels = [],[]

    # Load train data from the TW file
    with open(TW_train,'r', encoding='utf-8') as fi:
        for line in fi:
            data = line.strip().split('\t')
            # get sample
            samples.append(data[1])
            # get labels
            labels.append(data[2])

    return samples, labels




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
    elif embedding_file.endswith('txt'):
        embeds = KeyedVectors.load_word2vec_format(embedding_file, binary=False)
        vocab = embeds.wv.vocab

    return embeds, vocab



def clean_samples(samples):
    '''
    Simple cleaning: removing URLs, line breaks, abstracting away from user names etc.
    '''

    new_samples = []
    for string in samples:
        string = re.sub(r'\|LBR\|', '', string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\s{2,}", " ", string)
        string = re.sub(r"'", " ' ", string)
        #string = re.sub(r"[^A-Za-z0-9(),!?èéàòùì\'\`]", " ", string) # potremmo commentarlo
        pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('italian')) + r')\b\s*')
        string = pattern.sub('', string)
        string = string.strip().lower()
        new_samples.append(string)

    return new_samples



def evaluate(Ygold, Yguess):
    '''Evaluating model performance and printing out scores in readable way'''

    print('-'*50)
    print("Accuracy:", accuracy_score(Ygold, Yguess))
    print('-'*50)
    print("Precision, recall and F-score per class:")

    # get all labels in sorted way
    # Ygold is a regular list while Yguess is a numpy array
    labs = sorted(set(Ygold + Yguess.tolist()))


    # printing out precision, recall, f-score for each class in easily readable way
    PRFS = precision_recall_fscore_support(Ygold, Yguess, labels=labs)
    print('{:10s} {:>10s} {:>10s} {:>10s}'.format("", "Precision", "Recall", "F-score"))
    for idx, label in enumerate(labs):
        print("{0:10s} {1:10f} {2:10f} {3:10f}".format(label, PRFS[0][idx],PRFS[1][idx],PRFS[2][idx]))

    print('-'*50)
    print("Average (macro) F-score:", stats.mean(PRFS[2]))
    print('-'*50)
    print('Confusion matrix:')
    print('Labels:', labs)
    print(confusion_matrix(Ygold, Yguess, labels=labs))
    print()



if __name__ == '__main__':

    TASK = 'binary'
    #TASK = 'multi'

    '''
    Preparing data
    '''

    TW_train = '/home/p281734/projects/evalita2018-rug/Data/haspeede_TW-train.tsv'
#    espresso_train = '/home/p281734/projects/evalita2018-rug/Data/espresso-ita-hate.p'

    evalita_test = '/home/p281734/projects/evalita2018-rug/Data/Test/haspeede_FB-test.tsv'

    print('Reading in EVALITA training data...')
    Xtrain,Ytrain = read_corpus(TW_train)

    print('Reading in Test data...')

    Xtest_raw , Xtest_ids = [], []
    with open(evalita_test, 'r', encoding='utf-8') as fi:
        for line in fi:
            data = line.strip().split("\t")
            Xtest_raw.append(data[1])
            Xtest_ids.append(data[0])


    # Minimal preprocessing / cleaning
    Xtrain = clean_samples(Xtrain)
    Xtest = clean_samples(Xtest_raw)

    print(len(Xtrain), 'training samples!')
    print(len(Xtest), 'test samples!')


    '''
    Preparing vectorizer and classifier
    '''

    # Vectorizing data / Extracting features
    print('Preparing tools (vectorizer, classifier) ...')

    # n-grams
    count_word = TfidfVectorizer(analyzer='word',
                             ngram_range=(1,3),
                             binary=False,
                             sublinear_tf=False
                             )
    count_char = TfidfVectorizer(analyzer='char',
                             ngram_range=(2,4),
                             binary=False,
                             sublinear_tf=False
                             )


    # Getting embeddings
    #path_to_embs = '/media/flavio/1554-26B0/THESIS EXPERIMENTS/CNN/Embeddings/model_hate_300.bin'
    # path_to_embs = 'embeddings/model_reset_random.bin'
    path_to_embs = '/project/tcaselli/Documents/evalita2018-rug/Data/embeddings/model_hate_300.bin'
    print('Getting pretrained word embeddings from {}...'.format(path_to_embs))
    embeddings, vocab = load_embeddings(path_to_embs)
    print('Done')

    vectorizer = FeatureUnion([('word', count_word),
                                ('char', count_char),
                                ('word_embeds', features.Embeddings(embeddings, pool='pool'))])


    clf = LinearSVC()

    classifier = Pipeline([
                     ('vectorize', vectorizer),
                     ('classify', clf)])


    """
    10-cv on training
    """
    scores = cross_val_score(classifier, Xtrain, Ytrain, cv=10, scoring='f1_macro')

    with open("cross-fold_TW-FB.txt", 'w', encoding='utf-8') as fo:
            print(str(scores), file=fo)


    '''
    Actual training and predicting:
    '''

    print('Fitting on training data...')
    classifier.fit(Xtrain, Ytrain)
    print('Predicting...')
    Yguess = classifier.predict(Xtest)


    '''
    Outputting in format required
    '''

    print('Outputting predictions...')

    outdir = '/project/tcaselli/Documents/evalita2018-rug/Results'
    fname = 'evalita_rug_TW-FB_svm.txt'

    with open(outdir + '/' + fname, 'w', encoding='utf-8') as fo:
        assert len(Yguess) == len(Xtest_raw), 'Unequal length between samples and predictions!'
        for idx in range(len(Yguess)):
            print(Xtest_ids[idx] + "\t" + Xtest_raw[idx] + '\t' + Yguess[idx], file=fo) # binary task (coarse)
           #print(Xtest_raw[idx] + '\t' + Yguess[idx] + '\t' + 'XXX', file=fo) # binary task (coarse)
            #print(Xtest_raw[idx] + '\t' + 'XXX' + '\t' + Yguess[idx], file=fo) # multi task (fine)

    print('Done.')

