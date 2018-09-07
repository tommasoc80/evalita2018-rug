'''
This script implements an ensemble classifer for GermEval 2018.
The lower-level classifiers are SVM and CNN
the meta-level classifer is optionally a LinearSVC or a Logistic Regressor

Predictions outputted by SVM and CNN need to be obtained beforehand, stored as pickle, and loaded
'''

import argparse
import re
import statistics as stats
import stop_words
import features_ensamble
import json
import pickle
from scipy.sparse import hstack, csr_matrix

# from features import get_embeddings
from sklearn.base import TransformerMixin
from sklearn.model_selection import cross_val_predict, cross_validate, train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion

from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support

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
                print("Your are doing another task! Check the data!!")
#                # 4-class problem: OTHER, PROFANITY, INSULT, ABUSE
#                labels.append(data[2])

    return tweets, labels



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

    '''
    PART: TRAINING META-CLASSIFIER
    '''

    #evalita_train = '../../Data/haspeede_FB-train.tsv' # FB Train
    evalita_test = '../../Data/Test/haspeede_FB-test.tsv' # FB Test

    evalita_train = '../../Data/haspeede_TW-train-ensamble.tsv' # TW Train
    #evalita_test = '../../Data/Test/haspeede_TW-test.tsv' # TW Test



    # load training data of ensemble classifier
    Xtrain, Ytrain = read_corpus(evalita_train)
    assert len(Xtrain) == len(Ytrain), 'Unequal length for Xtrain and Ytrain!'
    print('{} training samples'.format(len(Xtrain)))

    # Set up vectorizer to get the SentLen and Lexicon look-up information
    # Meta classifier uses as features the predictions of the two lower-level classifiers + SentLen + Lexicon
    print('Setting up meta_vectorizer...')
    meta_vectorizer = FeatureUnion([('length', features_ensamble.TweetLength()),
                                    ('badwords', features_ensamble.Lexicon('lexicon_DeMauro.txt'))])
    X_feats = meta_vectorizer.fit_transform(Xtrain)

    # load in predictions for training data by 1) svm and 2) cnn
    # Predictions already saved as scipy sparse matrices
    print('Loading SVM and CNN predictions on train...')
    #f1 = open('NEW-train-svm-predict-FB-ensamble.p', 'rb') #FB SVM prediction train
    f1 = open('NEW-train-svm-predict-TW-ensamble.p', 'rb') #TW SVM prediction Train
    SVM_train_predict = pickle.load(f1)
    print(SVM_train_predict)
    f1.close()
    #f2 = open('NEW-train-cnn-predict-FB-ensamble.p', 'rb') # FB CNN prediction train
    f2 = open('NEW-train-cnn-predict-TW-ensamble.p', 'rb') # TW CNN prediction train
    CNN_train_predict = pickle.load(f2)
    print(CNN_train_predict)
    f2.close()



    # Combine all features to input to ensemble classifier
    print('Stacking all features...')
    Xtrain_feats = hstack((X_feats, SVM_train_predict, CNN_train_predict))
    print(type(Xtrain_feats))
    print('Shape of featurized Xtrain:', Xtrain_feats.shape)

    # Set-up meta classifier
    # meta_clf = Pipeline([('clf', LinearSVC(random_state=0))]) # LinearSVC
    meta_clf = Pipeline([('clf', LogisticRegression(random_state=0))]) # Logistic Regressor

    # Fit it
    print('Fitting meta-classifier...')
    meta_clf.fit(Xtrain_feats, Ytrain)


    '''
    PART: TESTING META-CLASSIFIER
    '''

    # load real test data of ensemble classifier without labels
    print('Reading in Test data...')
    Xtest, test_ids = [],[]
    with open(evalita_test, 'r', encoding='utf-8') as fi:
        for line in fi:
            data = line.strip().split("\t")
            Xtest.append(data[1])
            test_ids.append(data[0])


    #Xtest, Ytest = read_corpus(evalita_test)
    #assert len(Xtest) == len(Ytest), 'Unequal length for Xtest and Ytest!'
    print('{} test samples'.format(len(Xtest)))

    Xtest_feats = meta_vectorizer.transform(Xtest)
    #print('Shape of featurized Xtest:', Xtest_feats.shape)


    # Loading predictions of SVM and CNN on test data
    print('Loading SVM and CNN predictions on test...')
    #ft1 = open('TEST-FB-FB-svm-ensamble.p', 'rb') # FB prediction Test - SVM
    #ft1 = open('TEST-TW-TW-svm-ensamble.p', 'rb') # TW prediction Test - SVM
    #ft1 = open('TEST-FB-TW-svm-ensamble.p', 'rb') # TW prediction Test - SVM
    ft1 = open('TEST-TW-FB-svm-ensamble.p', 'rb') # TW prediction Test - SVM
    SVM_test_predict = pickle.load(ft1)
    print(SVM_test_predict)
    ft1.close()
    #ft2 = open('TEST-FB-FB-cnn.p', 'rb') # FB prediction Test - CNN
    #ft2 = open('TEST-TW-TW-cnn.p', 'rb') # TW prediction Test - CNN
    #ft2 = open('TEST-FB-TW-cnn.p', 'rb') # TW prediction Test - CNN
    ft2 = open('TEST-TW-FB-cnn.p', 'rb') # TW prediction Test - CNN
    CNN_test_predict = pickle.load(ft2)
    ft2.close()

    # Combine all features for test input to input to ensemble classifier
    print('Stacking all features...')
    Xtest_feats = hstack((Xtest_feats, SVM_test_predict, CNN_test_predict))
    print('Shape of featurized Xtest:', Xtest_feats.shape)

    # Use trained meta-classifier to get predictions on test set
    Yguess = meta_clf.predict(Xtest_feats)    # assert len(Xtest) == len(Yguess), 'Yguess not the same length as Xtest!'
    print(len(Yguess), 'predictions in total')

    ## Evaluate
    #evaluate(Ytest, Yguess)


    #Outputting in format required


    print('Outputting predictions...')

    outdir = '../../Results'
    #fname = 'evalita_rug_ensemble_TW-TW.txt'
    #fname = 'evalita_rug_ensemble_FB-TW.txt'
    fname = 'evalita_rug_ensemble_TW-FB.txt'


    with open(outdir + '/' + fname, 'w', encoding='utf-8') as fo:
        assert len(Yguess) == len(Xtest), 'Unequal length between samples and predictions!'
        for idx in range(len(Yguess)):
            print(test_ids[idx] + "\t" + Xtest[idx] + '\t' + Yguess[idx], file=fo) # binary task (coarse)
            # print(Xtest_raw[idx] + '\t' + 'XXX' + '\t' + Yguess[idx], file=fo) # multi task (fine)

    print('Done.')



    '''
    First Results:

    Meta classifier = LinearSVC
    --------------------------------------------------
    Accuracy: 0.7455089820359282
    --------------------------------------------------
    Precision, recall and F-score per class:
                Precision     Recall    F-score
    OFFENSE      0.726829   0.428161   0.538879
    OTHER        0.750314   0.914373   0.824259
    --------------------------------------------------
    Average (macro) F-score: 0.6815689871548336
    --------------------------------------------------
    Confusion matrix:
    Labels: ['OFFENSE', 'OTHER']
    [[149 199]
     [ 56 598]]



    Meta classifier = Logistic Regression
    --------------------------------------------------
    Accuracy: 0.7574850299401198
    --------------------------------------------------
    Precision, recall and F-score per class:
                Precision     Recall    F-score
    OFFENSE      0.741935   0.462644   0.569912
    OTHER        0.761783   0.914373   0.831133
    --------------------------------------------------
    Average (macro) F-score: 0.7005221177440085
    --------------------------------------------------
    Confusion matrix:
    Labels: ['OFFENSE', 'OTHER']
    [[161 187]
     [ 56 598]]


    New RES:
    With Espresso data!

    LogisticRegression
    --------------------------------------------------
    Accuracy: 0.7514970059880239
    --------------------------------------------------
    Precision, recall and F-score per class:
                Precision     Recall    F-score
    OFFENSE      0.710638   0.479885   0.572899
    OTHER        0.764016   0.896024   0.824771
    --------------------------------------------------
    Average (macro) F-score: 0.6988350435696844
    --------------------------------------------------
    Confusion matrix:
    Labels: ['OFFENSE', 'OTHER']
    [[167 181]
     [ 68 586]]



    '''
