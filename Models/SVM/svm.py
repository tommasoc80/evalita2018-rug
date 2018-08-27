'''
SVM systems for EVALITA
'''


import argparse
import re
import statistics as stats
import stop_words
import json
import random
import features
import csv
import gensim.models as gm
import pandas as pd



# from features import get_embeddings
from sklearn.base import TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import classification_report
from gensim.models import KeyedVectors
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from nltk.corpus import stopwords





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


df = pd.read_csv('mix.csv')
df = df.sample(frac=1, random_state=0)
X = df['message'].values.tolist()
Y = df['hate'].values.tolist()



# Minimal preprocessing: Removing line breaks
Data_X = []
for i in X:
    i = re.sub(r"[^A-Za-z0-9(),!?èéàòùì\'\`]", " ", i)
    i = re.sub(r",", " , ", i)
    i = re.sub(r"!", " ! ", i)
    i = re.sub(r"\(", " \( ", i)
    i = re.sub(r"\)", " \) ", i)
    i = re.sub(r"\s{2,}", " ", i)
    i = re.sub(r"'", " ' ", i)
    pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('italian')) + r')\b\s*')
    i = pattern.sub('', i)
    i = i.strip().lower()
    Data_X.append(i)
X = Data_X

'''
Preparing vectorizer + classifier to be used
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
path_to_embs = '/media/flavio/1554-26B0/THESIS EXPERIMENTS/CNN/Embeddings/model_hate_300.bin'
print('Getting pretrained word embeddings from {}...'.format(path_to_embs))
embeddings, vocab = load_embeddings(path_to_embs)
print('Done')


vectorizer = FeatureUnion([('word', count_word),
                           ('char', count_char),
                           ('word_embeds', features.Embeddings(embeddings, pool='pool'))
])


clf = LinearSVC()

classifier = Pipeline([
                     ('vectorize', vectorizer),
                     ('classify', clf)])



train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.20, random_state=42)


p_1 = classifier.fit(train_x, train_y)



# classification reports

print("LinearSVC:")
y_1 = p_1.predict(test_x)
print(classification_report(test_y, y_1))


print("Accuracy:", accuracy_score(test_y, y_1))
