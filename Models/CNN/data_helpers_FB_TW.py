import numpy as np
import re
import itertools
from collections import Counter
import pickle
from nltk.corpus import stopwords

"""
Original taken from https://github.com/dennybritz/cnn-text-classification-tf
This script assumes that we have a fixed train and a fixed test set.
The data loading method assumes that there are labels available for the fixed test set (for evaluation).
"""


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r'\|LBR\|', '', string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"'", " ' ", string)
    #string = re.sub(r"[^A-Za-z0-9(),!?èéàòùì\'\`]", " ", string)
    pattern = re.compile(r'\b(' + r'|'.join(stopwords.words('italian')) + r')\b\s*')
    string = pattern.sub('', string)
    return string.strip().lower()


def load_data_and_labels():
    """
    Loading both the train and the  test set.
    Adding the espresso dataset to train
    """
    # Load train data from files
    samples, labels = [],[]
    with open('/home/p281734/projects/evalita2018-rug/Data/haspeede_FB-train.tsv','r', encoding='utf-8') as fi:
        for line in fi:
            data = line.strip().split('\t')
            # get sample
            samples.append(data[1])
            # get label
            if data[2] == '1':
                labels.append([0,1]) # label of positive sample
            elif data[2] == '0':
                labels.append([1,0]) # label of negative sample
            else:
                raise ValueError('Unknown label!')

    # Adding Espresso data
#    samples, labels, idx_espresso = add_espresso_data(samples, labels)

    # Clean and split samples
    Xtrain = [clean_str(sample) for sample in samples]
    Xtrain = [s.split(" ") for s in Xtrain] # each sample as list of words/strings
    Ytrain = np.array(labels)
    len_train = len(Xtrain)
    # We need to remember the len of train, we will put train + test together to build the vocab. Then we will recognise the first len_train items as coming from the train set

    # Load test data, no labels
    Xtest = []
    test_sentences = []
    test_ids = []
    with open('/home/p281734/projects/evalita2018-rug/Data/Test/haspeede_TW-test.tsv','r', encoding='utf-8') as fi:
        for line in fi:
            data = line.strip().split('\t')
            # get sample
            Xtest.append(data[1])
            test_sentences.append(data[1])
            test_ids.append(data[0])

    Xtest = [clean_str(sample) for sample in Xtest]
    Xtest = [s.split(" ") for s in Xtest] # each sample as list of words/strings
    # Ytest = np.array(Ytest)

    return [Xtrain, Ytrain, Xtest, len_train, test_sentences, test_ids]


# def load_data_and_labels_test():
#     """
#     Copy of load_data_and_labels to be applied to separate set of test data
#     """
#     # Load data from files
#     samples, labels = [],[]
#     with open('../../Data/germeval.ensemble.test.txt','r', encoding='utf-8') as fi:
#         for line in fi:
#             data = line.strip().split('\t')
#             # get sample
#             samples.append(data[0])
#             # get label
#             if data[1] == 'OFFENSE':
#                 labels.append([0,1]) # label of positive sample
#             elif data[1] == 'OTHER':
#                 labels.append([1,0]) # label of negative sample
#             else:
#                 raise ValueError('Unknown label!')
#
#     # Clean and split samples
#     x_text = [clean_str(sample) for sample in samples]
#     x_text = [s.split(" ") for s in x_text] # each sample as list of words/strings
#     # Turn labels to np array
#     y = np.array(labels)
#
#     return [x_text, y]


def pad_sentences(sentences, padding_word="<PAD/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    sequence_length = max(len(x) for x in sentences)
    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)
        new_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(new_sentence)
    return padded_sentences


def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]


def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]


def load_data():
    """
    Loads and preprocessed data.
    Returns input vectors, labels, vocabulary, and inverse vocabulary.
    """
    # Load and preprocess data
    #Xtrain, Ytrain, Xtest, len_train, idx_espresso = load_data_and_labels()
    Xtrain, Ytrain, Xtest, len_train, test_text, test_ids = load_data_and_labels()
    # sentences, labels, idx_espresso = load_data_and_labels()
    # Vocab needs to be build on the basis of the whole dataset, so we put train and test together! TRAIN, then TEST
    X = Xtrain + Xtest # X is list while Y is np.array
    Y = Ytrain

    sentences_padded = pad_sentences(X)
    vocabulary, vocabulary_inv = build_vocab(sentences_padded)
    X, Y = build_input_data(sentences_padded, Y, vocabulary)
#    return [X, Y, vocabulary, vocabulary_inv, len_train, idx_espresso]
    return [X, Y, vocabulary, vocabulary_inv, len_train, test_text, test_ids]


def batch_iter(data, batch_size, num_epochs):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
    yield shuffled_data[start_index:end_index]
