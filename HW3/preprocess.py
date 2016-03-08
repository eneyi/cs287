#!/usr/bin/env python

"""Language modeling preprocessing
"""

import numpy as np
import h5py
import argparse
import sys
import re
import codecs
from collections import Counter

# Your preprocessing, features construction, and word2vec code.


def get_words2index(filename):
    '''
    Loading the tags to index mapping
    '''
    words2index = {}
    with open(filename) as f:
        for line in f:
            (val, key, num) = line.split()
            words2index[key] = int(val)
    return words2index


def get_index2words(filename):
    '''
    Loading the tags to index mapping
    '''
    index2words = {}
    with open(filename) as f:
        for line in f:
            (val, key, num) = line.split()
            index2words[int(val)] = key
    return index2words

index2words = get_index2words('data/words.dict')
index2words1000 = get_index2words('data/words.1000.dict')


def valid_test_Ngram(filepath, words2index, N, test=False):
    results = []
    if test == False:
        with open(filepath) as f:
            i = 1
            for line in f:
                lsplit = line.split()
                if lsplit[0] == 'Q':
                    topredict = np.array([words2index[x] for x in lsplit[1:]])
                if lsplit[0] == 'C':
                    l = np.append(
                        np.repeat(words2index['<s>'], N-1), [words2index[x] for x in lsplit[1:-1]])
                    lastNgram = l[-N+1:]
                    results.append((lastNgram, topredict))
    else:
        with open(filepath) as f:
            i = 1
            for line in f:
                lsplit = line.split()
                if lsplit[0] == 'Q':
                    topredict = np.array([words2index[x] for x in lsplit[1:]])
                if lsplit[0] == 'C':
                    l = np.append(
                        np.repeat(words2index['<s>'], N-1), [words2index[x] for x in lsplit[1:-1]])
                    lastNgram = l[-N+1:]
                    results.append((lastNgram, topredict))
    return results


def train_get_ngram(filename, words2index, N):
    '''
    Generating N-grams
    '''
    results = []
    with open(filename) as f:
        for line in f:
            lsplit = [words2index[x] for x in line.split()]
            l = np.append(np.repeat(words2index['<s>'], N-1), lsplit)

            for i in range(len(lsplit)):
                g = l[i:N-1+i]
                v = lsplit[i]
                results.append((g, v))
        results.append((l[-N+1:], words2index['</s>']))
    return results

def tomatrix(results, train=True, count = True):

    N = len(results[0][0])+1

    if train:
        if count:
            tuplelist = []
            for i in range(len(results)):
                tuplelist.append(
                    tuple(list(np.append(results[i][0], results[i][1]))))
            Count = Counter(tuplelist).most_common()
            tooutput = np.empty((len(Count),  N+1))

            for i in range(len(Count)):
                tooutput[i, :] = np.append(np.array(Count[i][0]), Count[i][1])

            return tooutput.astype(int)

        else:
            tooutput_ = np.empty((len(results),N))
            for i in range(len(results)):
                tooutput_[i,:] = np.append(results[i][0],results[i][1])
            return tooutput_

    else:
        tooutput = np.empty((len(results), 50+N-1))

        for i in range(len(results)):
            tooutput[i, :] = np.hstack((results[i][1], results[i][0]))

        return tooutput.astype(int)

def validation_kaggle(filepath):
    it = 0
    results = []
    with open(filepath) as f:
        for line in f:
            if it == 0 :
                it+=1
            else:
                lsplit = line.split(',')
                l = [int(x.rstrip()) for x in lsplit[1:]]
                results.append(l)
    return np.array(results)


def get_prior(filepath, words2index):
    '''
    Case N=1: ie prior on the word distribution from the train text
    '''
    counter = Counter()
    with open(filepath) as f:
        lines = f.readlines()
        for line in lines:
            # Adding the end of line prediction
            lsplit = line.split() + ['</s>']
            counter.update(lsplit[1:])
    # Build count matrix: (N_words, 2), col 1: word index, col2: word cout
    count_matrix = np.zeros((len(counter), 2), dtype=int)
    
    for i,t in enumerate(counter.iteritems()):
        k, v = t
        count_matrix[i, 0] = words2index[k]
        count_matrix[i, 1] = v
    return count_matrix


FILE_PATHS = ("data/train.txt",
			  "data/train.1000.txt",
              "data/valid_blanks.txt",
              "data/test_blanks.txt",
              "data/words.dict",
              "data/words.1000.dict",
              "data/valid_kaggle.txt")
args = {}


def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--N', default=3, type=int, help='Ngram size')
    args = parser.parse_args(arguments)
    N = args.N
    train, train1000, valid, test, word_dict, word_dict_1000, kaggle = FILE_PATHS

    words2index = get_words2index(word_dict)
    words2index1000 = get_words2index(word_dict_1000)
    index2words = get_index2words(word_dict)

    train_list = train_get_ngram(train, words2index, N)
    train_matrix_count = tomatrix(train_list)
    train_matrix = tomatrix(train_list,True,False)

    train_list_1000 = train_get_ngram(train1000, words2index1000, N)
    train_matrix_1000_count = tomatrix(train_list_1000,True,False)
    train_matrix_1000 = tomatrix(train_list_1000)

    valid_list = valid_test_Ngram(valid, words2index, N)
    valid_matrix = tomatrix(valid_list, False)

    test_list = valid_test_Ngram(test, words2index, N, True)
    test_matrix = tomatrix(test_list, False)

    valid_kaggle = validation_kaggle(kaggle)

    filename = str(N) + '-grams.hdf5'
    with h5py.File(filename, "w") as f:

        f['train'] = train_matrix_count
        f['train_1000_nocounts'] = train_matrix_1000
        f['train_1000'] = train_matrix_1000_count
        f['train_nocounts'] = train_matrix
        f['valid'] = valid_matrix
        f['valid_output'] = valid_kaggle
        f['test'] = test_matrix
        f['nwords'] = np.array([np.max(index2words.keys())])

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
