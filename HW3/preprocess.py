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
                        np.repeat(words2index['<s>'], N-1), [words2index[x] for x in lsplit[1:]])
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
    with open('data/train.txt') as f:
        for line in f:
            lsplit = [words2index[x] for x in line.split()]
            l = np.append(np.repeat(words2index['<s>'], N-1), lsplit)

            for i in range(len(lsplit)):
                g = l[i:N-1+i]
                v = lsplit[i]
                results.append((g, v))
        results.append((l[-N+1:], words2index['</s>']))
    return results

def tomatrix(results, train=True):

    N = len(results[0][0])+1

    if train:
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
        tooutput = np.empty((len(results), 50+N-1))

        for i in range(len(results)):
            tooutput[i, :] = np.hstack((results[i][1], results[i][0]))

        return tooutput.astype(int)


FILE_PATHS = ("data/train.txt",
              "data/valid_blanks.txt",
              "data/test_blanks.txt",
              "data/words.dict")
args = {}


def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--N', default=3, type=int, help='Ngram size')
    args = parser.parse_args(arguments)
    N = args.N
    train, valid, test, word_dict = FILE_PATHS

    words2index = get_words2index(word_dict)
    index2words = get_index2words(word_dict)

    train_list = train_get_ngram(train, words2index, N)
    print(len(train_list))
    train_matrix = tomatrix(train_list)

    valid_list = valid_test_Ngram(valid, words2index, N)
    valid_matrix = tomatrix(valid_list, False)

    test_list = valid_test_Ngram(test, words2index, N, True)
    test_matrix = tomatrix(test_list, False)

    filename = str(N) + '-grams.hdf5'
    with h5py.File(filename, "w") as f:
        f['train'] = train_matrix
        f['valid'] = valid_matrix
        f['test'] = test_matrix

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
