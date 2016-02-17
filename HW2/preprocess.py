#!/usr/bin/env python

"""Part-Of-Speech Preprocessing
"""

import numpy as np
import h5py
import argparse
import sys
import re
import codecs

# Your preprocessing, features construction, and word2vec code.


def get_tags2index(filename):
    '''
    Loading the tags to index mapping
    '''
    tags2index = {}
    with open(filename) as f:
        for line in f:
            (key, val) = line.split()
            tags2index[key] = int(val)
    return tags2index
tags2index = get_tags2index('data/tags.dict')


def get_words2index(filename):
    '''
    Loading the dictionary (words2index mapping)
    '''
    words2index = {'PADDING': 0, 'RARE': 1}
    with open(filename) as f:
        for i, line in enumerate(f):
            # Restricing to the first 100 000 words
            if i == 100000:
                break
            vect = line.split()
            # Shifting of two for padding
            words2index[vect[0]] = i + 2
    return words2index


def get_word_embeddings(filename, line_count, dimension):
    word_embeddings = np.zeros((line_count+2, dimension - 1))
    word_embeddings[0, :] = 2 * np.random.random(dimension - 1) - 1
    word_embeddings[1, :] = 2 * np.random.random(dimension - 1) - 1
    with open(filename) as f:
        for i, line in enumerate(f):
            vector = line.split()
            word_embeddings[i+2, :] = vector[1:]
    return word_embeddings


def pre_process(word, words2index):
    '''
    Function to pre-process the words
    Return (feature_1, feature_2)
    '''
    # Removing number if present
    word = re.sub("\d", "", word)
    # Case if only digits
    if not len(word):
        word = 'NUMBER'
    # Building feature 1
    word_lower = word.lower()
    if word_lower in words2index:
        feature1 = words2index[word_lower]
    else:
        word = 'RARE'
        feature1 = 1
    # Building feature 2
    if word.islower() or re.search('[.?\-",]+', word):
        feature2 = 0
    elif word.isupper():
        feature2 = 1
    elif word[0].isupper():
        feature2 = 2
    else:
        feature2 = 3
    return feature1, feature2


def get_number_elements(filename):
    '''
    Counting number of elements and dimension of element in a file
    '''
    line_count = 0
    dimension = 0
    with open(filename) as f:
        for line in f:
            sp = line.split()
            if sp:
                if not dimension:
                    dimension = len(sp)
                line_count += 1
    return line_count, dimension


def build_processed_input(filename, line_count, words2index, tags2index, test=False):
    '''
    Step 1: build the array (id_in_sentence, word_feature, cap_feature) and
    the output array
    '''
    output = np.zeros(line_count, dtype=int)
    # Contains: id_in_sentence, word_feature, cap_feature
    processed_input = np.zeros((line_count, 3), dtype=int)
    i = 0
    with open(filename) as f:
        for line in f:
            sp = line.split()
            # Check if blanck
            if sp:
                idword, id_in_sentence, word, tag = sp
                word_feature, cap_feature = pre_process(word, words2index)
                if not test:
                    output[i] = tags2index[tag]
                processed_input[i, :] = [id_in_sentence, word_feature, cap_feature]
                i += 1
    return processed_input, output


def build_feature_array(filename, line_count, processed_input):
    '''
    Step 2: building the two arrays for word_feature and cap_feature using
    window of dim 5 and the output vector
    '''
    # Initialization
    input_word = np.zeros((line_count, 5), dtype=int)
    input_cap = np.zeros((line_count, 5), dtype=int)

    for i in xrange(line_count - 2):
        # Last element of the window
        id_in_sentence_cur, feature1_cur, feature2_cur = tuple(
            processed_input[i, :])
        id_in_sentence_next1, feature1_next1, feature2_next1 = tuple(
            processed_input[i+1, :])
        id_in_sentence_next2, feature1_next2, feature2_next2 = tuple(
            processed_input[i+2, :])
        # Case current word is the first one of a sentence
        if id_in_sentence_cur == 1:
            input_word[i, :2] = 0
            input_cap[i, :2] = 1
            input_word[i, 2] = feature1_cur
            input_cap[i, 2] = feature2_cur
            input_word[i, 3] = feature1_next1
            input_cap[i, 3] = feature2_next1
            input_word[i, 4] = feature1_next2
            input_cap[i, 4] = feature2_next2
        else:
            input_word[i, :4] = input_word[i-1, 1:5]
            input_cap[i, :4] = input_cap[i-1, 1:5]
            # Case current word is within one position to the last one of a
            # sentence
            if id_in_sentence_next2 == 1:
                input_word[i, 4] = 0
                input_cap[i, 4] = 1
            # Case current word is the last one of a sentence
            elif id_in_sentence_next1 == 1:
                input_word[i, 3] = 0
                input_cap[i, 3] = 1
                input_word[i, 4] = 0
                input_cap[i, 4] = 1
            else:
                input_word[i, 4] = feature1_next2
                input_cap[i, 4] = feature2_next2
    # Corner Case: two last rows
    i = line_count - 2
    # Case one to last word at a beginning of a sentence
    id_in_sentence_last1, feature1_last1, feature2_last1 = tuple(
        processed_input[i + 1, :])
    id_in_sentence_last2, feature1_last2, feature2_last2 = tuple(
        processed_input[i, :])
    if id_in_sentence_last2 == 1:
        input_word[i, :2] = 0
        input_cap[i, :2] = 1
        input_word[i, 2] = feature1_last2
        input_cap[i, 2] = feature2_last2
        input_word[i, 3] = feature1_last1
        input_cap[i, 3] = feature2_last1
        input_word[i, 4] = 0
        input_cap[i, 4] = 1
    else:
        input_word[i, :4] = input_word[i-1, 1:5]
        input_cap[i, :4] = input_cap[i-1, 1:5]
        input_word[i, 4] = 0
        input_cap[i, 4] = 1
    # Last word case
    input_word[i+1, :4] = input_word[i, 1:5]
    input_cap[i+1, :4] = input_cap[i, 1:5]
    input_word[i+1, 4] = 0
    input_cap[i+1, 4] = 1

    return input_cap, input_word


FILE_PATHS = {"PTB": ("data/train.tags.txt",
                      "data/dev.tags.txt",
                      "data/test.tags.txt",
                      "data/tags.dict",
                      "data/glove.6B.50d.txt")}
args = {}


def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('dataset', help="Data set",
                        type=str)
    args = parser.parse_args(arguments)
    dataset = args.dataset
    train, valid, test, tag_dict, embedding = FILE_PATHS[dataset]

    tags2index = get_tags2index(tag_dict)
    print 'tags2index size', len(tags2index)
    C = len(tags2index)
    words2index = get_words2index(embedding)
    print 'words2index size', len(words2index)
    line_count_dict, dimension_dict = get_number_elements(embedding)
    word_embeddings = get_word_embeddings(embedding, line_count_dict,
                                          dimension_dict)

    input_features = {}
    for name, filename in zip(['train', 'valid', 'test'], [train, valid, test]):
        if name == 'test':
            test_bool = True
        else:
            test_bool = False
        line_count, dimension = get_number_elements(filename)
        processed_input, output = build_processed_input(filename, line_count,
                                                        words2index,
                                                        tags2index,
                                                        test=test_bool)
        input_cap, input_word = build_feature_array(filename, line_count,
                                                    processed_input)
        input_features[name] = input_word, input_cap, output

    filename = args.dataset + '.hdf5'
    with h5py.File(filename, "w") as f:
        for name in ['train', 'valid', 'test']:
            f['{}_input_word_windows'.format(name)] = input_features[name][0]
            f['{}_input_cap_windows'.format(name)] = input_features[name][1]
            if name != 'test':
                f['{}_output'.format(name)] = input_features[name][2]
        f['nwords'] = np.array([line_count_dict], dtype=np.int32)
        f['nclasses'] = np.array([C], dtype=np.int32)
        f['word_embeddings'] = word_embeddings


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
