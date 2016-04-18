import numpy as np
import h5py
import re
import pattern.en
import sys
import argparse

from itertools import product


def get_tag2index():
    # Tags mapping
    tag2index = {}

    with open('data/tags.txt', 'r') as f:
        for line in f:
            line_split = line[:-1].split(' ')
            tag2index[line_split[0]] = int(line_split[1])

    # Adding tags for end/start of sentence
    tag2index['<t>'] = 8
    tag2index['<\t>'] = 9
    return tag2index


def get_pos2index():
    '''
    Part of speech tagging tags to feature index mapping
    '''
    # mapping for the POS tags
    tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD',
            'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS',
            'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB',
            'VBZ', 'VBP', 'VBD', 'VBN', 'VBG', 'WDT', 'WP', 'WP$', 'WRB',
            '.', ',', ':', '(', ')']

    pos2index = {k: v+1 for v, k in enumerate(tags)}
    return pos2index


def count_elements(filename, tags=True):
    # Counting the number of elements to stored (ie num_words +
    # 2*num_sentences)
    num_words = 0
    num_sentences = 0
    with open(filename, 'r') as f:
        for line in f:
            if tags:
                line_split = line[:-1].split('\t')
            else:
                line_split = line[:-1].split(' ')
            # Case blank
            if len(line_split) == 1:
                num_sentences += 1
            else:
                num_words += 1

    return num_words, num_sentences


def get_cap_feature(word):
    # Return the caps feature for the given word
    # 1 - low caps; 2 - all caps; 3 - first cap; 4 - one cap; 5 - other
    if len(word) == 0 or word.islower() or re.search('[.?\-",]+', word):
        feature = 1
    elif word.isupper():
        feature = 2
    elif len(word) and word[0].isupper():
        feature = 3
    elif sum([w.isupper() for w in word]):
        feature = 4
    else:
        feature = 5
    return feature


def get_tokenized_sentences(filename, tags=True):
    # Build the part of speech tags
    with open(filename, 'r') as f:
        text = []
        for line in f:
            if tags:
                line_split = line[:-1].split('\t')
            else:
                line_split = line[:-1].split(' ')
            if len(line_split) != 1:
                text.append(line_split[2])

    return pattern.en.tag(' '.join(text))


def build_input_matrix(filename, num_rows, tag2index, pos2index, tags=True, word2index=None, memm = False):
    # Building input matrix with columns: (id, id_in_sentence, id_word, id_caps, id_token, id_tag)
    # caps feature:
    # 1 - low caps; 2 - all caps; 3 - first cap; 4 - one cap; 5 - other
    # Tags: if correct solution given (ie 4th column)
    # word2index: if use of previously built word2index mapping

    # Features for starting/ending of sentence (3 last columns)
    # For the POS tag, we use the same as a point (index 36)
    # initialization
    input_matrix = np.zeros((num_rows, 6), dtype=int)
  	if memm ==  False: 
    	input_matrix[0] = [1, 1, 1, 1, 36, 8]
    	start = [1, 1, 36, 8]
    	end = [2, 1, 36, 9]
    else:
    	input_matrix[0] = [1,1,word2index['<s>'],1,36,8]
    	start = [word2index['<s>'],1,36, 8]
   		end = [word2index['<\s>'],1,36, 9]
    row = 1

    # Get the POS tokken
    tokenized_sentences = get_tokenized_sentences(filename, tags=tags)
    pos_i = 0

    # Boolean to indicate if a sentence is starting
    starting = False
    # Boolean if a mapping is defined (last element of the mapping is for
    # unknown words)
    if word2index == None:
        test = False
        word2index = {'<s>': 1, '<\s>': 2}
        id_word = 3
    else:
        test = True
    with open(filename, 'r') as f:
        for line in f:
            if tags:
                line_split = line[:-1].split('\t')
            else:
                line_split = line[:-1].split(' ')
            if starting == True:
                # Start of sentence
                input_matrix[row, 0] = input_matrix[row-1, 0] + 1
                input_matrix[row, 1] = 1
                input_matrix[row, 2:] = start
                row += 1
                starting = False
            if len(line_split) == 1:
                # End of sentence
                input_matrix[row, :2] = input_matrix[row-1, :2] + 1
                input_matrix[row, 2:] = end
                row += 1
                starting = True
            else:
                # Indexing
                input_matrix[row, 0] = input_matrix[row-1, 0] + 1
                input_matrix[row, 1] = int(line_split[1]) + 1
                # Build cap feature
                word = line_split[2]
                input_matrix[row, 3] = get_cap_feature(word)
                # Build pos feature
                pos_tag = tokenized_sentences[pos_i][1].split('-')[0]
                if pos_tag in pos2index.keys():
                    input_matrix[row, 4] = pos2index[pos_tag]
                else:
                    input_matrix[row, 4] = len(pos2index) + 1
                pos_i += 1

                # Build word count feature
                word_clean = word.lower()
                if not test:
                    if word_clean not in word2index:
                        word2index[word_clean] = id_word
                        id_word += 1
                    input_matrix[row, 2] = word2index[word_clean]
                else:
                    # Unseen word during train
                    if word_clean not in word2index:
                        input_matrix[row, 2] = len(word2index)
                    else:
                        input_matrix[row, 2] = word2index[word_clean]
                if tags:
                    input_matrix[row, 5] = tag2index[line_split[3]]
                row += 1
    # Add special word if training
    if not test:
        word2index['<unk>'] = len(word2index)+1
    if tags:
        return input_matrix, word2index
    else:
        return input_matrix[:, :5], word2index

#Function that formats the output of the previous function in order to run MEMM:
def input_mm_pos(matrix):
    
    nwords = matrix.shape[0]
    
    res = np.zeros((nwords,1 + 9 + 5 + 43 + 1),dtype = int)
    
    res[:,0] = matrix[:,2]
    
    for i in range(nwords):
        tag_1_hot = np.zeros(9)
        tag_1_hot[matrix[i,5]-1] = 1
        tag_1_hot_cap = np.zeros(5)
        tag_1_hot_cap[matrix[i,3]-1] = 1
        tag_1_hot_pos = np.zeros(43)
        tag_1_hot_pos[matrix[i,4]] = 1
        res[i,1:10] = tag_1_hot
        res[i,10:15] = tag_1_hot_cap
        res[i,15:58] = tag_1_hot_pos
    res[:,58] = matrix[:,5]
    return res


def train_hmm(input_matrix, num_features, num_pos, num_tags):
    # Emission word_count matrix:
    # size (num_words, num_tags)
    # row: observation / colum: tag
    # (un-normalized if smoothing required)
    emission_w = np.zeros((num_features, num_tags), dtype=int)

    # Emission caos_count matrix:
    # size (5, num_tags)
    # row: observation / colum: caps
    # (un-normalized if smoothing required)
    emission_c = np.zeros((5, num_tags), dtype=int)

    # Emission pos_count matrix:
    # size (5, num_tags)
    # row: observation / colum: pos tag
    # (un-normalized if smoothing required)
    emission_p = np.zeros((num_pos, num_tags), dtype=int)

    # Building
    for r in input_matrix:
        emission_w[r[2]-1, r[5]-1] += 1
        emission_c[r[3]-1, r[5]-1] += 1
        emission_p[r[4]-1, r[5]-1] += 1

    # Transition matrix
    # size (num_tags, num_tags)
    # row: to / colum: from
    # (un-normalized if smoothing required)
    transition = np.zeros((num_tags, num_tags), dtype=int)
    for i in xrange(input_matrix.shape[0] - 1):
        transition[input_matrix[i+1, 5]-1, input_matrix[i, 5]-1] += 1

    return emission_w, emission_c, emission_p, transition


def main(arguments):
    # Args
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--f', default='data/features.hdf5',
                        type=str, help='Filename to save data')
    args = parser.parse_args(arguments)
    filename = args.f

    # Train
    pos2index = get_pos2index()
    tag2index = get_tag2index()
    num_words, num_sentences = count_elements('data/train.num.txt')
    num_rows = num_words + 2*num_sentences
    input_matrix_train, word2index = build_input_matrix('data/train.num.txt',
                                                        num_rows, tag2index,
                                                        pos2index)

    # Building the count matrix
    num_tags = len(tag2index)
    num_features = len(word2index)
    num_pos = len(pos2index) + 1
    emission_w, emission_c, emission_p, transition = train_hmm(input_matrix_train,
                                                               num_features, num_pos,
                                                               num_tags)

    # Dev & test
    num_words, num_sentences = count_elements('data/dev.num.txt')
    # Miss 1 blank line at the end of the file for the dev set
    num_rows = num_words + 2*num_sentences + 1
    input_matrix_dev, word2index = build_input_matrix('data/dev.num.txt',
                                                      num_rows, tag2index,
                                                      pos2index,
                                                      word2index=word2index)

    num_words, num_sentences = count_elements('data/test.num.txt',
                                              tags=False)
    num_rows = num_words + 2*num_sentences
    input_matrix_test, word2index = build_input_matrix('data/test.num.txt',
                                                       num_rows, tag2index,
                                                       pos2index,
                                                       tags=False,
                                                       word2index=word2index)

    # Saving pre-processing
    with h5py.File(filename, "w") as f:
        # Model
        f['emission_w'] = emission_w
        f['emission_c'] = emission_c
        f['emission_p'] = emission_p
        f['transition'] = transition

        f['input_matrix_train'] = input_matrix_train
        f['input_matrix_dev'] = input_matrix_dev
        f['input_matrix_test'] = input_matrix_test


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
