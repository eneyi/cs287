import numpy as np
import h5py
import argparse
import sys
import re
import codecs

from collections import Counter


FILE_PATHS = ("data/train_chars.txt",
              "data/valid_chars.txt",
              "data/test_chars.txt")


def get_input(filename, n, char_to_ind=None):
    # Contain the list of characters indices in the data
    # initialized with a padding
    if n > 2:
        input_data = [1]*(n-2)
    else:
        input_data = []
    if char_to_ind is None:
        # Map each character to an index with
        # Index of <space> set to 0
        char_to_ind = {'<space>': 0, '</s>': 1}
        count = 2
    with open(filename, 'r') as f:
        # Loop to index the char and store them inside the input
        for line in f:
            for c in line[:-1].split(' '):
                # Input data
                if c in char_to_ind:
                    input_data.append(char_to_ind[c])
                else:
                    char_to_ind[c] = count
                    count += 1
                    input_data.append(char_to_ind[c])
    return input_data, char_to_ind


def build_train_data(input_data, n):
    # Build the input matrix: (num_records, n-1)
    # and the output vector (num_records,1)
    # which stores the output for the given (n-1)gram
    input_matrix = np.zeros((len(input_data)-n, n-1))
    output_matrix = np.zeros(len(input_data)-n)
    for i in xrange(len(input_data)-n):
        # Countext is a (n-1)gram
        w = input_data[i:i+(n-1)]
        input_matrix[i, :] = w
        output_matrix[i] = (1 if input_data[i+(n-1)] == 0 else 2)
    return input_matrix, output_matrix


def build_count_matrix(input_matrix, output_matrix, n):
    count_matrix_raw = np.concatenate((input_matrix,
                                       output_matrix.reshape(output_matrix.shape[0], 1)), axis=1)

    num_rows = len(set([tuple(s) for s in input_matrix]))
    count = Counter([tuple(s) for s in count_matrix_raw])

    # count matrix: (num_(n-1grams, 2))
    F = np.zeros((num_rows, n + 1))
    gram_to_ind = {}
    i = 0
    for k, v in count.iteritems():
        gram = k[:(n-1)]
        if gram not in gram_to_ind:
            gram_to_ind[gram] = i
            i += 1
        F[gram_to_ind[gram], n-1 + int(k[-1]) - 1] = v
        F[gram_to_ind[gram], :n-1] = list(gram)

    return F


def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--N', default=2, type=int, help='Ngram size')
    args = parser.parse_args(arguments)
    N = args.N

    train, valid, test = FILE_PATHS

    # Train
    input_data_train, char_to_ind = get_input(train, N)
    input_matrix_train, output_matrix_train = build_train_data(
        input_data_train, N)
    F_train = build_count_matrix(input_matrix_train, output_matrix_train, N)

    # Valid
    input_data_valid, char_to_ind = get_input(valid, N, char_to_ind)

    # Test
    input_data_test, char_to_ind = get_input(test, N, char_to_ind)

    filename = 'data_preprocessed/' + str(N) + '-grams.hdf5'
    with h5py.File(filename, "w") as f:
        # Stores a matrix (num_records, N-1) with at each row
        # the (N-1) grams appearing in the input data
        f['input_matrix_train'] = input_matrix_train
        f['F_train'] = F_train
        # Vector (num_records) storing the class of the next word
        # after the (N-1) gram stored at the same index in input_matrix
        # 1 is space; 2 is character
        f['output_matrix_train'] = output_matrix_train
        # Stores the list of consecutives character (or space) as their
        # index from the mapping char_to_ind
        f['input_data_train'] = np.array(input_data_train)
        f['input_data_valid'] = np.array(input_data_valid)
        f['input_data_test'] = np.array(input_data_test)


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
