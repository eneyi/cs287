# Helper functions
import numpy as np
import h5py
import pickle

PREPROCESS_PATH = '../Data/preprocess/'


def read_preprocessed_matrix_data(filename):
    fname = PREPROCESS_PATH + filename + '.hdf5'

    # Loading array
    with h5py.File(fname, 'r') as hf:
        sentences = np.array(hf.get('sentences'), dtype=int)
        questions = np.array(hf.get('questions'), dtype=int)
        questions_sentences = np.array(hf.get('questions_sentences'), dtype=int)
        answers = np.array(hf.get('answers'), dtype=int)

    return sentences, questions, questions_sentences, answers


def read_preprocessed_mapping(filename):
    fname = PREPROCESS_PATH + filename

    with open(fname + '_word2index', 'rb') as file:
        word2index = pickle.load(file)
    with open(fname + '_index2sentence', 'rb') as file:
        index2sentence = pickle.load(file)
    with open(fname + '_index2question', 'rb') as file:
        index2question = pickle.load(file)

    return word2index, index2sentence, index2question
