import numpy as np
import argparse
import sys
import pickle
import h5py


from os import listdir
from os.path import isfile, join
from collections import Counter


DATA_PATH = '../Data'


def build_sentences_mapping(filenames):
    '''
    Build several mappings:
        index2sentence: sentence index to bow of sentence
        index2question: question index to bow of question
        index2supportings: question index to sentence index of supportings
            facts
        answers: vector of answers (index by order of question index)
        question_index2sentences_index: question index to the range
            [start_index, end_index] of sentences it corresponds to
        word2index: word to word index (first Nw indexes correspond to answer words)

    Indexes always start at 1 (for lua)
    '''
    index2sentence = {}
    s = 1
    index2question = {}
    index2supportings = {}
    answers = []
    question_index2sentences_index = {}
    q = 1
    word2index = {}
    w = 1

    # Set of unique words
    words = set()

    # Loop over all the filenames (in the given order)
    for filename in filenames:
        with open(DATA_PATH + '/en/' + filename, 'r') as f:
            for line in f:
                line_split = line[:-1].split('\t')
                # Check if question
                if len(line_split) > 1:
                    # Answer word
                    answer = line_split[1]
                    if answer not in word2index:
                        word2index[answer] = w
                        w += 1
                    bow = line_split[0].rstrip('? ').split()[1:]
                    index2question[q] = bow
                    index2supportings[q] = [int(u) for u in line_split[2].split()]
                    answers.append(word2index[answer])
                    question_index2sentences_index[q] = [current_start, s-1]
                    q += 1
                else:
                    bow = line_split[0].rstrip('.').split()
                    local_ind = int(bow[0])
                    # Restart current_start if new story
                    if local_ind == 1:
                        current_start = s
                    index2sentence[s] = bow[1:]
                    s += 1
                words.update(set(bow))
    # Set the number of answer words
    Nw = len(word2index)

    # Complete the mapping word2index
    for word in words:
        if word not in word2index:
            word2index[word] = w
            w += 1

    return Nw, word2index, index2sentence, index2question, index2supportings, question_index2sentences_index, answers


def build_bow_array(index2question, word2index):
    Nq = len(index2question)

    # Get max length
    max_len = 0
    for bow in index2question.values():
        if max_len < len(bow):
            max_len = len(bow)

    questions = np.zeros((Nq, max_len), dtype=int)
    for i in xrange(1, Nq+1):
        bow = index2question[i]
        row = [word2index[w] for w in bow]
        questions[i-1, :len(bow)] = row

    return questions


def build_questions_sentences_array(index2supportings, question_index2sentences_index):
    Nq = len(index2supportings)

    # We assume at most 3 supporting facts
    qs = np.zeros((Nq, 5), dtype=int)
    for i in xrange(1, Nq+1):
        qs[i-1, :2] = question_index2sentences_index[i]
        supportings = index2supportings[i]
        qs[i-1, 2:2+len(supportings)] = supportings
    return qs


# TESTED
def get_filenames(train=True, tasks=[]):
    '''
    Return filenames of the dataset in list of string.
    ARGS:
        train: boolean
        tasks: list of integers, to specify tasks (if empty all tasks)
    '''
    # Get filenames
    path = DATA_PATH + '/en/'
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    # Filter train/test
    if train:
        onlyfiles = [f for f in onlyfiles if 'train' in f]
    else:
        onlyfiles = [f for f in onlyfiles if 'test' in f]
    # Filter tasks
    if len(tasks):
        selection = []
        for f in onlyfiles:
            current_number = f.split('_')[0][2:]
            if int(current_number) in tasks:
                selection.append(f)
        if len(selection):
            return selection
        else:
            raise ValueError('Tasks not found {}'.format(tasks))
    return onlyfiles


def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--tasks', default=0, type=int, nargs='+', help='Tasks list')
    parser.add_argument('--f', default='new', type=str,
                        help='Filename to save preprocess data in hdf5 format')
    parser.add_argument('--train', default=1, type=int,
                        help='Dataset to preprocess')
    args = parser.parse_args(arguments)

    # Retrieve filename
    filenames = get_filenames(train=args.train, tasks=args.tasks)
    print(filenames)

    # ###### STEP 1: mappings
    Nw, word2index, index2sentence, index2question, index2supportings, question_index2sentences_index, answers = build_sentences_mapping(filenames)
    index2word = {v: k for k, v in word2index.iteritems()}

    # ###### STEP 2: arrays

    questions = build_bow_array(index2question, word2index)
    sentences = build_bow_array(index2sentence, word2index)
    questions_sentences = build_questions_sentences_array(index2supportings,
                                                          question_index2sentences_index)

    # ###### STEP 3: saving

    # Matrix in hdf5 format
    filename = DATA_PATH + '/preprocess/' + args.f
    print(filename)
    with h5py.File(filename + '.hdf5', "w") as f:
        f['sentences'] = sentences
        f['questions'] = questions
        f['questions_sentences'] = questions_sentences
        f['answers'] = answers

    # Mapping as python object
    for f, obj in zip(['_word2index', '_index2sentence', '_index2question'],
                      [word2index, index2sentence, index2question]):
        with open(filename + f, 'wb') as file:
            pickle.dump(obj, file)

    # Debugging
    print(len(index2word))
    print(sentences.shape)
    print(sentences[:5])
    print(questions.shape)
    print(questions[:5])
    print(questions_sentences.shape)
    print(questions_sentences[:5])
    print(len(answers))
    print([index2word[i] for i in answers[:5]])

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
