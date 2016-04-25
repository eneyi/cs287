# README: How to use me?
#
# Pre process the question from the bAbI dataset.
#
# Arguments:
#     [--tasks TASKS [TASKS ...]] list of tasks to pre process, if empty all tasks
#     [--f F]: filename to save pre-processed data
#     [--train TRAIN]: train data if 1, test ow
#
# Instructions example:
# $ python preprocess.py --task 2 --f task2_train
#
#
# Outputs: save all the preprocessed data on disk
# (2 functions are defined in helper.py to read them,
# with the same filename argument as this script)
#
#     with h5py.File(filename + '.hdf5', "w") as f:
#     sentences: np.array (rows: num_sentences, columns: padded list of words index)
#     questions: np.array (rows: num_questions, columns: padded list of words index)
#     questions_sentences: np.array (rows: num_questions,
#         columns: (start of sentences story, end of sentences story,
#         list supporting facts))
#     answers: np.array(rows: num_questions, cols: answers (singleton or list))
import numpy as np
import argparse
import sys
import pickle
import h5py

from os import listdir
from os.path import isfile, join

from helper import *


DATA_PATH = '../Data'


def build_sentences_mapping(filenames_tuples, word2index=None):
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

    # helper mapping for the supporting fact
    # Map the local index of the sentences in a story
    # to their global index
    localindex2globalindex = {}

    # Indexes start at 1, padding will be added to the end of the mapping
    if word2index is None:
        pretrained = False
        word2index = {}
        w = 1
    else:
        pretrained = True

    # Set of unique words
    words = set()

    # Loop over all the filenames (in the given order)
    for tup in filenames_tuples:
        task_id, filename = tup
        with open(DATA_PATH + '/en/' + filename, 'r') as f:
            for line in f:
                line_split = line[:-1].split('\t')
                # Check if question
                if len(line_split) > 1:
                    # Answer list (in case of list of words as answer)
                    answer = line_split[1].split(',')
                    if not pretrained:
                        for aw in answer:
                            if aw not in word2index:
                                word2index[aw] = w
                                w += 1
                    bow = line_split[0].rstrip('? ').split()[1:]
                    # tuple (id_task, bow)
                    index2question[q] = (task_id, bow)
                    index2supportings[q] = [localindex2globalindex[int(u)] for u in line_split[2].split()]
                    answers.append([word2index[aw] for aw in answer])
                    question_index2sentences_index[q] = [current_start, s-1]
                    q += 1
                else:
                    line_cleaned = line_split[0].rstrip('.').split()
                    local_ind = int(line_cleaned[0])

                    # Remove the local index
                    bow = line_cleaned[1:]
                    # Restart current_start if new story
                    if local_ind == 1:
                        current_start = s
                        # Restart mapping
                        localindex2globalindex = {}
                    # tuple (id_task, bow)
                    index2sentence[s] = (task_id, bow)
                    localindex2globalindex[local_ind] = s
                    s += 1
                words.update(set(bow))

    # Complete the mapping word2index
    if not pretrained:
        for word in words:
            if word not in word2index:
                word2index[word] = w
                w += 1

    # Convert answers to np.array
    # Compute the max number of answers for a question
    max_number = 1
    for a in answers:
        if len(a) > max_number:
            max_number = len(a)
    answers_matrix = np.zeros((len(answers), max_number))
    for i, a in enumerate(answers):
        answers_matrix[i, :len(a)] = a

    return word2index, index2sentence, index2question, index2supportings, question_index2sentences_index, answers_matrix


def build_bow_array(index2question, word2index, padding_index):
    Nq = len(index2question)

    # Get max length
    max_len = 0
    for tup in index2question.values():
        task_id, bow = tup
        if max_len < len(bow):
            max_len = len(bow)
    bows = padding_index*np.ones((Nq, max_len + 1), dtype=int)
    for i in xrange(1, Nq+1):
        task_id, bow = index2question[i]
        row = [word2index[w] for w in bow]
        # Stores task id
        bows[i-1, 0] = task_id
        # Stores bow
        bows[i-1, 1:1+len(row)] = row

    return bows


def build_questions_sentences_array(index2supportings, question_index2sentences_index):
    Nq = len(index2supportings)

    # Counting the length max of supporting facts
    max_num_supporting = max([len(v) for v in index2supportings.values()])

    # We assume at most 3 supporting facts
    qs = np.zeros((Nq, 2 + max_num_supporting), dtype=int)
    for i in xrange(1, Nq+1):
        qs[i-1, :2] = question_index2sentences_index[i]
        supportings = index2supportings[i]
        qs[i-1, 2:2+len(supportings)] = supportings
    return qs


def get_filenames(train=True, tasks=[]):
    '''
    Return filenames of the dataset in list of string.
    ARGS:
        train: boolean
        tasks: list of integers, to specify tasks (if empty all tasks)
    OUTPUTS:
        selection: list [(task_number, filename)]
    '''
    # Get filenames
    path = DATA_PATH + '/en/'
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    # Filter train/test
    if train:
        onlyfiles = [f for f in onlyfiles if 'train' in f]
    else:
        onlyfiles = [f for f in onlyfiles if 'test' in f]
    # Add the task number with the filename in the tuple
    tuples_list = []
    for f in onlyfiles:
        task_number = int(f.split('_')[0][2:])
        tuples_list.append((task_number, f))

    # Filter tasks
    if len(tasks):
        selection = []
        for tup in tuples_list:
            if tup[0] in tasks:
                selection.append(tup)
    else:
        selection = tuples_list

    # Order the filename list by task
    selection.sort(key=lambda x: x[0])

    return selection


def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--tasks', default=range(1, 21), type=int, nargs='+', help='Tasks list')
    parser.add_argument('--f', default='new', type=str,
                        help='Filename to save preprocess data in hdf5 format')
    args = parser.parse_args(arguments)

    # Filter out tasks with multiple anwsers expected
    tasks = args.tasks
    for t in [8, 19]:
        if t in tasks:
            tasks.remove(t)

    # ###### STEP 0: retrieving filenames
    # Train
    filenames_tuples_train = get_filenames(train=True, tasks=tasks)
    # Test
    filenames_tuples_test = get_filenames(train=False, tasks=tasks)

    # ###### STEP 1: mappings
    # Train
    word2index, index2sentence_train, index2question_train, index2supportings_train, question_index2sentences_index_train, answers_train = build_sentences_mapping(filenames_tuples_train)
    index2word = {v: k for k, v in word2index.iteritems()}
    # Test: use word2index from train
    word2index, index2sentence_test, index2question_test, index2supportings_test, question_index2sentences_index_test, answers_test = build_sentences_mapping(filenames_tuples_test, word2index=word2index)

    # ###### STEP 2: arrays
    # Train
    padding_index = len(word2index) + 1
    questions_train = build_bow_array(index2question_train, word2index, padding_index)
    sentences_train = build_bow_array(index2sentence_train, word2index, padding_index)
    questions_sentences_train = build_questions_sentences_array(index2supportings_train,
                                                                question_index2sentences_index_train)
    # Test
    questions_test = build_bow_array(index2question_test, word2index, padding_index)
    sentences_test = build_bow_array(index2sentence_test, word2index, padding_index)
    questions_sentences_test = build_questions_sentences_array(index2supportings_test,
                                                                question_index2sentences_index_test)

    # ###### STEP 3: saving

    # Matrix in hdf5 format
    # Train
    filename = DATA_PATH + '/preprocess/' + args.f + '_train'
    with h5py.File(filename + '.hdf5', "w") as f:
        f['sentences'] = sentences_train
        f['questions'] = questions_train
        f['questions_sentences'] = questions_sentences_train
        f['answers'] = answers_train
        f['voc_size'] = np.array([len(word2index) + 1])
    print('Matrix Train saved in {}.hdf5'.format(filename))

    # Test
    filename = DATA_PATH + '/preprocess/' + args.f + '_test'
    with h5py.File(filename + '.hdf5', "w") as f:
        f['sentences'] = sentences_test
        f['questions'] = questions_test
        f['questions_sentences'] = questions_sentences_test
        f['answers'] = answers_test
        f['voc_size'] = np.array([len(word2index) + 1])

    print('Matrix Test saved in {}.hdf5'.format(filename))

    # Mapping as python object (we save only the word mapping)
    # for f, obj in zip(['_word2index', '_index2sentence', '_index2question'],
    #                   [word2index, index2sentence, index2question]):
    #     with open(filename + f, 'wb') as file:
    #         pickle.dump(obj, file)

    with open(filename + '_word2index', 'wb') as file:
        pickle.dump(word2index, file)

    print('Mapping of words from train pickled in {}'.format(filename))


    # Debugging
    print('DEBUG print')
    print(len(index2word))
    print(sentences_train.shape)
    print(sentences_train[:5])
    print(questions_train.shape)
    print(questions_train[:5])
    print(questions_sentences_train.shape)
    print(questions_sentences_train[:5])
    print(answers_train.shape)
    print([[index2word[aw] for aw in r] for r in answers_train[:5]])

if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
