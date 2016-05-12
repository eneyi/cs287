import numpy as np
import argparse
import sys
import pickle
import h5py

from helper import *
import preprocess
import itertools

##########################
#
# README
#
# This code implements the count based baseline
# for the bAbI task. The preprocessing is done
# by the code in preprocess.py which is called here.
#
# Command args:
#   --task: tasks to solve (list of ints between 1 and 20,
#            tasks 8 and 19 are removed because the baseline)
#
# OUTPUT:
# None, it will print the result from the train and test set
# on average and by task in the command line
#
# HOW TO USE:
# For all tasks
# $  python count_based.py
# To precise tasks:
# $  python count_based.py --task 1 4
#
# Authors: Virgile Audi | Nicolas Drizard 2016

def sentence_relevance(story, ques, n=1, stopwords=None):

    nsentence, nword = story.shape

    relevance = np.zeros(nsentence)
    for i in range(nsentence):
        for j in range(nword):
            for w in ques:
                if story[i, j] == w and w not in stopwords:
                    relevance[i] += 1

    if n == 1:
        return np.argmax(relevance)
    else:
        res = list(np.arange(nsentence)[relevance.nonzero()]+1)
        for i in list(np.arange(nsentence)[relevance.nonzero()]):
            for ii in np.arange(nsentence):
                if ii+1 in res:
                    continue
                else:
                    count = 0
                    for j in range(nword):
                        for w in story[i, :]:
                            if story[ii, j] == w and w not in stopwords:
                                count += 1
                    if count > 0:
                        res.append(ii+1)

        return sorted(res)


def train_question_vector(questions, answers, aw_number, alpha=0.1):
    '''
    Build embeddings of the first word of the question based on the
    count of times each word is answer of the first question word.
    Output: dictionnary {q[0]: embeddings}
    '''
    # Build the embeddings
    questions_embeddings = {}
    for q, r in zip(questions, answers):
        # q[0] is the task_id
        # Index starts at 1
        if q[1] not in questions_embeddings:
            questions_embeddings[q[1]] = alpha*np.ones(aw_number)
        questions_embeddings[q[1]][r-1] += 1

    # Normalize
    for k in questions_embeddings.keys():
        questions_embeddings[k] /= np.sum(questions_embeddings[k])

    return questions_embeddings


def build_story_aw_distribution(facts, aw_number, alpha=0.1, decay=0.15):
    '''
    Compute the count of answer words in the fact. Weight down the
    old words.
    Output: normalized count vector
    '''
    count_vector = alpha*np.ones(aw_number)
    bow = facts.flatten()
    for i in xrange(len(bow)-1, -1, -1):
        b = bow[i]
        # check not padding and an answer word
        if b != 0 and b <= aw_number:
            # weighted coung
            count_vector[b-1] += 1 + decay*i
    # Normalization
    count_vector /= np.sum(count_vector)

    return count_vector


def batch_prediction(questions, questions_sentences, sentences, aw_number,
                     questions_embeddings, word2index, relevance=False):
    '''
    Predict the answer to all the questions in the batch as a distribution on
    the possible answer words.
    Use the whole story as facts.
    Assume feature independent.
    '''
    # To store predictions
    predictions = np.zeros((len(questions), aw_number))

    # Go over each question
    for qi, q in enumerate(questions):
        # Facts
        # Removing task id from the facts
        facts = sentences[
            questions_sentences[qi][0]-1: questions_sentences[qi][1], 1:]

        if relevance:
            facts_ = sentences[
                questions_sentences[qi][0]-1: questions_sentences[qi][1], 1:]
            the = word2index['the']
            to = word2index['to']
            facts = facts_[np.array(sentence_relevance(facts_, q[1:], 1, [0, the, to]), dtype=int)-1, :]

        # Features
        f1 = np.log(questions_embeddings[q[1]])
        f2 = np.log(build_story_aw_distribution(facts, aw_number))
        # print(q)
        # print(facts)
        # print(f1)
        # print(f2)
        # break

        predictions[qi, :] = np.exp(f1+f2)
        predictions[qi, :] /= np.sum(predictions[qi, :])

    return predictions


def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        '--tasks', default=range(1, 21), type=int, nargs='+', help='Tasks list')
    args = parser.parse_args(arguments)

    # Filter the tasks expecting more than one output
    tasks = args.tasks
    for t in [19]:
        if t in tasks:
            tasks.remove(t)

    # Wrap up in an argument
    new_arguments = ['--task'] + [str(t) for t in tasks]
    # Preprocessing
    preprocess.main(new_arguments)

    # Loading the data
    sentences_train, questions_train, questions_sentences_train, answers_train = read_preprocessed_matrix_data(
        'new_train')
    with open('../Data/preprocess/new_word2index', 'rb') as file:
        word2index = pickle.load(file)

    # ###### Training the questions embeddings
    # Count the answers words (indexed from 1 to len(answer_words))
    answer_words = set(answers_train.flatten())
    aw_number = len(answer_words)

    # Questions embeddings
    questions_embeddings_train = train_question_vector(questions_train, answers_train,
                                                 aw_number, alpha=0.1)

    # ##### Predictions
    # ##### Train
    # Batch predictions
    predictions_train = batch_prediction(questions_train, questions_sentences_train, sentences_train,
                                   aw_number, questions_embeddings_train, word2index)

    # Select response (index start at 1)
    output = np.argmax(predictions_train, axis=1) + 1

    # Compute global accuracy
    response = answers_train.flatten()
    print(len(response))
    accuracy = np.sum(output == response)/(1.*len(output))

    # Accuracy per tasks on train
    results_train = np.ones((len(tasks), 2))
    for i in xrange(len(tasks)):
        task_id = questions_train[1000*i, 0]
        local_acc = np.sum(
            output[1000*i:1000*(i+1)] == response[1000*i:1000*(i+1)])/(1000.)
        results_train[i, 0] = task_id
        results_train[i, 1] = local_acc

    print('---------------TRAIN------------------')

    for i in xrange(len(tasks)):
        print 'Results for task {}'.format(results_train[i, 0])
        print 'Average Accuracy is {}'.format(results_train[i, 1])
        print('----------------------------------------')

    print('----------------------------------------')
    print('Number of possible answers {}'.format(aw_number))
    print 'Results for {}'.format(tasks)
    print 'Average Accuracy is {}'.format(accuracy)
    print('----------------------------------------')

    # ##### Test
    sentences_test, questions_test, questions_sentences_test, answers_test = read_preprocessed_matrix_data(
        'new_test')

    # Batch predictions
    predictions_test = batch_prediction(questions_test, questions_sentences_test, sentences_test,
                                        aw_number, questions_embeddings_train, word2index)

    # Select response (index start at 1)
    output = np.argmax(predictions_test, axis=1) + 1

    # Compute global accuracy
    response = answers_test.flatten()
    print(len(response))
    accuracy = np.sum(output == response)/(1.*len(output))

    # Accuracy per tasks on train
    results_test = np.ones((len(tasks), 2))
    for i in xrange(len(tasks)):
        task_id = questions_test[1000*i, 0]
        local_acc = np.sum(
            output[1000*i:1000*(i+1)] == response[1000*i:1000*(i+1)])/(1000.)
        results_test[i, 0] = task_id
        results_test[i, 1] = local_acc

    print('---------------TEST------------------')

    for i in xrange(len(tasks)):
        print 'Results for task {}'.format(results_test[i, 0])
        print 'Average Accuracy is {}'.format(results_test[i, 1])
        print('----------------------------------------')

    print('----------------------------------------')
    print('Number of possible answers {}'.format(aw_number))
    print 'Results for {}'.format(tasks)
    print 'Average Accuracy is {}'.format(accuracy)
    print('----------------------------------------')


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
