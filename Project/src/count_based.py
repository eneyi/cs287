import numpy as np
import argparse
import sys
import pickle
import h5py

from helper import *
import preprocess


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
        questions_embeddings[q[1]][r[0]-1] += 1

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
    # Removing task id from the facts
    bow = facts[:, 1:].flatten()
    for i in xrange(len(bow)-1, -1, -1):
        b = bow[i]
        # check not padding and an answer word
        if b != 0 and b <= aw_number:
            # weighted coung
            count_vector[b-1] += 1 + decay*i
    # Normalization
    count_vector /= np.sum(count_vector)

    return count_vector


def question_matching_feature(facts, question, aw_number, alpha=0.1):
    '''
    Indicator feature on the set of possible answer words to indicate
    if a question word is inside the same sentence as an answer word
    in the facts.
    Output: normalized count vector.
    '''
    count_vector = alpha*np.ones(aw_number)
    question_set = set([q for q in question if q != 0])
    for tup in facts:
        task, fact = tup
        fact_set = set([q for q in fact if q != 0])
        intersection = question_set.intersection(fact_set)
        for i in intersection:
            pass


def batch_prediction(questions, questions_sentences, sentences, aw_number,
                     questions_embeddings):
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
        facts = sentences[questions_sentences[qi][0]-1: questions_sentences[qi][1], :]

        # Features
        f1 = np.log(questions_embeddings[q[1]])
        f2 = np.log(build_story_aw_distribution(facts, aw_number))
        # print(q)
        # print(facts)
        # print(f1)
        # print(f2)

        predictions[qi, :] = np.exp(f1+f2)
        predictions[qi, :] /= np.sum(predictions[qi, :])

    return predictions


def main(arguments):
    global args
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--f', default='new', type=str,
                        help='Filename to save preprocess data in hdf5 format')
    parser.add_argument('--tasks', default=range(1, 21), type=int, nargs='+', help='Tasks list')
    args = parser.parse_args(arguments)

    # # Storing the results (task number, num possible responses, accuracy)
    # results = np.zeros((len(args.tasks), 3))

    # # Looping over the tasks
    # for i, task in enumerate(args.tasks):
    #     # Pre-processing the data
    #     arguments = ['--task', str(task)]
    #     preprocess.main(arguments)

    #     # Loading the data
    #     sentences, questions, questions_sentences, answers = read_preprocessed_matrix_data('new')
    #     word2index, index2sentence, index2question = read_preprocessed_mapping('new')

    #     # Count the answers words (indexed from 1 to len(answer_words))
    #     answer_words = set(answers.flatten())
    #     aw_number = len(answer_words)

    #     # Questions embeddings
    #     questions_embeddings = train_question_vector(questions, answers, aw_number,
    #                                                  alpha=0.1)

    #     # Batch predictions
    #     predictions = batch_prediction(questions, questions_sentences, sentences,
    #                                    aw_number, questions_embeddings)
    #     # Select response (index start at 1)
    #     output = np.argmax(predictions, axis=1) + 1

    #     # Compute accuracy
    #     response = answers.flatten()
    #     accuracy = np.sum(output == response)/(1.*len(output))

    #     # Storing result
    #     results[i, :] = [task, aw_number, accuracy]

    # # VERBOSE
    # print(results)
    # print('----------------------------------------')
    # print 'Results for {}'.format(args.tasks)
    # print 'Average Accuracy is {}'.format(np.mean(results[:,-1]))
    # print('----------------------------------------')

    # Filter the tasks expecting more than one output
    tasks = args.tasks
    for t in [8, 19]:
        if t in tasks:
            tasks.remove(t)
    # Wrap up in an argument
    new_arguments = ['--task'] + [str(t) for t in tasks]

    # All in a row
    preprocess.main(new_arguments)

    # Loading the data
    sentences, questions, questions_sentences, answers = read_preprocessed_matrix_data('new')
    word2index, index2sentence, index2question = read_preprocessed_mapping('new')

    # Count the answers words (indexed from 1 to len(answer_words))
    answer_words = set(answers.flatten())
    aw_number = len(answer_words)

    # Questions embeddings
    questions_embeddings = train_question_vector(questions, answers, aw_number,
                                                 alpha=0.1)

    # Batch predictions
    predictions = batch_prediction(questions, questions_sentences, sentences,
                                   aw_number, questions_embeddings)

    # Select response (index start at 1)
    output = np.argmax(predictions, axis=1) + 1

    # Compute global accuracy
    response = answers.flatten()
    print(len(response))
    accuracy = np.sum(output == response)/(1.*len(output))

    # Accuracy per tasks
    for i in xrange(len(tasks)):
        task_id = questions[1000*i, 0]
        local_acc = np.sum(output[1000*i:1000*(i+1)] == response[1000*i:1000*(i+1)])/(1000.)
        print('----------------------------------------')
        print 'Results for task {}'.format(task_id)
        print 'Average Accuracy is {}'.format(local_acc)
        print('----------------------------------------')

    print('----------------------------------------')
    print('Number of possible answers {}'.format(aw_number))
    print 'Results for {}'.format(tasks)
    print 'Average Accuracy is {}'.format(accuracy)
    print('----------------------------------------')


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))