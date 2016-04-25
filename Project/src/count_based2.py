import numpy as np
import argparse
import sys
import pickle
import h5py

from helper import *
from sentences_matching import sentence_relevance
import preprocess
import itertools


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
    Build embeddings of the first word of the question.
    Output: dictionnary {q[0]: embeddings}
    '''
    # Build the embeddings
    questions_embeddings = {}
    for q, r in zip(questions, answers):
        # Index starts at 1
        if q[0] not in questions_embeddings:
            questions_embeddings[q[0]] = alpha*np.ones(aw_number)
        questions_embeddings[q[0]][r[0]-1] += 1

    # Normalize
    for k in questions_embeddings.keys():
        questions_embeddings[k] /= np.sum(questions_embeddings[k])

    return questions_embeddings


def build_story_aw_distribution(facts, aw_number, alpha=0.1, decay=0.1):
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


def question_matching_feature(facts, question, aw_number, alpha=0.1):
    '''
    Indicator feature on the set of possible answer words to indicate
    if a question word is inside the same sentence as an answer word
    in the facts.
    Output: normalized count vector.
    '''
    count_vector = alpha*np.ones(aw_number)
    question_set = set([q for q in question if q != 0])
    for fact in facts:
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

        facts_ = sentences[questions_sentences[qi][0]-1: questions_sentences[qi][1], :]
        facts =  facts_[np.array(sentence_relevance(facts_, questions[qi], 2, [0, 20, 90]), dtype=int)-1,:]
        # Features
        f1 = np.log(questions_embeddings[q[0]])
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

    # Compute accuracygit
    response = answers.flatten()
    accuracy = np.sum(output == response)/(1.*len(output))

    print('----------------------------------------')
    print('Number of possible answers {}'.format(aw_number))
    print 'Results for {}'.format(tasks)
    print 'Average Accuracy is {}'.format(accuracy)
    print('----------------------------------------')


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))