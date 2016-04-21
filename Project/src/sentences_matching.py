import numpy as np
import h5py
import itertools


def sentence_relevance(story, ques, n = 1, stopwords = None):

    nsentence, nword =  story.shape
    
    relevance = np.zeros(nsentence)
    for i in range(nsentence):
        for j in range(nword):
            for w in ques:
                if story[i,j] == w and w not in stopwords:
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
                        for w in story[i,:]:
                            if story[ii,j] == w and w not in stopwords:
                                count += 1
                    if count > 0:
                        res.append(ii+1)
                            
        return sorted(res)

# with h5py.File('../Data/preprocess/task2_train.hdf5','r') as hf:
#     ans = np.array(hf.get('answers'))
#     questions = np.array(hf.get('questions'))
#     questions_sentences = np.array(hf.get('questions_sentences'))
#     sentences = np.array(hf.get('sentences'))

# nq = 2
# story_1 = sentences[(questions_sentences[nq,0]-1):questions_sentences[nq,1],:]
# ques = questions[nq]

# print "Relevant sentences: "
# print sentence_relevance(story_1, ques, nq, [0, 21, 90])
# print "True relevant sentences: "
# print questions_sentences[nq][2]-nq,questions_sentences[nq][3]-nq