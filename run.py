from keras.models import load_model
import spacy
import numpy as np
import pandas as pd
import random

nlp = spacy.load('en_default')
test_data = pd.read_csv('test.csv')
total_records = len(test_data)

# Get GloVe vectors for each word
def get_vectors_for_text(nlp, text):
    return np.array([w.vector for w in nlp(text)])

# Take the mean of all the rows, thus getting a single
# row in the end
def mean_pool(text_vectors):
    return np.mean(text_vectors, axis=0)

# Take the max from all rows, thus getting a single
# row in the end
def max_pool(text_vectors):
    return np.max(text_vectors, axis=0)

# Concat of max and mean pool of all vectors
def features_for_text(nlp, text):
    vectors = get_vectors_for_text(nlp, unicode(text.strip()))
    
    return np.concatenate((max_pool(vectors), mean_pool(vectors)))

model = load_model('300_True_6.h5')

while True:
    question_type = raw_input('Select a random question? ')
    if question_type == 'y':
        random_id = random.randint(0, total_records-1)
        record = test_data.loc[random_id]
        question1 = features_for_text(nlp, str(record['question1']))
        question2 = features_for_text(nlp, str(record['question2']))

        print("Question 1:", record['question1'])
        print("Question 2:", record['question2'])
    else:
        question1 = features_for_text(nlp, raw_input('Question 1: '))
        question2 = features_for_text(nlp, raw_input('Question 2: '))

    q1 = np.asarray([question1, question2])
    q2 = np.asarray([question2, question1])

    prediction = model.predict([q1, q2])
    p1 = prediction[0][0]
    p2 = prediction[1][0]
    p = min(p1, p2)

    print("Normal order score: {}".format(str(p1)))
    print("Reverse order score: {}".format(str(p2)))
    print("Score: {}".format(str(p)))

    if p >= 0.5:
        print("IS DUPLICATE")
    else:
        print("NOT DUPLICATE")
    print("")