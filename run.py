from keras.models import load_model
import spacy
import numpy as np

nlp = spacy.load('en_default')

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
    question1 = features_for_text(nlp, raw_input('Question 1: '))
    question2 = features_for_text(nlp, raw_input('Question 2: '))

    q1 = np.asarray([question1, question2])
    q2 = np.asarray([question2, question1])

    prediction = model.predict([q1, q2])
    p1 = prediction[0][0]
    p2 = prediction[1][0]
    p = max(p1, p2)

    print("Score: {}".format(str(p)))

    if p >= 0.5:
        print("IS DUPLICATE")
    else:
        print("NOT DUPLICATE")
    print("")