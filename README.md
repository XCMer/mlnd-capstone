# MLND - Quora De-duplication

## Setup

Assuming that you already have a `virtualenv` or `conda` environment setup, run the following commands in bash:

```
# Install all the requirements
pip install -r requirements.txt

# Download spaCy models and GloVe vectors
python -m spacy download en_core_web_md

# Mark the above model as the default
python -m spacy link --force en_core_web_md en_default
```

## Dataset

Since the data was released in a Kaggle competition, the terms have to be accepted in order to get them. This is the link to the competition: https://www.kaggle.com/c/quora-question-pairs

The dataset has two files, `train.csv` and `test.csv`, and they have to be put in the same directory as this README.

## Running the jupyter notebook

To run the jupyter notebook (for training):

```
KERAS_BACKEND=theano jupyter notebook
```

## Running the command-line application

To run the command-line application, and provide it with your own question pairs or ones from Quora's test set:

```
KERAS_BACKEND=theano python run.py
```

And follow the instructions. To quit, press `CTRL + C`.
