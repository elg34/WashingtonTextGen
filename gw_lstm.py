import numpy as np
import sys
import re
import random as rd
from collections import Counter
import glob
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.models import load_model
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import random
from sklearn.metrics import roc_auc_score, confusion_matrix, accuracy_score
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from scipy.sparse import csr_matrix, vstack, lil_matrix
from scipy import io
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import optimizers
import os
import time
import theano
import pickle
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())


def preproc(text):
    'preprocessing of large text string'
    text = re.sub(' Mr.', ' Mr', text)
    text = re.sub(' Mrs.', ' Mrs', text)
    text = re.sub(' Messrs.', ' Messrs', text)
    text = text.lower()
    text = re.sub('\[(.*?)\]', '', text)
    text = re.sub('\n\n', ' ', text)
    text = re.sub('\n', ' ', text)
    text = re.sub(',', '', text)
    text = re.sub('_', '', text)
    text = re.sub(';', '', text)
    text = re.sub('_figure_', '', text)
    text = re.sub('\d+','',text)
    return text

# get all words in target text file, plus all files in a directory for optional pre-training
def get_data(targetname,pretrains = None):
    'load text from file, optional pretraining'
    rawwords=[]
    text = open(targetname).read() # now do the same for target file
    text = preproc(text)
    wds = re.findall(r"[\w']+|[.,!?;]", text)
    print('Words in target set: ', len(wds))
    for wd in wds:
        rawwords.append(wd)     
    with open('text_main.pkl', 'wb') as fp:
        pickle.dump(rawwords, fp)
    
    if pretrains: # optional
        filename = glob.glob(pretrains) # get directory and file ending for pretraining files
        count=0
        for i in filename:
            text = open(i).read()
            text = preproc(text) # preprocess file text
            wds = re.findall(r"[\w']+|[.,!?;]", text) # split text into individual words
            for wd in wds:
                rawwords.append(wd) # append individual words to our raw training array
            count +=1
            print('loading file...', count+1, '/',len(filename),' :: ',i)
        print('Words in pretraining set: ', len(rawwords))
        with open('text_pretrain.pkl', 'wb') as fp:
            pickle.dump(rawwords, fp)

    return rawwords
    
def batch_generator():
    count = 0
    shuffle_index = np.arange(n_examples)
    np.random.shuffle(shuffle_index)
    while 1:
        dataX = []
        dataY = []
        for i in shuffle_index[batch_size*count:batch_size*(count+1)]:
            seq_in = words[i:i + seq_length] # seq_length number of words from all training words
            seq_out = words[i + seq_length] # the word after that, training signal
            dataX.append([word_to_int[word] for word in seq_in]) # seq of words turned into int value
            dataY.append(word_to_int[seq_out]) # training signal word words turned into int value
        X = np.asarray(dataX) # [individual samples, length of sequence]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1)) # [individual samples, length of sequence, features]
        X = X / np.float32(n_uwords) # normalize
        y = np_utils.to_categorical(dataY,num_classes=n_uwords) # one hot encoder
        count += 1 # move one batch forward and repeat
        yield(X,y)
        if (count >= n_batches):
                np.random.shuffle(shuffle_index)
                count=0

def make_model(opt,loss):
        # Keras model
        nout = n_uwords
        nin = seq_length
        print('input size:',nin)
        print('output size:',nout)
        print('number of training exammples:',n_examples)
        model = Sequential()
        model.add(LSTM(64, input_shape=(nin,1), return_sequences=True, recurrent_dropout=0.1,dropout=0.5))
        model.add(LSTM(64, input_shape=(nin,1), return_sequences=True, recurrent_dropout=0.1,dropout=0.5))
        model.add(LSTM(64))
        model.add(Dense(nout, activation='softmax'))
        model.compile(loss=loss, optimizer=opt)
        model.compile(optimizer=opt,loss=loss,metrics=['accuracy'])
        return model
                
def fit_model(model,nb_epoch):                
        # fit
        #callb = [EarlyStopping(monitor='loss', min_delta=0, patience=0, verbose=0, mode='auto'), ModelCheckpoint('model.h5', monitor='loss', verbose=1, save_best_only=True)]
        callb=[ModelCheckpoint('model.h5', monitor='loss', verbose=1, save_best_only=True)]
        print('fitting...')
        model.fit_generator(batch_generator(),n_batches, nb_epoch, verbose=1,callbacks=callb)

def load_data(name):
        with open (name, 'rb') as fp:
            text = pickle.load(fp)
        print('using saved data... (',len(text),')')
        return text
    
def predict_words(n_pred):
    start = np.random.randint(0, len(words)-seq_length) # pick a random seed
    pattern = words[start:start+seq_length] # get a full sequence
    print("Seed: "," ".join(pattern))
    pattern = [word_to_int[pat] for pat in pattern] # turn to ints for model
    print("Prediction: ")
    for i in range(n_pred): # for n_pred words (length of predicted sequence)
        x = np.reshape(pattern, (1, len(pattern), 1)) # make data pretty for model
        x = x / float(n_words)
        prediction = model.predict(x, verbose=0) # get model predictions (probabilities of unique words)
        index = np.random.multinomial(1, np.squeeze(prediction)) # sample over word probabilities to get actual prediction
        result = int_to_word[list(index).index(1)]
        seq_in = [int_to_word[value] for value in pattern]
        #while result=='RARE': # if model predicts rare, sample again until it finds a more frequent word
        #    index = np.random.multinomial(1, np.squeeze(prediction))
        #    result = int_to_word[list(index).index(1)]
        sys.stdout.write(result) # print result
        sys.stdout.write(" ")
        pattern.append(list(index).index(1))
        pattern = pattern[1:len(pattern)] # delete first element of pattern and continue (slowly gets rid of seed)


# get raw text
#raw_pretrain = get_data('gwtext.txt','pretrain/*.txt') # also makes file for main
rawwords = load_data('text_main.pkl')

# get word counts and dictionaries and make a category for rare words
word_counts = Counter(word for word in rawwords)
words = [ word if word_counts[word]>5 else 'RARE' for word in rawwords ]
unique_words = sorted(list(set(words)))
word_to_int = dict((c, i) for i, c in enumerate(unique_words))
int_to_word = dict((i, c) for i, c in enumerate(unique_words))
n_words = len(words)
n_uwords = len(unique_words)
print('Total Words (without rare words): ', n_words)
print('Unique Words (without rare words): ', n_uwords)

batch_size = 64 # how many sequences to train concurrently per weight update
seq_length = 25 # number of words per training sequence
n_examples = len(words)-seq_length # total number of available example sequences
n_batches = n_examples/batch_size # how many batches from full set of examples

# optimizer
sgd = optimizers.SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

model = make_model(opt=sgd,loss='categorical_crossentropy')
#n_examples
fit_model(model,nb_epoch=15)



