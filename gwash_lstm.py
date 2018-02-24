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
    text = re.sub(' {2,}', '', text)
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

def get_dicts(rawwords):
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
        return words,n_words,n_uwords,word_to_int,int_to_word

def batch_generator(batches):
        count = 0
        shuffle_index = np.arange(n_examples)
        np.random.shuffle(shuffle_index) 
        while 1:
                dataX = []
                dataY = []
                for i in shuffle_index[batch_size*count:batch_size*(count+1)]:
                        seq_in = words[i:i + seq_length] # seq_length number of words from all training words
                        seq_out = words[i + seq_length] # the word after that, training signal
                        dataX.append([word_to_int[wd] for wd in seq_in]) # seq of words turned into int value
                        dataY.append(word_to_int[seq_out]) # training signal words turned into int value
                X = np.zeros((len(dataX), seq_length, n_uwords), dtype=np.bool)
                y = np.zeros((len(dataY), n_uwords), dtype=np.bool)
                for i, pat in enumerate(dataX):
                        for t, w in enumerate(pat):
                                X[i, t, w] = 1
                                y[i, dataY[i]] = 1
                count += 1 # move one batch forward and repeat
                yield(X,y)
                if (count >= batches):
                        np.random.shuffle(shuffle_index)
                        count=0

def load_data(name):
        with open (name, 'rb') as fp:
            text = pickle.load(fp)
        print('using saved data... (',len(text),')')
        return text
    
def predict_words(npred,div):
        start = np.random.randint(0, len(words)-seq_length) # pick a random seed
        pattern = words[start:start+seq_length] # get a full sequence
        print('Seed: "' + ' '.join(pattern) + '"')
        for i in range(npred):
                x_pred = np.zeros((1, seq_length, len(sorted(list(set(words))))))
                for t, wd in enumerate(pattern):
                        x_pred[0, t, word_to_int[wd]] = 1.
                preds = model.predict(x_pred, verbose=0)[0]
                preds = np.asarray(preds).astype('float64')
                preds = np.log(preds)/div
                exp_preds = np.exp(preds)
                preds = exp_preds / np.sum(exp_preds)
                res = int_to_word[np.argmax(np.random.multinomial(1, preds, 1))]
                while res=='RARE': #or res==pattern[-1]: # if model predicts rare, sample again until it finds a more frequent word
                        res = int_to_word[np.argmax(np.random.multinomial(1, preds,1))]
                pattern.append(res)
                pattern = pattern[1:]
                sys.stdout.write(res+' ')
                sys.stdout.flush()

def make_model(opt,loss):
        # Keras model
        nout = n_uwords
        nin = seq_length
        print('input size:',nin)
        print('output size:',nout)
        print('number of training exammples:',n_examples)
        model = Sequential()
        model.add(LSTM(500, input_shape=(nin,nout), return_sequences=True, recurrent_dropout=0.1,dropout=0.5))
        model.add(LSTM(500, input_shape=(nin,nout), recurrent_dropout=0.1,dropout=0.5)) # prev must return sequences
        model.add(Dense(nout, activation='softmax'))
        model.compile(optimizer=opt,loss=loss,metrics=['accuracy'])
        return model

# get text & words
raw_pretrain = get_data('gwtext.txt')
rawwords = load_data('text_main.pkl')
words,n_words,n_uwords,word_to_int,int_to_word = get_dicts(rawwords)

batch_size = 100 # how many sequences to train concurrently per weight update
seq_length = 30 # number of words per training sequence
n_examples = len(words)-seq_length # total number of available example sequences
n_batches = n_examples/batch_size # how many batches from full set of examples

rms = optimizers.RMSprop(lr=0.001)
model = make_model(opt=rms,loss='categorical_crossentropy')

ep=1
while ep<=10:
        model.fit_generator(batch_generator(n_batches),n_batches, 1, verbose=1)
        model.save('modelE'+str(ep)+'.h5')
        predict_words(200,1.0)
        ep+=1
        
