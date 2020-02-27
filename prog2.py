from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation, GRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.optimizers import Adam
from keras.models import Model
import pandas as pd
import pickle
import numpy as np


def build_model(embedding_matrix,nb_words,embedding_size):
    inp = Input(shape=(max_length,))
    x = Embedding(nb_words, embedding_size, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(0.3)(x)
    x1 = Bidirectional(LSTM(128, return_sequences=True))(x)
    max_pool1 = GlobalMaxPooling1D()(x1)
    predictions = Dense(1, activation='sigmoid')(max_pool1)
    model = Model(inputs=inp, outputs=predictions)
    adam = optimizers.Adam(lr=learning_rate)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model

max_length = 55
embedding_size = 300
learning_rate = 0.001
batch_size = 64
num_epoch = 2


train_word_sequences=pickle.load(open('train_word_sequences.txt','rb'))
embedding_matrix=pickle.load(open('embedding_matrix.txt','rb'))

train_word_sequences = pad_sequences(train_word_sequences, maxlen=max_length, padding='post')


train_y=pickle.load(open('y.txt','rb'))
model = build_model(embedding_matrix,embedding_matrix.shape[0], embedding_size)

for i in range(num_epoch):
    model.fit(train_word_sequences, train_y, batch_size=batch_size, epochs=1, verbose=1)
    model.save('modelnlp.h5')
