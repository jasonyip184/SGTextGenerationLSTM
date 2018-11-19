import json
import numpy as np
import os.path
import pickle
import time
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Activation, Bidirectional, Dense, Dropout, Embedding, LSTM
from keras.models import Sequential, load_model
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from matplotlib import pyplot as plt
from statistics import mean, median, mode


def previous_words_to_many_words_context(encoded_texts, sequence_length):
    X_sequences = []
    Y_last_words = []
    for text in encoded_texts:
        for i in range(0, len(text)-sequence_length):
            for j in range(1, sequence_length):
                seq = text[i:i+j]
                seq = [0] * (sequence_length-len(seq)-1) + seq
                X_sequences.append(seq)
                Y_last_words.append(text[i+j])
    X_sequences = np.array(X_sequences)
    Y_last_words = np.array(Y_last_words)
    print("Number of Sequences:", len(X_sequences))
    return X_sequences, Y_last_words


def generate_text(num_of_texts, input_text, name):
    model = load_model(name+'_model.h5')

    with open(name+'_objects.pickle', 'rb') as f:
        X_sequences = pickle.load(f)
        tokenizer = pickle.load(f)
        sequence_length = pickle.load(f)

    passage = input_text.split(' ')
    passage_length = num_of_texts*sequence_length

    for i in range(passage_length-sequence_length+1):
        if len(passage) < sequence_length:
            seed_text = [tokenizer.word_index[word] for word in passage]
            seed_text = [0]*(sequence_length-1-len(seed_text)) + seed_text
        else:
            seed_text = passage[-(sequence_length-1):]
            seed_text = [tokenizer.word_index[word] for word in seed_text]
        X_text = np.reshape(seed_text, (1, len(seed_text)))
        prediction = model.predict_classes(X_text, verbose=0)
        output_word = tokenizer.index_word[prediction[0]]
        passage.append(output_word)

    for i in range(0, len(passage), sequence_length):
        string = ' '.join(passage[i:i+sequence_length])
        time.sleep(1)
        print(string, "\n\n")

    

def learn_texts(context):
    with open('smsCorpus_en_2015.03.09_all.json') as f:
        corpus = json.load(f)

    users = set()
    texts = []
    for text in corpus['smsCorpus']['message']:
        users.add(text['source']['userProfile']['userID']['$'])
        texts.append(text_to_word_sequence(str(text['text']['$'])))
    print("Number of Users:", len(users))
    print("Number of Texts:", len(corpus['smsCorpus']['message']))
    texts = texts[:1000]
    lengths = [len(text) for text in texts]
    sequence_length = round(mean(lengths))
    print("Sequence Length: ", sequence_length)

    filtered_texts = [text for text in texts if len(text) >= sequence_length]
    print("Number of Filtered Texts:", len(filtered_texts))

    vocab = set()
    for text in filtered_texts:
        vocab.update(text)
    vocab_size = len(vocab)+1
    print("Vocab Size:", vocab_size)

    number_of_words = sum(lengths)
    print("Total words", number_of_words)
    print("Vocab / Total words ratio:", round(vocab_size/number_of_words, 3))

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(filtered_texts)
    encoded_texts = tokenizer.texts_to_sequences(filtered_texts)

    embedding_dims = round(vocab_size**0.25)
    print("Dimensions for Embedding Layer:", embedding_dims)

    if context == "previous_words_to_one_word":
        X_sequences, Y_last_words = previous_words_to_one_word_context(encoded_texts, sequence_length)
        mask_zero = False
    elif context == "previous_words_to_many_words":
        X_sequences, Y_last_words = previous_words_to_many_words_context(encoded_texts, sequence_length)
        mask_zero = True

    with open(context+'_objects.pickle', 'wb') as f:
        pickle.dump(X_sequences, f)
        pickle.dump(tokenizer, f)
        pickle.dump(sequence_length, f)

    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dims, input_length=sequence_length-1, mask_zero=mask_zero))
    model.add(Bidirectional(LSTM(128)))
    model.add(Dropout(0.5))
    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))
    print(model.summary())
    
    adam = optimizers.Adam(lr=0.001)
    if os.path.exists(context+"_best_weights.hdf5"):
        model.load_weights(context+"_best_weights.hdf5")

    model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['sparse_categorical_accuracy'])
    checkpoint = ModelCheckpoint(context+"_best_weights.hdf5", monitor='sparse_categorical_accuracy', verbose=1, save_best_only=True, mode='max')
    earlystopping = EarlyStopping(patience=4, monitor='sparse_val_categorical_accuracy')
    callbacks = [earlystopping, checkpoint]
    model.fit(X_sequences, Y_last_words, epochs=100, callbacks=callbacks, verbose=2, validation_split=0.1)
    model.save(context+'_model.h5')

# learn_texts(context="previous_words_to_many_words")
generate_text(50, "i will be", "previous_words_to_many_words")