# The example function below keeps track of the opponent's history and plays whatever the opponent played two plays ago. It is not a very good player so you will need to change the code to pass the challenge.

import tensorflow as tf

def player(prev_play, opponent_history=[]):
    opponent_history.append(prev_play)

    return guess 

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim,
                                batch_input_shape=[batch_size, None]), 
        tf.keras.layers.LSTM(rnn_units,
                            return_sequences=True,
                            stateful=True,
                            recurrent_initializer='glorot_uniform'),
        tf.keras.layers.Dense(vocab_size)
    ])
    return model

def shid_and_camed():
    mrugesh = open("database/mrugesh.txt")
    mrugesh_train_data = [[i for i in mrugesh.readline()]]
    mrugesh_train_data = tf.data.Dataset.from_tensor_slices(mrugesh_train_data)
    #
    seq_length = 100
    examples_per_epoch = len(mrugesh_train_data)//(seq_length+1)

    sequences = mrugesh_train_data.batch(seq_length+1, drop_remainder=True)
    dataset = sequences.map(split_input_target) 
    BATCH_SIZE = 1
    VOCAB_SIZE = 3
    EMBEDDING_DIM = 8
    RNN_UNITS = 1024

    model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
    model.build(tf.TensorShape([1,None]))
    model.summary() 
    model.compile(optimizer="adam", loss=loss)
    history = model.fit(dataset, epochs=40)

def split_input_target(chunk): 
    input_text = chunk[:-1]
    output_text = chunk[1:]
    return input_text, output_text

def loss(labels, logits): #logits = probability distribution
    return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

shid_and_camed()
