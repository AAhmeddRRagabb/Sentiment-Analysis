###############################################################################
# This file contains the required functions to 
# >> Load the word_2_vector glove embeddings
# >> Transform the input sequences into indices
# >> Generate batches for the model
# >> Construct the model architecture with the embedding layer

## الله المستعان
###############################################################################

import numpy as np
import tensorflow as tf


def load_glove_vecs(glove_file):
    """
    This function loads the glove vectors

    Args:
        glove_file: the glove file path

    Returns:
        word_to_vec_map: a dict contains the words and their corresponding vectors
    """
    with open(glove_file, 'r', encoding = 'utf-8') as f:
        word_to_vec_map = {}

        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            word_to_vec_map[curr_word] = np.array(line[1:], dtype = np.float32)
    return word_to_vec_map



def pretrained_embedding_layer(word_to_vec_map, word_to_idx):
    """
    Loads in the pre-trained GloVe vectors & uses them in a keras layer

    Args:
        word_to_vec_map: dictionary mapping words to their GloVe vector representation.
        word_to_idx  : dictionary mapping from words to their indices in the vocabulary

    Returns:
        embedding_layer: pretrained layer Keras instance
    """

    vocab_size = len(word_to_idx)
    any_word = next(iter(word_to_vec_map.keys()))
    embedding_dimesions = word_to_vec_map[any_word].shape[0]      # The 300 dimensions

    # Building the embedding layer
    embedding_matrix = np.zeros(shape = (vocab_size, embedding_dimesions))
    words = list(word_to_idx.keys())
    indices = np.array([word_to_idx[w] for w in words if w in word_to_vec_map])
    vectors = np.array([word_to_vec_map[w] for w in words if w in word_to_vec_map])
    embedding_matrix[indices] = vectors

    embedding_layer = tf.keras.layers.Embedding(
        input_dim  = vocab_size,
        output_dim = embedding_dimesions,
        trainable = True
    )

    embedding_layer.build((None,))
    embedding_layer.set_weights([embedding_matrix])

    return embedding_layer






def sentences_to_indices(batch, word_to_idx):
    """
    Maps each padded sentence in the given batch to an array of word indices.

    Args:
        batch       : list of padded sentences, length m
        word_to_idx : dict mapping words to indices

    Returns:
        indices: ndarray of shape (m, seq_len)
    """

    m = len(batch)
    seq_len = len(batch[0])
    indices = np.zeros(shape = (m, seq_len))

    for seq in range(m):
        sentence_words = batch[seq]
        word_index = 0
        for word in sentence_words:
            if word in word_to_idx.keys():
                indices[seq, word_index] = word_to_idx[word]
                word_index += 1
    return indices



def batch_train_generator(X, Y, word_to_idx, batch_size = 32):
    """
    This function generates the batches for training the model

    Args:
        X          : a list contains the input sequences
        Y          : a list contains the input labels
        word_to_idx: a dict map each word into its corresponding index
        batch_size : the size of the training batch
    """
    m = len(X)
    i = 0

    while True:
        X_batch = X[i : i + batch_size]
        Y_batch = Y[i : i + batch_size]
 
        X_batch_indices = sentences_to_indices(X_batch, word_to_idx)
        Y_batch = np.array(Y_batch).reshape(-1, 1)

        yield X_batch_indices, Y_batch

        i += batch_size
        if i >= m:
            i = 0


def batch_val_generator(X, Y, word_to_idx, batch_size = 64):
    """
    This function generates the cross-validation batches the model

    Args:
        X          : a list contains the input sequences
        Y          : a list contains the input labels
        word_to_idx: a dict map each word into its corresponding index
        batch_size : the size of the training batch
    """
    m = len(X)
    i = 0

    while True:
        X_batch = X[i : i + batch_size]
        Y_batch = Y[i : i + batch_size]

        X_batch_indices = sentences_to_indices(X_batch, word_to_idx)
        Y_batch = np.array(Y_batch).reshape(-1, 1)

        yield X_batch_indices, Y_batch

        if i >= m:
            i = 0


def build_model(lstm_units, keep_prob, input_shape, output_shape, word_to_vec_map, word_to_idx):
    """
    Builds an LSTM-based sentiment analysis model with pre-trained word embeddings.

    Args:
        lstm_units      : int, number of units in each LSTM layer
        keep_prob       : float, probability of keeping a unit during dropout (0 < keep_prob <= 1)
        input_shape     : int, length of input sequences (seq_len)
        output_shape    : int, number of output classes
                              1 ---> for binary classification
                              > 1 -> for multi-class classification
        word_to_vec_map : dict, maps each word to its pre-trained vector 
        word_to_idx     : dict, maps each word to its integer index in the vocabulary

    Returns:
        model : tf.keras.Model,  LSTM model 
    """

    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_idx)
    activation_fn = 'softmax' if output_shape > 1 else 'sigmoid'
    
    # Input layer
    input_indices = tf.keras.layers.Input(shape=(input_shape,), dtype='int32')
    embeddings = embedding_layer(input_indices)
    
    # LSTM layers with dropout
    X = tf.keras.layers.LSTM(units = lstm_units, return_sequences = True)(embeddings)
    X = tf.keras.layers.Dropout(rate = 1 - keep_prob)(X)
    X = tf.keras.layers.LSTM(units = lstm_units, return_sequences = False)(X)
    X = tf.keras.layers.Dropout(rate = 1 - keep_prob)(X)
    
    # Output layer
    Y = tf.keras.layers.Dense(units = output_shape, activation = activation_fn)(X)


    model = tf.keras.models.Model(inputs = input_indices, outputs = Y)
    return model