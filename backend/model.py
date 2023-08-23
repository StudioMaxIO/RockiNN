import tensorflow as tf
import regex as re
import subprocess
import os
import urllib
import numpy as np
from tqdm import tqdm

cwd = os.path.dirname(__file__)


def char_to_idx(vocab, character):
    char2idx = {u: i for i, u in enumerate(vocab)}
    # Returns None if character is not in vocab
    return char2idx.get(character, None)


def idx_to_char(vocab, index):
    idx2char = np.array(vocab)
    # Returns None if index is out of range
    return idx2char[index] if 0 <= index < len(vocab) else None


def load_training_data():
    with open(os.path.join(cwd, "data", "training_music.abc"), "r") as f:
        text = f.read()
    songs = extract_song_snippet(text)
    return songs


def extract_song_snippet(text):
    pattern = '(^|\n\n)(.*?)\n\n'
    search_results = re.findall(
        pattern, text, overlapped=True, flags=re.DOTALL)
    songs = [song[1] for song in search_results]
    print("Found {} songs in text".format(len(songs)))
    return songs


def vectorize_string(vocab, string):
    vectorized = np.array([char_to_idx(vocab, char) for char in string])
    return vectorized


def prepare_data():
    songs = load_training_data()
    # Join our list of song strings into a single string containing all songs
    songs_joined = "\n\n".join(songs)

    # Find all unique characters in the joined string
    vocab = sorted(set(songs_joined))
    # print("There are", len(vocab), "unique characters in the dataset")
    # print(vocab)
    vectorized_songs = vectorize_string(vocab, songs_joined)
    return vocab, vectorized_songs


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


def instantiate_optimizer(learning_rate, optimizer):
    if optimizer == "Adam":
        # return tf.keras.optimizers.Adam(learning_rate)
        # use legacy for Mac M1/M2
        return tf.keras.optimizers.legacy.Adam(learning_rate)
    return tf.keras.optimizers.SGD(learning_rate)


def compute_loss(labels, logits):
    loss = tf.keras.losses.sparse_categorical_crossentropy(
        labels, logits, from_logits=True)
    return loss


@tf.function
def train_step(model, optimizer, x, y):
    with tf.GradientTape() as tape:
        y_hat = model(x)
        loss = compute_loss(y, y_hat)

    # Compute the gradients
    grads = tape.gradient(loss, model.trainable_variables)

    # Apply the gradients to the optimizer so it can update the model accordingly
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss


def generate_text(model, vocab, start_string, generation_length=1000):
    # Evaluation step (generating ABC text using the learned RNN model)
    input_eval = [char_to_idx(vocab, s) for s in start_string]
    input_eval = tf.expand_dims(input_eval, 0)
    text_generated = []
    model.reset_states()
    tqdm._instances.clear()
    for i in tqdm(range(generation_length)):
        predictions = model(input_eval)
        predictions = tf.squeeze(predictions, 0)
        predicted_id = tf.random.categorical(
            predictions, num_samples=1)[-1, 0].numpy()
        input_eval = tf.expand_dims([predicted_id], 0)
        text_generated.append(idx_to_char(vocab, predicted_id))
    return (start_string + ''.join(text_generated))


def get_batch(vectorized_songs, seq_length, batch_size):
    # the length of the vectorized songs string
    n = vectorized_songs.shape[0] - 1
    # randomly choose the starting indices for the examples in the training batch
    idx = np.random.choice(n-seq_length, batch_size)
    # construct a list of input sequences for the training batch
    input_batch = [vectorized_songs[i: i+seq_length] for i in idx]
    # construct a list of output sequences for the training batch
    output_batch = [vectorized_songs[i+1: i+1+seq_length] for i in idx]
    # x_batch, y_batch provide the true inputs and targets for network training
    x_batch = np.reshape(input_batch, [batch_size, seq_length])
    y_batch = np.reshape(output_batch, [batch_size, seq_length])
    return x_batch, y_batch
